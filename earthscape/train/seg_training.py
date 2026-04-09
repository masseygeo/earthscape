
from earthscape.train import EarlyStopping
from earthscape.evaluation import calculate_dice_score
import os
import glob
from datetime import datetime
import pandas as pd
import torch



def train_epoch_seg(model, train_loader, criterion, optimizer, device, baseline=True, scheduler=None):
    """
    Run a single training epoch for a segmentation model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : iterable
        Data loader yielding batches with keys "mask" and input features.
    criterion : callable
        Loss function taking (logits, masks) as input.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    device : torch.device
        Device on which computations are performed.
    baseline : bool, optional
        If True, use a single input tensor instead of a dictionary of inputs.
    scheduler : object, optional
        Learning rate scheduler with a `.step()` method called after each batch.

    Returns
    -------
    tuple of float
        Mean loss and mean Dice score over the epoch.
    """
    
    model.train()

    # initialize running metrics...
    running_loss = 0.0
    running_dice = 0.0
    total_batches = 0

    # iterate over one epoch...
    for batch in train_loader:

        # get masks & features...
        masks = batch['mask'].to(device, non_blocking=True).long()
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'mask'}

        # single tensor to pass to model (baseline tests)
        if baseline:
            inputs = next(iter(inputs.values()))    

        # train one step...
        optimizer.zero_grad(set_to_none=True)      # zero the gradients
        logits = model(inputs)                     # forward pass -> [B, C, H, W]
        loss = criterion(logits, masks)            # calculate loss
        loss.backward()                            # backprop
        optimizer.step()                           # update weights

        # update scheduler if given
        if scheduler is not None:
            scheduler.step()

        # update running metrics
        running_loss += loss.detach().item()
        total_batches += 1

        # calculate dice score...
        with torch.no_grad():
            dice = calculate_dice_score(logits, masks)
            running_dice += dice

    # calculate average metrics over epoch...
    epoch_loss = running_loss / total_batches
    epoch_dice = running_dice / total_batches

    return epoch_loss, epoch_dice




def validate_epoch_seg(model, val_loader, criterion, device, baseline=True):
    """
    Run a single validation epoch for a segmentation model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    val_loader : iterable
        Data loader yielding batches with keys "mask" and input features.
    criterion : callable
        Loss function taking (logits, masks) as input.
    device : torch.device
        Device on which computations are performed.
    baseline : bool, optional
        If True, use a single input tensor instead of a dictionary of inputs.

    Returns
    -------
    tuple of float
        Mean loss and mean Dice score over the epoch.
    """

    model.eval()

    # initialize running metrics...
    running_loss = 0.0
    running_dice = 0.0
    total_batches = 0

    # iterate over each batch...
    with torch.no_grad():
        for batch in val_loader:

            # get masks & features...
            masks = batch['mask'].to(device, non_blocking=True).long()
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'mask'}

            # single tensor to pass to model (baseline tests)
            if baseline:
                inputs = next(iter(inputs.values()))

            # calculate model logits & loss...
            logits = model(inputs)            # [B, C, H, W]
            loss = criterion(logits, masks)

            # calculate dice score
            dice = calculate_dice_score(logits, masks)

            # update running metrics...
            running_loss += loss.detach().item()
            total_batches += 1
            running_dice += dice

        # calculate average metrics over epoch...
        epoch_loss = running_loss / total_batches
        epoch_dice = running_dice / total_batches

        return epoch_loss, epoch_dice




def seg_train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir, early_stop=None, warmup=True, cosine_decay=True, baseline=True):
    """
    Train a segmentation model over multiple epochs and record training history.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : iterable
        Data loader yielding training batches.
    val_loader : iterable
        Data loader yielding validation batches.
    criterion : callable
        Loss function used for training and validation.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    device : torch.device
        Device on which computations are performed.
    num_epochs : int
        Number of training epochs.
    output_dir : str or os.PathLike
        Directory where model checkpoints and the training log are written.
    early_stop : dict or None, optional
        Keyword arguments used to initialize `EarlyStopping`. If None, early
        stopping is disabled.
    warmup : bool, optional
        If True, apply linear learning-rate warmup.
    cosine_decay : bool, optional
        If True, apply cosine learning-rate decay.
    baseline : bool, optional
        If True, pass a single input tensor to the model instead of a dictionary
        of inputs.

    Returns
    -------
    pandas.DataFrame
        Training log containing per-epoch training and validation metrics.
    """

    # initialize variables...
    train_loss = []                   # training - list of epoch losses
    train_dice = []                   # training - list of epoch dice scores
    train_time = []                   # training - list of epoch times
    val_loss = []                     # validation - list of epoch losses
    val_dice = []                     # validation - list of epoch dice scores
    best_val_loss = float('inf')      # validation - best validation loss


    # early stopping...
    stopper = None
    if early_stop is not None:
        stopper = EarlyStopping(**early_stop)


    # scheduler...
    scheduler = None
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    if warmup and cosine_decay:
        warmup_epochs = 5
        warmup_steps = warmup_epochs * steps_per_epoch
        cosine_steps = total_steps - warmup_steps

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    elif warmup:
        warmup_epochs = 5
        warmup_steps = warmup_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)

    elif cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


    # iterate over epochs...
    for epoch in range(num_epochs):

        # training...
        print(f"\nEpoch {epoch+1}")
        t0 = datetime.now()
        epoch_train_loss, epoch_train_dice = train_epoch_seg(model, train_loader, criterion, optimizer, device, baseline, scheduler)
        t1 = datetime.now()

        train_loss.append(epoch_train_loss)
        train_dice.append(epoch_train_dice)
        tt = round((t1-t0).total_seconds() / 60, 2)
        train_time.append(tt)
        training_str = f"TRAINING   -- Loss: {epoch_train_loss:.3f}  |  Dice Score: {epoch_train_dice:.3f}  |  Time: {tt} mins."
        print(training_str)


        # validation...
        t2 = datetime.now()
        epoch_val_loss, epoch_val_dice = validate_epoch_seg(model, val_loader, criterion, device, baseline)
        t3 = datetime.now()

        val_loss.append(epoch_val_loss)
        val_dice.append(epoch_val_dice)
        val_tt = round((t3-t2).total_seconds() / 60, 2)
        val_str = f"VALIDATION -- Loss: {epoch_val_loss:.3f}  |  Dice Score: {epoch_val_dice:.3f}  |  Time: {val_tt} mins."
        print(val_str)


        # save best model...
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            previous_checkpoints = glob.glob(f"{output_dir}/best_loss_epoch*.pth")
            if len(previous_checkpoints) > 0:
                for path in previous_checkpoints:
                    os.remove(path)
            torch.save(model.state_dict(), f"{output_dir}/best_loss_epoch{epoch + 1}.pth")
            print(f"New best model saved!")


        # early stopping...
        if stopper is not None:
            stop = stopper.step(epoch_val_loss, epoch+1)
            if stop:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    # save loss, accuracy, time to training log
    df = pd.DataFrame({'train loss': train_loss, 'train dice score': train_dice, 'train time': train_time, 'val loss': val_loss, 'val dice score': val_dice})
    output_path = f"{output_dir}/training_log.csv"
    df.to_csv(output_path, index=False)

    return df