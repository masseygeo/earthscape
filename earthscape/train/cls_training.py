
from .earlystopping import EarlyStopping

import os
import glob
from datetime import datetime
import pandas as pd
import torch



def train_epoch(model, train_loader, criterion, optimizer, device, baseline=True, scheduler=None):
    """
    Train a multilabel classification model for a single training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained. The model is set to training mode.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding training batches. Each batch is a dictionary
        containing a ``'label'`` tensor and one or more input tensors.
    criterion : callable
        Loss function applied to model outputs and labels.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which model and tensors are located.
    baseline : bool, optional
        If True, a single input tensor is selected from the batch and passed
        to the model. If False, the full input dictionary is passed.
    scheduler : object, optional
        Learning-rate scheduler with a ``step()`` method called after each batch.

    Returns
    -------
    epoch_loss : float
        Mean loss across all training batches.
    epoch_accuracy : float
        Micro-averaged classification accuracy in percent, computed by
        thresholding sigmoid outputs at 0.5.
    """
    # set model for training
    model.train()

    # initialize variables running over epoch...
    running_loss = torch.zeros((), device=device)
    correct_preds = torch.zeros((), device=device)
    total_batches = 0
    total_elements = 0
  
    # iterate through batches...
    for batch in train_loader:

        # get labels and images from batch...
        labels = batch['label'].to(device, non_blocking=True).float()

        # dict of modality tensors to pass to model
        modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

        # single tensor to pass to model (baseline tests)
        if baseline:
            modalities = next(iter(modalities.values()))            

        # zero optimizer...
        optimizer.zero_grad(set_to_none=True)

        # model training...
        logits = model(modalities)                # forward pass
        loss = criterion(logits, labels)          # calculate loss
        loss.backward()                           # back propagation
        optimizer.step()                          # update parameters

        # set lr using scheduler...
        if scheduler is not None:
            scheduler.step()
        
        # update running totals for batch...
        running_loss += loss.detach()                  # running loss for batch
        total_batches += 1                             # running count of batches
        total_elements += labels.numel()               # running count of images
        probs = torch.sigmoid(logits)                  # probabilities of batch
        preds = (probs >= 0.5).to(labels.dtype)        # predictions of batch with 50% probability threshold
        correct_preds += (preds == labels).sum()       # numer of correct predictions vs. ground-truth labels

    # total epoch loss and accuracy...
    epoch_loss = (running_loss / total_batches).item()                # average batch loss
    epoch_accuracy = (correct_preds / total_elements * 100).item()    # total accuracy

    return epoch_loss, epoch_accuracy




def validate_epoch(model, val_loader, criterion, device, baseline=True):
    """
    Validate a multilabel classification model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    val_loader : torch.utils.data.DataLoader
        Validation DataLoader yielding batches as dicts with a ``'label'`` tensor
        and one or more modality tensors.
    criterion : callable
        Loss function.
    device : torch.device
        Device used for evaluation.
    baseline : bool, default True
        Controls how input tensors are extracted from each batch. If True,
        a single modality tensor is selected from the input dictionary and
        passed to the model. If False, the full modality dictionary is passed 
        to the model (e.g., for SGMap-Net).

    Returns
    -------
    epoch_loss : float
        Mean loss across validation batches.
    epoch_accuracy : float
        Micro-averaged classification accuracy (percentage) computed across all 
        label elements by thresholding sigmoid outputs at 0.5.
    """

    # set model for evaluation
    model.eval()

    # initialize variables running over epoch...
    running_loss = torch.zeros((), device=device)
    correct_preds = torch.zeros((), device=device)
    total_batches = 0
    total_elements = 0

    # iterate through batches...
    with torch.no_grad():
        for batch in val_loader:

            # get labels and images from batch...
            labels = batch['label'].to(device, non_blocking=True)
            
            # dict of modality tensors to pass to model 
            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

            # single tensor to pass to model (baseline tests)
            if baseline:
                modalities = next(iter(modalities.values()))

            # get model logits & calculated loss...
            logits = model(modalities)
            loss = criterion(logits, labels)

            # update running totals...
            running_loss += loss.detach()
            total_batches += 1
            total_elements += labels.numel()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            correct_preds += (preds == labels).sum()

    # calculate loss and accuracy for epoch...
    epoch_loss = (running_loss / total_batches).item()
    epoch_accuracy = (correct_preds / total_elements * 100).item()

    return epoch_loss, epoch_accuracy





def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir, baseline=True, early_stop=None, warmup=True, cosine_decay=True):
    """
    Train a multilabel classification model for multiple epochs and log training/validation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding training batches.
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding validation batches.
    criterion : callable
        Loss function accepting (logits, labels) and returning a scalar tensor.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device used for training and validation.
    num_epochs : int
        Number of training epochs.
    output_dir : str or os.PathLike
        Directory to save model checkpoints and the training log CSV.
    baseline : bool, optional
        If True, a single input tensor is selected from each batch and passed
        to the model. If False, the full input dictionary is passed.
    early_stop : dict or None, optional
        Keyword arguments used to initialize `EarlyStopping`. If None, early
        stopping is disabled.
    warmup : bool, optional
        If True, apply linear learning-rate warmup.
    cosine_decay : bool, optional
        If True, apply cosine learning-rate decay.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing per-epoch training and validation loss and accuracy,
        along with training time in minutes.
    """

    # initialize variables...
    train_loss = []                  # training - list of epoch losses
    train_acc = []                   # training - list of epoch accuracies
    train_time = []                  # training - list of epoch times
    val_loss = []                    # validation - list of epoch losses
    val_acc = []                     # validation - list of epoch accuracies
    best_val_loss = float('inf')     # validation - best validation loss

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
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device, baseline, scheduler)
        t1 = datetime.now()

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        tt = round((t1-t0).total_seconds() / 60, 2)
        train_time.append(tt)
        training_str = f"TRAINING   -- Loss: {epoch_train_loss:.3f}  |  Accuracy: {epoch_train_acc:.2f}%  |  Time: {tt} mins."
        print(training_str)


        # validation...
        t2 = datetime.now()
        epoch_val_loss, epoch_val_acc = validate_epoch(model, val_loader, criterion, device, baseline)
        t3 = datetime.now()

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        val_tt = round((t3-t2).total_seconds() / 60, 2)
        val_str = f"VALIDATION -- Loss: {epoch_val_loss:.3f}  |  Accuracy: {epoch_val_acc:.2f}%  |  Time: {val_tt} mins."
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
    df = pd.DataFrame({'train loss': train_loss, 'train accuracy': train_acc, 'train time': train_time, 'val loss': val_loss, 'val accuracy': val_acc})
    output_path = f"{output_dir}/training_log.csv"
    df.to_csv(output_path, index=False)

    return df
