
import os
import glob
import json
from datetime import datetime
import pandas as pd
import torch
import torchinfo
import matplotlib.pyplot as plt




def train_epoch(model, train_loader, criterion, optimizer, device, baseline=True):
    """
    Train a model for a single training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained. The model is set to training mode.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding training batches. Each batch is a dictionary containing
        a ``'label'`` tensor and one or more modality tensors.
    criterion : callable
        Loss function applied to model outputs and labels.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which model and tensors are located.
    baseline : bool, default True
        Controls how input tensors are extracted from each batch. If True,
        a single modality tensor is selected from the input dictionary and
        passed to the model. If False, the full modality dictionary is passed 
        to the model (e.g., for SGMap-Net).

    Returns
    -------
    epoch_loss : float
        Mean loss across all training batches.
    epoch_accuracy : float
        Micro-averaged classification accuracy (percentage), computed across all
        batches by thresholding sigmoid outputs at 0.5.
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

        # dict of modality tensors to pass to model (SGMap-Net)
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
    Validate a model for one epoch.

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
            
            # dict of modality tensors to pass to model (SGMap-Net)
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





def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir, baseline=True):
    """
    Train a model for multiple epochs and log training/validation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding training batches.
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding validation batches.
    criterion : callable
        PyTorch-compatible loss function that accepts (logits, labels) and
        returns a scalar tensor.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device used for training and validation.
    num_epochs : int
        Number of training epochs.
    output_dir : str or pathlib.Path
        Directory to save the best model checkpoint and the training log CSV.
    baseline : bool, default True
        Controls how input tensors are extracted from each batch. If True,
        a single modality tensor is selected from the input dictionary and
        passed to the model. If False, the full modality dictionary is passed 
        to the model (e.g., for SGMap-Net).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing per-epoch training & validation loss and accuracy, 
        plus training time in minutes.
    """

    # initialize variables...
    train_loss = []                  # training - list of epoch losses
    train_acc = []                   # training - list of epoch accuracies
    train_time = []                  # training - list of epoch times
    val_loss = []                    # validation - list of epoch losses
    val_acc = []                     # validation - list of epoch accuracies
    best_val_loss = float('inf')     # validation - best validation loss

    # iterate over epochs...
    for epoch in range(num_epochs):

        # training...
        print(f"Epoch {epoch+1}")
        t0 = datetime.now()
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device, baseline=baseline)
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

        print('\n')

    # save loss, accuracy, time to training log
    df = pd.DataFrame({'train loss': train_loss, 'train accuracy': train_acc, 'train time': train_time, 'val loss': val_loss, 'val accuracy': val_acc})
    output_path = f"{output_dir}/training_log.csv"
    df.to_csv(output_path, index=False)

    return df




def train_metadata(model_name, output_dir, seed, train_path, train_patches, val_path, val_patches, test_path, test_patches, cross_path, cross_patches, modality_configs, batch_size, num_epochs, optimizer, criterion, model, input_size):
    """
    Write experiment metadata and a model architecture summary to disk.

    Parameters
    ----------
    model_name : str
        Experiment/model identifier.
    output_dir : str or pathlib.Path
        Directory where `metadata.json` and `architecture.txt` are written.
    seed : int
        Random seed used for the experiment.
    train_patches, val_patches, test_patches, cross_patches : int or str
        Patch counts/identifiers to record.
    modality_configs : dict
        Modality configuration mapping (e.g., channels and optional normalization stats).
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    optimizer : torch.optim.Optimizer
        Optimizer instance (name and selected hyperparameters are logged).
    criterion : torch.nn.Module
        Loss function instance name; alpha and gamma also recorded if focal loss.
    model : torch.nn.Module
        Model whose architecture is summarized with torchinfo.

    Returns
    --------
    None
    """

    ##### collect setup info
    metadata = {
        'NAME': model_name,
        'DIRECTORY': str(output_dir),
        'SEED': seed
        }


    ##### collect modalitiy info
    modalities_meta = {}
    for mod_name, data in modality_configs.items():
        modalities_meta[mod_name] = {}
        modalities_meta[mod_name]['modalities'] = ', '.join(data['channels'])
        # modalities_meta[mod_name]['input shape'] = ', '.join(str(i) for i in input_size)
        if data['mean'] is not None:
            modalities_meta[mod_name]['normalization means'] = ', '.join([str(i) for i in data['mean']])
            modalities_meta[mod_name]['normalization sd'] = ', '.join([str(i) for i in data['sd']])
    metadata['INPUTS'] = modalities_meta


    ##### collect hyperparameters info
    hyper_meta = {
        'batch size': batch_size,
        'epochs': num_epochs, 
        'optimizer': type(optimizer).__name__,
        'learning rate': optimizer.param_groups[0]['lr'],
        'weight decay': optimizer.param_groups[0].get('weight_decay', None),
        'momentum': optimizer.param_groups[0].get('momentum', None),
        'loss': type(criterion).__name__
        }
    if 'Focal' in hyper_meta['loss']:
        # alpha = ', '.join(str(v.item()) for v in criterion.alpha)
        hyper_meta['alpha'] = criterion.alpha
        hyper_meta['gamma'] = criterion.gamma
        hyper_meta['reduction'] = criterion.reduction
        hyper_meta['pos_weight'] = criterion.pos_weight
    metadata['HYPERPARAMETERS'] = hyper_meta


    ##### collect patches info
    patches_meta = {
        'training set': train_path,
        'training n': len(train_patches),
        'validation set': val_path,
        'validation n': len(val_patches),
        'test set': test_path,
        'test n': len(test_patches),
        'cross test set': cross_path,
        'cross test n': len(cross_patches),
        }
    metadata['PATCHES'] = patches_meta


    ##### write log to json
    meta_output_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_output_path, 'w') as f:
        json.dump(metadata, f, indent=4)




def architecture_to_json(output_dir, model, loader):

    # input feature shape...
    input_size = next(iter(loader))
    input_size = {k: v for k, v in input_size.items() if k != "label"}
    input_size = list(next(iter(input_size.values()))[:1].shape)

    architecture = torchinfo.summary(model, input_size=input_size, depth=4, verbose=0, col_names=["input_size", "kernel_size", "output_size", "num_params"])
    output_path = os.path.join(output_dir, 'architecture.json')
    with open(output_path, 'w') as f:
        f.write(str(architecture))




def plot_training_curves(df):
    """
    Plot training and validation loss and accuracy over 
    epochs. 
    
    Generates a two-panel figure showing loss and micro-accuracy 
    for training and validation sets across epochs. The epoch with the
    minimum validation loss is marked with a vertical dashed line.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame ordered by epoch containing columns for ``train loss``, 
        ``val loss``, ``train accuracy``, and ``val accuracy``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the loss and accuracy subplots.
    """

    # setup figure and axes for two subplots
    fig, ax = plt.subplots(ncols=2, figsize=(10,6))

    # create generator for epochs
    epochs = range(1, len(df)+1)

    # plot loss subplot...
    ax[0].plot(epochs, df['train loss'], lw=0.75, label='Train',)
    ax[0].plot(epochs, df['val loss'], lw=0.75, label='Validation')
    ax[0].set_ylabel('Loss')

    # plot micro-averaged accuracy
    ax[1].plot(epochs, df['train accuracy'], lw=0.75, label='Train')
    ax[1].plot(epochs, df['val accuracy'], lw=0.75, label='Validation')
    ax[1].set_ylabel('Accuracy (%)')

    # plot selected model at correct epoch
    for axes in ax:
        axes.axvline(x=df['val loss'].values.argmin()+1, linestyle='--', color='darkred', label='Selected')
        axes.legend(frameon=False)
        axes.set_xticks(epochs)
        axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
        axes.set_xlabel('Epochs')

    plt.suptitle(f"Training and Validation Curves", y=0.92)

    return fig