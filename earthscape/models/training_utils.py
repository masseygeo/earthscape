
import os
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torchinfo
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt




def get_norm_stats(stats_path, modality_configs):
    """
    Compute per-channel normalization statistics for each modality.

    Parameters
    ----------
    stats_path : str or pathlib.Path
        Path to a CSV file containing training-set statistics. The first column
        contains channel identifiers and the CSV includes ``mean`` and ``sd`` columns.
    modality_configs : dict
        Dictionary of modality configurations. Each value must contain a
        ``'channels'`` list specifying channel identifiers.

    Returns
    -------
    dict
        The same ``modality_configs`` object, modified in-place. Each modality is
        extended with ``'mean'`` and ``'sd'`` lists aligned with ``'channels'``.
        Channels that should not be normalized have ``None`` entries.
    """

    # read stats CSV to df
    df = pd.read_csv(stats_path)

    # iterate through values in modality_configs dictionary
    for _, data in modality_configs.items():

        # add two additional values for mean and sd
        data.update({'mean': [], 'sd': []})
        
        # iterate through channels in dictionary list named 'channels' containing modality file suffixes.
        for c in data['channels']:
            
            # categorical images should not have normalization stats (0 or 1)
            if ('osm' in c) or ('nhd' in c) or ('mask' in c):
                data['mean'].append(None)
                data['sd'].append(None)
            
            # other images should have normalization stats from training dataset
            else:
                row = df.loc[df[df.columns[0]] == c]
                data['mean'].append(row['mean'].item())
                data['sd'].append(row['sd'].item())
    
    # return modified dictionary
    return modality_configs





def train_epoch(model, train_loader, criterion, optimizer, device):
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

    Returns
    -------
    epoch_loss : float
        Mean loss across all training batches.
    epoch_accuracy : float
        Micro-averaged classification accuracy (percentage), computed across all
        label elements by thresholding sigmoid outputs at 0.5.
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
        modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

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




def validate_epoch(model, val_loader, criterion, device):
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
            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

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





def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir):
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

    Returns
    -------
    pandas.DataFrame
        DataFrame containing per-epoch training/validation loss and accuracy, plus
        training time in minutes.
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
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        t1 = datetime.now()

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        tt = round((t1-t0).total_seconds() / 60, 2)
        train_time.append(tt)
        training_str = f"TRAINING   -- Loss: {epoch_train_loss:.3f}  |  Accuracy: {epoch_train_acc:.2f}%  |  Time: {tt} mins."
        print(training_str)


        # validation...
        t2 = datetime.now()
        epoch_val_loss, epoch_val_acc = validate_epoch(model, val_loader, criterion, device)
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




def test_model(model, test_loader, device):
    """
    Run inference on a test set and return probabilities and targets.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used for inference.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding test batches as dicts with a ``'label'`` tensor and one
        or more modality tensors.
    device : torch.device
        Device used for model inference.

    Returns
    -------
    probabilities : torch.Tensor
        Concatenated sigmoid probabilities for all test samples (on CPU).
    targets : torch.Tensor
        Concatenated ground-truth labels for all test samples (on CPU).
    """

    # set model for evaluation
    model.eval()

    # initialize variables for inference...
    probs = []     # model probabilities
    targs = []     # class labels

    # iterate over batches...
    with torch.inference_mode():
        for batch in test_loader:
            
            # get labels & images from batch...
            labels = batch['label'].to(device, non_blocking=True)
            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

            # model inference from input modalities
            logits = model(modalities)

            # model probabilities from inference output logits
            p = torch.sigmoid(logits)

            # append model probabilities & true class labels for batch...
            probs.append(p.cpu())
            targs.append(labels.cpu())
    
    # get array of probabilities & targets...
    probabilities = torch.cat(probs, dim=0)
    targets = torch.cat(targs, dim=0)

    return probabilities, targets




def training_log(model_name, output_dir, seed, train_patches, val_patches, test_patches, cross_patches, modality_configs, batch_size, num_epochs, optimizer, criterion, model):

    ##### collect setup info
    metadata = {}
    metadata['NAME'] = model_name
    metadata['DIRECTORY'] = output_dir
    metadata['SEED'] = seed


    ##### collect modalitiy info
    modalities_meta = {}
    for mod_name, data in modality_configs.items():
        modalities_meta[mod_name] = {}
        modalities_meta[mod_name]['modalities'] = ', '.join(data['channels'])
        if not data['mean'] == None:
            modalities_meta[mod_name]['normalization means'] = ', '.join([str(i) for i in data['mean']])
            modalities_meta[mod_name]['normalization sd'] = ', '.join([str(i) for i in data['sd']])
    metadata['MODALITIES'] = modalities_meta


    ##### collect hyperparameters info
    hyper_meta = {}
    hyper_meta['batch size'] = batch_size
    hyper_meta['epochs'] = num_epochs
    hyper_meta['optimizer'] = type(optimizer).__name__
    hyper_meta['learning rate'] = optimizer.param_groups[0]['lr']
    hyper_meta['weight decay'] = optimizer.param_groups[0].get('weight_decay', None)
    hyper_meta['momentum'] = optimizer.param_groups[0].get('momentum', None)
    hyper_meta['loss'] = type(criterion).__name__
    if 'Focal' in hyper_meta['loss']:
        a = ', '.join([str(a.item()) for a in criterion.alpha.detach().ravel()])
        hyper_meta['alpha'] = a
        hyper_meta['gamma'] = criterion.gamma
    metadata['HYPERPARAMETERS'] = hyper_meta


    ##### collect patches info
    patches_meta = {}
    patches_meta['training patches'] = train_patches
    patches_meta['validation patches'] = val_patches
    patches_meta['testing patches'] = test_patches
    patches_meta['cross-domain testing patches'] = cross_patches
    metadata['PATCHES'] = patches_meta


    ##### write log to json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)


    ##### write model summary to text file (model architecture, trainable parameters, kernel sizes)
    architecture = torchinfo.summary(model, depth=4, verbose=0, col_names=["num_params", "kernel_size"])
    with open(f"{output_dir}/architecture.txt", 'w') as f:
        f.write(str(architecture))




def calculate_optimal_thresholds(model, loader, device, default_threshold=0.5):
    """
    Compute per-class decision thresholds that maximize F1 score. Thresholds 
    are selected independently for each class by evaluating the precision-recall 
    curve on the given dataset and choosing the threshold that yields the maximum 
    F1 score. Classes with no positive or no negative samples fall back to the 
    default threshold.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to generate prediction probabilities.
    loader : torch.utils.data.DataLoader
        DataLoader providing the dataset used for threshold optimization.
    device : torch.device
        Device on which the model is evaluated.
    default_threshold : float, optional
        Threshold used for classes with degenerate targets or undefined
        precision-recall curves. Default is 0.5.

    Returns
    -------
    optimal_thresholds : np.ndarray of shape (n_classes,)
        Array of per-class optimal thresholds that maximize F1 score.
    """

    # model inference with loader dataset
    probabilities, targets = test_model(model, loader, device)

    # make sure model outputs are on CPU and numpy arrays; labels also cast to int
    probabilities = probabilities.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy().astype(np.int32)

    # initialize variables...
    n_classes = probabilities.shape[1]                                               # number of classes
    optimal_thresholds = np.full(n_classes, default_threshold, dtype=np.float32)     # array of shape n_classes with default threshold
    eps = 1e-8                                                                       # value to prevent divide by zero error

    # iterate over probabilities...
    for class_idx in range(n_classes):

        # class targets & probabilities
        y_targs = targets[:, class_idx]
        p_model = probabilities[:, class_idx]

        # handle no positives or no negatives for a class; use default threshold
        if (y_targs.max() == 0) or (y_targs.min() == 1):
            continue
        
        # calculate precision, recall across thresholds
        precision, recall, thresholds = precision_recall_curve(y_targs, p_model)

        # handle empty thresholds
        if thresholds.size == 0:
            continue
        
        # calculate f1 score
        f1 = 2.0 * ((precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + eps))

        # find best threshold & add to optimal threshold array
        best_f1 = f1.max()
        best_idxs = np.where(np.isclose(f1, best_f1))[0]
        best_idx = best_idxs[-1]
        # best_idx = np.argmax(f1)
        optimal_thresholds[class_idx] = thresholds[best_idx]

    return optimal_thresholds










def calculate_global_metrics(targets, probabilities, thresholds):

    df = pd.DataFrame()

    targs = targets.numpy().astype(np.int32)
    probs = probabilities.numpy()
    thresholds = np.asarray(thresholds)

    if thresholds.ndim == 0:
        binary_preds = (probs >= thresholds).astype(np.int32)

    else:
        binary_preds = (probs >= thresholds[None, :]).astype(np.int32)
    
    df.loc[0, 'Precision'] = precision_score(targs, binary_preds, average='macro', zero_division=0.0)
    df.loc[0, 'Recall'] = recall_score(targs, binary_preds, average='macro', zero_division=0.0)
    df.loc[0, 'F1'] = f1_score(targs, binary_preds, average='macro', zero_division=0.0)
    df.loc[0, 'AUC'] = roc_auc_score(targs, probs, average="macro")
    df.loc[0, 'mAP'] = average_precision_score(targs, probs, average='macro')
    
    df.loc[0, 'Precision (Wt.)'] = precision_score(targs, binary_preds, average='weighted', zero_division=0.0)
    df.loc[0, 'Recall (Wt.)'] = recall_score(targs, binary_preds, average='weighted', zero_division=0.0)
    df.loc[0, 'F1 (Wt.)'] = f1_score(targs, binary_preds, average='weighted', zero_division=0.0)
    df.loc[0, 'AUC (Wt.)'] = roc_auc_score(targs, probs, average="weighted")
    df.loc[0, 'mAP (Wt.)'] = average_precision_score(targs, probs, average='weighted')
    df.loc[0, 'Accuracy (Micro)'] = (binary_preds == targs).mean()

    return df



def calculate_class_metrics(targets, probabilities, thresholds, classes=['af1', 'Qal', 'Qaf', 'Qat', 'Qc', 'Qca', 'Qr']):
    df = pd.DataFrame()

    thresholds = np.asarray(thresholds)

    for idx, (unit, thresh) in enumerate(zip(classes, thresholds)):

        probs = probabilities[:, idx].numpy()
        targs = targets[:, idx].numpy().astype(np.int32)
        binary_preds = (probs >= thresh).astype(np.int32)

        df.loc[idx, 'Class'] = f"{unit} ({str(round(thresh, 2))})"
        df.loc[idx, 'True'] = targs.sum()
        df.loc[idx, 'Predicted'] = binary_preds.sum()
        df.loc[idx, 'Accuracy'] = accuracy_score(targs, binary_preds)
        df.loc[idx, 'Precision'] = precision_score(targs, binary_preds)
        df.loc[idx, 'Recall'] = recall_score(targs, binary_preds)
        df.loc[idx, 'F1'] = f1_score(targs, binary_preds)
        df.loc[idx, 'AUC'] = roc_auc_score(targs, probs)
        df.loc[idx, 'AP'] = average_precision_score(targs, probs)
    
    return df




def plot_training_curves(df, output_dir):

    fig, ax = plt.subplots(ncols=2, figsize=(10,6))

    epochs = range(1, len(df)+1)

    ax[0].plot(epochs, df['train loss'], label='Train')
    ax[0].plot(epochs, df['val loss'], label='Validation')
    ax[0].set_ylabel('Focal Loss')

    ax[1].plot(epochs, df['train accuracy'], label='Train')
    ax[1].plot(epochs, df['val accuracy'], label='Validation')
    ax[1].set_ylabel('Accuracy (%)')

    for axes in ax:
        axes.axvline(x=df['val loss'].argmin()+1, linestyle='--', color='k', label='Best model')
        axes.legend(frameon=False)
        axes.set_xticks(epochs)
        axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
        axes.set_xlabel('Epochs')

    plt.suptitle(f"Training and Validation Curves", y=0.92)

    return fig




def plot_label_pr_roc_curves(true, pred, class_cols=['af1', 'Qal', 'Qaf', 'Qat', 'Qc', 'Qca', 'Qr']):

    precisions = []
    recalls = []
    fprs = []
    tprs = []

    for idx, unit in enumerate(class_cols):

        Y_true = true[:, idx]
        y_pred = pred[:, idx]

        p, r, _ = precision_recall_curve(Y_true, y_pred)
        precisions.append(p)
        recalls.append(r)

        fpr, tpr, _ = roc_curve(Y_true, y_pred)
        fprs.append(fpr)
        tprs.append(tpr)


    fig, ax = plt.subplots(ncols=2, figsize=(10,5))

    for idx in range(len(class_cols)):

        ax[0].plot(recalls[idx], precisions[idx], linewidth=1, label=class_cols[idx])
        ax[0].set_xlabel('Recall')
        ax[0].set_ylabel('Precision')
        ax[0].set_title('Precision-Recall Curve', style='italic')
    
        ax[1].plot(fprs[idx], tprs[idx], linewidth=1, label=class_cols[idx])
        ax[1].plot([0,1], [0,1], color='k', linestyle='--', lw=2)
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('Receiver Operating Curve', style='italic')
    
    for axes in ax:
        axes.set_xlim(0,1)
        axes.set_ylim(0,1)
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(1.15, -0.15), ncols=7, frameon=False, fontsize=8)

    return fig