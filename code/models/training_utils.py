
import json
import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchinfo

import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score




def get_norm_stats(stats_path, modality_configs):

    df = pd.read_csv(stats_path)

    for mod_name, data in modality_configs.items():
        data.update({'mean': [], 'sd': []})
        
        for c in data['channels']:
            
            if ('osm' in c) or ('nhd' in c) or ('geology' in c):
                data['mean'] = None
                data['sd'] = None
            
            else:
                row = df.loc[df['channel'] == c]
                data['mean'].append(row['mean'].item())
                data['sd'].append(row['sd'].item())
    
    return modality_configs




def training_log(model_name, output_dir, seed, train_patch_path, val_patch_path, test_patch_path, cross_test_patch_path, modality_configs, batch_size, num_epochs, optimizer, criterion, model):

    ##### collect setup info
    metadata = {}
    metadata['NAME'] = model_name
    metadata['DIRECTORY'] = output_dir
    metadata['SEED'] = seed


    ##### collect patches info
    patches_meta = {}
    patches_meta['training patches'] = train_patch_path
    patches_meta['validation patches'] = val_patch_path
    patches_meta['testing patches'] = test_patch_path
    patches_meta['cross-domain testing patches'] = cross_test_patch_path
    metadata['PATCHES'] = patches_meta


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


    ##### write log to json
    with open(f"{output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # ##### write model architecture to json
    # json_str = model_to_json(model)
    # with open(f"{output_dir}/model_architecture.json", "w") as f:
    #     f.write(json_str)

    ##### writee model summary to text file (model architecture, trainable parameters, kernel sizes)
    architecture = torchinfo.summary(model, depth=4, verbose=0, col_names=["num_params", "kernel_size"])
    with open(f"{output_dir}/model_summary.txt", 'w') as f:
        f.write(str(architecture))




# def model_to_json(model):
#     model_dict = {
#         "model_class": model.__class__.__name__,
#         "layers": []
#     }
#     for name, module in model.named_children():
#         model_dict["layers"].append({
#             "name": name,
#             "type": module.__class__.__name__,
#             "params": {k: v.shape for k, v in module.state_dict().items()}
#         })
#     return json.dumps(model_dict, indent=4, default=str)



def train_epoch(model, train_loader, criterion, optimizer, device):

    model.train()

    running_loss = torch.zeros((), device=device)
    correct_preds = torch.zeros((), device=device)
    total_batches = 0
    count = 0
  
    # iterate through batches...
    for batch in train_loader:

        labels = batch['label'].to(device, non_blocking=True).float()
        modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

        optimizer.zero_grad(set_to_none=True)

        logits = model(modalities)                # forward pass
        loss = criterion(logits, labels)          # calculate loss
        loss.backward()                           # back propagation
        optimizer.step()                          # update parameters

        running_loss += loss.detach()
        total_batches += 1
        count += labels.numel()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(labels.dtype)
        correct_preds += (preds == labels).sum()

    epoch_loss = (running_loss / total_batches).item()
    epoch_accuracy = (correct_preds / count * 100).item()

    return epoch_loss, epoch_accuracy




def validate_epoch(model, val_loader, criterion, device):

    model.eval()

    running_loss = torch.zeros((), device=device)
    correct_preds = torch.zeros((), device=device)
    total_batches = 0
    count = 0

    with torch.no_grad():

        for batch in val_loader:

            labels = batch['label'].to(device, non_blocking=True)
            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

            logits = model(modalities)
            loss = criterion(logits, labels)

            running_loss += loss.detach()
            total_batches += 1
            count += labels.numel()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            correct_preds += (preds == labels).sum()

    epoch_loss = (running_loss / total_batches).item()
    epoch_accuracy = (correct_preds / count * 100).item()

    return epoch_loss, epoch_accuracy




def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir):

    train_loss = []
    train_acc = []
    train_time = []

    val_loss = []
    val_acc = []

    best_val_loss = float('inf')


    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}")

        # training...
        t0 = datetime.now()
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        t1 = datetime.now()

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        tt = round((t1-t0).seconds / 60, 2)
        train_time.append(tt)
        training_str = f"TRAINING   -- Loss: {epoch_train_loss:.4f}  |  Accuracy: {epoch_train_acc:.2f}%  |  Time: {tt} mins."
        print(training_str)

        # validation...
        t2 = datetime.now()
        epoch_val_loss, epoch_val_acc = validate_epoch(model, val_loader, criterion, device)
        t3 = datetime.now()

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        val_str = f"VALIDATION -- Loss: {epoch_val_loss:.4f}  |  Accuracy: {epoch_val_acc:.2f}%  |  Time: {round((t3-t2).seconds / 60, 2)} mins."
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

    df = pd.DataFrame({'train loss': train_loss, 'train accuracy': train_acc, 'train time': train_time, 
                       'val loss': val_loss, 'val accuracy': val_acc})
    output_path = f"{output_dir}/training_log.csv"
    df.to_csv(output_path, index=False)

    return df





def test_model(model, test_loader, device):

    model.eval()

    probs = []
    targs = []

    with torch.inference_mode():

        for batch in test_loader:
        
            labels = batch['label'].to(device, non_blocking=True)

            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

            logits = model(modalities)
            p = torch.sigmoid(logits)
    
            probs.append(p.cpu())
            targs.append(labels.cpu())
    
    probabilities = torch.cat(probs, dim=0)
    targets = torch.cat(targs, dim=0)

    return probabilities, targets





def calculate_optimal_thresholds(model, val_loader, device):
  
    probabilities, targets = test_model(model, val_loader, device)

    probabilities = probabilities.numpy()
    targets = targets.numpy().astype(np.int32)

    optimal_thresholds = []

    for class_idx in range (probabilities.shape[1]):

        precision, recall, thresholds = precision_recall_curve(targets[:, class_idx], probabilities[:, class_idx])

        f1 = 2.0 * ((precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8))

        best_idx = np.argmax(f1)

        optimal_thresholds.append(thresholds[best_idx])

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