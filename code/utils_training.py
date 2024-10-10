

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, hamming_loss, accuracy_score



class FocalLoss(nn.Module):
  def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, logits_input, binary_target):
    bce_loss = F.binary_cross_entropy_with_logits(logits_input, binary_target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss

    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()
    else:
      return focal_loss



def train_epoch(model, train_loader, criterion, optimizer, device):

  # ensure the model can train and update
  model.train()

  # initialize metrics
  running_loss = 0.0
  correct_predictions = 0
  total_batches = 0
  total_predictions = 0
  all_targets = []
  all_binary_predictions = []

  # iterate through batches
  for batch in train_loader:

    labels = batch.pop('label').squeeze(1).to(device)
    modalities = {modality: data.to(device) for modality, data in batch.items()}

    # zero the gradients
    optimizer.zero_grad()

    # forward pass & backprop
    outputs = model(modalities)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # get results
    running_loss += loss.item()
    total_predictions += labels.numel()
    total_batches += 1

    all_targets.append(labels.cpu().numpy())

    predictions = torch.sigmoid(outputs)
    binary_predictions = (predictions >= 0.5).float()
    all_binary_predictions.append(binary_predictions.cpu().numpy())
    correct_predictions += (binary_predictions == labels).sum().item()

  all_binary_predictions = np.concatenate(all_binary_predictions, axis=0)
  all_targets = np.concatenate(all_targets, axis=0)

  epoch_loss = running_loss / total_batches
  accuracy = correct_predictions / total_predictions * 100
  macro_f1 = f1_score(all_targets, all_binary_predictions, average='macro') 

  return epoch_loss, accuracy, macro_f1




def validate_epoch(model, val_loader, criterion, device):

  # set model to evaluate not update weights
  model.eval()

  # initialize metrics
  running_loss = 0.0
  correct_predictions = 0
  total_batches = 0
  total_predictions = 0
  all_targets = []
  all_binary_predictions = []

  with torch.no_grad():
    for batch in val_loader:

      labels = batch.pop('label').squeeze(1).to(device)
      modalities = {modality: data.to(device) for modality, data in batch.items()}

      outputs = model(modalities)
      loss = criterion(outputs, labels)

      # get results
      running_loss += loss.item()
      total_predictions += labels.numel()
      total_batches += 1

      all_targets.append(labels.cpu().numpy())

      predictions = torch.sigmoid(outputs)
      binary_predictions = (predictions >= 0.5).float()
      all_binary_predictions.append(binary_predictions.cpu().numpy())
      correct_predictions += (binary_predictions == labels).sum().item()

  all_binary_predictions = np.concatenate(all_binary_predictions, axis=0)
  all_targets = np.concatenate(all_targets, axis=0)

  epoch_loss = running_loss / total_batches
  accuracy = correct_predictions / total_predictions * 100
  macro_f1 = f1_score(all_targets, all_binary_predictions, average='macro') 

  return epoch_loss, accuracy, macro_f1




def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir):

  # best_val_accuracy = 0.0
  best_val_loss = float('inf')
  epoch_train_loss = []
  epoch_train_acc = []
  epoch_train_macro_f1 = []
  epoch_val_loss = []
  epoch_val_acc = []
  epoch_val_macro_f1 = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    t0 = datetime.now()
    round((datetime.now()-t0).seconds / 60, 2)

    # training
    train_loss, train_accuracy, train_macro_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_accuracy)
    epoch_train_macro_f1.append(train_macro_f1)
    t1 = datetime.now()
    print(f"TRAINING --   Time: {round((t1-t0).seconds / 60, 2)} mins.  |  Loss: {train_loss:.4f}  |  Accuracy: {train_accuracy:.2f}%  |  Macro F1: {train_macro_f1:.2f}")

    # validation
    val_loss, val_accuracy, val_macro_f1 = validate_epoch(model, val_loader, criterion, device)
    epoch_val_loss.append(val_loss)
    epoch_val_acc.append(val_accuracy)
    epoch_val_macro_f1.append(val_macro_f1)
    t2 = datetime.now()
    print(f"VALIDATION -- Time: {round((t2-t1).seconds / 60, 2)} mins.  |  Loss: {val_loss:.4f}  |  Accuracy: {val_accuracy:.2f}%  |  Macro F1: {val_macro_f1:.2f}")

    print('\n')

    # save best model based on loss
    if val_loss > best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
      print(f"New best model saved with accuracy {val_accuracy:.2f}%")
      print('\n')

  return epoch_train_loss, epoch_train_acc, epoch_train_macro_f1, epoch_val_loss, epoch_val_acc, epoch_val_macro_f1






def test_model(model, test_loader, device, output_dir):
  
  all_predictions = []
  all_targets = []

  model.eval()

  with torch.no_grad():
    for batch in test_loader:
      rgb = batch['rgb'].to(device)
      dem = batch['dem'].to(device)
      labels = batch['label'].squeeze(1).to(device)

      all_targets.append(labels.cpu().numpy())

      outputs = model(rgb, dem)
      outputs = torch.sigmoid(outputs)
      all_predictions.append(outputs)
  
  all_predictions = np.concatenate(all_predictions)
  all_targets = np.concatenate(all_targets)

  return all_predictions, all_targets






def calculate_label_precision_recall_f1_aucroc(predictions, targets, threshold=0.5):
  predictions_binary = (predictions >= threshold).astype(int)
  
  precision = precision_score(targets, predictions_binary)
  recall = recall_score(targets, predictions_binary)
  f1 = f1_score(targets, predictions_binary)
  auc_roc = roc_auc_score(targets, predictions_binary)
  
  return precision, recall, f1, auc_roc






def plot_label_pr_roc_curves(target_label, predictions, targets):
  fig, ax = plt.subplots(ncols=2, figsize=(10,5))

  precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
  ax[0].plot(recall_vals, precision_vals, linewidth=2)
  ax[0].set_xlabel('Recall')
  ax[0].set_ylabel('Precision')
  ax[0].set_title(f'Precision-Recall Curve, {target_label}')

  fpr, tpr, _ = roc_curve(targets, predictions)
  ax[1].plot(fpr, tpr, linewidth=2)
  ax[1].set_xlabel('False Positive Rate')
  ax[1].set_ylabel('True Positive Rate')
  ax[1].set_title(f'Receiver Operating Curve, {target_label}')

  for axes in ax:
    axes.set_xlim(0,1)
    axes.set_ylim(0,1)

  return fig






def calculate_global_metrics(targets, predictions, threshold=0.5):

  predictions_binary = (predictions >= threshold).astype(int)
  
  macro_precision = precision_score(targets, predictions_binary, average='macro')
  macro_recall = recall_score(targets, predictions_binary, average='macro')
  macro_f1 = f1_score(targets, predictions_binary, average='macro')

  mean_ap = average_precision_score(targets, predictions, average='macro')
  h_loss = hamming_loss(targets, predictions_binary)
  subset_acc = accuracy_score(targets, predictions_binary)

  return macro_precision, macro_recall, macro_f1, mean_ap, h_loss, subset_acc





def calculate_optimal_thresholds(model, val_loader, device, output_dir):
  
  all_predictions = []
  all_targets = []

  model.eval()

  with torch.no_grad():
    for batch in val_loader:
      rgb = batch['rgb'].to(device)
      dem = batch['dem'].to(device)
      labels = batch['label'].squeeze(1).to(device)

      all_targets.append(labels.cpu().numpy())

      outputs = model(rgb, dem)
      outputs = torch.sigmoid(outputs)
      all_predictions.append(outputs)
  
  all_predictions = np.concatenate(all_predictions)
  all_targets = np.concatenate(all_targets)

  val_precision, val_recall, val_thresholds = precision_recall_curve(all_targets, all_predictions)

  val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)

  best_f1_idx = np.argmax(val_f1)
  best_f1_threshold = val_f1[best_f1_idx]

  return best_f1_threshold






def plot_class_distributions(patch_id_list, patch_count_path, patch_area_path, title):

    ##### calculate counts of occurrences
    df_count = pd.read_csv(patch_count_path)
    df_count = df_count.loc[df_count['patch_id'].isin(patch_id_list)]
    counts = df_count.iloc[:, 1:].sum(axis=0)
    counts = pd.DataFrame(counts) 

    ##### calculate areas in patches
    df_area = pd.read_csv(patch_area_path)
    df_area = df_area.loc[df_area['patch_id'].isin(patch_id_list)]
    df_area_long = df_area.iloc[:, 1:].melt(var_name='Geologic Map Unit', value_name='Proportion')

    ##### plot class distributions
    fig, ax = plt.subplots(ncols=2, figsize=(10,4))

    # counts...
    sns.barplot(ax=ax[0], data=counts, x=counts.index, y=0)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Class Occurrence', style='italic')

    # areas...
    # sns.violinplot(ax=ax[1], data=df_area_long, x='Geologic Map Unit', y='Proportion', 
    #                split=True, width=2)
    sns.boxplot(ax=ax[1], data=df_area_long, x='Geologic Map Unit', y='Proportion', 
                showfliers=False, fill=False, color='k', width=0.5, linewidth=1)
    
    sns.stripplot(ax=ax[1], data=df_area_long, x='Geologic Map Unit', y='Proportion', 
                  jitter=True, edgecolor='k', linewidth=0.2, alpha=0.03, facecolor='#3A6D8C', zorder=0)
    


    ax[1].set_xlabel('')
    ax[1].set_ylabel('Proportion')
    ax[1].set_title('Exposed Area', style='italic')

    plt.ylim(0,1)
    plt.suptitle(f"{title} (n={len(patch_id_list)})")
    plt.show()

    return fig