

import os
# import glob
import rasterio
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from rasterio.plot import show

import torch
# from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

# from torchvision import transforms
from torchvision.transforms import v2
from torchvision import models

from datetime import datetime





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


  


class SensorFeatureExtractor(nn.Module):
  """Feature extractor for single modalities/remote sensing sensors. Modified from Liu et al. (2023)."""
  def __init__(self, input_channels):
    super().__init__()

    self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(16)

    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(32)

    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)

    # self.transconv = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

  def forward(self, x):                    # input is [batch, channels, 256, 256]
    x = F.relu(self.bn1(self.conv1(x)))    # convolution, batch norm, relu - [batch, 16, 256, 256]
    x = F.relu(self.bn2(self.conv2(x)))    # convolution, batch norm, relu - [batch, 32, 256, 256]
    x = self.maxpool(x)                    # max pooling - [batch, 32, 128, 128]
    x = F.relu(self.bn3(self.conv3(x)))    # convolution, batch norm, relu - [batch, 64, 128, 128]
    x = F.relu(self.bn4(self.conv4(x)))    # convolution, batch norm, relu - [batch, 32, 128, 128]
    # x = self.transconv(x)                  # transposed convolution - [batch, 16, 256, 256]
    return x



class SharedFeatureExtractor(nn.Module):
  """Shared feature extractor for multiple modalities/remote sensing sensors. ResNext50 model without final adaptive average pooling and fully connected layers."""
  # def __init__(self, input, target_classes):
  def __init__(self, input_channels):
    super().__init__()

    # resnext50 architecture without pretrained weights
    self.resnet = models.resnext50_32x4d(weights=None)

    # modify first resnet convolutional layer to accept fused channel dimensions
    self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # remove last two classification layers (adaptive pooling & fully connected layers)
    self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

  def forward(self, x):
    x = self.resnet(x)
    return x





class MultiLabelClassification(nn.Module):
  """Multilabel classification head. Same as final two ResNext50 adaptive average pooling and fully connected layers."""
  def __init__(self, out_classes):
    super().__init__()

    # same output layers as resnext50 
    self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.fc1 = nn.Linear(2048, out_classes, bias=True)

  def forward(self, x):
    x = self.avgpool1(x)      # adaptive average pooling output - [batch, 2048, 1, 1]
    x = torch.flatten(x, 1)   # flatten to tensor - [batch, 2048] (PyTorch does this automatically in full implementation)
    output = self.fc1(x)      # fully connected output - [batch, num_classes]
    return output






class FullModel(nn.Module):
  "Full model using separate feature extractors for each modality, channel concatenation of all modalities, shared feature extractor for all modalities, and multilable classification."
  def __init__(self, out_classes, modality_channels):
    super().__init__()
    self.n_modalities = len(modality_channels.keys())
    self.feature_extractors = nn.ModuleDict({modality:SensorFeatureExtractor(input_channels) for modality, input_channels in modality_channels.items()})
    self.shared_extractor = SharedFeatureExtractor(self.n_modalities * 32)
    self.classifier = MultiLabelClassification(out_classes)

  def forward(self, modalities):

    # separate feature extractors
    extracted_features = []
    for name, data in modalities.items():
      if name in self.feature_extractors:
        features = self.feature_extractors[name](data)
        extracted_features.append(features)

    # feature fusion
    fused_features = torch.cat(extracted_features, dim=1)

    # shared feature extraction
    shared_features = self.shared_extractor(fused_features)

    # multilabel classification
    output = self.classifier(shared_features)

    return output





def train_epoch(model, train_loader, criterion, optimizer, device):

  t0 = datetime.now()

  # ensure the model can train and update
  model.train()

  # initialize metrics
  running_loss = 0.0
  correct = 0
  total_batches = 0
  total_predictions = 0

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

    # sigmoid to get probabilities and apply a threshold of 0.5
    predictions = (torch.sigmoid(outputs) >= 0.5).float()    # convert to binary predictions
    correct += (predictions == labels).sum().item()          # count total labels that were correctly predicted

  epoch_loss = running_loss / total_batches
  accuracy = correct / total_predictions * 100

  print(round((datetime.now()-t0).seconds / 60, 2), ' minutes')

  return epoch_loss, accuracy




def validate_epoch(model, val_loader, criterion, device):

  # set model to evaluate not update weights
  model.eval()

  # initialize metrics
  running_loss = 0.0
  correct = 0
  total_batches = 0
  total_predictions = 0

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

      # sigmoid to get probabilities and apply a threshold of 0.5
      predictions = (torch.sigmoid(outputs) >= 0.5).float()    # convert to binary predictions
      correct += (predictions == labels).sum().item()          # count total labels that were correctly predicted

  epoch_loss = running_loss / total_batches
  accuracy = correct / total_predictions * 100

  return epoch_loss, accuracy




def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, output_dir):

  # best_val_accuracy = 0.0
  best_val_loss = float('inf')
  epoch_train_loss = []
  epoch_train_acc = []
  epoch_val_loss = []
  epoch_val_acc = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # training
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_accuracy)
    print(f"Training Loss: {train_loss:.4f}     |   Training Accuracy: {train_accuracy:.2f}%")

    # validation
    val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
    epoch_val_loss.append(val_loss)
    epoch_val_acc.append(val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}   |   Validation Accuracy: {val_accuracy:.2f}%")

    print('\n')

    # save best model based on loss
    if val_loss > best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
      print(f"New best model saved with accuracy {val_accuracy:.2f}%")
      print('\n')

  return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc






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






from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def calculate_label_precision_recall_f1_aucroc(predictions, targets, threshold=0.5):
  predictions_binary = (predictions >= threshold).astype(int)
  
  precision = precision_score(targets, predictions_binary)
  recall = recall_score(targets, predictions_binary)
  f1 = f1_score(targets, predictions_binary)
  auc_roc = roc_auc_score(targets, predictions_binary)
  
  return precision, recall, f1, auc_roc





import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

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





from sklearn.metrics import average_precision_score, hamming_loss, accuracy_score

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





import pandas as pd
import seaborn as sns

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