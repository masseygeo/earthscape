

import os
import glob
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.plot import show

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import models




class MultiModalDataset(Dataset):
  def __init__(self, ids, data_dir, transform_rgb=None, transform_dem=None, transform_labels=None):
    self.ids = ids                               # list of patch IDs
    self.data_dir = data_dir                     # directory containing all data
    self.transform_rgb = transform_rgb           # transform for aerial rgb
    self.transform_dem = transform_dem           # transform for dem
    self.transform_labels = transform_labels     # transform for label

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    unique_id = self.ids[idx]

    ##### Aerial (RGB) image
    r_path = os.path.join(self.data_dir, f"{unique_id}_aerialr.tif")
    g_path = os.path.join(self.data_dir, f"{unique_id}_aerialg.tif")
    b_path = os.path.join(self.data_dir, f"{unique_id}_aerialb.tif")
    rgb_image = self.stack_images([r_path, g_path, b_path])           # create tensor of size [3, h, w]
    if self.transform_rgb:
      rgb_image = self.transform_rgb(rgb_image)                       # apply transform if provided

    ##### DEM image
    dem_path = os.path.join(self.data_dir, f"{unique_id}_dem.tif")
    dem_image = self.stack_images([dem_path])                         # create tensor of size [1, h, w]
    if self.transform_dem:
      dem_image = self.transform_dem(dem_image)                       # apply transform if provided

    ##### Label vector
    label_path = os.path.join(self.data_dir, f"{unique_id}_labels.csv")
    label = np.loadtxt(label_path)                                    # read label as array
    label = torch.from_numpy(label).unsqueeze(0)                      # create tensor of size [1, 7]
    label = label.type(torch.float)

    return {'rgb': rgb_image, 'dem': dem_image, 'label': label}

  @staticmethod
  def stack_images(paths_list):
    """
    Function to extract image arrays, stack if multiple images provided, and return tensor with shape [Channels, Height, Width].
    """
    # initialize list to hold image arrays
    src_arrays = []

    # iterate through image paths
    for path in paths_list:

      # open image
      with rasterio.open(path) as src:
        data = src.read(1)                       # read channel 1 as array (all input should be 1 channel)
        src_arrays.append(data)                  # append array to list
    image_array = np.stack(src_arrays, axis=0)   # stack image arrays along channel dimension
    return torch.from_numpy(image_array)         # return tensor with shape [channels, h, w]
  



class SensorFeatureExtractor(nn.Module):
  """Feature extractor for single modalities/remote sensing sensors. From Liu et al. (2023)."""
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
  """Shared feature extractor (ResNext50) for multiple modalities/remote sensing sensors."""
  # def __init__(self, input, target_classes):
  def __init__(self):
    super().__init__()

    # resnext50 architecture without pretrained weights
    self.resnet = models.resnext50_32x4d(weights=None)

    # modify first resnet convolutional layer to accept fused channel dimensions
    self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # remove last two classification layers (adaptive pooling & fully connected layers)
    self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

  def forward(self, x):
    x = self.resnet(x)
    return x





class MultiLabelClassification(nn.Module):
  """Multilabel classification head (same as ResNext50 output layers)."""
  def __init__(self, num_classes):
    super().__init__()

    # same output layers as resnext50 
    self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.fc1 = nn.Linear(2048, num_classes, bias=True)

  def forward(self, x):
    x = self.avgpool1(x)      # adaptive average pooling output - [batch, 2048, 1, 1]
    x = torch.flatten(x, 1)   # flatten to tensor - [batch, 2048]
    output = self.fc1(x)      # fully connected output - [batch, num_classes]
    return output






class FullModel(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    # separate feature extractors for sensors - RGB and DEM
    self.rgb_extractor = SensorFeatureExtractor(input_channels=3)   # in [batch, 3, 256, 256], out [batch, 16, 256, 256]
    self.dem_extractor = SensorFeatureExtractor(input_channels=1)   # in [batch, 1, 256, 256], out [batch, 16, 256, 256]

    self.shared_extractor = SharedFeatureExtractor()
    self.classifier = MultiLabelClassification(num_classes)

  def forward(self, rgb, dem):

    # separate feature extractors
    rgb = self.rgb_extractor(rgb)
    dem = self.dem_extractor(dem)

    # feature fusion
    fused = torch.cat((rgb, dem), dim=1)

    # shared feature extraction
    features = self.shared_extractor(fused)

    # multilabel classification
    output = self.classifier(features)

    return output




def get_norm_data(image_paths):
  """
  Function to calculate mean and standard deviation of 1-channel images.
  """
  total_sum = 0
  total_sum_squares = 0
  total_pixels = 0

  for path in image_paths:
    with rasterio.open(path) as src:
      data = src.read(1, masked=True)             # should not be any masked values, but just in case
      data = data.compressed()                    # this will remove any masked nodata values (if any)

      total_sum += np.sum(data)
      total_sum_squares += np.sum(data**2)
      total_pixels += data.size
      
  mean = total_sum / total_pixels
  var = (total_sum_squares / total_pixels) - mean**2
  sd = np.sqrt(var)
  return mean, sd




def prep_image_for_plot(batch_image):
  image = batch_image.clone().detach().numpy()
  min_val = image.min()
  max_val = image.max()
  image = (image - min_val) / (max_val-min_val)
  image = np.transpose(image, (1, 2, 0))
  return image




def train_epoch(model, train_loader, criterion, optimizer, device):

  # ensure the model can train and update
  model.train()

  # initialize metrics
  running_loss = 0.0
  correct = 0
  total_batches = 0
  total_predictions = 0

  # iterate through batches
  for batch in train_loader:

    # attach batch to device
    rgb_input = batch['rgb'].to(device)
    dem_input = batch['dem'].to(device)
    labels = batch['label'].squeeze(1).to(device)  # Assuming labels are of shape [batch_size, 1, 7] (squeeze to [batch_size, 7])

    # zero the gradients
    optimizer.zero_grad()

    # forward pass & backprop
    outputs = model(rgb_input, dem_input)
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
      rgb_input = batch['rgb'].to(device)
      dem_input = batch['dem'].to(device)
      labels = batch['label'].squeeze(1).to(device)

      outputs = model(rgb_input, dem_input)
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

  best_val_accuracy = 0.0
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

    # save best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
      best_val_accuracy = val_accuracy
      torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
      print(f"New best model saved with accuracy {best_val_accuracy:.2f}%")
      print('\n')


  return epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc