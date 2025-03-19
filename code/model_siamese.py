


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


  


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
