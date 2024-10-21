

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm



class Encoder(nn.Module):
    def __init__(self, input_channels, backbone):
        super().__init__()

        if backbone == 'ResNext':
            # load ResNeXt-50 model
            self.backbone = models.resnext50_32x4d(pretrained=False)
            # modify first convolution layer to accept specified input channels
            self.backbone.conv1 = nn.Conv2d(input_channels, 
                                            64, 
                                            kernel_size = (7, 7), 
                                            stride = (2, 2), 
                                            padding = (3, 3), 
                                            bias = False)
            # remove final pooling & classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        elif backbone == 'EfficientNet':
            # load EfficientNet model
            self.backbone = models.efficientnet_b0(pretrained=False)
            # modify first convolution layer to accept specified input channels
            self.backbone.features[0][0] = nn.Conv2d(input_channels, 
                                                     self.backbone.features[0][0].out_channels,
                                                     kernel_size = self.backbone.features[0][0].kernel_size,
                                                     stride = self.backbone.features[0][0].stride,
                                                     padding = self.backbone.features[0][0].padding,
                                                     bias  =False)
            # remove final pooling & classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        elif backbone == 'ViT':
            # load ViT model
            self.backbone = timm.create_model('vit_base_patch16_256', pretrained=False, num_classes=0)
            # modify first convolution layer to accept specified input channels
            self.backbone.patch_embed.proj = nn.Conv2d(input_channels, 
                                                       self.backbone.patch_embed.proj.out_channels,
                                                       kernel_size = self.backbone.patch_embed.proj.kernel_size,
                                                       stride = self.backbone.patch_embed.proj.stride,
                                                       padding = self.backbone.patch_embed.proj.padding,
                                                       bias = False)

    def forward(self, x):
        output = self.backbone(x)
        return output

