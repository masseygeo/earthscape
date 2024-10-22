

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm



class ResNextEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = models.resnext50_32x4d(pretrained=False)
        # modify first convolution layer to accept custom input channels
        self.encoder.conv1 = nn.Conv2d(input_channels, 
                                       64, 
                                       kernel_size = (7, 7), 
                                       stride = (2, 2), 
                                       padding = (3, 3), 
                                       bias = False)
        # remove final pooling & classification layers
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
    
    def forward(self, x):
        return self.encoder(x)



class EffNetEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = models.efficientnet_b0(pretrained=False)
        # modify first convolution layer to accept custom input channels
        self.encoder.features[0][0] = nn.Conv2d(input_channels, 
                                                32, 
                                                kernel_size = 3, 
                                                stride = 2, 
                                                padding = 1, 
                                                bias  =False)
        # remove final pooling & classification layers
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
    
    def forward(self, x):
        return self.encoder(x)



class ViTEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = timm.create_model('vit_base_patch16_256', 
                                        pretrained=False, 
                                        num_classes=0, 
                                        in_chans=input_channels)
    
    def forward(self, x):
            return self.encoder(x)



class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dims, num_heads=num_heads)

    def forward(self, x):     
        # self attentionof one modality x
        # x shape -  [sequence_length, batch_size, embedding_dims]
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention: q, k, v are the same
        return attn_output



class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dims, num_heads=num_heads)

    def forward(self, q, k, v):
        # cross attention - q from one modality, k & v from another modality
        # q, k, & v shape - [sequence_length, batch_size, embedding_dims]
        attn_output, _ = self.multihead_attn(q, k, v) 
        return attn_output
    


class MultilabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Output layer for 7 classes

    def forward(self, x):
        # x shape - [batch, flattened input]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class ClassificationModel():
    def __init__(self, input_channels, embedding_dims, num_heads, mlp_input_dim):
        super().__init__()
        self.encoder = ResNextEncoder(input_channels = input_channels)
        self.attention = SelfAttentionBlock(embedding_dims = embedding_dims, num_heads = num_heads)
        self.mlp = MultilabelMLP(input_dim = mlp_input_dim, hidden_dim=512, output_dim=7)
    
    def forward(self, x):
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        flattened = None
        output = self.mlp(flattened)
        return output