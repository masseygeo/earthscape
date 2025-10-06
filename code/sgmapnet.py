

import torch
import torch.nn as nn
from torchvision import models



class Standardization_Module(nn.Module):
    def __init__(self, modality_configs, std_kernel=1):
        super().__init__()
        self.modalities = list(modality_configs.keys())
        self.modality_convs = nn.ModuleDict()                                                # initialize dict of standardization convolutions per modality
        for m, c in modality_configs.items():                                                # Dataset dict - {modality name: [list of path extensions for modality]}
            input_channels = len(c)                                                          # number of input channels for standardization module
            self.modality_convs[m] = nn.Conv2d(input_channels, 3, kernel_size=std_kernel)
    
    def forward(self, x):
        standardized = {}
        for modality, data in x.items():
            standardized[modality] = self.modality_convs[modality](data)
        return standardized
    



class ResNext_Encoder(nn.Module):
    def __init__(self, weights_config='DEFAULT'):
        super().__init__()
        self.encoder = models.resnext50_32x4d(weights=weights_config)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])      # remove last two layers for classification
        self.hidden_size = 2048 * 8 * 8                                        # flattened output size
    def forward(self, x):
                                                                               # [batch_size, 3, 256, 256]
        embeddings = self.encoder(x)                                           # [batch_size, 2048, 8, 8]
        return embeddings 
    



class ViT_Encoder(nn.Module):
    def __init__(self, weights_config='DEFAULT'):
        super().__init__()
        self.encoder = models.vit_b_16(weights=weights_config)
        self.encoder.heads = nn.Identity()
        self.hidden_size = 768
    def forward(self, x):
        embeddings = self.encoder(x)        # encode input - [batch_size, 3, 256, 256]
        return embeddings                   # output shape - [batch_size, ]
    



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, 512), 
                                 nn.RelU(), 
                                 nn.Linear(512, 7))
    def forward(self, x):
        return self.mlp(x)
        



class Multilabel_Classification(nn.Module):
    def __init__(self, modality_configs, encoder, weights_config='DEFAULT', std_kernel=1):
        super().__init__()

        self.standardized = Standardization_Module(modality_configs, std_kernel=std_kernel)


        self.encoder = encoder(weights_config=weights_config)

        per_modality_flattened_size = self.encoder.hidden_size
        total_flattened_size = per_modality_flattened_size * len(self.standardized.modalities)
        self.mlp = MLP(input_dim=total_flattened_size)

    
    def forward(self, x):
                                                                  # input - [B, C, 256, 256]
        standardized = self.standardized(x)                       # output - [B, 3, 256, 256]

        encoded = []
        for m in self.standardized.modalities:
            z = self.encoder(standardized[m])
            encoded.append(z)                                     # output - RexNeXt: [B, 2048, 8, 8] | ViT: [B, 768]

        flattened = [e.reshape(e.size(0), -1) for e in encoded]
        concatenated = torch.cat(flattened, dim=1)
        output = self.mlp(concatenated)

        return output