
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class Standardization_Module(nn.Module):
    def __init__(self, modality_configs, std_kernel=1):
        super().__init__()

        # custom convolution layers per modality...
        # NOTE: modality_configs is user parameter with form:
        # {'modality name': {
        #         'extensions': [list of file extensions for each channel image], 
        #         'mean': [list of means for normalization], 
        #         'sd': [list of standard deviations for norm.]}}
        self.modality_convs = nn.ModuleDict() 
        for mod_name, data in modality_configs.items():

            # number of input channels per modality
            input_channels = len(data['channels'])

            # define conv layer per modality 
            self.modality_convs[mod_name] = nn.Conv2d(input_channels, 3, kernel_size=std_kernel)
    
    def forward(self, x):
        standardized = {}
        for mod_name, data in x.items():
            standardized[mod_name] = self.modality_convs[mod_name](data)
        return standardized
    



class ResNext_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # resnext-50 encoder backbone (remove last two layers for custom MLP)
        self.encoder = models.resnext50_32x4d(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # self.encoder.fc = nn.Identity() # drops final clf head, but keeps adaptive global pooling

        # flattened output size
        self.hidden_size = 2048 * 8 * 8

    def forward(self, x):
                                    # input - [B, 3, 256, 256]
        return self.encoder(x)      # output - [B, 2048, 8, 8]
    



class ViT_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # vit encoder backbone (remove classification head)
        self.encoder = models.vit_b_16(weights='DEFAULT')
        self.encoder.heads = nn.Identity()

        # flattened output size
        self.hidden_size = 768
    
    def forward(self, x):
                                    # input - [B, 3, 256, 256]

        resize = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.encoder(resize)      # output - [B, 768]
    



class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # define simple MLP classifier head
        self.clf = nn.Sequential(nn.Linear(input_dim, 512), 
                                 nn.ReLU(), 
                                 nn.Linear(512, 7))
    def forward(self, x):
                                # input - [B, input_dim]
        return self.clf(x)      # output - [B, 7] | logits




class Multilabel_Classification(nn.Module):
    def __init__(self, modality_configs, encoder, std_kernel=1):
        super().__init__()

        self.modalities = list(modality_configs.keys())

        # standardize C channels per modality to 3
        self.standardizer = Standardization_Module(modality_configs, std_kernel=std_kernel)

        # encode features
        if encoder == 'resnext':
            self.encoder = ResNext_Encoder()
        elif encoder == 'vit':
            self.encoder = ViT_Encoder()

        # MLP classification...
        # get total flattened size for all modalities
        flattened_size = self.encoder.hidden_size * len(self.modalities)
        self.classifier = Classifier(input_dim=flattened_size)

    def forward(self, x):
                                                    # input - [B, C, 256, 256]
        standardized = self.standardizer(x)         # output - ResNeXt: [B, 2048, 8, 8]   |   ViT: [B, 768]

        encoded = []
        for mod_name in self.modalities:
            z = self.encoder(standardized[mod_name])
            encoded.append(z)
            # output - ResNeXt: [B, 2048, 8, 8]   |   ViT: [B, 768]

        flattened = [e.reshape(e.size(0), -1) for e in encoded]
        concatenated = torch.cat(flattened, dim=1)
        output = self.classifier(concatenated)

        return output