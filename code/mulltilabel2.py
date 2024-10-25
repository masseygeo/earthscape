

import torch
import torch.nn as nn
from torchvision import models
# import timm



class ResNextEncoder(nn.Module):
    def __init__(self, weights_config=None):
        super().__init__()
        self.encoder = models.resnext50_32x4d(weights=weights_config)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
    
    def forward(self, x):
        # input shape - [batch_size, 3, 256, 256]
        output = self.encoder(x)
        return output               # shape - [batch_size, 2048, 8, 8]



class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, q, kv=None):  
        # input shpae(s) - [sequence_length, batch_size, embed_dim]
        # handle self attention (one modality) or cross attention (two modalities)...
        if kv is None:
            kv = q
        
        # perform attention and layer normalization
        attn_output, _ = self.self_attention(q, kv, kv)  
        attn_output = self.layer_norm(attn_output)
        return attn_output



class MultilabelClassification(nn.Module):
    def __init__(self, modality_configs, encoder, attention_configs=None):
        super().__init__()
        self.modality_convs = nn.ModuleDict()
        for modality, in_channels in modality_configs.items():
            num_channels = len(in_channels)
            self.modality_convs[modality] = nn.Conv2d(num_channels, 3, kernel_size=1)

        self.encoder = encoder

        self.attention_configs = attention_configs
        self.attention = AttentionBlock(embed_dim=2048, num_heads=8)

        self.classifier_in_chans = len(modality_configs) * 2048 * 8 * 8
        self.classifier = nn.Sequential(nn.Linear(self.classifier_in_chans, 1024), 
                                        nn.ReLU(), 
                                        nn.Linear(1024, 512), 
                                        nn.ReLU(), 
                                        nn.Linear(512, 7))
    
    def forward(self, x):
        encoded_features = {}
        attention_ready = {}
        attention_maps = {}

        # encode each modality...
        for modality, data in x.items():
            preprocessed = self.modality_convs[modality](data)
            encoded = self.encoder(preprocessed)
            encoded_features[modality] = encoded
        
        # classification WITHOUT attention...
        if not self.attention_configs:
            flattened_features = [encoded.reshape(encoded.size(0), -1) for encoded in encoded_features.values()]
            concatenated_features = torch.cat(flattened_features, dim=1)
            output = self.classifier(concatenated_features)
            return output

        # clasification WITH attention...
        else:

            # process for attention
            for modality, encoded in encoded_features.items():
                batch_size, channels, height, width = encoded.shape     
                attention_preprocessed = encoded.view(batch_size, channels, height * width)
                attention_preprocessed = attention_preprocessed.permute(2, 0, 1)  
                attention_ready[modality] = attention_preprocessed                   # [h*w, batch_size, channels]
            
            # apply attention...
            for attn_name, attn_modalities in self.attention_configs.items():
                
                # self attention
                if len(attn_modalities) == 1:
                    q = attention_ready[attn_modalities[0]]
                    kv = attention_ready[attn_modalities[0]]
                    attention_maps[attn_name] = self.attention(q, kv)      # [sequence_length, batch_size, embed_dim]
                
                # cross attention
                elif len(attn_modalities) == 2:
                    q = attention_ready[attn_modalities[0]]
                    kv = attention_ready[attn_modalities[1]]
                    attention_maps[attn_name] = self.attention(q, kv)      # [sequence_length, batch_size, embed_dim]
            
            # classification...
            flattened_features = [attn.permute(1, 0, 2) for attn in attention_maps.values()]       # [batch_size, sequence_length, embed_dim]
            flattened_features = [attn.reshape(attn.size(0), -1) for attn in flattened_features]   # [batch_size, sequence_length * embed_dim]
            concatenated_features = torch.cat(flattened_features, dim=1)
            output = self.classifier(concatenated_features)
            return output


