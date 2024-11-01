

import torch
import torch.nn as nn
from torchvision import models
# import timm



class ResNextEncoder(nn.Module):
    def __init__(self, weights_config=None):
        super().__init__()
        self.encoder = models.resnext50_32x4d(weights=weights_config)       # initialize ResNext50 with user-defined weights
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])   # remove last two layers for classification
    
    def forward(self, x):
        # input shape - [batch_size, 3, 256, 256]
        output = self.encoder(x)   # encode input
        return output              # output shape - [batch_size, 2048, 8, 8]



class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)   # initialize attention
        self.pre_layernorm_q = nn.LayerNorm(embed_dim)           # initialize layer normalization for query (and key & value for self attention)
        self.pre_layernorm_kv = nn.LayerNorm(embed_dim)          # initialize layer normalization for key & value (for cross attention)

    def forward(self, q, kv=None):  
        # input shape (ResNext50) - [sequence length, batch size, embedding dimension]
        if kv is None:                 
            q = self.pre_layernorm_q(q)   # apply pre-attention layer normalization for self attention (one modality)
            kv = q                       # define key & value equal to query
        else:
            q = self.pre_layernorm_q(q)       # apply pre-attention layer normalization for query in cross attention
            kv = self.pre_layernorm_kv(kv)    # apply pre-attention layer normalization for key & value in cross attention
        
        attn_output, _ = self.self_attention(q, kv, kv)   # apply attention
        return attn_output                                # output shape - [sequence length, batch size, embedding dimension]



class MultilabelClassification(nn.Module):
    def __init__(self, modality_configs, encoder, attention_configs=None, embed_dim=2048):
        super().__init__()

        # preprocessing convolutional layers for each modality...
        self.modality_convs = nn.ModuleDict()
        for modality, input_channels in modality_configs.items():   # dictionary used for Dataset/Dataloader {modality name: [list of unique extensions for modality patch]} 
            num_channels = len(input_channels)                      # number of input channels for convolution
            self.modality_convs[modality] = nn.Conv2d(num_channels, 3, kernel_size=1)    # define convolution for each modality name

        # shared encoder...
        self.encoder = encoder

        # attention...
        self.attention_configs = attention_configs   # dictionary defining attention(s) to use {uniqe/informal name: [modality for query, modality for key & value]}
        if self.attention_configs:
            self.attention = nn.ModuleDict()
            self.post_layernorm = nn.ModuleDict()
            for attn_name, _ in attention_configs.items():
                self.attention[attn_name] = AttentionBlock(embed_dim=embed_dim, num_heads=8)   # define attention according to configs
                self.post_layernorm[attn_name] = nn.LayerNorm(embed_dim)
        
        # global average pooling...
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)

        # classification head...
        if not self.attention_configs:
            self.clf_num_input = len(modality_configs) * 2048 * 8 * 8     # ResNext50 size
        else:
            self.clf_num_input = len(attention_configs) * embed_dim             # ResNext50 with global average pooling size
        self.classifier = nn.Sequential(nn.Linear(self.clf_num_input, 512), 
                                        nn.ReLU(), 
                                        nn.Linear(512, 7))
    
    def forward(self, x):
        encoded_features = {}
        attention_ready = {}
        attended_features = {}

        # encode each modality...
        for modality, data in x.items():
            preprocessed = self.modality_convs[modality](data)    # standardize to 3 channels for encoder
            encoded = self.encoder(preprocessed)                  # shared encoder - [batch size, 2048, 8, 8]
            encoded_features[modality] = encoded                  # append modality name & encoded data to dictionary
        print('encoded')
        print(len(encoded_features))

        # classification WITHOUT attention...
        if not self.attention_configs:
            print('start')
            flattened_features = [encoded.reshape(encoded.size(0), -1) for encoded in encoded_features.values()]   # flatten encoder output for each modality - [2048 * 8 * 8]
            print(flattened_features[0].shape)
            concatenated_features = torch.cat(flattened_features, dim=1)                                           # concatenate all modalities to single tensor - [batch size, 2048 * 8 * 8]
            print(concatenated_features.shape)
            output = self.classifier(concatenated_features)       
            print(output.shape)                                                 # multilabel classification, output shape - [batch size, 7]
            return output

        # clasification WITH attention...
        else:
            # preprocess for attention...
            for modality, encoded in encoded_features.items():
                batch_size, channels, height, width = encoded.shape     
                attention_preprocessed = encoded.reshape(batch_size, channels, height * width).permute(2, 0, 1)   # reshape & permute - [8 * 8, batch size, channels]
                attention_ready[modality] = attention_preprocessed                                                # append attention name & preprocessed data to dictionary
            
            # apply attention...
            for attn_name, attn_modalities in self.attention_configs.items():

                # define query, key, & value for self attention
                if len(attn_modalities) == 1:
                    # access list of attention modalities (only one element in list for self attention)
                    q = attention_ready[attn_modalities[0]]      # define query (same as k,v)
                    kv = None
                
                # define query, key, & value for cross attention
                elif len(attn_modalities) == 2:
                    # access list of attention modalities (two elements in list for cross attention)
                    q = attention_ready[attn_modalities[0]]      # define query
                    kv = attention_ready[attn_modalities[1]]     # define key & value
                
                attn_output = self.attention[attn_name](q, kv)
                attn_output = attn_output + q                                # residual connection of attention output with input query
                attn_output = self.post_layernorm[attn_name](attn_output)
                attended_features[attn_name] = attn_output
            
            # fusion and pooling...
            # each attended feature has shape [sequence length, batch size, embedding dimensions]
            # concatenate features along embedding dimension - [sequence length, batch size, embedding dimensions * number attentions]
            fused_features = torch.cat([attended for attended in attended_features.values()], dim=-1)   
            fused_features = fused_features.permute(1, 2, 0)   # [batch size, embedding dimension * number attentions, sequence length]
            # global average pooling across sequence dimension [batch size, embed_dim * num_attentions, 1]...then remove last dimension of size 1
            fused_features = self.global_average_pool(fused_features).squeeze(-1)   # [batch size, embed_dim * num_attentions]
            
            # classification...
            output = self.classifier(fused_features)
            
            return output


