

# import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
import timm



class ResNextEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = models.resnext50_32x4d(weights=None)
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
        
        # encode...
        output = self.encoder(x)
        # shape - [batch_size, 2048, 8, 8]

        device = x.device
        output = output.to(device)
        
        # transform for attention...
        batch_size, channels, height, width = output.shape
        output = output.view(batch_size, channels, height * width)
        # shape - [batch_size, 2048, 8*8]
        output = output.permute(2, 0, 1)
        # shape - [64, batch_size, 2048]

        return output    



class EffNetEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = models.efficientnet_b0(weights=None)
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
        # encode...
        output = self.encoder(x)
        # shape - [batch_size, 1280, 8, 8]

        # transform for attention...
        batch_size, channels, height, width = output.shape
        output = output.view(batch_size, channels, height * width)
        # shape - [batch_size, 1280, 8*8]
        output = output.permute(2, 0, 1)
        # shape - [64, batch_size, 1280]

        return output   



class ViTEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # load model without pretrained weights
        self.encoder = timm.create_model('vit_base_patch16_256', 
                                        pretrained=False, 
                                        num_classes=0, 
                                        in_chans=input_channels)
    
    def forward(self, x):
        # encode...
        output = self.encoder(x)
        # shape - [batch_size, 256 (num patches), 768 (embedded dimsions for each patch)]

        # transform for attention...
        output = output.permute(1, 0, 2)
        # shape - [256, batch_size, 768]

        return output   



class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = None

    def forward(self, x):  

        # self attention of one modality x
        # x shape -  [sequence_length, batch_size, embed_dim]

        # set up attention based on input embedding dimension...
        if self.self_attention is None:
            embed_dim = x.shape[2]
            self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8).to(x.device)

        # perform self-attention...
        attn_output, _ = self.self_attention(x, x, x)  # Self-attention: q, k, v are the same
        
        return attn_output




class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = None

    def forward(self, q, k, v):
        # cross attention - q from one modality, k & v from another modality
        # q, k, & v shapes - [sequence_length, batch_size, embed_dim]
        
        # set up cross attention...
        if self.cross_attention is None:
            embed_dim = q.shape[2]
            self.cross_attention(embed_dim=embed_dim, num_heads=8)

        # perform cross attention...
        attn_output, _ = self.cross_attention(q, k, v) 

        return attn_output



class MultilabelMLP(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x):

        # input attention shape - [sequence length, batch size, embedding dimension]

        # set up fc1 to accept attention output from any network...
        if self.fc1 is None:
            x = x.permute(1, 0, 2)
            x = x.reshape(x.size(0), -1)
            input_dim = x.shape[1]
            self.fc1 = nn.Linear(input_dim, self.hidden_dim, bias=True).to(x.device)
            # x shape - [batch, flattened input]
        else:
            x = x.permute(1, 0, 2)
            x = x.reshape(x.size(0), -1)

        # classify...
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        # output shape - [batch size, 7]

        return output
    


class ClassificationModel(nn.Module):
    def __init__(self, encoders, attentions):
        """
        encoders : dict
            {modality name (corresponds with dataloader name) : encoder class}
        attentions : dict
            {informal attention name : (attention block, [modalitiy/modalities])}
        """
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)       # {modality1 name: encoder class, ...}

        self.attentions = nn.ModuleDict({name: block for name, (block, _) in attentions.items()})
        self.attention_configs  = {name: modalities for name, (_, modalities) in attentions.items()}

        self.classify = MultilabelMLP()
    
    def forward(self, data):

        device = next(iter(data.values())).device
        self.to(device)

        # step 1. encode each modality
        encoded_features = {}
        for modality_name, encoder in self.encoders.items():
            if modality_name in data:
                encoded_features[modality_name] = encoder(data[modality_name].to(device)).to(device)
        
        # Step 2: apply attention based on attention configuration
        attention_output = None

        for attn_name, attn_block in self.attentions.items():
            modalities = self.attention_configs[attn_name]

            if len(modalities) == 1:
                attention_output = attn_block(encoded_features[modalities[0]].to(device)).to(device)
            
            elif len(modalities) ==2:
                q_modality, kv_modality = modalities
                attention_output = attn_block(encoded_features[q_modality].to(device), 
                                              encoded_features[kv_modality].to(device), 
                                              encoded_features[kv_modality].to(device)).to(device)
            else:
                raise ValueError(f"Invalid configuration for attention block - {attn_name}")


        # step 3. apply final task head (classification or segmentation)
        output = self.classify(attention_output.to(device)).to(device)
        return output



# class Multilabel2(nn.Module):
#     def __init__(self, attnentions):
#         super.__init__()
#         self.conv1 = self.encoder.conv1 = nn.Conv2d(input_channels, 
#                                        64, 
#                                        kernel_size = (7, 7), 
#                                        stride = (2, 2), 
#                                        padding = (3, 3), 
#                                        bias = False)