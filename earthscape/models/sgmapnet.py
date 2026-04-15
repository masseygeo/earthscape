
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F




class SGMapNet_Classification(nn.Module):
    def __init__(self, modality_configs, encoder, enable_attention=True, embedding_dim=512, num_heads=8, p_attn=0.0, p_resid=0.0):
        super().__init__()

        # list of modality names...
        self.modalities = list(modality_configs.keys())

        # standardize C channels per modality to 3...
        self.standardizer = Standardization_Module(modality_configs)

        # encode features
        if encoder == 'resnext':
            self.encoder = ResNext_Encoder()
        elif encoder == 'vit':
            self.encoder = ViT_Encoder()

        # projector & modality-specific embedding...
        self.projector = Projection_Module(self.encoder.hidden_size, embedding_dim, self.modalities)

        # attention...
        self.use_attention = bool(enable_attention) 
        if self.use_attention:
            self.attention = nn.ModuleDict({m: Attention_Module(embed_dim=embedding_dim, num_heads=num_heads, p_attn=p_attn, p_resid=p_resid) for m in self.modalities})
            self.aggregator = Aggregation_Module(embedding_dim, self.modalities)

        # classification...
        if not self.use_attention:
            clf_in = self.encoder.hidden_size * len(self.modalities)
        else:
            clf_in = embedding_dim
        self.classifier = Classification_Module(input_dim=clf_in)


    def forward(self, x):
        #  x: {modality name: [B, C, 256, 256]}

        # 1. standardize channel dimension - {m: [B, 3, 256, 256]}
        standardized = self.standardizer(x)     

        # 2. shared pre-trained encoder - {m: [B, 2048 or 768]}
        encoded = {m: self.encoder(standardized[m]) for m in self.modalities}

        # 3A. no attention...
        if not self.use_attention:

            # concatenate embeddings
            z_concat = torch.cat([encoded[m] for m in self.modalities], dim=1)

            # multilabel classification logits output [B, 7]
            return self.classifier(z_concat)
        
        # 3B. project embeddings & learn modality tokens
        tokens = self.projector(encoded)

        # 4. attention...
        attention_by_modality = {}
        for mod_name in self.modalities:
            q = tokens[mod_name]                                               # query modality
            kv_list = [tokens[m] for m in self.modalities if m != mod_name]    # all other modalities are kv

            # self attention
            if len(kv_list) == 0:
                y = self.attention[mod_name](q, kv=None)
            # cross attention
            else:
                kv = torch.cat(kv_list, dim=1)
                y = self.attention[mod_name](q, kv=kv)
            attention_by_modality[mod_name] = y

        # 6. attention-weighted aggregation across modalities
        z_fused = self.aggregator(attention_by_modality)

        return self.classifier(z_fused)




class Standardization_Module(nn.Module):
    def __init__(self, modality_configs):
        super().__init__()

        # modality_configs is user parameter with form:
        # {'modality name': {
        #         'channels': [list of file extensions for each channel image], 
        #         'mean': [list of means for normalization], 
        #         'sd': [list of standard deviations for norm.]}}
        self.modalities = list(modality_configs.keys())
        self.modality_convs = nn.ModuleDict()
        for k in self.modalities:
            input_channels = len(modality_configs[k]['channels'])
            self.modality_convs[k] = nn.Sequential(
                nn.Conv2d(input_channels, 3, kernel_size=1, bias=False), 
                nn.BatchNorm2d(3), 
                nn.ReLU(inplace=True))
    
    def forward(self, x):
        standardized = {}
        # for mod_name, data in x.items():
        #     standardized[mod_name] = self.modality_convs[mod_name](data)
        for k in self.modalities:
            standardized[k] = self.modality_convs[k](x[k])
        return standardized
    



class ResNext_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # resnext-50 backbone
        self.encoder = models.resnext50_32x4d(weights='DEFAULT')
        # # remove last two layers for custom MLP
        # self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # drops final clf head, but keeps adaptive global pooling
        self.encoder.fc = nn.Identity()

        # flattened output size
        # self.hidden_size = 2048 * 8 * 8
        self.hidden_size = 2048

    def forward(self, x):
                                    # input - [B, 3, 256, 256]
        return self.encoder(x)      # output - [B, 2048, 8, 8]  |  [B, 2048]    




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




class Projection_Module(nn.Module):
    def __init__(self, input_dim, d, modality_names):
        super().__init__()
        self.projector = nn.Linear(input_dim, d, bias=True)
        self.modality_embedding = nn.ParameterDict({mod_name: nn.Parameter(torch.zeros(d)) for mod_name in modality_names})

    def forward(self, x):
        # input - [B, 2048/768]
        projected = {}
        for mod_name, data in x.items():
            z = self.projector(data) + self.modality_embedding[mod_name]
            projected[mod_name] = z.unsqueeze(1)     # output - [B, 1, d]
        return projected
        



class Attention_Module(nn.Module):
    def __init__(self, embed_dim, num_heads, p_attn=0.0, p_resid=0.0):
        super().__init__()
        
        self.ln_q  = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.mha   = nn.MultiheadAttention(embed_dim, num_heads, dropout=p_attn, batch_first=True)
        self.drop  = nn.Dropout(p_resid)
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, q, kv=None, attn_mask=None, key_padding_mask=None):  
        # input - [B, 1, embedding dimension]
        # pre-layer normalization
        qn  = self.ln_q(q)
        kvn = qn if kv is None else self.ln_kv(kv)

        # attention
        attn_out, _ = self.mha(qn, kvn, kvn,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)

        # residual connection
        x = q + self.drop(attn_out)

        # post-layer normalization
        x = self.ln_out(x)
        # output - [B, 1, embedding dimension]
        return x
    



class Aggregation_Module(nn.Module):
    def __init__(self, d, modality_names):
        super().__init__()
        self.modalities = list(modality_names)
        
        # learnable attention scorerer
        self.v = nn.Parameter(torch.zeros(d))
    
    def forward(self, x):
        # input - [B, 1, d]

        # squeeze & stack
        A_list = [x[m].squeeze(1) for m in self.modalities]     # each modality is [B, d]
        A = torch.stack(A_list, dim=1)     # [B, N, d]

        # score attentions (sum over embedding dimension)
        s = (A * self.v).sum(dim=-1)      # [B, N]

        # softmax scores (weights) over modalities
        w = torch.softmax(s, dim=1)      # [B, N]

        # weighted sum - z = E w_i * a_i 
        z_fused = (w.unsqueeze(-1) * A).sum(dim=1)

        # output - [B, d]
        return z_fused




class Classification_Module(nn.Module):
    def __init__(self, input_dim, output_dim=7):
        super().__init__()

        # define simple MLP classifier head
        self.clf = nn.Sequential(nn.Linear(input_dim, 512), 
                                 nn.ReLU(), 
                                 nn.Linear(512, output_dim))
    def forward(self, x):
                                # input - [B, input_dim]
        return self.clf(x)      # output - [B, 7] | logits
