

##### Updated Multilabel Classification

import torch
import torch.nn as nn
import torchvision.models as models



class PreprocessingModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_preprocess = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.dem_preprocess = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb, dem):
        rgb_out = self.rgb_preprocess(rgb)    # shape: [batch, 3, H, W]
        dem_out = self.dem_preprocess(dem)    # shape: [batch, 3, H, W]
        return rgb_out, dem_out
    


class ResNextEncoderModule(nn.Module):
    def __init__(self, weights_config=None):
        super().__init__()
        self.encoder = models.resnext50_32x4d(weights=weights_config)       # initialize ResNext50 with user-defined weights
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])   # remove last two layers for classification
    
    def forward(self, x):
        # input shape - [batch_size, 3, H, W] - ResNext [B, 3, 256, 256]
        output = self.encoder(x)   # encode input
        return output              # output shape - [batch_size, 2048, 8, 8]
    



class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key_value):

        query = self.norm_query(query)
        key_value = self.norm_kv(key_value)
        
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        return attn_output
    




class FullModel2(nn.Module):
    def __init__(self, num_heads=8, embed_dim=2048, num_classes=7, weights_config=None):
        super().__init__()
        self.preprocess = PreprocessingModule()
        self.encoder = ResNextEncoderModule(weights_config=weights_config)
        self.attn_dem_to_rgb = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads)
        self.attn_rgb_to_dem = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, rgb, dem):
        
        # Step 1: Preprocess inputs
        rgb_pre, dem_pre = self.preprocess(rgb, dem)
        

        # Step 2: Extract features with shared ResNext encoder
        rgb_feat = self.encoder(rgb_pre)    # shape: [batch, 2048, 8, 8]
        dem_feat = self.encoder(dem_pre)    # shape: [batch, 2048, 8, 8]
        
        batch_size, channels, h, w = rgb_feat.size()
        seq_len = h * w  # e.g., 8*8 = 64
        

        # Step 3: Reshape features for the attention modules
        # New shape: [sequence_length, batch_size, embed_dim]
        rgb_seq = rgb_feat.view(batch_size, channels, seq_len).permute(2, 0, 1)
        dem_seq = dem_feat.view(batch_size, channels, seq_len).permute(2, 0, 1)
        

        # Step 4: Apply cross-attention between modalities
        attn_dem = self.attn_dem_to_rgb(dem_seq, key_value=rgb_seq)
        attn_rgb = self.attn_rgb_to_dem(rgb_seq, key_value=dem_seq)
        

        # Step 5: Aggregate features from each modality (e.g., average pooling over sequence tokens)
        attn_dem_avg = attn_dem.mean(dim=0)  # shape: [batch, embed_dim]
        attn_rgb_avg = attn_rgb.mean(dim=0)  # shape: [batch, embed_dim]
        

        # Step 6: Concatenate the aggregated features and classify
        fused_features = torch.cat([attn_dem_avg, attn_rgb_avg], dim=1)  # shape: [batch, embed_dim*2]
        logits = self.classifier(fused_features)  # shape: [batch, num_classes]
        
        return logits

# # Example usage:
# if __name__ == "__main__":
#     # Create dummy inputs:
#     # RGB: [batch, 3, 256, 256]
#     # DEM: [batch, 1, 256, 256]
#     batch_size = 2
#     rgb_input = torch.randn(batch_size, 3, 256, 256)
#     dem_input = torch.randn(batch_size, 1, 256, 256)
    
#     # Instantiate model
#     model = FullModel(num_classes=5)  # adjust num_classes as needed
#     output = model(rgb_input, dem_input)
#     print("Logits shape:", output.shape)  # Expected: [batch, num_classes]