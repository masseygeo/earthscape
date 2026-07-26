
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F




class Channel_Adapter(nn.Module):
    def __init__(self, input_channels, input_adapter="adapt"):
        super().__init__()

        self.input_adapter = input_adapter

        if input_adapter == "direct":
            self.adapter = nn.Identity()

        elif input_adapter == "adapt":
            if input_channels == 3:
                self.adapter = nn.Identity()
            else:
                self.adapter = nn.Sequential(
                    nn.Conv2d(input_channels, 3, kernel_size=1, bias=False),
                    nn.BatchNorm2d(3),
                    nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        return self.adapter(x)



class Encoder(nn.Module):
    def __init__(self, encoder="resnext50_32x4d", input_channels=3, image_size=None, pretrained=True, representation="pooled"):
        super().__init__()

        self.encoder_name = encoder
        self.representation = representation
        self.image_size = image_size

        if encoder == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet18(weights=weights)
            if input_channels != 3:
                self.encoder.conv1 = self._replace_conv2d(self.encoder.conv1, input_channels)
            self.encoder.fc = nn.Identity()
            self.embedding_dim = 512


        elif encoder == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnext50_32x4d(weights=weights)
            if input_channels != 3:
                self.encoder.conv1 = self._replace_conv2d(self.encoder.conv1, input_channels)
            self.encoder.fc = nn.Identity()
            self.embedding_dim = 2048


        elif encoder == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.encoder = models.vit_b_16(weights=weights)
            if input_channels != 3:
                self.encoder.conv_proj = self._replace_conv2d(self.encoder.conv_proj, input_channels)
            self.encoder.heads = nn.Identity()
            self.embedding_dim = 768


        elif encoder == "swin_t":
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            self.encoder = models.swin_t(weights=weights)
            if input_channels != 3:
                self.encoder.features[0][0] = self._replace_conv2d(self.encoder.features[0][0], input_channels)
            self.encoder.head = nn.Identity()
            self.embedding_dim = 768



    def forward(self, x):

        if self.image_size is not None:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)


        if self.encoder_name in ["resnet18", "resnext50_32x4d"]:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)

            if self.representation == "tokens":
                return x.flatten(2).transpose(1, 2)

            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)

            return x


        elif self.encoder_name == "vit_b_16":
            x = self.encoder._process_input(x)
            n = x.shape[0]
            cls = self.encoder.class_token.expand(n, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.encoder.encoder(x)

            if self.representation == "tokens":
                return x[:,1:,:]

            return x[:,0]


        elif self.encoder_name == "swin_t":
            x = self.encoder.features(x)
            B, H, W, C = x.shape

            if self.representation == "tokens":
                return x.reshape(B, H*W, C)

            return x.mean(dim=(1,2))
        

    def _replace_conv2d(self, conv, input_channels):
        return nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )



class Projection_Module(nn.Module):
    def __init__(self, input_dim, modality_names, embedding_dim=512):
        super().__init__()

        self.projector = nn.Linear(input_dim, embedding_dim)
        self.modality_embedding = nn.ParameterDict({name: nn.Parameter(torch.zeros(embedding_dim)) for name in modality_names})

    def forward(self, x, modality_name):
        z = self.projector(x)
        z = z + self.modality_embedding[modality_name]

        return z
    


class Attention_Module(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, p_attn=0.0, p_resid=0.0):
        super().__init__()

        self.ln_q = nn.LayerNorm(embedding_dim)
        self.ln_kv = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=p_attn,
            batch_first=True
            )

        self.dropout = nn.Dropout(p_resid)
        self.ln_out = nn.LayerNorm(embedding_dim)


    def forward(self, q, kv=None):

        if kv is None:
            kv = q

        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)

        attention_output, _ = self.attention(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=False
            )

        x = q + self.dropout(attention_output)
        x = self.ln_out(x)

        return x



class Aggregation_Module(nn.Module):
    def __init__(self, embedding_dim=512, modality_names=None):
        super().__init__()

        self.modalities = list(modality_names)
        self.v = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # x: {modality name: [B, N, D]}

        # pool tokens within each modality
        pooled = [x[name].mean(dim=1)for name in self.modalities]

        # [B, M, D]
        A = torch.stack(pooled, dim=1)

        # content-dependent score for each modality
        scores = (A * self.v).sum(dim=-1)

        # normalize across modalities
        weights = torch.softmax(scores, dim=1)

        # attention-weighted modality aggregation
        z_fused = (weights.unsqueeze(-1) * A).sum(dim=1)

        return z_fused



class Classification_Head(nn.Module):
    def __init__(self, input_dim, output_dim=7):
        super().__init__()

        self.clf = nn.Sequential(nn.Linear(input_dim, 512), 
                                 nn.ReLU(), 
                                 nn.Linear(512, output_dim))
    
    def forward(self, x):
        return self.clf(x)



class SGMapNet_Classification(nn.Module):
    def __init__(self, modality_configs, input_adapter="adapt", encoder="resnext50_32x4d", encoder_sharing="shared", embedding_fusion="cross_attention", image_size=None, pretrained=True, embedding_dim=512, num_heads=8, p_attn=0.0, p_resid=0.0, output_dim=7):
        super().__init__()

        self._validate_config(modality_configs=modality_configs, pretrained=pretrained, input_adapter=input_adapter,encoder_sharing=encoder_sharing, embedding_fusion=embedding_fusion)

        self.modalities = list(modality_configs.keys())
        self.encoder_sharing = encoder_sharing
        self.embedding_fusion = embedding_fusion


        # channel adapters...
        self.channel_adapters = nn.ModuleDict()
        for name in self.modalities:
            input_channels = len(modality_configs[name]["channels"])
            self.channel_adapters[name] = Channel_Adapter(
                input_channels=input_channels,
                input_adapter=input_adapter
                )


        # encoder output type...
        if embedding_fusion in ("self_attention", "cross_attention"):
            encoder_output = "tokens"
        else:
            encoder_output = "pooled"


        # shared encoder...
        if encoder_sharing == "shared":
            first_name = self.modalities[0]
            original_channels = len(modality_configs[first_name]["channels"])
            if input_adapter == "adapt":
                encoder_input_channels = 3
            else:
                encoder_input_channels = original_channels
            self.encoder = Encoder(
                encoder=encoder, 
                input_channels=encoder_input_channels, 
                image_size=image_size, 
                pretrained=pretrained, 
                representation=encoder_output
                )
            encoder_dim = self.encoder.embedding_dim


        # separate encoders...
        elif encoder_sharing == "separate":
            self.encoders = nn.ModuleDict()
            for name in self.modalities:
                original_channels = len(modality_configs[name]["channels"])
                if input_adapter == "adapt":
                    encoder_input_channels = 3
                else:
                    encoder_input_channels = original_channels
                self.encoders[name] = Encoder(
                    encoder=encoder,
                    input_channels=encoder_input_channels,
                    image_size=image_size,
                    pretrained=pretrained,
                    representation=encoder_output,
                    )
            encoder_dim = self.encoders[self.modalities[0]].embedding_dim


        # attention components...
        if embedding_fusion in ("self_attention", "cross_attention"):
            self.projection = Projection_Module(
                input_dim=encoder_dim,
                embedding_dim=embedding_dim,
                modality_names=self.modalities
                )


        # self-attention...
        if embedding_fusion == "self_attention":
            self.attention = Attention_Module(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                p_attn=p_attn,
                p_resid=p_resid
                )


        # cross-attention...
        elif embedding_fusion == "cross_attention":
            self.attention = nn.ModuleDict({
                name: Attention_Module(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        p_attn=p_attn,
                        p_resid=p_resid) 
                for name in self.modalities
                })
            self.aggregation = Aggregation_Module(embedding_dim=embedding_dim, modality_names=self.modalities)


        # classification-head input dimension...
        if embedding_fusion == "concat":
            classifier_input_dim = encoder_dim * len(self.modalities)
        elif embedding_fusion in ("self_attention", "cross_attention"):
            classifier_input_dim = embedding_dim
        else:
            classifier_input_dim = encoder_dim
        self.classifier = Classification_Head(input_dim=classifier_input_dim, output_dim=output_dim)


    def forward(self, x):
        # x: {modality name: [B, C, H, W]}

        encoded = {}

        # channel adaptation and encoding...
        for name in self.modalities:
            adapted = self.channel_adapters[name](x[name])
            if self.encoder_sharing == "shared":
                encoded[name] = self.encoder(adapted)
            elif self.encoder_sharing == "separate":
                encoded[name] = self.encoders[name](adapted)


        # 1. No mid-level fusion...
        # NOTE: includes early fusion (stacking) or single modality inputs
        if self.embedding_fusion == "none":
            z = next(iter(encoded.values()))
            return self.classifier(z)


        # 2. Mid-level concatenation...
        # NOTE: includes multiple inputs with shared or separate encoders
        elif self.embedding_fusion == "concat":
            z = torch.cat([encoded[name] for name in self.modalities], dim=-1)
            return self.classifier(z)


        # 3. Self attention...
        # NOTE: includes single modality inputs or early fusion by stacking
        elif self.embedding_fusion == "self_attention":
            name = self.modalities[0]
            tokens = self.projection(encoded[name], name)
            attended = self.attention(q=tokens, kv=None)
            z = attended.mean(dim=1)                          # pool spatial/patch tokens...
            return self.classifier(z)


        # 4. Cross attention...
        # NOTE: includes multiple inputs/embeddings from shared or separate encoders
        elif self.embedding_fusion == "cross_attention":
            projected = {name: self.projection(encoded[name], name) for name in self.modalities}
            attended = {}
            for query_name in self.modalities:
                q = projected[query_name]
                kv = torch.cat([projected[name] for name in self.modalities if name != query_name], dim=1)
                attended[query_name] = self.attention[query_name](q=q, kv=kv)
            z = self.aggregation(attended)
            return self.classifier(z)


    def _validate_config(self, modality_configs, pretrained, input_adapter, encoder_sharing, embedding_fusion):
        num_modalities = len(modality_configs)
        channel_counts = [len(config["channels"]) for config in modality_configs.values()]

        if pretrained and input_adapter == "direct":
            raise ValueError("pretrained=True is not compatible with channel_mode='direct'...")

        if embedding_fusion in ("none", "self_attention") and num_modalities != 1:
            raise ValueError(f"embedding_fusion='{embedding_fusion}' requires one input branch...")

        if embedding_fusion in ("concat", "cross_attention") and num_modalities < 2:
            raise ValueError(f"embedding_fusion='{embedding_fusion}' requires multiple input branches...")

        if encoder_sharing == "shared" and input_adapter == "direct":
            if len(set(channel_counts)) != 1:
                raise ValueError("A shared direct-input encoder requires all input branches to have the same number of channels...")