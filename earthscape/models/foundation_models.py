
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224
from torchgeo.models import Panopticon_Weights, panopticon_vitb14
from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base

from terratorch.registry import BACKBONE_REGISTRY


class DOFAClassifier(nn.Module):
    """
    DOFA wrapper for EarthScape multilabel classification.

    Parameters
    ----------
    wavelengths : sequence of float, default=[0.665, 0.560, 0.475, 0.842]
        Center wavelength of each input channel in micrometers (um) for Red, Green, 
        Blue, and Near-Infrared (NIR) bands. Must be in the same order as the channels 
        supplied by the dataset.

    num_classes : int, default=7
        Number of output classes.

    pretrained : bool, default=True
        If True, initialize the DOFA encoder with pretrained weights.

    image_size : int, default=224
        Spatial input size expected by the pretrained DOFA model.
        Inputs are resized to (image_size, image_size) in forward().
    """

    def __init__(self, wavelengths=[0.665, 0.560, 0.475, 0.842], num_classes=7, image_size=224):
        super().__init__()
        self.wavelengths = wavelengths
        self.image_size = image_size
        self.model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE, num_classes=num_classes)

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.model(x, self.wavelengths)



class PanopticonClassifier(nn.Module):
    """
    Panopticon wrapper for EarthScape multilabel classification.

    Parameters
    ----------
    channel_ids : sequence of int, default=[665, 560, 475, 842]
        Channel identifiers. Optical channels use wavelength in nanometers.
        Synthetic positive identifiers may be used for experimental non-spectral
        channels such as DEM or terrain features.

    num_classes : int, default=7
        Number of output classes.

    image_size : int, default=224
        Spatial input size used during Panopticon pretraining.
    """

    def __init__(self, channel_ids=[665, 560, 475, 842], num_classes=7, image_size=224):
        super().__init__()
        self.channel_ids = channel_ids
        self.image_size = image_size
        self.model = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=image_size)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        channel_ids = torch.tensor(self.channel_ids, device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        features = self.model({"imgs": x, "chn_ids": channel_ids})
        return self.classifier(features)



class CopernicusFMClassifier(nn.Module):
    """
    Copernicus-FM wrapper for EarthScape multilabel classification.

    Parameters
    ----------
    wavelengths : sequence of float, default=[665, 560, 475, 842]
        Center wavelengths in nanometers for Red, Green, Blue, and NIR.

    bandwidths : sequence of float or None, default=None
        Spectral bandwidths in nanometers.

    language_embed : torch.Tensor or None, default=None
        2048-dimensional Llama 3.2 embedding for non-spectral variables.

    input_mode : str, default="spectral"
        Either "spectral" or "variable".

    num_classes : int, default=7
        Number of output classes.

    image_size : int, default=224
        Spatial input size.
    """

    def __init__(self, wavelengths=[665, 560, 475, 842], bandwidths=[30, 35, 65, 115], language_embed=None, input_mode="spectral", num_classes=7, image_size=224):
        super().__init__()
        self.wavelengths = wavelengths
        self.bandwidths = bandwidths
        self.language_embed = language_embed
        self.input_mode = input_mode
        self.image_size = image_size
        self.model = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        metadata = torch.full((x.shape[0], 4), float("nan"), device=x.device)

        if self.input_mode == "spectral":
            features = self.model(x, metadata, wavelengths=self.wavelengths, bandwidths=self.bandwidths, input_mode="spectral")
        else:
            features = self.model(x, metadata, language_embed=self.language_embed.to(x.device), input_mode="variable")

        return self.classifier(features)




class TerraMindClassifier(nn.Module):
    """
    TerraMind wrapper for EarthScape multilabel classification.

    Parameters
    ----------
    modalities : sequence of str
        Native TerraMind input modalities. EarthScape options are
        ["RGB"], ["DEM"], or ["RGB", "DEM"].

    num_classes : int, default=7
        Number of output classes.

    image_size : int, default=224
        Spatial input size used by TerraMind.
    """

    def __init__(self, modalities=["RGB"], num_classes=7, image_size=224):
        super().__init__()
        self.modalities = modalities
        self.image_size = image_size
        self.model = BACKBONE_REGISTRY.build("terramind_v1_small", pretrained=True, modalities=modalities, merge_method="mean")
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        inputs = {}

        for modality in self.modalities:
            z = x[modality.lower()]

            if z.shape[-2:] != (self.image_size, self.image_size):
                z = F.interpolate(z, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            inputs[modality] = z

        features = self.model(inputs)[-1]
        features = features.mean(dim=1)

        return self.classifier(features)