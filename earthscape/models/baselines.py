
import torch
from torchvision.models import resnet18, resnet50, vit_b_16, swin_t
import segmentation_models_pytorch as smp




def create_resnet_cls(architecture, in_channels, out_features):
    """
    Instantiate a ResNet classification model with modified input and output layers.

    A torchvision ResNet backbone is created without pretrained weights.
    The first convolutional layer is replaced to accept ``in_channels``
    input channels, and the final fully connected layer is replaced to
    produce ``out_features`` outputs.

    Parameters
    ----------
    architecture : str
        ResNet architecture identifier. Supported values are
        {"resnet18", "resnet50"}.
    in_channels : int
        Number of input channels for the first convolutional layer.
    out_features : int
        Number of output features for the final fully connected layer.

    Returns
    -------
    model : torch.nn.Module
        Modified ResNet classifier.
    """

    # initialize correct model & no pre-trained weights
    if architecture == 'resnet18':
        model = resnet18(weights=None)
    elif architecture == 'resnet50':
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # modify first conv layer to accept 1:C input channels...
    old_conv1 = model.conv1
    new_conv1 = torch.nn.Conv2d(
        in_channels = in_channels, 
        out_channels = old_conv1.out_channels,
        kernel_size = old_conv1.kernel_size,
        stride = old_conv1.stride,
        padding = old_conv1.padding,
        dilation = old_conv1.dilation,
        groups = old_conv1.groups,
        bias = (old_conv1.bias is not None),
        padding_mode = old_conv1.padding_mode,
        )
    
    # modify final fc layer to produce correct K output classes...
    old_fc = model.fc
    new_fc = torch.nn.Linear(
        in_features = old_fc.in_features, 
        out_features = out_features, 
        bias = (old_fc.bias is not None)
        )

    # return modified resnet classifier... 
    model.conv1 = new_conv1
    model.fc = new_fc
    
    return model




def create_vit_cls(in_channels, num_classes, image_size):
    """
    Instantiate a ViT-B/16 classification model with modified input layers and output size.

    A torchvision ViT-B/16 model is created without pretrained weights. The patch
    embedding projection (``conv_proj``) is replaced to accept ``in_channels`` input
    channels, and the classification head is configured to produce ``num_classes``
    outputs via the constructor.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the patch embedding projection.
    num_classes : int
        Number of output classes.
    image_size : int or tuple of int, length 2
        Input image size. For ViT-B/16, each dimension should be divisible by 16.

    Returns
    -------
    model : torch.nn.Module
        Modified ViT classifier.
    """

    # initialize torchvision vit b/16 model with modified image_size and num_classes output
    model = vit_b_16(weights=None, image_size=image_size, num_classes=num_classes)

    # modify patch embedding conv to accept custom input channels
    old = model.conv_proj
    model.conv_proj = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    return model



def create_swin_cls(in_channels, num_classes):
    """
    Construct a Swin-Tiny classification model with custom input channels.

    The patch embedding layer is modified to accept `in_channels` and the
    classifier head is replaced to produce `num_classes` outputs.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input images.
    num_classes : int
        Number of classes in the classification task.

    Returns
    -------
    torch.nn.Module
        Modified Swin-Tiny model.
    """

    # instantiate model
    model = swin_t(weights=None)

    # modify first convolution in patch embedding layer...
    old = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(
        in_channels=in_channels, 
        out_channels=old.out_channels, 
        kernel_size=old.kernel_size, 
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
        )
    
    # modify classifier head...
    old_head = model.head
    model.head = torch.nn.Linear(
        in_features=old_head.in_features, 
        out_features=num_classes, 
        bias=(old_head.bias is not None)
        )
    
    return model




def create_unet_seg(in_channels, num_classes, encoder_name='resnet50'):
    """
    Construct a Unet segmentation model with custom input channels and predicted classes.
    
    Parameters
    -----------
    in_channels : int
        Number of channels in the input images.
    num_classes : int
        Number of classes in the segmentation task.
    encoder_name : str, default=resnet50
        Encoder name used as the backbone; names listed in `segmentation-models-pytorch`.

    Returns
    --------
    torch.nn.Module
        Customized Unet model.    
    """

    model = smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=None, 
        in_channels=in_channels, 
        classes=num_classes
        )
    
    return model




def create_deeplabv3p_seg(in_channels, num_classes, encoder_name='resnet50'):
    """
    Construct a DeeplabV3+ segmentation model with custom input channels and predicted classes.
    
    Parameters
    -----------
    in_channels : int
        Number of channels in the input images.
    num_classes : int
        Number of classes in the segmentation task.
    encoder_name : str, default=resnet50
        Encoder name used as the backbone; names listed in `segmentation-models-pytorch`.

    Returns
    --------
    torch.nn.Module
        Customized DeeplabV3+ model.    
    """

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name, 
        encoder_weights=None, 
        in_channels=in_channels, 
        classes=num_classes
        )

    return model




def create_segformer_seg(in_channels, num_classes, encoder_name='resnet50'):
    """
    Construct a Segformer segmentation model with custom input channels and predicted classes.
    
    Parameters
    -----------
    in_channels : int
        Number of channels in the input images.
    num_classes : int
        Number of classes in the segmentation task.
    encoder_name : str, default=resnet50
        Encoder name used as the backbone; names listed in `segmentation-models-pytorch`.

    Returns
    --------
    torch.nn.Module
        Customized Segformer model.    
    """

    model = smp.Segformer(
        encoder_name=encoder_name, 
        encoder_weights=None, 
        in_channels=in_channels, 
        classes=num_classes
        )
    
    return model