
import torch
from torchvision.models import resnet18, resnet50, vit_b_16




def create_resnet_clf(arch, in_channels, out_classes):

    if arch == 'resnet18':
        model = resnet18(weights=None)
    elif arch == 'resnet50':
        model = resnet50(weights=None)
    
    new_conv1 = torch.nn.Conv2d(
        in_channels=in_channels, 
        out_channels = model.conv1.out_channels,
        kernel_size = model.conv1.kernel_size,
        stride = model.conv1.stride,
        padding = model.conv1.padding,
        dilation = model.conv1.dilation,
        groups = model.conv1.groups,
        bias = (model.conv1.bias is not None),
        padding_mode = model.conv1.padding_mode,
        )
    
    new_fc = torch.nn.Linear(
        in_features = model.fc.in_features, 
        out_features = out_classes, 
        bias = (model.fc.bias is not None)
        )

    model.conv1 = new_conv1
    model.fc = new_fc
    
    return model




def create_vit_clf(in_channels, num_classes, image_size):

    model = vit_b_16(weights=None, image_size=image_size, num_classes=num_classes)

    model.conv_proj = torch.nn.Conv2d(
        in_channels = in_channels,
        out_channels = model.out_channels,
        kernel_size = model.kernel_size,
        stride = model.stride,
        padding = model.padding,
        dilation = model.dilation,
        groups = model.groups,
        bias = (model.bias is not None),
        padding_mode = model.padding_mode,
    )

    return model