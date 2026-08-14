
# set global matplotlib backend...suppress potential windows opening when saving various plots...
import matplotlib
matplotlib.use("Agg")

# normal imports
from earthscape.utils import set_seed, set_worker_seed, config_load, config_update
from earthscape.loaders import ESDataset_Classification, get_norm_stats
from earthscape.models import create_resnet_cls, create_vit_cls, create_swin_cls, SGMapNet_Classification, SGMapNetGradCAMWrapper
from earthscape.evaluation import test_model, get_global_metrics, get_class_metrics, plot_pr_roc_curves

import os
import glob
import argparse
import yaml
import datetime
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import DataLoader


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Use a multilabel classification model for evaluation.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to trained model config.yml file.")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True, help="Directory paths containing data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output.")
    parser.add_argument("--patch_ids_path", type=str, default=None, help="(Optional) Path to GeoJSON file with test patch IDs (must contain column 'patch_id'). If left blank, must use --patch_ids flag.")
    parser.add_argument("--patch_ids", type=str, nargs="+", default=None, help="(Optional) List of patch IDs passed directly.")
    parser.add_argument("--seed", type=int, default=None, help="(Optional) Override seed.")
    parser.add_argument("--save_gradcams", action="store_true", help="(Optional) Save class-specific Grad-CAM arrays. No argument needed, only flag.")
    return parser.parse_args()



def main():

    ##### open args & configs.yml...
    args = parse_args()
    cfg = config_load(args.config_path)


    ##### record start time
    t0 = datetime.datetime.now()
    cfg['eval']['start'] = t0.strftime("%H:%M %m/%d/%Y")


    ##### reconcile optional args with configs...
    cfg = config_update(cfg, args)


    ##### set seeds...
    seed = cfg['experiment']['seed']
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


    ##### set device & add to configs.yml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['eval']['device'] = str(device)

    
    ##### build input dict... 
    # NOTE: input features dict -> {'dem': {'channels': ['dem.tif']}, ...}
    input_dict = cfg['data']['input']

    # path to training split statistics for normalization
    norm_stats_path = os.path.abspath(os.path.join(cfg['norm']['root'], cfg['norm']['glob']))
    norm_stats_path = glob.glob(norm_stats_path)[0]

    # final input features & noramlization stats for each channel
    # NOTE: input features dict -> {'dem': {'channels': ['dem.tif'], 'mean': [float], 'sd': [float]}, ...}
    input_dict = get_norm_stats(norm_stats_path, input_dict)


    ##### build dataset & dataloaders...
    # dataset parameters...
    patch_dirs = [os.path.abspath(d) for d in args.data_dir]
    areas_path = os.path.abspath(glob.glob(os.path.join(cfg['labels']['root'], cfg['labels']['glob']))[0])

    ds_params = {
        'patch_dirs': patch_dirs, 
        'input_features': input_dict, 
        'areas_path': areas_path,
        'label_threshold': cfg['labels']['area_threshold'],
        'normalize': cfg['norm']['normalize'],
        'augment': False
        }

    # dataloader parameters...
    dl_params = {
        **cfg['dataloader']['base'],
        'pin_memory': (device.type == 'cuda'),
        'generator': g, 
        'worker_init_fn': set_worker_seed,
        'shuffle': cfg['dataloader']['eval']['shuffle'], 
        'drop_last': cfg['dataloader']['eval']['drop_last']
        }

    # get patch IDs for evaluation...
    if args.patch_ids is not None:
        patch_ids = args.patch_ids

    elif args.patch_ids_path is not None:
        patches = gpd.read_file(args.patch_ids_path)
        patch_ids = patches["patch_id"].astype(str).to_list()

    else:
        raise ValueError("Provide either --patch_ids_path or --patch_ids.")

    # build dataset and dataloader...
    test_dataset = ESDataset_Classification(patch_ids, **ds_params)
    test_loader = DataLoader(test_dataset, **dl_params)


    ##### build model...
    model_name = cfg['model']['model_name']
    in_channels = cfg['model']['in_channels']
    output_size = cfg['model']['output_size']
    image_size = cfg['model']['image_size']

    # instantiate model...
    if model_name == 'resnet18':
        model = create_resnet_cls(architecture=model_name, in_channels=in_channels, out_features=output_size).to(device)
    
    elif model_name == 'resnet50':
        model = create_resnet_cls(architecture=model_name, in_channels=in_channels, out_features=output_size).to(device)
    
    elif model_name == 'vit':
        image_size = cfg['model']['image_size']
        model = create_vit_cls(in_channels=in_channels, num_classes=output_size, image_size=image_size).to(device)
    
    elif model_name == 'swin':
        model = create_swin_cls(in_channels=in_channels, num_classes=output_size)

    elif model_name == 'sgmap-net':
        params = {
            **cfg['model']['sgmapnet_params'],
            'encoder': cfg['model']['encoder_name']
            }
        model = SGMapNet_Classification(modality_configs=input_dict, output_dim=output_size, **params).to(device)


    ##### output directory...
    # output_root = cfg['experiment']['output_dir']
    output_root = args.output_dir
    input_label = list(input_dict.keys())[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join(output_root, f"{model_name}_{input_label}_{timestamp}"))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['eval']['output_dir'] = output_dir


    ##### load model...
    model_path = os.path.abspath(cfg['experiment']['best_model'])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


    ##### inference...
    baseline = model_name != "sgmap-net"
    class_cols = cfg['eval']['labels']
    optimal_thresholds = [float(t) for t in cfg['eval']['thresholds']]
    probabilities, targets = test_model(model, test_loader, device, baseline=baseline)


    ##### save individual sample predictions...
    predictions = pd.DataFrame(data=test_dataset.ids, columns=['patch_id'])
    predictions[class_cols] = probabilities.detach().cpu().numpy()
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


    ##### assess performance (if labels provided)...
    df_global = get_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    output_path = os.path.abspath(os.path.join(output_dir, 'global.csv'))
    df_global.to_csv(output_path, index=False)

    df_class = get_class_metrics(targets, probabilities, thresholds=optimal_thresholds, classes=class_cols)
    output_path = os.path.abspath(os.path.join(output_dir, 'class.csv'))
    df_class.to_csv(output_path, index=False)

    fig = plot_pr_roc_curves(targets, probabilities, class_cols)
    output_path = os.path.abspath(os.path.join(output_dir, 'idpr_roc_curves.png'))
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)


    ##### grad-cam (optional)...
    ##### optionally generate class-specific Grad-CAMs...
    if args.save_gradcams:

        # checks for valid experiment params...
        if model_name != "sgmap-net":
            raise ValueError("Grad-CAM export currently supports SGMap-Net only.")
        if len(input_dict) != 1:
            raise ValueError("Grad-CAM export currently expects one input branch, such as an early-stacked input.")
        if cfg["dataloader"]["eval"]["shuffle"]:
            raise ValueError("Set eval shuffle=False so CAM arrays match patch IDs.")
        if cfg["dataloader"]["eval"]["drop_last"]:
            raise ValueError("Set eval drop_last=False so all patches are processed.")

        # create output directory...
        gradcam_dir = os.path.join(output_dir, "gradcams")
        if not os.path.isdir(gradcam_dir):
            os.makedirs(gradcam_dir)

        # setup information for gradcam...
        input_name = next(iter(input_dict.keys()))                                                 # input modality name
        wrapped_model = SGMapNetGradCAMWrapper(model=model, input_name=input_name,).to(device)     # setup sgmap-net model
        wrapped_model.eval()                                                                       # set for evaluation
        target_layers = [model.encoder.encoder.layer4[-1]]                                         # last spatial layer in ResNet/ResNext encoder

        # iterate through batches in evaluation set; initialize index; open gradcam model...
        patch_index = 0
        with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
            for batch in test_loader:
                x = batch[input_name].to(device, non_blocking=True)

                # iterate through each sample in batch...
                for i in range(x.shape[0]):

                    # extract sample and patch ID; increase patch index...
                    sample = x[i:i + 1]
                    patch_id = str(test_dataset.ids[patch_index])
                    patch_index += 1

                    # extract metadata from sample...
                    height, width = sample.shape[-2:]
                    num_classes = len(class_cols)

                    # initialize array for gradcam
                    sample_cams = np.zeros((num_classes, height, width), dtype=np.float32)

                    # iterate through each class label in sample...
                    for class_index in range(num_classes):
                        grayscale_cam = cam(input_tensor=sample, targets=[ClassifierOutputTarget(class_index)])
                        sample_cams[class_index] = grayscale_cam[0]

                    # save gradcam array...
                    np.save(os.path.join(gradcam_dir, f"{patch_id}.npy"), sample_cams)


    ##### save updated config with evaluation parameters & finish...
    t1 = datetime.datetime.now() 
    cfg['eval']['end'] = t1.strftime("%H:%M %m/%d/%Y")
    cfg_output_path = os.path.abspath(os.path.join(output_dir, 'config.yml'))
    with open(cfg_output_path, "w") as f:
        yaml.safe_dump(cfg, f)
    elapsed = (t1 - t0).total_seconds() / 60
    print(f"Inference complete - Minutes: {elapsed:.2f}")#
    
    
if __name__ == "__main__":
    main()