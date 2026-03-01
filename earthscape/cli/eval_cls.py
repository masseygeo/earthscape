
# from earthscape.utils.constants import ES_SPLIT_DIR, SG_MAPPING
from earthscape.utils import set_seed, set_worker_seed, config_load, config_update
from earthscape.loaders import ESDataset_Classification, get_norm_stats
from earthscape.models import create_resnet_clf, create_vit_clf
from earthscape.evaluation import test_model, get_global_metrics, get_class_metrics, plot_pr_roc_curves


import os
import glob
import argparse
import yaml
import datetime
# import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import DataLoader




def parse_args():
    parser = argparse.ArgumentParser(description="Use a multilabel classification model for evaluation.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to trained model config.yml file.")
    # parser.add_argument("--mode", type=str, choices=('predict', 'evaluate'), required=True, help="Predict labels or evaluate performance with labels.")
    parser.add_argument("--patch_ids_path", type=str, required=True, help="Path to GeoJSON file with test patch IDs; must contain column 'patch_id'.")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True, help="Directory paths containing data. Example: --data_dir ../data/dir1  ../data/dir2 ...")

    parser.add_argument("--experiment_root", type=str, default=None, help="(Optional) Override output directory.")
    parser.add_argument("--seed", type=int, default=None, help="(Optional) Override seed.")
    parser.add_argument("--batch_size", type=int, default=None, help="(Optional) Override batch size.")

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
    # input features -> {'dem': {'channels': ['dem.tif']}, ...}
    input_dict = cfg['data']['input']

    # path to training split statistics for normalization
    norm_stats_path = os.path.abspath(os.path.join(cfg['norm']['root'], cfg['norm']['glob']))
    norm_stats_path = glob.glob(norm_stats_path)[0]

    # final input features & noramlization stats for each channel
    # -> {'dem': {'channels': ['dem.tif'], 'mean': [float], 'sd': [float]}, ...}
    input_dict = get_norm_stats(norm_stats_path, input_dict)


    ##### build dataset & dataloaders...
    # dataset parameters...
    patch_dirs = [os.path.abspath(d) for d in args.data_dir]

    # if args.mode == 'evaluate':
    areas_path = os.path.abspath(glob.glob(os.path.join(cfg['labels']['root'], cfg['labels']['glob']))[0])
    # else:
    #     areas_path = None
    # cfg['eval']['mode'] = args.mode

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

    # build evaluation set...
    patches = gpd.read_file(args.patch_ids_path)
    patch_ids = patches['patch_id'].to_list()
    test_dataset = ESDataset_Classification(patch_ids, **ds_params)
    test_loader = DataLoader(test_dataset, **dl_params)


    ##### build model...
    encoder = cfg['model']['encoder']
    in_channels = cfg['model']['in_channels']
    output_size = cfg['model']['output_size']
    image_size = cfg['model']['image_size']

    # instantiate model...
    if encoder == 'resnet18':
        model = create_resnet_clf(architecture=encoder, in_channels=in_channels, out_features=output_size).to(device)
    elif encoder == 'resnet50':
        model = create_resnet_clf(architecture=encoder, in_channels=in_channels, out_features=output_size).to(device)
    elif encoder == 'vit':
        image_size = cfg['model']['image_size']
        model = create_vit_clf(in_channels=in_channels, num_classes=output_size, image_size=image_size).to(device)


    ##### output directory...
    output_root = cfg['experiment']['output_dir']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}"
    output_dir = os.path.abspath(os.path.join(output_root, "inference", dir_name))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['eval']['output_dir'] = output_dir


    ##### load model...
    model_path = os.path.abspath(cfg['experiment']['best_model'])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


    ##### inference...
    baseline = len(input_dict.keys()) == 1
    class_cols = cfg['eval']['labels']
    optimal_thresholds = [float(t) for t in cfg['eval']['thresholds']]
    probabilities, targets = test_model(model, test_loader, device, baseline=baseline)
    

    ##### save individual sample predictions...
    predictions = pd.DataFrame(data=test_dataset.ids, columns=['patch_id'])
    predictions[class_cols] = probabilities.detach().cpu().numpy()
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'))


    ##### optionally assess performance (if labels provided)...
    # if args.mode == 'evaluate':
    df_global = get_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    output_path = os.path.abspath(os.path.join(output_dir, 'global.csv'))
    df_global.to_csv(output_path, index=False)

    df_class = get_class_metrics(targets, probabilities, thresholds=optimal_thresholds, classes=class_cols)
    output_path = os.path.abspath(os.path.join(output_dir, 'class.csv'))
    df_class.to_csv(output_path, index=False)

    fig = plot_pr_roc_curves(targets, probabilities, class_cols)
    output_path = os.path.abspath(os.path.join(output_dir, 'idpr_roc_curves.png'))
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)


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