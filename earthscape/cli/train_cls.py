
from earthscape.utils.constants import ES_VERSION_SPLIT_DIR, SG_MAPPING
from earthscape.utils import set_seed, set_worker_seed
from earthscape.loaders import ESDataset_Classification, get_norm_stats
from earthscape.models import create_resnet_clf, create_vit_clf
from earthscape.train import BCEFocalLogits, architecture_to_json, train_model, plot_training_curves
from earthscape.evaluation import get_optimal_thresholds, test_model, get_global_metrics, get_class_metrics, plot_pr_roc_curves

import os
import glob
import argparse
import yaml
import datetime
import numpy as np
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim



def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a multilabel classification model using config.yml file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yml; copy will be saved to the experiment output directory for reproducibility.")
    parser.add_argument("--mode", type=str, choices=("train", "train-test", "train-test-cross"), required=True, help="Execution mode: train, training with validation set model selection; train-test, training & validation plus in-domain test; train-test-cross, training & validation plus in- and cross-domain tests.")
    parser.add_argument("--custom_output_dir", type=str, default=None, help="Customize output directory (optional). Default output directory will be ../experiments/classification/{encoder name}_{input feature names}_{date and time}.")
    return parser.parse_args()



def main():

    ##### open args & configs.yml...
    args = parse_args()
    with open(os.path.abspath(args.config_path), "r") as f:
        cfg = yaml.safe_load(f)


    ##### record start time
    t0 = datetime.datetime.now()
    cfg['experiment']['start_time'] = t0.strftime("%H:%M %m/%d/%Y")


    ##### set seeds...
    seed = cfg['experiment']['seed']
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


    ##### set device & add to configs.yml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['training']['device'] = str(device)


    ##### setup input feature dict...
    input_configs = cfg['data']['input_features']
    norm_stats_path = cfg['data']['split_files']['norm_stats']
    norm_stats_path = os.path.abspath(os.path.join(ES_VERSION_SPLIT_DIR, norm_stats_path))
    input_configs = get_norm_stats(norm_stats_path, input_configs)


    ##### pytorch datasets & dataloaders...
    patch_dirs = [os.path.abspath(p) for p in cfg['data']['patch_dirs']]
    splits = cfg["data"]["split_files"]

    # dataset parameters
    dataset_params = {'patch_dirs': patch_dirs, 'modalities': input_configs, 'normalize': cfg['training']['normalize']}

    # dataloader parameters...
    dl_default_params = cfg['training']['default_params']     # batch_size, drop_last, num_workers, persistent_workers, prefetch_factor 
    dl_custom_params = {'pin_memory': (device.type == 'cuda'), 'generator': g, 'worker_init_fn': set_worker_seed}
    dl_train_params = {'shuffle': cfg['training']['train_shuffle'], **dl_default_params, **dl_custom_params}
    dl_ttv_params = {'shuffle': cfg['training']['ttv_shuffle'], **dl_default_params, **dl_custom_params}

    # taining set
    train_patch_path = os.path.abspath(os.path.join(ES_VERSION_SPLIT_DIR, splits['train']))
    train_patches = gpd.read_file(train_patch_path)
    train_patch_ids = train_patches['patch_id'].to_list()
    train_dataset = ESDataset_Classification(train_patch_ids, augment=cfg['training']['train_augment'], **dataset_params)
    train_loader = DataLoader(train_dataset, **dl_train_params)

    # validation set
    val_patch_path = os.path.abspath(os.path.join(ES_VERSION_SPLIT_DIR, splits['val']))
    val_patches = gpd.read_file(val_patch_path)
    val_patch_ids = val_patches['patch_id'].to_list()
    val_dataset = ESDataset_Classification(val_patch_ids, augment=cfg['training']['ttv_augment'], **dataset_params)
    val_loader = DataLoader(val_dataset, **dl_ttv_params)

    # test set (in-domain) (optional)
    if args.mode == "train-test" or args.mode == "train-test-cross":
        test_patch_path = os.path.abspath(os.path.join(ES_VERSION_SPLIT_DIR, splits['test']))
        test_patches = gpd.read_file(test_patch_path)
        test_patch_ids = test_patches['patch_id'].to_list()
        test_dataset = ESDataset_Classification(test_patch_ids, augment=cfg['training']['ttv_augment'], **dataset_params)
        test_loader = DataLoader(test_dataset, **dl_ttv_params)

    # test set (cross-domain) (optional)
    if args.mode == "train-test-cross":
        cross_patch_path = os.path.abspath(os.path.join(ES_VERSION_SPLIT_DIR, splits['cross_test']))
        cross_patches = gpd.read_file(cross_patch_path)
        cross_patch_ids = cross_patches['patch_id'].to_list()
        cross_dataset = ESDataset_Classification(cross_patch_ids, augment=cfg['training']['ttv_augment'], **dataset_params)
        cross_loader = DataLoader(cross_dataset, **dl_ttv_params)


    ##### build model...
    # define encoder
    encoder = cfg['model']['encoder']

    # define input channels (baselines use one list of channels & will be stacked in order)
    inputs = cfg['data']['input_features']
    in_channels = sum(len(m['channels']) for m in inputs.values())
    cfg['model']['in_channels'] = in_channels

    # output size
    output_size = cfg['model']['output_size']

    # instantiate model...
    if encoder == 'resnet18':
        model = create_resnet_clf(architecture=encoder, in_channels=in_channels, out_features=output_size).to(device)
    elif encoder == 'resnet50':
        model = create_resnet_clf(architecture=encoder, in_channels=in_channels, out_features=output_size).to(device)
    elif encoder == 'vit':
        image_size = cfg['model']['image_size']
        model = create_vit_clf(in_channels=in_channels, num_classes=output_size, image_size=image_size).to(device)

    # compile model for optimal performance
    if cfg['model']['compile']:
        model = torch.compile(model)


    ##### loss...
    loss_name = cfg['loss']['name']
    if loss_name == 'bcefocal':
        loss_params = cfg['loss']['params']
        criterion = BCEFocalLogits(**loss_params).to(device)


    ##### optimizer...
    optimizer_name = cfg['optimizer']['name']
    if optimizer_name == 'adam':
        optim_params = cfg['optimizer']['params']
        optimizer = optim.Adam(model.parameters(), **optim_params)
    

    ##### output directory...
    if args.custom_output_dir is None:
        output_root = cfg['experiment']['output_root']
        input_names = list(cfg['data']['input_features'].keys())
        input_names = '_'.join(input_names)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{encoder}_{input_names}_{timestamp}"
        output_dir = os.path.abspath(os.path.join(output_root, dir_name))
    else:
        output_dir = os.path.abspath(args.custom_output_dir)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['experiment']['output_dir'] = output_dir


    ##### training...
    df_train = train_model(model, train_loader, val_loader, criterion, optimizer, device, cfg['training']['num_epochs'], output_dir)
    fig = plot_training_curves(df_train, output_dir)
    fig.savefig(os.path.abspath(os.path.join(output_dir, 'training_curves.png')), dpi=300, bbox_inches='tight', pad_inches=0)   

    t1 = datetime.datetime.now() 
    elapsed = (t1 - t0).total_seconds() / 60
    print(f"Training complete - Minutes: {elapsed:.2f}")


    ##### testing (optional)...
    if args.mode == "train-test" or args.mode == "train-test-cross":


        ##### load best model...
        model_path = glob.glob(os.path.abspath(os.path.join(output_dir, '*best*.pth')))[0]
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)


        ##### optimize thresholds...
        if cfg['training']['optimize_thresholds']:
            optimal_thresholds = get_optimal_thresholds(model, val_loader, device)
        else:
            optimal_thresholds = np.full(shape=cfg['model']['output_size'], fill_value=0.5)

        cfg['testing'] = {}
        cfg['testing']['thresholds'] = list(optimal_thresholds)


        ##### testing (in-domain)...
        class_cols = list(SG_MAPPING.keys())
        cfg['data']['labels'] = class_cols

        probabilities, targets = test_model(model, test_loader, device)

        df_global = get_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
        output_path = os.path.abspath(os.path.join(output_dir, 'id_global.csv'))
        df_global.to_csv(output_path, index=False)

        df_class = get_class_metrics(targets, probabilities, thresholds=optimal_thresholds, classes=class_cols)
        output_path = os.path.abspath(os.path.join(output_dir, 'id_class.csv'))
        df_class.to_csv(output_path, index=False)

        fig = plot_pr_roc_curves(targets, probabilities, class_cols)
        output_path = os.path.abspath(os.path.join(output_dir, 'id_pr_roc_curves.png'))
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)

        t2 = datetime.datetime.now() 
        elapsed = (t2 - t1).total_seconds() / 60
        print(f"Testing (in-domain) complete - Minutes: {elapsed:.2f}")


    ###### testing (cross-domain)...
    if args.mode == "train-test-cross":
        probabilities, targets = test_model(model, cross_loader, device)

        df_global = get_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
        output_path = os.path.abspath(os.path.join(output_dir, 'cd_global.csv'))
        df_global.to_csv(output_path, index=False)

        df_class = get_class_metrics(targets, probabilities, thresholds=optimal_thresholds, classes=class_cols)
        output_path = os.path.abspath(os.path.join(output_dir, 'cd_class.csv'))
        df_class.to_csv(output_path, index=False)

        fig = plot_pr_roc_curves(targets, probabilities, class_cols)
        output_path = os.path.abspath(os.path.join(output_dir, 'cd_pr_roc_curves.png'))
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)

        t3 = datetime.datetime.now() 
        elapsed = (t3 - t2).total_seconds() / 60
        print(f"Testing (cross-domain) complete - Minutes: {elapsed:.2f}")


    ##### save experiment metadata files...
    architecture_to_json(output_dir, model, val_loader)                          # model architecture file
    cfg_output_path = os.path.abspath(os.path.join(output_dir, 'config.yml'))    # config file used for experiment
    with open(cfg_output_path, "w") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    main()