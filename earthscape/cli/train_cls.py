
from earthscape.utils.constants import SG_MAPPING
from earthscape.utils import set_seed, set_worker_seed, config_load, config_update
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

    parser.add_argument("--experiment_root", type=str, default=None, help="(Optional) Override with experiment output directory.")
    parser.add_argument("--seed", type=int, default=None, help="(Optional) Override with custom seed.")
    parser.add_argument("--compile", type=str, choices=('true', 'false'), default=None, help="(Optional) Override enable/disable torch.compile(model).")
    parser.add_argument("--encoder", type=str, choices=('resnet18', 'resnet50', 'vit'), default=None, help="(Optional) Override model.")
    parser.add_argument("--area_threshold", type=float, default=None, help="(Optional) Override class-area proportion threshold for target labels.")
    parser.add_argument("--batch_size", type=int, default=None, help="(Optional) Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="(Optional) Override learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=None, help="(Optional) Override weight decay for Adam.")
    parser.add_argument("--pos_weight", type=float, default=None, help="(Optional) Override pos_weight for BCE loss.")
    parser.add_argument("--gamma", type=float, default=None, help="(Optional) Override focal loss gamma.")
    parser.add_argument("--alpha", type=float, default=None, help="(Optional) Override focal loss alpha.")
    parser.add_argument("--reduction", type=str, choices=('mean', 'sum', 'none'), default=None, help="(Optional) Override reduction for loss.")
    parser.add_argument("--input", action="append", default=None, help="(Optional) Override inputs. Example: --input dem:dem.tif --input aerial:aerialr.tif,aerialg.tif,aerialb.tif")

    return parser.parse_args()




def main():

    ##### open args & configs.yml...
    args = parse_args()
    cfg = config_load(args.config_path)
    # with open(os.path.abspath(args.config_path), "r") as f:
    #     cfg = yaml.safe_load(f)


    ##### record start time
    t0 = datetime.datetime.now()
    cfg['experiment']['start_training'] = t0.strftime("%H:%M %m/%d/%Y")


    ##### reconcile optional args with configs...
    cfg = config_update(cfg, args)


    ##### set seeds...
    seed = cfg['experiment']['seed']
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


    ##### set device & add to configs.yml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['training']['device'] = str(device)


    ##### setup input feature dict, datasets, & dataloaders...
    
    # build input dict... 
    # input features -> {'dem': {'channels': ['dem.tif']}, ...}
    input_dict = cfg['data']['input']

    # path to training split statistics for normalization
    norm_stats_path = os.path.abspath(os.path.join(cfg['norm']['root'], cfg['norm']['glob']))
    norm_stats_path = glob.glob(norm_stats_path)[0]

    # final input features & noramlization stats for each channel
    # -> {'dem': {'channels': ['dem.tif'], 'mean': [float], 'sd': [float]}, ...}
    input_dict = get_norm_stats(norm_stats_path, input_dict)


    # dataset parameters...
    patch_dirs = [os.path.abspath(os.path.join(cfg['data']['root'], d)) for d in cfg['data']['dirs']]
    areas_path = os.path.abspath(glob.glob(os.path.join(cfg['labels']['root'], cfg['labels']['glob']))[0])
    ds_params = {
        'patch_dirs': patch_dirs, 
        'input_features': input_dict, 
        'areas_path': areas_path,
        'label_threshold': cfg['labels']['area_threshold'],
        'normalize': cfg['norm']['normalize'],
        }

    # dataloader parameters...
    dl_params = {
        **cfg['dataloader']['base'],
        'pin_memory': (device.type == 'cuda'),
        'generator': g, 
        'worker_init_fn': set_worker_seed
        }
    
    dl_train_params = {
        'shuffle': cfg['dataloader']['train']['shuffle'], 
        'drop_last': cfg['dataloader']['train']['drop_last'], 
        **dl_params
        }
    
    dl_eval_params = {
        'shuffle': cfg['dataloader']['eval']['shuffle'], 
        'drop_last': cfg['dataloader']['eval']['drop_last'], 
        **dl_params
        }

    # build taining set...
    train_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['train']))[0])
    train_patches = gpd.read_file(train_patch_path)
    train_patch_ids = train_patches['patch_id'].to_list()
    train_dataset = ESDataset_Classification(train_patch_ids, augment=cfg['dataloader']['train']['augment'], **ds_params)
    train_loader = DataLoader(train_dataset, **dl_train_params)

    # build validation set...
    val_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['val']))[0])
    val_patches = gpd.read_file(val_patch_path)
    val_patch_ids = val_patches['patch_id'].to_list()
    val_dataset = ESDataset_Classification(val_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
    val_loader = DataLoader(val_dataset, **dl_eval_params)

    # test set (in-domain) (optional)
    if args.mode == "train-test" or args.mode == "train-test-cross":
        test_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['test']))[0])
        test_patches = gpd.read_file(test_patch_path)
        test_patch_ids = test_patches['patch_id'].to_list()
        test_dataset = ESDataset_Classification(test_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
        test_loader = DataLoader(test_dataset, **dl_eval_params)

    # test set (cross-domain) (optional)
    if args.mode == "train-test-cross":
        cross_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['cross']))[0])
        cross_patches = gpd.read_file(cross_patch_path)
        cross_patch_ids = cross_patches['patch_id'].to_list()
        cross_dataset = ESDataset_Classification(cross_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
        cross_loader = DataLoader(cross_dataset, **dl_eval_params)


    ##### build model...
    # define encoder
    encoder = cfg['model']['encoder']

    # define input channels (baselines use one list of channels & will be stacked in order)
    in_channels = sum(len(m['channels']) for m in input_dict.values())
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
    output_root = cfg['experiment']['root']
    input_names = list(cfg['data']['input'].keys())
    input_names = '_'.join(input_names)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{encoder}_{input_names}_{timestamp}"
    output_dir = os.path.abspath(os.path.join(output_root, dir_name))
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['experiment']['output_dir'] = output_dir


    ##### training...
    # define epochs and train/validate model
    epochs = cfg['training']['num_epochs']
    baseline = len(input_dict.keys()) == 1
    df_train = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir, baseline=baseline, early_stop=cfg['early_stop'])

    # plot train/val loss & accuracy curves
    fig = plot_training_curves(df_train)
    output_path = os.path.abspath(os.path.join(output_dir, 'training_curves.png'))
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)   

    # record & display training time
    t1 = datetime.datetime.now() 
    cfg['experiment']['end_training'] = t1.strftime("%H:%M %m/%d/%Y")
    elapsed = (t1 - t0).total_seconds() / 60
    print(f"Training complete - Minutes: {elapsed:.2f}")


    ##### testing (optional)...
    if args.mode == "train-test" or args.mode == "train-test-cross":

        ##### load best model...
        model_path = glob.glob(os.path.abspath(os.path.join(output_dir, '*best*.pth')))[0]
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        cfg['experiment']['best_model'] = model_path

        class_cols = list(SG_MAPPING.keys())
        cfg['eval']['labels'] = class_cols


        ##### optimize thresholds...
        if cfg['eval']['optimize_clf_thresholds']:
            optimal_thresholds = get_optimal_thresholds(model, val_loader, device)
        else:
            optimal_thresholds = np.full(shape=cfg['model']['output_size'], fill_value=0.5)
        cfg['eval']['thresholds'] = optimal_thresholds.tolist()


        ##### testing (in-domain)...
        probabilities, targets = test_model(model, test_loader, device, baseline=baseline)
        
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