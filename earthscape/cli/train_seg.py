
##### imports...
# NOTE: suppress potential windows opening when saving various plots
import matplotlib
matplotlib.use("Agg")

# regular imports...
from earthscape.utils.constants import SG_MAPPING
from earthscape.utils import set_seed, set_worker_seed, config_load, config_update
from earthscape.loaders import ESDataset_Classification, get_norm_stats
from earthscape.models import create_unet_seg, create_deeplabv3p_seg, create_segformer_seg
from earthscape.train import seg_train_model, architecture_to_json, plot_training_curves
from earthscape.evaluation import test_model_seg, image_class_metrics_seg, image_overall_metrics_seg, overall_metrics_seg, overall_class_metrics_seg, plot_cm_seg
import argparse
import os
import glob
import yaml
import datetime
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim



def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a semantic segmentation model using config.yml file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yml; copy will be saved to the experiment output directory for reproducibility.")
    parser.add_argument("--mode", type=str, choices=("train", "train-test", "train-test-cross"), required=True, help="Execution mode: train, training with validation set model selection; train-test, training & validation plus in-domain test; train-test-cross, training & validation plus in- and cross-domain tests.")
    parser.add_argument("--experiment_root", type=str, default=None, help="(Optional) Override config output directory.")
    parser.add_argument("--seed", type=int, default=None, help="(Optional) Override config seed.")
    parser.add_argument("--model_name", type=str, choices=('unet', 'deeplabv3p', 'segformer'), default=None, help="(Optional) Override config model.")
    parser.add_argument("--encoder_name", type=str, choices=('resnet18', 'resnet34', 'resnet50', 'resnet101', 'mit_b0', 'mit_b2'), default=None, help="(Optional) Override config backbone.")
    parser.add_argument("--input", type=str, nargs="+", default=None, help="(Optional) Override input features. Example: --input dem:dem.tif  aerial:aerialr.tif,aerialg.tif,aerialb.tif ...")
    parser.add_argument("--batch_size", type=int, default=None, help="(Optional) Override config batch size.")
    parser.add_argument("--lr", type=float, default=None, help="(Optional) Override config learning rate.")
    parser.add_argument("--weight", type=float, default=None, help="(Optional) Override config weight decay.")
    parser.add_argument("--reduction", type=str, choices=('mean', 'sum', 'none'), default=None, help="(Optional) Override config reduction for loss.")
    parser.add_argument("--patience", type=int, default=None, help='(Optional) Override config early stopping epoch patience.')
    parser.add_argument("--min_delta", type=float, default=None, help='(Optional) Override config early stopping min_delta.')
    parser.add_argument("--warmup_epochs", type=int, default=None, help='(Optional) Override config early stopping warmup epochs.')

    return parser.parse_args()



def main():

    ##### open args & configs.yml...
    args = parse_args()
    cfg = config_load(args.config_path)


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
    # NOTE: input dict -> {'dem': {'channels': ['dem.tif']}, ...}
    input_dict = cfg['data']['input']

    # path to training split statistics for normalization
    norm_stats_path = os.path.abspath(os.path.join(cfg['norm']['root'], cfg['norm']['glob']))
    norm_stats_path = glob.glob(norm_stats_path)[0]

    # final input features & noramlization stats for each channel
    # NOTE: modified input dict -> {'dem': {'channels': ['dem.tif'], 'mean': [float], 'sd': [float]}, ...}
    input_dict = get_norm_stats(norm_stats_path, input_dict)


    ##### dataset parameters...
    patch_dirs = [os.path.abspath(os.path.join(cfg['data']['root'], d)) for d in cfg['data']['dirs']]
    areas_path = os.path.abspath(glob.glob(os.path.join(cfg['labels']['root'], cfg['labels']['glob']))[0])
    ds_params = {
        'patch_dirs': patch_dirs, 
        'input_features': input_dict, 
        'areas_path': areas_path,
        'label_threshold': cfg['labels']['area_threshold'],
        'normalize': cfg['norm']['normalize'],
        'task': cfg['experiment']['task']
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
    print('Training samples: ', len(train_loader.dataset))

    # build validation set...
    val_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['val']))[0])
    val_patches = gpd.read_file(val_patch_path)
    val_patch_ids = val_patches['patch_id'].to_list()
    val_dataset = ESDataset_Classification(val_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
    val_loader = DataLoader(val_dataset, **dl_eval_params)
    print('Validation samples: ', len(val_loader.dataset))

    # test set (in-domain) (optional)
    if args.mode == "train-test" or args.mode == "train-test-cross":
        test_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['test']))[0])
        test_patches = gpd.read_file(test_patch_path)
        test_patch_ids = test_patches['patch_id'].to_list()
        test_dataset = ESDataset_Classification(test_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
        test_loader = DataLoader(test_dataset, **dl_eval_params)
        print('Test samples (in-domain): ', len(test_loader.dataset))

    # test set (cross-domain) (optional)
    if args.mode == "train-test-cross":
        cross_patch_path = os.path.abspath(glob.glob(os.path.join(cfg['splits']['root'], cfg['splits']['glob']['cross']))[0])
        cross_patches = gpd.read_file(cross_patch_path)
        cross_patch_ids = cross_patches['patch_id'].to_list()
        cross_dataset = ESDataset_Classification(cross_patch_ids, augment=cfg['dataloader']['eval']['augment'], **ds_params)
        cross_loader = DataLoader(cross_dataset, **dl_eval_params)
        print('Test samples (cross-domain): ', len(cross_loader.dataset))


    ##### build model...
    # model name
    model_name = cfg['model']['model_name']
    encoder_name = cfg['model']['encoder_name']

    # define input channels (baselines use one list of channels & will be stacked in order)
    in_channels = sum(len(m['channels']) for m in input_dict.values())
    cfg['model']['in_channels'] = in_channels

    # output size
    output_size = cfg['model']['output_size']

    # instantiate model...
    if model_name == 'unet':
        model = create_unet_seg(in_channels=in_channels, num_classes=output_size, encoder_name=encoder_name).to(device)
    
    elif model_name == 'deeplabv3p':
        model = create_deeplabv3p_seg(in_channels=in_channels, num_classes=output_size, encoder_name=encoder_name).to(device)

    elif model_name == 'segformer':
        model = create_segformer_seg(in_channels=in_channels, num_classes=output_size, encoder_name=encoder_name).to(device)

    # # compile model for optimal performance
    # if cfg['model']['compile']:
    #     model = torch.compile(model)


    ##### loss...
    loss_name = cfg['loss']['name']
    if loss_name == 'crossentropyloss':
        loss_params = cfg['loss']['params']
        criterion = CrossEntropyLoss(**loss_params).to(device)


    ##### optimizer...
    optimizer_name = cfg['optimizer']['name']
    if optimizer_name == 'AdamW':
        optim_params = cfg['optimizer']['params']
        optimizer = optim.AdamW(model.parameters(), **optim_params)
    

    ##### output directory...
    output_root = cfg['experiment']['root']
    input_names = list(cfg['data']['input'].keys())
    input_names = '_'.join(input_names)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{input_names}_{timestamp}"
    output_dir = os.path.abspath(os.path.join(output_root, dir_name))
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['experiment']['output_dir'] = output_dir


    ##### training...
    # define epochs and train/validate model
    epochs = cfg['training']['num_epochs']
    baseline = len(input_dict.keys()) == 1
    early_stop = cfg['early_stop']
    warmup = cfg['optimizer']['warmup']
    cosine_decay = cfg['optimizer']['cosine_decay']
    df_train = seg_train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir, early_stop=early_stop, warmup=warmup, cosine_decay=cosine_decay, baseline=baseline)

    # plot train/val loss & accuracy curves
    fig = plot_training_curves(df_train, metric_col='dice')
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
        model.load_state_dict(state_dict)    # loads state dict into existing model object (restores best weights)
        cfg['experiment']['best_model'] = model_path

        class_cols = list(SG_MAPPING.keys())
        cfg['eval']['labels'] = class_cols


        ##### testing (in-domain)...
        # test evaluation
        predictions, masks = test_model_seg(model, test_loader, device, baseline)

        # calculate metrics & plots...
        patch_ids = test_dataset.ids
        df_image_class = image_class_metrics_seg(preds=predictions, masks=masks, patch_ids=patch_ids, class_cols=class_cols)
        df_image_overall = image_overall_metrics_seg(df_image_class)
        df_overall = overall_metrics_seg(df_image_class)
        df_overall_class = overall_class_metrics_seg(df_image_class)
        fig_raw = plot_cm_seg(predictions, masks, class_cols, mode='raw')
        fig_norm = plot_cm_seg(predictions, masks, class_cols, mode='row_norm')

        # save metrics & plots...
        df_image_class.to_csv(os.path.join(output_dir, 'id_img_class.csv'), index=False)
        df_image_overall.to_csv(os.path.join(output_dir, 'id_img.csv'), index=False)
        df_overall.to_csv(os.path.join(output_dir, 'id_overall.csv'), index=False)
        df_overall_class.to_csv(os.path.join(output_dir, 'id_overall_class.csv'), index=False)
        fig_raw.savefig(os.path.join(output_dir, 'id_cm_raw.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        fig_norm.savefig(os.path.join(output_dir, 'id_cm_norm.png'), dpi=300, bbox_inches='tight', pad_inches=0)

        # print testing elapsed time...
        t2 = datetime.datetime.now() 
        elapsed = (t2 - t1).total_seconds() / 60
        print(f"Testing (in-domain) complete - Minutes: {elapsed:.2f}")


    ###### testing (cross-domain)...
    if args.mode == "train-test-cross":
        # test evaluation
        predictions, masks = test_model_seg(model, cross_loader, device, baseline)

        # calculate metrics & plots...
        patch_ids = cross_dataset.ids
        df_image_class = image_class_metrics_seg(preds=predictions, masks=masks, patch_ids=patch_ids, class_cols=class_cols)
        df_image_overall = image_overall_metrics_seg(df_image_class)
        df_overall = overall_metrics_seg(df_image_class)
        df_overall_class = overall_class_metrics_seg(df_image_class)
        fig_raw = plot_cm_seg(predictions, masks, class_cols, mode='raw')
        fig_norm = plot_cm_seg(predictions, masks, class_cols, mode='row_norm')

        # save metrics & plots...
        df_image_class.to_csv(os.path.join(output_dir, 'cd_img_class.csv'), index=False)
        df_image_overall.to_csv(os.path.join(output_dir, 'cd_img.csv'), index=False)
        df_overall.to_csv(os.path.join(output_dir, 'cd_overall.csv'), index=False)
        df_overall_class.to_csv(os.path.join(output_dir, 'cd_overall_class.csv'), index=False)
        fig_raw.savefig(os.path.join(output_dir, 'cd_cm_raw.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        fig_norm.savefig(os.path.join(output_dir, 'cd_cm_norm.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        
        # print testing elapsed time...
        t3 = datetime.datetime.now() 
        elapsed = (t3 - t2).total_seconds() / 60
        print(f"Testing (cross-domain) complete - Minutes: {elapsed:.2f}\n")


    ##### save experiment metadata files...
    architecture_to_json(output_dir, model, val_loader)                          # model architecture file
    cfg_output_path = os.path.abspath(os.path.join(output_dir, 'config.yml'))    # config file used for experiment
    with open(cfg_output_path, "w") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    main()