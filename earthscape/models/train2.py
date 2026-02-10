
# add parent directory to path to import custom scripts...
import sys, os
sys.path.append(os.path.abspath('..'))
from utils import *
from earthscape.models.dataloaders import *
from sgmapnet import *
from training_utils import *
from earthscape.models.focal_loss import *

# import other libraries...
import argparse
import datetime
import json
from pathlib import Path
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EarthScape model using a YAML config.")
    parser.add_argument("--config", type=str, default=r"../configs_template.yml", help="Path to YAML config file.")
    parser.add_argument("--override_dir", type=str, default=None, help="Custom override directory path for output.")
    return parser.parse_args()



def main():
    args = parse_args()

    with open(os.path.abspath(args.config), "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['experiment']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['train']['device'] = device
    g = torch.Generator()
    g.manual_seed(cfg["experiment"]["seed"])

    print(f"Training SGMap-Net with {cfg['model']['encoder']} backbone for {cfg['train']['num_epochs']} epochs...")

    output_root = cfg['experiment']['output_root']
    if args.override_dir is not None:
        encoder_name = cfg['model']['encoder']
        modality_names = list(cfg['data']['modalities'].keys())
        modality_str = '_'.join(modality_names)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{encoder_name}_{modality_str}_{timestamp}"
        output_dir = os.path.join(output_root, model_name)
    else:
        output_dir = args.override_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    cfg['experiment']['output_dir'] = output_dir



    modalities = cfg['data']['modalities']
    if cfg['data']['apply_normalizations']:
        modalities = get_norm_stats(cfg['paths']['stats_csv'], modalities)



    cfg_output_path = os.path.joinn(output_dir, 'config.yml')
    with open(cfg_output_path, "w") as f:
        yaml.safe_dump(cfg, f)

    

    # in-domain training...
    gdf = gpd.read_file(cfg['paths']['splits']['train'])
    train_patch_ids = gdf['patch_id'].to_list()
    train_ds = EarthScape_Dataset(train_patch_ids, cfg['paths']['patch_dirs'], modalities, augment=True)
    train_dl = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=False, num_workers=cfg['train']['num_workers'], pin_memory=(device.type=='cuda'), worker_init_fn=seed_worker, generator=g)

    # in-domain validation...
    gdf = gpd.read_file(cfg['paths']['splits']['val'])
    val_patch_ids = gdf['patch_id'].to_list()
    val_ds = EarthScape_Dataset(val_patch_ids, cfg['paths']['patch_dirs'], modalities, augment=False)
    val_dl = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, drop_last=False, num_workers=cfg['train']['num_workers'], pin_memory=(device.type=='cuda'), worker_init_fn=seed_worker, generator=g)

    # in-domain test set...
    gdf = gpd.read_file(cfg['paths']['splits']['test'])
    test_patch_ids = gdf['patch_id'].to_list()
    test_ds = EarthScape_Dataset(test_patch_ids, cfg['paths']['patch_dirs'], modalities, augment=False)
    test_dl = DataLoader(test_ds, batch_size=cfg['train']['batch_size'], shuffle=False, drop_last=False, num_workers=cfg['train']['num_workers'], pin_memory=(device.type=='cuda'), worker_init_fn=seed_worker, generator=g)

    # cross-domain test set...
    gdf = gpd.read_file(cfg['paths']['splits']['cross_test'])
    cross_patch_ids = gdf['patch_id'].to_list()
    cross_ds = EarthScape_Dataset(cross_patch_ids, cfg['paths']['patch_dirs'], modalities, augment=False)
    cross_dl = DataLoader(cross_ds, batch_size=cfg['train']['batch_size'], shuffle=False, drop_last=False, num_workers=cfg['train']['num_workers'], pin_memory=(device.type=='cuda'), worker_init_fn=seed_worker, generator=g)



    # model setup...
    model = SGMapNet_Classification(modalities, cfg['model']['encoder'], enable_attention=cfg['model']['enable_attention'], embedding_dim=cfg['model']['embedding_dim'], num_heads=cfg['model']['num_heads'], p_attn=cfg['model']['p_attn'], p_resid=cfg['model']['p_resid']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'], betas=cfg['optimizer']['betas'], weight_decay=cfg['optimizer']['weight_decay'])
    criterion = FocalLoss(alpha=cfg['loss']['alpha'], gamma=cfg['loss']['gamma'], reduction=cfg['loss']['reduction']).to(device)



    # training...
    df_train = train_model(model, train_dl, val_dl, criterion, optimizer, device, cfg['train']['num_epochs'], output_dir)
    fig = plot_training_curves(df_train, output_dir)
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    model_path = glob.glob(f"{output_dir}/*best_loss_epoch*.pth")[0]
    
    
    print("Training Complete!")



    # optimal F1 thresholds from validation set...
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    optimal_thresholds = calculate_optimal_thresholds(model, val_dl, device)



    # in-domain testing...

    print('In-domain inference...')
    probabilities, targets = test_model(model, test_dl, device)
    df_global = calculate_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_global.to_csv(f"{output_dir}/indomain_global.csv", index=False)

    df_class = calculate_class_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_class.to_csv(f"{output_dir}/indomain_class.csv", index=False)

    fig = plot_label_pr_roc_curves(targets, probabilities)
    fig.savefig(f"{output_dir}/indomain_pr.jpg")



    # cross-domain testing...

    print('Cross-domain inference...')
    probabilities, targets = test_model(model, cross_dl, device)
    df_cross_global = calculate_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_cross_global.to_csv(f"{output_dir}/cross_global.csv", index=False)

    df_cross_class = calculate_class_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_cross_class.to_csv(f"{output_dir}/cross_class.csv", index=False)

    fig = plot_label_pr_roc_curves(targets, probabilities)
    fig.savefig(f"{output_dir}/crossd_pr.jpg")

    print("Experiment complete!")


if __name__ == "__main__":
    main()