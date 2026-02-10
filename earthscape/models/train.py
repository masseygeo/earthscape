
# add parent directory to path to import global utils
import sys, os
sys.path.append(os.path.abspath('..'))
from utils import set_seed
from earthscape.models.dataloaders import *
from sgmapnet import *
from training_utils import *
from earthscape.models.focal_loss import *


import argparse
import json
from pathlib import Path
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim





def parse_args():
    parser = argparse.ArgumentParser(description="Train and/or test a multilabel classification model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mode", type=str, choices=["train", "test"], default='train', help="Training script mode: train=training, validation, and testing; test=inference only.")

    parser.add_argument("--model_path", type=Path, default=None, help="Path to model checkpoint (.pth). Required when --mode test.")

    # parser.add_argument("--output_dir", type=Path, default=None, help="Optional output directory. If not set: train→timestamped run; test→<ckpt_parent>/eval_<timestamp>.")

    parser.add_argument("--seed", type=int, default=111, help="Seed for reprodcibility.")
    
    parser.add_argument("--patch_dirs", type=Path, nargs='+', help="List of directory(s) conttaining patches for training, validation, and testing.")
    
    parser.add_argument("--modality_configs", type=json.loads, help="JSON dict for modalities, e.g. '{\"dem\": {\"channels\": [\"dem.tif\"]}}'")
    
    parser.add_argument("--stats_path", type=Path, default=r'../../data/earthscape_image_stats.csv', help="Path to mean & standard deviation .csv file for normalization parameters.")
    
    parser.add_argument("--train_path", type=Path, default=r'../../models/splits/indomain_train.geojson', help="Path to GeoJSON file of training patches.")
    
    parser.add_argument("--val_path", type=Path, default=r'../../models/splits/indomain_val.geojson', help="Path to GeoJSON file of validation patches.")
    
    parser.add_argument("--test_path", type=Path, default=r'../../models/splits/indomain_test.geojson', help="Path to GeoJSON file of test patches.")
    
    parser.add_argument("--cross_path", type=Path, default=r'../../models/splits/crossdomain_test.geojson', help="Path to GeoJSON file containing cross-domain test patches.")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training, validation, and testing")
    
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs.")
    
    parser.add_argument("--encoder", type=str, choices=["resnext", "vit"], default='resnext', help="Informal name of backbone encoder.")
    
    parser.add_argument("--enable_attention", action="store_true", help="Enable cross-attention modality fusion in SGMap-Net.")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for ADAM.")
    
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha parameter for focal loss")
    
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], default='mean', help="Reduction type for focal loss.")
    
    return parser.parse_args()





def main():

    # setup...
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modality_configs = get_norm_stats(args.stats_path, args.modality_configs)

    print("Mode: ", args.mode)
    print("Using device: ", device)
    print("Epochs: ", args.num_epochs)
    print("Batch size: ", args.batch_size)
    print("Modalities: ", list(modality_configs.keys()))


    # in-domain training...
    gdf = gpd.read_file(args.train_path)
    train_patch_ids = gdf['patch_id'].to_list()
    train_ds = EarthScape_Dataset(train_patch_ids, args.patch_dirs, modality_configs, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=(device.type=='cuda'))

    # in-domain validation...
    gdf = gpd.read_file(args.val_path)
    val_patch_ids = gdf['patch_id'].to_list()
    val_ds = EarthScape_Dataset(val_patch_ids, args.patch_dirs, modality_configs, augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=(device.type=='cuda'))

    # in-domain test set...
    gdf = gpd.read_file(args.test_path)
    test_patch_ids = gdf['patch_id'].to_list()
    test_ds = EarthScape_Dataset(test_patch_ids, args.patch_dirs, modality_configs, augment=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=(device.type=='cuda'))

    # cross-domain test set...
    gdf = gpd.read_file(args.cross_path)
    cross_patch_ids = gdf['patch_id'].to_list()
    cross_ds = EarthScape_Dataset(cross_patch_ids, args.patch_dirs, modality_configs, augment=False)
    cross_dl = DataLoader(cross_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=(device.type=='cuda'))

    # model setup...
    model = SGMapNet_Classification(modality_configs, args.encoder, enable_attention=args.enable_attention, embedding_dim=512, num_heads=8, p_attn=0.0, p_resid=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction=args.reduction).to(device)

    # create output directory...
    if args.mode == "train":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        modality_str = '_'.join(list(modality_configs.keys()))
        model_name = f"{args.encoder}_{modality_str}_{timestamp}"
        output_dir = f"../../models/smoke/{model_name}"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    else:
        output_dir = os.path.dirname(args.model_path)
        model_name = f"{Path(args.model_path).stem}_eval"
    
    # log training data
    training_log(model_name, output_dir, args.seed,
             args.train_path, args.val_path, args.test_path, args.cross_path,
             modality_configs, 
             args.batch_size, args.num_epochs, optimizer, criterion, model)


    # training...
    if args.mode == 'train':
        df_train = train_model(model, train_dl, val_dl, criterion, optimizer, device, args.num_epochs, output_dir)
        fig = plot_training_curves(df_train, output_dir)
        fig.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight', pad_inches=0)
        model_path = glob.glob(f"{output_dir}/*best_loss_epoch*.pth")[0]
        print("Training Complete!")

    if args.mode == "test":
        model_path = args.model_path
        # state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # model.load_state_dict(state_dict)

    # optimal F1 thresholds from validation set...
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    optimal_thresholds = calculate_optimal_thresholds(model, val_dl, device)

    # in-domain testing...
    probabilities, targets = test_model(model, test_dl, device)
    df_global = calculate_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_global.to_csv(f"{output_dir}/indomain_global_metrics.csv", index=False)

    df_class = calculate_class_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_class.to_csv(f"{output_dir}/indomain_class_metrics.csv", index=False)

    fig = plot_label_pr_roc_curves(targets, probabilities)
    fig.savefig(f"{output_dir}/indomain_pr_curves.jpg")

    # cross-domain testing...
    probabilities, targets = test_model(model, cross_dl, device)
    df_cross_global = calculate_global_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_cross_global.to_csv(f"{output_dir}/crossdomain_global_metrics.csv", index=False)

    df_cross_class = calculate_class_metrics(targets, probabilities, thresholds=optimal_thresholds)
    df_cross_class.to_csv(f"{output_dir}/crossdomain_class_metrics.csv", index=False)

    fig = plot_label_pr_roc_curves(targets, probabilities)
    fig.savefig(f"{output_dir}/crossdomain_pr_curves.jpg")

    print("Testing complete!")


if __name__ == "__main__":
    main()