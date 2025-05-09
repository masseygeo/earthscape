import os
import json
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import average_precision_score

import random
import numpy as np
from torchvision import transforms


# Local imports
from utils_model_dataloader import MultiModalDataset
from utils_model_training import (
    train_model,
    FocalLoss,
    calculate_optimal_thresholds,
    test_model,
    calculate_label_precision_recall_f1_aucroc,
    plot_label_pr_roc_curves,
    calculate_global_metrics
)
from model_sgmap_munia import ResNextEncoder, VAE_encoder, MultilabelClassification, ViT_encoder
# SatMaEViTEncoder
# from model_cross_attn import CrossModalFusionPerMod, MultiLayerFusion




def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or test a multimodal classification model.")
    parser.add_argument("--model_name", type=str, default="ep_5x5_res",
                        help="Unique name for the model (used for directory and files)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training/testing")
    parser.add_argument("--num_epochs", type=int, default=15,
                        help="Number of training epochs")

    parser.add_argument("--seed", type=int, default=42,
                        help="Seed value")
    
    parser.add_argument("--weights_config", type=str, default="IMAGENET1K_V2",
                        help="Pretrained encoder weights (e.g., IMAGENET1K_V2 or None)")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3,                         # larger than 0.0001
                        help="Learning rate for optimizer")
    
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Alpha parameter for focal loss")
    
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "test", "all"],
                        help="Run mode: train, test only, or all (train+test)")
    
    parser.add_argument("--patch_dir", type=str, default="../data/patches_warren",
                        help="Directory where patch TIFFs are stored")
    parser.add_argument("--hardin_patch_dir", type=str, default="../data/patches_hardin",
                        help="Directory for cross-domain test patches")
    
    parser.add_argument("--norm_stats_path", type=str,
                        default="../data/warren/image_stats.csv",
                        help="Path to normalization stats CSV file")
    
    parser.add_argument("--train_patch_path", type=str,
                        default="../models/patches/warren_patches_train.geojson",
                        help="GeoJSON path for training patches")
    parser.add_argument("--val_patch_path", type=str,
                        default="../models/patches/warren_patches_val.geojson",
                        help="GeoJSON path for validation patches")
    parser.add_argument("--test_patch_path", type=str,
                        default="../models/patches/warren_patches_test.geojson",
                        help="GeoJSON path for in-domain test patches")
    parser.add_argument("--hardin_test_patch_path", type=str,
                        default="../models/patches/hardin_patches_test.geojson",
                        help="GeoJSON path for cross-domain test patches")
    return parser.parse_args()



def set_seed(seed: int):
    # 1) Python built-in RNG
    random.seed(seed)
    # 2) NumPy RNG
    np.random.seed(seed)
    # 3) PyTorch CPU RNG
    torch.manual_seed(seed)
    # 4) PyTorch GPU RNG (all devices)
    torch.cuda.manual_seed_all(seed)

    # 5) Make CuDNN deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Directories
    model_dir = os.path.join("../models/classification", args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    set_seed(args.seed)
    enc = args.model_name.split('_')[-1]
    print('encoder ', enc)
    modality  = args.model_name.split("_" + enc)[0]
    print('modality: ', modality)


    # Config
    if modality == 'rgb':
        modalities = {'aerial_rgb': ['aerialr.tif', 'aerialg.tif', 'aerialb.tif']}
    else:
        modalities = {modality : [f'{modality}.tif']}

    attention_configs = None


    # Build model
    # encoder = ResNextEncoder(args.weights_config)

    if enc == 'res':
        encoder = ResNextEncoder(args.weights_config)
    elif enc == 'vae':
        encoder = VAE_encoder().to(device)
    elif enc == 'vit':
        encoder = ViT_encoder().to(device)
    # elif enc == 'satmae':
    #     encoder = SatMaEViTEncoder().to(device)

    # model = MultiLayerFusion(modality_configs=modalities, encoder=encoder,  device=device).to(device)
    # model = CrossModalFusionPerMod(modality_configs=modalities, encoder=encoder, device=device).to(device)
    model = MultilabelClassification(modality_configs=modalities, encoder=encoder, attention_configs=attention_configs).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    with open(os.path.join(model_dir, 'model_architecture.txt'), 'w') as f:
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(str(model))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = torch.nn.DataParallel(model)
    
    encoder = encoder.to(device)
    model = model.to(device)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    alpha = 0.25
    # positive_counts = np.array([ 8690, 14793,   228,  1044, 11597,  6474, 21917])         # get class counts
    # alpha = 1 / positive_counts                                                           # calculate inverse class frequency
    # alpha = np.sqrt(alpha)                                                                # calculate square root ICF
    # alpha = alpha / alpha.mean()                                                          # normalize by mean
    alpha = torch.tensor(alpha, dtype=torch.float32).view(1, -1).to(device)               # convert to tensor

    criterion = FocalLoss(alpha=alpha, gamma=args.gamma, reduction='mean').to(device)   # initialize focal loss


    # Load normalization stats
    df_stats = pd.read_csv(args.norm_stats_path)
    norm_params = {
        m: [[df_stats.loc[df_stats['path']==ch, 'mean'].item() for ch in channels],
            [df_stats.loc[df_stats['path']==ch, 'std'].item() for ch in channels]]
        for m, channels in modalities.items()
    }

    # Load patch IDs
    gdf_train = gpd.read_file(args.train_patch_path)
    gdf_val = gpd.read_file(args.val_patch_path)
    gdf_test = gpd.read_file(args.test_patch_path)
    gdf_hardin = gpd.read_file(args.hardin_test_patch_path)


    # Datasets & loaders
    custom_tf = None
    if enc in ['vit', 'satmae']:
        custom_tf = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            transforms.ToTensor(),
        ])

    def make_loader(ids, data_dir, augment):
        ds = MultiModalDataset(
            ids=ids.tolist(), data_dir=data_dir,
            modalities=modalities, norm_params=norm_params,
            augment=augment, task='classification', transform=custom_tf)
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(gdf_train['patch_id'], args.patch_dir, True)
    val_loader   = make_loader(gdf_val['patch_id'], args.patch_dir, False)
    test_loader  = make_loader(gdf_test['patch_id'], args.patch_dir, False)
    hardin_loader= make_loader(gdf_hardin['patch_id'], args.hardin_patch_dir, False)


    # Metadata
    metadata = {
        'NAME': args.model_name,
        'DIRECTORY': model_dir,
        'MODALITIES': modalities,
        'HYPERPARAMETERS': vars(args),
        'MODEL': {
            'encoder': type(encoder).__name__,
            'weights': args.weights_config,
            'attention': attention_configs,
            'model': type(model).__name__
        }
    }
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


    # TRAIN
    if args.mode in ['train', 'all']:
        train_loss, train_acc, val_loss, val_acc, best_epoch = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, device, args.num_epochs, model_dir)
        metadata['TRAINING'] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)


        fig, ax = plt.subplots(ncols=2, figsize=(10,6))

        epochs = range(1, len(train_loss)+1)

        ax[0].plot(epochs, train_loss, label='Train')
        ax[0].plot(epochs, val_loss, label='Validation')
        ax[0].set_title('Focal Loss', style='italic')

        ax[1].plot(epochs, train_acc, label='Train')
        ax[1].plot(epochs, val_acc, label='Validation')
        ax[1].set_title('Overall Accuracy', style='italic')

        for axes in ax:
            axes.axvline(x=best_epoch, linestyle='--', color='k', label='Best model')
            axes.legend(frameon=False)
            axes.set_xticks(epochs)
            axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
            axes.set_xlabel('Epochs')

        modalities_str = list(modalities.keys())[0]
        if len(modalities.keys()) > 1:
            for modality in list(modalities.keys())[1:]:
                modalities_str = modalities_str + ' + ' + str(modality)

        plt.suptitle(f"Multilabel Classification\n{args.model_name} - {modalities_str}", y=0.99)
        plt.savefig(f"{model_dir}/training_results.jpg")



    # TEST
    if args.mode in ['test', 'all']:
        # load best weights
        state = torch.load(os.path.join(model_dir, 'best_loss.pth'),
                           map_location=device, weights_only=False)
        model.load_state_dict(state)

        # optimal thresholds
        thresholds = calculate_optimal_thresholds(model, val_loader, device)


        # evaluations
        def evaluate(loader, prefix):
            preds, targs = test_model(model, loader, device)

            # per-label
            df_label = pd.DataFrame(columns=[
                'Class','Targets','Predictions','Accuracy','Precision',
                'Recall','F1','AP','AUROC'
            ])
            for i, unit in enumerate(['af1','Qal','Qaf','Qat','Qc','Qca','Qr']):
                thresh = thresholds[i]
                p = preds[:,i]; t = targs[:,i]
                acc, prec, rec, f1, roc = \
                    calculate_label_precision_recall_f1_aucroc(p,t,threshold=thresh)
                ap = average_precision_score(t, p)
                df_label.loc[i] = [
                    f"{unit} ({thresh:.2f})", int(t.sum()), int((p>=thresh).sum()),
                    acc, prec, rec, f1, ap, roc
                ]
            df_label.to_csv(os.path.join(model_dir, f'label_metrics_{prefix}.csv'), index=False)


            # PR/ROC curves
            fig = plot_label_pr_roc_curves(targs, preds, ['af1','Qal','Qaf','Qat','Qc','Qca','Qr'])
            fig.savefig(os.path.join(model_dir, f'pr_roc_curves_{prefix}.png'))


            # global
            macro_precision, weighted_precision, macro_recall, weighted_recall, macro_f1, weighted_f1, macro_mAP, weighted_mAP, h_loss, subset_acc, overall_acc = calculate_global_metrics(targs, preds, thresholds)
            # cols = ['AUC','Overall Accuracy','Macro Precision','Macro Recall',
            #         'Macro F1','Weighted Precision','Weighted Recall','Weighted F1',
            #         'Macro mAP','Weighted mAP','Hamming Loss','Subset Accuracy']
            df_glob = pd.DataFrame({
                # 'AUC':auc,
                'Overall Accuracy': overall_acc, 
                'Macro Precision': macro_precision, 
                'Macro Recall': macro_recall, 
                'Macro F1': macro_f1, 
                'Weighted Precision': weighted_precision, 
                'Weighted Recall': weighted_recall,
                'Weighted F1': weighted_f1, 
                'Macro mAP': macro_mAP, 
                'Weighted mAP': weighted_mAP, 
                'Hamming Loss': h_loss, 
                'Subset Accuracy':subset_acc}, index=[0])
            
            # df_glob = pd.DataFrame([stats], columns=cols)
            df_glob.to_csv(os.path.join(model_dir, f'global_metrics_{prefix}.csv'), index=False)

        evaluate(test_loader, 'warren')
        evaluate(hardin_loader, 'hardin')
        print(f"Testing complete. Metrics saved to {model_dir}")

if __name__ == "__main__":
    main()
