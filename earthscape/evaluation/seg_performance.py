
# import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp 
from monai.metrics import hausdorff_distance
import matplotlib.pyplot as plt
import seaborn as sns



def image_class_metrics_seg(preds, masks, patch_ids, class_cols):

    num_classes = len(class_cols)

    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode="multiclass", num_classes=num_classes)       # [N, C]

    y_true = F.one_hot(masks.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    y_pred = F.one_hot(preds.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    hd = hausdorff_distance(y_pred=y_pred, y=y_true, include_background=True,percentile=None)      # [N, C]
    hd95 = hausdorff_distance(y_pred=y_pred, y=y_true, include_background=True, percentile=95.0,)  # [N, C]
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")                                          # [N, C]
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")                                          # [N, C]
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="none")                                    # [N, C]
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="none")                                          # [N, C]
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="none")                                      # [N, C]
    support = tp + fn
    predicted_support = tp + fp
    gt_present = (support > 0).long()
    pred_present = (predicted_support > 0).long()

    rows = []
    for i, patch_id in enumerate(patch_ids):
        for c, class_name in enumerate(class_cols):
            rows.append({
                "patch_id": patch_id,
                "class": class_name,
                "tp": tp[i, c].item(),
                "fp": fp[i, c].item(),
                "fn": fn[i, c].item(),
                "tn": tn[i, c].item(),
                "iou": iou[i, c].item(),
                "dice": dice[i, c].item(),
                "precision": precision[i, c].item(),
                "recall": recall[i, c].item(),
                "accuracy": accuracy[i, c].item(),
                "hd": hd[i, c].item(),
                "hd95": hd95[i, c].item(),
                "support": support[i, c].item(),
                "predicted_support": predicted_support[i, c].item(),
                "targets_present": gt_present[i, c].item(),
                "preds_present": pred_present[i, c].item(),
                })

    return pd.DataFrame(rows)




def image_overall_metrics_seg(df_image):
    df = (
        df_image.groupby("patch_id", as_index=False)
        .agg(
            mean_iou=("iou", "mean"),
            mean_dice=("dice", "mean"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_accuracy=("accuracy", "mean"),
            mean_hd=("hd", "mean"),
            mean_hd95=("hd95", "mean"),
            total_support=("support", "sum"),
            total_predicted_support=("predicted_support", "sum"),
            target_classes_present=("targets_present", "sum"),
            pred_classes_present=("preds_present", "sum"),
        ).copy())

    return df



def overall_metrics_seg(df_class):
    df = pd.DataFrame({
        "metric": ["miou", "mean_dice", "macro_precision", "macro_recall", "macro_acc", "hd", "hd95", "pixel_accuracy"], 
        "value": [df_class["iou"].mean(), df_class["dice"].mean(), df_class["precision"].mean(), df_class["recall"].mean(), df_class["accuracy"].mean(), df_class["hd"].mean(), df_class["hd95"].mean(), df_class["tp"].sum() / df_class["support"].sum()]
        })

    return df



def overall_class_metrics_seg(df_image):
    df_class = (
        df_image.groupby("class", as_index=False)
        .agg(
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            hd=("hd", "mean"),
            hd95=("hd95", "mean"),
            support=("support", "sum"),
            predicted_support=("predicted_support", "sum"),
            image_support=("targets_present", "sum"),
            predicted_image_support=("preds_present", "sum"),
        ).copy())

    df_class["iou"] = df_class["tp"] / (df_class["tp"] + df_class["fp"] + df_class["fn"])
    df_class["dice"] = 2 * df_class["tp"] / (2 * df_class["tp"] + df_class["fp"] + df_class["fn"])
    df_class["precision"] = df_class["tp"] / (df_class["tp"] + df_class["fp"])
    df_class["recall"] = df_class["tp"] / (df_class["tp"] + df_class["fn"])
    df_class["accuracy"] = (df_class["tp"] + df_class["tn"]) / (df_class["tp"] + df_class["fp"] + df_class["fn"] + df_class["tn"])

    return df_class[["class","iou","dice","precision","recall","accuracy","hd","hd95","support","predicted_support","image_support","predicted_image_support","tp","fp","fn","tn"]]




def plot_cm_seg(preds, masks, class_cols, mode="raw"):
    num_classes = len(class_cols)

    x = masks.reshape(-1) * num_classes + preds.reshape(-1)
    cm = torch.bincount(x, minlength=num_classes**2).reshape(num_classes, num_classes).cpu().numpy()

    if mode == "row_norm":
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix, Row-normalized"
        vmin, vmax = 0, 1
    else:
        fmt = "d"
        title = "Confusion Matrix, Raw pixels"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(cm, annot=True, annot_kws={'fontsize':10}, fmt=fmt, cmap='viridis', cbar_kws={"pad": 0.01}, xticklabels=class_cols, yticklabels=class_cols, vmin=vmin, vmax=vmax, square=True, linewidths=0.5, linecolor='k', ax=ax)

    ax.tick_params(labelsize=10)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    return fig




def calculate_dice_score(logits, masks):

    num_classes = logits.shape[1]

    # predicted class labels -> [B, H, W]
    preds = torch.argmax(logits, dim=1)

    # convert preds and masks to one-hot -> [B, C, H, W]
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    masks_one_hot = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # sum over batch and spatial dims
    intersection = (preds_one_hot * masks_one_hot).sum(dim=(0, 2, 3))                    # shape -> [C]
    denominator = preds_one_hot.sum(dim=(0, 2, 3)) + masks_one_hot.sum(dim=(0, 2, 3))    # shape -> [C]
    dice_per_class = (2 * intersection + 1e-7) / (denominator + 1e-7)                    # shape -> [C]
    dice = dice_per_class.mean()

    return dice.item()




# from scipy.ndimage import binary_erosion, distance_transform_edt

# def calculate_hausdorff_distance(preds, targets, num_classes, percentile=95):

#     preds = preds.cpu().numpy()
#     targets = targets.cpu().numpy()

#     hd_per_class = {}

#     for c in range(num_classes):
#         class_distances = []

#         for pred, target in zip(preds, targets):
#             pred_c = (pred == c)
#             target_c = (target == c)

#             pred_boundary = pred_c ^ binary_erosion(pred_c)
#             target_boundary = target_c ^ binary_erosion(target_c)

#             # distance to nearest boundary point in opposite mask
#             dt_target = distance_transform_edt(~target_boundary)
#             dt_pred = distance_transform_edt(~pred_boundary)

#             pred_to_target = dt_target[pred_boundary]
#             target_to_pred = dt_pred[target_boundary]

#             surface_distances = np.concatenate([pred_to_target, target_to_pred])

#             if surface_distances.size > 0:
#                 hd = np.percentile(surface_distances, percentile)
#                 class_distances.append(hd)

#         valid = [d for d in class_distances if not np.isnan(d)]

#         if len(valid) > 0:
#             hd_per_class[c] = float(np.mean(valid))
#         else:
#             hd_per_class[c] = np.nan

#     valid_values = [v for v in hd_per_class.values() if not np.isnan(v)]
#     mean_hd = float(np.mean(valid_values)) if len(valid_values) > 0 else np.nan

#     return mean_hd, hd_per_class