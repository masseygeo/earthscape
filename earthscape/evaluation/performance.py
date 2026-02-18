
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt




def get_global_metrics(targets, probabilities, thresholds):
    """
    Compute global multi-label classification metrics. Probabilities are 
    thresholded (using a scalar or per-class thresholds) to obtain binary 
    predictions.

    Parameters
    ----------
    targets : torch.Tensor
        Ground-truth binary labels.
    probabilities : torch.Tensor
        Predicted probabilities or scores.
    thresholds : float or array-like
        Decision threshold(s) applied to probabilities.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing computed metrics: Macro-averaged 
        and weighted precision, recall, F1, ROC AUC, and mean average 
        precision (mAP); micro-averaged accuracy (Hamming accuracy) and 
        balanced accuracy. 
    """

    # move targs, probs, thresh to CPU, move to arrays; cast targets to int...
    targs = targets.detach().cpu().numpy().astype(np.int32)
    probs = probabilities.detach().cpu().numpy()
    thresholds = np.asarray(thresholds)

    # calculate binary predictions from either single scalar or per-class array...
    if thresholds.ndim == 0:
        binary_preds = (probs >= thresholds).astype(np.int32)             # if scalar
    else:
        binary_preds = (probs >= thresholds[None, :]).astype(np.int32)    # if array
    
    # initialize dict to hold metrics
    df = {}

    # calculate global performance metrics (macro- and weighted-)...
    df['Precision (Macro)'] = precision_score(targs, binary_preds, average='macro', zero_division=0.0)
    df['Recall (Macro)'] = recall_score(targs, binary_preds, average='macro', zero_division=0.0)
    df['F1 (Macro)'] = f1_score(targs, binary_preds, average='macro', zero_division=0.0)
    # df['AUC (Macro)'] = roc_auc_score(targs, probs, average="macro")
    df['mAP (Macro)'] = average_precision_score(targs, probs, average='macro')
    
    df['Precision (Wt.)'] = precision_score(targs, binary_preds, average='weighted', zero_division=0.0)
    df['Recall (Wt.)'] = recall_score(targs, binary_preds, average='weighted', zero_division=0.0)
    df['F1 (Wt.)'] = f1_score(targs, binary_preds, average='weighted', zero_division=0.0)
    # df['AUC (Wt.)'] = roc_auc_score(targs, probs, average='weighted')
    df['mAP (Wt.)'] = average_precision_score(targs, probs, average='weighted')
    df['Accuracy (Micro)'] = (binary_preds == targs).mean()

    # calculate AUC over classes with at least one positive & one negative...
    # only keep classes (columns) with > 1 value
    valid_cols = [k for k in range(targets.shape[1]) if len(np.unique(targets[:, k])) == 2]

    if len(valid_cols) > 0:
        m = roc_auc_score(targs[:, valid_cols], probs[:, valid_cols], average="macro")
        w = roc_auc_score(targs[:, valid_cols], probs[:, valid_cols], average='weighted')
    else:
        m = np.nan
        w = np.nan

    df = pd.DataFrame(df, index=[0])
    df.insert(loc=3, column='AUC (Macro)', value=m)
    df.insert(loc=8, column='AUC (Wt.)', value=w)

    return df







def get_class_metrics(targets, probabilities, thresholds, classes):
    """
    Compute per-class classification metrics. Probabilities are 
    thresholded (using a scalar or per-class thresholds) to obtain 
    binary predictions.

    Parameters
    ----------
    targets : torch.Tensor with shape ``(n_samples, n_classes)``
        Ground-truth binary labels.
    probabilities : torch.Tensor with shape ``(n_samples, n_classes)``
        Predicted probabilities.
    thresholds : float or array-like with shape ``(n_classes,)``
        Decision threshold(s) applied to probabilities.
    classes : sequence of str
        Informal class names corresponding to each column.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing per-class metrics for each class: accuracy, 
        precision, recall (sensitivity, TPR), specificity (TNR), F1, 
        ROC AUC, and average precision (AP). In addtion, class name, 
        threshold used, total number of positives (target), and total 
        number of predicted positives (model)
    """

    # initialize dataframe
    df = pd.DataFrame()

    # move targs, probs, thresh to CPU, move to arrays; cast targets to int...
    targets = targets.detach().cpu().numpy().astype(np.int32)
    probabilities = probabilities.detach().cpu().numpy()
    thresholds = np.asarray(thresholds)

    # calculate binary predictions from either single scalar or per-class array...
    if thresholds.ndim == 0:
        thresholds = np.full(probabilities.shape[1], float(thresholds))
        binary_preds = (probabilities >= thresholds).astype(np.int32)             # if scalar
    else:
        binary_preds = (probabilities >= thresholds[None, :]).astype(np.int32)    # if array

    # iterate through each class and calculate performance metrics...
    for idx, (unit, thresh) in enumerate(zip(classes, thresholds)):

        # get targets, probabilities, and binary predictitons for class
        targs = targets[:, idx]
        probs = probabilities[:, idx]
        preds = binary_preds[:, idx]

        # calculate metrics
        df.loc[idx, 'Class'] = unit
        df.loc[idx, 'Threshold'] = thresh
        df.loc[idx, 'n True'] = targs.sum()
        df.loc[idx, 'n Predicted'] = preds.sum()
        df.loc[idx, 'Precision'] = precision_score(targs, preds, zero_division=0.0)
        df.loc[idx, 'Recall'] = recall_score(targs, preds, zero_division=0.0)
        df.loc[idx, 'Specificity'] = recall_score(1-targs, 1-preds, zero_division=0.0)
        df.loc[idx, 'F1'] = f1_score(targs, preds, zero_division=0.0)
        df.loc[idx, 'AUC'] = roc_auc_score(targs, probs)
        df.loc[idx, 'AP'] = average_precision_score(targs, probs)
        df.loc[idx, 'Accuracy'] = accuracy_score(targs, preds)

    return df




def plot_pr_roc_curves(targets, predictions, class_cols):
    """
    Plot per-class precision-recall and receiver operating curves (ROC).

    Generates a two-panel figure showing precision-recall curves (left panel) 
    and ROC curves (right panel) for model experiment.

    Parameters
    ----------
    targets : ndarray of shape (n_samples, n_classes)
        Ground-truth binary labels.
    predictions : ndarray of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class.
    class_cols : sequence of str, len n_classes
        Class names corresponding to each column.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing PR and ROC subplots with one curve per class.
    """

    # initialize figure and axes objects for two subplots 
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))

    # initialize list for skipped classes
    skipped = []

    # iterate through classes...
    for idx, unit in enumerate(class_cols):

        # get ground truth & predicted labels...
        Y_true = targets[:, idx]
        y_pred = predictions[:, idx]

        # if ground truth has no variance (same labels) then skip plots...
        if Y_true.max() == Y_true.min():
            skipped.append(unit)
            continue
        
        # get precision & recall for all thresholds for each class...
        p, r, _ = precision_recall_curve(Y_true, y_pred)

        # get FPR & TPR (recall/sensitivity) for all thresholds for each class...
        fpr, tpr, _ = roc_curve(Y_true, y_pred)

        # plot P-R curve & ROC for each class...
        ax[0].plot(r, p, linewidth=0.75, label=class_cols[idx])
        ax[1].plot(fpr, tpr, linewidth=0.75, label=class_cols[idx])
    
    # customize plots...
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title('Precision-Recall Curve', style='italic')

    ax[1].plot([0,1], [0,1], color='k', linestyle='--', lw=1)
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Curve', style='italic')
    
    for axes in ax:
        axes.set_xlim(0,1)
        axes.set_ylim(0,1)
    
    # add legend
    ax[0].legend(loc='upper center', bbox_to_anchor=(1.15, -0.15), ncols=len(class_cols), frameon=False, fontsize=8)

    # add note for any skipped classes...
    if skipped:
        fig.subplots_adjust(bottom=0.28)
        note = "*Not shown - no ground-truth label variation: " + ", ".join(skipped)
        fig.text(0.5, 0.03, note, ha='center', va='bottom', fontsize=8, fontstyle='italic')

    return fig