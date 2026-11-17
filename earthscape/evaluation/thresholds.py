
from earthscape.evaluation import test_model
import numpy as np
from sklearn.metrics import precision_recall_curve




def get_optimal_thresholds(model, loader, device, baseline, default_threshold=0.5):
    """
    Compute per-class decision thresholds that maximize F1 score. Thresholds 
    are selected independently for each class by evaluating the precision-recall 
    curve on the given dataset and choosing the threshold that yields the maximum 
    F1 score. Classes with no positive or no negative samples fall back to the 
    default threshold.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to generate prediction probabilities.
    loader : torch.utils.data.DataLoader
        DataLoader providing the dataset used for threshold optimization.
    device : torch.device
        Device on which the model is evaluated.
    default_threshold : float, optional
        Threshold used for classes with degenerate targets or undefined
        precision-recall curves. Default is 0.5.

    Returns
    -------
    optimal_thresholds : np.ndarray of shape (n_classes,)
        Array of per-class optimal thresholds that maximize F1 score.
    """

    # model inference with loader dataset
    probabilities, targets = test_model(model, loader, device, baseline=baseline)

    # make sure model outputs are on CPU and numpy arrays; labels also cast to int
    probabilities = probabilities.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy().astype(np.int32)

    # initialize variables...
    n_classes = probabilities.shape[1]                                               # number of classes
    optimal_thresholds = np.full(n_classes, default_threshold, dtype=np.float32)     # array of shape n_classes with default threshold
    eps = 1e-8                                                                       # value to prevent divide by zero error

    # iterate over probabilities...
    for class_idx in range(n_classes):

        # class targets & probabilities
        y_targs = targets[:, class_idx]
        p_model = probabilities[:, class_idx]

        # handle no positives or no negatives for a class; use default threshold
        if (y_targs.max() == 0) or (y_targs.min() == 1):
            continue
        
        # calculate precision, recall across thresholds
        precision, recall, thresholds = precision_recall_curve(y_targs, p_model)

        # handle empty thresholds
        if thresholds.size == 0:
            continue
        
        # calculate f1 score
        f1 = 2.0 * ((precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + eps))

        # find best threshold & add to optimal threshold array
        best_f1 = f1.max()
        best_idxs = np.where(np.isclose(f1, best_f1))[0]
        best_idx = best_idxs[-1]
        # best_idx = np.argmax(f1)
        optimal_thresholds[class_idx] = thresholds[best_idx]

    return optimal_thresholds