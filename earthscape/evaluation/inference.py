
import torch



def test_model(model, test_loader, device, baseline=True):
    """
    Run inference on a test set and return probabilities and targets.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used for inference.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding test batches as dicts with an optional ``'label'``
        tensor and one or more input tensors.
    device : torch.device
        Device used for model inference.
    baseline : bool, optional
        If True, a single input tensor is selected from each batch and passed
        to the model. If False, the full input dictionary is passed.

    Returns
    -------
    probabilities : torch.Tensor
        Concatenated sigmoid probabilities for all test samples (on CPU).
    targets : torch.Tensor or None
        Concatenated ground-truth labels for all test samples (on CPU), or None
        if labels are not provided in the test loader.
    """

    # set model for evaluation
    model.eval()

    # initialize variables for inference...
    probs = []     # model probabilities
    targs = []     # class labels

    # iterate over batches...
    with torch.inference_mode():
        for batch in test_loader:
            
            # get labels & images from batch...
            labels = None
            if "label" in batch:
                labels = batch['label'].to(device, non_blocking=True)
            
            # dict of modality tensors to pass to model (SGMap-Net)
            modalities = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'label'}

            # single tensor to pass to model (baseline tests)
            if baseline:
                modalities = next(iter(modalities.values())) 

            # model inference from input modalities
            logits = model(modalities)

            # model probabilities from inference output logits
            p = torch.sigmoid(logits)

            # append model probabilities & true class labels for batch...
            probs.append(p.cpu())

            if labels is not None:
                targs.append(labels.cpu())
    
    # get array of probabilities & targets...
    probabilities = torch.cat(probs, dim=0)

    if len(targs) > 0:
        targets = torch.cat(targs, dim=0)
    else:
        targets = None

    return probabilities, targets




def test_model_seg(model, test_loader, device, baseline=True):
    """
    Run inference on a segmentation test set and return predicted and target masks.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model used for inference.
    test_loader : torch.utils.data.DataLoader
        DataLoader yielding test batches with a ``'mask'`` tensor and one or more
        input tensors.
    device : torch.device
        Device used for model inference.
    baseline : bool, optional
        If True, a single input tensor is selected from each batch and passed
        to the model. If False, the full input dictionary is passed.

    Returns
    -------
    predictions : torch.Tensor
        Concatenated predicted class indices with shape [N, H, W] (on CPU).
    target_masks : torch.Tensor
        Concatenated ground-truth masks with shape [N, H, W] (on CPU).
    """
    model.eval()

    predictions = []
    target_masks = []

    with torch.no_grad():
        for batch in test_loader:
            masks = batch["mask"].to(device, non_blocking=True).long()
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "mask"}

            # single tensor to pass to model (baseline tests)
            if baseline:
                inputs = next(iter(inputs.values()))

            logits = model(inputs)                 # [B, C, H, W]
            preds = torch.argmax(logits, dim=1)    # [B, H, W]

            predictions.append(preds.cpu())
            target_masks.append(masks.cpu())

    predictions = torch.cat(predictions, dim=0)     # [N, H, W]
    target_masks = torch.cat(target_masks, dim=0)   # [N, H, W]

    return predictions, target_masks

