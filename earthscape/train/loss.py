
import torch
import torch.nn as nn
import torch.nn.functional as F




class BCEFocalLogits(nn.Module):
    """
    BCE-with-logits loss for multi-label classification with optional focal focusing,
    optional alpha balancing, and positive-class re-weighting.

    Parameters
    ----------
    gamma : float, default=0.0
        Focal parameter. gamma=0 disables focal scaling.
    alpha : float or torch.Tensor or None, default=None
        Classic focal alpha balancing for positives vs negatives. If provided, the
        per-element weight is alpha for target=1 and (1-alpha) for target=0.
        Scalar applies globally; tensor of shape (C,) applies per class. None disables.
    pos_weight : float or torch.Tensor or None, default=None
        Positive-class weight passed to `binary_cross_entropy_with_logits`. Scales only
        the positive term of BCE. Scalar applies globally; tensor of shape (C,) applies
        per class. None disables.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Reduction over the output tensor.

    Returns
    -------
    torch.Tensor
        Scalar loss if reduction in {'mean','sum'}, else tensor of shape (B, C).
    """

    def __init__(self, gamma=0.0, alpha=None, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):

        # cast target to same dtype of logits to ensure compatibility
        target = target.type_as(logits)


        # OPTIONAL: cast pos_weight to tensor -> scalar or (C,)...
        # NOTE: controls weights on positive samples in BCE
        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight
            if not isinstance(pw, torch.Tensor):
                pw = torch.tensor(pw, device=logits.device, dtype=logits.dtype)
            else:
                pw = pw.to(device=logits.device, dtype=logits.dtype)     


        # calculate BCE loss -> (B, C)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=pw)
        

        # calculate focal factor...
        # NOTE: focuses loss on low-confidence predictions 
        if self.gamma != 0.0:                 # focal term is on -> (B,C)
            p = torch.sigmoid(logits)
            pt = p * target + (1.0 - p) * (1.0 - target)
            focal_factor = (1.0 - pt).clamp_min(1e-8).pow(self.gamma)
        else:
            focal_factor = 1.0                # focal term is off -> scalar


        # OPTIONAL: calculate focal loss alpha positive/negative weighting -> scalar or (C,)...
        # NOTE: controls relative weights of positive vs. negatives
        if self.alpha is not None:            # alpha pos/neg scaling is on
            alpha = self.alpha
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha = alpha.to(device=logits.device, dtype=logits.dtype)
                
            alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        else:                                 # alpha pos/neg scaling is off
            alpha_t = 1.0


        # calculate loss (pos/neg weighting & focal term & BCE) -> (B,C)
        loss = alpha_t * focal_factor * bce_loss


        # aggregation of loss...
        if self.reduction == 'mean':
            return loss.mean()    # each class has equal weight
        
        elif self.reduction == 'sum':
            return loss.sum()     # sum of batch
        
        elif self.reduction == 'none':
            return loss           # per label loss for further use -> (B, C)
        