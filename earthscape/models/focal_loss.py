
import torch
import torch.nn as nn
import torch.nn.functional as F







class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha             # class balance weighting factor
        self.gamma = gamma             # focal param (how much to down-weight easy classes)
        self.reduction = reduction     # loss aggregation method

    def forward(self, logits_input, binary_target):
        
        # calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits_input, binary_target, reduction='none')
        
        # calculate predicted probability of correct class
        pt = torch.exp(-bce_loss)

        # calculate focal loss
        focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()    # aggregate by average over batch
        
        elif self.reduction == 'sum':
            return focal_loss.sum()     # aggregate by sum over batch
        
        else:
            return focal_loss           # per sample focal loss tensor; output - [B, 7]