import torch
import torch.nn as nn
import torch.nn.functional as F

from csrr.registry import MODELS
from .utils import weight_reduce_loss


@MODELS.register_module()
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor = None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = F.mse_loss(pred, target, reduction='none')
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
