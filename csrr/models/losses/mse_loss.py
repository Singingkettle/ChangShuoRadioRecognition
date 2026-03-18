import torch
import torch.nn as nn

from csrr.registry import MODELS


@MODELS.register_module()
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.mse_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss
