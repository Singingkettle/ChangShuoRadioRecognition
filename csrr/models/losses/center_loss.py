import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


def center_cluster(dis,
                   label,
                   weight=None,
                   reduction='mean',
                   avg_factor=None,
                   class_weight=None,
                   **kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        dis (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    b, c = dis.shape
    mask = F.one_hot(label, num_classes=c)
    loss = dis * mask.float()
    loss = loss.clamp(min=1e-12, max=1e+12)
    loss = torch.sum(loss, dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CenterLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CenterLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cluster_criterion = center_cluster

    def forward(self,
                x,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            x (torch.Tensor): The embedding vectors.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_center = self.loss_weight * self.cluster_criterion(
            x,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_center
