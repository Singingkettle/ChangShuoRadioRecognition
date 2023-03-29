import torch
import torch.nn.functional as F

from .cross_entropy_loss import CrossEntropyLoss
from .utils import weight_reduce_loss
from ..builder import LOSSES


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
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
    loss = F.binary_cross_entropy_with_logits(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class BinaryCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """BinaryCrossEntropyLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = binary_cross_entropy
