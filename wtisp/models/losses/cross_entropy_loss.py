import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# x = torch.rand(3, 5)
# y = torch.tensor([1, 0, 4])

# #
# m = nn.LogSoftmax(dim=1)
# loss = nn.NLLLoss()

# p = m(x)
# l = loss(p, y)
# print(l)

# #

# m = nn.Softmax(dim=1)
# p = m(x)
# p = torch.log(p)
# l = loss(p, y)
# print(l)

# #
# l = F.cross_entropy(x, y)
# print(l)


def cross_entropy(pred,
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
    label = label.view(-1)
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def custom_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the custom CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction with shape(N, C)
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    pred = F.log_softmax(pred, dim=1)
    loss = torch.mul(pred, label)
    loss = torch.mul(loss, -1)
    loss = torch.sum(loss, dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 multi_label=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            multi_label (bool, optional): Whether the input data has multilabel. 
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.multi_label = multi_label
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.multi_label:
            # TODO: support multi_label prediction
            # self.cls_criterion = multilabel_cross_entropy
            pass
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
