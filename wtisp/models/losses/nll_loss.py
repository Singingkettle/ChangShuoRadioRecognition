import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


def nll(pred,
        label,
        weight=None,
        reduction='mean',
        avg_factor=None,
        class_weight=None):
    """Calculate the negative log likelihood loss.

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
    loss = F.nll_loss(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def custom_nll(pred,
               label,
               weight=None,
               reduction='mean',
               avg_factor=None,
               class_weight=None):
    """Calculate the negative log likelihood loss.

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
    pred = torch.log(pred)
    loss = F.nll_loss(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class NLLLoss(nn.Module):

    def __init__(self,
                 multi_label=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """NLLLoss.

        Args:
            multi_label (bool, optional): Whether the input data has multilabel. 
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(NLLLoss, self).__init__()
        self.multi_label = multi_label
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.multi_label:
            # TODO: support multi_label prediction
            # self.cls_criterion = multilabel_cross_entropy
            pass
        else:
            self.cls_criterion = nll

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


@LOSSES.register_module()
class CustomNLLLoss(nn.Module):

    def __init__(self,
                 multi_label=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """NLLLoss.

        Args:
            multi_label (bool, optional): Whether the input data has multilabel. 
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CustomNLLLoss, self).__init__()
        self.multi_label = multi_label
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.multi_label:
            # TODO: support multi_label prediction
            # self.cls_criterion = multilabel_cross_entropy
            pass
        else:
            self.cls_criterion = custom_nll

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
