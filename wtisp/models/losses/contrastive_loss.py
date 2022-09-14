import torch
import torch.nn as nn

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


def contrastive_classic(x,
                        label,
                        contrastive=None,
                        weight=None,
                        reduction='mean',
                        avg_factor=None,
                        class_weight=None):
    """Calculate the Contrastive loss.

    Args:
        x (torch.Tensor): The embedding vectors with shape (N, M), M is the vector length.
        label (torch.Tensor): The sign label of x, where 2x=M, M must be even number.
        contrastive (dict): The config to build loss
        weight (torch.Tensor, optional): Sample-pair loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    label = label.view(-1)
    batch_size = x.shape[0]
    split_index = int(batch_size / 2)
    x0 = x[:split_index, :]
    x1 = x[split_index:, :]
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)

    mdist = contrastive['margin'] - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = label * dist_sq + (1 - label) * torch.pow(dist, 2)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def contrastive_exp(x,
                    label,
                    contrastive=None,
                    weight=None,
                    reduction='mean',
                    avg_factor=None,
                    class_weight=None):
    """Calculate the Contrastive loss.

    Args:
        x (torch.Tensor): The embedding vectors with shape (N, M), M is the vector length.
        label (torch.Tensor): The sign label of x, where 2x=M, M must be even number.
        contrastive (dict): The config to build loss
        weight (torch.Tensor, optional): Sample-pair loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    label = label.view(-1)
    batch_size = x.shape[0]
    split_index = int(batch_size / 2)
    x0 = x[:split_index, :]
    x1 = x[split_index:, :]
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)

    mdist = contrastive['margin'] - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = label * dist_sq + (1 - label) * torch.exp(dist)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def contrastive_quadratic(x,
                          label,
                          contrastive=None,
                          weight=None,
                          reduction='mean',
                          avg_factor=None,
                          class_weight=None):
    """Calculate the Contrastive loss.

    Args:
        x (torch.Tensor): The embedding vectors with shape (N, M), M is the vector length.
        label (torch.Tensor): The sign label of x, where 2x=M, M must be even number.
        contrastive (dict): The config to build loss
        weight (torch.Tensor, optional): Sample-pair loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    label = label.view(-1)
    batch_size = x.shape[0]
    split_index = int(batch_size / 2)
    x0 = x[:split_index, :]
    x1 = x[split_index:, :]
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), 1)

    mdist = contrastive['b'] - contrastive['a'] * dist_sq
    mdist = torch.clamp(mdist, min=0.0)
    loss = label * dist_sq + (1 - label) * mdist

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 contrastive=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.contrastive = contrastive
        if contrastive is None:
            raise ValueError('You must set the type of contrastive to calculate the contrastive loss!')
        elif contrastive['type'] == 'classic':
            self.cls_criterion = contrastive_classic
        elif contrastive['type'] == 'exp':
            self.cls_criterion = contrastive_exp
        elif contrastive['type'] == 'quadratic':
            self.cls_criterion = contrastive_quadratic
        else:
            raise ValueError(
                'Unknown contrastive type {} to calculate the contrastive loss!'.format(contrastive['type']))

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
        loss_cls = self.loss_weight * self.cls_criterion(
            x,
            label,
            self.contrastive,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
