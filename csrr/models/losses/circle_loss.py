import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


def circle_loss(x,
                label,
                m,
                gama,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
    similarity_matrix = x @ x.T  # need gard here
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    negative_matrix = label_matrix.logical_not()
    positive_matrix = label_matrix.fill_diagonal_(False)

    sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
    sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))

    ap = torch.clamp_min(1 + m - sp.detach(), min=0.)
    an = torch.clamp_min(sn.detach() + m, min=0.)

    dp = 1 - m
    dn = m
    logit_p = -gama * ap * (sp - dp)
    logit_n = gama * an * (sn - dn)

    logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
    logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1))

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CircleLoss(nn.Module):

    def __init__(self,
                 m=0.25,
                 gama=10,
                 reduction='mean',
                 loss_weight=1.0):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gama = gama
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.circle_criterion = circle_loss

    def forward(self,
                x,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_circle = self.loss_weight * self.circle_criterion(
            x,
            label,
            self.m,
            self.gama,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_circle
