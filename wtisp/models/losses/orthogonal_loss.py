#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: inter_orthogonal_loss.py
Author: Citybuster
Time: 2021/5/31 12:29
Email: chagshuo@bupt.edu.cn
"""

import torch
import torch.nn as nn

from .utils import weight_reduce_loss
from ..builder import LOSSES


@LOSSES.register_module()
class InterOrthogonalLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 temperature=1,
                 loss_weight=1.0):
        """InterOrthogonalLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            temperature (float, optional): . Defaults to 1
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(InterOrthogonalLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, inner_product, label):
        """Forward function.

        Args:
            inner_product (torch.Tensor): distance.
            label (torch.Tensor): 0 (self-inner_product which should be ignored) or 1 (from the same domain)
        Returns:
            torch.Tensor: The calculated loss
        """
        x = torch.log(1 + torch.exp(inner_product / self.temperature))
        x = torch.multiply(x, label)  # (N, (p*p))
        x = torch.sum(x, dim=1)  # (N,)
        loss_inter_orthogonal = self.loss_weight * weight_reduce_loss(x, reduction=self.reduction)
        return loss_inter_orthogonal


@LOSSES.register_module()
class IntraOrthogonalLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 temperature=1,
                 loss_weight=1.0):
        """IntraOrthogonalLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            temperature (float, optional): . Defaults to 1
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(IntraOrthogonalLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.loss_weight = loss_weight
        if self.temperature <= 0:
            raise ValueError('The temperature of IntraOrthogonalLoss should be > 0. '
                             'However it is set as {:f}'.format(self.temperature))

    def forward(self, inner_product, label):
        """Forward function.

        Args:
            inner_product (torch.Tensor): distance.
            label (torch.Tensor): 0 (from different modulation or self-inner_product which should be ignored)
                or 1 (from the same modulation)
        Returns:
            torch.Tensor: The calculated loss
        """
        x_left = torch.log((torch.sum(torch.exp(inner_product / self.temperature), dim=1) - 1))
        x_right = torch.log(torch.sum(torch.multiply(torch.exp(inner_product / self.temperature), label), dim=1))
        x = x_left - x_right
        loss_intra_orthogonal = self.loss_weight * weight_reduce_loss(x, reduction=self.reduction)
        return loss_intra_orthogonal
