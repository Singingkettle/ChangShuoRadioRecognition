#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: logistic_loss.py
Author: Citybuster
Time: 2021/6/1 20:57
Email: chagshuo@bupt.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


@LOSSES.register_module()
class LogisticLoss(nn.Module):

    def __init__(self,
                 reduction='sum',
                 temperature=1,
                 loss_weight=1.0):
        """LogisticLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            temperature (float, optional): . Defaults to 1
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(LogisticLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, inner_product, label, weight):
        """Forward function.

        Args:
            inner_product (torch.Tensor): distance.
            label (torch.Tensor): 0 (self-inner_product which should be ignored), -1, or 1
            weight (torch.Tensor): 0 (self-inner_product which should be ignored) or
                frac{1}{2N^+}, frac{1}{2N^-}
        Returns:
            torch.Tensor: The calculated loss
        """
        x = -1 * torch.multiply(inner_product, label) * self.temperature
        x = torch.log(1 + torch.exp(x))
        x = torch.multiply(x, weight)  # (N, C)
        x = torch.sum(x, dim=1)  # (N,)
        loss_logistic = self.loss_weight * weight_reduce_loss(x, reduction=self.reduction)
        return loss_logistic


@LOSSES.register_module()
class InfoNCELoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 temperature=1,
                 loss_weight=1.0):
        """LogisticLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            temperature (float, optional): . Defaults to 1
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(InfoNCELoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, inner_product, label, weight):
        """Forward function.

        Args:
            inner_product (torch.Tensor): distance.
            label (torch.Tensor): 0 (self-inner_product which should be ignored), -1, or 1
            weight (torch.Tensor): 0 (self-inner_product which should be ignored) or
                frac{1}{2N^+}, frac{1}{2N^-}
        Returns:
            torch.Tensor: The calculated loss
        """
        x = torch.multiply(inner_product * self.temperature, label)
        x = -1 * F.log_softmax(x, dim=1)
        x = torch.multiply(x, weight)  # (N, C)
        x = torch.sum(x, dim=1)  # (N,)
        loss = self.loss_weight * weight_reduce_loss(x, reduction=self.reduction)
        return loss
