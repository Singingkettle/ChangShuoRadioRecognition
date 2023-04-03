#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: inter_orthogonal_loss.py
Author: Citybuster
Time: 2021/5/31 12:29
Email: chagshuo@bupt.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


@LOSSES.register_module()
class ShrinkageLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 temperature=1,
                 loss_weight=1.0):
        """ShrinkageLoss.

        Args:
            Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            temperature (float, optional): . Defaults to 1
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(ShrinkageLoss, self).__init__()
        self.reduction = reduction
        self.temperature_reciprocal = 1.0 / temperature
        self.loss_weight = loss_weight
        if self.temperature_reciprocal >= 1:
            raise ValueError('The temperature of reciprocal should be < 1. '
                             'However it is set as {:f}'.format(self.temperature_reciprocal))

    def forward(self, x, label):
        """Forward function.

        Args:
            x (torch.Tensor): distance.
            label (torch.Tensor): 0 (from different modulation or self-inner_product which should be ignored)
                or 1 (from the same modulation)
        Returns:
            torch.Tensor: The calculated loss
        """
        x = torch.mul(x, self.temperature_reciprocal)
        loss = F.log_softmax(x, dim=1)
        loss = torch.mul(loss, label)
        loss = torch.mul(loss, -1)
        loss = torch.sum(loss, dim=1)
        loss = self.loss_weight * weight_reduce_loss(loss, reduction=self.reduction)
        return loss
