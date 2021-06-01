#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: auxiliary_head.py
Author: Citybuster
Time: 2021/5/31 19:15
Email: chagshuo@bupt.edu.cn
"""

import torch

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class InterOrthogonalHead(BaseHead):
    def __init__(self, num_bases, batch_size, loss_weight=1):
        super(InterOrthogonalHead, self).__init__()
        self.num_bases = num_bases
        self.batch_size = batch_size
        self.loss_inter_orthogonal = build_loss(dict(type='InterOrthogonalLoss', loss_weight=loss_weight))

    def init_weights(self):
        pass

    def loss(self, x, **kwargs):
        label = x.new_ones((self.batch_size, self.num_bases, self.num_bases))
        label[torch.arange(self.batch_size), torch.arange(self.num_bases), torch.arange(self.num_bases)] = 0
        label = label.view(self.batch_size, -1)
        loss_inter_orthogonal = self.loss_inter_orthogonal(x, label)
        return dict(loss_inter_orthogonal=loss_inter_orthogonal)

    def forward(self, x):
        x_ = torch.transpose(x, 1, 2)
        inter_inner_products = torch.bmm(x, x_)
        x = inter_inner_products.view(self.batch_size, -1)

        return x
