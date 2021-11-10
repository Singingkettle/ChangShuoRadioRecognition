#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: distillation_head.py
Author: Citybuster
Time: 2021/10/28 21:55
Email: chagshuo@bupt.edu.cn
"""

import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class DISHead(BaseHead):
    def __init__(self, num_classes, in_features=80, out_features=256, loss_cn=None, loss_kl=None):
        super(DISHead, self).__init__()
        if loss_cn is None:
            loss_cn = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if loss_kl is None:
            loss_kl = dict(
                type='KLDIVLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cn = build_loss(loss_cn)
        self.loss_kl = build_loss(loss_kl)

        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels, teacher_labels, weight=None):
        loss = dict()
        loss['cn_loss'] = self.loss_cn(x, mod_labels, weight=weight)
        loss['kl_loss'] = self.loss_kl(x, teacher_labels, weight=weight)

        return loss

    def forward(self, x, vis_fea=False):
        x = self.classifier(x)

        return x