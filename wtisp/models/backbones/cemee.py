#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: cemee.py
Author: Citybuster
Time: 2021/8/21 10:57
Email: chagshuo@bupt.edu.cn
"""
import logging

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        if in_channels < reduction:
            nhid = 2
        else:
            nhid = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, nhid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


@BACKBONES.register_module()
class CEMEENet(nn.Module):

    def __init__(self, dropout_rate=0.5, in_features=2, skip_connection=True, reduction=16):
        super(CEMEENet, self).__init__()
        self.skip_connection = skip_connection
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=(1, 3), stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=2),

            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=2),

            nn.Conv2d(256, 100, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=2),
        )
        self.se = SELayer(148, reduction=reduction)

        self.gru = nn.GRU(input_size=100, hidden_size=50,
                          num_layers=2, dropout=dropout_rate,
                          batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GRU):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(3, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(3, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)

        if self.skip_connection:
            x = x + c_x

        x = torch.sum(x, dim=1)

        return x
