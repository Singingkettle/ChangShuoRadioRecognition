#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: resclnet.py
Author: Citybuster
Time: 2021/6/7 10:30
Email: chagshuo@bupt.edu.cn
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ...runner import load_checkpoint


class ResCLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rnn='lstm', dropout_rate=0.5):
        super(ResCLBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        if rnn is 'gru':
            self.rnn = nn.GRU(input_size=out_channels, hidden_size=out_channels // 2,
                              batch_first=True, num_layers=1, bidirectional=True)
        elif rnn is 'lstm':
            self.rnn = nn.LSTM(input_size=out_channels, hidden_size=out_channels // 2,
                               batch_first=True, num_layers=1, bidirectional=True)
        else:
            raise ValueError('Unknown rnn mode {}!!!'.format(rnn))

    def forward(self, x):
        cx = self.conv(x)
        cx = torch.transpose(cx, 1, 2)
        rx, _ = self.rnn(cx)
        x = cx + rx
        x = torch.transpose(x, 1, 2)

        return x


@BACKBONES.register_module()
class ResCLNetV1(nn.Module):

    def __init__(self, dropout_rate=0.5, in_channels=4, down_sample_size=None):
        super(ResCLNetV1, self).__init__()
        if down_sample_size is not None:
            self.has_down_sample = True
            self.down_sample_size = down_sample_size
        else:
            self.has_down_sample = False

        self.backbone = nn.Sequential(
            ResCLBlock(in_channels=in_channels, out_channels=256, dropout_rate=dropout_rate),
            ResCLBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate),
            ResCLBlock(in_channels=256, out_channels=80, dropout_rate=dropout_rate),
        )

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)
                if isinstance(m, nn.GRU):
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
        x = torch.squeeze(x)
        if self.has_down_sample:
            x = F.interpolate(x, self.down_sample_size)
        x = self.backbone(x)
        x = torch.sum(x, dim=2)

        return x
