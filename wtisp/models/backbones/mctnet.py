#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: wtisignalprocessing
File: mctnet.py
Author: Citybuster
Time: 2021/9/16 18:57
Email: chagshuo@bupt.edu.cn
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.activation import MultiheadAttention

from ..builder import BACKBONES
from ...runner import load_checkpoint


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class MyTransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(12)
        self.norm2 = nn.BatchNorm1d(12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MyTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        # src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src


@BACKBONES.register_module()
class MCTNetV1(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(MCTNetV1, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 256, kernel_size=(1, 11), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ninp, kernel_size=(1, 5), stride=(1, 2)),
            nn.ReLU(inplace=True),
        )
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x = self.transformer_encoder(c_x)
        x = x[:, 0, :]

        return x


@BACKBONES.register_module()
class MCTNetV2(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(MCTNetV2, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 256, kernel_size=(1, 11), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ninp, kernel_size=(1, 5), stride=(1, 2)),
            nn.ReLU(inplace=True),
        )
        encoder_layers = MyTransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x = self.transformer_encoder(c_x)
        x = x[:, 0, :]

        return x
