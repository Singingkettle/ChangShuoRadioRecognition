#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: shrinkage_head.py
Author: Citybuster
Time: 2021/5/31 19:15
Email: chagshuo@bupt.edu.cn
"""

import torch

from .base_head import BaseHead
from ..builder import HEADS, build_loss


def _bmm_cosine_similarity(a, b, eps=1e-8):
    """
        added eps for numerical stability
        """
    a_n, b_n = a.norm(dim=2)[:, :, None], b.norm(dim=2)[:, :, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    x = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return x


def _mm_cosine_similarity(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    x = torch.mm(a_norm, b_norm.transpose(0, 1))
    return x


def _bmm_inner_product(a, b):
    x = torch.bmm(a, b.transpose(1, 2))
    return x


def _mm_inner_product(a, b):
    x = torch.mm(a, b.transpose(0, 1))
    return x


_MMFUNCTIONS = dict(inner_product=_mm_inner_product, cosine=_mm_cosine_similarity)
_BMMFUNCTIONS = dict(inner_product=_bmm_inner_product, cosine=_bmm_cosine_similarity)


@HEADS.register_module()
class ShrinkageHead(BaseHead):
    def __init__(self, num_classes, expansion=1, mm='inner_product', loss_shrinkage=None):
        super(ShrinkageHead, self).__init__()
        self.num_classes = num_classes
        self.expansion = expansion
        self.mm = mm
        if mm in _MMFUNCTIONS:
            self.mm_f = _MMFUNCTIONS[mm]
        else:
            raise ValueError('Unknown mm mode {}!!!'.format(mm))
        self.loss_shrinkage = build_loss(loss_shrinkage)

    def init_weights(self):
        pass

    def loss(self, x, mod_labels=None, **kwargs):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.mm_f(x, x)
        x = x * self.expansion

        label = x.new_full((batch_size, self.num_classes), 0)
        label[torch.arange(batch_size), mod_labels[:]] = 1
        label = torch.mm(label, torch.transpose(label, 0, 1))
        label[torch.arange(batch_size), torch.arange(batch_size)] = 0
        loss_shrinkage = self.loss_shrinkage(x, label)
        return dict(loss_shrinkage=loss_shrinkage)
