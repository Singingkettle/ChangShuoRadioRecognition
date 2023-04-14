import numpy as np
import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss
import torch.nn.functional as F
from ...runner import Sequential


@HEADS.register_module()
class FastMLDNNHead(BaseHead):
    def __init__(self, num_classes, alpha=-0.6, beta=1.1, in_size=10560,
                 out_size=256, loss_cls=None, loss_se=None, init_cfg=None, is_shallow=False, dp=0.5):
        super(FastMLDNNHead, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                weight=1
            )
        if loss_se is None:
            loss_se = dict(
                type='CrossEntropyLoss',
                weight=0.1
            )

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.in_size = in_size
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.loss_se = build_loss(loss_se)
        if is_shallow:
            self.fea = nn.Identity()
        else:
            self.fea = Sequential(
                nn.Linear(self.in_size, self.out_size),
                nn.ReLU(inplace=True),
            )
        self.pre = Sequential(
            nn.Dropout(dp),
            nn.Linear(self.out_size, self.num_classes),
        )

    def loss(self, inputs, targets, weight=None, **kwargs):

        loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)

        mask = F.one_hot(targets['modulations'], num_classes=inputs.shape[1])
        inputs = self.beta + (self.alpha - self.beta) * mask + inputs
        loss_se = self.loss_se(inputs, targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls, loss_se=loss_se)

    def forward(self, x, vis_fea=False, is_test=False):
        x = x.reshape(-1, self.in_size)
        fea = self.fea(x)
        pre = self.pre(fea)
        if vis_fea:
            if is_test:
                pre = torch.softmax(pre, dim=1)
            return dict(fea=fea, pre=pre)
        else:
            if is_test:
                pre = torch.softmax(pre, dim=1)
            return pre
