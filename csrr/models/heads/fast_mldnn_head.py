import numpy as np
import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss
from ...ops import sim_matrix
from ...runner import Sequential


@HEADS.register_module()
class FastMLDNNHead(BaseHead):
    def __init__(self, num_classes, alpha=-0.0001, beta=1, in_size=10560,
                 out_size=256, loss_cls=None, balance=0.0, init_cfg=None, is_shallow=False, dp=0.5):
        super(FastMLDNNHead, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
            )

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.in_size = in_size
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.balance = balance

        self.is_shallow = is_shallow
        if self.is_shallow:
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
        if self.balance > 0.0:
            p = sim_matrix(self.pre[1].weight, self.pre[1].weight) * self.balance
            loss_reg = self.loss_cls(p, targets['modulations'].new_tensor(np.arange(inputs.shape[1])))
            loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)
            return dict(loss_cls=loss_cls, loss_reg=loss_reg)
        else:
            loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)
            return dict(loss_cls=loss_cls)

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


@HEADS.register_module()
class FastMLDNNHeadV2(BaseHead):
    def __init__(self, num_classes, alpha=-0.0001, beta=1,
                 out_size=256, levels=(256, 256, 100, 100), loss_cls=None, init_cfg=None, dp=0.5):
        super(FastMLDNNHeadV2, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
            )

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.num_level = len(levels)
        self.heads = []
        for i, in_size in enumerate(levels):
            head = Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Linear(out_size, num_classes)
            )
            self.add_module(f'head_{i}', head)
            self.heads.append(f'head_{i}')

    def loss(self, inputs, targets, weight=None, **kwargs):

        losses = dict()
        for i in range(self.num_level):
            loss_cls = self.loss_cls(inputs[i], targets['modulations'], weight=weight)
            losses[f'loss_head_{i}'] = loss_cls
        return losses

    def forward(self, x, vis_fea=False, is_test=False):
        if is_test:
            pre = getattr(self, self.heads[-1])(x[-1])
            pres = torch.softmax(pre, dim=1)
        else:
            pres = []
            for i in range(self.num_level):
                pre = getattr(self, self.heads[i])(x[i])
                pres.append(pre)
        return pres
