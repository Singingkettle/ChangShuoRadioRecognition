import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss, build_head
from ...runner import Sequential


@HEADS.register_module()
class SSRCNNHead(BaseHead):
    """
    Refer the official code: https://github.com/YihongDong/Semi-supervised-Signal-Recognition/blob/main/train.py,
    We rewrite for supervised version.
    Specifically, the original: loss = loss_cross + loss_unlabeled_cross + lam1 * loss_kl + lam2 * loss_cent
    while the supervised version is: loss = loss_cross + lam2 * loss_cent
    """

    def __init__(self, num_classes, in_size=10560, out_size=256, loss_cls=None,
                 center_head=None, is_reg=True, init_cfg=None):
        super(SSRCNNHead, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
            )
        if center_head is None:
            center_head = dict(
                type='CenterHead',
                in_size=out_size,  # keep the same as snr head
                num_classes=num_classes,
                loss_center=dict(
                    type='CenterLoss',
                    loss_weight=0.003,
                ),
            )

        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.center_head = build_head(center_head)
        self.is_reg = is_reg

        self.fea = Sequential(
            nn.Linear(self.in_size, self.out_size),
        )
        self.pre = Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_size, self.num_classes),
        )

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss_center = self.center_head.forward_train(inputs['fea'], targets['modulations'], weight=weight)
        loss_cls = self.loss_cls(inputs['pre'], targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls, loss_center=loss_center['loss_center'])

    def forward(self, x, vis_fea=True, is_test=False):
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
