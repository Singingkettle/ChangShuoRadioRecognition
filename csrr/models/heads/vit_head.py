import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class VitHead(BaseHead):
    def __init__(self, num_classes, in_size=80, loss_cls=None, init_cfg=None):
        super(VitHead, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.loss_cls = build_loss(loss_cls)
        self.in_size = in_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_size),
            nn.Linear(in_size, num_classes)
        )

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, vis_fea=False, is_test=False):
        x = x.reshape(-1, self.in_size)

        if is_test:
            if vis_fea:
                if not self.is_shallow:
                    for i in range(3):
                        x = self.classifier[i](x)
                pre = self.classifier[-1](x)
                pre = torch.softmax(pre, dim=1)
                return dict(fea=x, pre=pre, center=self.classifier[-1].weight)
            else:
                pre = self.classifier(x)
                pre = torch.softmax(pre, dim=1)
                return pre
        else:
            pre = self.classifier(x)
            return pre
