import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss
from ...ops import Euclidean


@HEADS.register_module()
class EuclideanHead(BaseHead):
    def __init__(self, num_classes, in_size=10560, out_size=256, is_shallow=False, loss_cls=None):
        super(EuclideanHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.is_shallow = is_shallow
        if is_shallow:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_size, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_size, self.out_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                Euclidean(self.out_size, self.num_classes),
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, vis_fea=False, is_test=False):
        x = x.reshape(-1, self.in_size)
        if vis_fea:
            if not self.is_shallow:
                for i in range(3):
                    x = self.classifier[i](x)
            pre = self.classifier[-1](x)
            if is_test:
                pre = torch.softmax(pre, dim=1)
            return dict(fea=x, pre=pre)
        else:
            pre = self.classifier(x)
            if is_test:
                pre = torch.softmax(pre, dim=1)
            return pre
