import torch.nn as nn
import torch
from .base_head import BaseHead
from ..builder import HEADS, build_loss, build_head


@HEADS.register_module()
class FastMLDNNHead(BaseHead):
    def __init__(self, num_classes, in_size=10560, out_size=256, loss_cls=None, shrinkage_head=None):
        super(FastMLDNNHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if shrinkage_head is None:
            dict(
                type='IntraOrthogonalHead',
                in_size=out_size,  # keep the same as snr head
                num_classes=num_classes,
                mm='inner_product',
                is_abs=False,
                loss_shrinkage=dict(
                    type='LogisticLoss',
                    loss_weight=1,
                    temperature=100,
                ),
            ),
        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = out_size
        self.loss_cls = build_loss(loss_cls)
        self.shrinkage_head = build_head(shrinkage_head)

        self.fea = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            nn.ReLU(inplace=True),
        )
        self.pre = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_size, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss_cls = self.loss_cls(inputs['pre'], targets['modulations'], weight=weight)
        loss_shrinkage = self.shrinkage_head.loss(inputs['fea'], targets['modulations'])
        return dict(loss_cls=loss_cls, loss_shrinkage=loss_shrinkage['loss_shrinkage'])

    def forward(self, x, vis_fea=False, is_test=False):
        x = x.reshape(-1, self.in_size)
        fea = self.fea(x)
        pre = self.pre(fea)
        if is_test:
            pre = torch.softmax(pre, dim=1)
            return pre
        else:
            return dict(fea=x, pre=pre)
