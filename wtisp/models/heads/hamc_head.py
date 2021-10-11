import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class HAMCHead(BaseHead):
    def __init__(self, num_classes, in_features=80, out_features=256, loss_cls=None):
        super(HAMCHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cls = build_loss(loss_cls)

        self.fea = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
        )
        self.pre = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels, weight=None):
        loss = dict()
        for pre in x:
            loss_name = pre.split('_')[0]
            loss[loss_name + '_loss'] = self.loss_cls(x[pre], mod_labels, weight=weight)

        return loss

    def forward(self, x, vis_fea=False):
        outs = dict()
        for bx in x:
            fea = self.fea(x[bx])
            pre = self.pre(fea)
            if vis_fea:
                outs[bx + '_fea'] = fea
            outs[bx + '_pre'] = pre

        return outs
