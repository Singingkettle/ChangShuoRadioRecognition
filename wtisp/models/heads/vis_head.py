import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class VisHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, out_features=256,
                 loss_cls=None):
        super(VisHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_Final = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_Final=loss_Final)

    def forward(self, x, vis_fea=False):
        x = x.reshape(-1, self.in_features)
        if vis_fea:
            x = self.classifier[0](x)
            fea = self.classifier[1](x)
            x = self.classifier[2](fea)
            pre = self.classifier[3](x)
            return dict(fea=fea, Final=pre)
        else:
            pre = self.classifier(x)
            return pre