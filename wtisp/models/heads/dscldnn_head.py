import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class DSCLDNNHead(BaseHead):
    def __init__(self, num_classes, in_features=2500, loss_cls=None):
        super(DSCLDNNHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, self.in_features)
        pre = self.classifier(x)

        return pre
