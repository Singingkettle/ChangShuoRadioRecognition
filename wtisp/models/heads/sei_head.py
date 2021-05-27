import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class SEIHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, out_features=256, loss_cls=None):
        super(SEIHead, self).__init__()
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
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, dev_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, dev_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x):
        x = x.reshape(-1, self.in_features)
        x = self.classifier(x)

        return x


@HEADS.register_module()
class SEICCHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, out_features=256, balance_weight=0.1, loss_cls=None,
                 loss_con=None):
        super(SEICCHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if loss_con is None:
            loss_con = dict(
                type='ContrastiveLoss',
                margin=3,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.balance_weight = balance_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_con = build_loss(loss_con)
        self.embedding_fea = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
        )

        self.class_fea = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, dev_labels=None, from_same_dev=None, weight=None, con_weigh=None, **kwargs):
        loss_cls = self.loss_cls(x['c_x'], dev_labels, weight=weight)
        loss_con = self.loss_con(x['e_x'], from_same_dev, weight=con_weigh)
        return dict(loss_cls=loss_cls, loss_con=loss_con)

    def forward(self, x):
        x = x.reshape(-1, self.in_features)
        e_x = self.embedding_fea(x)
        c_x = self.class_fea(e_x)

        return dict(e_x=e_x, c_x=c_x)
