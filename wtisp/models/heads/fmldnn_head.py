import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss, build_head


@HEADS.register_module()
class FMLDNNHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, batch_size=None,
                 out_features=256, loss_cls=None,
                 aux_head=None):
        super(FMLDNNHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if aux_head is None:
            dict(
                type='IntraOrthogonalHead',
                in_features=out_features,  # keep the same as snr head
                batch_size=batch_size,  # keep the same as samples_per_gpu
                num_classes=num_classes,
                mm='inner_product',
                is_abs=False,
                loss_aux=dict(
                    type='LogisticLoss',
                    loss_weight=1,
                    temperature=100,
                ),
            ),
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cls = build_loss(loss_cls)
        self.aux_head = build_head(aux_head)

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

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x['Final'], mod_labels, weight=weight)
        loss_aux = self.aux_head.loss(x['fea'], mod_labels)
        return dict(loss_Final=loss_cls, loss_AUX=loss_aux['loss_shrinkage'])

    def forward(self, x, vis_fea=False):
        x = x.reshape(-1, self.in_features)
        fea = self.fea(x)
        Final = self.pre(fea)
        return dict(fea=fea, Final=Final)
