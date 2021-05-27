import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_head


@HEADS.register_module()
class LocationHead(BaseHead):
    def __init__(self, heads, loss_cls=None):
        super(LocationHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, px=None, py=None):
        losses = dict()
        loss_pos_x = self.classifier_head[0].loss(
            x['fea_x'], mod_labels=px)
        loss_pos_y = self.classifier_head[0].loss(
            x['fea_x'], mod_labels=py)

        losses['loss_pos_x'] = loss_pos_x['loss_cls']
        losses['loss_pos_y'] = loss_pos_y['loss_cls']

        return losses

    def forward(self, x):
        fea_x = self.classifier_head[0](x)
        fea_y = self.classifier_head[1](x)

        return dict(fea_x=fea_x, fea_y=fea_y)
