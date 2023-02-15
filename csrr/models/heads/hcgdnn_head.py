import torch.nn as nn

from .classification_head import ClassificationHead
from .base_head import BaseHead
from ..builder import HEADS


@HEADS.register_module()
class HCGDNNHead(BaseHead):
    def __init__(self, num_classes, heads=None, in_features=80, out_features=256, loss_cls=None):
        super(HCGDNNHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if heads is None:
            heads = ['CNN', 'BIGRU1', 'BIGRU2']
        self.heads = heads
        for layer_name in self.heads:
            self.add_module(layer_name, ClassificationHead(num_classes, in_features, out_features, loss_cls))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss = dict()
        for layer_name in self.heads:
            loss['loss_' + layer_name] = getattr(self, layer_name).loss(x[layer_name], mod_labels, weight=weight)[
                'loss_Final']

        return loss

    def forward(self, x, vis_fea=False, is_test=False):
        outs = dict()
        for layer_name in self.heads:
            sub_outs = getattr(self, layer_name)(x[layer_name], vis_fea)
            if vis_fea:
                sub_outs[layer_name + '_fea'] = sub_outs['fea']
                sub_outs.pop('fea', None)
                sub_outs[layer_name] = sub_outs['Final']
                sub_outs.pop('Final', None)
            else:
                sub_outs = {layer_name: sub_outs}
            outs.update(sub_outs)

        return outs
