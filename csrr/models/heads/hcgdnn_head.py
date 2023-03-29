from .amc_head import AMCHead
from .base_head import BaseHead
from .euclidean_head import EuclideanHead
from ..builder import HEADS


@HEADS.register_module()
class HCGDNNHead(BaseHead):
    def __init__(self, num_classes, heads=None, in_size=80, out_size=256, loss_cls=None, use_eh=False, init_cfg=None):
        super(HCGDNNHead, self).__init__(init_cfg)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if heads is None:
            heads = ['CNN', 'BIGRU1', 'BIGRU2']
        self.heads = heads
        for layer_name in self.heads:
            if use_eh:
                self.add_module(layer_name, EuclideanHead(num_classes, in_size, out_size, loss_cls=loss_cls))
            else:
                self.add_module(layer_name, AMCHead(num_classes, in_size, out_size, loss_cls))

    def loss(self, inputs, targets, weight=None, **kwargs):
        loss = dict()
        for layer_name in self.heads:
            loss['loss_' + layer_name] = getattr(self, layer_name).loss(inputs[layer_name], targets, weight=weight)[
                'loss_cls']

        return loss

    def forward(self, inputs, vis_fea=False, is_test=False):
        outputs = dict()
        for layer_name in self.heads:
            sub_outs = getattr(self, layer_name)(inputs[layer_name], vis_fea, is_test)
            if vis_fea:
                sub_outs[f'fea_{layer_name}'] = sub_outs['fea']
                sub_outs.pop('fea', None)
                sub_outs[layer_name] = sub_outs['pre']
                sub_outs.pop('pre', None)
            else:
                sub_outs = {layer_name: sub_outs}
            outputs.update(sub_outs)

        return outputs
