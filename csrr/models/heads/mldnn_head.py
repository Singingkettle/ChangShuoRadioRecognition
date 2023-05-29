import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_head, build_loss


@HEADS.register_module()
class MergeAMCHead(BaseHead):
    def __init__(self, loss_cls=None):
        super(MergeAMCHead, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.loss_cls = build_loss(loss_cls)
        if loss_cls['type'] == 'NLLLoss':
            self.with_log = True
        else:
            self.with_log = False

    def init_weights(self):
        pass

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x):
        snr = self.softmax(x['snr'])
        low = self.softmax(x['low'])
        high = self.softmax(x['high'])

        low_snr = torch.mul(low, snr[:, -1:])
        high_snr = torch.mul(high, snr[:, :1])
        merge = torch.add(low_snr, high_snr)

        if self.with_log:
            merge = torch.where(merge > 0, merge, merge.new_tensor(1))
            merge = torch.log(merge)

        return dict(snr=snr, low=low, high=high, merge=merge)


@HEADS.register_module()
class MLDNNHead(BaseHead):
    def __init__(self, heads):
        super(MLDNNHead, self).__init__()
        self.num_head = len(heads)
        self.heads = []
        for name in heads:
            self.heads.append(name)
            head_block = build_head(heads[name])
            self.add_module(name, head_block)

    def init_weights(self):
        for name in self.heads:
            getattr(self, name).init_weights()

    def loss(self, inputs, targets, **kwargs):
        losses = dict()
        for key in inputs:
            if 'snr' in key:
                losses[f'loss_{key}'] = getattr(self, key).loss(inputs[key], targets['snrs'])['loss_cls']
            else:
                losses[f'loss_{key}'] = getattr(self, key).loss(inputs[key], targets['modulations'])['loss_cls']

        return losses

    def forward(self, inputs, vis_fea=False, is_test=False):
        outputs = dict()
        for key in inputs:
            outputs[key] = getattr(self, key)(inputs[key])
        outputs = getattr(self, 'merge')(outputs)
        if is_test:
            outputs = outputs['merge']
        if vis_fea:
            raise NotImplementedError('The vis fea for MLDNN is not supported!')

        return outputs
