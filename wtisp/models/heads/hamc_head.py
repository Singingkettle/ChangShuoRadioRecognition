import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_head


@HEADS.register_module()
class HAMCHead(BaseHead):
    def __init__(self, loss_prefix, heads):
        super(HAMCHead, self).__init__()
        self.loss_prefix = loss_prefix
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels, weight=None):
        loss = dict()
        for i in range(self.num_head):
            loss['loss_{}'.format(self.loss_prefix[i])] = self.classifier_head[i].loss(
                x[self.loss_prefix[i]], mod_labels=mod_labels)['loss_cls']

        return loss

    def forward(self, x):
        outs = dict()
        for i in range(self.num_head):
            outs[self.loss_prefix[i]] = self.classifier_head[i](x[i])

        return outs
