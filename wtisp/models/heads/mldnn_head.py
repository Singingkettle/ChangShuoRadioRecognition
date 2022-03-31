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
        if loss_cls['type'] is 'NLLLoss':
            self.with_log = True
        else:
            self.with_log = False

    def init_weights(self):
        pass

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x):
        SNR_p = self.softmax(x['SNR'])
        Low_p = self.softmax(x['Low'])
        High_p = self.softmax(x['High'])

        Low_snr_pred = torch.mul(Low_p, SNR_p[:, -1:])

        High_snr_pred = torch.mul(High_p, SNR_p[:, :1])
        Final_pre = torch.add(Low_snr_pred, High_snr_pred)

        if self.with_log:
            Final_pre = torch.where(Final_pre > 0, Final_pre, Final_pre.new_tensor(1))
            Final_pre = torch.log(Final_pre)

        return Final_pre


@HEADS.register_module()
class MLDNNHead(BaseHead):
    def __init__(self, heads):
        super(MLDNNHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['SNR'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['Low'], mod_labels=mod_labels)
        high_loss = self.classifier_head[2].loss(
            x['High'], mod_labels=mod_labels)
        Final_loss = self.classifier_head[3].loss(
            x['Final'], mod_labels=mod_labels)

        losses['loss_SNR'] = snr_loss['loss_cls']
        losses['loss_Low'] = low_loss['loss_cls']
        losses['loss_High'] = high_loss['loss_cls']
        losses['loss_Final'] = Final_loss['loss_cls']

        return losses

    def forward(self, x):
        SNR = self.classifier_head[0](x[0])
        Low = self.classifier_head[1](x[1])
        High = self.classifier_head[2](x[2])
        x = dict(SNR=SNR, Low=Low, High=High)
        Final = self.classifier_head[3](x)

        return dict(SNR=SNR, Low=Low, High=High, Final=Final)
