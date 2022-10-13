import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class SEDNNHead(BaseHead):
    def __init__(self, num_mod, num_snr, snrs=None,
                 in_features=256, mod_out_features=256, snr_out_features=256,
                 loss_mod=None, loss_snr=None, loss_merge=None):
        super(SEDNNHead, self).__init__()
        self.num_mod = num_mod
        self.num_snr = num_snr
        self.snrs = snrs
        self.amc_classifier = nn.Sequential(
            nn.Conv2d(in_features, num_snr * mod_out_features, stride=1, padding=0, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(num_snr * mod_out_features, num_snr * num_mod, groups=num_snr, stride=1, padding=0, kernel_size=1)
        )
        self.snr_classifier = nn.Sequential(
            nn.Conv2d(in_features, snr_out_features, stride=1, padding=0, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(snr_out_features, num_snr, stride=1, padding=0, kernel_size=1)
        )
        if loss_mod is None:
            loss_mod = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        if loss_snr is None:
            loss_snr = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        if loss_merge is None:
            loss_merge = dict(
                type='NLLLoss',
            )

        self.loss_mod = build_loss(loss_mod)
        self.loss_snr = build_loss(loss_snr)
        self.loss_merge = build_loss(loss_merge)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def merge(self, s, m):
        s = F.softmax(s, dim=1)  # B * K1
        s = torch.unsqueeze(s, dim=2)
        m = F.softmax(m, dim=2)  # B * K1 * K2
        f = torch.mul(s, m)
        f = torch.sum(f, dim=1)

        return f

    def loss(self, x, mod_labels=None, snr_labels=None):
        losses = dict()

        MOD = []
        for mod_head_index in range(self.num_snr):
            head_name = 'Mod-' + self.snrs[mod_head_index]
            MOD.append(torch.unsqueeze(x[head_name], dim=1))
        MOD = torch.concat(MOD, dim=1)

        s = []
        for batch_index in range(MOD.shape[0]):
            s.append(torch.unsqueeze(MOD[batch_index, snr_labels[batch_index], :], dim=0))
        s = torch.concat(s, dim=0)

        Final = torch.where(x['Final'] > 0, x['Final'], x['Final'].new_tensor(1))
        Final = torch.log(Final)

        snr_loss = self.loss_snr(x['SNR'], snr_labels)
        mod_loss = self.loss_mod(s, mod_labels)
        final_loss = self.loss_merge(Final, mod_labels)

        losses['loss_SNR'] = snr_loss
        losses['loss_MOD'] = mod_loss
        losses['loss_Final'] = final_loss

        return losses

    def forward(self, x, vis_fea=False, is_test=False):
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=3)
        SNR = self.snr_classifier(x)  # B * K1 * 1 * 1
        SNR = torch.squeeze(SNR)  # B * K1
        MOD = self.amc_classifier(x)  # B * (K1 * K2) * 1 * 1
        MOD = torch.squeeze(MOD)  # B * (K1 * K2)
        MOD = torch.reshape(MOD, [-1, self.num_snr, self.num_mod])  # B * K1 * K2
        MERGE = self.merge(SNR, MOD)

        y = dict(SNR=SNR)
        for mod_head_index in range(self.num_snr):
            head_name = 'Mod-' + self.snrs[mod_head_index]
            y[head_name] = MOD[:, mod_head_index, :]

        y['Final'] = MERGE

        return y
