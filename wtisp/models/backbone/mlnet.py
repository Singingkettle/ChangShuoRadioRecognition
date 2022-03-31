import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ...runner import load_checkpoint


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


class SingleBranch(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=True, is_BIGRU=True, fusion_method='safn'):
        super(SingleBranch, self).__init__()
        self.dropout_rate = dropout_rate
        self.avg_pool = avg_pool
        self.use_GRU = use_GRU
        self.is_BIGRU = is_BIGRU
        self.fusion_method = fusion_method
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )
        if self.avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(self.avg_pool)
        else:
            self.has_avg_pool = False

        if self.use_GRU:
            self.gru_net = nn.GRU(input_size=80, hidden_size=50,
                                  num_layers=2, dropout=self.dropout_rate,
                                  batch_first=True, bidirectional=self.is_BIGRU)

            if self.fusion_method is 'safn':
                if avg_pool is not None:
                    self.fusion = SELayer(127)
                else:
                    self.fusion = SELayer(122)
            elif self.fusion_method is 'attention':
                self.fusion = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv_net(x)
        if self.use_GRU:
            if self.has_avg_pool:
                x = self.avg_pool_layer(x)
            x = torch.squeeze(x, dim=2)
            c_x = torch.transpose(x, 1, 2)

            if 'safn' is self.fusion_method:
                w = self.fusion(c_x)
                x, _ = self.gru_net(c_x)
                x = torch.mul(x, w)
                x = torch.sum(x, dim=1)
                return x
            elif 'attention' is self.fusion_method:
                x, _ = self.gru_net(c_x)
                w = self.fusion(x)
                w = F.softmax(w, dim=1)
                x = torch.mul(x, w)
                x = torch.sum(x, dim=1)
                return x
            elif 'add' is self.fusion_method:
                x, _ = self.gru_net(c_x)
                return torch.add(x[:, 0, :], x[:, -1, :])
            elif 'last' is self.fusion_method:
                x, _ = self.gru_net(c_x)
                return x[:, -1, :]
            else:
                x = x.view(x.shape[0], -1)
                return x
        else:
            x = x.view(x.shape[0], -1)
            return x


@BACKBONES.register_module()
class MLNet(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=True,
                 is_BIGRU=True, fusion_method='safn', gradient_truncation=False):
        super(MLNet, self).__init__()
        self.iq_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method)
        self.ap_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method)
        self.gradient_truncation = gradient_truncation

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GRU):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(3, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(3, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, iqs, aps):
        iq_fea = self.iq_net(iqs)
        ap_fea = self.ap_net(aps)
        if self.gradient_truncation:
            with torch.no_grad():
                snr_fea = torch.add(iq_fea, ap_fea)
        else:
            snr_fea = torch.add(iq_fea, ap_fea)

        return [snr_fea, iq_fea, ap_fea]
