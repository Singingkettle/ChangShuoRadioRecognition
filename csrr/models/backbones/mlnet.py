import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone
from ..builder import BACKBONES


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
    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=False, is_BIGRU=False, fusion_method=''):
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

            if self.fusion_method == 'safn':
                if avg_pool is not None:
                    self.fusion = SELayer(127)
                else:
                    self.fusion = SELayer(122)
            elif self.fusion_method == 'attention':
                self.fusion = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv_net(x)
        if self.use_GRU:
            if self.has_avg_pool:
                x = self.avg_pool_layer(x)
            x = torch.squeeze(x, dim=2)
            c_x = torch.transpose(x, 1, 2)

            if 'safn' == self.fusion_method:
                w = self.fusion(c_x)
                x, _ = self.gru_net(c_x)
                x = torch.mul(x, w)
                x = torch.sum(x, dim=1)
                return x
            elif 'attention' == self.fusion_method:
                x, _ = self.gru_net(c_x)
                w = self.fusion(x)
                w = F.softmax(w, dim=1)
                x = torch.mul(x, w)
                x = torch.sum(x, dim=1)
                return x
            elif 'add' == self.fusion_method:
                x, _ = self.gru_net(c_x)
                return torch.add(x[:, 0, :], x[:, -1, :])
            elif 'last' == self.fusion_method:
                x, _ = self.gru_net(c_x)
                return x[:, -1, :]
            else:
                x, _ = self.gru_net(c_x)
                x = x.reshape(x.shape[0], -1)
                return x
        else:
            x = x.reshape(x.shape[0], -1)
            return x


@BACKBONES.register_module()
class MLNet(BaseBackbone):

    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=False, is_BIGRU=False, fusion_method='',
                 gradient_truncation=False, init_cfg=None):
        super(MLNet, self).__init__(init_cfg)
        self.iq_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method)
        self.ap_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method)
        self.gradient_truncation = gradient_truncation

    def forward(self, iqs, aps):
        low = self.iq_net(iqs)
        high = self.ap_net(aps)
        if self.gradient_truncation:
            with torch.no_grad():
                snr = torch.add(low, high)
        else:
            snr = torch.add(low, high)

        return dict(snr=snr, low=low, high=high)
