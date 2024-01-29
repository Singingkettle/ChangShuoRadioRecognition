import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
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
    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=False, is_BIGRU=False, fusion_method='', is_init=False):
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
        if is_init:
            self._init_weights()

    def _init_weights(self):
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
class MLDNNSingleBranch(BaseBackbone):
    """`MLDNN <https://ieeexplore.ieee.org/document/9462447>`_ backbone
    The input for MLDNN is two mod (iq [1*2*L] and ap [1*2*L] ) for the smae frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, dropout_rate=0.5, avg_pool=None, use_GRU=True, is_BIGRU=True,
                 fusion_method='safn', init_cfg=None):
        super(MLDNNSingleBranch, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.features = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method, is_init=True)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)


@BACKBONES.register_module()
class MLDNN(BaseBackbone):
    """`MLDNN <https://link.springer.com/chapter/10.1007/978-3-319-44188-7_16>`_ backbone
    The input for MLDNN is two mod (iq [1*2*L] and ap [1*2*L] ) for the smae frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, dropout_rate=0.5, avg_pool=None, use_GRU=False, is_BIGRU=False, fusion_method='',
                 gradient_truncation=False, init_cfg=None):
        super(MLDNN, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.ap_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method, is_init=True)
        self.iq_net = SingleBranch(dropout_rate, avg_pool, use_GRU, is_BIGRU, fusion_method, is_init=True)
        self.gradient_truncation = gradient_truncation
        if self.num_classes > 0:
            self.classifier_ap = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes),
            )
            self.classifier_iq = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes),
            )
            self.classifier_snr = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )

    def forward(self, x):
        ap = self.ap_net(x['ap'])
        iq = self.iq_net(x['iq'])
        if self.gradient_truncation:
            with torch.no_grad():
                snr = torch.add(ap, iq)
        else:
            snr = torch.add(ap, iq)

        if self.num_classes > 0:
            ap = self.classifier_ap(ap)
            iq = self.classifier_iq(iq)
            snr = self.classifier_snr(snr)

            ap_p = F.softmax(ap, dim=1)
            iq_p = F.softmax(iq, dim=1)
            snr_p = F.softmax(snr, dim=1)
            merge = torch.mul(ap_p, snr_p[:, :1]) + torch.mul(iq_p, snr_p[:, -1:])
            if self.training:
                return (merge, ap, iq, snr)
            else:
                return (merge,)

        return (ap, iq, snr)
