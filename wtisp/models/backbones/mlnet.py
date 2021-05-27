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


@BACKBONES.register_module()
class MLNetV1(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV1, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate, batch_first=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate, batch_first=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.iq_gru(c_x)

        return x[:, -1, :]

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.ap_gru(c_x)

        return x[:, -1, :]

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV2(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV2, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate, batch_first=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate, batch_first=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.iq_gru(c_x)

        return x[:, -1, :]

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.ap_gru(c_x)

        return x[:, -1, :]

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV3(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV3, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.iq_gru(c_x)

        return x[:, -1, :]

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.ap_gru(c_x)

        return x[:, -1, :]

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV4(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV4, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.iq_gru(c_x)

        return torch.add(x[:, 0, :], x[:, -1, :])

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.ap_gru(c_x)

        return torch.add(x[:, 0, :], x[:, -1, :])

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV5(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(MLNetV5, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.iq_se = SELayer(127)
            self.ap_se = SELayer(127)
        else:
            self.iq_se = SELayer(122)
            self.ap_se = SELayer(122)
            self.has_avg_pool = False

        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.iq_se(c_x)
        x, _ = self.iq_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.ap_se(c_x)
        x, _ = self.ap_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV6(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV6, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate, batch_first=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=3, dropout=dropout_rate, batch_first=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.iq_gru(c_x)

        return x[:, -1, :]

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.ap_gru(c_x)

        return x[:, -1, :]

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV7(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV7, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)
        self.iq_se = SELayer(122)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.iq_se(c_x)
        x, _ = self.iq_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, iqs):
        iq_fea = self.forward_iqs(iqs)
        return iq_fea


@BACKBONES.register_module()
class MLNetV8(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV8, self).__init__()
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)
        self.ap_se = SELayer(122)

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

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.ap_se(c_x)
        x, _ = self.ap_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, aps):
        ap_fea = self.forward_aps(aps)

        return ap_fea


@BACKBONES.register_module()
class MLNetV9(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(MLNetV9, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True)

        self.iq_se = SELayer(122)
        self.ap_se = SELayer(122)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.iq_se(c_x)
        x, _ = self.iq_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.ap_se(c_x)
        x, _ = self.ap_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV10(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(MLNetV10, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)
        self.iq_att = nn.Linear(100, 1)
        self.ap_att = nn.Linear(100, 1)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        x, _ = self.iq_gru(c_x)
        w = self.iq_att(x)
        w = F.softmax(w, dim=1)
        x = torch.mul(x, w)
        x = torch.sum(x, dim=1)

        return x

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        x, _ = self.ap_gru(c_x)
        w = self.ap_att(x)
        w = F.softmax(w, dim=1)
        x = torch.mul(x, w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV11(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(MLNetV11, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.iq_se = SELayer(127)
            self.ap_se = SELayer(127)
        else:
            self.iq_se = SELayer(122)
            self.ap_se = SELayer(122)
            self.has_avg_pool = False

        self.iq_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.ap_gru = nn.GRU(input_size=80, hidden_size=50,
                             num_layers=2, dropout=dropout_rate,
                             batch_first=True, bidirectional=True)

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.iq_se(c_x)
        x, _ = self.iq_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.ap_se(c_x)
        x, _ = self.ap_gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)


@BACKBONES.register_module()
class MLNetV12(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(MLNetV12, self).__init__()
        # For low snr
        self.iq_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # For high snr
        self.ap_conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

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

    def forward_iqs(self, x):
        x = self.iq_conv_net(x)
        x = x.view(-1, 9760)

        return x

    def forward_aps(self, x):
        x = self.ap_conv_net(x)
        x = x.view(-1, 9760)

        return x

    def forward(self, iqs, aps):
        iq_fea = self.forward_iqs(iqs)
        ap_fea = self.forward_aps(aps)
        with torch.no_grad():
            snr_fea = torch.add(iq_fea, ap_fea)

        return (snr_fea, iq_fea, ap_fea)
