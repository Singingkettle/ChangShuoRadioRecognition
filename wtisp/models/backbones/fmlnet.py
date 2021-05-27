import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ...runner import load_checkpoint


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        if in_channels < reduction:
            nhid = 2
        else:
            nhid = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, nhid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


class SELayerV2(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayerV2, self).__init__()
        if in_channels < reduction:
            nhid = 2
        else:
            nhid = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, nhid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, in_channels, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


@BACKBONES.register_module()
class FMLNetV1(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV1, self).__init__()
        # For low snr
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127)
        else:
            self.se = SELayer(122)
            self.has_avg_pool = False

        self.gru = nn.GRU(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV2(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV2, self).__init__()
        # For low snr
        self.conv_net = nn.Sequential(
            nn.Conv2d(8, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127)
        else:
            self.se = SELayer(122)
            self.has_avg_pool = False

        self.gru = nn.GRU(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV3(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV3, self).__init__()
        # For low snr
        self.conv_net = nn.Sequential(
            nn.Conv2d(8, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127)
        else:
            self.se = SELayer(122)
            self.has_avg_pool = False

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV4(nn.Module):

    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV4, self).__init__()
        # For low snr
        self.conv_net = nn.Sequential(
            nn.Conv2d(8, 128, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127)
        else:
            self.se = SELayer(122)
            self.has_avg_pool = False

        self.gru = nn.GRU(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV5(nn.Module):
    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV5, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(122)
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV6(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV6, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV7(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV7, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayerV2(15)
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV8(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV8, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV9(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV9, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV10(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV10, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.ninp = ninp
        self.pos_encoder1 = PositionalEncoding(ninp, dropout_rate)
        encoder_layers1 = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layers1, nlayers)

        self.pos_encoder2 = PositionalEncoding(ninp, dropout_rate)
        encoder_layers2 = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layers2, nlayers)

        self.pos_encoder3 = PositionalEncoding(ninp, dropout_rate)
        encoder_layers3 = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate)
        self.transformer_encoder3 = nn.TransformerEncoder(
            encoder_layers3, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        x1 = self.pos_encoder1(c_x)
        x1 = self.transformer_encoder1(x1)
        x1 = torch.mean(x1, dim=1)

        c_x2 = c_x.reshape(-1, 8, self.ninp)
        x2 = self.pos_encoder2(c_x2)
        x2 = self.transformer_encoder2(x2)
        x2 = x2.reshape(-1, 16, self.ninp)
        x2 = torch.mean(x2, dim=1)

        c_x3 = c_x.reshape(-1, 4, self.ninp)
        x3 = self.pos_encoder3(c_x3)
        x3 = self.transformer_encoder3(x3)
        x3 = x3.reshape(-1, 16, self.ninp)
        x3 = torch.mean(x3, dim=1)

        return x1 + x2 + x3


@BACKBONES.register_module()
class FMLNetV11(nn.Module):

    def __init__(self, ninp=80, nhead=2, nhid=32, nlayers=2, dropout_rate=0.5):
        super(FMLNetV11, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, ninp, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x = self.pos_encoder(c_x)
        x = self.transformer_encoder(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV12(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV12, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)

        self.gru = nn.GRU(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV13(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV13, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayerV2(15)

        self.gru = nn.GRU(input_size=80, hidden_size=50,
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

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV14(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV14, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV15(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV15, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV16(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV16, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV17(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(FMLNetV17, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 40, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.se = SELayer(15)

        self.lstm = nn.LSTM(input_size=40, hidden_size=20,
                            num_layers=3, dropout=dropout_rate,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV18(nn.Module):

    def __init__(self, ninp=32, nhead=2, nhid=32, nlayers=6, dropout_rate=0.5):
        super(FMLNetV18, self).__init__()
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            ninp, nhead, nhid, dropout_rate, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

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
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()

    def forward(self, x):
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(-1, 512 // self.ninp, self.ninp)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        return x


@BACKBONES.register_module()
class FMLNetV19(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV19, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV20(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV20, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV21(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV21, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.SELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.SELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.SELU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV22(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV22, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV23(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV23, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.CELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.CELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.CELU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.lstm(c_x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV24(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV24, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.pos_encoder = PositionalEncoding(80, dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(
            80, 4, 32, dropout_rate, activation='gelu')
        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
                            num_layers=1, batch_first=True, bidirectional=True)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x = self.pos_encoder(c_x)
        x = self.encoder_layer(x)
        x, _ = self.lstm(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV25(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV25, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se = SELayerV2(15)

        self.lstm1 = nn.LSTM(
            input_size=80,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.lstm2 = nn.LSTM(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x1, _ = self.lstm1(c_x)
        x2, _ = self.lstm2(x1)
        x = x1 + x2
        x = self.dropout(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV26(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV26, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se1 = SELayerV2(15)
        self.se2 = SELayerV2(15)

        self.lstm1 = nn.LSTM(
            input_size=80,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.lstm2 = nn.LSTM(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        x0 = torch.transpose(x, 1, 2)

        se_w1 = self.se1(x0)
        x1, _ = self.lstm1(x0)

        se_w2 = self.se2(x1)
        x2, _ = self.lstm2(x1)

        x1 = torch.mul(x1, se_w1)
        x1 = torch.sum(x1, dim=1)

        x2 = torch.mul(x2, se_w2)
        x2 = torch.sum(x2, dim=1)

        return dict(cnn=x0, lstm1=x1, lstm2=x2)


@BACKBONES.register_module()
class FMLNetV27(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV27, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se1 = SELayerV2(15)
        self.se2 = SELayerV2(15)

        self.lstm1 = nn.LSTM(
            input_size=80,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.lstm2 = nn.LSTM(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        x0 = torch.transpose(x, 1, 2)

        se_w1 = self.se1(x0)
        x1, _ = self.lstm1(x0)

        se_w2 = self.se2(x1)
        x2, _ = self.lstm2(x1)

        x1 = torch.mul(x1, se_w1)
        x1 = torch.sum(x1, dim=1)

        x2 = torch.mul(x2, se_w2)
        x2 = torch.sum(x2, dim=1)

        return dict(cnn=x0, lstm1=x1, lstm2=x2)


@BACKBONES.register_module()
class FMLNetV28(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV28, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False
        self.se1 = SELayerV2(15)
        self.se2 = SELayerV2(15)

        self.lstm1 = nn.LSTM(
            input_size=80,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.lstm2 = nn.LSTM(
            input_size=100,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        x0 = torch.transpose(x, 1, 2)

        se_w1 = self.se1(x0)
        x1, _ = self.lstm1(x0)

        se_w2 = self.se2(x1)
        x2, _ = self.lstm2(x1)

        x1 = torch.mul(x1, se_w1)
        x1 = torch.sum(x1, dim=1)

        x2 = torch.mul(x2, se_w2)
        x2 = torch.sum(x2, dim=1)

        return dict(cnn=x0, lstm1=x1, lstm2=x2)


@BACKBONES.register_module()
class FMLNetV29(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV29, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False

        self.lstm1 = nn.LSTM(
            input_size=80,
            hidden_size=40,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.lstm2 = nn.LSTM(
            input_size=80,
            hidden_size=40,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        x0 = torch.transpose(x, 1, 2)

        x1, _ = self.lstm1(x0)
        x1 = x1 + x0

        x2, _ = self.lstm2(x1)
        x2 = x2 + x1

        x0 = torch.sum(x0, dim=1)
        x1 = torch.sum(x1, dim=1)
        x2 = torch.sum(x2, dim=1)

        return dict(cnn=x0, lstm1=x1, lstm2=x2)


@BACKBONES.register_module()
class FMLNetV30(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV30, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 40, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False

        self.lstm1 = nn.LSTM(
            input_size=40,
            hidden_size=40,
            num_layers=1,
            batch_first=True)
        self.lstm2 = nn.LSTM(
            input_size=40,
            hidden_size=40,
            num_layers=1,
            batch_first=True)

        self.fpn1 = nn.Sequential(
            nn.Conv1d(15, 15, 3, padding=1),
            nn.ReLU(),
        )

        self.fpn2 = nn.Sequential(
            nn.Conv1d(15, 15, 3, padding=1),
            nn.ReLU(),
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
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        x0 = torch.transpose(x, 1, 2)

        x1, _ = self.lstm1(x0)

        rx1 = torch.flip(x1, [2])
        fx2, _ = self.lstm2(F.dropout(rx1, self.dropout_rate))

        fx1 = self.fpn1(fx2) + x1
        fx0 = self.fpn2(fx1) + x0

        fx0 = torch.mean(fx0, dim=1)
        fx1 = torch.mean(fx1, dim=1)
        fx2 = torch.mean(fx2, dim=1)

        with torch.no_grad():
            fx3 = torch.cat([fx0, fx1, fx2], dim=1)

        return dict(fx0=fx0, fx1=fx1, fx2=fx2, fx3=fx3)


@BACKBONES.register_module()
class FMLNetV31(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV31, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
                            num_layers=2, dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(c_x)

        fx0 = x[:, 0, :]
        fx1 = x[:, 7, :]
        fx2 = x[:, -1, :]

        with torch.no_grad():
            fx3 = torch.cat([c_x, x], dim=2)

        return dict(fx0=fx0, fx1=fx1, fx2=fx2, fx3=fx3)


@BACKBONES.register_module()
class FMLNetV32(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV32, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.res12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.res13 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.res23 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.scales = [511, 255, 127]
        else:
            self.has_avg_pool = False
            self.scales = [63, 31, 15]

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
                            num_layers=2, dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        rx12 = F.interpolate(self.res12(x1), size=(1, self.scales[1]), mode='bilinear', align_corners=False)
        dx2 = rx12 + x2
        x3 = self.conv3(dx2)

        rx13 = F.interpolate(self.res13(x1), size=(1, self.scales[2]), mode='bilinear', align_corners=False)
        rx23 = F.interpolate(self.res23(x2), size=(1, self.scales[2]), mode='bilinear', align_corners=False)

        x = x3 + rx13 + rx23

        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(c_x)

        fx0 = x[:, 0, :]
        fx1 = x[:, 7, :]
        fx2 = x[:, -1, :]

        with torch.no_grad():
            fx3 = torch.cat([c_x, x], dim=2)

        return dict(fx0=fx0, fx1=fx1, fx2=fx2, fx3=fx3)


@BACKBONES.register_module()
class FMLNetV33(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV33, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
                            num_layers=2, dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(c_x)

        fx0 = x[:, 0, :]
        fx1 = x[:, 7, :]
        fx2 = x[:, -1, :]


        fx3 = torch.cat([c_x, x], dim=2)

        return dict(fx0=fx0, fx1=fx1, fx2=fx2, fx3=fx3)


@BACKBONES.register_module()
class FMLNetV34(nn.Module):
    def __init__(self, dropout_rate=0.5, avg_pool=None):
        super(FMLNetV34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.res12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.res13 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.res23 = nn.Sequential(
            nn.Conv2d(128, 80, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.scales = [511, 255, 127]
        else:
            self.has_avg_pool = False
            self.scales = [63, 31, 15]

        self.lstm = nn.LSTM(input_size=80, hidden_size=50,
                            num_layers=2, dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        rx12 = F.interpolate(self.res12(x1), size=(1, self.scales[1]), mode='bilinear', align_corners=False)
        dx2 = rx12 + x2
        x3 = self.conv3(dx2)

        rx13 = F.interpolate(self.res13(x1), size=(1, self.scales[2]), mode='bilinear', align_corners=False)
        rx23 = F.interpolate(self.res23(x2), size=(1, self.scales[2]), mode='bilinear', align_corners=False)

        x = x3 + rx13 + rx23

        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(c_x)

        fx0 = x[:, 0, :]
        fx1 = x[:, 7, :]
        fx2 = x[:, -1, :]

        fx3 = torch.cat([c_x, x], dim=2)

        return dict(fx0=fx0, fx1=fx1, fx2=fx2, fx3=fx3)


