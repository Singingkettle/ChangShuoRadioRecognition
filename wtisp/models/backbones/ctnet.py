import logging

import torch
import torch.nn as nn

from .cnnnet import Truncation2d
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


class ConvBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=1):
        super(ConvBasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=(1, stride)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvNet(nn.Module):
    arch_settings = {
        2: ((256, 1), (80, 2)),
        3: ((256, 1), (80, 1), (80, 2)),
        4: ((256, 1), (80, 1), (80, 1), (80, 2)),
    }

    def __init__(self, depth, in_channels, sequence_length=128, kernel_size=(1, 3), avg_pool=None, out_indices=(1,)):
        super(ConvNet, self).__init__()
        self.out_indices = out_indices
        filters = self.arch_settings[depth]
        self.out_channels = filters[-1][0]
        self.cnnlayers = []
        self.num_step = sequence_length
        for i, filter_config in enumerate(filters):
            conv_layer = self.make_conv_layer(
                in_channels=in_channels, out_channels=filter_config[0],
                kernel_size=kernel_size, stride=filter_config[1])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, conv_layer)
            self.cnnlayers.append(layer_name)
            in_channels = filter_config[0]
            self.num_step = (
                                    self.num_step - kernel_size[1]) // filter_config[1] + 1

        if avg_pool is not None:
            avg_pool_layer = nn.Sequential(
                Truncation2d(len(filters)),
                nn.AvgPool2d(avg_pool),
            )
            layer_name = 'avg_pool_layer'
            self.add_module(layer_name, avg_pool_layer)
            self.cnnlayers.append(layer_name)
        else:
            self.has_avg_pool = False

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
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight.data, mean=1, std=0.02)
                    nn.init.constant_(m.bias.data, 0)

    def make_conv_layer(self, **kwargs):
        return ConvBasicBlock(**kwargs)

    def forward(self, x):
        for i, layer_name in enumerate(self.cnnlayers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x


class TransformerNet(nn.Module):

    def __init__(self, depth, input_size, num_step):
        super(TransformerNet, self).__init__()
        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=8, dim_feedforward=100, dropout=0.5)
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=8, dim_feedforward=100, dropout=0.5)
        self.se_channel_attention = SELayer(num_step)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder1(x)
        se_w = self.se_channel_attention(x)
        x = self.encoder2(x)
        x = torch.mul(x, se_w)
        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class CTNet(nn.Module):

    def __init__(self, in_channels, cnn_depth, tnn_depth, sequence_length, avg_pool=None, out_indices=(1,)):
        super(CTNet, self).__init__()
        self.cnn_net = self.make_cnn_net(
            depth=cnn_depth, in_channels=in_channels,
            sequence_length=sequence_length, avg_pool=avg_pool, out_indices=out_indices)
        self.tnn_net = self.make_tnn_net(
            depth=tnn_depth, input_size=self.cnn_net.out_channels, num_step=self.cnn_net.num_step)

    def make_cnn_net(self, **kwargs):
        return ConvNet(**kwargs)

    def make_tnn_net(self, **kwargs):
        return TransformerNet(**kwargs)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            self.cnn_net.init_weights()
            self.tnn_net.init_weights()

    def forward(self, x):
        x = self.cnn_net(x)
        x = torch.squeeze(x, dim=2)  # (N, C, 1, W)
        x = torch.transpose(x, 1, 2)  # (N, W, C)
        x = self.tnn_net(x)
        return x


@BACKBONES.register_module()
class CTNetV2(nn.Module):

    def __init__(self):
        super(CTNetV2, self).__init__()
        self.tnn_net = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=32),
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=32),
            nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=32),
        )

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.squeeze(x, dim=2)  # (N, C, W)
        x = torch.transpose(x, 1, 2)  # (N, W, C)
        x = torch.reshape(x, (-1, 8, 32))
        x = self.tnn_net(x)
        return x
