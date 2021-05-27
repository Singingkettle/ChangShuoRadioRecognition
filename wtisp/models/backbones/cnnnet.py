import logging

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


class Truncation2d(nn.Module):
    def __init__(self, length=0):
        super(Truncation2d, self).__init__()
        self.left_index = length
        self.right_index = -1 * length

    def forward(self, x):
        x = x[:, :, :, self.left_index:self.right_index]

        return x


class CNNBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad=None, dropout_rate=None):
        super(CNNBasicBlock, self).__init__()
        self.cnnblock = []
        if pad is not None:
            if isinstance(pad, int):
                pad_layer = nn.ZeroPad2d((pad, pad, 0, 0))
                self.add_module('pad_layer', pad_layer)
                self.cnnblock.append('pad_layer')
            else:
                raise NotImplementedError(
                    'The type of augment pad is only support int. Please check your code')

        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        )

        self.add_module('conv_layer', conv_layer)
        self.cnnblock.append('conv_layer')

        if dropout_rate is not None:
            if isinstance(dropout_rate, float):
                dropout_layer = nn.Dropout(p=dropout_rate)
                self.add_module('dropout_layer', dropout_layer)
                self.cnnblock.append('dropout_layer')
            else:
                raise NotImplementedError(
                    'The type of augment dropout_rate is only support float ([0.0, 1.0]) or int ((0.0, 1.0)). Please check your code')

    def forward(self, x):
        for layer_name in self.cnnblock:
            layer = getattr(self, layer_name)
            x = layer(x)

        return x


@BACKBONES.register_module()
class CNNNet(nn.Module):
    arch_settings = {
        2: ((256, 80), (2, 0.5, 1)),
        3: ((256, 256, 80), (2, 0.5, 1)),
        4: ((256, 256, 80, 80), (2, 0.5, 1)),
    }

    def __init__(self, depth, in_channels, in_height=2, avg_pool=None, out_indices=(1,)):
        super(CNNNet, self).__init__()
        self.out_indices = out_indices
        filters, configs = self.arch_settings[depth]

        self.cnnlayers = []
        for i, out_channels in enumerate(filters):
            if i == configs[2]:
                kernel_size = (in_height, 3)
            else:
                kernel_size = (1, 3)
            conv_layer = self.make_conv_layer(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, pad=configs[0],
                dropout_rate=configs[1])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, conv_layer)
            self.cnnlayers.append(layer_name)
            in_channels = out_channels

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

    def make_conv_layer(self, **kwargs):
        return CNNBasicBlock(**kwargs)

    def forward(self, x):
        for i, layer_name in enumerate(self.cnnlayers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x


@BACKBONES.register_module()
class ResNet(nn.Module):

    def __init__(self, avg_pool=None):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.connect = nn.Conv2d(256, 80, kernel_size=(1, 1), stride=1)
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
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

    def forward(self, iq):
        x1 = self.conv1(iq)
        xc = self.connect(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = torch.add(x3, xc)
        x4 = self.conv4(x3)
        if self.has_avg_pool:
            x4 = self.avg_pool_layer(x4)
        return x4


@BACKBONES.register_module()
class DensNet(nn.Module):

    def __init__(self, avg_pool=None):
        super(DensNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        # self.connect1 = nn.Conv2d(256, 256, kernel_size=(2, 1), stride=1)
        self.connect1 = nn.AvgPool2d((2, 1), stride=1)
        self.connect2 = nn.Conv2d(256, 80, kernel_size=(1, 1), stride=1)
        self.connect3 = nn.Conv2d(256, 80, kernel_size=(2, 1), stride=1)

        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
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

    def forward(self, iq):
        x1 = self.conv1(iq)
        xc1 = self.connect1(x1)
        xc3 = self.connect3(x1)

        x2 = self.conv2(x1)
        x2 = torch.add(x2, xc1)
        xc2 = self.connect2(x2)

        x3 = self.conv3(x2)
        x3 = torch.add(x3, xc2)
        x3 = torch.add(x3, xc3)

        x4 = self.conv4(x3)
        if self.has_avg_pool:
            x4 = self.avg_pool_layer(x4)
        return x4
