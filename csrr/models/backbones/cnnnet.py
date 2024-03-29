import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


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
class CNNNet(BaseBackbone):
    arch_settings = {
        2: ((256, 80), (2, 0.5, 1)),
        3: ((256, 256, 80), (2, 0.5, 1)),
        4: ((256, 256, 80, 80), (2, 0.5, 1)),
    }

    def __init__(self, depth, in_channels, in_height=2, avg_pool=None, out_indices=(1,), init_cfg=None):
        super(CNNNet, self).__init__(init_cfg)
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

    def make_conv_layer(self, **kwargs):
        return CNNBasicBlock(**kwargs)

    def forward(self, iqs=None, aps=None):
        if iqs is None:
            x = aps
        elif aps is None:
            x = iqs
        else:
            raise ValueError('Either input iq sequence or ap sequence!')
        for i, layer_name in enumerate(self.cnnlayers):
            layer = getattr(self, layer_name)
            x = layer(x)

        return x


@BACKBONES.register_module()
class DetCNN(BaseBackbone):

    def __init__(self, init_cfg=None):
        super(DetCNN, self).__init__(init_cfg)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=(1, 6), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(16, 16, kernel_size=(1, 5), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(16, 16, kernel_size=(1, 4), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(32, 32, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(32, 32, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(64, 64, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(128, 128, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(128, 128, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        # )
        #
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(1, 3), padding=0),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, iqs):
        x1 = self.conv1(iqs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        # x5 = self.conv5(x4)

        return x4
