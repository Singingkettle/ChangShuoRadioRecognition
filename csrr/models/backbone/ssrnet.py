import math

import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class SSRNet(BaseBackbone):
    """
    Refer the official code: https://github.com/YihongDong/Semi-supervised-Signal-Recognition/blob/main/SSRCNN.py
    The CNN network defined here is inherited from official, and the last two fc layers are placed in the
    ssrcnn_head.py, which is redesigned in the csrr manner. Of course, the core idea is not changed. (Some of
    unused layers are deleted.)
    """

    def __init__(self, num_channels=2):
        super(SSRNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=[1, 5], stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])

        self.conv2 = nn.Conv2d(32, 32, kernel_size=[1, 3], stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=[1, 3], stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, iqs):
        x = self.conv1(iqs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x
