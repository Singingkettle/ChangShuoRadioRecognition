import torch
import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class ResCNN(BaseBackbone):

    def __init__(self, avg_pool=None, init_cfg=None):
        super(ResCNN, self).__init__(init_cfg)
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

    def forward(self, iqs):
        x1 = self.conv1(iqs)
        xc = self.connect(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = torch.add(x3, xc)
        x4 = self.conv4(x3)
        if self.has_avg_pool:
            x4 = self.avg_pool_layer(x4)
        return x4


@BACKBONES.register_module()
class AMRBResCNN(BaseBackbone):

    def __init__(self, drop_rate=0.6, init_cfg=None):
        super(AMRBResCNN, self).__init__(init_cfg)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(2, 3), stride=1, padding='same'),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 80, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=(1, 3), stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )

    def forward(self, iqs):
        x1 = self.conv1(iqs)
        x1 = iqs + x1
        x2 = self.conv2(x1)
        return x2
