from typing import Union, List, Dict, cast

import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


@BACKBONES.register_module()
class VGGNet(BaseBackbone):

    def __init__(self, net_type='D', batch_norm=False, init_cfg=None):
        super(VGGNet, self).__init__(init_cfg)
        layers: List[nn.Module] = []
        in_channels = 1
        cfg = cfgs[net_type]
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, cos):
        x = self.features(cos)
        x = self.avg_pool(x)

        return x
