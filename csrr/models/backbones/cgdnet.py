import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev + 1
            return x * epsilon
        else:
            return x


@BACKBONES.register_module()
class CGDNet(BaseBackbone):
    """`CGDNet <https://ieeexplore.ieee.org/abstract/document/9349627>`_ backbone
    The input for CNN1 is a 2*L frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, init_cfg=None):
        super(CGDNet, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 6)),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            GaussianDropout(0.2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 6)),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            GaussianDropout(0.2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 6)),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            GaussianDropout(0.2),
        )
        self.gru = nn.GRU(input_size=472, hidden_size=50, batch_first=True)
        self.dp = GaussianDropout(0.2)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(50, 256),
                nn.ReLU(inplace=True),
                GaussianDropout(0.2),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):

        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x3 = self.cnn3(x2)
        x4 = torch.cat([x1, x3], dim=3)
        x4 = x4.view(-1, 50, 472)
        _, x5 = self.gru(x4)
        x5 = torch.squeeze(x5)
        x5 = self.dp(x5)
        if self.num_classes > 0:
            x = self.classifier(x5)

        return (x,)
