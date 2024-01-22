import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


class _Pre_Block(nn.Module):
    def __init__(self):
        super(_Pre_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)

        return x


class _M_BlockA(nn.Module):
    def __init__(self, ):
        super(_M_BlockA, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(64, 32, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2)),
            nn.ReLU()
        )

    def forward(self, x):
        sx = self.skip(x)
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = sx + x

        return x


class _M_BlockB(nn.Module):
    def __init__(self, has_pooling=False):
        super(_M_BlockB, self).__init__()
        self.has_pooling = has_pooling
        self.conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
        )
        if has_pooling:
            self.skip = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
            )
            self.conv2 = nn.Sequential(
                nn.ZeroPad2d(padding=(1, 1, 0, 0)),
                nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2)),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2)),
                nn.ReLU()
            )
        else:
            self.skip = nn.Identity()
            self.conv1 = nn.Sequential(
                nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.ZeroPad2d(padding=(1, 1, 0, 0)),
                nn.Conv2d(32, 48, kernel_size=(1, 3)),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 1)),
                nn.ReLU()
            )

    def forward(self, x):
        sx = self.skip(x)
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = sx + x

        return x


class _M_BlockC(nn.Module):
    def __init__(self):
        super(_M_BlockC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
        )
        self.skip = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(32, 96, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        sx = self.skip(x)
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([sx, x1, x2, x3], dim=1)

        return x


@BACKBONES.register_module()
class MCNet(BaseBackbone):
    """`MCNet <https://ieeexplore.ieee.org/abstract/document/8963964>`_ backbone
    The input for MCNet is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(MCNet, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.fea = nn.Sequential(
            nn.ZeroPad2d(padding=(3, 3, 1, 1)),
            nn.Conv2d(1, 64, kernel_size=(3, 7), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            _Pre_Block(),
            _M_BlockA(),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockC(),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, frame_length // 128)),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(384, num_classes),
            )

    def forward(self, x):
        x = self.fea(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
