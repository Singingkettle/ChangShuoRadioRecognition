import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class DensCNN(BaseBackbone):
    """`DensCNN <https://ieeexplore.ieee.org/abstract/document/9128408>`_ backbone
    The input for DensCNN is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(DensCNN, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding='same'),
            nn.ReLU(inplace=True),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(2, 3), padding='same'),
        )
        self.cnn3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 80, kernel_size=(1, 3), padding='same'),
        )
        self.cnn4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(592, 80, kernel_size=(1, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Flatten()
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.frame_length*80*2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.6),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x1)
        x2 = torch.concatenate((x1, x2), dim=1)
        x3 = self.cnn3(x2)
        x3 = torch.concatenate((x2, x3), dim=1)
        x4 = self.cnn4(x3)

        if self.num_classes > 0:
            x = self.classifier(x4)

        return (x,)
