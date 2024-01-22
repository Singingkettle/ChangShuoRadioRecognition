import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class CNN4(BaseBackbone):
    """`CNN4 <https://ieeexplore.ieee.org/abstract/document/9128408>`_ backbone
    The input for CNN2 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(CNN4, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(256, 128, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(128, 64, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.frame_length//16*2*64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
