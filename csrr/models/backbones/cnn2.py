import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class CNN2(BaseBackbone):
    """`CNN2 <https://link.springer.com/chapter/10.1007/978-3-319-44188-7_16>`_ backbone
    The input for CNN2 is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(CNN2, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(50, 50, kernel_size=(2, 8), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.frame_length-7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
