import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class CNN2(BaseBackbone):
    """`CNN1 <https://link.springer.com/chapter/10.1007/978-3-319-44188-7_16>`_ backbone
    The input for CNN1 is a 2*L frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, init_cfg=None):
        super(CNN2, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(50, 50, kernel_size=(2, 8), padding='valid'),
            nn.ReLU(inplace=True),

        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(6050, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
