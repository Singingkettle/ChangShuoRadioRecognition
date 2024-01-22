import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class DAE(BaseBackbone):
    """`DAE <https://ieeexplore.ieee.org/abstract/document/9487492>`_ backbone
    The input for DAE is an N*L*2 frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, depth=2, hidden_size=32, init_cfg=None):
        super(DAE, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=depth, batch_first=True)
        self.time_distributed = nn.Linear(hidden_size, 2)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(16),
                nn.Linear(16, self.num_classes)
            )

    def forward(self, x):

        xc, _ = self.lstm(x)
        xd = self.time_distributed(xc)
        if self.num_classes > 0:
            xc = self.classifier(xc[:, -1, :])

        return (xc, x, xd)
