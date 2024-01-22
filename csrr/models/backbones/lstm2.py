import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class LSTM2(BaseBackbone):
    """`LSTM2 <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8357902>`_ backbone
    The input for LSTM is an N*L*2 frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, depth=2, hidden_size=128, init_cfg=None):
        super(LSTM2, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.features = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=depth, batch_first=True)
        if self.num_classes > 0:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x, _ = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)
