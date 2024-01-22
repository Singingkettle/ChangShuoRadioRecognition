import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class CNN1DPF(BaseBackbone):
    """`CNN1DPF <https://lirias.kuleuven.be/retrieve/546033>`_ backbone
    The input for CNN1DPF is a 2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        super(CNN1DPF, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.a_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.p_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        def _calculate_in_features(x):
            x = x - 8
            for i in range(5):
                x = x - 2
                x = x // 2

            return x

        self._in_features = _calculate_in_features(self.frame_length) * 64
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self._in_features, 128),
                nn.SELU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.SELU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x_a = self.a_features(x[:, :, 0, :])
        x_p = self.p_features(x[:, :, 1, :])
        x = torch.cat([x_a, x_p], dim=1)
        x = self.merge(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
