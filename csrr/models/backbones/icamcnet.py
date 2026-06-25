import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class ICAMCNet(BaseBackbone):
    """`IC-AMCNet <https://ieeexplore.ieee.org/abstract/document/8977561>`_ backbone.

    PyTorch port of the AMR-Benchmark Keras reference
    https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML201610a/IC-AMCNet/rmlmodels/ICAMC.py.
    Four ``(1, k)`` convolutional blocks (64, 64, 128, 128 filters) with
    a single ``MaxPool(2, 2)`` after the first block, a hidden
    ``Dense(128, ReLU)`` layer and an additive Gaussian noise layer
    inserted between the hidden Dense and the classifier (training only)
    to improve generalisation under low-SNR signals.

    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
        dropout (float): dropout probability applied between convolutional
            stages and inside the classifier head. Defaults to ``0.4`` to
            match the Keras reference.
        noise_std (float): standard deviation of the Gaussian noise applied
            to the hidden representation during training. Defaults to ``1.0``
            to match the Keras ``GaussianNoise(1)`` layer.
    """

    def __init__(self,
                 frame_length=128,
                 num_classes=-1,
                 dropout=0.4,
                 noise_std=1.0,
                 init_cfg=None):
        super(ICAMCNet, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.noise_std = noise_std

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1)),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if self.num_classes > 0:
            in_features = 128 * 1 * (frame_length // 2)
            self.classifier_pre = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier_pre(x)
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x = self.classifier_out(x)

        return (x,)
