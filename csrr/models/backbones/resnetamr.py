import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class ResNetAMR(BaseBackbone):
    """`ResNet <https://ieeexplore.ieee.org/abstract/document/8335483>`_
    backbone as packaged by AMR-Benchmark.

    PyTorch port of the AMR-Benchmark Keras reference
    https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML201610a/ResNet/rmlmodels/ResNet.py.
    The model is intentionally renamed to ``ResNetAMR`` to avoid clashing
    with classification ResNet variants in other registries.

    Despite its name this is not a multi-block residual network: it is a
    shallow stack with a single broadcast residual skip from the raw
    ``(1, 2, L)`` input to the second convolutional output (whose channel
    dimension is expanded from 1 to 256 via PyTorch broadcasting). The
    classifier is a hidden ``Dense(128, ReLU)`` projection followed by the
    output Linear; dropout (default ``0.6``) is applied after the conv
    stack and inside the head, matching the Keras reference.

    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
        dropout (float): dropout probability applied after the conv stack
            and in the hidden Dense layer. Defaults to ``0.6``.
    """

    def __init__(self, frame_length=128, num_classes=-1, dropout=0.6, init_cfg=None):
        super(ResNetAMR, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding='same')
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3), padding='same')
        self.conv3 = nn.Conv2d(256, 80, kernel_size=(1, 3), padding='same')
        self.conv4 = nn.Conv2d(80, 80, kernel_size=(1, 3), padding='same')
        self.drop = nn.Dropout(dropout)

        if self.num_classes > 0:
            in_features = frame_length * 80 * 2
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # Broadcast residual: the 1-channel input is added to every output
        # channel of conv2 to match the Keras Add()([input, x]) semantics.
        out = F.relu(identity + out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.drop(out)
        out = torch.flatten(out, 1)
        if self.num_classes > 0:
            out = self.classifier(out)

        return (out,)
