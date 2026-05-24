import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class ResNetAMR(BaseBackbone):
    """AMR-Benchmark style residual CNN for automatic modulation recognition.

    This module mirrors the lightweight Keras ResNet used in AMR-Benchmark for
    RadioML2018.01A.  The input is a ``1 x 2 x L`` IQ frame.  The residual
    addition intentionally relies on broadcasting from the single input channel
    to the 256-channel convolution output, matching TensorFlow's elementwise
    add behavior in the reference implementation.
    """

    def __init__(self,
                 frame_length=1024,
                 num_classes=-1,
                 dropout=0.6,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding='same'),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3), padding='same')
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 80, kernel_size=(1, 3), padding='same'),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(1, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.frame_length * 2 * 80, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, self.num_classes),
            )

    def _format_input(self, x):
        """Convert CSRR IQ layouts to ``N x 1 x 2 x L`` for 2-D convs."""
        if x.dim() == 3:
            # Common RCPS pipeline layout: N x L x 2.
            if x.size(-1) == 2:
                x = x.permute(0, 2, 1).unsqueeze(1)
            # Classic AMR layout: N x 2 x L.
            elif x.size(1) == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(
                    'ResNetAMR expects 3-D IQ input as N x L x 2 or N x 2 x L, '
                    f'but got {tuple(x.shape)}.')
        elif x.dim() == 4:
            # Already N x 1 x 2 x L.
            if x.size(1) == 1 and x.size(2) == 2:
                pass
            # Keras-like N x 2 x L x 1.
            elif x.size(1) == 2 and x.size(-1) == 1:
                x = x.permute(0, 3, 1, 2)
            # Occasionally N x L x 2 x 1.
            elif x.size(2) == 2 and x.size(-1) == 1:
                x = x.permute(0, 3, 2, 1)
            else:
                raise ValueError(
                    'ResNetAMR expects 4-D IQ input as N x 1 x 2 x L, '
                    f'N x 2 x L x 1, or N x L x 2 x 1, but got {tuple(x.shape)}.')
        else:
            raise ValueError(f'ResNetAMR expects a 3-D or 4-D tensor, got {tuple(x.shape)}.')
        return x.contiguous()

    def forward(self, x):
        x = self._format_input(x)
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.relu(x + residual)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.num_classes > 0:
            x = self.classifier(x)
        return (x,)
