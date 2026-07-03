# Copyright (c) Shuo Chang. All Rights Reserved.
"""Backbones of the JDM framework.

Reference: H. Xing et al., "Joint Signal Detection and Automatic Modulation
Classification via Deep Learning", IEEE TWC, vol. 23, no. 11, 2024
(https://arxiv.org/abs/2405.00736).

Two networks are defined:

- :class:`JDMDetectionBackbone` (paper Fig. 4): five CNN blocks, each with
  three conv layers followed by ReLU + BatchNorm, applied to the ``2 x L``
  frequency-domain representation (FFT amplitude/phase) of a frame. Unlike the
  historical ``DetCNN`` (valid padding, which made the feature-grid geometry
  inconsistent with the anchor stride), convolutions use same padding and the
  temporal resolution is reduced only by the pooling stages, giving an exact
  ``L / 8`` detection grid.
- :class:`JDMClassificationBackbone` (paper Fig. 5): three conv layers
  (256 / 256 / 80 filters, the last one collapsing the I/Q axis), each with
  ReLU + Dropout(0.5), followed by a sum over the time axis producing an
  80-dim feature and a final linear projection to the modulation classes.
"""
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def _conv_block(in_channels, out_channels, num_convs, pool):
    layers = []
    for i in range(num_convs):
        layers += [
            nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        ]
    if pool:
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


@BACKBONES.register_module()
class JDMDetectionBackbone(BaseBackbone):
    """CNN feature extractor of the JDM signal-detection module.

    Args:
        in_channels (int): channels of the input spectrum representation
            (2 = FFT amplitude + phase). Defaults to 2.
        stage_channels (tuple[int]): output channels of the five blocks.
        out_channels (int): channels of the returned feature map (equals
            ``stage_channels[-1]``).

    The input has shape ``(N, in_channels, L)``; the output has shape
    ``(N, stage_channels[-1], L // 8)`` (three pooling stages).
    """

    def __init__(self,
                 in_channels=2,
                 stage_channels=(16, 32, 64, 128, 256),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(stage_channels) == 5, 'the paper uses five CNN blocks'
        self.out_channels = stage_channels[-1]
        # Pool after the first three blocks only: L=1200 -> grid 150
        # (stride 8, matching the anchor stride of the detection head).
        pools = (True, True, True, False, False)
        blocks = []
        channels = in_channels
        for out, pool in zip(stage_channels, pools):
            blocks.append(_conv_block(channels, out, num_convs=3, pool=pool))
            channels = out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


@BACKBONES.register_module()
class JDMClassificationBackbone(BaseBackbone):
    """CNN of the JDM modulation-classification module (paper Fig. 5).

    Input: single-signal baseband I/Q of shape ``(N, 1, 2, frame_length)``.
    Three conv stages (ReLU + Dropout 0.5, valid padding, stride 1 — the
    paper's quoted output size ``80 x 1194`` implies stride 1), the third
    collapsing the I/Q axis; features are then summed over the time axis
    (the paper's "Sum layer") and projected to ``num_classes`` logits.

    Args:
        num_classes (int): number of modulation classes.
        dropout_rate (float): dropout probability. Defaults to 0.5.
    """

    def __init__(self, num_classes, dropout_rate=0.5, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.features = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.features(x)  # (N, 80, 1, L - 6)
        x = x.squeeze(2).sum(dim=-1)  # Sum layer -> (N, 80)
        return (self.classifier(x),)
