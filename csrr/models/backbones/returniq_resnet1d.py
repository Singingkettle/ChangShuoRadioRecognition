# Copyright (c) Shuo Chang and contributors. Licensed under the Apache-2.0 License.
"""1-D residual backbone for the return-to-IQ recognizer of the paper
"Detection Is Easy, Recognition Is Hard".

The recognizer classifies a single narrowband signal from its channelized
baseband IQ (a ``[B, 2, L]`` real tensor holding I and Q). It is a small 1-D
ResNet: a wide stem, three residual stages with progressive channel widening
and temporal pooling, and a global-average-pooled feature. The paired
:class:`~csrr.models.heads.HierarchicalAMCHead` turns that feature into the
single-vs-multi-carrier routing and the fine-grained class.

An optional differential-phase input representation is supported: ``diff`` maps
a constant carrier offset to a constant rotation of ``z[n] = x[n] * conj(x[n-1])``
(a probe of CFO invariance), and ``iqdiff`` concatenates it with raw I/Q.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def to_input_rep(x: torch.Tensor, rep: str) -> torch.Tensor:
    """Map a ``[B, 2, L]`` I/Q tensor to the requested input representation."""
    if rep == 'iq':
        return x
    c = torch.complex(x[:, 0], x[:, 1])
    z = c[:, 1:] * torch.conj(c[:, :-1])
    z = F.pad(z, (1, 0))
    zr, zi = z.real.unsqueeze(1), z.imag.unsqueeze(1)
    if rep == 'diff':
        return torch.cat([zr, zi], dim=1)
    if rep == 'iqdiff':
        return torch.cat([x, zr, zi], dim=1)
    raise ValueError(f'unknown input-rep {rep!r}')


_REP2CH = {'iq': 2, 'diff': 2, 'iqdiff': 4}


class _ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.b1 = nn.BatchNorm1d(channels)
        self.c2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.b2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + x)


@BACKBONES.register_module()
class ReturnIQResNet1D(BaseBackbone):
    """1-D ResNet over channelized baseband IQ.

    Args:
        input_rep (str): One of ``'iq'`` (2ch I/Q), ``'diff'`` (2ch differential
            phase) or ``'iqdiff'`` (4ch I/Q + differential). Defaults to ``'iq'``.
        stem_channels (int): Width of the stem conv. Defaults to 64.
        stage_channels (tuple[int]): Channels of the three residual stages.
            Defaults to ``(64, 128, 256)``.
        blocks_per_stage (int): Residual blocks per stage. Defaults to 2.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """

    def __init__(self,
                 input_rep: str = 'iq',
                 stem_channels: int = 64,
                 stage_channels=(64, 128, 256),
                 blocks_per_stage: int = 2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert input_rep in _REP2CH, f'input_rep must be one of {list(_REP2CH)}'
        self.input_rep = input_rep
        self.feat_dim = stage_channels[-1]

        in_ch = _REP2CH[input_rep]
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_channels, 7, padding=3),
            nn.BatchNorm1d(stem_channels), nn.ReLU(inplace=True))

        layers, cin = [], stem_channels
        for c in stage_channels:
            if c != cin:
                layers += [nn.Conv1d(cin, c, 1), nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
                cin = c
            for _ in range(blocks_per_stage):
                layers.append(_ResBlock1d(c))
            layers.append(nn.MaxPool1d(2))
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2, L] real I/Q -> representation -> [B, feat_dim]
        x = to_input_rep(x, self.input_rep)
        feat = self.body(self.stem(x))
        return (feat,)
