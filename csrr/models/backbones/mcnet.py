import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def _conv_same(in_channels, out_channels, kernel_size, stride=(1, 1)):
    """Wrap a Conv2d that emulates Keras ``padding='same'``.

    PyTorch's built-in ``padding='same'`` only supports stride 1, so for
    strided convolutions we replicate Keras behaviour with explicit
    symmetric padding of ``((k - 1) // 2)`` along each axis. For the even/
    odd input sizes used by the supported datasets (L in {128, 1024},
    H = 2) the resulting output spatial size matches ``ceil(input / stride)``.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    if sh == 1 and sw == 1:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=(sh, sw), padding=(pad_h, pad_w))


def _pool_same(kernel_size, stride, pool_cls):
    kh, kw = kernel_size
    sh, sw = stride
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    return pool_cls(kernel_size=(kh, kw), stride=(sh, sw), padding=(pad_h, pad_w))


class _MBlockDown4(nn.Module):
    """First M-block: pre-pool plus strided convs (Keras ``Mblockp1``).

    Downsamples the time axis by a factor of 4 relative to its input.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pre_pool = _pool_same((1, 3), (1, 2), nn.MaxPool2d)
        self.conv1 = _conv_same(in_channels, 32, (1, 1))
        self.conv2 = _conv_same(32, 48, (3, 1))
        self.pool2 = _pool_same((1, 3), (1, 2), nn.MaxPool2d)
        self.conv3 = _conv_same(32, 48, (1, 3), stride=(1, 2))
        self.conv4 = _conv_same(32, 32, (1, 1), stride=(1, 2))

    def forward(self, x):
        x = self.pre_pool(x)
        x = F.relu(self.conv1(x))
        x31 = self.pool2(F.relu(self.conv2(x)))
        x32 = F.relu(self.conv3(x))
        x33 = F.relu(self.conv4(x))
        x31 = torch.cat([x31, x32], dim=1)
        return torch.cat([x33, x31], dim=1)


class _MBlockDown2(nn.Module):
    """Downsampling M-block without pre-pool (Keras ``Mblockp3``/``Mblockp5``).

    Downsamples the time axis by a factor of 2 relative to its input.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = _conv_same(in_channels, 32, (1, 1))
        self.conv2 = _conv_same(32, 48, (3, 1))
        self.pool2 = _pool_same((1, 3), (1, 2), nn.MaxPool2d)
        self.conv3 = _conv_same(32, 48, (1, 3), stride=(1, 2))
        self.conv4 = _conv_same(32, 32, (1, 1), stride=(1, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x51 = self.pool2(F.relu(self.conv2(x)))
        x52 = F.relu(self.conv3(x))
        x53 = F.relu(self.conv4(x))
        x51 = torch.cat([x51, x52], dim=1)
        return torch.cat([x53, x51], dim=1)


class _MBlockKeep(nn.Module):
    """M-block that keeps spatial dimensions (Keras ``Mblock2``/``Mblockp4``).

    Args:
        in_channels (int): channels of the input feature map.
        out_widths (tuple[int, int, int]): output channels of the three
            parallel paths ``(conv_h, conv_w, conv_1x1)``. Defaults to the
            standard ``(48, 48, 32)`` and is overridden by ``Mblockp6``.
    """

    def __init__(self, in_channels, out_widths=(48, 48, 32)):
        super().__init__()
        ow_h, ow_w, ow_p = out_widths
        self.conv1 = _conv_same(in_channels, 32, (1, 1))
        self.conv2 = _conv_same(32, ow_h, (3, 1))
        self.conv3 = _conv_same(32, ow_w, (1, 3))
        self.conv4 = _conv_same(32, ow_p, (1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x41 = F.relu(self.conv2(x))
        x42 = F.relu(self.conv3(x))
        x43 = F.relu(self.conv4(x))
        x41 = torch.cat([x41, x42], dim=1)
        return torch.cat([x43, x41], dim=1)


@BACKBONES.register_module()
class MCNet(BaseBackbone):
    """`MCNet <https://ieeexplore.ieee.org/abstract/document/8963964>`_ backbone.

    PyTorch port of the AMR-Benchmark Keras reference
    https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML201610a/MCNET/rmlmodels/MCNET.py.

    Stem + pre-block + six asymmetric M-blocks operating on ``(B, 1, 2, L)``
    inputs. Downsampling alternates between strided convolutions and
    ``MaxPool`` layers so that the time axis is progressively halved while
    the I/Q rows are kept intact. The final ``AveragePool`` kernel switches
    between ``(2, 1)`` for ``frame_length=128`` (RML2016.10A/B) and
    ``(2, 8)`` for ``frame_length=1024`` (RML2018.01A / HisarMod) to
    collapse the spatial map to a single feature vector regardless of
    dataset.

    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
        dropout (float): dropout probability applied before the classifier.
            Defaults to ``0.5`` to match the Keras reference.
    """

    def __init__(self, frame_length=128, num_classes=-1, dropout=0.5, init_cfg=None):
        super(MCNet, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Stem.
        self.stem_conv = _conv_same(1, 64, (3, 7), stride=(1, 2))
        self.stem_pool = _pool_same((1, 3), (1, 2), nn.MaxPool2d)

        # Pre-block (concat of an avg-pool branch and a strided conv branch).
        self.pre_conv_a = _conv_same(64, 32, (3, 1))
        self.pre_pool_a = _pool_same((1, 3), (1, 2), nn.AvgPool2d)
        self.pre_conv_b = _conv_same(64, 32, (1, 3), stride=(1, 2))

        # Skip path consumed by the first residual addition.
        self.skip_conv = _conv_same(64, 128, (1, 1), stride=(1, 2))
        self.skip_pool = _pool_same((1, 3), (1, 2), nn.MaxPool2d)

        # M-blocks.
        self.mblock1 = _MBlockDown4(64)            # Mblockp1
        self.mblock2 = _MBlockKeep(128)            # Mblock2
        self.mblock3 = _MBlockDown2(128)           # Mblockp3
        self.add3_pool = _pool_same((2, 2), (1, 2), nn.MaxPool2d)
        self.mblock4 = _MBlockKeep(128)            # Mblockp4
        self.mblock5 = _MBlockDown2(128)           # Mblockp5
        self.add5_pool = _pool_same((2, 2), (1, 2), nn.MaxPool2d)
        # Mblockp6 widens the per-branch outputs: (96, 96, 64) -> 256 channels.
        self.mblock6 = _MBlockKeep(128, out_widths=(96, 96, 64))

        if frame_length == 128:
            final_pool_w = 1
        elif frame_length == 1024:
            final_pool_w = 8
        else:
            # For arbitrary lengths, collapse whatever residual width remains.
            # Total downsampling from input to add5 is 64x (stem 2, pool 2,
            # pre-block 2, mblock1 4, mblock3 2, mblock5 2), so the residual
            # width is roughly frame_length // 64. For 128 -> 2 -> pool 2 = 1;
            # for 1024 -> 16 -> pool 2 = 8.
            final_pool_w = max(1, frame_length // 128)
        self.final_pool = nn.AvgPool2d(kernel_size=(2, final_pool_w))
        self.drop = nn.Dropout(dropout)

        if self.num_classes > 0:
            # After concat x888 (256 ch) + add5 (128 ch) and the final
            # 2 x final_pool_w average pool, the feature map collapses to
            # (B, 384, 1, 1).
            self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x1 = F.relu(self.stem_conv(x))
        x1 = self.stem_pool(x1)

        x2 = self.pre_pool_a(F.relu(self.pre_conv_a(x1)))
        x22 = F.relu(self.pre_conv_b(x1))
        x222 = torch.cat([x2, x22], dim=1)

        xx1 = F.relu(self.skip_conv(x222))
        xx1 = self.skip_pool(xx1)

        x333 = self.mblock1(x222)
        add1 = x333 + xx1

        x444 = self.mblock2(add1)
        add2 = x444 + add1

        x555 = self.mblock3(add2)
        ad3 = self.add3_pool(add2)
        add3 = x555 + ad3

        x666 = self.mblock4(add3)
        add4 = x666 + add3

        x777 = self.mblock5(add4)
        ad5 = self.add5_pool(add4)
        add5 = x777 + ad5

        x888 = self.mblock6(add5)
        x_con = torch.cat([x888, add5], dim=1)
        xout = self.final_pool(x_con)
        xout = self.drop(xout)
        xout = torch.flatten(xout, 1)

        if self.num_classes > 0:
            xout = self.classifier(xout)

        return (xout,)
