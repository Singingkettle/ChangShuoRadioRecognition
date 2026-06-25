import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class MCLDNN(BaseBackbone):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=-1, init_cfg=None):
        # The AMR-Benchmark Keras reference initialises every Conv/Dense with
        # `glorot_uniform`. Without an explicit init the PyTorch default
        # (Kaiming) leaves the deep multi-branch + SELU head unable to escape
        # the random-chance solution (loss frozen at ln(num_classes)); match
        # Keras by defaulting to Xavier-uniform on conv and linear layers.
        if init_cfg is None:
            init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Conv1d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform'),
                # Keras CuDNNLSTM uses glorot(input) + orthogonal(recurrent);
                # rnn_init reproduces that (and sets forget-gate bias = 1),
                # which converges to a markedly better optimum than the
                # PyTorch default uniform LSTM init.
                dict(type='LSTM', layer='LSTM', gain=1),
            ]
        super(MCLDNN, self).__init__(init_cfg=init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        # x5 is NCHW [B, 100, 1, L-4]. The Keras reference reshapes the NHWC
        # tensor [B, 1, L-4, 100] to (L-4, 100), i.e. time = the L-4 conv
        # positions, features = the 100 channels. A plain torch.reshape here
        # interleaves channels into the time axis and scrambles the sequence
        # fed to the LSTM, which freezes training at random chance. Squeeze the
        # singleton height and permute so time/feature axes match Keras.
        x = x5.squeeze(2).permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)
