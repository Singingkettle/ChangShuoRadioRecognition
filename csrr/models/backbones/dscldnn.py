import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


class _CNNBlock(nn.Module):
    def __init__(self, ):
        super(_CNNBlock, self).__init__()
        self.cnn_net = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 256, kernel_size=(2, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 80, kernel_size=(1, 3), padding='valid'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.cnn_net(x)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)
        return x


class _LSTMBlock(nn.Module):
    def __init__(self):
        super(_LSTMBlock, self).__init__()
        self.lstm1 = nn.LSTM(input_size=80, hidden_size=100, batch_first=True)
        self.dp1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dp1(x)
        x, _ = self.lstm2(x)
        x = self.dp2(x)

        return x[:, -1, :]


@BACKBONES.register_module()
class DSCLDNN(BaseBackbone):
    """`DSCLDNN <https://ieeexplore.ieee.org/document/9220797>`_ backbone
        The input for CNN1 is two 1*2*L frames.
        In addition, the author said that an 80 × 134 feature map can be obtained after three convolutional layers.
        Therefore, the author may add a padding layer with (2, 2, 0, 0) before each convolutional layer.
        Args:
            num_classes (int): number of classes for classification.
                The default value is -1, which uses the backbone as
                a feature extractor without the top classifier.
        """

    def __init__(self, num_classes=-1, init_cfg=None):
        super(DSCLDNN, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.ap_net = nn.Sequential(
            _CNNBlock(),
            _LSTMBlock()
        )
        self.iq_net = nn.Sequential(
            _CNNBlock(),
            _LSTMBlock()
        )

        if self.num_classes > 0:
            self.classifier = nn.Linear(2500, num_classes)

    def _bmm_fusion(self, iq, ap):
        iq = torch.unsqueeze(iq, 2)
        ap = torch.unsqueeze(ap, 1)  # (N, 1, C)

        # (N, C, C) Please refer the https://pytorch.org/docs/0.2.0/torch.html#torch.bmm for the detail of torch.bmm
        x = torch.bmm(iq, ap)
        x = torch.reshape(x, (-1, 2500))

        return x

    def forward(self, x):
        ap = self.ap_net(x['ap'])
        iq = self.iq_net(x['iq'])
        x = self._bmm_fusion(iq, ap)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)
