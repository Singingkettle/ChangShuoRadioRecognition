import torch
import torch.nn as nn

from .mlnet import SELayer
from ..builder import BACKBONES


@BACKBONES.register_module()
class FSNet(nn.Module):

    def __init__(self, dropout_rate=0.5, in_size=4,
                 channel_mode=False, skip_connection=False,
                 reduction=16, avg_pool=None):
        super(FSNet, self).__init__()
        self.channel_mode = channel_mode
        self.skip_connection = skip_connection
        if self.channel_mode:
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_size, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 100, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            )
        else:
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 100, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.AvgPool2d((in_size, 1)),
            )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127, reduction=reduction)
        else:
            self.se = SELayer(122, reduction=reduction)
            self.has_avg_pool = False

        self.gru = nn.GRU(input_size=100, hidden_size=50,
                          num_layers=2, dropout=dropout_rate,
                          batch_first=True, bidirectional=True)

    def forward(self, iqs, aps):
        if self.channel_mode:
            x = torch.concat([iqs, aps], dim=1)
        else:
            x = torch.concat([iqs, aps], dim=2)
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)

        if self.skip_connection:
            x = x + c_x

        x = torch.sum(x, dim=1)

        return x
