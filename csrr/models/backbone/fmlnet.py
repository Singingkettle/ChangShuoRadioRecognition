import logging

import torch
import torch.nn as nn

from .mlnet import SELayer
from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class FMLNet(nn.Module):

    def __init__(self, dropout_rate=0.5, in_size=4,
                 channel_mode=False, skip_connection=False,
                 reduction=16, avg_pool=None):
        super(FMLNet, self).__init__()
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

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GRU):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(3, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(3, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)

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
