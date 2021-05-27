import logging

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class HCLNetV1(nn.Module):

    def __init__(self):
        super(HCLNetV1, self).__init__()
        # For low snr
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.gru1 = nn.GRU(input_size=80,
                           hidden_size=100, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(input_size=100,
                           hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.gru3 = nn.GRU(input_size=50,
                           hidden_size=50, batch_first=True)

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
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            for ih in param.chunk(4, 0):
                                nn.init.xavier_uniform_(ih)
                        elif 'weight_hh' in name:
                            for hh in param.chunk(4, 0):
                                nn.init.orthogonal_(hh)
                        elif 'bias_ih' in name:
                            nn.init.zeros_(param)
                        elif 'bias_hh' in name:
                            nn.init.zeros_(param)
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

    def forward(self, x):
        c_fea = self.conv_net(x)

        x = torch.squeeze(c_fea, dim=2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru1(x)
        fea1 = self.dropout1(x)

        x, _ = self.gru2(fea1)
        fea2 = self.dropout2(x)

        fea3, _ = self.gru3(fea2)

        return (c_fea, fea1[:, -1, :], fea2[:, -1, :], fea3[:, -1, :])
