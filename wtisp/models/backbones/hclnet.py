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
            nn.Conv2d(2, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 80, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.gru1 = nn.GRU(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.gru2 = nn.GRU(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)

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

        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)

        return dict(cnn=torch.mean(c_fea, dim=1), gru1=torch.mean(g_fea1, dim=1), gru2=torch.mean(g_fea2, dim=1))
