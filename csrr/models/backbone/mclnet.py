import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone
from ..builder import BACKBONES



@BACKBONES.register_module()
class MCLNet(BaseBackbone):

    def __init__(self, dropout_rate=0.5, avg_pool=None, use_GRU=False, is_BIGRU=False, fusion_method='',
                 gradient_truncation=False, init_cfg=None):
        super(MCLNet, self).__init__(init_cfg)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 50, kernel_size=8, padding=(7, 0)),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 50, kernel_size=8, padding=(7, 0)),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

    def forward(self, iqs):
        x1 = self.conv1(iqs)
        x2 = self.conv2(iqs[:, :, 0, :])
        x3 = self.conv3(iqs[:, :, 1, :])
        x = torch.concatenate([x2, x3], dim=1)


