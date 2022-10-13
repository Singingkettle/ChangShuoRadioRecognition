import logging

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class MCTNet(nn.Module):
    """
    MCTNet: Modulation Classification Neural Network
    """

    def __init__(self, depth=2, input_size=80, head_size=4, hidden_size=32, dp=0.5):
        super(MCTNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(depth, 256, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, input_size, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
        )
        self.input_size = input_size
        self.tnn1 = nn.TransformerEncoderLayer(input_size, head_size, hidden_size, dropout=dp, batch_first=True)
        self.tnn2 = nn.TransformerEncoderLayer(input_size, head_size, hidden_size, dropout=dp, batch_first=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, iqs):
        x = self.cnn(iqs)
        x = torch.transpose(x, 1, 2)

        x = torch.reshape(x, [-1, 16, self.input_size])
        x = self.tnn1(x)
        x = torch.mean(x, dim=1)
        x = torch.reshape(x, [-1, 8, self.input_size])
        x = self.tnn2(x)

        x = torch.mean(x, dim=1)

        return x
