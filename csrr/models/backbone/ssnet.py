import logging

import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


@BACKBONES.register_module()
class SSNet(nn.Module):

    def __init__(self) -> None:
        super(SSNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=7, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, kernel_size=5, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, kernel_size=5, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, iqs):
        N, S, W, L = iqs.shape
        iqs = iqs.view(N*S, W, L)
        x = self.block1(iqs)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 192)

        return x
