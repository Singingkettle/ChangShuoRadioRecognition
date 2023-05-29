import torch
import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES
from ...runner import Sequential


@BACKBONES.register_module()
class FMLNet(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None,
                 is_freeze=False, groups=(2, 16, 4), stride=1, tnn='t', merge='sum', channel_mode=True):
        super(FMLNet, self).__init__(init_cfg)
        if channel_mode:
            self.cnn = Sequential(
                nn.Conv1d(depth, hidden_size, kernel_size=3, groups=groups[0], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, groups=groups[1], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, input_size, kernel_size=3, groups=groups[2], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )
        else:
            self.cnn = Sequential(
                nn.Conv2d(depth//2, hidden_size, kernel_size=(1, 3), groups=groups[0], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 3), groups=groups[1], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv2d(hidden_size, input_size, kernel_size=(2, 3), groups=groups[2], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )

        self.merge = merge
        self.is_rnn = False if tnn == 't' else True
        if tnn == 't':
            self.tnn = nn.TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)
        elif tnn == 'l':
            self.tnn = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
            self.merge = 'last'
        elif tnn == 'g':
            self.tnn = nn.LSTM(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
            self.merge = 'last'
        elif tnn == 'r':
            self.tnn = nn.RNN(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
            self.merge = 'last'

        self.is_freeze = is_freeze

    def _freeze_layers(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)
        if self.is_rnn:
            fea, _ = self.tnn(c_fea)
        else:
            fea = self.tnn(c_fea)
        if self.merge == 'sum':
            return torch.sum(fea, dim=1)  # this is best
        elif self.merge == 'mean':
            return torch.mean(fea, dim=1)
        elif self.merge == 'max':  # this is best
            return torch.max(fea, dim=1)[0]
        elif self.merge == 'min':
            return torch.min(fea, dim=1)[0]
        elif self.merge == 'logsumexp':
            return torch.logsumexp(fea, dim=1)
        elif self.merge == 'median':
            return torch.median(fea, dim=1)[0]
        elif self.merge == 'std':
            return torch.std(fea, dim=1)
        elif self.merge == 'quantile':
            return torch.quantile(fea, 0.6, dim=1, interpolation='higher')
        elif self.merge == 'last':
            return fea[:, -1, :]
        else:
            raise NotImplementedError(f'There is no torch.{self.merge} operation!')

    def train(self, mode=True):
        super(FMLNet, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()
