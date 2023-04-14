import torch
import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES
from ...runner import Sequential


@BACKBONES.register_module()
class FMLNet(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None,
                 is_freeze=False, groups=(2, 16, 4), stride=1, merge='sum'):
        super(FMLNet, self).__init__(init_cfg)
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

        self.merge = merge
        self.tnn = nn.TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)

        self.is_freeze = is_freeze

    def _freeze_layers(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.transpose(c_fea, 1, 2)
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
        else:
            raise NotImplementedError(f'There is no torch.{self.merge} operation!')

    def train(self, mode=True):
        super(FMLNet, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()



