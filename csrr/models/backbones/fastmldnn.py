import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
from ..builder import BACKBONES


def unit_vector(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)

    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_norm = unit_vector(a, eps)
    b_norm = unit_vector(b, eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@BACKBONES.register_module()
class FastMLDNN(BaseBackbone):
    """`FastMLDNN <https://ieeexplore.ieee.org/abstract/document/10239249/>`_ backbone
    The input for FastMLDNN is two mod (iq+ap [4*1*L]) for the smae frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, depth=4, input_size=100, output_size=288, hidden_size=256, dp=0.5,
                 init_cfg=None, groups=(2, 16, 4), stride=1, tnn='t', merge='sum', scale=1, channel_mode=True):
        super(FastMLDNN, self).__init__(init_cfg)
        self.num_classes = num_classes
        if channel_mode:
            self.cnn = nn.Sequential(
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
            self.cnn = nn.Sequential(
                nn.Conv2d(depth // 2, hidden_size, kernel_size=(1, 3), groups=groups[0], stride=stride),
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

        self.scale = scale
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(output_size, self.num_classes, bias=False),
            )

    def forward(self, x):
        x = torch.concat([x['ap'], x['iq']], dim=1)
        x = self.cnn(x)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        if self.is_rnn:
            x, _ = self.tnn(x)
        else:
            x = self.tnn(x)
        if self.merge == 'sum':
            x = torch.sum(x, dim=1)  # this is best
        elif self.merge == 'mean':
            x = torch.mean(x, dim=1) * self.scale
        elif self.merge == 'max':  # this is best
            x = torch.max(x, dim=1)[0]
        elif self.merge == 'min':
            x = torch.min(x, dim=1)[0]
        elif self.merge == 'logsumexp':
            x = torch.logsumexp(x, dim=1)
        elif self.merge == 'median':
            x = torch.median(x, dim=1)[0]
        elif self.merge == 'std':
            x = torch.std(x, dim=1)
        elif self.merge == 'quantile':
            x = torch.quantile(x, 0.6, dim=1, interpolation='higher')
        elif self.merge == 'last':
            x = x[:, -1, :]
        else:
            raise NotImplementedError(f'There is no torch.{self.merge} operation!')

        if self.num_classes > 0:
            x = self.classifier(x)
            p = sim_matrix(self.classifier[3].weight, self.classifier[3].weight)
            if self.training:
                return (x, p, )

        return (x,)
