import torch
import torch.nn as nn

from .base import BaseBackbone
from ..builder import BACKBONES


class _CNN(nn.Module):

    def __init__(self, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, padding=(0, 0)):
        super(_CNN, self).__init__()
        if has_stride:
            stride = 2
        else:
            stride = 1
        self.conv_net = nn.Sequential(
            nn.Conv2d(depth, 256, kernel_size=(1, 3), stride=(1, stride), padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, stride), padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv2d(256, input_size, kernel_size=(1, 3), stride=(1, stride), padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
        else:
            self.has_avg_pool = False

    def forward(self, x):
        c_fea = self.conv_net(x)
        if self.has_avg_pool:
            c_fea = self.avg_pool_layer(c_fea)

        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)

        return c_fea


@BACKBONES.register_module()
class HCGNet(BaseBackbone):

    def __init__(self, heads, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNet, self).__init__(init_cfg)
        self.heads = heads
        if len(heads) < 2:
            assert ValueError('The CHGNet must have multi heads!')

        self.cnn = _CNN(depth, input_size, avg_pool, has_stride, dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        outs = dict()
        c_fea = self.cnn(iqs)
        if 'CNN' in self.heads:
            outs['CNN'] = torch.sum(c_fea, dim=1)
        g_fea1, _ = self.gru1(c_fea)
        if 'BiGRU1' in self.heads:
            outs['BiGRU1'] = torch.sum(g_fea1, dim=1)
        if 'BiGRU2' in self.heads:
            fea = self.dropout(g_fea1)
            g_fea2, _ = self.gru2(fea)
            outs['BiGRU2'] = torch.sum(g_fea2, dim=1)

        return outs


@BACKBONES.register_module()
class HCGNetCNN(BaseBackbone):

    def __init__(self, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetCNN, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)

        return torch.sum(c_fea, dim=1)


@BACKBONES.register_module()
class HCGNetGRU1(BaseBackbone):

    def __init__(self, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetGRU1, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)

        return torch.sum(g_fea1, dim=1)


@BACKBONES.register_module()
class HCGNetGRU2(BaseBackbone):

    def __init__(self, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetGRU2, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)

        return torch.sum(g_fea2, dim=1)


@BACKBONES.register_module()
class HCGNetCG1(BaseBackbone):

    def __init__(self, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetCG1, self).__init__(init_cfg)
        self.cnn = _CNN(input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)

        return dict(cnn=torch.sum(c_fea, dim=1), gru1=torch.sum(g_fea1, dim=1))


@BACKBONES.register_module()
class HCGNetCG2(BaseBackbone):
    def __init__(self, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetCG2, self).__init__(init_cfg)
        self.cnn = _CNN(avg_pool=avg_pool, input_size=input_size, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)

        return dict(cnn=torch.sum(c_fea, dim=1), gru2=torch.sum(g_fea2, dim=1))


@BACKBONES.register_module()
class HCGNetG1G2(BaseBackbone):
    def __init__(self, input_size=80, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetG1G2, self).__init__(init_cfg)
        self.cnn = _CNN(input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)

        return dict(gru1=torch.sum(g_fea2, dim=1), gru2=torch.sum(g_fea2, dim=1))
