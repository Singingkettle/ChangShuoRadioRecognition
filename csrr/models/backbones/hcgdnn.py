import torch
import torch.nn as nn

from .base_backbone import BaseBackbone
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
        self._init_weights()

    def _init_weights(self):
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

    def forward(self, x):
        c_fea = self.conv_net(x)
        if self.has_avg_pool:
            c_fea = self.avg_pool_layer(c_fea)

        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)

        return c_fea


class HCGNetCNN(BaseBackbone):

    def __init__(self, depth=2, input_size=100, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetCNN, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)

        return dict(cnn=torch.sum(c_fea, dim=1))


class HCGNetGRU1(BaseBackbone):

    def __init__(self, depth=2, input_size=100, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetGRU1, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self._init_weights()

    def _init_weights(self):
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

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)

        return dict(cnn=torch.sum(c_fea, dim=1), gru1=torch.sum(g_fea1, dim=1))


class HCGNetGRU2(BaseBackbone):

    def __init__(self, depth=2, input_size=100, avg_pool=None, has_stride=False, dp=0.5, init_cfg=None):
        super(HCGNetGRU2, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self._init_weights()

    def _init_weights(self):
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

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)

        return dict(cnn=torch.sum(c_fea, dim=1), gru1=torch.sum(g_fea1, dim=1), gru2=torch.sum(g_fea2, dim=1))


@BACKBONES.register_module()
class HCGDNN(BaseBackbone):
    """`HCGDNN <https://ieeexplore.ieee.org/document/9764618>`_ backbone
    The input for HCGDNN is two mod (iq+ap [4*1*L]) for the smae frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1, input_size=100, avg_pool=None,
                 has_stride=False, dp=0.5, init_cfg=None, outputs=['cnn', 'gru1', 'gru2']):
        super(HCGDNN, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.outputs = outputs
        if 'gru2' in self.outputs:
            self.features = HCGNetGRU2(input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        elif 'gru1' in self.outputs:
            self.features = HCGNetGRU1(input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        elif 'cnn' in self.outputs:
            self.features = _CNN(input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)

        if self.num_classes > 0:
            for output_name in self.outputs:
                classifier = nn.Sequential(
                    nn.Linear(100, 288),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(288, self.num_classes),
                )
                self.add_module(f'classifier_{output_name}', classifier)

    def forward(self, x):
        x = self.features(x)
        y = dict()
        if self.num_classes > 0:
            for head_name in self.outputs:
                layer = getattr(self, f'classifier_{head_name}')
                y[head_name] = layer(x[head_name])
        else:
            y = x

        return y
