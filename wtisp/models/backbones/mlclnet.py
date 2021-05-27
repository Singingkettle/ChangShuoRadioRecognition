import logging

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor(
            [1.] + [0.] * (2 * k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2 * self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2 *
                                self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :,
                              :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :,
                              :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


@BACKBONES.register_module()
class MLCLNet(nn.Module):

    def __init__(self):
        super(MLCLNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            DyReLUB(256, conv_type='2d'),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            DyReLUB(256, conv_type='2d'),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            DyReLUB(80, conv_type='2d'),
            nn.Dropout(0.5),
        )
        self.gru1 = nn.GRU(input_size=80,
                           hidden_size=100, batch_first=True)

        self.gru2 = nn.GRU(input_size=100,
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
        cx1 = self.conv1(x)
        cx2 = self.conv2(cx1)
        cx3 = self.conv3(cx2)

        cx3 = torch.squeeze(cx3, dim=2)
        cx3 = torch.transpose(cx3, 1, 2)
        gx1, _ = self.gru1(cx3)
        gx2, _ = self.gru2(gx1)
        return gx2[:, -1, :]
