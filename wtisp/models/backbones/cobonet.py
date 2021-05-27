# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from ...runner import load_checkpoint


# 复现论文《Specific Emitter Identification Against Unreliable Features Interference Based on Time-Series Classification Network Structure》


# Resnet的base结构，修改为适合序列卷积的一维残差结构
# 网络输入为I-Q-P组成的3通道1xN序列
class ResBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=3, stride=stride, padding=5, dilation=5),
            # 按照原文中的配置，第一层padding和dilation均为5
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),

            nn.Conv1d(out_chans, out_chans, kernel_size=3, stride=stride, padding=3, dilation=3),
            # 按照原文中的配置，第二层padding和dilation均为3
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),

            nn.Conv1d(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, dilation=1),
            # 按照原文中的配置，第二层padding和dilation均为1
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),
        )

        # 判断shortcut 是直接连接还是需要做升维和降采样处理
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_chans, out_chans, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_chans)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResBlock_SE(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()

        reduction_rate = 16  # SE模块的超参数，表示降维和升维倍数
        self.left = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=3, stride=stride, padding=5, dilation=5),
            # 按照原文中的配置，第一层padding和dilation均为5
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),

            nn.Conv1d(out_chans, out_chans, kernel_size=3, stride=stride, padding=3, dilation=3),
            # 按照原文中的配置，第二层padding和dilation均为3
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),

            nn.Conv1d(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, dilation=1),
            # 按照原文中的配置，第二层padding和dilation均为1
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),
        )

        self.SE_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(out_chans, out_chans // reduction_rate, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(out_chans // reduction_rate, out_chans, bias=False),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        # 判断shortcut 是直接连接还是需要做升维和降采样处理
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_chans, out_chans, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_chans)
            )
        else:
            self.shortcut = nn.Sequential(nn.BatchNorm1d(out_chans))

    def forward(self, x):
        b, c, _ = x.size()
        out = self.left(x)
        factor = self.SE_block(out).view(b, c, 1)
        out = out * factor.expand_as(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DBi_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.BiLSTM = nn.LSTM(input_size=3, hidden_size=45, num_layers=5, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=90, out_features=128),
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (h, c) = self.BiLSTM(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out


@BACKBONES.register_module()
class CoBoNet(nn.Module):
    def __init__(self):
        super(CoBoNet, self).__init__()
        self.left_branch = nn.Sequential(
            ResBlock(in_chans=3, out_chans=64),
            ResBlock(in_chans=64, out_chans=128),
            ResBlock_SE(in_chans=128, out_chans=128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.right_branch = DBi_LSTM()

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
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, iqs, aps):
        iqs = torch.squeeze(iqs)
        aps = torch.squeeze(aps)
        x = torch.cat((iqs, aps), dim=1)
        x = x[:, :3, :]
        left_out = self.left_branch(x)
        right_out = self.right_branch(x)
        out = torch.cat((left_out, right_out), dim=1)
        return out
