import torch
import torch.nn as nn

from .base import BaseBackbone
from .cnnnet import CNNNet
from ..builder import BACKBONES


class RNNBasicBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=None, rnn_mode='LSTM'):
        super(RNNBasicBlock, self).__init__()
        self.rnn_block = []
        if rnn_mode == 'LSTM':
            rnn_layer = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size, batch_first=True)
        elif rnn_mode == 'GRU':
            rnn_layer = nn.GRU(input_size=input_size,
                               hidden_size=hidden_size, batch_first=True)
        else:
            raise NotImplementedError(
                f'The rnn mode of {rnn_mode} is not supported in current RNNBasicBlock!')

        self.add_module('rnn_layer', rnn_layer)
        self.rnn_block.append('rnn_layer')

        if dropout_rate is not None:
            if isinstance(dropout_rate, float) or isinstance(dropout_rate, int):
                dropout_layer = nn.Dropout(p=dropout_rate)
                self.add_module('dropout_layer', dropout_layer)
                self.rnn_block.append('dropout_layer')
            else:
                raise NotImplementedError(
                    'The type of augment dropout_rate is only support float ([0.0, 1.0]) or '
                    'int ((0.0, 1.0)). Please check your code')

    def forward(self, x):
        for layer_name in self.rnn_block:
            layer = getattr(self, layer_name)
            if 'rnn' in layer_name:
                x, _ = layer(x)
            else:
                x = layer(x)

        return x


class RNNNet(nn.Module):
    arch_settings = {
        1: ((50,), 0.5),
        2: ((100, 50), 0.5),
    }

    def __init__(self, depth, input_size, is_last=True, rnn_mode='LSTM'):
        super(RNNNet, self).__init__()
        self.is_last = is_last
        hiddens, dropout_rate = self.arch_settings[depth]

        self.rnnlayers = []
        for i, hidden_size in enumerate(hiddens):
            rnn_layer = self.make_rnn_layer(
                input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate, rnn_mode=rnn_mode)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, rnn_layer)
            self.rnnlayers.append(layer_name)
            input_size = hidden_size

    def make_rnn_layer(self, **kwargs):
        return RNNBasicBlock(**kwargs)

    def forward(self, x):
        for i, layer_name in enumerate(self.rnnlayers):
            layer = getattr(self, layer_name)
            x = layer(x)

        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 2)  # (N, C, 1, W)
        if self.is_last:
            x = x[:, :, 0, -1]

        return x


@BACKBONES.register_module()
class CRNet(BaseBackbone):

    def __init__(self, in_channels, cnn_depth, rnn_depth, input_size, in_height=2, avg_pool=None, out_indices=(1,),
                 is_last=True, rnn_mode='LSTM', init_cfg=None):
        super(CRNet, self).__init__(init_cfg)
        self.cnn_net = self._make_cnn_net(
            depth=cnn_depth, in_channels=in_channels, in_height=in_height, avg_pool=avg_pool, out_indices=out_indices)
        self.rnn_net = self._make_rnn_net(
            depth=rnn_depth, input_size=input_size, is_last=is_last, rnn_mode=rnn_mode)

    def _make_cnn_net(self, **kwargs):
        return CNNNet(**kwargs)

    def _make_rnn_net(self, **kwargs):
        return RNNNet(**kwargs)

    def forward(self, iqs=None, aps=None):
        if iqs is None:
            x = aps
        elif aps is None:
            x = iqs
        else:
            raise ValueError('Either input iq sequence or ap sequence!')
        x = self.cnn_net(x)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x = self.rnn_net(x)
        return x
