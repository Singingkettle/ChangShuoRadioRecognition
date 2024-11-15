from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
from mmengine.model.weight_init import BaseInit, _get_bases_name, update_init_info
from csrr.registry import WEIGHT_INITIALIZERS


def _svd(w, gain=1.0):
    rows = w.size(0)
    cols = w.numel() // rows
    a = np.random.normal(0, 1, (cols, rows))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == w.shape else v

    q = np.transpose(q)
    with torch.no_grad():
        w.copy_(torch.from_numpy(q).float())
        w.mul_(gain)
    return w


def rnn_init(module: nn.Module, gain: float = 1.0) -> None:
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data, gain=gain)
        elif 'bias_ih' in name:
            nn.init.zeros_(param)
            n = param.size(0)
            param.data[(n // 4):(n // 2)].fill_(1)
        elif 'bias_hh' in name:
            nn.init.zeros_(param)


@WEIGHT_INITIALIZERS.register_module(name='LSTM')
class LSTMInit(BaseInit):
    def __init__(self, gain: float, layer: Union[str, List, None] = None):
        super().__init__(layer=layer)
        self.gain = gain

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be a str or a list of str, \
                    but got a {type(layer)}')
        else:
            layer = []

        self.layer = [layer] if isinstance(layer, str) else layer

    def __call__(self, module) -> None:

        def init(m):
            if self.wholemodule:
                rnn_init(m, self.gain)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    rnn_init(m, self.gain)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}, gain={self.gain}'
        return info
