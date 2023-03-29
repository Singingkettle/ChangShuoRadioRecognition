from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from .base import BaseBackbone
from ..builder import BACKBONES
from ...runner import Sequential


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)


@BACKBONES.register_module()
class FMLNet(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None, use_my=False, use_group=False):
        super(FMLNet, self).__init__(init_cfg)
        if use_group:
            self.cnn = Sequential(
                nn.Conv1d(depth, hidden_size, kernel_size=3, groups=2),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, groups=16),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, input_size, kernel_size=3, groups=4),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )
        else:
            self.cnn = Sequential(
                nn.Conv1d(depth, hidden_size, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
                nn.Conv1d(hidden_size, input_size, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )
        if use_my:
            self.tnn = TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)
        else:
            self.tnn = nn.TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.transpose(c_fea, 1, 2)
        fea = self.tnn(c_fea)

        return torch.sum(fea, dim=1)
