import logging
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from csrr.models.backbone.hcgnet import _CNN
from csrr.models.backbone.mlnet import SELayer
from csrr.models.builder import BACKBONES
from csrr.runner import load_checkpoint, BaseModule


@BACKBONES.register_module()
class FMLNetOld(nn.Module):

    def __init__(self, dropout_rate=0.5, in_size=4,
                 channel_mode=False, skip_connection=False,
                 reduction=16, avg_pool=None):
        super(FMLNetOld, self).__init__()
        self.channel_mode = channel_mode
        self.skip_connection = skip_connection
        if self.channel_mode:
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_size, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 100, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            )
        else:
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 256, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(256, 100, kernel_size=(1, 3)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.AvgPool2d((in_size, 1)),
            )
        if avg_pool is not None:
            self.has_avg_pool = True
            self.avg_pool_layer = nn.AvgPool2d(avg_pool)
            self.se = SELayer(127, reduction=reduction)
        else:
            self.se = SELayer(122, reduction=reduction)
            self.has_avg_pool = False

        self.gru = nn.GRU(input_size=100, hidden_size=50,
                          num_layers=2, dropout=dropout_rate,
                          batch_first=True, bidirectional=True)

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

    def forward(self, iqs, aps):
        if self.channel_mode:
            x = torch.concat([iqs, aps], dim=1)
        else:
            x = torch.concat([iqs, aps], dim=2)
        x = self.conv_net(x)
        if self.has_avg_pool:
            x = self.avg_pool_layer(x)
        x = torch.squeeze(x, dim=2)
        c_x = torch.transpose(x, 1, 2)

        se_w = self.se(c_x)
        x, _ = self.gru(c_x)
        x = torch.mul(x, se_w)

        if self.skip_connection:
            x = x + c_x

        x = torch.sum(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNet(nn.Module):

    def __init__(self, depth=4, input_size=80, head_size=1, hidden_size=16, dp=0.1):
        super(FMLNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(depth, input_size // 2, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(input_size // 2),
            nn.Conv1d(input_size // 2, input_size, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(input_size),
        )
        self.input_size = input_size
        self.tnn1 = nn.TransformerEncoderLayer(input_size, head_size, hidden_size, dropout=dp, batch_first=True)
        self.tnn2 = nn.TransformerEncoderLayer(input_size, head_size, hidden_size, dropout=dp, batch_first=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        x = self.cnn(x)
        x = torch.transpose(x, 1, 2)

        x = torch.reshape(x, [-1, 16, self.input_size])
        x = self.tnn1(x)
        x = torch.mean(x, dim=1)
        x = torch.reshape(x, [-1, 8, self.input_size])
        x = self.tnn2(x)

        x = torch.mean(x, dim=1)

        return x


@BACKBONES.register_module()
class FMLNetV2(nn.Module):

    def __init__(self, depth=4, input_size=80, avg_pool=None, has_stride=False, dp=0.5):
        super(FMLNetV2, self).__init__()
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp,
                        padding=(0, 1))
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dp)
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)
        self.input_size = input_size

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.LSTM):
                    m.reset_parameters()
                elif isinstance(m, nn.GRU):
                    m.reset_parameters()

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)

        c_fea = torch.reshape(c_fea, [-1, 16, self.input_size])
        g_fea1, _ = self.gru1(c_fea)
        g_fea1 = torch.sum(g_fea1, dim=1)
        fea = self.dropout(g_fea1)

        fea = torch.reshape(fea, [-1, 8, self.input_size])
        g_fea2, _ = self.gru2(fea)

        return torch.sum(g_fea2, dim=1)


@BACKBONES.register_module()
class FMLNetV3(nn.Module):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2):
        super(FMLNetV3, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(depth, hidden_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, input_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.transpose(c_fea, 1, 2)
        g_fea1, _ = self.gru1(c_fea)

        return torch.sum(g_fea1, dim=1)


@BACKBONES.register_module()
class FMLNetV3(nn.Module):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2):
        super(FMLNetV3, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(depth, hidden_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, input_size, kernel_size=3, stride=1, bias=False, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=input_size // 2, batch_first=True, bidirectional=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.transpose(c_fea, 1, 2)
        g_fea1, _ = self.gru1(c_fea)

        return torch.sum(g_fea1, dim=1)


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
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        # self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)

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
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            # x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            # x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    # def _ff_block(self, x: Tensor) -> Tensor:
    #     x = self.activation(self.linear1(x))
    #     return self.dropout2(x)


@BACKBONES.register_module()
class FMLNetV5(nn.Module):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2):
        super(FMLNetV5, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(depth, hidden_size, kernel_size=3, stride=1, bias=False, padding=0, groups=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, bias=False, padding=0, groups=8),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(hidden_size, input_size, kernel_size=3, stride=1, bias=False, padding=0, groups=4),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.tnn = TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c_fea = self.cnn(x)
        c_fea = torch.transpose(c_fea, 1, 2)
        fea = self.tnn(c_fea)

        return torch.sum(fea, dim=1)


@BACKBONES.register_module()
class FMLNetV6(BaseModule):

    def __init__(self, depth=2, input_size=80, avg_pool=None, has_stride=False, dp=0.5, merge='sum', init_cfg=None):
        super(FMLNetV6, self).__init__(init_cfg)
        self.cnn = _CNN(depth=depth, input_size=input_size, avg_pool=avg_pool, has_stride=has_stride, dp=dp)
        self.merge = merge

    def forward(self, iqs):
        c_fea = self.cnn(iqs)

        if self.merge == 'sum':
            return torch.sum(c_fea, dim=1)  # this is best
        elif self.merge == 'mean':
            return torch.mean(c_fea, dim=1)
        elif self.merge == 'max':  # this is best
            return torch.max(c_fea, dim=1)[0]
        elif self.merge == 'min':
            return torch.min(c_fea, dim=1)[0]
        elif self.merge == 'logsumexp':
            return torch.logsumexp(c_fea, dim=1)
        elif self.merge == 'median':
            return torch.median(c_fea, dim=1)[0]
        elif self.merge == 'std':
            return torch.std(c_fea, dim=1)
        elif self.merge == 'quantile':
            return torch.quantile(c_fea, 0.6, dim=1, interpolation='higher')
        else:
            raise NotImplementedError(f'There is no torch.{self.merge} operation!')


@BACKBONES.register_module()
class FMLNetV2(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None,
                 use_group=False, is_freeze=False, has_stride=False, groups=(16, 4)):
        super(FMLNetV2, self).__init__(init_cfg)
        if has_stride:
            stride = 2
        else:
            stride = 1
        if use_group:
            self.cnn1 = Sequential(
                nn.Conv1d(depth, hidden_size, kernel_size=3, groups=2, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp)
            )
            self.cnn2 = Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, groups=groups[0], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp)
            )
            self.cnn3 = Sequential(
                nn.Conv1d(hidden_size, input_size, kernel_size=3, groups=groups[1], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )
        else:
            self.cnn1 = Sequential(
                nn.Conv1d(depth, hidden_size, kernel_size=3, groups=2, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp)
            )
            self.cnn2 = Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, groups=groups[0], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp)
            )
            self.cnn3 = Sequential(
                nn.Conv1d(hidden_size, input_size, kernel_size=3, groups=groups[1], stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dp),
            )

        self.tnn = nn.TransformerEncoderLayer(input_size, 1, hidden_size, dropout=dp, batch_first=True)

        self.is_freeze = is_freeze

    def _freeze_layers(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        c1 = self.cnn1(x)
        c2 = self.cnn2(c1)
        c3 = self.cnn3(c2)

        f1 = torch.transpose(c1, 1, 2)
        f2 = torch.transpose(c2, 1, 2)
        f3 = torch.transpose(c3, 1, 2)
        f4 = self.tnn(f3)

        f1 = torch.sum(f1, dim=1)
        f2 = torch.sum(f2, dim=1)
        f3 = torch.sum(f3, dim=1)
        f4 = torch.sum(f4, dim=1)
        return (f1, f2, f3, f4)

    def train(self, mode=True):
        super(FMLNetV2, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()


@BACKBONES.register_module()
class FMLNetV3(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None,
                 is_freeze=False, groups=(2, 16, 4), stride=2, dilation=2):
        super(FMLNetV3, self).__init__(init_cfg)
        self.cnn1 = Sequential(
            nn.Conv1d(depth, hidden_size, kernel_size=3, groups=groups[0], stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dp)
        )
        self.cnn2 = Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, groups=groups[1], stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dp)
        )
        self.cnn3 = Sequential(
            nn.Conv1d(hidden_size, input_size, kernel_size=3, groups=groups[2], stride=stride, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )

        self.is_freeze = is_freeze

    def _freeze_layers(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, iqs, fts):
        x = torch.concat([iqs, fts], dim=1)
        c1 = self.cnn1(x)
        c2 = self.cnn2(c1)
        c3 = self.cnn3(c2)

        f1 = torch.transpose(c1, 1, 2)
        f2 = torch.transpose(c2, 1, 2)
        f3 = torch.transpose(c3, 1, 2)

        f1 = torch.sum(f1, dim=1)
        f2 = torch.sum(f2, dim=1)
        f3 = torch.sum(f3, dim=1)
        return (f1, f2, f3)

    def train(self, mode=True):
        super(FMLNetV3, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()


@BACKBONES.register_module()
class FasterMLNet(BaseBackbone):

    def __init__(self, depth=2, input_size=80, dp=0.2, init_cfg=None,
                 is_freeze=False, is_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FasterMLNet, self).__init__(init_cfg)
        self.cnn = Sequential(
            nn.Conv1d(depth, 256, kernel_size=3, groups=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(256, 256, kernel_size=3, groups=16, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Conv1d(256, input_size, kernel_size=3, groups=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.embed_dim = input_size
        self.num_heads = 4
        self.q_head_dim = self.embed_dim // self.num_heads // 4
        self.k_head_dim = self.embed_dim // self.num_heads // 4
        self.v_head_dim = self.embed_dim // self.num_heads
        self.q_proj_weight = Parameter(torch.empty((self.embed_dim // 4, input_size), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.embed_dim // 4, input_size), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((self.embed_dim, input_size), **factory_kwargs))
        self.dropout = nn.Dropout(dp)
        self.is_freeze = is_freeze
        self.is_residual = is_residual
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def _freeze_layers(self):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, iqs):
        fea = self.cnn(iqs)
        fea = fea.transpose(1, 2)
        fea = fea.transpose(0, 1)

        tgt_len, bsz, embed_dim = fea.shape
        q = F.linear(fea, self.q_proj_weight)
        k = F.linear(fea, self.k_proj_weight)
        v = F.linear(fea, self.v_proj_weight)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.v_head_dim).transpose(0, 1)

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        fea = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return torch.sum(fea.transpose(1, 0), dim=1)

    def train(self, mode=True):
        super(FasterMLNet, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()