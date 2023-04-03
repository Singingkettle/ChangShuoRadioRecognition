import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone
from ..builder import BACKBONES
from ...runner import Sequential


@BACKBONES.register_module()
class FMLNet(BaseBackbone):

    def __init__(self, depth=4, input_size=80, hidden_size=256, dp=0.2, init_cfg=None,
                 use_group=False, is_freeze=False):
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
        return torch.sum(fea, dim=1)

    def train(self, mode=True):
        super(FMLNet, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()


from torch.nn.parameter import Parameter


@BACKBONES.register_module()
class FMLNetV2(BaseBackbone):

    def __init__(self, depth=4, input_size=80, dp=0.2, init_cfg=None,
                 is_freeze=False, is_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FMLNetV2, self).__init__(init_cfg)
        self.cnn1 = Sequential(
            nn.Conv1d(depth, input_size, kernel_size=3, groups=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.cnn2 = Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, groups=16, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
        )
        self.cnn3 = Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, groups=4, padding=1),
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

    def forward(self, iqs, aps):
        x = torch.concat([iqs, aps], dim=1)
        fea1 = self.cnn1(x)
        fea2 = self.cnn2(fea1)
        fea3 = self.cnn3(fea2 + fea1)
        fea = fea1 + fea2 + fea3
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
        super(FMLNetV2, self).train(mode)
        if self.is_freeze:
            self._freeze_layers()
