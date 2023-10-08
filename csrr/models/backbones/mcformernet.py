import torch
from torch import nn

from .base import BaseBackbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class MCformerNet(BaseBackbone):

    def __init__(self, dim=32, init_cfg=None):
        super(MCformerNet, self).__init__(init_cfg)
        self.cnn = nn.Sequential(
            nn.Conv1d(2, dim, kernel_size=65, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim, batch_first=True)
        self.tnn = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, iqs):
        c_fea = self.cnn(iqs)
        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)
        x = self.tnn(c_fea)
        x = x[:, :4, :]
        x = torch.reshape(x, [-1, 4 * self.dim])
        return x
