"""MCformer with SNR-conditioned FiLM modulation (P2 architecture level, 2nd backbone).

Same design as PETCGDNNFiLM: two modulation points (per-channel on the conv
feature sequence entering the transformer, per-dim on the pooled pre-classifier
feature), and a zero-initialised SNR head so the model starts exactly equal to
the unconditioned MCformer.
"""
import torch
import torch.nn as nn

from csrr.registry import MODELS
from .mcformer import MCformer


@MODELS.register_module()
class MCformerFiLM(MCformer):

    def __init__(self, snr_min=-20.0, snr_max=18.0, embed_dim=32,
                 film_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.film_scale = float(film_scale)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        c = self.fea_dim
        h = 4 * self.fea_dim
        self._film_dims = (c, h)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2 * (c + h)))
        nn.init.zeros_(self.snr_mlp[-1].weight)
        nn.init.zeros_(self.snr_mlp[-1].bias)

    def _film_params(self, snr):
        r = ((snr.float() - self.snr_min) /
             (self.snr_max - self.snr_min)).clamp(0.0, 1.0)
        out = self.snr_mlp(r.view(-1, 1))
        c, h = self._film_dims
        g1, b1, g2, b2 = torch.split(out, [c, c, h, h], dim=1)
        s = self.film_scale
        return 1.0 + s * g1, s * b1, 1.0 + s * g2, s * b2

    def forward(self, x, snr=None):
        if snr is None:
            return super().forward(x)
        g1, b1, g2, b2 = self._film_params(snr)
        x = self.cnn(x)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x = x * g1.unsqueeze(1) + b1.unsqueeze(1)
        x = self.tnn(x)
        x = x[:, :4, :]
        phi = torch.reshape(x, [-1, 4 * self.fea_dim])
        phi = phi * g2 + b2
        if self.num_classes > 0:
            phi = self.classifier(phi)
        return (phi,)
