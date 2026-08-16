"""PETCGDNN with SNR-conditioned FiLM modulation (P2 architecture-level rep.).

Two modulation points: per-channel on the conv feature sequence and per-dim on
the pre-classifier feature phi. The SNR head is zero-initialised so the model
starts exactly equal to the unconditioned PETCGDNN (gamma=1, beta=0).
"""
import torch
import torch.nn as nn

from csrr.registry import MODELS
from .petcgdnn import PETCGDNN


@MODELS.register_module()
class PETCGDNNFiLM(PETCGDNN):

    def __init__(self, snr_min=-20.0, snr_max=18.0, embed_dim=32,
                 film_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.film_scale = float(film_scale)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        conv_ch = 25
        hidden = self.gru.hidden_size
        self._film_dims = (conv_ch, hidden)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2 * (conv_ch + hidden)))
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
        x = self.features(x)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)
        x = x * g1.unsqueeze(1) + b1.unsqueeze(1)
        x, _ = self.gru(x)
        phi = x[:, -1, :]
        phi = phi * g2 + b2
        if self.num_classes > 0:
            phi = self.classifier(phi)
        return (phi,)
