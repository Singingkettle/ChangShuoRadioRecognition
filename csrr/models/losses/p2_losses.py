"""P2 spectrum losses for the null-ladder audit (pre-registered in P2_PREREG).

SoftmaxFocalLoss: multi-class focal loss (softmax form) as a published
target-level representative. SNRCurriculumCELoss: loss-weight-level curriculum
that unlocks low-SNR samples progressively during training.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from csrr.registry import MODELS
from csrr.structures import DataSample
from .rcps_loss import collect_reliability


@MODELS.register_module()
class SoftmaxFocalLoss(nn.Module):
    """Multi-class focal loss: FL = -(1 - p_y)^gamma * log p_y."""

    requires_data_samples = False

    def __init__(self,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 loss_weight: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> torch.Tensor:
        logp = F.log_softmax(cls_score, dim=-1)
        logp_y = logp.gather(1, label.view(-1, 1)).squeeze(1)
        p_y = logp_y.exp()
        loss = -torch.pow(1.0 - p_y, self.gamma) * logp_y
        if weight is not None:
            loss = loss * weight
        red = reduction_override or self.reduction
        if red == "mean":
            loss = loss.mean()
        elif red == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


@MODELS.register_module()
class SNRCurriculumCELoss(nn.Module):
    """High-SNR-first curriculum CE.

    A soft gate w = min_w + (1-min_w)*sigmoid((snr - thr)/tau) weights each
    sample; the threshold thr moves linearly from snr_max down past snr_min
    over warmup_iters, after which all samples are fully included. The
    weighted mean divides by the weight sum so the effective LR is stable.
    """

    requires_data_samples = True

    def __init__(self,
                 reliability_key: str = "snr",
                 snr_min: float = -20.0,
                 snr_max: float = 18.0,
                 warmup_iters: int = 75000,
                 tau: float = 2.0,
                 min_weight: float = 0.0,
                 loss_weight: float = 1.0):
        super().__init__()
        self.reliability_key = reliability_key
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.warmup_iters = int(warmup_iters)
        self.tau = float(tau)
        self.min_weight = float(min_weight)
        self.loss_weight = float(loss_weight)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long),
                             persistent=True)

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> torch.Tensor:
        if data_samples is None:
            raise ValueError("SNRCurriculumCELoss requires data_samples.")
        snr = collect_reliability(data_samples, self.reliability_key,
                                  cls_score.device).float()
        if self.training:
            self._step += 1
        progress = (self._step.item() / max(1, self.warmup_iters))
        progress = min(1.0, progress)
        # threshold sweeps from snr_max down to (snr_min - 4*tau) so the gate
        # fully opens for every sample by the end of the warmup.
        thr = self.snr_max - progress * (self.snr_max - self.snr_min + 4.0 * self.tau)
        w = self.min_weight + (1.0 - self.min_weight) * torch.sigmoid(
            (snr - thr) / self.tau)
        ce = F.cross_entropy(cls_score, label, reduction="none")
        loss = (w * ce).sum() / w.sum().clamp_min(1e-6)
        return self.loss_weight * loss
