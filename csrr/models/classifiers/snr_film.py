"""Classifier wrapper that feeds per-sample SNR into a FiLM backbone."""
from typing import List, Optional

import torch

from csrr.registry import MODELS
from csrr.structures import DataSample
from ..losses.rcps_loss import collect_reliability
from .signal import SignalClassifier


@MODELS.register_module()
class SNRFiLMClassifier(SignalClassifier):

    def __init__(self, reliability_key="snr", shuffle_snr=False, **kwargs):
        super().__init__(**kwargs)
        self.reliability_key = reliability_key
        self.shuffle_snr = shuffle_snr  # sanity mode: permute SNR at test time

    def _snr(self, inputs, data_samples):
        snr = collect_reliability(data_samples, self.reliability_key,
                                  inputs.device)
        if self.shuffle_snr:
            snr = snr[torch.randperm(snr.numel(), device=snr.device)]
        return snr

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        feats = self.backbone(inputs, self._snr(inputs, data_samples))
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.backbone(inputs, self._snr(inputs, data_samples))
        return self.head.predict(feats, data_samples, **kwargs)
