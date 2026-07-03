# Copyright (c) Shuo Chang. All Rights Reserved.
"""Joint framework for detection and modulation classification (JDM).

Inference-only wrapper chaining the two separately trained JDM modules
(paper Sec. V-A): the detection module produces frequency-band "proposals"
from the FFT of the received frame; each proposal is converted to a
single-signal baseband crop (carrier removal + brick-wall low-pass in the FFT
domain) that the classification module labels with a modulation type.

Training is done per-module with the dedicated configs
(``configs/jdm/jdm-det_*.py`` and ``configs/jdm/jdm-amc_*.py``); this wrapper
is used with ``tools/test.py`` for end-to-end evaluation. Load the trained
weights via per-submodule ``init_cfg=dict(type='Pretrained', checkpoint=...,
prefix=...)`` entries or a merged checkpoint.
"""
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel

from csrr.registry import MODELS
from csrr.structures import DataSample


@MODELS.register_module()
class JDMFramework(BaseModel):
    """Chained detector + classifier for end-to-end JDM evaluation.

    Args:
        detector (dict): config of a :class:`SignalDetector`.
        classifier (dict): config of a model whose ``(feats)`` tensor output
            are modulation logits, e.g. ``SignalClassifier`` with
            :class:`JDMClassificationBackbone`.

    The dataloader must feed **time-domain** I/Q frames of shape
    ``(N, 2, frame_length)`` (i.e. the CSRD pipeline *without*
    ``IQToSpectrum``); the spectrum for the detector is computed internally so
    that proposals can be cut out of the same FFT.
    """

    def __init__(self,
                 detector: dict,
                 classifier: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        data_preprocessor = data_preprocessor or {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'SignalDetDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(init_cfg=init_cfg,
                         data_preprocessor=data_preprocessor)
        self.detector = MODELS.build(detector)
        self.classifier = MODELS.build(classifier)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'predict') -> Union[list, torch.Tensor]:
        if mode != 'predict':
            raise RuntimeError(
                'JDMFramework is an inference-only wrapper; train the '
                'detection and classification modules separately '
                f'(got mode="{mode}").')
        return self.predict(inputs, data_samples)

    @torch.no_grad()
    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None
                ) -> List[DataSample]:
        frame = inputs[:, 0].to(torch.float32) \
            + 1j * inputs[:, 1].to(torch.float32)  # (N, L)
        spectrum = torch.fft.fftshift(torch.fft.fft(frame, dim=-1), dim=-1)
        det_inputs = torch.stack(
            [spectrum.abs(), spectrum.angle()], dim=1)  # (N, 2, L)

        data_samples = self.detector(det_inputs, data_samples,
                                     mode='predict')

        crops, owners = [], []
        for i, sample in enumerate(data_samples):
            for box in sample.pred_boxes:
                crops.append(self._to_baseband(spectrum[i], box))
                owners.append(i)

        num_dets = [int(s.pred_boxes.size(0)) for s in data_samples]
        if crops:
            batch = torch.stack(crops).unsqueeze(1)  # (K, 1, 2, L)
            logits = self.classifier(batch, mode='tensor')
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]
            scores = torch.softmax(logits, dim=1)
            labels = scores.argmax(dim=1)
        for i, sample in enumerate(data_samples):
            start = sum(num_dets[:i])
            stop = start + num_dets[i]
            if num_dets[i] > 0:
                sample.set_field(labels[start:stop], 'pred_box_labels')
                sample.set_field(
                    scores[start:stop].max(dim=1).values,
                    'pred_box_label_scores')
            else:
                device = sample.pred_boxes.device
                sample.set_field(
                    torch.zeros(0, dtype=torch.long, device=device),
                    'pred_box_labels')
                sample.set_field(
                    torch.zeros(0, device=device), 'pred_box_label_scores')
        return data_samples

    @staticmethod
    def _to_baseband(spectrum: torch.Tensor, box: torch.Tensor
                     ) -> torch.Tensor:
        """Cut one proposal out of a (fftshift-ed) frame spectrum.

        Keeps only the bins inside the proposal interval (ideal low-pass),
        rolls the band center to DC (carrier removal) and returns the
        time-domain baseband crop as a real ``(2, L)`` tensor.
        """
        num_bins = spectrum.size(-1)
        left = int(box[0].round().clamp(0, num_bins - 1))
        right = int(box[1].round().clamp(left + 1, num_bins))
        masked = torch.zeros_like(spectrum)
        masked[left:right] = spectrum[left:right]
        center = (left + right) // 2
        masked = torch.roll(masked, num_bins // 2 - center, dims=-1)
        baseband = torch.fft.ifft(torch.fft.ifftshift(masked, dim=-1), dim=-1)
        return torch.stack([baseband.real, baseband.imag])
