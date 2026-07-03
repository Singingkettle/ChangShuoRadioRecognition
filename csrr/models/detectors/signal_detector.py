# Copyright (c) Shuo Chang. All Rights Reserved.
"""Single-stage signal detector (detection module of the JDM framework)."""
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor, BaseModel

from csrr.registry import MODELS
from csrr.structures import DataSample


@MODELS.register_module()
class SignalDetDataPreprocessor(BaseDataPreprocessor):
    """Move inputs *and* data samples (with gt/pred interval tensors) to the
    target device and stack list inputs into a batch tensor."""

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        return {'inputs': inputs, 'data_samples': data.get('data_samples')}


@MODELS.register_module()
class SignalDetector(BaseModel):
    """Backbone + detection-head model for frame-level signal detection.

    Follows the same loss/predict contract as :class:`SignalClassifier`, so it
    plugs into ``tools/train.py`` / ``tools/test.py`` unchanged. Ground truth
    is carried by ``DataSample.gt_boxes`` / ``gt_box_labels`` (1-D frequency
    intervals in FFT-bin units and per-signal modulation indices); predictions
    are written to ``pred_boxes`` / ``pred_box_scores``.
    """

    def __init__(self,
                 backbone: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        data_preprocessor = data_preprocessor or {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'SignalDetDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(init_cfg=init_cfg,
                         data_preprocessor=data_preprocessor)
        self.backbone = backbone if isinstance(backbone, nn.Module) \
            else MODELS.build(backbone)
        self.head = head if isinstance(head, nn.Module) else MODELS.build(head)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor') -> Union[dict, list, torch.Tensor]:
        if mode == 'tensor':
            return self.head(self.extract_feat(inputs))
        elif mode == 'loss':
            return self.head.loss(self.extract_feat(inputs), data_samples)
        elif mode == 'predict':
            return self.head.predict(self.extract_feat(inputs), data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs)
