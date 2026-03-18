# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseDataPreprocessor

from csrr.registry import MODELS
from csrr.structures import (DataSample, MultiTaskDataSample,
                             cat_batch_labels, tensor_split)


@MODELS.register_module()
class SignalDataPreprocessor(BaseDataPreprocessor):
    """Signal pre-processor for classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = self.cast_data(data['inputs'])
        data_samples = data.get('data_samples', None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, DataSample):
            batch_label = None
            batch_score = None
            if 'gt_label' in sample_item:
                gt_labels = [sample.gt_label for sample in data_samples]
                batch_label, label_indices = cat_batch_labels(gt_labels)
                batch_label = batch_label.to(self.device)
            if 'gt_score' in sample_item:
                gt_scores = [sample.gt_score for sample in data_samples]
                batch_score = torch.stack(gt_scores).to(self.device)

            if batch_label is not None:
                for sample, label in zip(
                        data_samples, tensor_split(batch_label,
                                                   label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)
        elif isinstance(sample_item, MultiTaskDataSample):
            data_samples = self.cast_data(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}
