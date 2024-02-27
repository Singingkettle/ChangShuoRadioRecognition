from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from csrr.registry import MODELS
from csrr.structures import MultiTaskDataSample, DataSample
from csrr.structures.utils import format_score


@MODELS.register_module()
class HCGDNNHead(BaseModule):
    """Classification head.

    Args:
        loss (dict): List of Config of cross entropy loss. Defaults to None.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 init_cfg: Optional[dict] = None):
        super(HCGDNNHead, self).__init__(init_cfg=init_cfg)

        if loss is None:
            loss = dict(cnn=dict(type='CrossEntropyLoss', loss_weight=1),
                        gru1=dict(type='CrossEntropyLoss', loss_weight=1),
                        gru2=dict(type='CrossEntropyLoss', loss_weight=1)
                        )
        _loss = dict()
        for head_name in loss:
            if not isinstance(loss[head_name], nn.Module):
                sub_loss = MODELS.build(loss[head_name])
            else:
                sub_loss = loss[head_name]

            _loss[head_name] = sub_loss

        self.topk = topk
        self.loss_module = _loss

        weights = [0 / len(loss)] * len(loss)
        weights[-1] = 1 - sum(weights[0:-1])
        head_index = 0
        for head_name in loss:
            self.register_buffer(head_name, torch.tensor(weights[head_index]).float(), persistent=False)
            head_index += 1

    def set_weights(self, weights: Dict[str, float]) -> None:
        for head_name in weights:
            setattr(self, head_name,
                    torch.tensor(weights[head_name], device=getattr(self, head_name).device, dtype=torch.float32))

    def pre_logits(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``HCGDNNHead``, we just obtain the feature
        of the last stage.
        """
        scores = sum(F.softmax(feats[head_name], dim=1) * getattr(self, head_name) for head_name in feats)
        return scores

    def forward(self, feats: Dict[str, torch.Tensor], is_diagnose=False) -> Union[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """The forward process."""

        if self.training or is_diagnose:
            return feats
        else:
            pre_logits = self.pre_logits(feats)
            return pre_logits

    def loss(self, feats: Dict[str, torch.Tensor], data_samples: List[MultiTaskDataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[MultiTaskDataSample]): The annotation data of
                every sample.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(feats, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_scores: Dict[str, torch.Tensor],
                  data_samples: List[MultiTaskDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()

        for head_name in cls_scores:
            losses[f'loss_{head_name}'] = self.loss_module[head_name](cls_scores[head_name], target,
                                                                      avg_factor=cls_scores[head_name].size(0),
                                                                      **kwargs)

        return losses

    def predict(
            self,
            feats: Dict[str, torch.Tensor],
            data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every sample. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples

    def diagnose(
            self,
            feats: Dict[str, torch.Tensor],
            data_samples: List[Optional[DataSample]],
            **kwargs,
    ) -> List[DataSample]:
        """Diagnose without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every sample. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.
            **kwargs: Other keyword arguments to forward the loss module.
        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results and losses.
        """

        # The part can be traced by torch.fx
        cls_score = self(feats, is_diagnose=True)

        # The part can not be traced by torch.fx
        diagnoses = self._get_diagnoses(cls_score, data_samples, **kwargs)
        return diagnoses

    def _get_diagnoses(
            self, cls_score: Dict[str, torch.Tensor],
            data_samples: List[DataSample],
            **kwargs
    ) -> List[DataSample]:
        """Unpack data samples and compute loss."""

        # compute scores
        pred_scores = {head_name: F.softmax(cls_score[head_name], dim=1) for head_name in cls_score}

        pred_scores = [dict(zip(pred_scores, t)) for t in zip(*pred_scores.values())]

        out_data_samples = []
        if data_samples is None:
            raise ValueError("No data samples")

        for data_sample, score in zip(data_samples, pred_scores):

            for head_name in score:
                data_sample.set_field(format_score(score[head_name]), f'{head_name}_pred_score', dtype=torch.Tensor)
            out_data_samples.append(data_sample)
        return out_data_samples
