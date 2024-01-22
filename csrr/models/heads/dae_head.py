# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from csrr.evaluation.metrics import Accuracy
from csrr.registry import MODELS
from csrr.structures import DataSample


@MODELS.register_module()
class DAEHead(BaseModule):
    """Classification head.

    Args:
        loss_cls (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=0.1)``.
        loss_mse (dict): Config of mse loss. Defaults to
            ``dict(type='MSELoss', loss_weight=0.9, reduction='mean')
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss_cls: dict = dict(type='CrossEntropyLoss', loss_weight=0.1),
                 loss_mse: dict = dict(type='MSELoss', loss_weight=0.9, reduction='mean'),
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(DAEHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss_cls, nn.Module):
            loss_cls = MODELS.build(loss_cls)
        if not isinstance(loss_mse, nn.Module):
            loss_mse = MODELS.build(loss_mse)
        self.loss_cls = loss_cls
        self.loss_mse = loss_mse
        self.cal_acc = cal_acc

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``DAEHead``, we just obtain the feature
        of the last stage.
        """
        return feats[0]

    def forward(self, feats: Tuple[torch.Tensor], return_loss=False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        if return_loss:
            return feats
        else:
            return pre_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every sample.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        feats = self(feats, return_loss=True)

        # The part can not be traced by torch.fx
        losses = self._get_loss(feats[0], feats[1], feats[2], data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  x: torch.Tensor,
                  xd: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        cls_loss = self.loss_cls(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        mse_loss = self.loss_mse(x, xd)

        losses['loss_cls'] = cls_loss
        losses['loss_mse'] = mse_loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                                     'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(
            self,
            feats: Tuple[torch.Tensor],
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
            feats: Tuple[torch.Tensor],
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
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        diagnoses = self._get_diagnoses(cls_score, data_samples, **kwargs)
        return diagnoses

    def _get_diagnoses(
            self, cls_score: torch.Tensor,
            data_samples: List[DataSample],
            **kwargs
    ) -> List[DataSample]:
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = F.cross_entropy(cls_score, target, reduction='none')
        losses = losses.reshape(-1, 1)

        # compute scores
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, loss, score, label in zip(data_samples, losses, pred_scores,
                                                   pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_loss(loss, 'classification_loss').set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
