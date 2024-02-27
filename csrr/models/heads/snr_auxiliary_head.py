from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from csrr.registry import MODELS
from csrr.structures import DataSample, MultiTaskDataSample


@MODELS.register_module()
class SNRAuxiliaryHead(BaseModule):
    """Classification head.

    Args:
        loss (dict): List of Config of cross entropy loss. Defaults to None.
        loss_snr (dict): List of Config of cross entropy loss. Defaults to None.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 num_snr: int,
                 input_size: int,
                 output_size: int,
                 snr_output_size: int,
                 is_share: bool = False,
                 loss_cls=None,
                 loss_snr=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(SNRAuxiliaryHead, self).__init__(init_cfg=init_cfg)

        if loss_cls is None:
            loss_cls = dict(type='CrossEntropyLoss', loss_weight=1.0)
        if loss_snr is None:
            loss_snr = dict(type='CrossEntropyLoss', loss_weight=1.0)

        if not isinstance(loss_cls, nn.Module):
            loss_cls = MODELS.build(loss_cls)

        if not isinstance(loss_snr, nn.Module):
            loss_snr = MODELS.build(loss_snr)

        self.num_classes = num_classes
        self.num_snr = num_snr
        self.loss_cls = loss_cls
        self.loss_snr = loss_snr

        self.topk = topk
        self.cal_acc = cal_acc

        if is_share:
            self.classifier = nn.Sequential(
                nn.Conv1d(input_size, output_size, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(output_size, self.num_classes * self.num_snr, kernel_size=1, bias=False),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Conv1d(input_size, output_size * self.num_snr, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(output_size * self.num_snr, self.num_classes * self.num_snr,
                          kernel_size=1, bias=False, groups=self.num_snr),
            )

        self.snr_classifier = nn.Sequential(
            nn.Conv1d(input_size, snr_output_size, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(snr_output_size, self.num_snr, kernel_size=1, bias=False),
        )

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Dict[str, torch.Tensor]) -> Union[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """The forward process."""

        x = self.pre_logits(feats)
        x = torch.unsqueeze(x, dim=2)
        pres = self.classifier(x)
        snr_pres = self.snr_classifier(x)
        pres = torch.squeeze(pres)

        if self.training:
            pres = torch.squeeze(pres)
            snr_pres = torch.squeeze(snr_pres)
            return dict(pres=pres, snr_pres=snr_pres)
        else:
            pres = torch.reshape(pres, (-1, self.num_snr, self.num_classes))
            pres = F.softmax(pres, dim=2)
            snr_pres = F.softmax(snr_pres, dim=1)
            pres = pres * snr_pres
            pres = torch.sum(pres, dim=1)
            return pres


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
        cls_scores = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_scores, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_scores: Dict[str, torch.Tensor],
                  data_samples: List[MultiTaskDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0].get('amc'):
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.get('amc').gt_score for i in data_samples])
        else:
            target = torch.cat([i.get('amc').gt_label for i in data_samples])

        if 'gt_score' in data_samples[0].get('snr'):
            # Batch augmentation may convert labels to one-hot format scores.
            snr_target = torch.stack([i.get('snr').gt_score for i in data_samples])
        else:
            snr_target = torch.cat([i.get('snr').gt_label for i in data_samples])

        pres = torch.reshape(cls_scores['pres'], (-1, self.num_classes))
        snr_pres = cls_scores['snr_pres']

        target_ = target.new_zeros(snr_pres.size(0), self.num_snr)
        mask = target.new_zeros(snr_pres.size(0), self.num_snr)
        target_[torch.arange(snr_pres.size(0)), snr_target[:]] = target
        mask[torch.arange(snr_pres.size(0)), snr_target[:]] = 1
        target = torch.reshape(target_, (snr_pres.size(0) * self.num_snr,))
        mask = torch.reshape(mask, (snr_pres.size(0) * self.num_snr,))

        # compute loss
        losses = dict()

        losses['loss_amc'] = self.loss_cls(pres, target, avg_factor=snr_pres.size(0), weight=mask, **kwargs)
        losses['loss_snr'] = self.loss_snr(snr_pres, snr_target, avg_factor=snr_pres.size(0), **kwargs)

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
