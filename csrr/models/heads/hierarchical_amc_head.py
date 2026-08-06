# Copyright (c) Shuo Chang and contributors. Licensed under the Apache-2.0 License.
"""Hierarchical AMC head for the return-to-IQ recognizer.

The 57 wideband classes split into single-carrier families (constellations,
frequency/analog) and multi-carrier OFDM. One decimated OFDM subcarrier looks
like noise, so a flat 57-way head wastes capacity separating the two regimes.
This head makes that split explicit: a stage-1 router predicts single vs.
multi-carrier, a stage-2 single head classifies the 45 single-carrier classes,
and a stage-2 multi head classifies the 12 OFDM classes. The hierarchy is
entirely inside the head, so the dataset stays a plain 57-class AMC dataset.
Training minimizes the sum of the three cross-entropies; inference routes on the
router and emits a proper 57-way score for standard metrics.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from csrr.registry import MODELS
from csrr.structures import DataSample


@MODELS.register_module()
class HierarchicalAMCHead(BaseModule):
    """Stage-1 single/multi router + stage-2 single(45)/multi(12) heads.

    Args:
        feat_dim (int): Backbone feature dimension. Defaults to 256.
        num_classes (int): Total number of fine-grained classes. Defaults to 57.
        multi_class_indices (list[int]): Indices (in ``[0, num_classes)``) of the
            multi-carrier (OFDM) classes. The rest are single-carrier.
        label_smoothing (float): Label-smoothing for every cross-entropy.
            Defaults to 0.0.
        dropout (float): Dropout in the single-carrier head. Defaults to 0.3.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """

    def __init__(self,
                 feat_dim: int = 256,
                 num_classes: int = 57,
                 multi_class_indices: Optional[List[int]] = None,
                 label_smoothing: float = 0.0,
                 dropout: float = 0.3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        multi_class_indices = sorted(set(multi_class_indices or []))
        multi_set = set(multi_class_indices)
        single_ids = [i for i in range(num_classes) if i not in multi_set]
        multi_ids = list(multi_class_indices)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # 57-class index -> (is_multi, position within its branch)
        is_multi = torch.zeros(num_classes, dtype=torch.bool)
        single_pos = torch.full((num_classes,), -1, dtype=torch.long)
        multi_pos = torch.full((num_classes,), -1, dtype=torch.long)
        for p, c in enumerate(single_ids):
            single_pos[c] = p
        for p, c in enumerate(multi_ids):
            is_multi[c] = True
            multi_pos[c] = p
        self.register_buffer('is_multi', is_multi)
        self.register_buffer('single_pos', single_pos)
        self.register_buffer('multi_pos', multi_pos)
        self.register_buffer('single_ids', torch.tensor(single_ids, dtype=torch.long))
        self.register_buffer('multi_ids', torch.tensor(multi_ids, dtype=torch.long))

        n_single, n_multi = len(single_ids), max(len(multi_ids), 1)
        self.coarse = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.single = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feat_dim, n_single))
        self.multi = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, n_multi))

    def forward(self, feats: Tuple[torch.Tensor]):
        f = feats[-1]
        return self.coarse(f), self.single(f), self.multi(f)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        coarse, single, multi = self(feats)
        target = torch.cat([s.gt_label for s in data_samples]).to(coarse.device)
        ls = self.label_smoothing

        coarse_t = self.is_multi[target].long()
        losses = {'loss_coarse': F.cross_entropy(coarse, coarse_t, label_smoothing=ls)}

        smask = ~self.is_multi[target]
        if smask.any():
            losses['loss_single'] = F.cross_entropy(
                single[smask], self.single_pos[target[smask]], label_smoothing=ls)
        mmask = self.is_multi[target]
        if mmask.any():
            losses['loss_multi'] = F.cross_entropy(
                multi[mmask], self.multi_pos[target[mmask]], label_smoothing=ls)
        return losses

    def predict(self, feats: Tuple[torch.Tensor],
                data_samples: Optional[List[Optional[DataSample]]] = None) -> List[DataSample]:
        coarse, single, multi = self(feats)
        cp = F.softmax(coarse, dim=1)                       # [B, 2]
        sp = F.softmax(single, dim=1)                       # [B, n_single]
        mp = F.softmax(multi, dim=1)                        # [B, n_multi]
        b = coarse.size(0)
        score = coarse.new_zeros(b, self.num_classes)       # proper 57-way joint
        score[:, self.single_ids] = cp[:, 0:1] * sp
        score[:, self.multi_ids] = cp[:, 1:2] * mp
        labels = score.argmax(dim=1, keepdim=True).detach()

        if data_samples is None:
            data_samples = [None] * b
        out = []
        for ds, sc, lb in zip(data_samples, score, labels):
            ds = ds if ds is not None else DataSample()
            ds.set_pred_score(sc).set_pred_label(lb)
            out.append(ds)
        return out

    def diagnose(self, feats: Tuple[torch.Tensor],
                 data_samples: List[Optional[DataSample]], **kwargs) -> List[DataSample]:
        """Validation-time forward (CSRR ``val_step`` convention): per-sample
        loss + prediction packed into each data sample, mirroring ClsHead."""
        coarse, single, multi = self(feats)
        cp = F.softmax(coarse, dim=1)
        sp = F.softmax(single, dim=1)
        mp = F.softmax(multi, dim=1)
        b = coarse.size(0)
        score = coarse.new_zeros(b, self.num_classes)       # proper 57-way joint
        score[:, self.single_ids] = cp[:, 0:1] * sp
        score[:, self.multi_ids] = cp[:, 1:2] * mp
        labels = score.argmax(dim=1, keepdim=True).detach()

        # per-sample NLL of the joint hierarchical distribution
        target = torch.cat([ds.gt_label for ds in data_samples])
        losses = F.nll_loss(torch.log(score.clamp_min(1e-12)), target,
                            reduction='none').reshape(-1, 1)

        out = []
        for ds, loss, sc, lb in zip(data_samples, losses, score, labels):
            if ds is None:
                ds = DataSample()
            ds.set_loss(loss, 'classification_loss').set_pred_score(sc).set_pred_label(lb)
            out.append(ds)
        return out
