# Copyright (c) Shuo Chang. All Rights Reserved.
"""YOLO-style 1-D detection head of the JDM framework.

Reference: H. Xing et al., "Joint Signal Detection and Automatic Modulation
Classification via Deep Learning", IEEE TWC, 2024, Sec. V-B.

The frequency axis (``frame_length`` FFT bins) is divided into ``G`` grid
cells of ``stride`` bins; each cell holds ``len(anchor_widths)`` anchors of
different base bandwidths. Every anchor predicts three attributes
(paper: "each bounding box corresponds to three predictions"):

- ``t_c`` — within-cell center-frequency offset, squashed by a sigmoid;
- ``t_w`` — log-scale bandwidth relative to the anchor width;
- ``t_o`` — objectness/confidence logit.

The signal boxes are 1-D frequency intervals ``(left, right)``; the time axis
is always fully overlapped, so IoU is the 1-D interval IoU
(:func:`csrr.models.utils.interval_iou`). The head is class-agnostic —
modulation classification is done by the second JDM stage.

Target assignment (YOLOv3 convention, which the historical
``SignalDetectionHead`` approximated through mmdet's ``GridAssigner``):

- the grid cell containing a ground-truth center is "responsible"; among its
  anchors the one with the highest IoU against that ground truth is positive;
- non-positive anchors whose best IoU against any ground truth exceeds
  ``ignore_iou_thr`` are ignored by the confidence loss;
- all remaining anchors are negative (confidence target 0).
"""
from typing import List, Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from csrr.structures import DataSample
from ..builder import HEADS, build_loss
from ..utils import interval_iou, interval_nms


@HEADS.register_module()
class JDMDetectionHead(BaseModule):
    """Signal detection head with 1-D anchors.

    Args:
        in_channels (int): channels of the backbone feature map.
        frame_length (int): number of FFT bins of the input frame.
        stride (int): FFT bins per grid cell. ``frame_length`` must be
            divisible by ``stride`` and match the backbone's downsampling.
        anchor_widths (Sequence[float]): base bandwidths (in bins) of the
            anchors of each cell. The defaults straddle the three bandwidth
            clusters of the CSRD/CRML23 data (~96/120/146 bins).
        ignore_iou_thr (float): IoU above which a non-positive anchor is
            excluded from the negative confidence loss.
        loss_conf / loss_cf / loss_bw (dict): loss configs for confidence,
            center offset and log-bandwidth (paper/historical recipe:
            BCE / BCE / MSE with weight 2).
        test_cfg (dict): ``score_thr``, ``nms_iou_thr`` and ``max_per_frame``
            used by :meth:`predict`.
    """

    def __init__(self,
                 in_channels: int = 256,
                 frame_length: int = 1200,
                 stride: int = 8,
                 anchor_widths=(100.0, 120.0, 140.0),
                 ignore_iou_thr: float = 0.5,
                 loss_conf: Optional[dict] = None,
                 loss_cf: Optional[dict] = None,
                 loss_bw: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        assert frame_length % stride == 0, \
            'frame_length must be divisible by the grid stride'
        self.frame_length = frame_length
        self.stride = stride
        self.num_cells = frame_length // stride
        self.anchor_widths = tuple(float(w) for w in anchor_widths)
        self.num_anchors = len(self.anchor_widths)
        self.num_attrib = 3  # (t_c, t_w, t_o)
        self.ignore_iou_thr = ignore_iou_thr

        loss_conf = loss_conf or dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        loss_cf = loss_cf or dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        loss_bw = loss_bw or dict(type='MSELoss', loss_weight=2.0)
        self.loss_conf = build_loss(loss_conf)
        self.loss_cf = build_loss(loss_cf)
        self.loss_bw = build_loss(loss_bw)

        self.test_cfg = dict(score_thr=0.05, nms_iou_thr=0.45, max_per_frame=20)
        if test_cfg is not None:
            self.test_cfg.update(test_cfg)

        self.conv_pred = nn.Conv1d(
            in_channels, self.num_anchors * self.num_attrib, kernel_size=1)

        # anchors as (left, right) intervals, shape (num_cells*num_anchors, 2)
        centers = (torch.arange(self.num_cells, dtype=torch.float32) + 0.5) \
            * stride
        widths = torch.tensor(self.anchor_widths, dtype=torch.float32)
        left = centers[:, None] - widths[None, :] / 2
        right = centers[:, None] + widths[None, :] / 2
        anchors = torch.stack([left, right], dim=-1).reshape(-1, 2)
        self.register_buffer('anchors', anchors, persistent=False)
        anchor_w = widths[None, :].expand(self.num_cells, -1).reshape(-1)
        self.register_buffer('anchor_w', anchor_w, persistent=False)

    def forward(self, feats) -> torch.Tensor:
        """Return raw prediction map of shape ``(N, num_priors, 3)``.

        ``num_priors = num_cells * num_anchors``; the last dim holds
        ``(t_c, t_w, t_o)`` logits (anchor-major within each cell).
        """
        x = feats[-1] if isinstance(feats, (tuple, list)) else feats
        pred = self.conv_pred(x)  # (N, A*3, G)
        n, _, g = pred.shape
        assert g == self.num_cells, (
            f'feature grid ({g}) does not match head geometry '
            f'({self.num_cells}); check backbone stride/frame_length')
        # (N, A*3, G) -> (N, G, A, 3) -> (N, G*A, 3)
        pred = pred.permute(0, 2, 1).reshape(n, g, self.num_anchors,
                                             self.num_attrib)
        return pred.reshape(n, g * self.num_anchors, self.num_attrib)

    # --------------------------------------------------------------- train
    def loss(self, feats, data_samples: List[DataSample]) -> dict:
        """Compute detection losses from backbone features."""
        pred = self(feats)  # (N, P, 3)
        num_priors = pred.size(1)

        target_cf = pred.new_zeros(pred.shape[0], num_priors)
        target_bw = pred.new_zeros(pred.shape[0], num_priors)
        target_conf = pred.new_zeros(pred.shape[0], num_priors)
        pos_mask = torch.zeros_like(target_conf, dtype=torch.bool)
        neg_mask = torch.ones_like(pos_mask)

        for i, sample in enumerate(data_samples):
            gt = sample.gt_boxes.to(pred.device).float()
            if gt.numel() == 0:
                continue
            pos_inds, ignore = self._assign(gt)
            neg_mask[i, ignore] = False
            neg_mask[i, pos_inds] = False
            pos_mask[i, pos_inds] = True
            centers = (gt[:, 0] + gt[:, 1]) / 2
            widths = (gt[:, 1] - gt[:, 0]).clamp(min=1e-6)
            cells = pos_inds // self.num_anchors
            target_cf[i, pos_inds] = \
                (centers / self.stride - cells.float()).clamp(1e-6, 1 - 1e-6)
            target_bw[i, pos_inds] = torch.log(widths /
                                               self.anchor_w[pos_inds])
            target_conf[i, pos_inds] = 1.0

        num_pos = max(int(pos_mask.sum()), 1)
        conf_mask = pos_mask | neg_mask
        losses = dict(
            loss_conf=self.loss_conf(
                pred[..., 2][conf_mask], target_conf[conf_mask],
                avg_factor=num_pos),
            loss_cf=self.loss_cf(
                pred[..., 0][pos_mask], target_cf[pos_mask],
                avg_factor=num_pos),
            loss_bw=self.loss_bw(
                pred[..., 1][pos_mask], target_bw[pos_mask]),
        )
        return losses

    def _assign(self, gt: torch.Tensor):
        """Assign ground truths to anchors of a single sample.

        Returns:
            tuple(Tensor, Tensor): indices of positive anchors (one per GT)
            and boolean mask of anchors to ignore in the confidence loss.
        """
        ious = interval_iou(self.anchors, gt)  # (P, num_gt)
        ignore = ious.max(dim=1).values > self.ignore_iou_thr

        centers = (gt[:, 0] + gt[:, 1]) / 2
        cells = (centers / self.stride).long().clamp(0, self.num_cells - 1)
        pos_inds = []
        for j in range(gt.size(0)):
            cell_anchor_inds = cells[j] * self.num_anchors + torch.arange(
                self.num_anchors, device=gt.device)
            best = ious[cell_anchor_inds, j].argmax()
            pos_inds.append(cell_anchor_inds[best])
        return torch.stack(pos_inds), ignore

    # ---------------------------------------------------------------- test
    def predict(self, feats,
                data_samples: Optional[List[DataSample]] = None
                ) -> List[DataSample]:
        """Decode predictions, run 1-D NMS and write results into samples.

        Each returned :class:`DataSample` carries ``pred_boxes`` (K, 2),
        ``pred_box_scores`` (K,) sorted by descending confidence.
        """
        pred = self(feats)
        boxes, scores = self.decode(pred)

        score_thr = self.test_cfg['score_thr']
        nms_iou_thr = self.test_cfg['nms_iou_thr']
        max_per_frame = self.test_cfg['max_per_frame']

        if data_samples is None:
            data_samples = [DataSample() for _ in range(pred.size(0))]

        for i, sample in enumerate(data_samples):
            keep = scores[i] > score_thr
            sample_boxes, sample_scores = boxes[i][keep], scores[i][keep]
            if sample_boxes.numel() > 0:
                inds = interval_nms(sample_boxes, sample_scores,
                                    iou_threshold=nms_iou_thr,
                                    max_num=max_per_frame)
                sample_boxes, sample_scores = sample_boxes[inds], \
                    sample_scores[inds]
            sample.set_field(sample_boxes, 'pred_boxes')
            sample.set_field(sample_scores, 'pred_box_scores')
        return data_samples

    def decode(self, pred: torch.Tensor):
        """Decode raw predictions to intervals and confidence scores.

        Args:
            pred (Tensor): raw map of shape (N, P, 3) from :meth:`forward`.

        Returns:
            tuple(Tensor, Tensor): intervals (N, P, 2) clamped to the frame
            and confidences (N, P).
        """
        cells = torch.arange(
            self.num_cells, device=pred.device,
            dtype=pred.dtype).repeat_interleave(self.num_anchors)
        centers = (cells + pred[..., 0].sigmoid()) * self.stride
        widths = self.anchor_w * pred[..., 1].exp()
        boxes = torch.stack([centers - widths / 2, centers + widths / 2],
                            dim=-1)
        boxes = boxes.clamp(min=0, max=self.frame_length)
        return boxes, pred[..., 2].sigmoid()
