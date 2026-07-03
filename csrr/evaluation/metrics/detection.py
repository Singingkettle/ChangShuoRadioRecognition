# Copyright (c) Shuo Chang. All Rights Reserved.
"""COCO-style evaluation over 1-D frequency intervals for signal detection.

Implements the metric suite of the JDM paper (Sec. VI-A) without pycocotools:
mAP@[.5:.95], AP@.50, AP@.75, bandwidth-binned AP (small/medium/large) and the
matching average-recall numbers, all computed with the 1-D interval IoU
(a signal always spans the full time axis, so boxes only differ along the
frequency axis).
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from csrr.registry import METRICS

# bandwidth (FFT-bin) ranges of the paper's small/medium/large buckets
DEFAULT_SIZE_RANGES = dict(
    small=(0.0, 110.0),
    medium=(110.0, 130.0),
    large=(130.0, 150.0 * 1e6),
)
DEFAULT_IOU_THRS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def interval_iou_numpy(intervals1: np.ndarray,
                       intervals2: np.ndarray) -> np.ndarray:
    """Pairwise 1-D interval IoU, shapes (N, 2) x (M, 2) -> (N, M)."""
    lt = np.maximum(intervals1[:, None, 0], intervals2[None, :, 0])
    rb = np.minimum(intervals1[:, None, 1], intervals2[None, :, 1])
    inter = np.clip(rb - lt, 0, None)
    len1 = np.clip(intervals1[:, 1] - intervals1[:, 0], 0, None)
    len2 = np.clip(intervals2[:, 1] - intervals2[:, 0], 0, None)
    union = len1[:, None] + len2[None, :] - inter
    return inter / np.maximum(union, np.finfo(np.float64).eps)


def _match_sample(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                  gt_boxes: np.ndarray, iou_thr: float
                  ) -> Tuple[np.ndarray, int]:
    """Greedy matching of one sample's detections to ground truths.

    Returns per-detection TP flags (detections sorted by score beforehand)
    and the number of ground truths.
    """
    num_gt = gt_boxes.shape[0]
    tp = np.zeros(pred_boxes.shape[0], dtype=bool)
    if num_gt == 0 or pred_boxes.shape[0] == 0:
        return tp, num_gt
    ious = interval_iou_numpy(pred_boxes, gt_boxes)
    taken = np.zeros(num_gt, dtype=bool)
    for det_idx in range(pred_boxes.shape[0]):
        cand = np.where(~taken & (ious[det_idx] >= iou_thr))[0]
        if cand.size:
            best = cand[np.argmax(ious[det_idx, cand])]
            taken[best] = True
            tp[det_idx] = True
    return tp, num_gt


def _average_precision(tp: np.ndarray, scores: np.ndarray,
                       num_gt: int) -> float:
    """COCO-style 101-point interpolated AP from pooled detections."""
    if num_gt == 0:
        return float('nan')
    if tp.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    recall = cum_tp / num_gt
    precision = cum_tp / (np.arange(tp.size) + 1)
    # make precision monotonically decreasing, then sample 101 recall points
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.linspace(0, 1, 101)
    inds = np.searchsorted(recall, recall_points, side='left')
    sampled = np.where(inds < precision.size, precision[np.minimum(
        inds, precision.size - 1)], 0.0)
    return float(sampled.mean())


@METRICS.register_module()
class SignalDetectionMetric(BaseMetric):
    """mAP/AR over 1-D frequency intervals.

    Consumes data samples with ``pred_boxes`` (K, 2), ``pred_box_scores``
    (K,), ``gt_boxes`` (M, 2) and (optionally, for joint JDM evaluation)
    ``pred_box_labels`` / ``gt_box_labels``.

    Args:
        iou_thrs (Sequence[float]): IoU thresholds of the mAP average.
        size_ranges (dict | None): name -> (min_bw, max_bw) buckets in FFT
            bins for size-binned AP/AR (paper: (0,110)/(110,130)/(130,∞)).
        max_detections (Sequence[int]): caps for AR@k (the paper reports
            AR@4/5/6, the dominant per-frame signal counts).
        classwise (bool): if True, evaluate per modulation class using
            ``pred_box_labels``/``gt_box_labels`` and report their mean
            (end-to-end JDM metric); otherwise class-agnostic detection.
    """
    default_prefix: Optional[str] = 'detection'

    def __init__(self,
                 iou_thrs: Sequence[float] = DEFAULT_IOU_THRS,
                 size_ranges: Optional[dict] = None,
                 max_detections: Sequence[int] = (4, 5, 6),
                 classwise: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = tuple(iou_thrs)
        self.size_ranges = DEFAULT_SIZE_RANGES if size_ranges is None \
            else size_ranges
        self.max_detections = tuple(max_detections)
        self.classwise = classwise

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict(
                pred_boxes=self._to_numpy(
                    data_sample['pred_boxes']).reshape(-1, 2),
                pred_scores=self._to_numpy(
                    data_sample['pred_box_scores']).reshape(-1),
                gt_boxes=self._to_numpy(
                    data_sample['gt_boxes']).reshape(-1, 2))
            if self.classwise:
                result['pred_labels'] = self._to_numpy(
                    data_sample['pred_box_labels']).reshape(-1)
                result['gt_labels'] = self._to_numpy(
                    data_sample['gt_box_labels']).reshape(-1)
            self.results.append(result)

    # ------------------------------------------------------------------
    def compute_metrics(self, results: List[dict]) -> dict:
        if self.classwise:
            class_ids = sorted(
                {int(c) for res in results for c in res['gt_labels']})
            groups = [
                [self._select_class(res, cid) for res in results]
                for cid in class_ids
            ]
        else:
            groups = [results]

        metrics = dict(
            mAP=self._mean_over_groups(groups, self._ap_at, self.iou_thrs),
            AP50=self._mean_over_groups(groups, self._ap_at, (0.5,)),
            AP75=self._mean_over_groups(groups, self._ap_at, (0.75,)),
            AR=self._mean_over_groups(groups, self._ar_at, self.iou_thrs,
                                      None),
        )
        for name, size_range in self.size_ranges.items():
            metrics[f'AP_{name}'] = self._mean_over_groups(
                groups, self._ap_at, self.iou_thrs, size_range)
            metrics[f'AR_{name}'] = self._mean_over_groups(
                groups, self._ar_at, self.iou_thrs, None, size_range)
        for k in self.max_detections:
            metrics[f'AR@{k}'] = self._mean_over_groups(
                groups, self._ar_at, self.iou_thrs, k)
        return {key: round(value, 4) for key, value in metrics.items()}

    @staticmethod
    def _select_class(result: dict, class_id: int) -> dict:
        pred_keep = result['pred_labels'] == class_id
        gt_keep = result['gt_labels'] == class_id
        return dict(
            pred_boxes=result['pred_boxes'][pred_keep],
            pred_scores=result['pred_scores'][pred_keep],
            gt_boxes=result['gt_boxes'][gt_keep])

    @staticmethod
    def _filter_size(boxes: np.ndarray, size_range) -> np.ndarray:
        widths = boxes[:, 1] - boxes[:, 0]
        return (widths >= size_range[0]) & (widths < size_range[1])

    def _prepare(self, result: dict, size_range=None, max_dets=None):
        pred_boxes, pred_scores = result['pred_boxes'], result['pred_scores']
        gt_boxes = result['gt_boxes']
        order = np.argsort(-pred_scores)
        pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
        if max_dets is not None:
            pred_boxes, pred_scores = pred_boxes[:max_dets], \
                pred_scores[:max_dets]
        if size_range is not None:
            gt_boxes = gt_boxes[self._filter_size(gt_boxes, size_range)]
            keep = self._filter_size(pred_boxes, size_range)
            pred_boxes, pred_scores = pred_boxes[keep], pred_scores[keep]
        return pred_boxes, pred_scores, gt_boxes

    def _ap_at(self, results: List[dict], iou_thrs, size_range=None) -> float:
        aps = []
        for iou_thr in iou_thrs:
            tps, scores, num_gt = [], [], 0
            for result in results:
                pred_boxes, pred_scores, gt_boxes = self._prepare(
                    result, size_range)
                tp, n = _match_sample(pred_boxes, pred_scores, gt_boxes,
                                      iou_thr)
                tps.append(tp)
                scores.append(pred_scores)
                num_gt += n
            ap = _average_precision(
                np.concatenate(tps) if tps else np.zeros(0, dtype=bool),
                np.concatenate(scores) if scores else np.zeros(0),
                num_gt)
            if not np.isnan(ap):
                aps.append(ap)
        return float(np.mean(aps)) if aps else float('nan')

    def _ar_at(self, results: List[dict], iou_thrs, max_dets=None,
               size_range=None) -> float:
        recalls = []
        for iou_thr in iou_thrs:
            matched, num_gt = 0, 0
            for result in results:
                pred_boxes, pred_scores, gt_boxes = self._prepare(
                    result, size_range, max_dets)
                tp, n = _match_sample(pred_boxes, pred_scores, gt_boxes,
                                      iou_thr)
                matched += int(tp.sum())
                num_gt += n
            if num_gt > 0:
                recalls.append(matched / num_gt)
        return float(np.mean(recalls)) if recalls else float('nan')

    def _mean_over_groups(self, groups, fn, *args) -> float:
        values = [fn(group, *args) for group in groups]
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else float('nan')
