# Copyright (c) Shuo Chang. All Rights Reserved.
"""COCO-style evaluation over 1-D frequency intervals for signal detection.

Implements the metric suite of the JDM paper (Sec. VI-A) without pycocotools:
mAP@[.5:.95], AP@.50, AP@.75, bandwidth-binned AP (small/medium/large) and the
matching average-recall numbers, all computed with the 1-D interval IoU
(a signal always spans the full time axis, so boxes only differ along the
frequency axis).
"""
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

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


def _match_sample_with_ignore(pred_boxes: np.ndarray,
                              pred_scores: np.ndarray,
                              gt_boxes: np.ndarray,
                              ignore_gt_boxes: np.ndarray,
                              iou_thr: float
                              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Greedy TP matching plus ignored detections for non-target GT boxes."""
    tp, num_gt = _match_sample(pred_boxes, pred_scores, gt_boxes, iou_thr)
    keep = np.ones(pred_boxes.shape[0], dtype=bool)
    if ignore_gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return tp, keep, num_gt
    ious = interval_iou_numpy(pred_boxes, ignore_gt_boxes)
    # Detections already matched to target-SNR positives remain valid TPs.
    keep = tp | (ious.max(axis=1) < iou_thr)
    return tp, keep, num_gt


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
        snrwise (bool): if True, also compute AP/AR curves grouped by each
            ground-truth signal's own SNR value. This requires a per-box SNR
            array in the data sample metainfo, parallel to ``gt_boxes``.
        snr_key (str): data sample key/metainfo key containing per-box SNRs.
        snr_curve_out (str | None): optional JSON output path for the compact
            SNR curve payload.
        snr_plot_out (str | None): optional image/PDF output path for the SNR
            mAP curve.
        snr_plot_title (str | None): optional plot title.
        snrwise_metrics (Sequence[str]): SNR-curve metrics to compute. Allowed
            names are ``mAP``, ``AP50``, ``AP75`` and ``AR``.
    """
    default_prefix: Optional[str] = 'detection'

    def __init__(self,
                 iou_thrs: Sequence[float] = DEFAULT_IOU_THRS,
                 size_ranges: Optional[dict] = None,
                 max_detections: Sequence[int] = (4, 5, 6),
                 classwise: bool = False,
                 snrwise: bool = False,
                 snr_key: str = 'snr',
                 snr_curve_out: Optional[str] = None,
                 snr_plot_out: Optional[str] = None,
                 snr_plot_title: Optional[str] = None,
                 snrwise_metrics: Sequence[str] = ('mAP', 'AP50', 'AP75',
                                                   'AR'),
                 per_iou_ap: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = tuple(iou_thrs)
        self.size_ranges = DEFAULT_SIZE_RANGES if size_ranges is None \
            else size_ranges
        self.max_detections = tuple(max_detections)
        self.classwise = classwise
        # When True, also emit AP at each individual IoU threshold
        # (AP_iou_0.50 .. AP_iou_0.95). Diagnostic only; localizes where the
        # mAP gap to the paper radar sits (paper Fig. 8 ideal mAP 0.91 vs our
        # 0.80 is concentrated in the high-IoU / box-tightness regime).
        self.per_iou_ap = per_iou_ap
        self.snrwise = snrwise
        self.snr_key = snr_key
        self.snr_curve_out = snr_curve_out
        self.snr_plot_out = snr_plot_out
        self.snr_plot_title = snr_plot_title
        allowed_snr_metrics = {'mAP', 'AP50', 'AP75', 'AR'}
        self.snrwise_metrics = tuple(snrwise_metrics)
        unknown = set(self.snrwise_metrics) - allowed_snr_metrics
        if unknown:
            raise ValueError(
                f'Unsupported SNR-wise metrics: {sorted(unknown)}')

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _get_sample_value(data_sample, key: str):
        if isinstance(data_sample, dict):
            return data_sample.get(key, None)
        if hasattr(data_sample, 'get'):
            value = data_sample.get(key, None)
            if value is not None:
                return value
        try:
            return data_sample[key]
        except Exception:
            return getattr(data_sample, key, None)

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict(
                pred_boxes=self._to_numpy(
                    data_sample['pred_boxes']).reshape(-1, 2),
                pred_scores=self._to_numpy(
                    data_sample['pred_box_scores']).reshape(-1),
                gt_boxes=self._to_numpy(
                    data_sample['gt_boxes']).reshape(-1, 2))
            snr = self._get_sample_value(data_sample, self.snr_key)
            if snr is not None:
                gt_snr = self._to_numpy(snr).reshape(-1)
                num_gt = result['gt_boxes'].shape[0]
                if gt_snr.shape[0] != num_gt:
                    raise ValueError(
                        f'Expected per-box "{self.snr_key}" with {num_gt} '
                        f'values, got shape {gt_snr.shape}. Frame-level SNR '
                        'cannot be used for SNR-wise detection curves.')
                result['gt_snr'] = gt_snr
            elif self.snrwise:
                raise KeyError(
                    f'SNR-wise detection evaluation requires "{self.snr_key}" '
                    'in each data sample.')
            if self.classwise:
                result['pred_labels'] = self._to_numpy(
                    data_sample['pred_box_labels']).reshape(-1)
                result['gt_labels'] = self._to_numpy(
                    data_sample['gt_box_labels']).reshape(-1)
            self.results.append(result)

    # ------------------------------------------------------------------
    def compute_metrics(self, results: List[dict]) -> dict:
        groups, class_ids = self._build_groups(results)

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

        if self.per_iou_ap:
            for iou_thr in self.iou_thrs:
                metrics[f'AP_iou_{iou_thr:.2f}'] = self._mean_over_groups(
                    groups, self._ap_at, (iou_thr,))

        if self.snrwise:
            snr_metrics, snr_curve = self._compute_snrwise(
                results, class_ids)
            metrics.update(snr_metrics)
            self._dump_snr_curve(snr_curve)

        return {key: round(value, 4) for key, value in metrics.items()}

    def _build_groups(self,
                      results: List[dict],
                      snr=None,
                      class_ids: Optional[Sequence[int]] = None):
        if self.classwise:
            if class_ids is None:
                class_ids = sorted(
                    {int(c) for res in results for c in res['gt_labels']})
            groups = [
                [
                    self._select_snr(class_res, snr)
                    if snr is not None else class_res
                    for class_res in [
                        self._select_class(res, cid) for res in results
                    ]
                ]
                for cid in class_ids
            ]
        else:
            class_ids = []
            if snr is not None:
                results = [self._select_snr(res, snr) for res in results]
            groups = [results]
        return groups, tuple(class_ids)

    @staticmethod
    def _select_class(result: dict, class_id: int) -> dict:
        pred_keep = result['pred_labels'] == class_id
        gt_keep = result['gt_labels'] == class_id
        selected = dict(
            pred_boxes=result['pred_boxes'][pred_keep],
            pred_scores=result['pred_scores'][pred_keep],
            gt_boxes=result['gt_boxes'][gt_keep])
        if 'gt_snr' in result:
            selected['gt_snr'] = result['gt_snr'][gt_keep]
        return selected

    @staticmethod
    def _select_snr(result: dict, snr) -> dict:
        gt_keep = result['gt_snr'] == snr
        selected = dict(
            pred_boxes=result['pred_boxes'],
            pred_scores=result['pred_scores'],
            gt_boxes=result['gt_boxes'][gt_keep],
            ignore_gt_boxes=result['gt_boxes'][~gt_keep])
        if 'pred_labels' in result:
            selected['pred_labels'] = result['pred_labels']
        if 'gt_labels' in result:
            selected['gt_labels'] = result['gt_labels'][gt_keep]
        selected['gt_snr'] = result['gt_snr'][gt_keep]
        return selected

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

    def _compute_snrwise(self, results: List[dict], class_ids):
        snr_values = sorted(
            np.unique(np.concatenate([res['gt_snr'] for res in results])),
            key=self._snr_sort_key)
        scalar_metrics = {}
        curve = []
        for snr in snr_values:
            groups, _ = self._build_groups(results, snr, class_ids)
            values = {}
            if 'mAP' in self.snrwise_metrics:
                values['mAP'] = self._mean_over_groups(
                    groups, self._ap_at_with_ignore, self.iou_thrs)
            if 'AP50' in self.snrwise_metrics:
                values['AP50'] = self._mean_over_groups(
                    groups, self._ap_at_with_ignore, (0.5,))
            if 'AP75' in self.snrwise_metrics:
                values['AP75'] = self._mean_over_groups(
                    groups, self._ap_at_with_ignore, (0.75,))
            if 'AR' in self.snrwise_metrics:
                values['AR'] = self._mean_over_groups(
                    groups, self._ar_at_with_ignore, self.iou_thrs, None)
            label = self._format_snr_for_key(snr)
            if 'mAP' in values:
                scalar_metrics[f'mAP_snr_{label}'] = values['mAP']
            if 'AR' in values:
                scalar_metrics[f'AR_snr_{label}'] = values['AR']
            curve.append({
                'snr': self._to_json_scalar(snr),
                'num_gt': self._count_gt_at_snr(results, snr),
                **{
                    key: self._to_json_metric(value)
                    for key, value in values.items()
                },
            })
        return scalar_metrics, curve

    def _ap_at_with_ignore(self, results: List[dict], iou_thrs) -> float:
        aps = []
        for iou_thr in iou_thrs:
            tps, scores, num_gt = [], [], 0
            for result in results:
                pred_boxes, pred_scores, gt_boxes = self._prepare(result)
                ignore_gt_boxes = result.get(
                    'ignore_gt_boxes',
                    np.zeros((0, 2), dtype=gt_boxes.dtype))
                tp, keep, n = _match_sample_with_ignore(
                    pred_boxes, pred_scores, gt_boxes, ignore_gt_boxes,
                    iou_thr)
                tps.append(tp[keep])
                scores.append(pred_scores[keep])
                num_gt += n
            ap = _average_precision(
                np.concatenate(tps) if tps else np.zeros(0, dtype=bool),
                np.concatenate(scores) if scores else np.zeros(0),
                num_gt)
            if not np.isnan(ap):
                aps.append(ap)
        return float(np.mean(aps)) if aps else float('nan')

    def _ar_at_with_ignore(self, results: List[dict], iou_thrs,
                           max_dets=None) -> float:
        recalls = []
        for iou_thr in iou_thrs:
            matched, num_gt = 0, 0
            for result in results:
                pred_boxes, pred_scores, gt_boxes = self._prepare(
                    result, max_dets=max_dets)
                ignore_gt_boxes = result.get(
                    'ignore_gt_boxes',
                    np.zeros((0, 2), dtype=gt_boxes.dtype))
                tp, _, n = _match_sample_with_ignore(
                    pred_boxes, pred_scores, gt_boxes, ignore_gt_boxes,
                    iou_thr)
                matched += int(tp.sum())
                num_gt += n
            if num_gt > 0:
                recalls.append(matched / num_gt)
        return float(np.mean(recalls)) if recalls else float('nan')

    @staticmethod
    def _count_gt_at_snr(results: List[dict], snr) -> int:
        return int(sum(np.count_nonzero(res['gt_snr'] == snr)
                       for res in results))

    @staticmethod
    def _format_snr_for_key(snr) -> str:
        parsed = SignalDetectionMetric._parse_snr_number(snr)
        if parsed is not None:
            if parsed.is_integer():
                return str(int(parsed))
            return str(parsed).replace('.', 'p')
        return str(SignalDetectionMetric._to_json_scalar(snr)).replace(
            ' ', '_').replace('/', '_')

    @staticmethod
    def _snr_sort_key(snr):
        parsed = SignalDetectionMetric._parse_snr_number(snr)
        if parsed is not None:
            return (0, parsed, '')
        label = str(SignalDetectionMetric._to_json_scalar(snr))
        if label.lower() in ('infdb', '+infdb', 'inf', '+inf',
                             'infinity', '+infinity'):
            return (1, float('inf'), label)
        return (2, 0.0, label)

    @staticmethod
    def _parse_snr_number(snr):
        value = SignalDetectionMetric._to_json_scalar(snr)
        if isinstance(value, str):
            value = value.strip()
            if value.lower().endswith('db'):
                value = value[:-2]
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isfinite(parsed):
            return parsed
        return None

    @staticmethod
    def _to_json_scalar(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _to_json_metric(value):
        if np.isnan(value):
            return None
        return round(float(value), 4)

    def _dump_snr_curve(self, curve: List[dict]) -> None:
        if not self.snr_curve_out and not self.snr_plot_out:
            return

        payload = dict(
            metric='SignalDetectionMetric',
            classwise=self.classwise,
            snr_key=self.snr_key,
            snrwise_metrics=list(self.snrwise_metrics),
            iou_thrs=[float(thr) for thr in self.iou_thrs],
            points=curve)

        if self.snr_curve_out:
            self._ensure_parent_dir(self.snr_curve_out)
            with open(self.snr_curve_out, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            MMLogger.get_current_instance().info(
                f'Wrote SNR detection curve to {self.snr_curve_out}')

        if self.snr_plot_out:
            self._write_snr_plot(curve)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _write_snr_plot(self, curve: List[dict]) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self._ensure_parent_dir(self.snr_plot_out)
        xs = [point['snr'] for point in curve]
        if any(self._parse_snr_number(x) is None for x in xs):
            xs = [str(x) for x in xs]
        map_values = [np.nan if point['mAP'] is None else point['mAP']
                      for point in curve] if 'mAP' in self.snrwise_metrics \
            else None
        ar_values = [np.nan if point['AR'] is None else point['AR']
                     for point in curve] if 'AR' in self.snrwise_metrics \
            else None

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if map_values is not None:
            ax.plot(xs, map_values, marker='o', label='mAP@[.50:.95]')
        if ar_values is not None:
            ax.plot(xs, ar_values, marker='s', label='AR@[.50:.95]')
        ax.set_xlabel('Per-signal SNR (dB)')
        ax.set_ylabel('Metric')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        if self.snr_plot_title:
            ax.set_title(self.snr_plot_title)
        fig.tight_layout()
        fig.savefig(self.snr_plot_out)
        plt.close(fig)
        MMLogger.get_current_instance().info(
            f'Wrote SNR detection plot to {self.snr_plot_out}')
