# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def time_frequency_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU for boxes [t_start, t_end, f_center, bandwidth]."""

    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    a_t0 = torch.minimum(boxes_a[:, 0], boxes_a[:, 1]).clamp(0.0, 1.0)
    a_t1 = torch.maximum(boxes_a[:, 0], boxes_a[:, 1]).clamp(0.0, 1.0)
    a_f0 = (boxes_a[:, 2] - 0.5 * boxes_a[:, 3].abs()).clamp(-0.5, 0.5)
    a_f1 = (boxes_a[:, 2] + 0.5 * boxes_a[:, 3].abs()).clamp(-0.5, 0.5)

    b_t0 = torch.minimum(boxes_b[:, 0], boxes_b[:, 1]).clamp(0.0, 1.0)
    b_t1 = torch.maximum(boxes_b[:, 0], boxes_b[:, 1]).clamp(0.0, 1.0)
    b_f0 = (boxes_b[:, 2] - 0.5 * boxes_b[:, 3].abs()).clamp(-0.5, 0.5)
    b_f1 = (boxes_b[:, 2] + 0.5 * boxes_b[:, 3].abs()).clamp(-0.5, 0.5)

    inter_t = (torch.minimum(a_t1[:, None], b_t1[None, :]) - torch.maximum(a_t0[:, None], b_t0[None, :])).clamp_min(0.0)
    inter_f = (torch.minimum(a_f1[:, None], b_f1[None, :]) - torch.maximum(a_f0[:, None], b_f0[None, :])).clamp_min(0.0)
    inter = inter_t * inter_f
    area_a = ((a_t1 - a_t0) * (a_f1 - a_f0)).clamp_min(0.0)
    area_b = ((b_t1 - b_t0) * (b_f1 - b_f0)).clamp_min(0.0)
    union = (area_a[:, None] + area_b[None, :] - inter).clamp_min(torch.finfo(boxes_a.dtype).eps)
    return inter / union


def aligned_time_frequency_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Elementwise IoU for aligned box tensors shaped [N, 4].

    This is equivalent to ``time_frequency_iou(a, b).diag()`` but avoids the
    quadratic memory cost, which matters for dense time-frequency lattices.
    """

    if boxes_a.shape != boxes_b.shape:
        raise ValueError(f"aligned_time_frequency_iou expects matching shapes, got {tuple(boxes_a.shape)} and {tuple(boxes_b.shape)}")
    if boxes_a.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0],))

    a_t0 = torch.minimum(boxes_a[:, 0], boxes_a[:, 1]).clamp(0.0, 1.0)
    a_t1 = torch.maximum(boxes_a[:, 0], boxes_a[:, 1]).clamp(0.0, 1.0)
    a_f0 = (boxes_a[:, 2] - 0.5 * boxes_a[:, 3].abs()).clamp(-0.5, 0.5)
    a_f1 = (boxes_a[:, 2] + 0.5 * boxes_a[:, 3].abs()).clamp(-0.5, 0.5)

    b_t0 = torch.minimum(boxes_b[:, 0], boxes_b[:, 1]).clamp(0.0, 1.0)
    b_t1 = torch.maximum(boxes_b[:, 0], boxes_b[:, 1]).clamp(0.0, 1.0)
    b_f0 = (boxes_b[:, 2] - 0.5 * boxes_b[:, 3].abs()).clamp(-0.5, 0.5)
    b_f1 = (boxes_b[:, 2] + 0.5 * boxes_b[:, 3].abs()).clamp(-0.5, 0.5)

    inter_t = (torch.minimum(a_t1, b_t1) - torch.maximum(a_t0, b_t0)).clamp_min(0.0)
    inter_f = (torch.minimum(a_f1, b_f1) - torch.maximum(a_f0, b_f0)).clamp_min(0.0)
    inter = inter_t * inter_f
    area_a = ((a_t1 - a_t0) * (a_f1 - a_f0)).clamp_min(0.0)
    area_b = ((b_t1 - b_t0) * (b_f1 - b_f0)).clamp_min(0.0)
    union = (area_a + area_b - inter).clamp_min(torch.finfo(boxes_a.dtype).eps)
    return inter / union


def greedy_match(pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float = 0.5) -> list[tuple[int, int, float]]:
    ious = time_frequency_iou(pred_boxes, gt_boxes)
    matches: list[tuple[int, int, float]] = []
    if ious.numel() == 0:
        return matches

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    flat_order = torch.argsort(ious.flatten(), descending=True)
    num_gt = gt_boxes.shape[0]
    for flat_idx in flat_order.tolist():
        pred_idx = flat_idx // num_gt
        gt_idx = flat_idx % num_gt
        iou = float(ious[pred_idx, gt_idx])
        if iou < iou_threshold:
            break
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, iou))
    return matches


@dataclass
class DetectionSummary:
    precision: float
    recall: float
    mean_iou: float
    num_predictions: int
    num_targets: int
    num_matches: int


def detection_summary(pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float = 0.5) -> DetectionSummary:
    matches = greedy_match(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
    num_predictions = int(pred_boxes.shape[0])
    num_targets = int(gt_boxes.shape[0])
    num_matches = len(matches)
    mean_iou = sum(match[2] for match in matches) / max(num_matches, 1)
    return DetectionSummary(
        precision=num_matches / max(num_predictions, 1),
        recall=num_matches / max(num_targets, 1),
        mean_iou=mean_iou,
        num_predictions=num_predictions,
        num_targets=num_targets,
        num_matches=num_matches,
    )


def average_precision_at_iou(
    pred_boxes_by_image: list[Tensor],
    pred_scores_by_image: list[Tensor],
    gt_boxes_by_image: list[Tensor],
    iou_threshold: float,
) -> float:
    """Class-agnostic AP for normalized time-frequency boxes."""

    return _average_precision_by_threshold(
        pred_boxes_by_image,
        pred_scores_by_image,
        gt_boxes_by_image,
        [float(iou_threshold)],
    )[float(iou_threshold)]


def _average_precision_by_threshold(
    pred_boxes_by_image: list[Tensor],
    pred_scores_by_image: list[Tensor],
    gt_boxes_by_image: list[Tensor],
    thresholds: list[float],
) -> dict[float, float]:
    """Compute AP at multiple IoU thresholds with shared sorting and IoU matrices."""

    pred_boxes_cpu = [boxes.detach().cpu() for boxes in pred_boxes_by_image]
    pred_scores_cpu = [scores.detach().cpu() for scores in pred_scores_by_image]
    gt_boxes_cpu = [boxes.detach().cpu() for boxes in gt_boxes_by_image]
    total_gt = sum(int(boxes.shape[0]) for boxes in gt_boxes_cpu)
    if total_gt <= 0:
        return {threshold: 0.0 for threshold in thresholds}

    detections: list[tuple[float, int, int]] = []
    ious_by_image: list[Tensor] = []
    for boxes, gt_boxes in zip(pred_boxes_cpu, gt_boxes_cpu):
        if boxes.numel() == 0 or gt_boxes.numel() == 0:
            ious_by_image.append(boxes.new_zeros((boxes.shape[0], gt_boxes.shape[0])))
        else:
            ious_by_image.append(time_frequency_iou(boxes, gt_boxes))
    for image_idx, (boxes, scores) in enumerate(zip(pred_boxes_cpu, pred_scores_cpu)):
        if boxes.numel() == 0 or scores.numel() == 0:
            continue
        for pred_idx, score in enumerate(scores.tolist()):
            detections.append((float(score), image_idx, pred_idx))
    if not detections:
        return {threshold: 0.0 for threshold in thresholds}

    detections.sort(key=lambda item: item[0], reverse=True)
    recall_grid = torch.linspace(0.0, 1.0, 101)
    ap_by_threshold: dict[float, float] = {}
    for threshold in thresholds:
        matched_gt: dict[int, set[int]] = {idx: set() for idx in range(len(gt_boxes_cpu))}
        true_positive = torch.zeros(len(detections), dtype=torch.float32)
        false_positive = torch.zeros(len(detections), dtype=torch.float32)

        for det_idx, (_, image_idx, pred_idx) in enumerate(detections):
            gt_boxes = gt_boxes_cpu[image_idx]
            if gt_boxes.numel() == 0:
                false_positive[det_idx] = 1.0
                continue

            ious = ious_by_image[image_idx][pred_idx]
            best_iou, best_gt = torch.max(ious, dim=0)
            best_gt_idx = int(best_gt.item())
            if float(best_iou.item()) >= threshold and best_gt_idx not in matched_gt[image_idx]:
                true_positive[det_idx] = 1.0
                matched_gt[image_idx].add(best_gt_idx)
            else:
                false_positive[det_idx] = 1.0

        cum_tp = true_positive.cumsum(dim=0)
        cum_fp = false_positive.cumsum(dim=0)
        recall = cum_tp / max(float(total_gt), 1.0)
        precision = cum_tp / (cum_tp + cum_fp).clamp_min(torch.finfo(torch.float32).eps)
        interpolated = []
        for recall_level in recall_grid:
            valid = precision[recall >= recall_level]
            interpolated.append(valid.max() if valid.numel() else torch.tensor(0.0))
        ap_by_threshold[threshold] = float(torch.stack(interpolated).mean().item())
    return ap_by_threshold


def detection_map(
    pred_boxes_by_image: list[Tensor],
    pred_scores_by_image: list[Tensor],
    gt_boxes_by_image: list[Tensor],
    iou_thresholds: Tensor | None = None,
) -> dict[str, float]:
    """COCO-style class-agnostic mAP for IQDet time-frequency detections."""

    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    thresholds = [round(float(value), 2) for value in iou_thresholds.detach().cpu().tolist()]
    ap_by_threshold = _average_precision_by_threshold(
        pred_boxes_by_image,
        pred_scores_by_image,
        gt_boxes_by_image,
        thresholds,
    )
    return {
        "bbox_mAP": float(sum(ap_by_threshold.values()) / max(len(ap_by_threshold), 1)),
        "bbox_mAP_50": float(ap_by_threshold.get(0.5, 0.0)),
        "bbox_mAP_75": float(ap_by_threshold.get(0.75, 0.0)),
    }


def class_aware_detection_map(
    pred_boxes_by_image: list[Tensor],
    pred_scores_by_image: list[Tensor],
    pred_labels_by_image: list[Tensor],
    gt_boxes_by_image: list[Tensor],
    gt_labels_by_image: list[Tensor],
    *,
    num_classes: int,
    iou_thresholds: Tensor | None = None,
) -> dict[str, float]:
    """COCO-style AP averaged over classes with at least one GT instance."""

    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    thresholds = [round(float(value), 2) for value in iou_thresholds.detach().cpu().tolist()]
    class_metrics: dict[float, list[float]] = {threshold: [] for threshold in thresholds}

    for class_idx in range(num_classes):
        gt_count = sum(int((labels == class_idx).sum().item()) for labels in gt_labels_by_image)
        if gt_count <= 0:
            continue
        class_pred_boxes = []
        class_pred_scores = []
        class_gt_boxes = []
        for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
            pred_boxes_by_image,
            pred_scores_by_image,
            pred_labels_by_image,
            gt_boxes_by_image,
            gt_labels_by_image,
        ):
            pred_mask = pred_labels == class_idx
            gt_mask = gt_labels == class_idx
            class_pred_boxes.append(pred_boxes[pred_mask])
            class_pred_scores.append(pred_scores[pred_mask])
            class_gt_boxes.append(gt_boxes[gt_mask])
        ap_by_threshold = _average_precision_by_threshold(
            class_pred_boxes,
            class_pred_scores,
            class_gt_boxes,
            thresholds,
        )
        for threshold in thresholds:
            class_metrics[threshold].append(ap_by_threshold[threshold])

    ap_by_threshold = {
        threshold: float(sum(values) / max(len(values), 1)) if values else 0.0
        for threshold, values in class_metrics.items()
    }
    return {
        "class_bbox_mAP": float(sum(ap_by_threshold.values()) / max(len(ap_by_threshold), 1)),
        "class_bbox_mAP_50": float(ap_by_threshold.get(0.5, 0.0)),
        "class_bbox_mAP_75": float(ap_by_threshold.get(0.75, 0.0)),
    }


def classification_accuracy(logits: Tensor, target: Tensor) -> float:
    if logits.numel() == 0 or target.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return float((pred == target.to(pred.device)).float().mean().detach().cpu())


def masked_rmse(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    if pred.numel() == 0 or mask.sum().item() <= 0:
        return 0.0
    error = (pred - target.to(pred.device)).pow(2) * mask.to(pred.device)
    return float((error.sum() / mask.to(pred.device).sum().clamp_min(1.0)).sqrt().detach().cpu())
