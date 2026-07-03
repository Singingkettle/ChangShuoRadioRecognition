# Copyright (c) Shuo Chang. All Rights Reserved.
"""1-D interval (frequency-band) operations for signal detection.

A detected signal occupies its full extent along the time axis, so a
"bounding box" degenerates to a frequency interval ``(left, right)`` in FFT-bin
units. IoU/NMS therefore operate on 1-D intervals; no 2-D box machinery
(mmdet/mmcv) is required.
"""
from typing import Union

import torch
from torch import Tensor


def interval_iou(intervals1: Tensor, intervals2: Tensor) -> Tensor:
    """Pairwise IoU between two sets of 1-D intervals.

    Args:
        intervals1 (Tensor): shape (N, 2), ``(left, right)`` pairs.
        intervals2 (Tensor): shape (M, 2), ``(left, right)`` pairs.

    Returns:
        Tensor: IoU matrix of shape (N, M).
    """
    lt = torch.max(intervals1[:, None, 0], intervals2[None, :, 0])
    rb = torch.min(intervals1[:, None, 1], intervals2[None, :, 1])
    inter = (rb - lt).clamp(min=0)
    len1 = (intervals1[:, 1] - intervals1[:, 0]).clamp(min=0)
    len2 = (intervals2[:, 1] - intervals2[:, 0]).clamp(min=0)
    union = len1[:, None] + len2[None, :] - inter
    return inter / union.clamp(min=torch.finfo(inter.dtype).eps)


def interval_nms(intervals: Tensor, scores: Tensor,
                 iou_threshold: float = 0.45,
                 max_num: Union[int, None] = None) -> Tensor:
    """Greedy non-maximum suppression on 1-D intervals.

    Args:
        intervals (Tensor): shape (N, 2).
        scores (Tensor): shape (N,).
        iou_threshold (float): overlapping intervals with IoU above this value
            are suppressed.
        max_num (int, optional): maximum number of intervals to keep.

    Returns:
        Tensor: indices of the kept intervals, sorted by descending score.
    """
    if intervals.numel() == 0:
        return intervals.new_zeros((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if max_num is not None and len(keep) >= max_num:
            break
        if order.numel() == 1:
            break
        ious = interval_iou(intervals[i].unsqueeze(0), intervals[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)
