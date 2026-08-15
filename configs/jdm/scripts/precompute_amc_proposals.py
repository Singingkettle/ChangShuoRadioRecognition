# Copyright (c) Shuo Chang. All Rights Reserved.
"""Precompute detector proposal boxes for AMC domain-adaptation training.

For each annotated signal in the CSRD splits, runs the JDM detector on the
received frame and stores the best-IoU proposal interval (FFT bins). The cache
is consumed by :class:`LoadDetProposal` so AMC fine-tuning sees crops filtered
with detector-localized boxes instead of ground-truth intervals.

Example::

    python configs/jdm/scripts/precompute_amc_proposals.py \\
        configs/jdm/jdm-det_fft-csrd.py \\
        work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \\
        --out work_dirs/jdm/amc_proposals/all_splits.json
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from csrr.evaluation.metrics.detection import interval_iou_numpy
from csrr.registry import DATASETS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute detector proposals for AMC training.')
    parser.add_argument('config', help='detector config file')
    parser.add_argument('checkpoint', help='detector checkpoint')
    parser.add_argument(
        '--out',
        default='work_dirs/jdm/amc_proposals/all_splits.json',
        help='output JSON cache path')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=('train', 'validation', 'test'),
        help='dataset splits to process')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--min-iou',
        type=float,
        default=0.0,
        help='fallback to GT box when best proposal IoU is below this value')
    parser.add_argument(
        '--hard-neg-iou-thr',
        type=float,
        default=0.3,
        help='store detector proposals whose max IoU to any GT is below '
        'this value under the ``_unmatched`` cache key for AMC hard-negative '
        'mining')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options')
    return parser.parse_args()


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample_value(sample, key, default=None):
    value = sample.get(key, None) if hasattr(sample, 'get') else None
    if value is not None:
        return value
    try:
        return sample[key]
    except Exception:
        return getattr(sample, key, default)


def _select_dataloader_cfg(cfg, split):
    loader = cfg.train_dataloader if split == 'train' else \
        (cfg.val_dataloader if split == 'validation' else cfg.test_dataloader)
    loader = loader.copy()
    loader.dataset = loader.dataset.copy()
    loader.dataset.type = 'CSRDDetectionDataset'
    loader.dataset.split = split
    return loader


def _build_dataloader(cfg, split, batch_size, num_workers):
    dataloader_cfg = _select_dataloader_cfg(cfg, split)
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.setdefault('collate_fn', dict(type='default_collate'))
    return Runner.build_dataloader(dataloader_cfg)


def _match_proposals(sample, cache, min_iou, hard_neg_iou_thr):
    pred_boxes = _as_numpy(_sample_value(sample, 'pred_boxes')).reshape(-1, 2)
    gt_boxes = _as_numpy(_sample_value(sample, 'gt_boxes')).reshape(-1, 2)
    file_name = _sample_value(sample, 'file_name')

    if pred_boxes.size and gt_boxes.size:
        ious = interval_iou_numpy(gt_boxes, pred_boxes)
        best_pred = ious.argmax(axis=1)
        best_iou = ious[np.arange(gt_boxes.shape[0]), best_pred]
    else:
        best_pred = np.full(gt_boxes.shape[0], -1, dtype=np.int64)
        best_iou = np.zeros(gt_boxes.shape[0], dtype=np.float64)
        ious = None

    frame_cache = cache.setdefault(file_name, {})
    for gt_idx, gt_box in enumerate(gt_boxes):
        if best_pred[gt_idx] >= 0 and best_iou[gt_idx] >= min_iou:
            box = pred_boxes[best_pred[gt_idx]]
        else:
            box = gt_box
        frame_cache[str(gt_idx)] = [float(box[0]), float(box[1])]

    unmatched = []
    if pred_boxes.size:
        if gt_boxes.size:
            max_iou_per_pred = ious.max(axis=0)
        else:
            max_iou_per_pred = np.zeros(pred_boxes.shape[0], dtype=np.float64)
        for pred_idx, pred_box in enumerate(pred_boxes):
            if max_iou_per_pred[pred_idx] < hard_neg_iou_thr:
                unmatched.append([float(pred_box[0]), float(pred_box[1])])
    frame_cache['_unmatched'] = unmatched


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'csrr'))

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith('cuda') else 'cpu')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device)
    model.eval()

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    cache = {}
    stats = dict(
        num_frames=0,
        num_signals=0,
        used_proposal=0,
        used_gt=0,
        num_unmatched=0,
    )

    for split in args.splits:
        dataloader = _build_dataloader(
            cfg, split, args.batch_size, args.num_workers)
        print(f'Processing split={split!r} ({len(dataloader.dataset)} frames)')
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = model.data_preprocessor(data, training=False)
                data['inputs'] = data['inputs'].to(device)
                results = model(
                    data['inputs'], data.get('data_samples'), mode='predict')
                for sample in results:
                    pred_boxes = _as_numpy(
                        _sample_value(sample, 'pred_boxes')).reshape(-1, 2)
                    gt_boxes = _as_numpy(
                        _sample_value(sample, 'gt_boxes')).reshape(-1, 2)
                    _match_proposals(
                        sample, cache, args.min_iou, args.hard_neg_iou_thr)
                    stats['num_frames'] += 1
                    stats['num_signals'] += gt_boxes.shape[0]
                    file_name = _sample_value(sample, 'file_name')
                    stats['num_unmatched'] += len(
                        cache[file_name]['_unmatched'])
                    if pred_boxes.size:
                        ious = interval_iou_numpy(gt_boxes, pred_boxes)
                        best_iou = ious.max(axis=1)
                        stats['used_proposal'] += int(
                            (best_iou >= args.min_iou).sum())
                        stats['used_gt'] += int((best_iou < args.min_iou).sum())
                    else:
                        stats['used_gt'] += gt_boxes.shape[0]
                if (batch_idx + 1) % 50 == 0:
                    print(f'  [{batch_idx + 1}/{len(dataloader)}] frames')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    stats['out'] = args.out
    stats['num_files'] = len(cache)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
