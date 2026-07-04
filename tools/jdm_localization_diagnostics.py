"""Diagnose JDM detector localization errors.

This script runs detector inference on a CSRD split, greedily associates each
ground-truth interval with its best-overlap prediction, and writes compact CSV,
JSON, and histogram artifacts for center, bandwidth, IoU, SNR, and size bins.
"""
import argparse
import csv
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from csrr.evaluation.metrics.detection import (DEFAULT_SIZE_RANGES,
                                               interval_iou_numpy)
from csrr.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run JDM detector localization diagnostics.')
    parser.add_argument('config', help='detector config file')
    parser.add_argument('checkpoint', help='detector checkpoint')
    parser.add_argument(
        '--split',
        choices=('validation', 'test'),
        default='test',
        help='dataset split to diagnose')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/jdm/diagnostics',
        help='directory for diagnostic artifacts')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='optional smoke-test cap on the number of frames')
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


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _snr_sort_key(value):
    value = _json_scalar(value)
    if isinstance(value, str) and value.lower().endswith('db'):
        value = value[:-2]
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None and np.isfinite(parsed):
        return (0, parsed, '')
    text = str(_json_scalar(value))
    if text.lower() in ('infdb', '+infdb', 'inf', '+inf', 'infinity'):
        return (1, float('inf'), text)
    return (2, 0.0, text)


def _size_bucket(width):
    for name, (lo, hi) in DEFAULT_SIZE_RANGES.items():
        if lo <= width < hi:
            return name
    return 'other'


def _summarize_numeric(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(count=0)
    return dict(
        count=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std()),
        p50=float(np.quantile(arr, 0.50)),
        p75=float(np.quantile(arr, 0.75)),
        p90=float(np.quantile(arr, 0.90)),
        p95=float(np.quantile(arr, 0.95)),
        max=float(arr.max()),
    )


def _summarize_rows(rows):
    matched = [r for r in rows if r['matched']]
    return dict(
        num_gt=len(rows),
        num_matched=len(matched),
        recall_best_iou_50=float(np.mean([r['best_iou'] >= 0.5
                                          for r in rows])) if rows else 0.0,
        recall_best_iou_75=float(np.mean([r['best_iou'] >= 0.75
                                          for r in rows])) if rows else 0.0,
        iou=_summarize_numeric([r['best_iou'] for r in rows]),
        center_abs_bins=_hist(
            [r['center_abs_err'] for r in matched],
            [0, 1, 2, 4, 8, 16, 32, 64, np.inf]),
        width_abs_bins=_hist(
            [r['width_abs_err'] for r in matched],
            [0, 1, 2, 4, 8, 16, 32, 64, np.inf]),
        center_abs_err=_summarize_numeric(
            [r['center_abs_err'] for r in matched]),
        center_abs_err_norm_width=_summarize_numeric(
            [r['center_abs_err_norm_width'] for r in matched]),
        width_signed_err=_summarize_numeric(
            [r['width_signed_err'] for r in matched]),
        width_abs_err=_summarize_numeric(
            [r['width_abs_err'] for r in matched]),
        width_rel_err=_summarize_numeric(
            [r['width_rel_err'] for r in matched]),
        left_abs_err=_summarize_numeric([r['left_abs_err'] for r in matched]),
        right_abs_err=_summarize_numeric(
            [r['right_abs_err'] for r in matched]),
        score=_summarize_numeric([r['pred_score'] for r in matched]),
    )


def _hist(values, bins):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=np.asarray(bins, dtype=np.float64))
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f'[{left:g},{right:g})')
    return {label: int(count) for label, count in zip(labels, counts)}


def _write_plots(rows, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matched = [r for r in rows if r['matched']]
    plot_specs = [
        ('iou_hist.png', [r['best_iou'] for r in rows], np.linspace(0, 1, 41),
         'Best prediction IoU per GT', 'IoU'),
        ('center_abs_err_hist.png',
         [r['center_abs_err'] for r in matched],
         [0, 1, 2, 4, 8, 16, 32, 64, 128],
         'Center absolute error for matched GT', 'FFT bins'),
        ('width_abs_err_hist.png',
         [r['width_abs_err'] for r in matched],
         [0, 1, 2, 4, 8, 16, 32, 64, 128],
         'Bandwidth absolute error for matched GT', 'FFT bins'),
        ('width_rel_err_hist.png',
         [r['width_rel_err'] for r in matched],
         np.linspace(-0.8, 0.8, 65),
         'Signed relative bandwidth error', '(pred - gt) / gt'),
    ]
    for filename, values, bins, title, xlabel in plot_specs:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        if arr.size:
            ax.hist(arr, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.35)
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, filename))
        plt.close(fig)

    by_size = defaultdict(list)
    for row in rows:
        by_size[row['size_bucket']].append(row['best_iou'])
    labels = [key for key in ('small', 'medium', 'large') if key in by_size]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if labels:
        ax.boxplot([by_size[key] for key in labels], tick_labels=labels)
    ax.set_title('Best IoU by GT bandwidth bucket')
    ax.set_ylabel('IoU')
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, 'iou_by_size.png'))
    plt.close(fig)


def _select_dataloader_cfg(cfg, split):
    loader = cfg.val_dataloader if split == 'validation' else cfg.test_dataloader
    loader = loader.copy()
    loader.dataset = loader.dataset.copy()
    loader.dataset.split = split
    return loader


def _build_dataloader(cfg, split, batch_size, num_workers, max_samples):
    dataloader_cfg = _select_dataloader_cfg(cfg, split)
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.setdefault('collate_fn', dict(type='default_collate'))
    if max_samples is not None:
        dataloader_cfg.dataset.indices = max_samples
    return Runner.build_dataloader(dataloader_cfg)


def _match_sample(sample, sample_idx, rows):
    pred_boxes = _as_numpy(_sample_value(sample, 'pred_boxes')).reshape(-1, 2)
    pred_scores = _as_numpy(
        _sample_value(sample, 'pred_box_scores')).reshape(-1)
    gt_boxes = _as_numpy(_sample_value(sample, 'gt_boxes')).reshape(-1, 2)
    gt_labels = _as_numpy(_sample_value(sample, 'gt_box_labels')).reshape(-1)
    gt_snr = _sample_value(sample, 'snr', ['unknown'] * len(gt_boxes))
    gt_snr = np.asarray(gt_snr, dtype=object).reshape(-1)

    if pred_boxes.size:
        ious = interval_iou_numpy(gt_boxes, pred_boxes)
        best_pred = ious.argmax(axis=1)
        best_iou = ious[np.arange(gt_boxes.shape[0]), best_pred]
    else:
        best_pred = np.full(gt_boxes.shape[0], -1, dtype=np.int64)
        best_iou = np.zeros(gt_boxes.shape[0], dtype=np.float64)

    for gt_idx, gt_box in enumerate(gt_boxes):
        gt_center = float((gt_box[0] + gt_box[1]) / 2)
        gt_width = float(gt_box[1] - gt_box[0])
        row = dict(
            sample_idx=int(sample_idx),
            gt_idx=int(gt_idx),
            gt_label=int(gt_labels[gt_idx]),
            gt_snr=_json_scalar(gt_snr[gt_idx]),
            gt_left=float(gt_box[0]),
            gt_right=float(gt_box[1]),
            gt_center=gt_center,
            gt_width=gt_width,
            gt_center_cell=int(np.floor(gt_center / 8.0)),
            gt_center_cell_frac=float(gt_center / 8.0 - np.floor(
                gt_center / 8.0)),
            size_bucket=_size_bucket(gt_width),
            best_iou=float(best_iou[gt_idx]),
            matched=bool(best_pred[gt_idx] >= 0),
            pred_idx=int(best_pred[gt_idx]),
            pred_score=float('nan'),
            pred_left=float('nan'),
            pred_right=float('nan'),
            pred_center=float('nan'),
            pred_width=float('nan'),
            center_signed_err=float('nan'),
            center_abs_err=float('nan'),
            center_abs_err_norm_width=float('nan'),
            width_signed_err=float('nan'),
            width_abs_err=float('nan'),
            width_rel_err=float('nan'),
            left_signed_err=float('nan'),
            left_abs_err=float('nan'),
            right_signed_err=float('nan'),
            right_abs_err=float('nan'),
        )
        if row['matched']:
            pred_box = pred_boxes[row['pred_idx']]
            pred_center = float((pred_box[0] + pred_box[1]) / 2)
            pred_width = float(pred_box[1] - pred_box[0])
            center_err = pred_center - gt_center
            width_err = pred_width - gt_width
            left_err = float(pred_box[0] - gt_box[0])
            right_err = float(pred_box[1] - gt_box[1])
            row.update(
                pred_score=float(pred_scores[row['pred_idx']]),
                pred_left=float(pred_box[0]),
                pred_right=float(pred_box[1]),
                pred_center=pred_center,
                pred_width=pred_width,
                center_signed_err=float(center_err),
                center_abs_err=float(abs(center_err)),
                center_abs_err_norm_width=float(abs(center_err) / gt_width),
                width_signed_err=float(width_err),
                width_abs_err=float(abs(width_err)),
                width_rel_err=float(width_err / gt_width),
                left_signed_err=left_err,
                left_abs_err=abs(left_err),
                right_signed_err=right_err,
                right_abs_err=abs(right_err),
            )
        rows.append(row)


def _write_csv(rows, out_path):
    if not rows:
        return
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

    dataloader = _build_dataloader(
        cfg, args.split, args.batch_size, args.num_workers, args.max_samples)
    os.makedirs(args.work_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = model.data_preprocessor(data, training=False)
            data['inputs'] = data['inputs'].to(device)
            results = model(
                data['inputs'], data.get('data_samples'), mode='predict')
            batch_base = batch_idx * args.batch_size
            for offset, sample in enumerate(results):
                sample_idx = _sample_value(sample, 'sample_idx',
                                           batch_base + offset)
                _match_sample(sample, sample_idx, rows)
            if (batch_idx + 1) % 50 == 0:
                print(f'[{batch_idx + 1}/{len(dataloader)}] frames processed')

    summary = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        num_frames=len(dataloader.dataset),
        overall=_summarize_rows(rows),
        by_size={
            name: _summarize_rows(
                [r for r in rows if r['size_bucket'] == name])
            for name in ('small', 'medium', 'large')
        },
        by_snr={
            str(_json_scalar(snr)): _summarize_rows(
                [r for r in rows if r['gt_snr'] == snr])
            for snr in sorted({r['gt_snr'] for r in rows}, key=_snr_sort_key)
        },
    )

    _write_csv(rows, osp.join(args.work_dir, f'{args.split}_localization.csv'))
    with open(osp.join(args.work_dir, f'{args.split}_localization.json'),
              'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    _write_plots(rows, args.work_dir)
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    main()
