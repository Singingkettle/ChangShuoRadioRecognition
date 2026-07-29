# Copyright (c) Shuo Chang. All Rights Reserved.
"""Fig. 10 point-by-point AMC audit (per-modulation x per-SNR accuracy).

The JDM paper's Fig. 10 reports modulation-classification accuracy as a curve
per modulation over SNR (AWGN solid, simulate hollow). Our aggregate metrics
(top1) cannot be compared point-by-point to that figure. This script runs a
trained classification model over a chosen set of CSRD versions and reports
top-1 accuracy for every (modulation, SNR) cell, then dumps a JSON table and a
matplotlib plot overlaying the digitized paper points (docs/.../paper_figure_targets.md).

It is evaluation-only (sequential read -> fast even on the slow local disk) and
uses the exact dataset/pipeline of the given config, so it works for both the
GT-box classifier (jdm-amc_iq-csrd.py) and proposal-crop classifiers
(amc_detprops_*.py).

Example::

    python tools/jdm_fig10_audit.py \\
        configs/jdm/jdm-amc_iq-csrd.py \\
        work_dirs/jdm/jdm-amc_iq-csrd/best_accuracy_top1_epoch_60.pth \\
        --versions v89 v90 v91 v92 v93 v94 v95 v96 v97 v98 \\
        --out-prefix work_dirs/jdm/retune/fig10_audit/gtbox_awgn
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from csrr.registry import MODELS


def _parse_snr(value):
    """CSRD annotations store SNR as e.g. '12dB' / '-8dB' / 'infdB'."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().replace('db', '')
    if 'inf' in s:
        return float('inf')
    return float(s)


# Digitized Fig. 10 simulate curve (docs/csrd_jointdet/paper_figure_targets.md);
# accuracy fraction per modulation over SNR 12..30 step 2 (BPSK ~1.0 at high SNR).
PAPER_FIG10_SIMULATE = {
    'BPSK': {12: 0.80, 14: 0.80, 16: 0.80, 18: 0.80, 20: 0.85, 22: 0.85,
             24: 0.90, 26: 0.90, 28: 0.95, 30: 0.98},
    'QPSK': {12: 0.40, 14: 0.55, 16: 0.56, 18: 0.56, 20: 0.61, 22: 0.65,
             24: 0.67, 26: 0.72, 28: 0.75, 30: 0.77},
    '8PSK': {12: 0.30, 14: 0.43, 16: 0.45, 18: 0.49, 20: 0.53, 22: 0.53,
             24: 0.54, 26: 0.60, 28: 0.61, 30: 0.63},
    '16QAM': {12: 0.17, 14: 0.39, 16: 0.40, 18: 0.40, 20: 0.41, 22: 0.49,
              24: 0.51, 26: 0.51, 28: 0.59, 30: 0.62},
    '64QAM': {12: 0.05, 14: 0.13, 16: 0.24, 18: 0.24, 20: 0.30, 22: 0.30,
              24: 0.32, 26: 0.34, 28: 0.39, 30: 0.43},
}


def parse_args():
    p = argparse.ArgumentParser(description='Fig.10 per-mod per-SNR AMC audit')
    p.add_argument('config', help='classification config (GT-box or proposal)')
    p.add_argument('checkpoint', help='classifier checkpoint')
    p.add_argument('--versions', nargs='+', required=True,
                   help='CSRD versions to evaluate (e.g. v89..v98 for AWGN)')
    p.add_argument('--out-prefix',
                   default='work_dirs/jdm/retune/fig10_audit/audit',
                   help='output prefix for <prefix>.json / <prefix>.pdf')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--paper-overlay', default='simulate',
                   choices=['simulate', 'none'],
                   help='which digitized paper curve to overlay')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'csrr'))

    classes = list(cfg.get('classes',
                           ('16QAM', '64QAM', '8PSK', 'BPSK', 'QPSK')))

    loader_cfg = cfg.test_dataloader.copy()
    loader_cfg['dataset'] = loader_cfg['dataset'].copy()
    loader_cfg['dataset']['versions'] = list(args.versions)
    loader_cfg['batch_size'] = args.batch_size
    loader_cfg['num_workers'] = args.num_workers
    loader_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)
    dataloader = Runner.build_dataloader(loader_cfg)
    dataset = dataloader.dataset

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith('cuda') else 'cpu')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device).eval()

    # Per-sample SNR / gt_label come from the dataset data_list in loader order
    # (shuffle=False, drop_last=False -> index i aligns with the i-th output).
    snrs = [_parse_snr(dataset.get_data_info(i)['snr'])
            for i in range(len(dataset))]
    gts = [int(dataset.get_data_info(i)['gt_label'])
           for i in range(len(dataset))]

    preds = []
    with torch.no_grad():
        for data in dataloader:
            data = model.data_preprocessor(data, training=False)
            out = model(data['inputs'].to(device),
                        data.get('data_samples'), mode='predict')
            for s in out:
                if hasattr(s, 'pred_label'):
                    preds.append(int(s.pred_label.item()))
                else:
                    preds.append(int(s.pred_score.argmax().item()))
    preds = preds[:len(gts)]

    # (mod, snr) -> [correct, total]
    cell = defaultdict(lambda: [0, 0])
    for pred, gt, snr in zip(preds, gts, snrs):
        cell[(gt, snr)][1] += 1
        if pred == gt:
            cell[(gt, snr)][0] += 1

    snr_values = sorted({s for (_, s) in cell})
    table = {}
    for ci, cname in enumerate(classes):
        row = {}
        for s in snr_values:
            c, n = cell[(ci, s)]
            row[str(int(s)) if float(s).is_integer() else str(s)] = \
                dict(acc=(c / n if n else float('nan')), n=n)
        table[cname] = row

    overall_correct = sum(v[0] for v in cell.values())
    overall_total = sum(v[1] for v in cell.values())
    result = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        versions=list(args.versions),
        classes=classes,
        snr_values=[int(s) if float(s).is_integer() else s
                    for s in snr_values],
        overall_top1=overall_correct / max(overall_total, 1),
        overall_n=overall_total,
        per_mod_per_snr=table,
    )

    os.makedirs(osp.dirname(args.out_prefix) or '.', exist_ok=True)
    with open(f'{args.out_prefix}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'overall_top1': result['overall_top1'],
                      'overall_n': overall_total,
                      'snr_values': result['snr_values']}, indent=2))

    _plot(table, classes, snr_values, args, result)


def _plot(table, classes, snr_values, args, result):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f'[fig10] plotting skipped: {e}')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    for ci, cname in enumerate(classes):
        xs = [s for s in snr_values]
        ys = [table[cname].get(
            str(int(s)) if float(s).is_integer() else str(s),
            dict(acc=float('nan')))['acc'] for s in snr_values]
        ax.plot(xs, ys, '-o', color=cmap(ci), label=f'{cname} (ours)')
        if args.paper_overlay == 'simulate' and cname in PAPER_FIG10_SIMULATE:
            px = sorted(PAPER_FIG10_SIMULATE[cname])
            py = [PAPER_FIG10_SIMULATE[cname][s] for s in px]
            ax.plot(px, py, '--x', color=cmap(ci), alpha=0.5,
                    label=f'{cname} (paper Fig.10 sim)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Top-1 accuracy')
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Fig.10 audit: {osp.basename(args.out_prefix)} "
                 f"(overall {result['overall_top1']:.3f})")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{args.out_prefix}.pdf')
    fig.savefig(f'{args.out_prefix}.png', dpi=120)
    print(f'[fig10] wrote {args.out_prefix}.json/.pdf/.png')


if __name__ == '__main__':
    main()
