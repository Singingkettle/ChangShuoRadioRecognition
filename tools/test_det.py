# Copyright (c) Shuo Chang. All Rights Reserved.
"""Generic mmengine test entry for detection configs.

``tools/test.py`` is specialized for classification models (it collects
``pred_score`` per sample for the performance/plotting library); detection
models report interval-based metrics through the standard mmengine test loop
instead:

    python tools/test_det.py configs/jdm/jdm-det_fft-csrd.py <checkpoint.pth>
"""
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detection model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the dir to save logs and results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config file')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.test_dataloader.setdefault('collate_fn', dict(type='default_collate'))
    _set_default_snr_outputs(cfg.test_evaluator, cfg.work_dir)

    runner = Runner.from_cfg(cfg)
    runner.test()


def _set_default_snr_outputs(evaluator, work_dir: str) -> None:
    """Place SNR curve artifacts in the active work dir by default."""
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            _set_default_snr_outputs(item, work_dir)
        return
    if not isinstance(evaluator, dict) or not evaluator.get('snrwise', False):
        return
    evaluator.setdefault('snr_curve_out',
                         osp.join(work_dir, 'snr_curve.json'))
    evaluator.setdefault('snr_plot_out',
                         osp.join(work_dir, 'snr_curve.pdf'))


if __name__ == '__main__':
    main()
