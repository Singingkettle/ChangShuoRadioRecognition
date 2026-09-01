#!/usr/bin/env python3
"""Build HCGDNN release configs and optionally audit the public split."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from mmengine.config import Config


PAPER_DIR = Path(__file__).resolve().parent


def find_repo_root():
    for parent in (PAPER_DIR, *PAPER_DIR.parents):
        if (parent / 'tools/train.py').is_file() and (parent / 'csrr').is_dir():
            return parent
    raise RuntimeError('could not locate the CSRR repository root')


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from csrr.registry import MODELS  # noqa: E402
from csrr.utils import register_all_modules  # noqa: E402


CONFIGS = (
    'configs/hcgdnn/hcgdnn_iq-deepsig-201610a.py',
    'configs/hcgdnn/experiments/hcgdnn_iq-deepsig-201610a_final.py',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    return parser.parse_args()


def check_config(path):
    cfg = Config.fromfile(REPO_ROOT / path)
    final = '/experiments/' in path
    expected_train = 'train_and_validation.json' if final else 'train.json'
    if cfg.train_dataloader.dataset.ann_file != expected_train:
        raise AssertionError(f'wrong training split in {path}')
    if final:
        if any(cfg.get(key) is not None
               for key in ('val_dataloader', 'val_evaluator', 'val_cfg')):
            raise AssertionError(f'final validation is enabled in {path}')
        if cfg.test_dataloader.dataset.ann_file != 'test.json':
            raise AssertionError(f'wrong test split in {path}')
    else:
        if cfg.val_dataloader.dataset.ann_file != 'validation.json':
            raise AssertionError(f'wrong validation split in {path}')
        if cfg.get('test_dataloader') is not None:
            raise AssertionError(f'calibration can access test in {path}')
    if cfg.train_dataloader.batch_size != 640:
        raise AssertionError(f'paper batch size changed in {path}')
    if cfg.train_cfg.max_epochs != 1600:
        raise AssertionError(f'paper epoch bound changed in {path}')
    if (cfg.optim_wrapper.optimizer.type != 'Adam'
            or cfg.optim_wrapper.optimizer.lr != 4.4e-4):
        raise AssertionError(f'paper optimizer changed in {path}')
    return cfg


def build_and_forward(path, cfg):
    model = MODELS.build(cfg.model)
    if sum(parameter.numel() for parameter in model.backbone.parameters()) \
            != 463557:
        raise AssertionError('HCGDNN parameter count differs from the paper')
    inputs = torch.randn(2, 2, 1, 128)
    model.backbone.eval()
    model.head.eval()
    with torch.no_grad():
        logits = model.backbone(inputs)
        probabilities = model.head.pre_logits(logits)
        predictions = model.head.predict(logits)
    if set(logits) != {'cnn', 'gru1', 'gru2'}:
        raise AssertionError(f'head outputs differ in {path}')
    if probabilities.shape != (2, 11):
        raise AssertionError(f'forward shape differs in {path}')
    actual = torch.stack([sample.pred_score for sample in predictions])
    torch.testing.assert_close(actual, probabilities)
    torch.testing.assert_close(actual.sum(1), torch.ones(2))


def read_split(root, name, expected):
    records = json.loads(
        (root / f'{name}.json').read_text(encoding='utf-8'))['data_list']
    mapped = {item['file_name']: (item['modulation'], item['snr'])
              for item in records}
    if len(records) != expected or len(mapped) != expected:
        raise AssertionError(f'{name} count differs from {expected}')
    return mapped


def check_data():
    root = (REPO_ROOT / 'data/ModulationClassification/DeepSig/'
            'RadioML.2016.10A')
    counts = {'train': 110000, 'validation': 22000,
              'train_and_validation': 132000, 'test': 88000}
    per_stratum = {'train': 500, 'validation': 100,
                   'train_and_validation': 600, 'test': 400}
    splits = {name: read_split(root, name, count)
              for name, count in counts.items()}
    train = set(splits['train'])
    validation = set(splits['validation'])
    test = set(splits['test'])
    if train & validation or train & test or validation & test:
        raise AssertionError('2016 split intersection is non-empty')
    if splits['train_and_validation'] != {
            **splits['train'], **splits['validation']}:
        raise AssertionError('merged 60% split is not the exact union')
    if set(splits['train_and_validation']) & test:
        raise AssertionError('merged 60% split intersects test')
    for name, records in splits.items():
        strata = Counter(records.values())
        if len(strata) != 220 or set(strata.values()) != {per_stratum[name]}:
            raise AssertionError(f'{name} is not stratified exactly')


def main():
    args = parse_args()
    register_all_modules()
    for path in CONFIGS:
        cfg = check_config(path)
        build_and_forward(path, cfg)
        print(f'[OK] {path}')
    if args.check_data:
        check_data()
        print('[OK] 201610a 50/10/40 split')
    print('HCGDNN release check passed')


if __name__ == '__main__':
    main()
