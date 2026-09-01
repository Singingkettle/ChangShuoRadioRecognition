#!/usr/bin/env python3
"""Build MLDNN release configs and optionally audit public dataset splits."""

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
    'configs/mldnn/mldnn_iq-ap-deepsig-201610a.py',
    'configs/mldnn/experiments/mldnn_iq-ap-deepsig-201610a_final.py',
    'configs/mldnn/mldnn_iq-ap-deepsig-201801a.py',
    'configs/mldnn/experiments/mldnn_iq-ap-deepsig-201801a_final.py',
)
SPLITS = {
    '201610a': {
        'root': ('data/ModulationClassification/DeepSig/'
                 'RadioML.2016.10A'),
        'counts': {'train': 110000, 'validation': 22000,
                   'train_and_validation': 132000, 'test': 88000},
        'strata': 220,
        'per_stratum': {'train': 500, 'validation': 100,
                        'train_and_validation': 600, 'test': 400},
    },
    '201801a': {
        'root': ('data/ModulationClassification/DeepSig/'
                 'RadioML.2018.01A'),
        'counts': {'train': 1277952, 'validation': 255216,
                   'train_and_validation': 1533168, 'test': 1022736},
        'strata': 624,
        'per_stratum': {'train': 2048, 'validation': 409,
                        'train_and_validation': 2457, 'test': 1639},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    return parser.parse_args()


def check_config_contract(config_path):
    cfg = Config.fromfile(REPO_ROOT / config_path)
    final = '/experiments/' in config_path
    expected_train = 'train_and_validation.json' if final else 'train.json'
    if cfg.train_dataloader.dataset.ann_file != expected_train:
        raise AssertionError(f'wrong training split in {config_path}')
    if final:
        if any(cfg.get(key) is not None
               for key in ('val_dataloader', 'val_evaluator', 'val_cfg')):
            raise AssertionError(f'final validation is enabled in {config_path}')
        if cfg.test_dataloader.dataset.ann_file != 'test.json':
            raise AssertionError(f'wrong test split in {config_path}')
    else:
        if cfg.val_dataloader.dataset.ann_file != 'validation.json':
            raise AssertionError(f'wrong validation split in {config_path}')
        if cfg.get('test_dataloader') is not None:
            raise AssertionError(f'calibration can access test in {config_path}')
    if cfg.train_dataloader.batch_size != 640:
        raise AssertionError(f'paper batch size changed in {config_path}')
    if cfg.optim_wrapper.optimizer.type != 'Adam':
        raise AssertionError(f'paper optimizer changed in {config_path}')
    if cfg.optim_wrapper.optimizer.lr != 4e-4:
        raise AssertionError(f'paper learning rate changed in {config_path}')
    return cfg


def build_and_forward(config_path, cfg):
    model = MODELS.build(cfg.model)
    backbone = model.backbone
    length = 1024 if '201801a' in config_path else 128
    classes = 24 if length == 1024 else 11
    inputs = {
        'iq': torch.randn(1, 1, 2, length),
        'ap': torch.randn(1, 1, 2, length),
    }
    backbone.train()
    outputs = backbone(inputs)
    if len(outputs) != 4 or outputs[0].shape != (1, classes):
        raise AssertionError(f'training forward failed for {config_path}')
    probabilities = outputs[0].exp().sum(dim=1)
    if not torch.allclose(probabilities, torch.ones_like(probabilities),
                          atol=1e-5, rtol=1e-5):
        raise AssertionError(f'mixture is not a log probability: {config_path}')
    backbone.eval()
    with torch.no_grad():
        output = backbone(inputs)
    if len(output) != 1 or output[0].shape != (1, classes):
        raise AssertionError(f'evaluation forward failed for {config_path}')


def read_split(root, name, expected):
    path = root / f'{name}.json'
    records = json.loads(path.read_text(encoding='utf-8'))['data_list']
    mapped = {item['file_name']: (item['modulation'], item['snr'])
              for item in records}
    if len(records) != expected or len(mapped) != expected:
        raise AssertionError(f'{path} count differs from {expected}')
    return mapped


def check_dataset(name, spec):
    root = REPO_ROOT / spec['root']
    if not root.is_dir():
        raise FileNotFoundError(root)
    splits = {split: read_split(root, split, count)
              for split, count in spec['counts'].items()}
    train = set(splits['train'])
    validation = set(splits['validation'])
    test = set(splits['test'])
    merged = set(splits['train_and_validation'])
    if train & validation or train & test or validation & test:
        raise AssertionError(f'{name} split intersection is non-empty')
    if splits['train_and_validation'] != {
            **splits['train'], **splits['validation']}:
        raise AssertionError(f'{name} merged split is not the exact union')
    if merged & test:
        raise AssertionError(f'{name} final train intersects test')
    for split, records in splits.items():
        strata = Counter(records.values())
        if len(strata) != spec['strata']:
            raise AssertionError(f'{name} {split} stratum count differs')
        if set(strata.values()) != {spec['per_stratum'][split]}:
            raise AssertionError(f'{name} {split} is not stratified exactly')


def main():
    args = parse_args()
    register_all_modules()
    for config_path in CONFIGS:
        cfg = check_config_contract(config_path)
        build_and_forward(config_path, cfg)
        print(f'[OK] {config_path}')
    if args.check_data:
        for name, spec in SPLITS.items():
            check_dataset(name, spec)
            print(f'[OK] {name} 50/10/40 split')
    print('MLDNN release check passed')


if __name__ == '__main__':
    main()
