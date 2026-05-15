import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create AMR-Benchmark compatible 600/200/200 splits for RadioML2016.10A-style data.')
    parser.add_argument('--source-root', required=True, help='Existing converted CSRR dataset root with train/test JSON and iq/.')
    parser.add_argument('--out-root', required=True, help='Output dataset root. A symlink to source iq/ is created.')
    parser.add_argument('--seed', type=int, default=2016)
    parser.add_argument('--train-per-group', type=int, default=600)
    parser.add_argument('--val-per-group', type=int, default=200)
    parser.add_argument(
        '--test-per-group',
        type=int,
        default=200,
        help='Number of held-out test samples per modulation/SNR group. '
        'Use a negative value to keep the full remainder as test.')
    return parser.parse_args()


def load_all_items(source_root):
    merged = []
    metainfo = None
    seen = set()
    for name in ('train.json', 'validation.json', 'test.json'):
        path = source_root / name
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if metainfo is None:
            metainfo = payload['metainfo']
        for item in payload['data_list']:
            key = item['file_name']
            if key in seen:
                raise ValueError(f'Duplicate file_name across source splits: {key}')
            seen.add(key)
            merged.append(item)
    return merged, metainfo


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    iq_link = out_root / 'iq'
    if not iq_link.exists():
        os.symlink(source_root / 'iq', iq_link, target_is_directory=True)

    items, metainfo = load_all_items(source_root)
    groups = defaultdict(list)
    for item in items:
        groups[(item['modulation'], item['snr'])].append(item)

    rng = np.random.RandomState(args.seed)
    train, val, test = [], [], []
    summary = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x['file_name'])
        n = len(group)
        fixed_test = args.test_per_group >= 0
        requested = args.train_per_group + args.val_per_group + (args.test_per_group if fixed_test else 0)
        if n < requested:
            raise ValueError(f'Group {key} has only {n} samples')
        indices = np.arange(n)
        train_idx = rng.choice(indices, size=args.train_per_group, replace=False)
        rest = np.array(sorted(set(indices.tolist()) - set(train_idx.tolist())))
        val_idx = rng.choice(rest, size=args.val_per_group, replace=False)
        rest_after_val = np.array(sorted(set(rest.tolist()) - set(val_idx.tolist())))
        if fixed_test:
            test_idx = rng.choice(rest_after_val, size=args.test_per_group, replace=False)
        else:
            test_idx = rest_after_val
        train.extend(group[i] for i in train_idx.tolist())
        val.extend(group[i] for i in val_idx.tolist())
        test.extend(group[i] for i in test_idx.tolist())
        summary.append((key[0], key[1], len(train_idx), len(val_idx), len(test_idx)))

    payloads = {
        'train.json': train,
        'validation.json': val,
        'test.json': test,
    }
    for name, split_items in payloads.items():
        with (out_root / name).open('w', encoding='utf-8') as f:
            json.dump({'data_list': split_items, 'metainfo': metainfo}, f, indent=2)

    with (out_root / 'split_manifest.md').open('w', encoding='utf-8') as f:
        f.write('# AMR-compatible split manifest\n\n')
        f.write(f'- source_root: `{source_root}`\n')
        f.write(f'- out_root: `{out_root}`\n')
        f.write(f'- seed: `{args.seed}`\n')
        test_desc = 'remainder' if args.test_per_group < 0 else str(args.test_per_group)
        f.write(f'- per modulation/SNR split: train `{args.train_per_group}`, validation `{args.val_per_group}`, test `{test_desc}`\n')
        f.write(f'- total train: `{len(train)}`\n')
        f.write(f'- total validation: `{len(val)}`\n')
        f.write(f'- total test: `{len(test)}`\n\n')
        f.write('| modulation | SNR | train | validation | test |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for mod, snr, n_train, n_val, n_test in summary:
            f.write(f'| {mod} | {snr} | {n_train} | {n_val} | {n_test} |\n')

    print(f'Wrote AMR-compatible split to {out_root}')
    print(f'train={len(train)} validation={len(val)} test={len(test)}')


if __name__ == '__main__':
    main()
