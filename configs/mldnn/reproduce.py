#!/usr/bin/env python3
"""Run the leakage-controlled two-stage MLDNN reproduction."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from release_utils import (aggregate_predictions, atomic_write_json,
                           calculate_metrics, dump_prediction,
                           select_validation_epoch)


PAPER_DIR = Path(__file__).resolve().parent


def find_repo_root():
    for parent in (PAPER_DIR, *PAPER_DIR.parents):
        if (parent / 'tools/train.py').is_file() and (parent / 'csrr').is_dir():
            return parent
    raise RuntimeError('could not locate the CSRR repository root')


REPO_ROOT = find_repo_root()
DATASETS = {
    '201610a': {
        'calibration': 'configs/mldnn/mldnn_iq-ap-deepsig-201610a.py',
        'final': ('configs/mldnn/experiments/'
                  'mldnn_iq-ap-deepsig-201610a_final.py'),
        'seeds': (31, 37, 41),
        'phase_views': 8,
        'paper_maa': 63.40,
    },
    '201801a': {
        'calibration': 'configs/mldnn/mldnn_iq-ap-deepsig-201801a.py',
        'final': ('configs/mldnn/experiments/'
                  'mldnn_iq-ap-deepsig-201801a_final.py'),
        'seeds': (17,),
        'phase_views': 1,
        'paper_maa': 60.70,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    parser.add_argument('--devices', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--work-dir', default='work_dirs/mldnn_reproduction')
    parser.add_argument(
        '--stage', choices=('all', 'calibration', 'final', 'test', 'aggregate'),
        default='all')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return os.path.relpath(Path(path).resolve(), REPO_ROOT)


def run_command(command, device):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    env['PYTHONPATH'] = str(REPO_ROOT)
    print(f'[GPU {device}]', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def run_device_queue(device, jobs):
    for command in jobs:
        run_command(command, device)


def run_parallel(commands, devices):
    queues = [[] for _ in devices]
    for index, command in enumerate(commands):
        queues[index % len(devices)].append(command)
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run_device_queue, device, queue)
                   for device, queue in zip(devices, queues) if queue]
        for future in futures:
            future.result()


def seed_dir(root, dataset, seed):
    return root / dataset / f'seed_{seed}'


def write_selections(root, dataset, recipe):
    for seed in recipe['seeds']:
        current = seed_dir(root, dataset, seed)
        epoch, accuracy = select_validation_epoch(current / 'calibration_50_10')
        selection_path = current / 'selection.json'
        payload = {
            'schema_version': 1,
            'dataset': dataset,
            'seed': seed,
            'selected_epoch': epoch,
            'validation_accuracy_top1': accuracy,
            'selection_rule': 'maximum validation top-1; earliest epoch on tie',
            'selection_split': 'validation.json',
            'test_split_seen': False,
        }
        if selection_path.exists():
            existing = json.loads(selection_path.read_text(encoding='utf-8'))
            if existing != payload:
                raise RuntimeError(f'selection changed: {selection_path}')
        else:
            atomic_write_json(selection_path, payload)


def calibration(root, dataset, recipe, devices):
    commands = []
    for seed in recipe['seeds']:
        current = seed_dir(root, dataset, seed)
        if (current / 'selection.json').is_file():
            continue
        work_dir = current / 'calibration_50_10'
        commands.append([
            sys.executable, 'tools/train.py', recipe['calibration'],
            '--work-dir', str(work_dir), '--cfg-options',
            f'randomness.seed={seed}', 'randomness.deterministic=False',
        ])
    run_parallel(commands, devices)
    write_selections(root, dataset, recipe)


def final_training(root, dataset, recipe, devices):
    commands = []
    completions = []
    for seed in recipe['seeds']:
        current = seed_dir(root, dataset, seed)
        selection_path = current / 'selection.json'
        if not selection_path.is_file():
            raise RuntimeError(f'missing calibration selection: {selection_path}')
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if selection.get('test_split_seen') is not False:
            raise RuntimeError(f'invalid selection provenance: {selection_path}')
        final_dir = current / 'final_60'
        complete = current / 'final_complete.json'
        if complete.exists():
            continue
        started = current / '.final_training_started'
        if started.exists():
            raise RuntimeError(
                f'incomplete final run must be audited before retry: {started}')
        atomic_write_json(started, {
            'seed': seed,
            'selected_epoch': selection['selected_epoch'],
            'training_split': 'train_and_validation.json',
            'validation': False,
        })
        commands.append([
            sys.executable, 'tools/train.py', recipe['final'],
            '--work-dir', str(final_dir), '--cfg-options',
            f'randomness.seed={seed}', 'randomness.deterministic=False',
            f'train_cfg.max_epochs={selection["selected_epoch"]}',
        ])
        completions.append((seed, selection['selected_epoch'], final_dir,
                            complete))
    run_parallel(commands, devices)
    for seed, epoch, final_dir, complete in completions:
        matches = sorted(final_dir.rglob(f'epoch_{epoch}.pth'))
        if len(matches) != 1:
            raise RuntimeError(
                f'expected one epoch_{epoch}.pth below {final_dir}')
        atomic_write_json(complete, {
            'seed': seed,
            'selected_epoch': epoch,
            'checkpoint': relative(matches[0]),
            'checkpoint_sha256': sha256(matches[0]),
            'training_split': 'train_and_validation.json',
            'validation': False,
        })


def test_once(root, dataset, recipe, devices):
    commands = []
    completions = []
    for seed in recipe['seeds']:
        current = seed_dir(root, dataset, seed)
        final_record = current / 'final_complete.json'
        if not final_record.is_file():
            raise RuntimeError(f'missing final record: {final_record}')
        tested = current / 'test_complete.json'
        if tested.exists():
            continue
        marker = current / '.test_started'
        if marker.exists():
            raise RuntimeError(f'refusing to test twice: {marker}')
        final = json.loads(final_record.read_text(encoding='utf-8'))
        checkpoint = REPO_ROOT / final['checkpoint']
        if sha256(checkpoint) != final['checkpoint_sha256']:
            raise RuntimeError(f'checkpoint digest changed: {checkpoint}')
        test_dir = current / 'test_once'
        atomic_write_json(marker, {
            'seed': seed,
            'checkpoint_sha256': final['checkpoint_sha256'],
            'test_split': 'test.json',
        })
        command = [
            sys.executable, 'tools/test.py', recipe['final'], str(checkpoint),
            '--work-dir', str(test_dir),
        ]
        if recipe['phase_views'] > 1:
            command.extend(['--phase-rotation-tta-views',
                            str(recipe['phase_views'])])
        commands.append(command)
        completions.append((seed, test_dir, tested))
    run_parallel(commands, devices)
    for seed, test_dir, tested in completions:
        prediction = test_dir / 'res/paper.pkl'
        metrics = test_dir / 'amc_test_metrics.json'
        if not prediction.is_file() or not metrics.is_file():
            raise RuntimeError(f'test outputs are incomplete: {test_dir}')
        atomic_write_json(tested, {
            'seed': seed,
            'prediction': relative(prediction),
            'prediction_sha256': sha256(prediction),
            'metrics': relative(metrics),
            'test_evaluations': 1,
        })


def aggregate(root, dataset, recipe):
    paths = []
    for seed in recipe['seeds']:
        record_path = seed_dir(root, dataset, seed) / 'test_complete.json'
        if not record_path.is_file():
            raise RuntimeError(f'missing test record: {record_path}')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        prediction = REPO_ROOT / record['prediction']
        if sha256(prediction) != record['prediction_sha256']:
            raise RuntimeError(f'prediction digest changed: {prediction}')
        paths.append(prediction)
    result = aggregate_predictions(paths)
    metrics = calculate_metrics(result['pps'], result['gts'], result['snrs'])
    output_dir = root / dataset / 'aggregate'
    prediction_path = output_dir / 'paper.pkl'
    dump_prediction(prediction_path, result)
    atomic_write_json(output_dir / 'metrics.json', {
        **metrics,
        'paper_maa': recipe['paper_maa'],
        'target_met': metrics['accuracy/maa'] >= recipe['paper_maa'],
        'seeds': list(recipe['seeds']),
        'phase_views_per_model': recipe['phase_views'],
        'aggregation': 'equal_probability_mean',
        'prediction_sha256': sha256(prediction_path),
    })
    print(f'{dataset} MAA: {metrics["accuracy/maa"]:.4f}%')


def main():
    args = parse_args()
    if not args.devices or len(set(args.devices)) != len(args.devices):
        raise ValueError('--devices must contain unique device indices')
    root = Path(args.work_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    datasets = DATASETS if args.dataset == 'all' else {
        args.dataset: DATASETS[args.dataset]}
    stages = ('calibration', 'final', 'test', 'aggregate') \
        if args.stage == 'all' else (args.stage,)
    for dataset, recipe in datasets.items():
        for stage in stages:
            if stage == 'calibration':
                calibration(root, dataset, recipe, args.devices)
            elif stage == 'final':
                final_training(root, dataset, recipe, args.devices)
            elif stage == 'test':
                test_once(root, dataset, recipe, args.devices)
            else:
                aggregate(root, dataset, recipe)


if __name__ == '__main__':
    main()
