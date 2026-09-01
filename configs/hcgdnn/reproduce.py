#!/usr/bin/env python3
"""Run the leakage-controlled two-stage HCGDNN reproduction."""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from release_utils import (aggregate_predictions, atomic_write_json,
                           average_checkpoints, calculate_metrics,
                           dump_prediction, select_validation_epoch, sha256,
                           transplant_fusion)


PAPER_DIR = Path(__file__).resolve().parent


def find_repo_root():
    for parent in (PAPER_DIR, *PAPER_DIR.parents):
        if (parent / 'tools/train.py').is_file() and (parent / 'csrr').is_dir():
            return parent
    raise RuntimeError('could not locate the CSRR repository root')


REPO_ROOT = find_repo_root()
CALIBRATION_CONFIG = 'configs/hcgdnn/hcgdnn_iq-deepsig-201610a.py'
FINAL_CONFIG = ('configs/hcgdnn/experiments/'
                'hcgdnn_iq-deepsig-201610a_final.py')
SEEDS = (31, 37, 41, 43, 47, 53)
PAPER_MAA = 63.75


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--work-dir', default='work_dirs/hcgdnn_reproduction')
    parser.add_argument(
        '--stage', choices=('all', 'calibration', 'final', 'test', 'aggregate'),
        default='all')
    return parser.parse_args()


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


def seed_dir(root, seed):
    return root / f'seed_{seed}'


def calibration_checkpoint(work_dir, epoch):
    exact = sorted(work_dir.rglob(
        f'best_accuracy_top1_epoch_{epoch}.pth'))
    if len(exact) != 1:
        raise RuntimeError(
            f'expected one validation-selected checkpoint below {work_dir}')
    return exact[0]


def calibration(root, devices):
    commands = []
    for seed in SEEDS:
        current = seed_dir(root, seed)
        if (current / 'selection.json').is_file():
            continue
        work_dir = current / 'calibration_50_10'
        commands.append([
            sys.executable, 'tools/train.py', CALIBRATION_CONFIG,
            '--work-dir', str(work_dir), '--cfg-options',
            f'randomness.seed={seed}', 'randomness.deterministic=False',
        ])
    run_parallel(commands, devices)
    for seed in SEEDS:
        current = seed_dir(root, seed)
        calibration_dir = current / 'calibration_50_10'
        epoch, accuracy = select_validation_epoch(calibration_dir)
        checkpoint = calibration_checkpoint(calibration_dir, epoch)
        payload = {
            'schema_version': 1,
            'seed': seed,
            'selected_epoch': epoch,
            'validation_accuracy_top1': accuracy,
            'calibration_checkpoint': relative(checkpoint),
            'calibration_checkpoint_sha256': sha256(checkpoint),
            'selection_rule': 'maximum validation top-1; earliest epoch on tie',
            'selection_split': 'validation.json',
            'fusion_split': 'validation.json',
            'test_split_seen': False,
        }
        selection = current / 'selection.json'
        if selection.exists():
            if json.loads(selection.read_text(encoding='utf-8')) != payload:
                raise RuntimeError(f'selection changed: {selection}')
        else:
            atomic_write_json(selection, payload)


def retained_checkpoints(final_dir, selected_epoch):
    parsed = []
    for path in final_dir.rglob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    parsed.sort()
    if len(parsed) != 3 or parsed[-1][0] != selected_epoch:
        raise RuntimeError(
            f'expected exactly three retained checkpoints ending at '
            f'epoch {selected_epoch}: {parsed}')
    return [path for _, path in parsed]


def final_training(root, devices):
    commands = []
    postprocess = []
    for seed in SEEDS:
        current = seed_dir(root, seed)
        selection_path = current / 'selection.json'
        if not selection_path.is_file():
            raise RuntimeError(f'missing calibration selection: {selection_path}')
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        calibration_path = REPO_ROOT / selection['calibration_checkpoint']
        if sha256(calibration_path) != selection['calibration_checkpoint_sha256']:
            raise RuntimeError(f'calibration checkpoint changed: {calibration_path}')
        complete = current / 'final_complete.json'
        if complete.exists():
            continue
        marker = current / '.final_training_started'
        if marker.exists():
            raise RuntimeError(
                f'incomplete final run must be audited before retry: {marker}')
        atomic_write_json(marker, {
            'seed': seed,
            'selected_epoch': selection['selected_epoch'],
            'training_split': 'train_and_validation.json',
            'validation': False,
        })
        final_dir = current / 'final_60'
        commands.append([
            sys.executable, 'tools/train.py', FINAL_CONFIG,
            '--work-dir', str(final_dir), '--cfg-options',
            f'randomness.seed={seed}', 'randomness.deterministic=False',
            f'train_cfg.max_epochs={selection["selected_epoch"]}',
        ])
        postprocess.append((seed, selection, calibration_path, final_dir,
                            complete))
    run_parallel(commands, devices)
    for seed, selection, calibration_path, final_dir, complete in postprocess:
        retained = retained_checkpoints(
            final_dir, selection['selected_epoch'])
        model_dir = seed_dir(root, seed) / 'final_model'
        averaged = model_dir / 'averaged.pth'
        final_checkpoint = model_dir / 'averaged_calibrated.pth'
        average_record = average_checkpoints(retained, averaged)
        fusion_record = transplant_fusion(
            calibration_path, averaged, final_checkpoint)
        atomic_write_json(complete, {
            'seed': seed,
            'selected_epoch': selection['selected_epoch'],
            'retained_checkpoints': [relative(path) for path in retained],
            'average': average_record,
            'fusion': fusion_record,
            'checkpoint': relative(final_checkpoint),
            'checkpoint_sha256': sha256(final_checkpoint),
            'training_split': 'train_and_validation.json',
            'validation': False,
        })


def test_once(root, devices):
    commands = []
    completions = []
    for seed in SEEDS:
        current = seed_dir(root, seed)
        final_record = current / 'final_complete.json'
        if not final_record.is_file():
            raise RuntimeError(f'missing final record: {final_record}')
        completed = current / 'test_complete.json'
        if completed.exists():
            continue
        marker = current / '.test_started'
        if marker.exists():
            raise RuntimeError(f'refusing to test twice: {marker}')
        final = json.loads(final_record.read_text(encoding='utf-8'))
        checkpoint = REPO_ROOT / final['checkpoint']
        if sha256(checkpoint) != final['checkpoint_sha256']:
            raise RuntimeError(f'final checkpoint changed: {checkpoint}')
        atomic_write_json(marker, {
            'seed': seed,
            'checkpoint_sha256': final['checkpoint_sha256'],
            'test_split': 'test.json',
        })
        test_dir = current / 'test_once'
        commands.append([
            sys.executable, 'tools/test.py', FINAL_CONFIG, str(checkpoint),
            '--work-dir', str(test_dir),
        ])
        completions.append((seed, test_dir, completed))
    run_parallel(commands, devices)
    for seed, test_dir, completed in completions:
        prediction = test_dir / 'res/paper.pkl'
        if not prediction.is_file():
            raise RuntimeError(f'test output is incomplete: {test_dir}')
        atomic_write_json(completed, {
            'seed': seed,
            'prediction': relative(prediction),
            'prediction_sha256': sha256(prediction),
            'test_evaluations': 1,
        })


def aggregate(root):
    paths = []
    for seed in SEEDS:
        record_path = seed_dir(root, seed) / 'test_complete.json'
        if not record_path.is_file():
            raise RuntimeError(f'missing test record: {record_path}')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        prediction = REPO_ROOT / record['prediction']
        if sha256(prediction) != record['prediction_sha256']:
            raise RuntimeError(f'prediction changed: {prediction}')
        paths.append(prediction)
    result = aggregate_predictions(paths)
    result['members'] = [relative(path) for path in paths]
    metrics = calculate_metrics(result['pps'], result['gts'], result['snrs'])
    output_dir = root / 'aggregate'
    prediction_path = output_dir / 'paper.pkl'
    dump_prediction(prediction_path, result)
    atomic_write_json(output_dir / 'metrics.json', {
        **metrics,
        'paper_maa': PAPER_MAA,
        'target_met': metrics['accuracy/maa'] >= PAPER_MAA,
        'seeds': list(SEEDS),
        'checkpoint_rule': 'equal mean of last three retained checkpoints',
        'prediction_aggregation': 'equal_probability_mean',
        'prediction_sha256': sha256(prediction_path),
    })
    print(f'HCGDNN MAA: {metrics["accuracy/maa"]:.4f}%')


def main():
    args = parse_args()
    if not args.devices or len(set(args.devices)) != len(args.devices):
        raise ValueError('--devices must contain unique device indices')
    root = Path(args.work_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    stages = ('calibration', 'final', 'test', 'aggregate') \
        if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if stage == 'calibration':
            calibration(root, args.devices)
        elif stage == 'final':
            final_training(root, args.devices)
        elif stage == 'test':
            test_once(root, args.devices)
        else:
            aggregate(root)


if __name__ == '__main__':
    main()
