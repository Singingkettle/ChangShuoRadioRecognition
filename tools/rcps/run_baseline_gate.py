import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_amc_matrix import CONFIGS
from run_calibration_grid import checkpoint_for, collect_cmd, train_cmd, worker_options


def run(cmd, execute, env=None):
    print(' '.join(str(item) for item in cmd), flush=True)
    if execute:
        subprocess.run([str(item) for item in cmd], check=True, env=env)


def analyze_cmd(dataset, model, method, pred_path, out_csv):
    return [
        sys.executable, 'tools/rcps/analyze_reliability.py',
        '--run', f'{dataset}={model}={method}={pred_path.as_posix()}',
        '--out-csv', out_csv.as_posix(),
    ]


def efficiency_cmd(dataset, model, method, seed, work_dir, out_csv, target_accuracy):
    cmd = [
        sys.executable, 'tools/rcps/analyze_training_efficiency.py',
        '--run', f'{dataset}={model}={method}={seed}={work_dir.as_posix()}',
        '--out-csv', out_csv.as_posix(),
    ]
    if target_accuracy is not None:
        cmd.extend(['--target-accuracy', str(target_accuracy)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Run hard-label baseline parity gates.')
    parser.add_argument('--models', nargs='+', default=['cgdnet', 'petcgdnn', 'mcformer', 'fastmldnn'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--dataset', default='deepsig201610A')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs/baseline_gate_v2')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--collect-splits', nargs='+', default=['validation', 'test'])
    parser.add_argument('--target-accuracy', type=float, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg_worker_options = worker_options(args.num_workers)
    work_root = Path(args.work_root)
    metrics_root = work_root / 'metrics'
    efficiency_root = work_root / 'efficiency'

    for model in args.models:
        config = CONFIGS[(model, 'hard-ce')]
        for seed in args.seeds:
            method = 'hard-ce'
            work_dir = work_root / 'amc' / args.dataset / f'{model}_{method}' / f'seed_{seed}'
            test_csv = metrics_root / f'{args.dataset}_{model}_{method}_seed{seed}_test.csv'
            if args.skip_existing and test_csv.exists():
                print(f'Skip existing baseline metric: {test_csv}', flush=True)
                continue

            run(train_cmd(config, work_dir, seed, args.max_epochs, cfg_worker_options), args.execute, env)
            checkpoint = checkpoint_for(work_dir)
            for split in args.collect_splits:
                cmd, pred_path = collect_cmd(config, checkpoint, work_dir, split, args.num_workers)
                run(cmd, args.execute, env)
                out_csv = metrics_root / f'{args.dataset}_{model}_{method}_seed{seed}_{split}.csv'
                run(analyze_cmd(args.dataset, model, method, pred_path, out_csv), args.execute, env)
            eff_csv = efficiency_root / f'{args.dataset}_{model}_{method}_seed{seed}.csv'
            run(efficiency_cmd(args.dataset, model, method, seed, work_dir, eff_csv, args.target_accuracy), args.execute, env)


if __name__ == '__main__':
    main()
