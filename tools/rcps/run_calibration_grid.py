import argparse
import subprocess
from pathlib import Path

from run_amc_matrix import CONFIGS


def fmt_float(value):
    return str(value).replace('.', 'p').replace('-', 'm')


def run(cmd, execute):
    print(' '.join(str(item) for item in cmd))
    if execute:
        subprocess.run([str(item) for item in cmd], check=True)


def worker_options(num_workers):
    if num_workers is None:
        return []
    opts = [
        f'train_dataloader.num_workers={num_workers}',
        f'val_dataloader.num_workers={num_workers}',
        f'test_dataloader.num_workers={num_workers}',
    ]
    if num_workers == 0:
        opts.extend([
            'train_dataloader.persistent_workers=False',
            'val_dataloader.persistent_workers=False',
            'test_dataloader.persistent_workers=False',
        ])
    return opts


def split_worker_options(split, num_workers):
    if num_workers is None:
        return []
    dataloader = 'val_dataloader' if split in {'validation', 'val'} else f'{split}_dataloader'
    opts = [f'{dataloader}.num_workers={num_workers}']
    if num_workers == 0:
        opts.append(f'{dataloader}.persistent_workers=False')
    return opts


def checkpoint_for(work_dir):
    matches = sorted(work_dir.glob('best_accuracy_top1_epoch_*.pth'))
    if matches:
        return matches[-1]
    matches = sorted(work_dir.glob('epoch_*.pth'))
    if matches:
        return matches[-1]
    return work_dir / 'best_accuracy_top1_epoch_*.pth'


def train_cmd(config, work_dir, seed, max_epochs, cfg_options):
    options = [f'randomness.seed={seed}', f'work_dir={work_dir.as_posix()}']
    if max_epochs is not None:
        options.append(f'train_cfg.max_epochs={max_epochs}')
    options.extend(cfg_options)
    return ['python', 'tools/train.py', config, '--cfg-options', *options]


def collect_cmd(config, checkpoint, work_dir, split, num_workers):
    pred_path = work_dir / 'predictions' / f'{split}.pkl'
    cmd = [
        'python', 'tools/rcps/collect_predictions.py', config, checkpoint.as_posix(),
        '--split', split, '--work-dir', work_dir.as_posix(), '--out', pred_path.as_posix(),
    ]
    opts = split_worker_options(split, num_workers)
    if opts:
        cmd.extend(['--cfg-options', *opts])
    return cmd, pred_path


def analyze_cmd(dataset, model, method, pred_path, out_csv):
    return [
        'python', 'tools/rcps/analyze_reliability.py',
        '--run', f'{dataset}={model}={method}={pred_path.as_posix()}',
        '--out-csv', out_csv.as_posix(),
    ]


def loss_option_prefixes(model):
    if model == 'mldnn':
        return ['model.head.loss_amc_merge', 'model.head.loss_amc_ap', 'model.head.loss_amc_iq']
    return ['model.head.loss']


def candidate_specs(model, method, epsilon_max, epsilon_gamma, smoothing):
    prefixes = loss_option_prefixes(model)
    if method == 'hard-ce':
        yield method, 'hard', []
    elif method == 'static-ls':
        for value in smoothing:
            suffix = f'ls{fmt_float(value)}'
            yield method, suffix, [f'{prefix}.smoothing={value}' for prefix in prefixes]
    elif method.startswith('rcps-'):
        for eps in epsilon_max:
            for gamma in epsilon_gamma:
                suffix = f'eps{fmt_float(eps)}_g{fmt_float(gamma)}'
                opts = []
                for prefix in prefixes:
                    opts.extend([f'{prefix}.epsilon.max={eps}', f'{prefix}.epsilon.gamma={gamma}'])
                yield method, suffix, opts
    else:
        raise KeyError(f'Unsupported calibration method: {method}')


def main():
    parser = argparse.ArgumentParser(description='Run RCPS/label-smoothing calibration grids.')
    parser.add_argument('--models', nargs='+', default=['mcldnn', 'cgdnet', 'petcgdnn', 'mcformer'])
    parser.add_argument('--methods', nargs='+', default=['hard-ce', 'static-ls', 'rcps-uniform'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026])
    parser.add_argument('--dataset', default='deepsig201610A')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs')
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--epsilon-max', nargs='+', type=float, default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument('--epsilon-gamma', nargs='+', type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument('--smoothing', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--collect-splits', nargs='+', default=['validation'])
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    train_worker_opts = worker_options(args.num_workers)
    for model in args.models:
        for method in args.methods:
            config = CONFIGS[(model, method)]
            for seed in args.seeds:
                for base_method, suffix, method_options in candidate_specs(
                        model, method, args.epsilon_max, args.epsilon_gamma, args.smoothing):
                    run_name = f'{model}_{base_method}_{suffix}'
                    work_dir = Path(args.work_root) / 'amc' / args.dataset / run_name / f'seed_{seed}'
                    cfg_options = [*method_options, *train_worker_opts]
                    run(train_cmd(config, work_dir, seed, args.max_epochs, cfg_options), args.execute)
                    if not args.collect_splits:
                        continue
                    checkpoint = checkpoint_for(work_dir)
                    for split in args.collect_splits:
                        cmd, pred_path = collect_cmd(config, checkpoint, work_dir, split, args.num_workers)
                        run(cmd, args.execute)
                        if args.analyze:
                            out_dir = Path(args.work_root) / 'metrics' / 'calibration'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_csv = out_dir / f'{args.dataset}_{run_name}_seed{seed}_{split}.csv'
                            run(analyze_cmd(args.dataset, model, f'{base_method}_{suffix}', pred_path, out_csv),
                                args.execute)


if __name__ == '__main__':
    main()
