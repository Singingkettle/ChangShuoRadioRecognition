import argparse
import csv
import subprocess
from pathlib import Path

from run_amc_matrix import CONFIGS
from run_calibration_grid import checkpoint_for, collect_cmd, split_worker_options, train_cmd, worker_options


def run(cmd, execute):
    print(' '.join(str(item) for item in cmd), flush=True)
    if execute:
        subprocess.run([str(item) for item in cmd], check=True)


def read_selected(path, models, families):
    models = set(models) if models else None
    families = set(families) if families else None
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows:
        if models is not None and row['model'] not in models:
            continue
        if families is not None and row['family'] not in families:
            continue
        selected.append(row)
    return selected


def method_options(row):
    opts = []
    if row['config_method'] == 'static-ls' and row.get('smoothing'):
        opts.append(f'model.head.loss.smoothing={float(row["smoothing"])}')
    if row['config_method'].startswith('rcps-'):
        if row.get('epsilon_max'):
            opts.append(f'model.head.loss.epsilon.max={float(row["epsilon_max"])}')
        if row.get('epsilon_gamma'):
            opts.append(f'model.head.loss.epsilon.gamma={float(row["epsilon_gamma"])}')
    return opts


def analyze_cmd(dataset, model, method, pred_path, out_csv):
    return [
        'python', 'tools/rcps/analyze_reliability.py',
        '--run', f'{dataset}={model}={method}={pred_path.as_posix()}',
        '--out-csv', out_csv.as_posix(),
    ]


def main():
    parser = argparse.ArgumentParser(description='Run main AMC experiments from selected calibration configs.')
    parser.add_argument('--selected', default='/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed/selected_main_configs.csv')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--families', nargs='+', default=['hard-ce', 'static-ls', 'rcps-uniform', 'rcps-confusion'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--dataset', default='deepsig201610A')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--collect-splits', nargs='+', default=['validation', 'test'])
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    train_worker_opts = worker_options(args.num_workers)
    selected = read_selected(args.selected, args.models, args.families)
    if not selected:
        raise SystemExit(f'No selected configs matched filters from {args.selected}')

    metrics_root = Path(args.work_root) / 'metrics' / 'main'
    for row in selected:
        model = row['model']
        config_method = row['config_method']
        method = row['method']
        config = CONFIGS[(model, config_method)]
        for seed in args.seeds:
            run_name = f'{model}_{method}'
            work_dir = Path(args.work_root) / 'amc' / args.dataset / run_name / f'seed_{seed}'
            split_csvs = [
                metrics_root / f'{args.dataset}_{run_name}_seed{seed}_{split}.csv'
                for split in args.collect_splits
            ]
            if args.skip_existing and split_csvs and all(path.exists() for path in split_csvs):
                print(f'Skip existing metrics for {run_name} seed {seed}', flush=True)
                continue

            cfg_options = [*method_options(row), *train_worker_opts]
            run(train_cmd(config, work_dir, seed, args.max_epochs, cfg_options), args.execute)
            checkpoint = checkpoint_for(work_dir)
            for split in args.collect_splits:
                cmd, pred_path = collect_cmd(config, checkpoint, work_dir, split, args.num_workers)
                run(cmd, args.execute)
                if args.analyze:
                    metrics_root.mkdir(parents=True, exist_ok=True)
                    out_csv = metrics_root / f'{args.dataset}_{run_name}_seed{seed}_{split}.csv'
                    run(analyze_cmd(args.dataset, model, method, pred_path, out_csv), args.execute)


if __name__ == '__main__':
    main()
