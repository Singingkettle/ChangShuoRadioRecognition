import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


VAL_RE = re.compile(
    r'(?P<time>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}).*?'
    r'Epoch\(val\) \[(?P<epoch>\d+)\].*?'
    r'accuracy/top1:\s*(?P<acc>[-+0-9.]+)'
    r'(?:.*?loss/classification:\s*(?P<loss>[-+0-9.]+))?')


def parse_time(value):
    return datetime.strptime(value, '%Y/%m/%d %H:%M:%S')


def latest_log(work_dir):
    logs = sorted(Path(work_dir).glob('20*/20*.log'), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f'No MMEngine log found under {work_dir}')
    return logs[-1]


def parse_validation_log(log_path):
    points = []
    for line in Path(log_path).read_text(errors='ignore').splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        points.append({
            'time': parse_time(match.group('time')),
            'epoch': int(match.group('epoch')),
            'accuracy': float(match.group('acc')),
            'loss': '' if match.group('loss') is None else float(match.group('loss')),
        })
    if not points:
        raise ValueError(f'No validation accuracy points found in {log_path}')
    return points


def summarize(points, target_accuracy=None):
    best = max(points, key=lambda item: item['accuracy'])
    first_time = points[0]['time']
    last_time = points[-1]['time']
    wall_seconds = max(0.0, (last_time - first_time).total_seconds())
    target = float(target_accuracy) if target_accuracy is not None else 0.95 * best['accuracy']
    reached = [item for item in points if item['accuracy'] >= target]
    first_reached = reached[0] if reached else None
    if len(points) > 1:
        epoch_seconds = wall_seconds / max(1, points[-1]['epoch'] - points[0]['epoch'])
    else:
        epoch_seconds = 0.0
    mean_val_acc = sum(item['accuracy'] for item in points) / len(points)
    return {
        'num_val_points': len(points),
        'first_epoch': points[0]['epoch'],
        'last_epoch': points[-1]['epoch'],
        'best_epoch': best['epoch'],
        'best_val_accuracy': best['accuracy'],
        'final_val_accuracy': points[-1]['accuracy'],
        'target_accuracy': target,
        'epoch_to_target': '' if first_reached is None else first_reached['epoch'],
        'seconds_to_target': '' if first_reached is None else max(0.0, (first_reached['time'] - first_time).total_seconds()),
        'validation_aulc': mean_val_acc,
        'wall_seconds_observed': wall_seconds,
        'mean_seconds_per_epoch': epoch_seconds,
    }


def parse_run(spec):
    parts = spec.split('=', 4)
    if len(parts) != 5:
        raise ValueError('Run must be dataset=model=method=seed=work_dir')
    dataset, model, method, seed, work_dir = parts
    return dataset, model, method, seed, Path(work_dir)


def main():
    parser = argparse.ArgumentParser(description='Summarize validation-curve training efficiency from MMEngine logs.')
    parser.add_argument('--run', action='append', required=True, help='dataset=model=method=seed=work_dir')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--target-accuracy', type=float, default=None)
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        dataset, model, method, seed, work_dir = parse_run(spec)
        log_path = latest_log(work_dir)
        row = {
            'dataset': dataset,
            'model': model,
            'method': method,
            'seed': seed,
            'work_dir': work_dir.as_posix(),
            'log_path': log_path.as_posix(),
        }
        row.update(summarize(parse_validation_log(log_path), args.target_accuracy))
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'dataset', 'model', 'method', 'seed', 'work_dir', 'log_path',
        'num_val_points', 'first_epoch', 'last_epoch', 'best_epoch',
        'best_val_accuracy', 'final_val_accuracy', 'target_accuracy',
        'epoch_to_target', 'seconds_to_target', 'validation_aulc',
        'wall_seconds_observed', 'mean_seconds_per_epoch',
    ]
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
