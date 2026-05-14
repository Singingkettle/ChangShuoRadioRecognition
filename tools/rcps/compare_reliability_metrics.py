import argparse
import csv
import re
from pathlib import Path


METRICS = [
    'accuracy',
    'nll',
    'ece',
    'brier',
    'mean_confidence',
    'mean_entropy',
]


def read_rows(paths):
    rows = []
    for path in paths:
        p = Path(path)
        with p.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                row['_source'] = p.as_posix()
                rows.append(row)
    return rows


def to_float(value):
    if value in {'', None}:
        return None
    return float(value)


def bin_value(label):
    if label == 'all':
        return None
    numbers = re.findall(r'-?\d+(?:\.\d+)?', label)
    if not numbers:
        return None
    return min(float(v) for v in numbers)


def mean(rows, field):
    vals = [to_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return ''
    return sum(vals) / len(vals)


def collect_groups(rows, low_max, high_min):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['model'], row['seed'], row['method'])
        item = groups.setdefault(key, {
            'all': None,
            'low': [],
            'high': [],
            'bins': [],
            'source': row['_source'],
        })
        if row['reliability_bin'] == 'all':
            item['all'] = row
            continue
        value = bin_value(row['reliability_bin'])
        if value is None:
            continue
        item['bins'].append(row)
        if value <= low_max:
            item['low'].append(row)
        if value >= high_min:
            item['high'].append(row)
    return groups


def prefix_metrics(out, prefix, rows_or_row):
    if isinstance(rows_or_row, dict):
        for field in METRICS:
            value = to_float(rows_or_row.get(field))
            out[f'{prefix}_{field}'] = '' if value is None else f'{value:.6f}'
        return
    for field in METRICS:
        value = mean(rows_or_row, field)
        out[f'{prefix}_{field}'] = '' if value == '' else f'{value:.6f}'


def add_delta(out, prefix, candidate, baseline):
    for field in METRICS:
        c = out.get(f'{prefix}_{field}', '')
        b = baseline.get(f'{prefix}_{field}', '')
        if c == '' or b == '':
            out[f'{prefix}_delta_{field}'] = ''
            continue
        out[f'{prefix}_delta_{field}'] = f'{float(c) - float(b):.6f}'


def make_summary(groups, baseline_method, ignore_model=False):
    baselines = {}
    summarized = {}
    for key, group in groups.items():
        dataset, model, seed, method = key
        if group['all'] is None:
            continue
        out = {
            'dataset': dataset,
            'model': model,
            'seed': seed,
            'method': method,
            'source': group['source'],
            'num_low_bins': len(group['low']),
            'num_high_bins': len(group['high']),
        }
        prefix_metrics(out, 'all', group['all'])
        prefix_metrics(out, 'low', group['low'])
        prefix_metrics(out, 'high', group['high'])
        summarized[key] = out
        if method == baseline_method:
            base_key = (dataset, seed) if ignore_model else (dataset, model, seed)
            baselines[base_key] = out

    rows = []
    for key, out in sorted(summarized.items()):
        dataset, model, seed, method = key
        base_key = (dataset, seed) if ignore_model else (dataset, model, seed)
        base = baselines.get(base_key)
        if base is None or method == baseline_method:
            for prefix in ['all', 'low', 'high']:
                for field in METRICS:
                    out[f'{prefix}_delta_{field}'] = ''
        else:
            for prefix in ['all', 'low', 'high']:
                add_delta(out, prefix, out, base)
        rows.append(out)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Compare reliability metrics against a same-seed baseline.')
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--baseline-method', default='hard-ce')
    parser.add_argument('--low-max', type=float, default=-10.0)
    parser.add_argument('--high-min', type=float, default=10.0)
    parser.add_argument('--ignore-model', action='store_true',
                        help='Match baselines by dataset and seed only. Use for alias-compatible backbones such as petcgdnn and petcgdnn_kerasinit.')
    args = parser.parse_args()

    rows = make_summary(
        collect_groups(read_rows(args.metrics), args.low_max, args.high_min),
        args.baseline_method,
        ignore_model=args.ignore_model,
    )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        'dataset', 'model', 'seed', 'method', 'source', 'num_low_bins',
        'num_high_bins',
    ]
    for prefix in ['all', 'low', 'high']:
        for field in METRICS:
            fields.append(f'{prefix}_{field}')
        for field in METRICS:
            fields.append(f'{prefix}_delta_{field}')

    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {out} with {len(rows)} rows')


if __name__ == '__main__':
    main()
