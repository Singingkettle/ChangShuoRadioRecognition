import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    'accuracy', 'nll', 'ece', 'brier', 'mean_confidence', 'mean_entropy',
    'high_r_retention', 'low_r_calibration_gain',
]


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    return rows


def summarize(values):
    values = [float(value) for value in values if value not in {'', None}]
    if not values:
        return '', ''
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f'{mean:.6f}', '0.000000'
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f'{mean:.6f}', f'{var ** 0.5:.6f}'


def main():
    parser = argparse.ArgumentParser(description='Aggregate RCPS reliability metrics across seeds.')
    parser.add_argument('--metric', action='append', required=True, help='CSV metric file')
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()

    groups = defaultdict(list)
    for row in read_rows(args.metric):
        key = (
            row.get('dataset', ''),
            row.get('model', ''),
            row.get('method', ''),
            row.get('reliability_bin', ''),
        )
        groups[key].append(row)

    out_rows = []
    for (dataset, model, method, reliability_bin), rows in sorted(groups.items()):
        out = {
            'dataset': dataset,
            'model': model,
            'method': method,
            'reliability_bin': reliability_bin,
            'num_seeds': len({row.get('seed', '') for row in rows}),
            'seeds': ' '.join(sorted({row.get('seed', '') for row in rows})),
        }
        for field in NUMERIC_FIELDS:
            mean, std = summarize([row.get(field, '') for row in rows])
            out[f'{field}_mean'] = mean
            out[f'{field}_std'] = std
        out_rows.append(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['dataset', 'model', 'method', 'reliability_bin', 'num_seeds', 'seeds']
    for field in NUMERIC_FIELDS:
        fields.extend([f'{field}_mean', f'{field}_std'])
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'Wrote {out_path} with {len(out_rows)} rows')


if __name__ == '__main__':
    main()
