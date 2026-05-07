import argparse
import csv
from pathlib import Path


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                row['_source'] = str(path)
                rows.append(row)
    return rows


def as_float(row, key):
    return float(row[key])


def summarize(rows, high_min):
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['model'], row['seed'], row['method'])
        grouped.setdefault(key, {'bins': [], 'all': None, 'source': row['_source']})
        if row['reliability_bin'] == 'all':
            grouped[key]['all'] = row
        else:
            grouped[key]['bins'].append(row)

    summaries = {}
    for key, value in grouped.items():
        high_bins = [
            row for row in value['bins']
            if row['reliability_bin'] != 'all' and float(row['reliability_bin']) >= high_min
        ]
        high_acc = None
        if high_bins:
            high_acc = sum(as_float(row, 'accuracy') for row in high_bins) / len(high_bins)
        summaries[key] = {
            'all': value['all'],
            'high_acc': high_acc,
            'source': value['source'],
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description='Select calibration candidates by validation NLL/ECE.')
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--baseline-method', default='hard-ce_hard')
    parser.add_argument('--high-min', type=float, default=10.0)
    parser.add_argument('--max-high-drop', type=float, default=1.0)
    args = parser.parse_args()

    summaries = summarize(read_rows(args.metrics), args.high_min)
    baselines = {}
    for (dataset, model, seed, method), summary in summaries.items():
        if method == args.baseline_method:
            baselines[(dataset, model, seed)] = summary

    selected = []
    candidates = {}
    for (dataset, model, seed, method), summary in summaries.items():
        if method == args.baseline_method or summary['all'] is None:
            continue
        baseline = baselines.get((dataset, model, seed))
        if baseline is None or baseline['high_acc'] is None or summary['high_acc'] is None:
            retained = True
            high_drop = ''
        else:
            high_drop_value = baseline['high_acc'] - summary['high_acc']
            retained = high_drop_value <= args.max_high_drop
            high_drop = f'{high_drop_value:.6f}'
        if not retained:
            continue
        key = (dataset, model, seed)
        row = summary['all']
        item = {
            'dataset': dataset,
            'model': model,
            'seed': seed,
            'method': method,
            'nll': row['nll'],
            'ece': row['ece'],
            'accuracy': row['accuracy'],
            'high_accuracy_drop': high_drop,
            'source': summary['source'],
        }
        candidates.setdefault(key, []).append(item)

    for key, items in candidates.items():
        items.sort(key=lambda item: (float(item['nll']), float(item['ece'])))
        selected.append(items[0])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['dataset', 'model', 'seed', 'method', 'nll', 'ece', 'accuracy',
              'high_accuracy_drop', 'source']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f'Wrote {out_path} with {len(selected)} selected candidates')


if __name__ == '__main__':
    main()
