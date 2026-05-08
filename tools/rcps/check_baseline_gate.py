import argparse
import csv
from pathlib import Path


def read_registry(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_metric(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for key in ('all', 'overall'):
        for row in rows:
            if row.get('reliability_bin') == key:
                return row
    if len(rows) == 1:
        return rows[0]
    raise ValueError(f'Cannot find all/overall row in {path}')


def main():
    parser = argparse.ArgumentParser(description='Check hard-label baselines against the RCPS registry.')
    parser.add_argument('--registry', default='docs/rcps/baseline_reference_registry.csv')
    parser.add_argument('--metric', action='append', default=[], help='model=dataset=path/to/reliability_metrics.csv')
    parser.add_argument('--out-csv', default='docs/rcps/baseline_gate_report.csv')
    args = parser.parse_args()

    metrics = {}
    for spec in args.metric:
        parts = spec.split('=', 2)
        if len(parts) != 3:
            raise SystemExit(f'Invalid --metric spec: {spec}')
        model, dataset, path = parts
        row = read_metric(path)
        metrics[(model, dataset)] = dict(row, _path=path)

    report = []
    for row in read_registry(args.registry):
        threshold = row.get('threshold_accuracy', '').strip()
        metric = metrics.get((row['model'], row['dataset']))
        if not threshold:
            status = 'reference_only'
            observed = '' if metric is None else metric.get('accuracy', '')
            margin = ''
        elif metric is None:
            status = 'missing_metric'
            observed = ''
            margin = ''
        else:
            observed_value = float(metric['accuracy'])
            threshold_value = float(threshold)
            observed = f'{observed_value:.6f}'
            margin_value = observed_value - threshold_value
            margin = f'{margin_value:.6f}'
            status = 'pass' if margin_value >= 0.0 else 'fail'
        report.append({
            'scenario': row['scenario'],
            'dataset': row['dataset'],
            'model': row['model'],
            'reference': row['reference'],
            'threshold_accuracy': threshold,
            'observed_accuracy': observed,
            'margin': margin,
            'gate_status': status,
            'metric_path': '' if metric is None else metric['_path'],
            'notes': row.get('notes', ''),
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['scenario', 'dataset', 'model', 'reference', 'threshold_accuracy', 'observed_accuracy', 'margin', 'gate_status', 'metric_path', 'notes']
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)
    for item in report:
        print(f"{item['dataset']} {item['model']}: {item['gate_status']} {item['observed_accuracy']}")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
