import argparse
import csv
import re
from pathlib import Path


FAMILIES = ['hard-ce', 'static-ls', 'rcps-uniform', 'rcps-confusion']
FLOAT_TOKEN = re.compile(r'(?P<kind>eps|g|ls)(?P<value>[0-9]+p[0-9]+|[0-9]+)')


def read_metric_file(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    all_row = next(row for row in rows if row['reliability_bin'] == 'all')
    high_rows = [
        row for row in rows
        if row['reliability_bin'] != 'all' and float(row['reliability_bin']) >= 10.0
    ]
    high_acc = None
    if high_rows:
        high_acc = sum(float(row['accuracy']) for row in high_rows) / len(high_rows)
    return all_row, high_acc


def parse_value(token):
    return float(token.replace('p', '.'))


def parse_method(method):
    params = {'smoothing': '', 'epsilon_max': '', 'epsilon_gamma': ''}
    for match in FLOAT_TOKEN.finditer(method):
        kind = match.group('kind')
        value = parse_value(match.group('value'))
        if kind == 'ls':
            params['smoothing'] = value
        elif kind == 'eps':
            params['epsilon_max'] = value
        elif kind == 'g':
            params['epsilon_gamma'] = value
    if method == 'hard-ce_hard':
        config_method = 'hard-ce'
        family = 'hard-ce'
    elif method.startswith('static-ls_'):
        config_method = 'static-ls'
        family = 'static-ls'
    elif method.startswith('rcps-uniform_'):
        config_method = 'rcps-uniform'
        family = 'rcps-uniform'
    elif method.startswith('rcps-confusion_'):
        config_method = 'rcps-confusion'
        family = 'rcps-confusion'
    else:
        raise ValueError(f'Unsupported method name: {method}')
    params.update(config_method=config_method, family=family)
    return params


def collect_candidates(root):
    candidates = []
    for path in sorted(Path(root).glob('deepsig201610A_*_seed2026_validation.csv')):
        all_row, high_acc = read_metric_file(path)
        parsed = parse_method(all_row['method'])
        item = {
            'dataset': all_row['dataset'],
            'model': all_row['model'],
            'seed': all_row['seed'],
            'family': parsed['family'],
            'method': all_row['method'],
            'config_method': parsed['config_method'],
            'smoothing': parsed['smoothing'],
            'epsilon_max': parsed['epsilon_max'],
            'epsilon_gamma': parsed['epsilon_gamma'],
            'accuracy': float(all_row['accuracy']),
            'nll': float(all_row['nll']),
            'ece': float(all_row['ece']),
            'high_acc': high_acc,
            'source': str(path),
        }
        candidates.append(item)
    return candidates


def choose_by_family(candidates, max_high_drop):
    baselines = {}
    for item in candidates:
        if item['family'] == 'hard-ce':
            baselines[item['model']] = item

    selected = []
    for model in sorted({item['model'] for item in candidates}):
        baseline = baselines.get(model)
        if baseline is None:
            continue
        for family in FAMILIES:
            family_items = [item for item in candidates if item['model'] == model and item['family'] == family]
            if not family_items:
                continue
            for item in family_items:
                if item['high_acc'] is None or baseline['high_acc'] is None:
                    item['high_accuracy_drop'] = ''
                    item['retention_pass'] = True
                else:
                    drop = baseline['high_acc'] - item['high_acc']
                    item['high_accuracy_drop'] = drop
                    item['retention_pass'] = drop <= max_high_drop
            retained = [item for item in family_items if item['retention_pass']]
            pool = retained if retained else family_items
            pool.sort(key=lambda item: (float(item['nll']), float(item['ece'])))
            chosen = dict(pool[0])
            chosen['retention_pass'] = str(bool(chosen['retention_pass']))
            if isinstance(chosen['high_accuracy_drop'], float):
                chosen['high_accuracy_drop'] = f'{chosen["high_accuracy_drop"]:.6f}'
            selected.append(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description='Select main-run configs from RCPS calibration metrics.')
    parser.add_argument('--calibration-root', default='/home/citybuster/Data/RCPS/work_dirs/calibration_10ep/metrics/calibration')
    parser.add_argument('--confusion-root', default='/home/citybuster/Data/RCPS/work_dirs/calibration_10ep_confusion/metrics/calibration')
    parser.add_argument('--out-csv', default='/home/citybuster/Data/RCPS/work_dirs/main_10ep_3seed/selected_main_configs.csv')
    parser.add_argument('--max-high-drop', type=float, default=1.0)
    args = parser.parse_args()

    candidates = collect_candidates(args.calibration_root) + collect_candidates(args.confusion_root)
    selected = choose_by_family(candidates, args.max_high_drop)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'dataset', 'model', 'seed', 'family', 'method', 'config_method', 'smoothing',
        'epsilon_max', 'epsilon_gamma', 'accuracy', 'nll', 'ece',
        'high_acc', 'high_accuracy_drop', 'retention_pass', 'source'
    ]
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f'Wrote {out} with {len(selected)} selected rows')


if __name__ == '__main__':
    main()
