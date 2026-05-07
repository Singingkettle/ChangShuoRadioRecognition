import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


METRIC_FIELDS = [
    'dataset', 'model', 'method', 'seed', 'reliability_bin',
    'accuracy', 'nll', 'ece', 'brier', 'mean_confidence',
    'mean_entropy', 'high_r_retention', 'low_r_calibration_gain'
]


def parse_run(spec):
    parts = spec.split('=')
    if len(parts) != 4:
        raise ValueError(
            'Run spec must be dataset=model=method=path/to/paper.pkl')
    dataset, model, method, path = parts
    return dataset, model, method, Path(path)


def expected_calibration_error(probs, labels, n_bins=15):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(probs, labels, n_bins=15):
    eps = 1e-12
    labels = labels.astype(np.int64)
    pred = probs.argmax(axis=1)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    clipped = np.clip(probs, eps, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return dict(
        accuracy=float((pred == labels).mean() * 100.0),
        nll=float(-np.log(clipped[np.arange(labels.size), labels]).mean()),
        ece=expected_calibration_error(clipped, labels, n_bins=n_bins),
        brier=float(np.square(clipped - one_hot).sum(axis=1).mean()),
        mean_confidence=float(clipped.max(axis=1).mean()),
        mean_entropy=float(entropy.mean()),
    )


def load_paper(path):
    with path.open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    labels = np.asarray(data['gts'], dtype=np.int64)
    snrs = np.asarray(data.get('snrs', np.zeros(labels.shape[0])), dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs, labels, snrs


def infer_seed(path):
    for part in path.parts:
        if part.startswith('seed_'):
            return part.replace('seed_', '')
    return ''


def add_rows(rows, dataset, model, method, path, n_bins):
    probs, labels, snrs = load_paper(path)
    seed = infer_seed(path)
    for snr in sorted(np.unique(snrs)):
        mask = snrs == snr
        row = dict(
            dataset=dataset,
            model=model,
            method=method,
            seed=seed,
            reliability_bin=str(int(snr)) if float(snr).is_integer() else str(snr),
            high_r_retention='',
            low_r_calibration_gain='',
        )
        row.update(compute_metrics(probs[mask], labels[mask], n_bins=n_bins))
        rows.append(row)

    row = dict(
        dataset=dataset,
        model=model,
        method=method,
        seed=seed,
        reliability_bin='all',
        high_r_retention='',
        low_r_calibration_gain='',
    )
    row.update(compute_metrics(probs, labels, n_bins=n_bins))
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(
        description='Compute RCPS reliability-stratified metrics from paper.pkl files.')
    parser.add_argument(
        '--run',
        action='append',
        required=True,
        help='Run descriptor: dataset=model=method=path/to/paper.pkl')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-json', default=None)
    parser.add_argument('--ece-bins', type=int, default=15)
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        dataset, model, method, path = parse_run(spec)
        add_rows(rows, dataset, model, method, path, args.ece_bins)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open('w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
