import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


METRICS = [
    'accuracy', 'nll', 'ece', 'brier', 'mean_confidence', 'mean_entropy'
]


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
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(probs, labels, n_bins=15):
    eps = 1e-12
    labels = labels.astype(np.int64)
    clipped = np.clip(probs, eps, 1.0)
    clipped = clipped / np.clip(clipped.sum(axis=1, keepdims=True), eps, None)
    one_hot = np.eye(clipped.shape[1], dtype=np.float64)[labels]
    pred = clipped.argmax(axis=1)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return dict(
        accuracy=float((pred == labels).mean() * 100.0),
        nll=float(-np.log(clipped[np.arange(labels.size), labels]).mean()),
        ece=expected_calibration_error(clipped, labels, n_bins=n_bins),
        brier=float(np.square(clipped - one_hot).sum(axis=1).mean()),
        mean_confidence=float(clipped.max(axis=1).mean()),
        mean_entropy=float(entropy.mean()),
    )


def load_pkl(path):
    with Path(path).open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    labels = np.asarray(data['gts'], dtype=np.int64)
    return data, probs, labels


def temperature_scale_probs(probs, temperature):
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def fit_temperature(val_probs, val_labels, t_min, t_max, num_grid, n_bins):
    grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), num_grid))
    best = None
    for t in grid:
        scaled = temperature_scale_probs(val_probs, t)
        metrics = compute_metrics(scaled, val_labels, n_bins=n_bins)
        key = (metrics['nll'], metrics['ece'], abs(t - 1.0))
        if best is None or key < best[0]:
            best = (key, float(t), metrics)
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(
        description='Fit scalar temperature on validation probabilities and apply it to test probabilities.')
    parser.add_argument('--validation-pkl', required=True)
    parser.add_argument('--test-pkl', required=True)
    parser.add_argument('--out-pkl', required=True)
    parser.add_argument('--summary-csv', default=None)
    parser.add_argument('--t-min', type=float, default=0.5)
    parser.add_argument('--t-max', type=float, default=5.0)
    parser.add_argument('--num-grid', type=int, default=401)
    parser.add_argument('--ece-bins', type=int, default=15)
    args = parser.parse_args()

    _, val_probs, val_labels = load_pkl(args.validation_pkl)
    test_data, test_probs, test_labels = load_pkl(args.test_pkl)

    temperature, val_scaled_metrics = fit_temperature(
        val_probs, val_labels, args.t_min, args.t_max, args.num_grid, args.ece_bins)
    raw_val_metrics = compute_metrics(val_probs, val_labels, n_bins=args.ece_bins)
    raw_test_metrics = compute_metrics(test_probs, test_labels, n_bins=args.ece_bins)

    scaled_test_probs = temperature_scale_probs(test_probs, temperature)
    scaled_test_metrics = compute_metrics(scaled_test_probs, test_labels, n_bins=args.ece_bins)

    out_data = dict(test_data)
    out_data['pps'] = scaled_test_probs
    out_data['temperature'] = temperature
    out_data['temperature_source'] = str(args.validation_pkl)
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump(out_data, f)

    print(f'temperature={temperature:.6f}')
    print('validation raw', raw_val_metrics)
    print('validation scaled', val_scaled_metrics)
    print('test raw', raw_test_metrics)
    print('test scaled', scaled_test_metrics)
    print(f'wrote {out_path}')

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['temperature'] + [f'val_raw_{m}' for m in METRICS] + [f'val_scaled_{m}' for m in METRICS] + [f'test_raw_{m}' for m in METRICS] + [f'test_scaled_{m}' for m in METRICS]
        row = {'temperature': temperature}
        for prefix, metrics in [('val_raw', raw_val_metrics), ('val_scaled', val_scaled_metrics), ('test_raw', raw_test_metrics), ('test_scaled', scaled_test_metrics)]:
            for m in METRICS:
                row[f'{prefix}_{m}'] = metrics[m]
        with summary_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
