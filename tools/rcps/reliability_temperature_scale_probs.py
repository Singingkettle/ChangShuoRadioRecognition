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
    if 'snrs' not in data:
        raise KeyError('Reliability-conditioned calibration requires snrs metadata.')
    snrs = np.asarray(data['snrs'], dtype=np.float64)
    return data, probs, labels, snrs


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


def assign_bins(snrs, edges):
    if edges is None or len(edges) == 0:
        return snrs.astype(np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    idx = np.digitize(snrs, edges, right=True)
    return idx.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description='Fit reliability-bin temperatures on validation predictions and apply them to test predictions.')
    parser.add_argument('--validation-pkl', required=True)
    parser.add_argument('--test-pkl', required=True)
    parser.add_argument('--out-pkl', required=True)
    parser.add_argument('--summary-csv', default=None)
    parser.add_argument('--bin-edges', type=float, nargs='*', default=None,
                        help='Optional SNR edges for grouped calibration. Omit to fit one temperature per SNR value.')
    parser.add_argument('--t-min', type=float, default=0.2)
    parser.add_argument('--t-max', type=float, default=3.0)
    parser.add_argument('--num-grid', type=int, default=561)
    parser.add_argument('--ece-bins', type=int, default=15)
    args = parser.parse_args()

    _, val_probs, val_labels, val_snrs = load_pkl(args.validation_pkl)
    test_data, test_probs, test_labels, test_snrs = load_pkl(args.test_pkl)

    val_bins = assign_bins(val_snrs, args.bin_edges)
    test_bins = assign_bins(test_snrs, args.bin_edges)
    unique_bins = np.unique(val_bins)
    scaled_test = np.zeros_like(test_probs)
    rows = []

    for bin_value in unique_bins:
        val_mask = val_bins == bin_value
        if not np.any(val_mask):
            continue
        t, val_scaled = fit_temperature(
            val_probs[val_mask], val_labels[val_mask],
            args.t_min, args.t_max, args.num_grid, args.ece_bins)
        test_mask = test_bins == bin_value
        if np.any(test_mask):
            scaled_test[test_mask] = temperature_scale_probs(test_probs[test_mask], t)
            test_raw = compute_metrics(test_probs[test_mask], test_labels[test_mask], args.ece_bins)
            test_scaled = compute_metrics(scaled_test[test_mask], test_labels[test_mask], args.ece_bins)
        else:
            test_raw = {m: '' for m in METRICS}
            test_scaled = {m: '' for m in METRICS}
        raw_val = compute_metrics(val_probs[val_mask], val_labels[val_mask], args.ece_bins)
        row = dict(bin=bin_value, temperature=t, validation_count=int(val_mask.sum()),
                   test_count=int(test_mask.sum()))
        for prefix, metrics in [('val_raw', raw_val), ('val_scaled', val_scaled),
                                ('test_raw', test_raw), ('test_scaled', test_scaled)]:
            for metric in METRICS:
                row[f'{prefix}_{metric}'] = metrics[metric]
        rows.append(row)

    missing_test = ~np.isin(test_bins, unique_bins)
    if np.any(missing_test):
        raise ValueError('Test contains reliability bins absent from validation.')

    out_data = dict(test_data)
    out_data['pps'] = scaled_test
    out_data['temperature_source'] = str(args.validation_pkl)
    out_data['temperature_mode'] = 'reliability_bin'
    out_data['temperature_bin_edges'] = args.bin_edges
    out_data['temperature_table'] = {
        str(row['bin']): row['temperature']
        for row in rows
    }
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump(out_data, f)

    print('test raw', compute_metrics(test_probs, test_labels, args.ece_bins))
    print('test scaled', compute_metrics(scaled_test, test_labels, args.ece_bins))
    print(f'wrote {out_path}')

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else ['bin', 'temperature']
        with summary_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
