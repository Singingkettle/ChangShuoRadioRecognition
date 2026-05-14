import argparse
import pickle
from pathlib import Path

import numpy as np


def load_predictions(path):
    with Path(path).open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    snrs = np.asarray(data.get('snrs'), dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs, snrs


def target_entropy(eps, num_classes):
    q_y = (1.0 - eps) + eps / num_classes
    q_o = eps / num_classes
    if eps <= 0.0:
        return 0.0
    values = [q_y] + [q_o] * (num_classes - 1)
    return -sum(q * np.log(max(q, 1e-12)) for q in values)


def invert_entropy(value, num_classes, eps_max):
    lo, hi = 0.0, eps_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if target_entropy(mid, num_classes) < value:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    parser = argparse.ArgumentParser(description='Build an entropy-matched RCPS epsilon table from validation predictions.')
    parser.add_argument('--pred', required=True, help='Validation prediction pkl from collect_predictions.py')
    parser.add_argument('--out', required=True, help='Output .npz containing bins and values')
    parser.add_argument('--snr-min', type=float, default=-20.0)
    parser.add_argument('--snr-max', type=float, default=18.0)
    parser.add_argument('--epsilon-max', type=float, default=0.7)
    parser.add_argument('--retain-min', type=float, default=0.8)
    parser.add_argument('--min-epsilon', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    probs, snrs = load_predictions(args.pred)
    if snrs is None:
        raise ValueError('Prediction file does not contain snrs.')
    num_classes = probs.shape[1]
    clipped = np.clip(probs, 1e-12, 1.0)
    if args.temperature != 1.0:
        clipped = np.power(clipped, 1.0 / args.temperature)
        clipped = clipped / np.clip(clipped.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    max_entropy = np.log(num_classes)

    bins = []
    values = []
    rows = []
    for snr in sorted(np.unique(snrs)):
        mask = snrs == snr
        reliability = np.clip((snr - args.snr_min) / (args.snr_max - args.snr_min), 0.0, 1.0)
        desired = float(np.clip(entropy[mask].mean(), 0.0, max_entropy))
        eps = invert_entropy(desired, num_classes, args.epsilon_max)
        eps = max(args.min_epsilon, min(args.epsilon_max, eps))
        if reliability >= args.retain_min:
            eps = 0.0
        bins.append(float(reliability))
        values.append(float(eps))
        rows.append((float(snr), float(reliability), desired, float(eps)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bins=np.asarray(bins, dtype=np.float32), values=np.asarray(values, dtype=np.float32), rows=np.asarray(rows, dtype=np.float32))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
