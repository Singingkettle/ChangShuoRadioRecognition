import argparse
import pickle
from pathlib import Path

import numpy as np


def map_snr_to_reliability(snrs, snr_min, snr_max):
    if np.isclose(snr_max, snr_min):
        raise ValueError('snr_max must differ from snr_min.')
    return np.clip((snrs - snr_min) / (snr_max - snr_min), 0.0, 1.0)


def load_paper(path):
    with Path(path).open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    labels = np.asarray(data['gts'], dtype=np.int64)
    snrs = np.asarray(data['snrs'], dtype=np.float64)
    return probs, labels, snrs


def main():
    parser = argparse.ArgumentParser(
        description='Build reliability-conditioned class posterior bases from validation predictions.')
    parser.add_argument('paper_pkl', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['mean-prob', 'hard-confusion'], default='mean-prob')
    parser.add_argument('--snr-min', type=float, default=-20.0)
    parser.add_argument('--snr-max', type=float, default=18.0)
    parser.add_argument('--laplace', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    probs_list, labels_list, snrs_list = [], [], []
    for paper in args.paper_pkl:
        probs, labels, snrs = load_paper(paper)
        probs_list.append(probs)
        labels_list.append(labels)
        snrs_list.append(snrs)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    snrs = np.concatenate(snrs_list, axis=0)

    if probs.ndim != 2:
        raise ValueError(f'Expected probabilities with shape (N, C), got {probs.shape}.')
    num_classes = probs.shape[1]
    raw_bins = np.array(sorted(np.unique(snrs)), dtype=np.float64)
    bins = map_snr_to_reliability(raw_bins, args.snr_min, args.snr_max).astype(np.float32)
    base = np.full((len(raw_bins), num_classes, num_classes), args.laplace, dtype=np.float64)
    counts = np.zeros((len(raw_bins), num_classes), dtype=np.int64)

    for b_idx, snr in enumerate(raw_bins):
        snr_mask = snrs == snr
        for cls in range(num_classes):
            mask = snr_mask & (labels == cls)
            counts[b_idx, cls] = int(mask.sum())
            if not np.any(mask):
                continue
            if args.mode == 'mean-prob':
                base[b_idx, cls] += probs[mask].mean(axis=0)
            else:
                pred = probs[mask].argmax(axis=1)
                for pr in pred:
                    base[b_idx, cls, pr] += 1.0

    if not np.isclose(args.temperature, 1.0):
        base = np.power(np.clip(base, 1e-12, None), 1.0 / args.temperature)
    base = base / np.clip(base.sum(axis=-1, keepdims=True), 1e-12, None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        base=base.astype(np.float32),
        bins=bins.astype(np.float32),
        raw_bins=raw_bins.astype(np.float32),
        counts=counts)
    print(f'Saved reliability-conditioned base to {out}')
    print(f'  sources: {len(args.paper_pkl)}')
    print(f'  samples: {len(labels)}')
    print(f'  classes: {num_classes}')
    print(f'  raw bins: {raw_bins.tolist()}')


if __name__ == '__main__':
    main()
