import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Build a confusion-aware RCPS base from a validation paper.pkl file.')
    parser.add_argument('paper_pkl')
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--mode',
        choices=['mean-prob', 'hard-confusion'],
        default='mean-prob')
    parser.add_argument('--laplace', type=float, default=1e-4)
    args = parser.parse_args()

    with Path(args.paper_pkl).open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    labels = np.asarray(data['gts'], dtype=np.int64)
    num_classes = probs.shape[1]
    base = np.full((num_classes, num_classes), args.laplace, dtype=np.float64)

    if args.mode == 'mean-prob':
        for cls in range(num_classes):
            mask = labels == cls
            if np.any(mask):
                base[cls] += probs[mask].mean(axis=0)
    else:
        pred = probs.argmax(axis=1)
        for gt, pr in zip(labels, pred):
            base[gt, pr] += 1.0

    base = base / np.clip(base.sum(axis=1, keepdims=True), 1e-12, None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, base.astype(np.float32))
    print(f'Saved confusion base to {out}')


if __name__ == '__main__':
    main()
