import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


def load_predictions(path):
    with Path(path).open('rb') as f:
        data = pickle.load(f)
    probs = np.asarray(data['pps'], dtype=np.float64)
    labels = np.asarray(data['gts'], dtype=np.int64)
    reliability = np.asarray(data.get('snrs', np.zeros(labels.shape)), dtype=np.float64)
    if 'sample_idx' in data:
        sample_idx = np.asarray(data['sample_idx'], dtype=np.int64)
    else:
        sample_idx = np.arange(labels.shape[0], dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f'Expected probabilities with shape (N, C), got {probs.shape}.')
    if not (probs.shape[0] == labels.shape[0] == reliability.shape[0] == sample_idx.shape[0]):
        raise ValueError('Prediction arrays have inconsistent first dimensions.')
    return probs, labels, reliability, sample_idx, data.get('split', '')


def normalize(probs, eps=1e-12):
    probs = np.clip(probs, eps, None)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)


def apply_temperature(probs, temperature):
    if np.isclose(temperature, 1.0):
        return normalize(probs)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    return normalize(np.exp(logits))


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description='Build sample-level posterior artifact for DPC-RCPS from collected predictions.')
    parser.add_argument('prediction_pkl', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--teacher-config', default='')
    parser.add_argument('--split', default='')
    parser.add_argument('--allow-duplicates', action='store_true')
    args = parser.parse_args()

    probs_list, labels_list, rel_list, idx_list, split_list = [], [], [], [], []
    for path in args.prediction_pkl:
        probs, labels, reliability, sample_idx, split = load_predictions(path)
        probs_list.append(probs)
        labels_list.append(labels)
        rel_list.append(reliability)
        idx_list.append(sample_idx)
        split_list.append(split)

    probs = apply_temperature(np.concatenate(probs_list, axis=0), args.temperature)
    labels = np.concatenate(labels_list, axis=0)
    reliability = np.concatenate(rel_list, axis=0)
    sample_idx = np.concatenate(idx_list, axis=0)

    if not args.allow_duplicates and len(np.unique(sample_idx)) != sample_idx.shape[0]:
        raise ValueError('Duplicate sample_idx values detected. Use --allow-duplicates only for diagnostics.')

    order = np.argsort(sample_idx, kind='mergesort')
    sample_idx = sample_idx[order]
    labels = labels[order]
    reliability = reliability[order]
    probs = probs[order]
    logits = np.log(np.clip(probs, 1e-12, 1.0))

    metadata = dict(
        split=args.split or ','.join(sorted({str(s) for s in split_list if s})),
        temperature=float(args.temperature),
        teacher_config=args.teacher_config,
        teacher_commit=git_commit(),
        sources=[str(Path(p)) for p in args.prediction_pkl],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_idx=sample_idx.astype(np.int64),
        label=labels.astype(np.int64),
        reliability=reliability.astype(np.float32),
        probs=probs.astype(np.float32),
        logits=logits.astype(np.float32),
        temperature=np.array([args.temperature], dtype=np.float32),
        metadata=np.array(json.dumps(metadata, sort_keys=True)))
    print(f'Saved sample posterior artifact to {out}')
    print(f'  samples: {sample_idx.shape[0]}')
    print(f'  classes: {probs.shape[1]}')
    print(f'  temperature: {args.temperature}')
    print(f'  split: {metadata["split"]}')


if __name__ == '__main__':
    main()
