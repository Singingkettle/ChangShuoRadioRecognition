#!/usr/bin/env python3
"""Build strict clean-paired AWGN-DPC sample-posterior targets.

The artifact is compatible with RCPSCrossEntropyLoss base.type='sample_posterior'.
Teacher predictions should be collected on high-SNR paired views from the same
clean_id. For each target sample, this script averages teacher posteriors over
available high-SNR views sharing that clean_id.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


def load_annotation(path):
    payload = json.loads(Path(path).read_text())
    rows = payload["data_list"]
    out = {}
    for item in rows:
        for key in ("sample_idx", "clean_id", "clean_file_name", "modulation", "snr"):
            if key not in item:
                raise ValueError(f"Strict AWGN-DPC requires annotation key {key}; missing in {path}")
        out[int(item["sample_idx"])] = item
    return out


def load_predictions(path):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    probs = np.asarray(payload["pps"], dtype=np.float64)
    labels = np.asarray(payload["gts"], dtype=np.int64)
    sample_idx = np.asarray(payload.get("sample_idx", np.arange(labels.shape[0])), dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0] or labels.shape[0] != sample_idx.shape[0]:
        raise ValueError(f"Invalid prediction payload shapes in {path}")
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return sample_idx, labels, probs


def apply_temperature(probs, temperature):
    if np.isclose(temperature, 1.0):
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(args):
    ann = {}
    target_indices = []
    for ann_file in args.ann_files:
        loaded = load_annotation(ann_file)
        ann.update(loaded)
        target_indices.extend(sorted(loaded))

    high_by_clean = {}
    label_by_sample = {}
    for pred_path in args.teacher_predictions:
        sample_idx, labels, probs = load_predictions(pred_path)
        probs = apply_temperature(probs, args.temperature)
        for idx, label, prob in zip(sample_idx, labels, probs):
            idx = int(idx)
            if idx not in ann:
                if args.strict_prediction_coverage:
                    raise KeyError(f"Prediction sample_idx={idx} not found in annotations.")
                continue
            item = ann[idx]
            if float(item["snr"]) < args.high_snr_min:
                continue
            clean_id = int(item["clean_id"])
            high_by_clean.setdefault(clean_id, []).append((float(item["snr"]), int(label), prob))
            label_by_sample[idx] = int(label)

    out_idx, out_label, out_clean, out_snr, out_high_snr, out_probs = [], [], [], [], [], []
    missing_clean = []
    for idx in sorted(set(target_indices)):
        item = ann[idx]
        clean_id = int(item["clean_id"])
        candidates = high_by_clean.get(clean_id, [])
        if not candidates:
            missing_clean.append(clean_id)
            continue
        probs = np.stack([c[2] for c in candidates], axis=0).mean(axis=0)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum()
        label = label_by_sample.get(idx)
        if label is None:
            label = int(args.class_names.index(item["modulation"])) if args.class_names else -1
        out_idx.append(idx)
        out_label.append(label)
        out_clean.append(clean_id)
        out_snr.append(float(item["snr"]))
        out_high_snr.append(float(np.mean([c[0] for c in candidates])))
        out_probs.append(probs)

    if missing_clean:
        unique_missing = sorted(set(missing_clean))
        raise KeyError(
            f"Missing high-SNR teacher posterior for {len(unique_missing)} clean_id values; "
            f"examples={unique_missing[:8]}")

    metadata = {
        "type": "strict_awgn_dpc_paired_high_snr_average",
        "ann_files": [str(Path(p)) for p in args.ann_files],
        "teacher_predictions": [str(Path(p)) for p in args.teacher_predictions],
        "high_snr_min": float(args.high_snr_min),
        "temperature": float(args.temperature),
        "teacher_config": args.teacher_config,
        "teacher_commit": args.teacher_commit or git_commit(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_idx=np.asarray(out_idx, dtype=np.int64),
        clean_id=np.asarray(out_clean, dtype=np.int64),
        label=np.asarray(out_label, dtype=np.int64),
        reliability=np.asarray(out_snr, dtype=np.float32),
        snr_low=np.asarray(out_snr, dtype=np.float32),
        snr_high=np.asarray(out_high_snr, dtype=np.float32),
        probs=np.asarray(out_probs, dtype=np.float32),
        temperature=np.asarray([args.temperature], dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved strict AWGN-DPC target artifact to {out}")
    print(f"  targets: {len(out_idx)}")
    print(f"  high-SNR clean pools: {len(high_by_clean)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-files", nargs="+", required=True)
    parser.add_argument("--teacher-predictions", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--high-snr-min", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--teacher-config", default="")
    parser.add_argument("--teacher-commit", default="")
    parser.add_argument("--strict-prediction-coverage", action="store_true")
    parser.add_argument("--class-names", nargs="*", default=[
        "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
        "4PAM", "16QAM", "64QAM", "QPSK", "WBFM",
    ])
    return parser.parse_args()


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
