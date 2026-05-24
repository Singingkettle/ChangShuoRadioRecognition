#!/usr/bin/env python3
"""Export deterministic Speech Commands teacher posteriors for phi-RCPS.

This utility reuses the standalone audio runner components but does not train.
It loads an admitted hard-label checkpoint, evaluates one split with deterministic
noisy views, and writes both the existing paper.pkl-compatible format and an
NPZ artifact with sample indices and teacher confidence/entropy. The artifact is
intended for next-stage audio phi-RCPS experiments, where scalar SNR is not a
sufficient sample-level order parameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_crossmodal_audio import (  # noqa: E402
    LogMelFeature,
    SpeechCommandsReliability,
    build_loader,
    build_model,
    evaluate_loader,
    write_prediction_pkl,
)


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1).astype(np.float32)


def numeric_snrs(snrs: np.ndarray) -> np.ndarray:
    values = []
    for value in snrs:
        values.append(30.0 if str(value) == "clean" else float(value))
    return np.asarray(values, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["ds-cnn", "logmel-resnet"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--max-per-label-snr", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            "/home/citybuster/Data/RCPS/processed/ReliabilityClassification/Audio/"
            "SpeechCommands-v0.02"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_seed_offsets = {"train": 0, "validation": 17, "test": 31}
    split_seed = args.seed + split_seed_offsets[args.split]
    dataset = SpeechCommandsReliability(
        args.processed_root / f"{args.split}.json",
        args.split,
        max_per_label_snr=args.max_per_label_snr,
        seed=split_seed,
        train=False,
    )
    loader_args = argparse.Namespace(batch_size=args.batch_size, workers=args.workers)
    loader = build_loader(dataset, loader_args, shuffle=False, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = LogMelFeature().to(device)
    model = build_model(args.model, len(dataset.classes)).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)

    probs, labels, reliabilities, snrs = evaluate_loader(model, feature_extractor, loader, device)
    sample_idx = np.arange(labels.shape[0], dtype=np.int64)
    conf = probs.max(axis=1).astype(np.float32)
    ent = entropy(probs)
    snr_values = numeric_snrs(snrs)
    snr_labels = np.asarray([str(v) for v in snrs])

    write_prediction_pkl(args.out_dir / f"{args.split}.pkl", probs, labels, reliabilities, snrs)
    np.savez_compressed(
        args.out_dir / f"{args.split}_teacher_posteriors.npz",
        sample_idx=sample_idx,
        probs=probs.astype(np.float32),
        labels=labels.astype(np.int64),
        reliability=reliabilities.astype(np.float32),
        snrs=snr_values,
        snr_labels=snr_labels,
        confidence=conf,
        entropy=ent,
    )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "split_seed": split_seed,
        "max_per_label_snr": args.max_per_label_snr,
        "num_samples": int(labels.shape[0]),
        "num_classes": int(probs.shape[1]),
        "classes": dataset.classes,
        "prob_sum_min": float(probs.sum(axis=1).min()),
        "prob_sum_max": float(probs.sum(axis=1).max()),
        "confidence_mean": float(conf.mean()),
        "entropy_mean": float(ent.mean()),
        "deterministic_views": True,
    }
    (args.out_dir / f"{args.split}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
