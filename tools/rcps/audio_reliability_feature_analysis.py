#!/usr/bin/env python3
"""Diagnose whether audio needs a multidimensional reliability coordinate.

This script is intentionally post-hoc: it uses an existing hard-label teacher
prediction artifact and reconstructs deterministic Speech Commands noisy views
from the same annotation order.  It then compares scalar SNR against a small
set of signal features as order-parameter candidates for sample-level posterior
quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_crossmodal_audio import SpeechCommandsReliability  # noqa: E402

SNR_VALUES = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0)


def load_prediction(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return {k: np.asarray(v) for k, v in payload.items()}


def spectral_features(wav: torch.Tensor, clean: torch.Tensor, sample_rate: int = 16000) -> Dict[str, float]:
    x = wav.flatten().float()
    s = clean.flatten().float()
    noise = x - s

    signal_power = float(s.pow(2).mean().clamp_min(1e-10))
    noise_power = float(noise.pow(2).mean().clamp_min(1e-10))
    actual_snr = 10.0 * math.log10(signal_power / noise_power)

    spec = torch.fft.rfft(x)
    power = spec.abs().pow(2).float() + 1e-10
    freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / sample_rate)
    flatness = float(torch.exp(torch.log(power).mean()) / power.mean())

    total_power = float(power.sum().clamp_min(1e-10))
    speech_band = (freqs >= 300.0) & (freqs <= 3400.0)
    low_band = freqs < 300.0
    high_band = freqs > 3400.0
    speech_ratio = float(power[speech_band].sum() / total_power)
    low_ratio = float(power[low_band].sum() / total_power)
    high_ratio = float(power[high_band].sum() / total_power)

    frame = 400
    hop = 160
    if s.numel() >= frame:
        frames = s.unfold(0, frame, hop)
        frame_rms = frames.pow(2).mean(dim=1).sqrt()
        threshold = max(0.01, float(frame_rms.max()) * 0.1)
        vad_ratio = float((frame_rms > threshold).float().mean())
    else:
        vad_ratio = float(s.abs().mean() > 0.01)

    zcr = float((x[1:] * x[:-1] < 0).float().mean()) if x.numel() > 1 else 0.0
    return {
        "actual_snr": actual_snr,
        "clean_log_energy": math.log(signal_power + 1e-10),
        "mixed_log_energy": math.log(float(x.pow(2).mean().clamp_min(1e-10))),
        "noise_log_energy": math.log(noise_power + 1e-10),
        "spectral_flatness": flatness,
        "speech_band_ratio": speech_ratio,
        "low_band_ratio": low_ratio,
        "high_band_ratio": high_ratio,
        "vad_ratio": vad_ratio,
        "zero_crossing_rate": zcr,
    }


def build_feature_table(split: str, prediction_path: Path, processed_root: Path,
                        seed: int, max_per_label_snr: int) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    pred = load_prediction(prediction_path)
    dataset = SpeechCommandsReliability(
        processed_root / f"{split}.json",
        split,
        max_per_label_snr=max_per_label_snr,
        seed=seed,
        train=False,
        return_clean_pair=True,
    )
    if len(dataset) != len(pred["gts"]):
        raise ValueError(f"{split}: dataset length {len(dataset)} != prediction length {len(pred['gts'])}")

    probs = np.clip(pred["pps"].astype(np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = pred["gts"].astype(np.int64)
    guess = probs.argmax(axis=1)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    confidence = probs.max(axis=1)
    onehot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    nll = -np.log(probs[np.arange(labels.size), labels])
    brier = ((probs - onehot) ** 2).sum(axis=1)
    correct = (guess == labels).astype(np.float64)

    rows: List[Dict[str, float]] = []
    for idx in range(len(dataset)):
        mixed, clean, label, reliability, snr_label = dataset[idx]
        snr_numeric = 30.0 if str(snr_label) == "clean" else float(snr_label)
        feats = spectral_features(mixed, clean)
        feats.update({
            "idx": idx,
            "label": int(label),
            "snr": snr_numeric,
            "reliability": float(reliability),
            "entropy": float(entropy[idx]),
            "confidence": float(confidence[idx]),
            "nll": float(nll[idx]),
            "brier": float(brier[idx]),
            "correct": float(correct[idx]),
        })
        rows.append(feats)

    outcomes = {
        "entropy": entropy,
        "confidence": confidence,
        "nll": nll,
        "brier": brier,
        "correct": correct,
    }
    return rows, outcomes


def matrix(rows: List[Dict[str, float]], names: Iterable[str]) -> np.ndarray:
    cols = []
    for name in names:
        if name == "snr_poly3":
            snr = np.asarray([float(row["snr"]) for row in rows], dtype=np.float64)
            cols.extend([snr, snr ** 2, snr ** 3])
        elif name == "snr_onehot":
            snr = np.asarray([float(row["snr"]) for row in rows], dtype=np.float64)
            # Drop the first bin to avoid exact collinearity with the intercept.
            for value in SNR_VALUES[1:]:
                cols.append((snr == value).astype(np.float64))
        else:
            cols.append(np.asarray([float(row[name]) for row in rows], dtype=np.float64))
    return np.stack(cols, axis=1)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def ridge_fit(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    x = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    reg = np.eye(x.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ train_y)


def predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x1 = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x1 @ coef


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(((y - y.mean()) ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - ((y - pred) ** 2).sum() / denom)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_rows(path: Path, rows: List[Dict[str, float]]) -> None:
    keep = [
        "idx", "label", "snr", "reliability", "actual_snr", "clean_log_energy",
        "mixed_log_energy", "noise_log_energy", "spectral_flatness",
        "speech_band_ratio", "low_band_ratio", "high_band_ratio", "vad_ratio",
        "zero_crossing_rate", "entropy", "confidence", "nll", "brier", "correct",
    ]
    write_csv(path, [{k: row[k] for k in keep} for row in rows])


def quantile_bins(phi: np.ndarray, rows: List[Dict[str, float]], n_bins: int = 10) -> List[Dict]:
    edges = np.quantile(phi, np.linspace(0.0, 1.0, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i + 1 == n_bins:
            mask = (phi >= lo) & (phi <= hi)
        else:
            mask = (phi >= lo) & (phi < hi)
        if not mask.any():
            continue
        sub = [rows[j] for j in np.where(mask)[0]]
        out.append({
            "bin": i,
            "count": len(sub),
            "phi_mean": float(phi[mask].mean()),
            "snr_mean": float(np.mean([r["snr"] for r in sub])),
            "accuracy": float(np.mean([r["correct"] for r in sub]) * 100.0),
            "entropy": float(np.mean([r["entropy"] for r in sub])),
            "nll": float(np.mean([r["nll"] for r in sub])),
            "brier": float(np.mean([r["brier"] for r in sub])),
        })
    return out


def maybe_plot(out_dir: Path, r2_rows: List[Dict], phi_rows: List[Dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot: {exc}")
        return

    outcomes = ["entropy", "nll", "brier", "correct"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    x = np.arange(len(outcomes))
    snr = [next(r["test_r2"] for r in r2_rows if r["outcome"] == o and r["feature_set"] == "snr_onehot") for o in outcomes]
    multi = [next(r["test_r2"] for r in r2_rows if r["outcome"] == o and r["feature_set"] == "snr_onehot_audio") for o in outcomes]
    axes[0].bar(x - 0.18, snr, width=0.36, label="SNR bins")
    axes[0].bar(x + 0.18, multi, width=0.36, label="SNR+audio")
    axes[0].set_xticks(x, ["Entropy", "NLL", "Brier", "Correct"], rotation=20)
    axes[0].set_ylabel("Test $R^2$")
    axes[0].set_title("Order-parameter explanatory power")
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.asarray([r["bin"] for r in phi_rows])
    entropy = np.asarray([r["entropy"] for r in phi_rows])
    acc = np.asarray([r["accuracy"] for r in phi_rows])
    ax = axes[1]
    ax.plot(bins, entropy, marker="o", label="Entropy")
    ax.set_xlabel("Predicted reliability quantile")
    ax.set_ylabel("Entropy")
    ax2 = ax.twinx()
    ax2.plot(bins, acc, marker="s", color="#D55E00", label="Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax.set_title("Multifeature reliability ordering")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"audio_multifeature_order_parameter.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("/home/citybuster/Data/RCPS/processed/ReliabilityClassification/Audio/SpeechCommands-v0.02"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-max-per-label-snr", type=int, default=200)
    parser.add_argument("--test-max-per-label-snr", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_rows, val_out = build_feature_table(
        "validation", args.prediction_root / "validation.pkl", args.processed_root,
        seed=args.seed + 17, max_per_label_snr=args.val_max_per_label_snr)
    test_rows, test_out = build_feature_table(
        "test", args.prediction_root / "test.pkl", args.processed_root,
        seed=args.seed + 31, max_per_label_snr=args.test_max_per_label_snr)
    write_feature_rows(args.out_dir / "validation_features.csv", val_rows)
    write_feature_rows(args.out_dir / "test_features.csv", test_rows)

    feature_sets = {
        "snr_linear": ["snr"],
        "snr_poly3": ["snr_poly3"],
        "snr_onehot": ["snr_onehot"],
        "audio": [
            "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
        ],
        "snr_linear_audio": [
            "snr", "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
        ],
        "snr_poly3_audio": [
            "snr_poly3", "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
        ],
        "snr_onehot_audio": [
            "snr_onehot", "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
        ],
        "teacher_conf": ["confidence"],
        "snr_teacher_conf": ["snr_onehot", "confidence"],
        "snr_audio_teacher_conf": [
            "snr_onehot", "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
            "confidence",
        ],
        # Diagnostic only: entropy is a direct teacher-output statistic, so this
        # feature set should not be interpreted as an independent physical
        # reliability coordinate.
        "snr_audio_teacher_entropy_conf": [
            "snr_onehot", "actual_snr", "clean_log_energy", "mixed_log_energy",
            "noise_log_energy", "spectral_flatness", "speech_band_ratio",
            "low_band_ratio", "high_band_ratio", "vad_ratio", "zero_crossing_rate",
            "entropy", "confidence",
        ],
    }
    r2_rows: List[Dict] = []
    coefficients = {}
    for outcome in ["entropy", "confidence", "nll", "brier", "correct"]:
        for name, feats in feature_sets.items():
            train_x = matrix(val_rows, feats)
            test_x = matrix(test_rows, feats)
            train_x, test_x = standardize(train_x, test_x)
            train_y = val_out[outcome].astype(np.float64)
            test_y = test_out[outcome].astype(np.float64)
            coef = ridge_fit(train_x, train_y)
            train_pred = predict(train_x, coef)
            test_pred = predict(test_x, coef)
            r2_rows.append({
                "outcome": outcome,
                "feature_set": name,
                "n_features": len(feats),
                "train_r2": r2_score(train_y, train_pred),
                "test_r2": r2_score(test_y, test_pred),
            })
            coefficients[f"{outcome}:{name}"] = {"features": feats, "coef": coef.tolist()}

    write_csv(args.out_dir / "feature_set_r2.csv", r2_rows)
    (args.out_dir / "ridge_coefficients.json").write_text(json.dumps(coefficients, indent=2), encoding="utf-8")

    feats = feature_sets["snr_onehot_audio"]
    train_x = matrix(val_rows, feats)
    test_x = matrix(test_rows, feats)
    train_x, test_x = standardize(train_x, test_x)
    coef = ridge_fit(train_x, val_out["entropy"].astype(np.float64))
    pred_entropy = predict(test_x, coef)
    phi = -pred_entropy
    phi_rows = quantile_bins(phi, test_rows, n_bins=10)
    write_csv(args.out_dir / "multifeature_phi_bins.csv", phi_rows)
    maybe_plot(args.out_dir, r2_rows, phi_rows)

    summary = {
        "prediction_root": str(args.prediction_root),
        "validation_size": len(val_rows),
        "test_size": len(test_rows),
        "snr_onehot_entropy_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "entropy" and r["feature_set"] == "snr_onehot"),
        "snr_audio_entropy_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "entropy" and r["feature_set"] == "snr_onehot_audio"),
        "teacher_conf_entropy_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "entropy" and r["feature_set"] == "teacher_conf"),
        "snr_audio_teacher_conf_entropy_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "entropy" and r["feature_set"] == "snr_audio_teacher_conf"),
        "snr_onehot_correct_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "correct" and r["feature_set"] == "snr_onehot"),
        "snr_audio_correct_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "correct" and r["feature_set"] == "snr_onehot_audio"),
        "teacher_conf_correct_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "correct" and r["feature_set"] == "teacher_conf"),
        "snr_audio_teacher_conf_correct_test_r2": next(r["test_r2"] for r in r2_rows if r["outcome"] == "correct" and r["feature_set"] == "snr_audio_teacher_conf"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
