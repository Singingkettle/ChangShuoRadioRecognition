#!/usr/bin/env python3
"""Analyze whether scalar SNR is a sufficient order parameter for audio runs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    score = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            score += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(score)


def _logistic(x: np.ndarray, amin: float, amax: float, k: float, gamma_c: float) -> np.ndarray:
    return amin + (amax - amin) / (1.0 + np.exp(-k * (x - gamma_c)))


def analyze(prediction_pkl: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with prediction_pkl.open("rb") as f:
        payload = pickle.load(f)

    probs = np.asarray(payload["pps"], dtype=float)
    labels = np.asarray(payload["gts"], dtype=int)
    snrs = np.asarray(payload["snrs"])
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    confidence = probs.max(axis=1)
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
    brier = ((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(axis=1)

    rows = []
    class_rows = []
    for snr in sorted(np.unique(snrs), key=float):
        mask = snrs == snr
        rows.append(
            dict(
                snr=float(snr),
                n=int(mask.sum()),
                accuracy=100.0 * correct[mask].mean(),
                nll=nll[mask].mean(),
                ece=_ece(probs[mask], labels[mask]),
                brier=brier[mask].mean(),
                confidence_mean=confidence[mask].mean(),
                confidence_std=confidence[mask].std(),
                entropy_mean=entropy[mask].mean(),
                entropy_std=entropy[mask].std(),
                entropy_iqr=np.percentile(entropy[mask], 75) - np.percentile(entropy[mask], 25),
            )
        )
        accs = []
        for cls in range(probs.shape[1]):
            cls_mask = mask & (labels == cls)
            if cls_mask.any():
                accs.append(100.0 * correct[cls_mask].mean())
        class_rows.append(
            dict(
                snr=float(snr),
                class_accuracy_min=np.min(accs),
                class_accuracy_max=np.max(accs),
                class_accuracy_range=np.max(accs) - np.min(accs),
                class_accuracy_std=np.std(accs),
            )
        )

    per_snr = pd.DataFrame(rows)
    class_spread = pd.DataFrame(class_rows)
    per_snr.to_csv(out_dir / "per_snr_dispersion.csv", index=False)
    class_spread.to_csv(out_dir / "per_snr_class_spread.csv", index=False)

    explained_rows = []
    for name, values in [
        ("correct", correct),
        ("entropy", entropy),
        ("confidence", confidence),
        ("nll", nll),
        ("brier", brier),
    ]:
        grand = values.mean()
        total = ((values - grand) ** 2).sum()
        between = 0.0
        within = 0.0
        for snr in np.unique(snrs):
            vals = values[snrs == snr]
            between += len(vals) * (vals.mean() - grand) ** 2
            within += ((vals - vals.mean()) ** 2).sum()
        explained_rows.append(
            dict(
                variable=name,
                eta2_snr=between / total if total > 0 else np.nan,
                within_snr_share=within / total if total > 0 else np.nan,
            )
        )
    explained = pd.DataFrame(explained_rows)
    explained.to_csv(out_dir / "snr_explained_variance.csv", index=False)

    x = per_snr["snr"].to_numpy(dtype=float)
    y = per_snr["accuracy"].to_numpy(dtype=float)
    p0 = [y.min(), y.max(), 0.2, x[np.argmin(np.abs(y - 0.5 * (y.min() + y.max())))]]
    popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=10000)
    fit_y = _logistic(x, *popt)
    r2 = 1.0 - ((y - fit_y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fit = pd.DataFrame([dict(amin=popt[0], amax=popt[1], k=popt[2], gamma_c=popt[3], r2=r2)])
    fit.to_csv(out_dir / "snr_logistic_fit.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
    axes[0].plot(x, y, marker="o", label="Accuracy")
    axes[0].plot(x, fit_y, linestyle="--", label=f"Logistic fit, R2={r2:.3f}")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Waterfall curve")
    axes[0].legend(fontsize=6, frameon=False)

    axes[1].plot(x, per_snr["entropy_std"], marker="o", color="#D55E00")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Entropy std.")
    axes[1].set_title("Within-SNR uncertainty")

    axes[2].plot(x, class_spread["class_accuracy_range"], marker="o", color="#009E73")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Class acc. range (pp)")
    axes[2].set_title("Within-SNR class spread")
    for ax in axes:
        ax.grid(color="0.9", linewidth=0.6)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(9)
    fig.tight_layout()
    fig.savefig(out_dir / "snr_order_parameter_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "snr_order_parameter_diagnostic.png", dpi=300, bbox_inches="tight")

    print(out_dir)
    print(explained.to_string(index=False))
    print(fit.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-pkl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.prediction_pkl, args.out_dir)


if __name__ == "__main__":
    main()
