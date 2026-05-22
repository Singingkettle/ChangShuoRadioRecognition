#!/usr/bin/env python3
"""Plot synthetic AWGN-DPC mismatch stress-test summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CB = {
    "hard_ce": "#0072B2",
    "strict_awgn_dpc": "#D55E00",
    "critical_strict_dpc": "#009E73",
}
LABELS = {
    "hard_ce": "Hard CE",
    "strict_awgn_dpc": "Strict DPC",
    "critical_strict_dpc": "Critical DPC",
}


def load_delta(root: Path, stem: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / stem)


def panel_delta(ax, df: pd.DataFrame, title: str) -> None:
    metrics = [
        ("delta_overall_accuracy", "Acc. (pp)", 1.0),
        ("delta_overall_nll", "NLL", -1.0),
        ("delta_overall_ece", "ECE", -1.0),
        ("delta_overall_brier", "Brier", -1.0),
        ("delta_high_accuracy", "High-SNR acc. (pp)", 1.0),
        ("delta_transition_nll", "Trans. NLL", -1.0),
    ]
    methods = ["strict_awgn_dpc", "critical_strict_dpc"]
    x = np.arange(len(metrics))
    width = 0.36
    for i, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        means = [float(row[f"{m}_mean"]) for m, _, _ in metrics]
        stds = [float(row[f"{m}_std"]) for m, _, _ in metrics]
        # Plot signed deltas directly; beneficial direction is marked by arrows in labels.
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=2.5,
               color=CB[method], label=LABELS[method], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="0.25", linewidth=0.7)
    ax.set_xticks(x)
    tick_labels = []
    for _, label, sign in metrics:
        direction = "higher" if sign > 0 else "lower"
        tick_labels.append(f"{label}\n({direction} better)")
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def panel_snr(ax, bin_df: pd.DataFrame, title: str) -> None:
    grouped = bin_df.groupby(["method", "snr"])["accuracy"].agg(["mean", "std"]).reset_index()
    for method in ["hard_ce", "strict_awgn_dpc", "critical_strict_dpc"]:
        sub = grouped[grouped["method"] == method].sort_values("snr")
        ax.plot(sub["snr"], sub["mean"], marker="o", markersize=2.5, linewidth=1.2,
                color=CB[method], label=LABELS[method])
        ax.fill_between(sub["snr"].to_numpy(),
                        (sub["mean"] - sub["std"]).to_numpy(),
                        (sub["mean"] + sub["std"]).to_numpy(),
                        color=CB[method], alpha=0.12, linewidth=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="0.9", linewidth=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, default=Path("/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/mismatch_phasefreqmultipath_3seed_eval"))
    parser.add_argument("--impulsive-root", type=Path, default=Path("/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/mismatch_impulsive_3seed_eval"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/paper_figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase_delta = load_delta(args.phase_root, "petcgdnn_mismatch_phasefreqmultipath_delta_mean_std.csv")
    impulsive_delta = load_delta(args.impulsive_root, "petcgdnn_impulsive_delta_mean_std.csv")
    phase_bins = pd.read_csv(args.phase_root / "metrics" / "petcgdnn_mismatch_phasefreqmultipath_snr_bin_metrics.csv")
    impulsive_bins = pd.read_csv(args.impulsive_root / "metrics" / "petcgdnn_impulsive_snr_bin_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), constrained_layout=True)
    panel_delta(axes[0, 0], phase_delta, "Phase/frequency/multipath mismatch: paired deltas")
    panel_delta(axes[0, 1], impulsive_delta, "Impulsive-noise mismatch: paired deltas")
    panel_snr(axes[1, 0], phase_bins, "Phase/frequency/multipath mismatch")
    panel_snr(axes[1, 1], impulsive_bins, "Impulsive-noise mismatch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.14)

    pdf = args.out_dir / "petcgdnn_mismatch_stress_tests.pdf"
    png = args.out_dir / "petcgdnn_mismatch_stress_tests.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
