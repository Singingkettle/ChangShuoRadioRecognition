#!/usr/bin/env python3
"""Summarize real-AMC feature geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--input must have the form label=/path/to/dir")
    label, path = value.split("=", 1)
    return label, Path(path)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="label=/path/to/geometry_dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for item in args.input:
        label, path = parse_input(item)
        corr_rows = read_csv(path / "real_amc_geometry_correlations.csv")
        cp_rows = read_csv(path / "real_amc_geometry_critical_points.csv")
        summary_path = path / "summary.json"
        metadata = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        cp = {row["metric"]: float(row["midpoint_snr"]) for row in cp_rows}
        corr = {row["metric"]: row for row in corr_rows}
        for metric in ("fisher_ratio", "margin_ratio", "silhouette_proxy", "overlap_proxy"):
            row = corr[metric]
            summary_rows.append({
                "setting": label,
                "metric": metric,
                "pearson_with_accuracy": float(row["pearson_with_accuracy"]),
                "pearson_with_entropy": float(row["pearson_with_entropy"]),
                "metric_midpoint_snr": cp.get(metric, float("nan")),
                "accuracy_midpoint_snr": cp.get("accuracy", float("nan")),
                "entropy_midpoint_snr": cp.get("mean_entropy", float("nan")),
                "n_total": metadata.get("n_total", ""),
                "n_used": metadata.get("n_used", ""),
                "feature_dim": metadata.get("feature_dim", ""),
                "feature_source": metadata.get("feature_source", ""),
                "path": str(path),
            })

    write_csv(out_dir / "real_amc_geometry_summary.csv", summary_rows)

    settings = list(dict.fromkeys(row["setting"] for row in summary_rows))
    metrics = ["fisher_ratio", "silhouette_proxy", "overlap_proxy"]
    nice = {
        "fisher_ratio": "Fisher",
        "silhouette_proxy": "Silhouette",
        "overlap_proxy": "Overlap",
    }
    values = np.zeros((len(settings), len(metrics)), dtype=float)
    for i, setting in enumerate(settings):
        for j, metric in enumerate(metrics):
            row = next(r for r in summary_rows if r["setting"] == setting and r["metric"] == metric)
            values[i, j] = row["pearson_with_accuracy"]

    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    x = np.arange(len(settings))
    width = 0.24
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for j, metric in enumerate(metrics):
        ax.bar(x + (j - 1) * width, values[:, j], width=width, label=nice[metric], color=colors[j])
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson correlation with accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=18, ha="right")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.set_title("Real AMC feature geometry aligns with reliability transitions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "real_amc_geometry_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "real_amc_geometry_summary.png", bbox_inches="tight", dpi=220)
    print(json.dumps({
        "out_dir": str(out_dir),
        "rows": len(summary_rows),
        "settings": settings,
        "summary_csv": str(out_dir / "real_amc_geometry_summary.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
