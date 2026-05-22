#!/usr/bin/env python3
"""Summarize synthetic AWGN RCPS/DPC prediction artifacts.

The script reads ``predictions/test.pkl`` files produced by the CSRR
evaluation pipeline and writes seed-level, SNR-bin, and paired-delta CSVs.
It intentionally recomputes metrics from probabilities instead of relying on
training logs.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - scipy should be present in the run env
    curve_fit = None


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn"),
        help="Root directory containing stage1_petcgdnn_seed*_30ep folders.",
    )
    parser.add_argument("--model-tag", default="petcgdnn")
    parser.add_argument("--dataset-tag", default="synthetic_awgn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hard_ce", "strict_awgn_dpc", "critical_strict_dpc"],
    )
    parser.add_argument(
        "--method-dirs",
        nargs="+",
        default=["hard_ce=hard_ce", "strict_awgn_dpc=strict_awgn_dpc", "critical_strict_dpc=critical_strict_dpc"],
        help="Mapping method=directory under each seed folder.",
    )
    parser.add_argument("--stage-template", default="stage1_{model_tag}_seed{seed}_30ep")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to root/stage1_<model-tag>_<n>seed_summary.",
    )
    parser.add_argument("--transition-low", type=float, default=-6.0)
    parser.add_argument("--transition-high", type=float, default=6.0)
    parser.add_argument("--high-snr", type=float, default=10.0)
    parser.add_argument("--low-snr", type=float, default=-10.0)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def metric_dict(probs: np.ndarray, labels: np.ndarray, ece_bins: int) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0

    out = {
        "accuracy": float((pred == labels).mean() * 100.0),
        "nll": float(-np.log(np.clip(probs[np.arange(n), labels], EPS, 1.0)).mean()),
        "brier": float(np.sum((probs - one_hot) ** 2, axis=1).mean()),
        "mean_confidence": float(conf.mean()),
        "mean_entropy": float(-(probs * np.log(np.clip(probs, EPS, 1.0))).sum(axis=1).mean()),
        "n": int(n),
    }

    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    correct = pred == labels
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    out["ece"] = ece
    return out


def logistic(x: np.ndarray, amin: float, amax: float, k: float, gamma_c: float) -> np.ndarray:
    return amin + (amax - amin) / (1.0 + np.exp(-k * (x - gamma_c)))


def fit_transition(snrs: list[float], accs: list[float]) -> dict[str, float | str]:
    if curve_fit is None:
        return {
            "fit_amin": np.nan,
            "fit_amax": np.nan,
            "fit_k": np.nan,
            "gamma_c": np.nan,
            "transition_20": np.nan,
            "transition_80": np.nan,
            "fit_r2": np.nan,
            "fit_error": "scipy unavailable",
        }

    xs = np.asarray(snrs, dtype=np.float64)
    ys = np.asarray(accs, dtype=np.float64)
    amin0 = float(np.nanmin(ys))
    amax0 = float(np.nanmax(ys))
    mid = 0.5 * (amin0 + amax0)
    gc0 = float(xs[np.argmin(np.abs(ys - mid))])
    try:
        popt, _ = curve_fit(
            logistic,
            xs,
            ys,
            p0=[amin0, amax0, 0.5, gc0],
            bounds=([0.0, 0.0, 1e-3, -40.0], [100.0, 100.0, 10.0, 40.0]),
            maxfev=20000,
        )
        pred = logistic(xs, *popt)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        amin, amax, k, gamma_c = map(float, popt)
        half_width = math.log(4.0) / k if k > 0 else np.nan
        return {
            "fit_amin": amin,
            "fit_amax": amax,
            "fit_k": k,
            "gamma_c": gamma_c,
            "transition_20": gamma_c - half_width,
            "transition_80": gamma_c + half_width,
            "fit_r2": r2,
            "fit_error": "",
        }
    except Exception as exc:  # pragma: no cover - retained for robust batch runs
        return {
            "fit_amin": np.nan,
            "fit_amax": np.nan,
            "fit_k": np.nan,
            "gamma_c": np.nan,
            "transition_20": np.nan,
            "transition_80": np.nan,
            "fit_r2": np.nan,
            "fit_error": str(exc),
        }


def main() -> None:
    args = parse_args()
    method_dirs = dict(item.split("=", 1) for item in args.method_dirs)
    out_dir = args.out or args.root / f"stage1_{args.model_tag}_{len(args.seeds)}seed_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, float | str | int]] = []
    bin_rows: list[dict[str, float | str | int]] = []

    for seed in args.seeds:
        stage_dir = args.root / args.stage_template.format(model_tag=args.model_tag, seed=seed)
        for method in args.methods:
            pred_path = stage_dir / method_dirs[method] / "predictions" / "test.pkl"
            if not pred_path.exists():
                print(f"[WARN] missing prediction file: {pred_path}")
                continue
            with pred_path.open("rb") as f:
                pred_obj = pickle.load(f)
            probs = pred_obj["pps"]
            labels = pred_obj["gts"]
            snrs = np.asarray(pred_obj["snrs"], dtype=np.float64)

            row: dict[str, float | str | int] = {"seed": seed, "method": method}
            masks = {
                "overall": np.ones_like(snrs, dtype=bool),
                "high_snr": snrs >= args.high_snr,
                "low_snr": snrs <= args.low_snr,
                "fixed_transition": (snrs >= args.transition_low) & (snrs <= args.transition_high),
            }
            for prefix, mask in masks.items():
                values = metric_dict(probs[mask], labels[mask], args.ece_bins)
                for key, value in values.items():
                    row[f"{prefix}_{key}"] = value

            unique_snrs = sorted(np.unique(snrs))
            accs = []
            for snr_value in unique_snrs:
                mask = snrs == snr_value
                values = metric_dict(probs[mask], labels[mask], args.ece_bins)
                values.update({"seed": seed, "method": method, "snr": float(snr_value)})
                bin_rows.append(values)
                accs.append(values["accuracy"])
            row.update(fit_transition(unique_snrs, accs))
            seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    bin_df = pd.DataFrame(bin_rows)
    seed_df.to_csv(out_dir / "seed_method_summary.csv", index=False)
    bin_df.to_csv(out_dir / "snr_bin_metrics.csv", index=False)

    metric_cols = [c for c in seed_df.columns if c not in {"seed", "method", "fit_error"}]
    aggregate_rows = []
    for method, group in seed_df.groupby("method"):
        out = {"method": method, "n_seeds": len(group), "seeds": ",".join(map(str, sorted(group.seed.unique())))}
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(group[col]):
                out[f"{col}_mean"] = group[col].mean()
                out[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        aggregate_rows.append(out)
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(out_dir / "method_mean_std.csv", index=False)

    delta_rows = []
    for seed in args.seeds:
        hard = seed_df[(seed_df.seed == seed) & (seed_df.method == "hard_ce")]
        if hard.empty:
            continue
        hard_row = hard.iloc[0]
        for method in args.methods:
            if method == "hard_ce":
                continue
            current = seed_df[(seed_df.seed == seed) & (seed_df.method == method)]
            if current.empty:
                continue
            current_row = current.iloc[0]
            out = {"seed": seed, "method": method}
            for col in metric_cols:
                if pd.api.types.is_numeric_dtype(seed_df[col]):
                    out[f"delta_{col}"] = current_row[col] - hard_row[col]
            delta_rows.append(out)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "paired_deltas_vs_hard_by_seed.csv", index=False)

    if not delta_df.empty:
        delta_aggregate_rows = []
        for method, group in delta_df.groupby("method"):
            out = {"method": method, "n_seeds": len(group)}
            for col in [c for c in delta_df.columns if c.startswith("delta_")]:
                out[f"{col}_mean"] = group[col].mean()
                out[f"{col}_std"] = group[col].std(ddof=1) if len(group) > 1 else 0.0
            delta_aggregate_rows.append(out)
        pd.DataFrame(delta_aggregate_rows).to_csv(out_dir / "paired_deltas_vs_hard_mean_std.csv", index=False)

    print(f"Wrote summary to {out_dir}")
    display_cols = [
        "method",
        "n_seeds",
        "overall_accuracy_mean",
        "overall_nll_mean",
        "overall_ece_mean",
        "overall_brier_mean",
        "high_snr_accuracy_mean",
        "fixed_transition_accuracy_mean",
        "fixed_transition_nll_mean",
        "gamma_c_mean",
        "fit_r2_mean",
    ]
    print(aggregate_df[[c for c in display_cols if c in aggregate_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
