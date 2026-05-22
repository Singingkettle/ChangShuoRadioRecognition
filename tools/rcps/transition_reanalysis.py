#!/usr/bin/env python3
"""Fit reliability-induced transition curves from existing RCPS SNR-bin CSVs.

This script is intentionally read-only with respect to experiment artifacts.  It
loads completed aggregate CSVs, fits a four-parameter logistic curve to
accuracy-vs-SNR, and reports critical SNR, slope, SNR-AUC, fit residuals, and
transition-band posterior metrics.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path("/home/citybuster/Data/RCPS/work_dirs")
OUT = ROOT / "transition_reanalysis"


SOURCES = [
    {
        "name": "RML2016.10A / PETCGDNN",
        "dataset": "RadioML2016.10A",
        "model": "PETCGDNN",
        "path": ROOT
        / "paper_evidence/paper_tables/amc_petcgdnn_rml201610a_snr_bins.csv",
        "format": "method_rows",
        "hard": "Hard CE",
        "method": "DPC-RCPS",
    },
    {
        "name": "RML2016.10B / MCformer",
        "dataset": "RadioML2016.10B",
        "model": "MCformer",
        "path": ROOT
        / "paper_evidence/paper_tables/amc_mcformer_rml201610b_snr_bins.csv",
        "format": "method_rows",
        "hard": "Hard CE",
        "method": "DPC-RCPS",
    },
    {
        "name": "RML2018.01A / PETCGDNN",
        "dataset": "RadioML2018.01A",
        "model": "PETCGDNN",
        "path": ROOT
        / "rcps_hybrid_2018A/summary/deepsig201801A_petcgdnn_rcps-hybrid-eps01_vs_hard_3seed_by_snr_aggregate.csv",
        "format": "paired_aggregate",
        "hard": "Hard CE",
        "method": "RCPS-Hybrid",
    },
]


def read_method_rows(path: Path, dataset: str, model: str):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": r["method_group"],
                    "snr": float(r["reliability_bin_num"]),
                    "accuracy": float(r["accuracy_mean"]) / 100.0,
                    "accuracy_std": float(r.get("accuracy_std", 0.0)) / 100.0,
                    "nll": float(r["nll_mean"]),
                    "ece": float(r["ece_mean"]),
                    "brier": float(r["brier_mean"]),
                }
            )
    return rows


def read_paired_aggregate(path: Path, dataset: str, model: str, method: str):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                snr = float(r["reliability_bin"])
            except ValueError:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": "Hard CE",
                    "snr": snr,
                    "accuracy": float(r["hard_accuracy_mean"]) / 100.0,
                    "accuracy_std": float(r.get("hard_accuracy_std", 0.0)) / 100.0,
                    "nll": float(r["hard_nll_mean"]),
                    "ece": float(r["hard_ece_mean"]),
                    "brier": float(r["hard_brier_mean"]),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "snr": snr,
                    "accuracy": float(r["rcps_accuracy_mean"]) / 100.0,
                    "accuracy_std": float(r.get("rcps_accuracy_std", 0.0)) / 100.0,
                    "nll": float(r["rcps_nll_mean"]),
                    "ece": float(r["rcps_ece_mean"]),
                    "brier": float(r["rcps_brier_mean"]),
                }
            )
    return rows


def logistic(x, amin, amax, k, gc):
    return amin + (amax - amin) / (1.0 + np.exp(-k * (x - gc)))


def fit_logistic(x, y):
    """Small dependency-light logistic fit.

    Prefer scipy when present.  Fall back to a deterministic grid search, which
    is adequate for the low-dimensional SNR-bin summaries used here.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    amin0 = max(0.0, float(np.percentile(y, 5)))
    amax0 = min(1.0, float(np.percentile(y, 95)))
    gc0 = float(x[np.argmin(np.abs(y - (amin0 + amax0) / 2.0))])
    k0 = 0.35

    try:
        from scipy.optimize import curve_fit  # type: ignore

        bounds = ([0.0, 0.0, 0.001, float(x.min() - 10)], [1.0, 1.0, 5.0, float(x.max() + 10)])
        popt, _ = curve_fit(
            logistic,
            x,
            y,
            p0=[amin0, amax0, k0, gc0],
            bounds=bounds,
            maxfev=50000,
        )
        pred = logistic(x, *popt)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return [float(v) for v in popt], rmse
    except Exception:
        best = None
        amin_grid = np.linspace(max(0, y.min() - 0.08), min(0.35, y.min() + 0.08), 17)
        amax_grid = np.linspace(max(y.max() - 0.08, 0.2), min(1.0, y.max() + 0.08), 21)
        k_grid = np.linspace(0.05, 1.5, 40)
        gc_grid = np.linspace(x.min() - 4, x.max() + 4, 57)
        for amin in amin_grid:
            for amax in amax_grid:
                if amax <= amin:
                    continue
                for k in k_grid:
                    z = 1.0 / (1.0 + np.exp(-k * (x[:, None] - gc_grid[None, :])))
                    pred = amin + (amax - amin) * z
                    mse = np.mean((pred - y[:, None]) ** 2, axis=0)
                    j = int(np.argmin(mse))
                    score = float(mse[j])
                    if best is None or score < best[0]:
                        best = (score, amin, amax, k, float(gc_grid[j]))
        assert best is not None
        _, amin, amax, k, gc = best
        pred = logistic(x, amin, amax, k, gc)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return [float(amin), float(amax), float(k), float(gc)], rmse


def trapz_auc(x, y):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    return float(np.trapz(y, x) / (x.max() - x.min()))


def band_from_hard_fit(x, hard_params):
    amin, amax, k, gc = hard_params
    lo = amin + 0.2 * (amax - amin)
    hi = amin + 0.8 * (amax - amin)
    pred = logistic(np.asarray(x, dtype=float), amin, amax, k, gc)
    return lo, hi, pred


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for src in SOURCES:
        if not src["path"].exists():
            raise FileNotFoundError(src["path"])
        if src["format"] == "method_rows":
            all_rows.extend(read_method_rows(src["path"], src["dataset"], src["model"]))
        else:
            all_rows.extend(read_paired_aggregate(src["path"], src["dataset"], src["model"], src["method"]))

    summary = []
    deltas = []
    band_rows = []
    fit_series = []

    for src in SOURCES:
        subset = [r for r in all_rows if r["dataset"] == src["dataset"] and r["model"] == src["model"]]
        methods = [src["hard"], src["method"]]
        hard_rows = sorted([r for r in subset if r["method"] == src["hard"]], key=lambda r: r["snr"])
        hard_x = np.array([r["snr"] for r in hard_rows], dtype=float)
        hard_y = np.array([r["accuracy"] for r in hard_rows], dtype=float)
        hard_params, _ = fit_logistic(hard_x, hard_y)
        lo, hi, hard_pred = band_from_hard_fit(hard_x, hard_params)
        hard_band_snrs = set(float(hard_x[i]) for i, v in enumerate(hard_pred) if lo <= v <= hi)
        if not hard_band_snrs:
            gc = hard_params[3]
            hard_band_snrs = set(float(s) for s in hard_x if abs(float(s) - gc) <= 4.0)

        fits = {}
        for method in methods:
            rows = sorted([r for r in subset if r["method"] == method], key=lambda r: r["snr"])
            x = np.array([r["snr"] for r in rows], dtype=float)
            y = np.array([r["accuracy"] for r in rows], dtype=float)
            params, rmse = fit_logistic(x, y)
            amin, amax, k, gc = params
            auc = trapz_auc(x, y)
            fits[method] = dict(amin=amin, amax=amax, k=k, gamma_c=gc, auc=auc, rmse=rmse)
            summary.append(
                {
                    "dataset": src["dataset"],
                    "model": src["model"],
                    "method": method,
                    "A_min": amin,
                    "A_max": amax,
                    "slope_k": k,
                    "gamma_c": gc,
                    "snr_auc": auc,
                    "fit_rmse": rmse,
                    "hard_transition_band_snrs": " ".join(str(int(s)) if float(s).is_integer() else str(s) for s in sorted(hard_band_snrs)),
                }
            )
            for r in rows:
                fit_series.append(
                    {
                        "dataset": src["dataset"],
                        "model": src["model"],
                        "method": method,
                        "snr": r["snr"],
                        "accuracy": r["accuracy"],
                        "fit_accuracy": float(logistic(np.array([r["snr"]]), *params)[0]),
                    }
                )
            br = [r for r in rows if float(r["snr"]) in hard_band_snrs]
            if br:
                band_rows.append(
                    {
                        "dataset": src["dataset"],
                        "model": src["model"],
                        "method": method,
                        "band_snrs": " ".join(str(int(s)) if float(s).is_integer() else str(s) for s in sorted(hard_band_snrs)),
                        "band_accuracy": float(np.mean([r["accuracy"] for r in br])),
                        "band_nll": float(np.mean([r["nll"] for r in br])),
                        "band_ece": float(np.mean([r["ece"] for r in br])),
                        "band_brier": float(np.mean([r["brier"] for r in br])),
                    }
                )

        h = fits[src["hard"]]
        m = fits[src["method"]]
        deltas.append(
            {
                "dataset": src["dataset"],
                "model": src["model"],
                "method": src["method"],
                "delta_gamma_c_method_minus_hard": m["gamma_c"] - h["gamma_c"],
                "delta_snr_auc": m["auc"] - h["auc"],
                "delta_A_max": m["amax"] - h["amax"],
                "delta_A_min": m["amin"] - h["amin"],
                "delta_slope_k": m["k"] - h["k"],
                "hard_fit_rmse": h["rmse"],
                "method_fit_rmse": m["rmse"],
            }
        )

    write_csv(
        OUT / "transition_fit_summary.csv",
        summary,
        ["dataset", "model", "method", "A_min", "A_max", "slope_k", "gamma_c", "snr_auc", "fit_rmse", "hard_transition_band_snrs"],
    )
    write_csv(
        OUT / "transition_deltas.csv",
        deltas,
        ["dataset", "model", "method", "delta_gamma_c_method_minus_hard", "delta_snr_auc", "delta_A_max", "delta_A_min", "delta_slope_k", "hard_fit_rmse", "method_fit_rmse"],
    )
    write_csv(
        OUT / "transition_band_metrics.csv",
        band_rows,
        ["dataset", "model", "method", "band_snrs", "band_accuracy", "band_nll", "band_ece", "band_brier"],
    )
    write_csv(
        OUT / "transition_fit_series.csv",
        fit_series,
        ["dataset", "model", "method", "snr", "accuracy", "fit_accuracy"],
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"Hard CE": "#0072B2", "DPC-RCPS": "#D55E00", "RCPS-Hybrid": "#009E73"}
        fig, axes = plt.subplots(1, len(SOURCES), figsize=(11.0, 3.0), sharey=True)
        if len(SOURCES) == 1:
            axes = [axes]
        for ax, src in zip(axes, SOURCES):
            for method in [src["hard"], src["method"]]:
                rows = [r for r in fit_series if r["dataset"] == src["dataset"] and r["model"] == src["model"] and r["method"] == method]
                rows = sorted(rows, key=lambda r: r["snr"])
                ax.plot([r["snr"] for r in rows], [100 * r["accuracy"] for r in rows], "o", ms=3.5, color=colors.get(method, "black"), label=method)
                ax.plot([r["snr"] for r in rows], [100 * r["fit_accuracy"] for r in rows], "-", lw=1.4, color=colors.get(method, "black"))
            ax.set_title(f"{src['dataset']}\n{src['model']}", fontsize=8)
            ax.set_xlabel("SNR (dB)", fontsize=8)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("Accuracy (%)", fontsize=8)
        axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout(pad=0.5)
        fig.savefig(OUT / "fig_transition_fits_amc.pdf")
        fig.savefig(OUT / "fig_transition_fits_amc.svg")

        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        labels = [f"{d['dataset']}\n{d['model']}" for d in deltas]
        vals = [float(d["delta_gamma_c_method_minus_hard"]) for d in deltas]
        ax.axhline(0, color="0.25", lw=0.8)
        ax.bar(range(len(vals)), vals, color="#CC79A7", width=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(r"$\Delta\gamma_c$ (dB)", fontsize=8)
        ax.set_title("Critical-SNR shift relative to Hard CE", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(pad=0.5)
        fig.savefig(OUT / "fig_transition_delta_gamma_c.pdf")
        fig.savefig(OUT / "fig_transition_delta_gamma_c.svg")
    except Exception as exc:
        (OUT / "figure_error.txt").write_text(str(exc))

    print(f"Wrote transition reanalysis to {OUT}")
    for d in deltas:
        print(
            f"{d['dataset']} {d['model']} {d['method']}: "
            f"delta_gamma_c={float(d['delta_gamma_c_method_minus_hard']):.3f} dB, "
            f"delta_auc={float(d['delta_snr_auc']):.4f}"
        )


if __name__ == "__main__":
    main()
