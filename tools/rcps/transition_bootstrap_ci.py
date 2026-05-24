#!/usr/bin/env python3
"""Bootstrap transition metrics from seed-level reliability-bin CSVs.

The script is intentionally post-hoc and read-only with respect to training
artifacts. It strengthens the transition reanalysis by computing paired
bootstrap intervals and a normalized transition-AUC decomposition that includes
plateau, floor, critical-point, and slope-change terms.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("/home/citybuster/Data/RCPS/work_dirs")
OUT = ROOT / "transition_reanalysis"
SEEDS = (2026, 2027, 2028)
N_BOOT = 5000
RNG_SEED = 240524


@dataclass(frozen=True)
class Source:
    dataset: str
    model: str
    method: str
    hard_pattern: str
    method_pattern: str


SOURCES = [
    Source(
        dataset="RadioML2016.10A",
        model="PETCGDNN",
        method="DPC-RCPS",
        hard_pattern=str(ROOT / "baseline_gate_v2/metrics/deepsig201610A_petcgdnn_hard-ce_seed{seed}_test.csv"),
        method_pattern=str(ROOT / "dpc_main/metrics/deepsig201610A_petcgdnn_dpc-rcps_seed{seed}_test.csv"),
    ),
    Source(
        dataset="RadioML2016.10B",
        model="MCformer",
        method="DPC-RCPS",
        hard_pattern=str(ROOT / "mcformer_gate_10B_400ep/metrics/deepsig201610B_amr_mcformer_hard-ce_seed{seed}_test.csv"),
        method_pattern=str(ROOT / "dpc_main/metrics/deepsig201610B_mcformer_dpc-rcps_seed{seed}_test.csv"),
    ),
    Source(
        dataset="RadioML2018.01A",
        model="PETCGDNN",
        method="RCPS-Hybrid",
        hard_pattern=str(ROOT / "baseline_gate_2018A/metrics/deepsig201801A_petcgdnn_hard-ce_seed{seed}_test.csv"),
        method_pattern=str(ROOT / "rcps_hybrid_2018A/metrics/deepsig201801A_petcgdnn_rcps-hybrid-eps01_seed{seed}_test.csv"),
    ),
]


METRIC_COLUMNS = {
    "accuracy": "accuracy",
    "nll": "nll",
    "ece": "ece",
    "brier": "brier",
}


def logistic(x: np.ndarray, amin: float, amax: float, k: float, gc: float) -> np.ndarray:
    return amin + (amax - amin) / (1.0 + np.exp(-k * (x - gc)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
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
        return np.asarray(popt, dtype=float), float(np.sqrt(np.mean((pred - y) ** 2)))
    except Exception:
        best = None
        amin_grid = np.linspace(max(0.0, y.min() - 0.08), min(0.35, y.min() + 0.08), 17)
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
        params = np.asarray([amin, amax, k, gc], dtype=float)
        pred = logistic(x, *params)
        return params, float(np.sqrt(np.mean((pred - y) ** 2)))


def normalized_auc_fit(params: np.ndarray, xmin: float, xmax: float, n: int = 1001) -> float:
    xs = np.linspace(xmin, xmax, n)
    ys = logistic(xs, *params)
    return float(np.trapz(ys, xs) / (xmax - xmin))


def normalized_auc_points(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    return float(np.trapz(y, x) / (x.max() - x.min()))


def read_metric_file(path: Path) -> dict[float, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[float, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                snr = float(row["reliability_bin"])
            except ValueError:
                continue
            values = {
                "accuracy": float(row["accuracy"]) / 100.0,
                "nll": float(row["nll"]),
                "ece": float(row["ece"]),
                "brier": float(row["brier"]),
            }
            out[snr] = values
    return out


def load_source(src: Source) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    per_seed_hard = []
    per_seed_method = []
    snrs_ref = None
    for seed in SEEDS:
        hard = read_metric_file(Path(src.hard_pattern.format(seed=seed)))
        method = read_metric_file(Path(src.method_pattern.format(seed=seed)))
        snrs = np.array(sorted(set(hard) & set(method)), dtype=float)
        if snrs_ref is None:
            snrs_ref = snrs
        elif not np.array_equal(snrs_ref, snrs):
            raise ValueError(f"SNR mismatch for {src.dataset}/{src.model}/seed{seed}")
        per_seed_hard.append(hard)
        per_seed_method.append(method)
    assert snrs_ref is not None

    def stack(rows: list[dict[float, dict[str, float]]]) -> dict[str, np.ndarray]:
        data = {}
        for metric in METRIC_COLUMNS:
            data[metric] = np.asarray([[r[float(s)][metric] for s in snrs_ref] for r in rows], dtype=float)
        return data

    return snrs_ref, stack(per_seed_hard), stack(per_seed_method)


def transition_band_snrs(x: np.ndarray, hard_params: np.ndarray) -> np.ndarray:
    amin, amax, k, gc = hard_params
    lo = amin + 0.2 * (amax - amin)
    hi = amin + 0.8 * (amax - amin)
    pred = logistic(x, *hard_params)
    mask = (pred >= lo) & (pred <= hi)
    if not mask.any():
        mask = np.abs(x - gc) <= 4.0
    return x[mask]


def auc_decomposition(hard_params: np.ndarray, method_params: np.ndarray, xmin: float, xmax: float) -> dict[str, float]:
    amin_h, amax_h, k_h, gc_h = [float(v) for v in hard_params]
    amin_m, amax_m, k_m, gc_m = [float(v) for v in method_params]
    L = xmax - xmin
    xs = np.linspace(xmin, xmax, 4001)
    sig = 1.0 / (1.0 + np.exp(-k_h * (xs - gc_h)))
    amp = amax_h - amin_h
    wh = float(np.trapz(sig, xs) / L)
    wl = 1.0 - wh
    sig_max = 1.0 / (1.0 + math.exp(-k_h * (xmax - gc_h)))
    sig_min = 1.0 / (1.0 + math.exp(-k_h * (xmin - gc_h)))
    cc = (sig_max - sig_min) / L
    wk = float(amp * np.trapz((xs - gc_h) * sig * (1.0 - sig), xs) / L)
    d_amin = amin_m - amin_h
    d_amax = amax_m - amax_h
    d_k = k_m - k_h
    d_gc = gc_m - gc_h
    plateau = wh * d_amax
    floor = wl * d_amin
    critical = -amp * cc * d_gc
    slope = wk * d_k
    fit_delta_auc = normalized_auc_fit(method_params, xmin, xmax) - normalized_auc_fit(hard_params, xmin, xmax)
    first_order = plateau + floor + critical + slope
    return {
        "fit_delta_auc": fit_delta_auc,
        "plateau_term": plateau,
        "floor_term": floor,
        "critical_shift_term": critical,
        "slope_term": slope,
        "first_order_sum": first_order,
        "second_order_residual": fit_delta_auc - first_order,
        "W_h": wh,
        "W_l": wl,
        "C_c": cc,
        "W_k": wk,
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def analyze_source(src: Source, rng: np.random.Generator) -> tuple[dict[str, object], dict[str, object]]:
    snrs, hard, method = load_source(src)
    xmin, xmax = float(snrs.min()), float(snrs.max())

    hard_mean = hard["accuracy"].mean(axis=0)
    method_mean = method["accuracy"].mean(axis=0)
    hard_params, hard_rmse = fit_logistic(snrs, hard_mean)
    method_params, method_rmse = fit_logistic(snrs, method_mean)
    band = transition_band_snrs(snrs, hard_params)
    band_mask = np.isin(snrs, band)

    obs_decomp = auc_decomposition(hard_params, method_params, xmin, xmax)
    observed = {
        "dataset": src.dataset,
        "model": src.model,
        "method": src.method,
        "delta_gamma_c": float(method_params[3] - hard_params[3]),
        "delta_A_max": float(method_params[1] - hard_params[1]),
        "delta_A_min": float(method_params[0] - hard_params[0]),
        "delta_slope_k": float(method_params[2] - hard_params[2]),
        "delta_auc_points": normalized_auc_points(snrs, method_mean) - normalized_auc_points(snrs, hard_mean),
        "hard_fit_rmse": hard_rmse,
        "method_fit_rmse": method_rmse,
        "band_snrs": " ".join(str(int(s)) if float(s).is_integer() else str(s) for s in band),
        "delta_band_accuracy": float(method["accuracy"][:, band_mask].mean() - hard["accuracy"][:, band_mask].mean()),
        "delta_band_nll": float(method["nll"][:, band_mask].mean() - hard["nll"][:, band_mask].mean()),
        "delta_band_ece": float(method["ece"][:, band_mask].mean() - hard["ece"][:, band_mask].mean()),
        "delta_band_brier": float(method["brier"][:, band_mask].mean() - hard["brier"][:, band_mask].mean()),
        **obs_decomp,
    }

    boot: dict[str, list[float]] = {
        "delta_gamma_c": [],
        "delta_A_max": [],
        "delta_A_min": [],
        "delta_slope_k": [],
        "delta_auc_points": [],
        "fit_delta_auc": [],
        "plateau_term": [],
        "floor_term": [],
        "critical_shift_term": [],
        "slope_term": [],
        "second_order_residual": [],
        "delta_band_nll": [],
        "delta_band_brier": [],
    }
    n_seed = len(SEEDS)
    for _ in range(N_BOOT):
        # Paired, SNR-stratified seed bootstrap: for each reliability bin, resample
        # the same seed indices for hard and RCPS so bin difficulty is preserved.
        hard_acc = []
        method_acc = []
        hard_nll_band = []
        method_nll_band = []
        hard_brier_band = []
        method_brier_band = []
        for j, s in enumerate(snrs):
            idx = rng.integers(0, n_seed, size=n_seed)
            hard_acc.append(float(hard["accuracy"][idx, j].mean()))
            method_acc.append(float(method["accuracy"][idx, j].mean()))
            if s in set(float(v) for v in band):
                hard_nll_band.append(float(hard["nll"][idx, j].mean()))
                method_nll_band.append(float(method["nll"][idx, j].mean()))
                hard_brier_band.append(float(hard["brier"][idx, j].mean()))
                method_brier_band.append(float(method["brier"][idx, j].mean()))
        hard_acc_arr = np.asarray(hard_acc, dtype=float)
        method_acc_arr = np.asarray(method_acc, dtype=float)
        hp, _ = fit_logistic(snrs, hard_acc_arr)
        mp, _ = fit_logistic(snrs, method_acc_arr)
        dec = auc_decomposition(hp, mp, xmin, xmax)
        boot["delta_gamma_c"].append(float(mp[3] - hp[3]))
        boot["delta_A_max"].append(float(mp[1] - hp[1]))
        boot["delta_A_min"].append(float(mp[0] - hp[0]))
        boot["delta_slope_k"].append(float(mp[2] - hp[2]))
        boot["delta_auc_points"].append(normalized_auc_points(snrs, method_acc_arr) - normalized_auc_points(snrs, hard_acc_arr))
        for key in ["fit_delta_auc", "plateau_term", "floor_term", "critical_shift_term", "slope_term", "second_order_residual"]:
            boot[key].append(float(dec[key]))
        boot["delta_band_nll"].append(float(np.mean(method_nll_band) - np.mean(hard_nll_band)))
        boot["delta_band_brier"].append(float(np.mean(method_brier_band) - np.mean(hard_brier_band)))

    ci_row = {
        "dataset": src.dataset,
        "model": src.model,
        "method": src.method,
        "n_boot": N_BOOT,
    }
    for key, vals in boot.items():
        vals_arr = np.asarray(vals, dtype=float)
        s = summarize(vals_arr)
        ci_row[f"{key}_mean"] = s["mean"]
        ci_row[f"{key}_ci_low"] = s["ci_low"]
        ci_row[f"{key}_ci_high"] = s["ci_high"]
        if key.startswith("delta_") or key.endswith("_term"):
            ci_row[f"{key}_sign_p"] = float(2.0 * min(np.mean(vals_arr <= 0), np.mean(vals_arr >= 0)))
    return observed, ci_row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    observed_rows = []
    ci_rows = []
    for src in SOURCES:
        obs, ci = analyze_source(src, rng)
        observed_rows.append(obs)
        ci_rows.append(ci)
        print(
            f"{src.dataset} / {src.model} / {src.method}: "
            f"delta_gamma_c={obs['delta_gamma_c']:.3f}, "
            f"delta_Amax={obs['delta_A_max']:.4f}, "
            f"fit_delta_auc={obs['fit_delta_auc']:.4f}, "
            f"plateau={obs['plateau_term']:.4f}, "
            f"slope={obs['slope_term']:.4f}"
        )

    observed_fields = [
        "dataset",
        "model",
        "method",
        "delta_gamma_c",
        "delta_A_max",
        "delta_A_min",
        "delta_slope_k",
        "delta_auc_points",
        "fit_delta_auc",
        "plateau_term",
        "floor_term",
        "critical_shift_term",
        "slope_term",
        "first_order_sum",
        "second_order_residual",
        "W_h",
        "W_l",
        "C_c",
        "W_k",
        "hard_fit_rmse",
        "method_fit_rmse",
        "band_snrs",
        "delta_band_accuracy",
        "delta_band_nll",
        "delta_band_ece",
        "delta_band_brier",
    ]
    write_csv(OUT / "transition_auc_decomposition_v2.csv", observed_rows, observed_fields)

    ci_fields = ["dataset", "model", "method", "n_boot"]
    metrics = [
        "delta_gamma_c",
        "delta_A_max",
        "delta_A_min",
        "delta_slope_k",
        "delta_auc_points",
        "fit_delta_auc",
        "plateau_term",
        "floor_term",
        "critical_shift_term",
        "slope_term",
        "second_order_residual",
        "delta_band_nll",
        "delta_band_brier",
    ]
    for metric in metrics:
        ci_fields += [f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"]
        if metric.startswith("delta_") or metric.endswith("_term"):
            ci_fields += [f"{metric}_sign_p"]
    write_csv(OUT / "transition_bootstrap_ci_v2.csv", ci_rows, ci_fields)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"Hard CE": "#0072B2", "DPC-RCPS": "#D55E00", "RCPS-Hybrid": "#009E73"}
        fig, axes = plt.subplots(1, len(SOURCES), figsize=(11.0, 3.0), sharey=True)
        if len(SOURCES) == 1:
            axes = [axes]
        for ax, src in zip(axes, SOURCES):
            snrs, hard, method = load_source(src)
            hard_y = hard["accuracy"].mean(axis=0)
            method_y = method["accuracy"].mean(axis=0)
            hard_params, _ = fit_logistic(snrs, hard_y)
            method_params, _ = fit_logistic(snrs, method_y)
            xs = np.linspace(float(snrs.min()), float(snrs.max()), 400)
            ax.plot(snrs, 100 * hard_y, "o", ms=3.5, color=colors["Hard CE"], label="Hard CE")
            ax.plot(xs, 100 * logistic(xs, *hard_params), "-", lw=1.4, color=colors["Hard CE"])
            ax.plot(snrs, 100 * method_y, "o", ms=3.5, color=colors.get(src.method, "black"), label=src.method)
            ax.plot(xs, 100 * logistic(xs, *method_params), "-", lw=1.4, color=colors.get(src.method, "black"))
            ax.set_title(f"{src.dataset}\n{src.model}", fontsize=8)
            ax.set_xlabel("SNR (dB)", fontsize=8)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("Accuracy (%)", fontsize=8)
        axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout(pad=0.5)
        fig.savefig(OUT / "fig_transition_fits_amc_v2.pdf")
        fig.savefig(OUT / "fig_transition_fits_amc_v2.svg")

        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        labels = [f"{row['dataset']}\n{row['model']}" for row in observed_rows]
        vals = [float(row["delta_gamma_c"]) for row in observed_rows]
        lows = [float(row["delta_gamma_c"]) - float(ci["delta_gamma_c_ci_low"]) for row, ci in zip(observed_rows, ci_rows)]
        highs = [float(ci["delta_gamma_c_ci_high"]) - float(row["delta_gamma_c"]) for row, ci in zip(observed_rows, ci_rows)]
        ax.axhline(0, color="0.25", lw=0.8)
        ax.bar(range(len(vals)), vals, color="#CC79A7", width=0.6)
        ax.errorbar(range(len(vals)), vals, yerr=[lows, highs], fmt="none", ecolor="0.15", elinewidth=0.8, capsize=3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(r"$\Delta\gamma_c$ (dB)", fontsize=8)
        ax.set_title("Critical-SNR shift relative to Hard CE", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(pad=0.5)
        fig.savefig(OUT / "fig_transition_delta_gamma_c_v2.pdf")
        fig.savefig(OUT / "fig_transition_delta_gamma_c_v2.svg")
    except Exception as exc:
        (OUT / "transition_bootstrap_figure_error.txt").write_text(str(exc))

    print(f"Wrote {OUT / 'transition_auc_decomposition_v2.csv'}")
    print(f"Wrote {OUT / 'transition_bootstrap_ci_v2.csv'}")


if __name__ == "__main__":
    main()
