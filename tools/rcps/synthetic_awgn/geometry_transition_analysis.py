#!/usr/bin/env python3
"""Geometry-transition diagnostics for clean-paired synthetic AWGN AMC data.

The analysis links the empirical waterfall curve to class-geometry quantities:
class-center separation, within-class radius, Fisher ratio, and a silhouette
proxy.  It is intentionally data-space first, because the controlled AWGN
dataset exposes the physical noisy observation directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction(path: Path | None) -> Dict | None:
    if path is None:
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload


def iq_vector(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float64)
    if arr.shape[0] != 2:
        raise ValueError(f"Expected IQ shape (2,L), got {arr.shape} at {path}")
    # Keep raw scale. Per-sample normalization would remove the SNR-dependent
    # noise radius that the geometry diagnostic is designed to measure.
    return arr.reshape(-1).astype(np.float32)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)


def nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(labels.size), labels.astype(int)])


def select_items(data_list: List[Dict], max_per_class_snr: int, seed: int) -> List[Dict]:
    if max_per_class_snr <= 0:
        return data_list
    rng = np.random.default_rng(seed)
    buckets: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for item in data_list:
        buckets[(item["modulation"], int(item["snr"]))].append(item)
    selected: List[Dict] = []
    for key in sorted(buckets):
        values = buckets[key]
        idx = np.arange(len(values))
        rng.shuffle(idx)
        selected.extend(values[i] for i in idx[:max_per_class_snr])
    return selected


def prediction_index(pred: Dict | None) -> Dict[int, int]:
    if pred is None:
        return {}
    return {int(idx): pos for pos, idx in enumerate(np.asarray(pred["sample_idx"]).astype(int))}


def class_geometry(x: np.ndarray, y: np.ndarray, num_classes: int) -> Dict[str, float]:
    centers = []
    counts = []
    within_sum = 0.0
    within_sq_sum = 0.0
    own_dist = np.zeros(x.shape[0], dtype=np.float64)
    global_mu = x.mean(axis=0)

    for cls in range(num_classes):
        mask = y == cls
        if not mask.any():
            continue
        sub = x[mask]
        center = sub.mean(axis=0)
        centers.append(center)
        counts.append(int(mask.sum()))
        d = np.linalg.norm(sub - center[None, :], axis=1)
        own_dist[mask] = d
        within_sum += float(d.sum())
        within_sq_sum += float((d ** 2).sum())

    centers_arr = np.stack(centers, axis=0)
    counts_arr = np.asarray(counts, dtype=np.float64)
    center_dists = []
    for i in range(centers_arr.shape[0]):
        for j in range(i + 1, centers_arr.shape[0]):
            center_dists.append(float(np.linalg.norm(centers_arr[i] - centers_arr[j])))
    center_dists_arr = np.asarray(center_dists, dtype=np.float64)

    between = 0.0
    for center, count in zip(centers_arr, counts_arr):
        between += float(count * np.sum((center - global_mu) ** 2))
    within = max(within_sq_sum, 1e-12)

    # Center-based silhouette proxy: own-center distance vs nearest other center.
    other_dist = np.zeros(x.shape[0], dtype=np.float64)
    for idx, row in enumerate(x):
        cls = int(y[idx])
        distances = np.linalg.norm(centers_arr - row[None, :], axis=1)
        distances[cls] = np.inf
        other_dist[idx] = float(np.min(distances))
    denom = np.maximum(own_dist, other_dist)
    silhouette_proxy = np.where(denom > 1e-12, (other_dist - own_dist) / denom, 0.0)

    mean_radius = within_sum / max(x.shape[0], 1)
    rms_radius = float(np.sqrt(within_sq_sum / max(x.shape[0], 1)))
    min_center_dist = float(center_dists_arr.min())
    mean_center_dist = float(center_dists_arr.mean())
    noise_tube_ratio = float(min_center_dist / max(2.0 * mean_radius, 1e-12))
    overlap_proxy = float((2.0 * mean_radius) / max(min_center_dist, 1e-12))
    bhattacharyya_proxy = float(np.exp(-(min_center_dist ** 2) / max(8.0 * (rms_radius ** 2), 1e-12)))

    return {
        "num_samples": int(x.shape[0]),
        "mean_within_radius": float(mean_radius),
        "rms_within_radius": float(rms_radius),
        "min_center_distance": min_center_dist,
        "mean_center_distance": mean_center_dist,
        "noise_tube_ratio": noise_tube_ratio,
        "overlap_proxy": overlap_proxy,
        "fisher_ratio": float(between / within),
        "silhouette_proxy": float(np.mean(silhouette_proxy)),
        "bhattacharyya_proxy": bhattacharyya_proxy,
    }


def paired_noise_geometry(x: np.ndarray, clean: np.ndarray, y: np.ndarray, num_classes: int) -> Dict[str, float]:
    clean_centers = []
    for cls in range(num_classes):
        mask = y == cls
        if mask.any():
            clean_centers.append(clean[mask].mean(axis=0))
    centers_arr = np.stack(clean_centers, axis=0)
    center_dists = []
    for i in range(centers_arr.shape[0]):
        for j in range(i + 1, centers_arr.shape[0]):
            center_dists.append(float(np.linalg.norm(centers_arr[i] - centers_arr[j])))
    center_dists_arr = np.asarray(center_dists, dtype=np.float64)
    noise_norm = np.linalg.norm(x - clean, axis=1)
    mean_noise_radius = float(noise_norm.mean())
    rms_noise_radius = float(np.sqrt((noise_norm ** 2).mean()))
    min_clean_center_dist = float(center_dists_arr.min())
    mean_clean_center_dist = float(center_dists_arr.mean())
    paired_tube_ratio = float(min_clean_center_dist / max(2.0 * mean_noise_radius, 1e-12))
    paired_overlap = float((2.0 * mean_noise_radius) / max(min_clean_center_dist, 1e-12))
    paired_bhattacharyya = float(np.exp(-(min_clean_center_dist ** 2) / max(8.0 * (rms_noise_radius ** 2), 1e-12)))
    return {
        "mean_noise_radius": mean_noise_radius,
        "rms_noise_radius": rms_noise_radius,
        "min_clean_center_distance": min_clean_center_dist,
        "mean_clean_center_distance": mean_clean_center_dist,
        "paired_noise_tube_ratio": paired_tube_ratio,
        "paired_overlap_proxy": paired_overlap,
        "paired_bhattacharyya_proxy": paired_bhattacharyya,
    }


def normalized_midpoint_snr(snrs: np.ndarray, values: np.ndarray, increasing: bool = True) -> float:
    order = np.argsort(snrs)
    xs = snrs[order].astype(np.float64)
    ys = values[order].astype(np.float64)
    if not increasing:
        ys = -ys
    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
    target = lo + 0.5 * (hi - lo)
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            if abs(y1 - y0) < 1e-12:
                return float(xs[i])
            t = (target - y0) / (y1 - y0)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return float(xs[np.argmin(np.abs(ys - target))])


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(out_dir: Path, rows: List[Dict], critical_rows: List[Dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot: {exc}")
        return

    rows = sorted(rows, key=lambda r: r["snr"])
    snr = np.asarray([r["snr"] for r in rows], dtype=float)
    acc = np.asarray([r.get("accuracy", np.nan) for r in rows], dtype=float)
    ent = np.asarray([r.get("mean_entropy", np.nan) for r in rows], dtype=float)
    fisher = np.asarray([r["fisher_ratio"] for r in rows], dtype=float)
    margin = np.asarray([r["noise_tube_ratio"] for r in rows], dtype=float)
    sil = np.asarray([r["silhouette_proxy"] for r in rows], dtype=float)
    overlap = np.asarray([r["paired_overlap_proxy"] for r in rows], dtype=float)
    paired_tube = np.asarray([r["paired_noise_tube_ratio"] for r in rows], dtype=float)

    def z(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return (v - lo) / max(hi - lo, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
    axes[0].plot(snr, acc, marker="o", label="Accuracy")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Accuracy (%)")
    ax0b = axes[0].twinx()
    ax0b.plot(snr, ent, marker="s", color="#D55E00", label="Entropy")
    ax0b.set_ylabel("Entropy")
    axes[0].set_title("Waterfall/posterior")

    axes[1].plot(snr, z(fisher), marker="o", label="Fisher")
    axes[1].plot(snr, z(paired_tube), marker="s", label="Paired tube")
    axes[1].plot(snr, z(sil), marker="^", label="Silhouette")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Normalized value")
    axes[1].set_title("Geometry order parameters")
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].plot(snr, overlap, marker="o", label="2 noise radius / clean dmin")
    axes[2].axhline(1.0, color="0.35", lw=1.0, ls="--", label="tube crossing")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Overlap proxy")
    axes[2].set_title("Noise-tube crossing")
    axes[2].legend(frameon=False, fontsize=7)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"synthetic_awgn_geometry_transition.{ext}", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--prediction", type=Path, default=Path("/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/stage1_petcgdnn_seed2026_30ep/hard_ce/predictions/test.pkl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-class-snr", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = load_payload(args.dataset_root / f"{args.split}.json")
    classes = list(payload["metainfo"].get("modulations", []))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    items = select_items(payload["data_list"], args.max_per_class_snr, args.seed)
    pred = load_prediction(args.prediction) if args.prediction else None
    pred_idx = prediction_index(pred)
    probs = np.asarray(pred["pps"]) if pred is not None else None
    pred_labels = np.asarray(pred["gts"]).astype(int) if pred is not None else None

    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for item in items:
        grouped[int(item["snr"])].append(item)

    rows: List[Dict] = []
    for snr in sorted(grouped):
        xs = []
        cs = []
        ys = []
        pred_positions = []
        for item in grouped[snr]:
            xs.append(iq_vector(args.dataset_root / "iq" / item["file_name"]))
            cs.append(iq_vector(args.dataset_root / "clean" / item["clean_file_name"]))
            ys.append(class_to_idx[item["modulation"]])
            if pred is not None:
                pos = pred_idx.get(int(item["sample_idx"]))
                if pos is not None:
                    pred_positions.append(pos)
        x = np.stack(xs, axis=0)
        clean = np.stack(cs, axis=0)
        y = np.asarray(ys, dtype=int)
        geom = class_geometry(x, y, len(classes))
        geom.update(paired_noise_geometry(x, clean, y, len(classes)))
        row = {"snr": int(snr), **geom}
        if pred is not None and pred_positions:
            pp = probs[np.asarray(pred_positions, dtype=int)]
            gt = pred_labels[np.asarray(pred_positions, dtype=int)]
            row.update({
                "accuracy": float((pp.argmax(axis=1) == gt).mean() * 100.0),
                "mean_entropy": float(entropy(pp).mean()),
                "nll": float(nll(pp, gt).mean()),
            })
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "synthetic_awgn_geometry_by_snr.csv", rows)

    snrs = np.asarray([r["snr"] for r in rows], dtype=float)
    critical_rows = []
    metric_specs = [
        ("accuracy", True), ("mean_entropy", False), ("fisher_ratio", True),
        ("noise_tube_ratio", True), ("silhouette_proxy", True), ("overlap_proxy", False),
        ("paired_noise_tube_ratio", True), ("paired_overlap_proxy", False),
    ]
    for metric, inc in metric_specs:
        vals = np.asarray([r.get(metric, np.nan) for r in rows], dtype=float)
        if np.isfinite(vals).all():
            critical_rows.append({
                "metric": metric,
                "midpoint_snr": normalized_midpoint_snr(snrs, vals, increasing=inc),
                "increasing": inc,
            })
    write_csv(args.out_dir / "synthetic_awgn_geometry_critical_points.csv", critical_rows)

    corr_rows = []
    acc = np.asarray([r.get("accuracy", np.nan) for r in rows], dtype=float)
    for metric in [
        "fisher_ratio", "noise_tube_ratio", "silhouette_proxy", "overlap_proxy",
        "bhattacharyya_proxy", "paired_noise_tube_ratio", "paired_overlap_proxy",
        "paired_bhattacharyya_proxy",
    ]:
        vals = np.asarray([r[metric] for r in rows], dtype=float)
        if np.isfinite(acc).all():
            corr_rows.append({
                "metric": metric,
                "pearson_with_accuracy": float(np.corrcoef(vals, acc)[0, 1]),
                "pearson_with_entropy": float(np.corrcoef(vals, np.asarray([r["mean_entropy"] for r in rows], dtype=float))[0, 1]),
            })
    write_csv(args.out_dir / "synthetic_awgn_geometry_correlations.csv", corr_rows)
    maybe_plot(args.out_dir, rows, critical_rows)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "rows": len(rows),
        "classes": classes,
        "max_per_class_snr": args.max_per_class_snr,
        "critical_points": critical_rows,
        "correlations": corr_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
