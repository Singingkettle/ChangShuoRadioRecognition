#!/usr/bin/env python3
"""Feature-geometry diagnostics for real AMC reliability transitions.

This script reuses a trained CSRR checkpoint and computes SNR-stratified
feature separability.  It is intentionally analysis-only: no training state is
modified, and all outputs are written under an explicit out directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import os.path as osp
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


SPLIT_TO_DATALOADER = {
    "train": "train_dataloader",
    "validation": "val_dataloader",
    "val": "val_dataloader",
    "test": "test_dataloader",
}


def set_default_dataloader_cfg(cfg, field: str) -> None:
    if cfg.get(field, None) is None:
        return
    dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type="default_collate"),
    )
    if digit_version(TORCH_VERSION) < digit_version("1.8.0"):
        dataloader_cfg.persistent_workers = False
    dataloader_cfg.update(deepcopy(cfg[field]))
    cfg[field] = dataloader_cfg


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def numeric_snr(value) -> float:
    if isinstance(value, str) and value.lower() == "clean":
        return 30.0
    return float(value)


def find_last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            last = submodule
    return last


class FeatureHook:
    def __init__(self, model: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self.handle = None
        self.source = "logits"
        candidate = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "classifier"):
            candidate = find_last_linear(model.backbone.classifier)
        if candidate is not None:
            self.source = "last_linear_input"
            self.handle = candidate.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        self.features = inputs[0].detach()

    def pop(self, fallback: torch.Tensor) -> torch.Tensor:
        if self.features is None:
            return fallback.detach()
        out = self.features
        self.features = None
        return out

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def maybe_subsample(indices: np.ndarray, labels: np.ndarray, snrs: np.ndarray,
                    max_per_class_snr: int, seed: int) -> np.ndarray:
    if max_per_class_snr <= 0:
        return indices
    rng = np.random.default_rng(seed)
    keep = []
    for snr in sorted(np.unique(snrs), key=numeric_snr):
        for label in sorted(np.unique(labels)):
            bucket = indices[(snrs[indices] == snr) & (labels[indices] == label)]
            if bucket.size > max_per_class_snr:
                bucket = rng.choice(bucket, size=max_per_class_snr, replace=False)
            keep.append(bucket)
    if not keep:
        return indices
    return np.sort(np.concatenate(keep))


def center_geometry(features: np.ndarray, labels: np.ndarray,
                    num_classes: int) -> Dict[str, float]:
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    labels = labels.astype(np.int64)
    global_center = features.mean(axis=0)
    centers = []
    radii = []
    within_scatter = 0.0
    valid_classes = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_feat = features[mask]
        center = cls_feat.mean(axis=0)
        dist = np.linalg.norm(cls_feat - center, axis=1)
        centers.append(center)
        radii.append(float(dist.mean()))
        within_scatter += float(((cls_feat - center) ** 2).sum())
        valid_classes.append(cls)
    if len(centers) < 2:
        return {
            "n_classes_present": len(centers),
            "fisher_ratio": float("nan"),
            "min_center_dist": float("nan"),
            "mean_radius": float("nan"),
            "overlap_proxy": float("nan"),
            "margin_ratio": float("nan"),
            "silhouette_proxy": float("nan"),
        }

    centers = np.stack(centers)
    radii_arr = np.asarray(radii)
    diffs = centers[:, None, :] - centers[None, :, :]
    center_dists = np.linalg.norm(diffs, axis=-1)
    center_dists = center_dists + np.eye(center_dists.shape[0]) * 1e12
    min_center_dist = float(center_dists.min())
    mean_radius = float(radii_arr.mean())
    between_scatter = 0.0
    for center in centers:
        between_scatter += float(np.sum((center - global_center) ** 2))
    fisher_ratio = between_scatter / max(within_scatter / max(features.shape[0], 1), 1e-12)

    # Prototype silhouette: distance to own class center vs nearest other center.
    own_center = np.zeros_like(features)
    for i, cls in enumerate(valid_classes):
        own_center[labels == cls] = centers[i]
    own_dist = np.linalg.norm(features - own_center, axis=1)
    all_center_dist = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=-1)
    for i, cls in enumerate(valid_classes):
        all_center_dist[labels == cls, i] = np.inf
    other_dist = np.min(all_center_dist, axis=1)
    denom = np.maximum(own_dist, other_dist).clip(min=1e-12)
    silhouette = float(np.mean((other_dist - own_dist) / denom))

    return {
        "n_classes_present": len(centers),
        "fisher_ratio": float(fisher_ratio),
        "min_center_dist": min_center_dist,
        "mean_radius": mean_radius,
        "overlap_proxy": float((2.0 * mean_radius) / max(min_center_dist, 1e-12)),
        "margin_ratio": float(min_center_dist / max(2.0 * mean_radius, 1e-12)),
        "silhouette_proxy": silhouette,
    }


def entropy_np(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    return -(probs * np.log(probs)).sum(axis=1)


def summarize_by_snr(features: np.ndarray, probs: np.ndarray, labels: np.ndarray,
                     snrs: np.ndarray, classes: List[str]) -> List[Dict]:
    rows = []
    for snr in sorted(np.unique(snrs), key=numeric_snr):
        mask = snrs == snr
        if mask.sum() < 10:
            continue
        feat = features[mask]
        y = labels[mask]
        p = probs[mask]
        pred = p.argmax(axis=1)
        geom = center_geometry(feat, y, len(classes))
        row = {
            "snr": str(snr),
            "snr_value": numeric_snr(snr),
            "n": int(mask.sum()),
            "accuracy": float((pred == y).mean() * 100.0),
            "mean_entropy": float(entropy_np(p).mean()),
            "mean_confidence": float(p.max(axis=1).mean()),
            **geom,
        }
        rows.append(row)
    return rows


def midpoint_snr(rows: List[Dict], metric: str, increasing: bool) -> float:
    xs = np.asarray([r["snr_value"] for r in rows], dtype=float)
    ys = np.asarray([r[metric] for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    lo, hi = float(np.min(ys)), float(np.max(ys))
    target = lo + 0.5 * (hi - lo)
    if not increasing:
        ys = -ys
        target = -target
    idx = np.argsort(ys)
    y_sorted = ys[idx]
    x_sorted = xs[idx]
    unique_y, unique_idx = np.unique(y_sorted, return_index=True)
    unique_x = x_sorted[unique_idx]
    if unique_y.size < 2:
        return float("nan")
    return float(np.interp(target if increasing else -target, unique_y, unique_x))


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(out_dir: Path, rows: List[Dict], title: str) -> None:
    snr = np.asarray([r["snr_value"] for r in rows], dtype=float)
    acc = np.asarray([r["accuracy"] for r in rows], dtype=float)
    ent = np.asarray([r["mean_entropy"] for r in rows], dtype=float)
    fisher = np.asarray([r["fisher_ratio"] for r in rows], dtype=float)
    overlap = np.asarray([r["overlap_proxy"] for r in rows], dtype=float)
    margin = np.asarray([r["margin_ratio"] for r in rows], dtype=float)

    def norm(v):
        v = np.asarray(v, dtype=float)
        finite = np.isfinite(v)
        out = np.full_like(v, np.nan)
        if finite.sum() < 2:
            return out
        lo, hi = np.nanmin(v[finite]), np.nanmax(v[finite])
        out[finite] = (v[finite] - lo) / max(hi - lo, 1e-12)
        return out

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    axes[0].plot(snr, acc, marker="o", label="Accuracy")
    ax2 = axes[0].twinx()
    ax2.plot(snr, ent, marker="s", color="#D55E00", label="Entropy")
    axes[0].set_title("Waterfall and entropy")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Entropy")

    axes[1].plot(snr, norm(fisher), marker="o", label="Fisher")
    axes[1].plot(snr, norm(margin), marker="s", label="Margin")
    axes[1].set_title("Feature separability")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(frameon=False, fontsize=7)

    log_overlap = np.log10(np.maximum(overlap, 1e-12))
    axes[2].plot(snr, log_overlap, marker="o", label=r"$\log_{10}(2r/d_{\min})$")
    axes[2].axhline(0.0, color="0.35", lw=1.0, ls="--", label="unit overlap")
    axes[2].set_title("Feature overlap")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Log overlap")
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle(title, y=1.03, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"real_amc_geometry_transition.{ext}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=sorted(SPLIT_TO_DATALOADER), default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="Real AMC geometry transition")
    parser.add_argument("--max-per-class-snr", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    set_default_dataloader_cfg(cfg, SPLIT_TO_DATALOADER[args.split])
    dataloader_cfg = cfg[SPLIT_TO_DATALOADER[args.split]]
    dataloader_cfg["num_workers"] = 0
    dataloader_cfg["persistent_workers"] = False

    init_default_scope(cfg.get("default_scope", "csrr"))
    from csrr.registry import MODELS

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    hook = FeatureHook(model)
    dataloader = Runner.build_dataloader(dataloader_cfg)
    dataset = dataloader.dataset
    classes = list(dataset.CLASSES)

    all_features, all_logits, all_labels, all_snrs, all_idx = [], [], [], [], []
    sample_counter = 0
    print(json.dumps({
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples": len(dataset),
        "classes": classes,
        "feature_source": hook.source,
    }), flush=True)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            batch = model.data_preprocessor(data, False)
            inputs = batch["inputs"].to(device, non_blocking=True)
            data_samples = batch.get("data_samples", None)
            logits = model(inputs, data_samples, mode="tensor")
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            features = hook.pop(logits)
            all_features.append(to_numpy(features).reshape(features.size(0), -1))
            all_logits.append(to_numpy(logits))
            if data_samples is None:
                raise ValueError("Dataloader did not provide data_samples.")
            for sample in data_samples:
                all_labels.append(int(sample.gt_label.item()))
                packed_idx = sample.get("sample_idx")
                global_idx = sample.get("global_sample_idx")
                if packed_idx is not None:
                    local_idx = int(packed_idx)
                    info = dataset.get_data_info(local_idx)
                else:
                    local_idx = sample_counter
                    info = dataset.get_data_info(sample_counter)
                idx = int(global_idx) if global_idx is not None else int(info.get("global_sample_idx", local_idx))
                all_idx.append(idx)
                all_snrs.append(info.get("snr", sample.get("snr", 0)))
                sample_counter += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(dataloader)}]", flush=True)
    hook.close()

    features = np.concatenate(all_features, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.asarray(all_labels, dtype=np.int64)
    snrs = np.asarray(all_snrs)
    sample_idx = np.asarray(all_idx, dtype=np.int64)
    probs = softmax_np(logits)

    indices = np.arange(labels.size)
    keep = maybe_subsample(indices, labels, snrs, args.max_per_class_snr, args.seed)
    rows = summarize_by_snr(features[keep], probs[keep], labels[keep], snrs[keep], classes)
    write_csv(out_dir / "real_amc_geometry_by_snr.csv", rows)

    metric_specs = [
        ("fisher_ratio", True),
        ("margin_ratio", True),
        ("silhouette_proxy", True),
        ("overlap_proxy", False),
    ]
    critical_rows = [
        {"metric": "accuracy", "midpoint_snr": midpoint_snr(rows, "accuracy", True), "increasing": True},
        {"metric": "mean_entropy", "midpoint_snr": midpoint_snr(rows, "mean_entropy", False), "increasing": False},
    ]
    for metric, increasing in metric_specs:
        critical_rows.append({
            "metric": metric,
            "midpoint_snr": midpoint_snr(rows, metric, increasing),
            "increasing": increasing,
        })
    write_csv(out_dir / "real_amc_geometry_critical_points.csv", critical_rows)

    corr_rows = []
    for metric, _ in metric_specs:
        corr_rows.append({
            "metric": metric,
            "pearson_with_accuracy": pearson([r[metric] for r in rows], [r["accuracy"] for r in rows]),
            "pearson_with_entropy": pearson([r[metric] for r in rows], [r["mean_entropy"] for r in rows]),
        })
    write_csv(out_dir / "real_amc_geometry_correlations.csv", corr_rows)
    maybe_plot(out_dir, rows, args.title)

    summary = {
        "out_dir": str(out_dir),
        "config": args.config,
        "checkpoint": osp.abspath(args.checkpoint),
        "split": args.split,
        "n_total": int(labels.size),
        "n_used": int(keep.size),
        "feature_dim": int(features.shape[1]),
        "feature_source": hook.source,
        "classes": classes,
        "critical_points": critical_rows,
        "correlations": corr_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_npz:
        np.savez_compressed(
            out_dir / "real_amc_features_logits.npz",
            features=features[keep].astype(np.float32),
            logits=logits[keep].astype(np.float32),
            labels=labels[keep],
            snrs=snrs[keep],
            sample_idx=sample_idx[keep],
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
