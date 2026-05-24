#!/usr/bin/env python3
"""Run Speech Commands phi-RCPS with teacher-derived reliability.

This is a minimal experimental runner for the audio boundary case.  It keeps the
same log-mel models and splits as ``run_crossmodal_audio.py`` but replaces scalar
SNR reliability with a sample-level teacher confidence projection.  The runner is
intended for gated experiments only; results should not enter the manuscript
unless validation NLL/Brier and high-confidence retention pass the evidence gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_crossmodal_audio import (  # noqa: E402
    DEFAULT_SNR_ORDER,
    LogMelFeature,
    SpeechCommandsReliability,
    append_metrics,
    build_model,
    evaluate_loader,
    set_seed,
    soft_cross_entropy,
    write_metric_rows,
    write_prediction_pkl,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return (idx, *self.dataset[idx])


def build_indexed_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        persistent_workers=args.workers > 0,
    )


def load_teacher(path: Path, expected_len: int, num_classes: int) -> Dict[str, torch.Tensor]:
    payload = np.load(path)
    required = {"sample_idx", "probs", "labels", "confidence"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"Teacher posterior file is missing keys: {sorted(missing)}")
    sample_idx = torch.as_tensor(payload["sample_idx"], dtype=torch.long)
    probs = torch.as_tensor(payload["probs"], dtype=torch.float32)
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    confidence = torch.as_tensor(payload["confidence"], dtype=torch.float32).clamp(0.0, 1.0)
    if sample_idx.numel() != expected_len:
        raise ValueError(f"Teacher length {sample_idx.numel()} != dataset length {expected_len}")
    if probs.shape != (expected_len, num_classes):
        raise ValueError(f"Teacher probs shape {tuple(probs.shape)} != ({expected_len}, {num_classes})")
    if not torch.equal(sample_idx, torch.arange(expected_len, dtype=torch.long)):
        raise ValueError("Teacher sample_idx must be contiguous and aligned with the deterministic dataset order.")
    return {"probs": probs, "labels": labels, "confidence": confidence}


def phi_targets(method: str, labels: torch.Tensor, idx: torch.Tensor, teacher: Dict[str, torch.Tensor],
                num_classes: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    teacher_probs = teacher["probs"][idx.cpu()].to(device)
    teacher_labels = teacher["labels"][idx.cpu()].to(device)
    phi = teacher["confidence"][idx.cpu()].to(device)
    if not torch.equal(teacher_labels, labels.long()):
        bad = idx[teacher_labels.detach().cpu() != labels.detach().cpu()][:8].tolist()
        raise ValueError(f"Teacher labels do not match dataset labels for sample_idx: {bad}")

    eps = float(args.epsilon_max) * torch.pow(1.0 - phi, float(args.gamma))
    eps = torch.where(phi >= float(args.retain_min), torch.zeros_like(eps), eps).reshape(-1, 1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
    if method == "phi-uniform":
        base = torch.full_like(one_hot, 1.0 / num_classes)
    elif method == "phi-teacher":
        base = teacher_probs
    else:
        raise ValueError(f"Unsupported phi method: {method}")
    target = (1.0 - eps) * one_hot + eps * base
    return target.clamp_min(1e-12) / target.sum(dim=1, keepdim=True).clamp_min(1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["phi-uniform", "phi-teacher"], required=True)
    parser.add_argument("--teacher-posterior-source", type=Path, required=True)
    parser.add_argument("--model", choices=["ds-cnn", "logmel-resnet"], default="logmel-resnet")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epsilon-max", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--retain-min", type=float, default=0.85)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--train-max-per-label-snr", type=int, default=600)
    parser.add_argument("--val-max-per-label-snr", type=int, default=200)
    parser.add_argument("--test-max-per-label-snr", type=int, default=0)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/home/citybuster/Data/RCPS/processed/ReliabilityClassification/Audio/SpeechCommands-v0.02"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_phi"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SpeechCommandsReliability(
        args.processed_root / "train.json", "train",
        args.train_max_per_label_snr, args.seed, train=True)
    val_ds = SpeechCommandsReliability(
        args.processed_root / "validation.json", "validation",
        args.val_max_per_label_snr, args.seed + 17, train=False)
    test_ds = SpeechCommandsReliability(
        args.processed_root / "test.json", "test",
        args.test_max_per_label_snr, args.seed + 31, train=False)
    num_classes = len(train_ds.classes)
    teacher = load_teacher(args.teacher_posterior_source, len(train_ds), num_classes)

    train_loader = build_indexed_loader(train_ds, args, shuffle=True, seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                             persistent_workers=args.workers > 0)

    work_dir = args.work_root / "audio" / "speechcommands" / args.model / args.method / f"seed_{args.seed}"
    metrics_dir = args.work_root / "metrics"
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")
    (work_dir / "classes.json").write_text(json.dumps(train_ds.classes, indent=2), encoding="utf-8")

    feature_extractor = LogMelFeature().to(device)
    model = build_model(args.model, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score = -1.0
    best_path = work_dir / "best.pt"
    history_rows: List[Dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx, wav, y, _, _ in train_loader:
            idx = idx.long()
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_extractor(wav))
            targets = phi_targets(args.method, y, idx, teacher, num_classes, args, device)
            loss = soft_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_probs, val_labels, _, _ = evaluate_loader(model, feature_extractor, val_loader, device)
        from run_crossmodal_audio import metrics_for  # local import avoids copying implementation
        val_metrics = metrics_for(val_probs, val_labels, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_accuracy": val_metrics["macro_accuracy"],
            "val_nll": val_metrics["nll"],
            "val_ece": val_metrics["ece"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - start,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_accuracy"] > best_score:
            best_score = val_metrics["macro_accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, best_path)
    write_metric_rows(work_dir / "history.csv", history_rows)

    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload["model"])
    val_probs, val_labels, val_reliabilities, val_snrs = evaluate_loader(model, feature_extractor, val_loader, device)
    probs, labels, reliabilities, snrs = evaluate_loader(model, feature_extractor, test_loader, device)
    if args.save_predictions:
        write_prediction_pkl(work_dir / "predictions" / "validation.pkl", val_probs, val_labels, val_reliabilities, val_snrs)
        write_prediction_pkl(work_dir / "predictions" / "test.pkl", probs, labels, reliabilities, snrs)
    metric_rows: List[Dict] = []
    append_metrics(metric_rows, "speechcommands-noisy", args.model, args.method, args.seed, "test", "all", "all", probs, labels, num_classes)
    for snr in DEFAULT_SNR_ORDER:
        mask = snrs == snr
        append_metrics(metric_rows, "speechcommands-noisy", args.model, args.method, args.seed, "test", "snr", snr, probs[mask], labels[mask], num_classes)
    out_csv = metrics_dir / f"speechcommands_{args.model}_{args.method}_seed{args.seed}_test.csv"
    write_metric_rows(out_csv, metric_rows)
    print(f"Wrote metrics: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
