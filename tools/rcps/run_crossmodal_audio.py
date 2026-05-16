#!/usr/bin/env python3
"""Standalone Speech Commands reliability experiment runner for RCPS.

The runner mirrors ``run_crossmodal_vision.py`` but uses log-mel audio
features and additive background noise.  It is deliberately task-local so that
we can validate the degraded-observation claim outside AMC without coupling the
experiment to radio-specific data structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from csrr.models.losses.rcps_loss import build_rcps_targets

SAMPLE_RATE = 16000
TARGET_SECONDS = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)
DEFAULT_SNR_ORDER = ('-10', '-5', '0', '5', '10', '20', 'clean')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@lru_cache(maxsize=8192)
def cached_load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.flatten().float()
    return wav


def fit_length(wav: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
    if wav.numel() == TARGET_SAMPLES:
        return wav
    if wav.numel() < TARGET_SAMPLES:
        return F.pad(wav, (0, TARGET_SAMPLES - wav.numel()))
    if rng is None:
        start = (wav.numel() - TARGET_SAMPLES) // 2
    else:
        start = rng.randint(0, wav.numel() - TARGET_SAMPLES)
    return wav[start:start + TARGET_SAMPLES]


def tile_to_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.numel() >= length:
        return wav
    reps = int(math.ceil(length / max(1, wav.numel())))
    return wav.repeat(reps)


def add_noise_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float,
                     rng: random.Random) -> torch.Tensor:
    signal = fit_length(signal, rng)
    noise = tile_to_length(noise, TARGET_SAMPLES)
    if noise.numel() > TARGET_SAMPLES:
        start = rng.randint(0, noise.numel() - TARGET_SAMPLES)
        noise = noise[start:start + TARGET_SAMPLES]
    noise = noise - noise.mean()
    signal_power = signal.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scale = torch.sqrt(target_noise_power / noise_power)
    mixed = signal + scale * noise
    return mixed.clamp(-1.0, 1.0)


def load_annotations(path: Path) -> Tuple[List[Dict], Dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload['data_list']
    meta = payload['metainfo']
    return rows, meta


def balanced_subset(rows: Sequence[Dict], max_per_label_snr: int, seed: int) -> List[Dict]:
    if max_per_label_snr <= 0:
        return list(rows)
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        buckets[(row['label_name'], str(row['snr']))].append(row)
    selected: List[Dict] = []
    for key in sorted(buckets):
        values = buckets[key]
        rng.shuffle(values)
        selected.extend(values[:max_per_label_snr])
    rng.shuffle(selected)
    return selected


class SpeechCommandsReliability(Dataset):
    def __init__(self, ann_path: Path, split: str, max_per_label_snr: int,
                 seed: int, train: bool = True):
        rows, meta = load_annotations(ann_path)
        self.rows = balanced_subset(rows, max_per_label_snr, seed)
        self.meta = meta
        self.classes = list(meta['classes'])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.raw_root = Path(meta['raw_root'])
        self.background_paths = sorted((self.raw_root / '_background_noise_').glob('*.wav'))
        if not self.background_paths:
            raise FileNotFoundError(self.raw_root / '_background_noise_')
        self.split = split
        self.seed = int(seed)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rng = random if self.train else random.Random(self.seed * 1000003 + idx)
        wav = cached_load_wav(str(self.raw_root / row['file_name']))
        snr = row['snr']
        if str(snr) == 'clean':
            mixed = fit_length(wav, rng if self.train else None)
            snr_label = 'clean'
        else:
            bg_path = rng.choice(self.background_paths)
            noise = cached_load_wav(str(bg_path))
            mixed = add_noise_at_snr(wav, noise, float(snr), rng)
            snr_label = str(snr)
        label = self.class_to_idx[row['label_name']]
        reliability = float(row['reliability'])
        return mixed.reshape(1, -1), int(label), reliability, snr_label


class LogMelFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40,
            f_min=20.0,
            f_max=7600.0,
            power=2.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feat = self.mel(wav).clamp_min(1e-8).log()
        dims = tuple(range(1, feat.ndim))
        mean = feat.mean(dim=dims, keepdim=True)
        std = feat.std(dim=dims, keepdim=True).clamp_min(1e-5)
        return (feat - mean) / std


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(64),
            DepthwiseSeparableBlock(64, stride=(2, 2)),
            DepthwiseSeparableBlock(64),
            DepthwiseSeparableBlock(64, stride=(2, 2)),
            DepthwiseSeparableBlock(64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


class LogMelResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(48, 64, stride=(1, 1))
        self.stage2 = self._stage(64, 96, stride=(2, 2))
        self.stage3 = self._stage(96, 128, stride=(2, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    @staticmethod
    def _stage(cin: int, cout: int, stride: Tuple[int, int]) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(self.pool(x).flatten(1))


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == 'ds-cnn':
        return DSCNN(num_classes)
    if name == 'logmel-resnet':
        return LogMelResNet(num_classes)
    raise ValueError(f'Unsupported model: {name}')


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def compute_loss(method: str, logits: torch.Tensor, labels: torch.Tensor,
                 reliability: torch.Tensor, num_classes: int, args) -> torch.Tensor:
    if method == 'hard-ce':
        return F.cross_entropy(logits, labels)
    if method == 'static-ls':
        return F.cross_entropy(logits, labels, label_smoothing=args.smoothing)
    if method == 'rcps-retention':
        targets = build_rcps_targets(
            labels,
            reliability,
            num_classes=num_classes,
            reliability_map=dict(type='identity'),
            epsilon=dict(type='retention_power', max=args.epsilon_max,
                         gamma=args.gamma, retain_min=args.retain_min),
            base=dict(type='uniform'))
        return soft_cross_entropy(logits, targets)
    raise ValueError(f'Unsupported method: {method}')


@dataclass
class EvalAccumulator:
    probs: List[np.ndarray]
    labels: List[np.ndarray]
    reliabilities: List[np.ndarray]
    snrs: List[str]

    @classmethod
    def empty(cls):
        return cls([], [], [], [])

    def add(self, prob, label, reliability, snr):
        self.probs.append(prob.detach().cpu().numpy())
        self.labels.append(label.detach().cpu().numpy())
        self.reliabilities.append(reliability.detach().cpu().numpy())
        self.snrs.extend([str(x) for x in snr])

    def arrays(self):
        return (np.concatenate(self.probs, axis=0),
                np.concatenate(self.labels, axis=0),
                np.concatenate(self.reliabilities, axis=0),
                np.asarray(self.snrs))


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-12)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def metrics_for(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    onehot = np.eye(num_classes, dtype=np.float64)[labels]
    entropy = -(probs * np.log(probs)).sum(axis=1)
    per_class = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            per_class.append(float((pred[mask] == labels[mask]).mean() * 100.0))
    return {
        'accuracy': float((pred == labels).mean() * 100.0),
        'macro_accuracy': float(np.mean(per_class)) if per_class else float('nan'),
        'nll': float(-np.log(probs[np.arange(labels.size), labels]).mean()),
        'ece': ece_score(probs, labels),
        'brier': float(((probs - onehot) ** 2).sum(axis=1).mean()),
        'mean_confidence': float(probs.max(axis=1).mean()),
        'mean_entropy': float(entropy.mean()),
    }


def evaluate_loader(model: nn.Module, feature_extractor: nn.Module,
                    loader: DataLoader, device: torch.device):
    model.eval()
    feature_extractor.eval()
    acc = EvalAccumulator.empty()
    with torch.no_grad():
        for wav, y, r, snr in loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = feature_extractor(wav)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            acc.add(probs, y, r, snr)
    return acc.arrays()


def write_metric_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError('No metric rows to write')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(rows: List[Dict], dataset: str, model_name: str, method: str, seed: int,
                   split: str, group: str, reliability_bin: str,
                   probs: np.ndarray, labels: np.ndarray, num_classes: int):
    if labels.size == 0:
        return
    m = metrics_for(probs, labels, num_classes)
    rows.append({
        'dataset': dataset,
        'model': model_name,
        'method': method,
        'seed': seed,
        'split': split,
        'group': group,
        'reliability_bin': reliability_bin,
        **m,
    })


def build_loader(dataset: Dataset, args, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        generator=gen if shuffle else None,
        persistent_workers=args.workers > 0)


def main():
    parser = argparse.ArgumentParser(description='Run Speech Commands RCPS cross-modal experiments.')
    parser.add_argument('--method', choices=['hard-ce', 'static-ls', 'rcps-retention'], required=True)
    parser.add_argument('--model', choices=['ds-cnn', 'logmel-resnet'], default='ds-cnn')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--epsilon-max', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--retain-min', type=float, default=0.85)
    parser.add_argument('--train-max-per-label-snr', type=int, default=600)
    parser.add_argument('--val-max-per-label-snr', type=int, default=200)
    parser.add_argument('--test-max-per-label-snr', type=int, default=0)
    parser.add_argument('--processed-root', default='/home/citybuster/Data/RCPS/processed/ReliabilityClassification/Audio/SpeechCommands-v0.02')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs/crossmodal_audio_speechcommands_20ep')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_root = Path(args.processed_root)
    train_ds = SpeechCommandsReliability(processed_root / 'train.json', 'train',
                                         args.train_max_per_label_snr, args.seed, train=True)
    val_ds = SpeechCommandsReliability(processed_root / 'validation.json', 'validation',
                                       args.val_max_per_label_snr, args.seed + 17, train=False)
    test_ds = SpeechCommandsReliability(processed_root / 'test.json', 'test',
                                        args.test_max_per_label_snr, args.seed + 31, train=False)
    num_classes = len(train_ds.classes)
    work_dir = Path(args.work_root) / 'audio' / 'speechcommands' / args.model / args.method / f'seed_{args.seed}'
    metrics_dir = Path(args.work_root) / 'metrics'
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    (work_dir / 'classes.json').write_text(json.dumps(train_ds.classes, indent=2), encoding='utf-8')

    train_loader = build_loader(train_ds, args, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_ds, args, shuffle=False, seed=args.seed)
    test_loader = build_loader(test_ds, args, shuffle=False, seed=args.seed)
    print(json.dumps({
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'num_classes': num_classes,
        'classes': train_ds.classes,
    }), flush=True)

    feature_extractor = LogMelFeature().to(device)
    model = build_model(args.model, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score = -1.0
    best_path = work_dir / 'best.pt'
    history_rows: List[Dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for wav, y, r, _ in train_loader:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            x = feature_extractor(wav)
            logits = model(x)
            loss = compute_loss(args.method, logits, y, r, num_classes, args)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_probs, val_labels, _, _ = evaluate_loader(model, feature_extractor, val_loader, device)
        val_metrics = metrics_for(val_probs, val_labels, num_classes)
        elapsed = time.time() - start
        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_accuracy': val_metrics['macro_accuracy'],
            'val_nll': val_metrics['nll'],
            'val_ece': val_metrics['ece'],
            'lr': scheduler.get_last_lr()[0],
            'elapsed_sec': elapsed,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics['macro_accuracy'] > best_score:
            best_score = val_metrics['macro_accuracy']
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_metrics': val_metrics, 'args': vars(args)}, best_path)
    write_metric_rows(work_dir / 'history.csv', history_rows)

    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload['model'])
    probs, labels, reliabilities, snrs = evaluate_loader(model, feature_extractor, test_loader, device)
    metric_rows: List[Dict] = []
    append_metrics(metric_rows, 'speechcommands-noisy', args.model, args.method, args.seed,
                   'test', 'all', 'all', probs, labels, num_classes)
    for snr in DEFAULT_SNR_ORDER:
        mask = snrs == snr
        append_metrics(metric_rows, 'speechcommands-noisy', args.model, args.method, args.seed,
                       'test', 'snr', snr, probs[mask], labels[mask], num_classes)
    out_csv = metrics_dir / f'speechcommands_{args.model}_{args.method}_seed{args.seed}_test.csv'
    write_metric_rows(out_csv, metric_rows)
    print(f'Wrote metrics: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
