#!/usr/bin/env python3
"""Standalone CIFAR-10-C reliability experiment runner for RCPS.

This script intentionally stays outside the AMC-specific MMEngine pipeline.  It
uses the same RCPS target builder from ``csrr.models.losses.rcps_loss`` but a
small torchvision ResNet18 baseline so that cross-modal checks remain simple,
traceable, and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset

from csrr.models.losses.rcps_loss import build_rcps_targets

CLASS_NAMES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)
DEFAULT_CORRUPTIONS = ('gaussian_noise', 'motion_blur', 'brightness')
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def list_image_folder(root: Path) -> List[Tuple[Path, int]]:
    rows: List[Tuple[Path, int]] = []
    for label, name in enumerate(CLASS_NAMES):
        class_dir = root / name
        if not class_dir.exists():
            raise FileNotFoundError(class_dir)
        for path in sorted(class_dir.glob('*.jpg')):
            rows.append((path, label))
    if not rows:
        raise RuntimeError(f'No jpg images found under {root}')
    return rows


def stratified_split(rows: Sequence[Tuple[Path, int]], val_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {i: [] for i in range(len(CLASS_NAMES))}
    for idx, (_, label) in enumerate(rows):
        by_class[int(label)].append(idx)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label, indices in by_class.items():
        indices = np.asarray(indices)
        rng.shuffle(indices)
        val_idx.extend(indices[:val_per_class].tolist())
        train_idx.extend(indices[val_per_class:].tolist())
    return sorted(train_idx), sorted(val_idx)


def reliability_from_severity(severity: int | str) -> float:
    if severity == 'clean' or int(severity) <= 0:
        return 1.0
    severity = int(severity)
    return float((5 - severity) / 4.0)


def corrupt_image(img: Image.Image, corruption: str, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0 or corruption == 'clean':
        return img
    if corruption == 'brightness':
        factors = {1: 0.90, 2: 0.75, 3: 0.60, 4: 0.45, 5: 0.30}
        return ImageEnhance.Brightness(img).enhance(factors[int(severity)])
    if corruption == 'motion_blur':
        sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
        k = sizes[int(severity)]
        pad = k // 2
        arr = np.asarray(img).astype(np.float32)
        if rng.random() < 0.5:
            padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode='edge')
            out = sum(padded[:, offset:offset + arr.shape[1], :] for offset in range(k)) / k
        else:
            padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode='edge')
            out = sum(padded[offset:offset + arr.shape[0], :, :] for offset in range(k)) / k
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if corruption == 'gaussian_noise':
        sigmas = {1: 7.0, 2: 14.0, 3: 21.0, 4: 32.0, 5: 45.0}
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0.0, sigmas[int(severity)], size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f'Unsupported corruption: {corruption}')


def train_transform(img: Image.Image, rng: random.Random) -> Image.Image:
    img = TF.pad(img, 4, padding_mode='reflect')
    i, j, h, w = torch.randint(0, 9, (1,)).item(), torch.randint(0, 9, (1,)).item(), 32, 32
    img = TF.crop(img, i, j, h, w)
    if rng.random() < 0.5:
        img = TF.hflip(img)
    return img


def normalize_tensor(img: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(img)
    return TF.normalize(x, MEAN, STD)


class ReliabilityCIFARTrain(Dataset):
    def __init__(self, rows: Sequence[Tuple[Path, int]], corruptions: Sequence[str], seed: int, train: bool = True):
        self.rows = list(rows)
        self.corruptions = tuple(corruptions)
        self.seed = int(seed)
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label = self.rows[idx]
        rng = random if self.train else random.Random(self.seed + idx)
        img = Image.open(path).convert('RGB')
        if self.train:
            img = train_transform(img, rng)
            severity = rng.choice([0, 1, 2, 3, 4, 5])
            corruption = 'clean' if severity == 0 else rng.choice(self.corruptions)
            img = corrupt_image(img, corruption, severity, rng)
        else:
            severity = 0
            corruption = 'clean'
        reliability = reliability_from_severity(severity)
        return normalize_tensor(img), int(label), float(reliability), str(corruption), int(severity)


class CIFAR10CTest(Dataset):
    def __init__(self, raw_dir: Path, corruptions: Sequence[str], severity: int):
        self.raw_dir = raw_dir
        self.corruptions = tuple(corruptions)
        self.severity = int(severity)
        self.labels = np.load(raw_dir / 'labels.npy').astype(np.int64)
        self.arrays = {name: np.load(raw_dir / f'{name}.npy', mmap_mode='r') for name in self.corruptions}
        self.length_per_corruption = 10000
        self.offset = (self.severity - 1) * 10000

    def __len__(self) -> int:
        return len(self.corruptions) * self.length_per_corruption

    def __getitem__(self, idx: int):
        cidx = idx // self.length_per_corruption
        local = idx % self.length_per_corruption
        corruption = self.corruptions[cidx]
        arr = self.arrays[corruption][self.offset + local]
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert('RGB')
        label = int(self.labels[local])
        reliability = reliability_from_severity(self.severity)
        return normalize_tensor(img), label, float(reliability), corruption, self.severity


class CleanImageFolderEval(Dataset):
    def __init__(self, root: Path):
        self.rows = list_image_folder(root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label = self.rows[idx]
        img = Image.open(path).convert('RGB')
        return normalize_tensor(img), int(label), 1.0, 'clean', 0


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(num_classes: int = 10) -> nn.Module:
    model = tv_models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def compute_loss(method: str, logits: torch.Tensor, labels: torch.Tensor, reliability: torch.Tensor, args) -> torch.Tensor:
    if method == 'hard-ce':
        return F.cross_entropy(logits, labels)
    if method == 'static-ls':
        return F.cross_entropy(logits, labels, label_smoothing=args.smoothing)
    if method == 'rcps-retention':
        targets = build_rcps_targets(
            labels,
            reliability,
            num_classes=10,
            reliability_map=dict(type='identity'),
            epsilon=dict(type='retention_power', max=args.epsilon_max, gamma=args.gamma, retain_min=args.retain_min),
            base=dict(type='uniform'))
        return soft_cross_entropy(logits, targets)
    raise ValueError(f'Unsupported method: {method}')


@dataclass
class EvalAccumulator:
    probs: List[np.ndarray]
    labels: List[np.ndarray]
    reliabilities: List[np.ndarray]
    corruptions: List[str]
    severities: List[int]

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [])

    def add(self, prob, label, reliability, corruption, severity):
        self.probs.append(prob.detach().cpu().numpy())
        self.labels.append(label.detach().cpu().numpy())
        self.reliabilities.append(reliability.detach().cpu().numpy())
        self.corruptions.extend(list(corruption))
        self.severities.extend([int(s) for s in severity])

    def arrays(self):
        return (np.concatenate(self.probs, axis=0),
                np.concatenate(self.labels, axis=0),
                np.concatenate(self.reliabilities, axis=0),
                np.asarray(self.corruptions),
                np.asarray(self.severities))


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-12)
        if not mask.any():
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def metrics_for(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    onehot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    entropy = -(probs * np.log(probs)).sum(axis=1)
    return {
        'accuracy': float((pred == labels).mean() * 100.0),
        'nll': float(-np.log(probs[np.arange(labels.size), labels]).mean()),
        'ece': ece_score(probs, labels),
        'brier': float(((probs - onehot) ** 2).sum(axis=1).mean()),
        'mean_confidence': float(probs.max(axis=1).mean()),
        'mean_entropy': float(entropy.mean()),
    }


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    acc = EvalAccumulator.empty()
    with torch.no_grad():
        for x, y, r, c, s in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            acc.add(probs, y, r, c, s)
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
                   split: str, group: str, reliability_bin: str, probs: np.ndarray, labels: np.ndarray):
    m = metrics_for(probs, labels)
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


def evaluate_cifar10c(model, args, device, batch_size, workers):
    raw_dir = Path(args.cifar10c_raw)
    rows: List[Dict] = []
    all_probs = []
    all_labels = []
    all_corruptions = []
    all_severities = []
    for severity in range(1, 6):
        ds = CIFAR10CTest(raw_dir, args.corruptions, severity)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        probs, labels, _, corruptions, severities = evaluate_loader(model, loader, device)
        append_metrics(rows, 'cifar10c', 'resnet18-cifar', args.method, args.seed,
                       'test', 'severity', str(severity), probs, labels)
        for corruption in args.corruptions:
            mask = corruptions == corruption
            append_metrics(rows, 'cifar10c', 'resnet18-cifar', args.method, args.seed,
                           'test', corruption, str(severity), probs[mask], labels[mask])
        all_probs.append(probs)
        all_labels.append(labels)
        all_corruptions.append(corruptions)
        all_severities.append(severities)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    append_metrics(rows, 'cifar10c', 'resnet18-cifar', args.method, args.seed,
                   'test', 'all', 'all', probs, labels)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Run CIFAR-10-C RCPS cross-modal experiments.')
    parser.add_argument('--method', choices=['hard-ce', 'static-ls', 'rcps-retention'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--epsilon-max', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--retain-min', type=float, default=0.8)
    parser.add_argument('--train-root', default='/home/citybuster/Data/Visual/CIFAR-10/train')
    parser.add_argument('--test-root', default='/home/citybuster/Data/Visual/CIFAR-10/test')
    parser.add_argument('--cifar10c-raw', default='/home/citybuster/Data/RCPS/raw/CIFAR-10-C/CIFAR-10-C')
    parser.add_argument('--work-root', default='/home/citybuster/Data/RCPS/work_dirs/crossmodal_vision_cifar10c_30ep')
    parser.add_argument('--corruptions', nargs='+', default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument('--max-train-samples', type=int, default=0)
    parser.add_argument('--max-val-samples', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    work_dir = Path(args.work_root) / 'vision' / 'cifar10c' / 'resnet18-cifar' / args.method / f'seed_{args.seed}'
    metrics_dir = Path(args.work_root) / 'metrics'
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')

    rows = list_image_folder(Path(args.train_root))
    train_idx, val_idx = stratified_split(rows, val_per_class=500, seed=args.seed)
    if args.max_train_samples:
        train_idx = train_idx[:args.max_train_samples]
    if args.max_val_samples:
        val_idx = val_idx[:args.max_val_samples]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]

    train_ds = ReliabilityCIFARTrain(train_rows, args.corruptions, seed=args.seed, train=True)
    val_ds = ReliabilityCIFARTrain(val_rows, args.corruptions, seed=args.seed, train=False)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = work_dir / 'best.pt'
    history_rows = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, r, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(args.method, logits, y, r, args)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_probs, val_labels, _, _, _ = evaluate_loader(model, val_loader, device)
        val_metrics = metrics_for(val_probs, val_labels)
        elapsed = time.time() - start
        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'val_clean_accuracy': val_metrics['accuracy'],
            'val_clean_nll': val_metrics['nll'],
            'val_clean_ece': val_metrics['ece'],
            'lr': scheduler.get_last_lr()[0],
            'elapsed_sec': elapsed,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics, 'args': vars(args)}, best_path)

    write_metric_rows(work_dir / 'history.csv', history_rows)
    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload['model'])

    clean_loader = DataLoader(CleanImageFolderEval(Path(args.test_root)), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    clean_probs, clean_labels, _, _, _ = evaluate_loader(model, clean_loader, device)
    metric_rows: List[Dict] = []
    append_metrics(metric_rows, 'cifar10c', 'resnet18-cifar', args.method, args.seed,
                   'test', 'clean', 'clean', clean_probs, clean_labels)
    metric_rows.extend(evaluate_cifar10c(model, args, device, args.batch_size, args.workers))
    out_csv = metrics_dir / f'cifar10c_resnet18-cifar_{args.method}_seed{args.seed}_test.csv'
    write_metric_rows(out_csv, metric_rows)
    print(f'Wrote metrics: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
