# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Create derived floating-point STFT feature tensors from real/imag STFT data."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_features(tensor: np.ndarray, mode: str) -> tuple[np.ndarray, list[str], str]:
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError(f"Expected [2,F,T] real/imag tensor, got {tensor.shape}")
    real = tensor[0].astype(np.float32, copy=False)
    imag = tensor[1].astype(np.float32, copy=False)
    magnitude = np.sqrt(np.square(real) + np.square(imag)).astype(np.float32, copy=False)
    log_magnitude = np.log1p(magnitude).astype(np.float32, copy=False)
    if mode == "logpower2ch":
        return np.stack([log_magnitude, log_magnitude], axis=0), ["log_magnitude", "log_magnitude"], "diagnostic_logpower_duplicated_2ch"
    if mode == "realimag_logpower3ch":
        return np.stack([real, imag, log_magnitude], axis=0), ["real", "imag", "log_magnitude"], "realimag_plus_logpower_3ch"
    raise ValueError(f"Unsupported mode: {mode}")


def mode_metadata(mode: str) -> tuple[list[str], str]:
    if mode == "logpower2ch":
        return ["log_magnitude", "log_magnitude"], "diagnostic_logpower_duplicated_2ch"
    if mode == "realimag_logpower3ch":
        return ["real", "imag", "log_magnitude"], "realimag_plus_logpower_3ch"
    raise ValueError(f"Unsupported mode: {mode}")


def feature_stats(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = out.reshape(out.shape[0], -1).astype(np.float64)
    return flat.sum(axis=1), np.square(flat).sum(axis=1), flat.shape[1]


def convert_tensor(
    src_path: Path,
    dst_path: Path,
    mode: str,
    *,
    skip_existing: bool = False,
    skip_existing_fast_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, list[str], str]:
    if skip_existing and dst_path.exists():
        channels, representation = mode_metadata(mode)
        if skip_existing_fast_stats:
            zeros = np.zeros(len(channels), dtype=np.float64)
            return zeros, zeros.copy(), 0, channels, representation
        out = np.load(dst_path).astype(np.float32, copy=False)
        sums, sumsq, count = feature_stats(out)
        return sums, sumsq, count, channels, representation

    tensor = np.load(src_path).astype(np.float32, copy=False)
    out, channels, representation = build_features(tensor, mode)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, out)
    sums, sumsq, count = feature_stats(out)
    return sums, sumsq, count, channels, representation


def convert_tensor_task(task: tuple[str, str, str, bool, bool]) -> tuple[np.ndarray, np.ndarray, int, list[str], str]:
    src_path, dst_path, mode, skip_existing, skip_existing_fast_stats = task
    return convert_tensor(
        Path(src_path),
        Path(dst_path),
        mode,
        skip_existing=skip_existing,
        skip_existing_fast_stats=skip_existing_fast_stats,
    )


def convert_split(
    src_root: Path,
    out_root: Path,
    split: str,
    mode: str,
    *,
    workers: int,
    skip_existing: bool,
    skip_existing_fast_stats: bool,
) -> tuple[int, int, np.ndarray, np.ndarray, int, list[str], str]:
    ann = load_json(src_root / "annotations" / f"instances_{split}.json")
    write_json(out_root / "annotations" / f"instances_{split}.json", ann)
    channel_sum: np.ndarray | None = None
    channel_sumsq: np.ndarray | None = None
    channel_count = 0
    channels: list[str] | None = None
    representation = ""
    src_dir = src_root / split / "tensors"
    dst_dir = out_root / split / "tensors"
    images = ann.get("images", [])
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(src_dir / image["file_name"]), str(dst_dir / image["file_name"]), mode, skip_existing, skip_existing_fast_stats)
        for image in images
    ]
    if workers > 1:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            iterator = pool.imap(convert_tensor_task, tasks, chunksize=16)
            for index, (sums, sumsq, count, channels, representation) in enumerate(iterator, start=1):
                if channel_sum is None:
                    channel_sum = np.zeros_like(sums)
                    channel_sumsq = np.zeros_like(sumsq)
                channel_sum += sums
                channel_sumsq += sumsq
                channel_count += count
                if index % 500 == 0 or index == len(images):
                    print(f"[make-stft-features] {split}: converted {index}/{len(images)}", flush=True)
    else:
        for index, image in enumerate(images, start=1):
            sums, sumsq, count, channels, representation = convert_tensor(
                src_dir / image["file_name"],
                dst_dir / image["file_name"],
                mode,
                skip_existing=skip_existing,
                skip_existing_fast_stats=skip_existing_fast_stats,
            )
            if channel_sum is None:
                channel_sum = np.zeros_like(sums)
                channel_sumsq = np.zeros_like(sumsq)
            channel_sum += sums
            channel_sumsq += sumsq
            channel_count += count
            if index % 500 == 0 or index == len(images):
                print(f"[make-stft-features] {split}: converted {index}/{len(images)}", flush=True)
    if not tasks:
        raise ValueError(f"No images found for split {split}")
    if channel_sum is None or channel_sumsq is None or channels is None:
        raise ValueError(f"No images found for split {split}")
    return len(images), len(ann.get("annotations", [])), channel_sum, channel_sumsq, channel_count, channels, representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, help="COCO root with [2,F,T] real/imag STFT tensors.")
    parser.add_argument("--out-root", required=True, help="Output dataset root; may be either parent or COCO root.")
    parser.add_argument("--mode", choices=["logpower2ch", "realimag_logpower3ch"], default="realimag_logpower3ch")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names, or auto.")
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU worker processes for tensor conversion.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing output tensors and only convert missing files while still including existing tensors in statistics.",
    )
    parser.add_argument(
        "--skip-existing-fast-stats",
        action="store_true",
        help="When --skip-existing is set, do not reload existing outputs for statistics; compute stats from newly converted tensors only.",
    )
    args = parser.parse_args()

    src_root = ROOT / args.src_root
    out_arg = ROOT / args.out_root
    out_root = out_arg / "coco" if out_arg.name != "coco" else out_arg
    out_parent = out_root.parent
    if out_parent.exists() and args.force:
        shutil.rmtree(out_parent)
    out_root.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    train_sum: np.ndarray | None = None
    train_sumsq: np.ndarray | None = None
    train_count = 0
    train_channels: list[str] | None = None
    representation = ""
    if args.splits == "auto":
        splits = sorted(path.stem.removeprefix("instances_") for path in (src_root / "annotations").glob("instances_*.json"))
    else:
        splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    workers = max(int(args.workers), 1)
    for split in splits:
        images, boxes, sums, sumsq, count, channels, representation = convert_split(
            src_root,
            out_root,
            split,
            args.mode,
            workers=workers,
            skip_existing=args.skip_existing,
            skip_existing_fast_stats=args.skip_existing_fast_stats,
        )
        src_meta = src_root.parent / "metadata" / f"{split}.jsonl"
        if src_meta.exists():
            dst_meta = out_parent / "metadata" / f"{split}.jsonl"
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_meta, dst_meta)
        split_stats[split] = {"images": images, "boxes": boxes, "channel_count": count}
        if split == "train" or split.endswith("_train_all") or "_train_" in split:
            if train_sum is None:
                train_sum = np.zeros_like(sums)
                train_sumsq = np.zeros_like(sumsq)
                train_channels = channels
            train_sum += sums
            train_sumsq += sumsq
            train_count += count

    if train_channels is None:
        train_channels, representation = mode_metadata(args.mode)
    if train_sum is None or train_sumsq is None or train_count <= 0:
        mean = np.zeros(len(train_channels), dtype=np.float64)
        std = np.ones(len(train_channels), dtype=np.float64)
    else:
        mean = train_sum / max(train_count, 1)
        var = train_sumsq / max(train_count, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-12))
    src_summary = src_root.parent / "summary.json"
    summary = load_json(src_summary) if src_summary.exists() else {}
    summary.update(
        {
            "source_root": str(src_root),
            "out_root": str(out_parent),
            "stft_representation": representation,
            "tensor_layout": "C,F,T",
            "tensor_channels": train_channels,
            "splits": split_stats,
            "stft_tensor_stats": {
                "mean": [float(v) for v in mean],
                "std": [float(max(v, 1e-6)) for v in std],
                "computed_on": [split for split in splits if split == "train" or split.endswith("_train_all") or "_train_" in split],
            },
        }
    )
    write_json(out_parent / "summary.json", summary)
    print(f"[make-stft-features] train mean={mean.tolist()} std={std.tolist()}", flush=True)
    print(f"[make-stft-features] wrote {out_parent}", flush=True)


if __name__ == "__main__":
    main()
