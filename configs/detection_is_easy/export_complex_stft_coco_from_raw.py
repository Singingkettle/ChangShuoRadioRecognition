# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def repo_root() -> Path:
    _p = Path(__file__).resolve()
    for _up in [_p, *_p.parents]:
        if (_up / "tools" / "train.py").exists() and (_up / "csrr").is_dir():
            return _up
    raise RuntimeError("CSRR repo root not found above " + str(_p))


ROOT = repo_root()
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from prepare_torchsig_coco import instance_frequency_bounds, render_preview  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(src_root: Path, split: str) -> list[dict[str, Any]]:
    path = src_root / "metadata" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rewrite_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / path


def load_iq_for_row(raw_path: Path, raw_cache_root: Path | None) -> np.ndarray:
    if raw_cache_root is not None:
        cache_path = raw_cache_root / raw_path.parent.name / f"{raw_path.stem}.npy"
        if cache_path.exists():
            return np.asarray(np.load(cache_path), dtype=np.complex64)
    with np.load(raw_path) as npz:
        return np.asarray(npz["iq"], dtype=np.complex64)


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalized_class_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_")


def category_id_for_instance(
    instance: dict[str, Any],
    *,
    valid_category_ids: set[int],
    category_name_to_id: dict[str, int],
) -> int | None:
    for key in ("category_id", "class_index"):
        value = instance.get(key)
        if value is None:
            continue
        try:
            category_id = int(value)
        except (TypeError, ValueError):
            continue
        if category_id in valid_category_ids:
            return category_id
    name = instance.get("class_name")
    if name is not None:
        category_id = category_name_to_id.get(normalized_class_name(name))
        if category_id is not None:
            return int(category_id)
    return None


def make_window(n_fft: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(n_fft).astype(np.float32)
    if window == "blackman-harris":
        n = np.arange(n_fft, dtype=np.float32)
        denom = max(float(n_fft - 1), 1.0)
        return (
            0.35875
            - 0.48829 * np.cos(2.0 * np.pi * n / denom)
            + 0.14128 * np.cos(4.0 * np.pi * n / denom)
            - 0.01168 * np.cos(6.0 * np.pi * n / denom)
        ).astype(np.float32)
    if window == "rect":
        return np.ones(n_fft, dtype=np.float32)
    raise ValueError(f"Unsupported STFT window: {window}")


def compute_complex_stft_tensor(iq: np.ndarray, n_fft: int, hop: int, window: str = "hann") -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size < n_fft:
        iq = np.pad(iq, (0, n_fft - iq.size))
    starts = np.arange(0, max(1, iq.size - n_fft + 1), hop)
    if starts.size == 0:
        starts = np.array([0])
    stft_window = make_window(n_fft, window)
    frames = np.stack([iq[start : start + n_fft] * stft_window for start in starts], axis=0)
    spec = np.fft.fftshift(np.fft.fft(frames, n=n_fft, axis=1), axes=1)
    spec = spec.T[::-1, :]  # high frequency at the top, matching the COCO y-axis mapping.
    tensor = np.stack([spec.real, spec.imag], axis=0).astype(np.float32, copy=False)
    return tensor


def instance_box_grid(
    instance: dict[str, Any],
    num_iq_samples: int,
    sample_rate: float,
    width: int,
    height: int,
) -> list[float] | None:
    start = instance.get("start")
    stop = instance.get("stop")
    if start is None:
        start_samples = float(instance.get("start_in_samples", 0))
        start = start_samples / max(num_iq_samples, 1)
    if stop is None:
        if instance.get("stop_in_samples") is not None:
            stop_samples = float(instance["stop_in_samples"])
        else:
            stop_samples = float(instance.get("start_in_samples", 0)) + float(instance.get("duration_in_samples", 0))
        stop = stop_samples / max(num_iq_samples, 1)

    bounds = instance_frequency_bounds(instance)
    if bounds is None:
        return None
    lower, upper = bounds

    start = float(np.clip(start, 0.0, 1.0))
    stop = float(np.clip(stop, 0.0, 1.0))
    lower_norm = float(np.clip(float(lower) / sample_rate + 0.5, 0.0, 1.0))
    upper_norm = float(np.clip(float(upper) / sample_rate + 0.5, 0.0, 1.0))

    x0 = min(start, stop) * width
    x1 = max(start, stop) * width
    y0 = (1.0 - max(lower_norm, upper_norm)) * height
    y1 = (1.0 - min(lower_norm, upper_norm)) * height
    w = x1 - x0
    h = y1 - y0
    if w <= 1.0 or h <= 1.0 or not all(math.isfinite(v) for v in [x0, y0, w, h]):
        return None
    return [round(x0, 3), round(y0, 3), round(w, 3), round(h, 3)]


def preview_tensor(tensor_path: Path, annotations: list[dict[str, Any]], out_path: Path) -> None:
    from PIL import Image

    tensor = np.load(tensor_path)
    power = np.log1p(np.sqrt(tensor[0] ** 2 + tensor[1] ** 2))
    lo, hi = np.percentile(power, [2.0, 98.0])
    image = np.clip((power - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    png = Image.fromarray((255.0 * image).astype(np.uint8), mode="L").convert("RGB")
    tmp = out_path.with_suffix(".tmp.png")
    png.save(tmp)
    render_preview(tmp, annotations, out_path)
    tmp.unlink(missing_ok=True)


def process_row_task(task: tuple[Any, ...]) -> tuple[int, dict[str, Any], list[dict[str, Any]], str, np.ndarray, np.ndarray, int]:
    (
        sample_idx,
        row,
        categories,
        stft_fft,
        stft_hop,
        stft_window,
        tensor_dtype_name,
        tensor_dir_text,
        raw_cache_root_text,
        skip_existing_stats,
    ) = task
    tensor_dir = Path(tensor_dir_text)
    raw_path = rewrite_path(str(row["raw_path"]))
    raw_cache_root = Path(raw_cache_root_text) if raw_cache_root_text else None

    sample_id = str(row.get("sample_id", f"sample_{sample_idx:06d}"))
    tensor_name = f"{sample_id}.npy"
    tensor_path = tensor_dir / tensor_name
    iq_length = int(row.get("num_iq_samples") or row.get("num_samples") or 0)
    if tensor_path.exists():
        try:
            if bool(skip_existing_stats):
                tensor_view = np.load(tensor_path, mmap_mode="r")
                tensor_shape = tuple(tensor_view.shape)
                del tensor_view
                tensor = None
            else:
                tensor = np.asarray(np.load(tensor_path))
                tensor_shape = tuple(tensor.shape)
        except Exception:
            iq = load_iq_for_row(raw_path, raw_cache_root)
            iq_length = len(iq)
            tensor = compute_complex_stft_tensor(iq, int(stft_fft), int(stft_hop), str(stft_window))
            np.save(tensor_path, tensor.astype(np.dtype(tensor_dtype_name), copy=False))
            tensor_shape = tuple(tensor.shape)
    else:
        iq = load_iq_for_row(raw_path, raw_cache_root)
        iq_length = len(iq)
        tensor = compute_complex_stft_tensor(iq, int(stft_fft), int(stft_hop), str(stft_window))
        np.save(tensor_path, tensor.astype(np.dtype(tensor_dtype_name), copy=False))
        tensor_shape = tuple(tensor.shape)
    if iq_length <= 0:
        iq = load_iq_for_row(raw_path, raw_cache_root)
        iq_length = len(iq)
    _, height, width = tensor_shape

    if tensor is None:
        channel_sum = np.zeros(2, dtype=np.float64)
        channel_sumsq = np.zeros(2, dtype=np.float64)
        channel_count = 0
    else:
        flat = tensor.reshape(2, -1).astype(np.float64)
        channel_sum = flat.sum(axis=1)
        channel_sumsq = np.square(flat).sum(axis=1)
        channel_count = int(flat.shape[1])

    image_id = int(sample_idx) + 1
    image = {"id": image_id, "file_name": tensor_name, "width": width, "height": height}

    valid_category_ids = {int(cat["id"]) for cat in categories}
    category_name_to_id = {normalized_class_name(cat.get("name", "")): int(cat["id"]) for cat in categories}
    sample_rate = float(row.get("sample_rate", 1_000_000.0))
    annotation_rows: list[dict[str, Any]] = []
    serial_instances: list[dict[str, Any]] = []
    for inst in row.get("instances", []):
        inst = dict(inst)
        bbox = instance_box_grid(inst, iq_length, sample_rate, width, height)
        serial_instances.append(inst)
        if bbox is None:
            continue
        category_id = category_id_for_instance(
            inst,
            valid_category_ids=valid_category_ids,
            category_name_to_id=category_name_to_id,
        )
        if category_id is None:
            continue
        annotation_rows.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 3),
                "iscrowd": 0,
            }
        )

    metadata_line = json.dumps(
        {
            **row,
            "tensor_path": portable_path(tensor_path),
            "stft_fft": int(stft_fft),
            "stft_hop": int(stft_hop),
            "stft_window": str(stft_window),
            "stft_height": height,
            "stft_width": width,
            "instances": serial_instances,
        },
        ensure_ascii=False,
    )
    return int(sample_idx), image, annotation_rows, metadata_line, channel_sum, channel_sumsq, channel_count


def write_split(
    src_root: Path,
    out_root: Path,
    split: str,
    categories: list[dict[str, Any]],
    *,
    stft_fft: int,
    stft_hop: int,
    stft_window: str,
    tensor_dtype: np.dtype,
    workers: int,
    raw_cache_root: Path | None,
    skip_existing_stats: bool,
) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    rows = load_rows(src_root, split)
    tensor_dir = out_root / "coco" / split / "tensors"
    ann_dir = out_root / "coco" / "annotations"
    meta_dir = out_root / "metadata"
    for path in (tensor_dir, ann_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_lines: list[str] = []
    channel_sum = np.zeros(2, dtype=np.float64)
    channel_sumsq = np.zeros(2, dtype=np.float64)
    channel_count = 0
    ann_id = 1

    tasks = [
        (
            sample_idx,
            row,
            categories,
            stft_fft,
            stft_hop,
            stft_window,
            tensor_dtype.name,
            str(tensor_dir),
            str(raw_cache_root or ""),
            bool(skip_existing_stats),
        )
        for sample_idx, row in enumerate(rows)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(process_row_task, tasks, chunksize=16)
            for exported, image, annotation_rows, metadata_line, sums, sumsq, count in iterator:
                images.append(image)
                for annotation in annotation_rows:
                    annotations.append({"id": ann_id, **annotation})
                    ann_id += 1
                metadata_lines.append(metadata_line)
                channel_sum += sums
                channel_sumsq += sumsq
                channel_count += count
                if (exported + 1) % 500 == 0 or exported + 1 == len(rows):
                    print(f"[export-complex-stft] {split}: exported {exported + 1}/{len(rows)}", flush=True)
    else:
        for task in tasks:
            exported, image, annotation_rows, metadata_line, sums, sumsq, count = process_row_task(task)
            images.append(image)
            for annotation in annotation_rows:
                annotations.append({"id": ann_id, **annotation})
                ann_id += 1
            metadata_lines.append(metadata_line)
            channel_sum += sums
            channel_sumsq += sumsq
            channel_count += count
            if (exported + 1) % 500 == 0 or exported + 1 == len(rows):
                print(f"[export-complex-stft] {split}: exported {exported + 1}/{len(rows)}", flush=True)

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    if (split == "train" or split.endswith("_train_all")) and images:
        first_tensor = tensor_dir / images[0]["file_name"]
        first_anns = [ann for ann in annotations if ann["image_id"] == 1]
        preview_tensor(first_tensor, first_anns, out_root / f"preview_{split}_000000.png")

    print(f"[export-complex-stft] {split}: {len(images)} tensors, {len(annotations)} boxes", flush=True)
    return len(images), len(annotations), channel_sum, channel_sumsq, channel_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stft-fft", type=int, required=True)
    parser.add_argument("--stft-hop", type=int, required=True)
    parser.add_argument("--stft-window", choices=["hann", "blackman-harris", "rect"], default="hann")
    parser.add_argument("--tensor-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--raw-cache-root", default="", help="Optional split-organized raw IQ .npy cache root.")
    parser.add_argument("--skip-existing-stats", action="store_true", help="Do not read full existing tensors for intermediate summary statistics.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for raw-to-STFT export.")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated metadata split names. Use official TorchSig names for latest configs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = ROOT / args.src_root
    out_root = ROOT / args.out_root
    raw_cache_root = ROOT / args.raw_cache_root if args.raw_cache_root else None
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_summary = load_json(src_root / "summary.json")
    categories = source_summary.get("categories", [])
    split_stats = {}
    total_images = 0
    total_boxes = 0
    train_sum = np.zeros(2, dtype=np.float64)
    train_sumsq = np.zeros(2, dtype=np.float64)
    train_count = 0
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if args.splits == "auto":
        requested_splits = sorted(path.stem for path in (src_root / "metadata").glob("*.jsonl"))
    if not requested_splits:
        raise ValueError("No splits requested.")

    for split in requested_splits:
        images, boxes, channel_sum, channel_sumsq, channel_count = write_split(
            src_root,
            out_root,
            split,
            categories,
            stft_fft=args.stft_fft,
            stft_hop=args.stft_hop,
            stft_window=args.stft_window,
            tensor_dtype=np.dtype(args.tensor_dtype),
            workers=max(int(args.workers), 1),
            raw_cache_root=raw_cache_root,
            skip_existing_stats=bool(args.skip_existing_stats),
        )
        split_stats[split] = {"images": images, "boxes": boxes, "channel_count": channel_count}
        total_images += images
        total_boxes += boxes
        if split == "train" or split.endswith("_train_all") or "_train_" in split:
            train_sum += channel_sum
            train_sumsq += channel_sumsq
            train_count += channel_count

    mean = train_sum / max(train_count, 1)
    var = train_sumsq / max(train_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    summary = {
        **source_summary,
        "out_root": str(out_root),
        "source_root": str(src_root),
        "stft_representation": "complex_real_imag_tensor",
        "stft_fft": args.stft_fft,
        "stft_hop": args.stft_hop,
        "stft_window": args.stft_window,
        "tensor_dtype": args.tensor_dtype,
        "tensor_layout": "C,F,T",
        "tensor_channels": ["real", "imag"],
        "categories": categories,
        "splits": split_stats,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "stft_tensor_stats": {
            "mean": [float(v) for v in mean],
            "std": [float(max(v, 1e-6)) for v in std],
            "computed_on": [split for split in requested_splits if split == "train" or split.endswith("_train_all") or "_train_" in split],
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export-complex-stft] train mean={mean.tolist()} std={std.tolist()}")
    print(f"[export-complex-stft] wrote {out_root}")


if __name__ == "__main__":
    main()
