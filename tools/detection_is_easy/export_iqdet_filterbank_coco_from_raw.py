# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from export_complex_stft_coco_from_raw import (  # noqa: E402
    category_id_for_instance,
    instance_box_grid,
    load_json,
    load_rows,
    normalized_class_name,
    portable_path,
    rewrite_path,
)
from prepare_torchsig_coco import render_preview  # noqa: E402
from iqdet_complex import ComplexGaborFilterbank  # noqa: E402


def _iq_to_pair(iq: np.ndarray, device: torch.device) -> torch.Tensor:
    iq = np.asarray(iq, dtype=np.complex64)
    pair = np.stack([iq.real, iq.imag], axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(pair).unsqueeze(0).to(device=device)


def build_channel_tensor(tokens: torch.Tensor, channel_mode: str) -> np.ndarray:
    if tokens.ndim != 3 or not torch.is_complex(tokens):
        raise ValueError(f"Expected complex filterbank tokens [B,F,T], got {tuple(tokens.shape)}")
    if tokens.shape[0] != 1:
        raise ValueError("The exporter processes one IQ sample at a time.")

    # High frequency should appear at the top of the COCO image, matching the
    # existing complex-STFT exporter and bbox y-axis convention.
    spec = torch.flip(tokens[0], dims=(0,))
    real = spec.real.float()
    imag = spec.imag.float()
    if channel_mode == "realimag":
        out = torch.stack([real, imag], dim=0)
    elif channel_mode == "realimag_logmag":
        logmag = torch.log1p(torch.abs(spec).float())
        out = torch.stack([real, imag, logmag], dim=0)
    elif channel_mode == "logmag2ch":
        logmag = torch.log1p(torch.abs(spec).float())
        out = torch.stack([logmag, logmag], dim=0)
    else:
        raise ValueError(f"Unsupported channel_mode: {channel_mode!r}")
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_filterbank_tensor(
    iq: np.ndarray,
    filterbank: ComplexGaborFilterbank,
    *,
    device: torch.device,
    channel_mode: str,
) -> np.ndarray:
    with torch.no_grad():
        x = _iq_to_pair(iq, device)
        tokens = filterbank(x)
    return build_channel_tensor(tokens, channel_mode)


def channel_names(channel_mode: str) -> list[str]:
    if channel_mode == "realimag":
        return ["real", "imag"]
    if channel_mode == "realimag_logmag":
        return ["real", "imag", "log_magnitude"]
    if channel_mode == "logmag2ch":
        return ["log_magnitude", "log_magnitude"]
    raise ValueError(f"Unsupported channel_mode: {channel_mode!r}")


def preview_filterbank_tensor(tensor_path: Path, annotations: list[dict[str, Any]], out_path: Path) -> None:
    from PIL import Image

    tensor = np.load(tensor_path)
    if tensor.shape[0] >= 2 and "real" in channel_names_for_tensor(tensor):
        power = np.log1p(np.sqrt(tensor[0] ** 2 + tensor[1] ** 2))
    else:
        power = tensor[0]
    lo, hi = np.percentile(power, [2.0, 98.0])
    image = np.clip((power - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    png = Image.fromarray((255.0 * image).astype(np.uint8), mode="L").convert("RGB")
    tmp = out_path.with_suffix(".tmp.png")
    png.save(tmp)
    render_preview(tmp, annotations, out_path)
    tmp.unlink(missing_ok=True)


def channel_names_for_tensor(tensor: np.ndarray) -> list[str]:
    if tensor.shape[0] == 2:
        return ["real", "imag"]
    if tensor.shape[0] == 3:
        return ["real", "imag", "log_magnitude"]
    return [f"channel_{idx}" for idx in range(tensor.shape[0])]


def write_split(
    src_root: Path,
    out_root: Path,
    split: str,
    categories: list[dict[str, Any]],
    *,
    filterbank: ComplexGaborFilterbank,
    device: torch.device,
    channel_mode: str,
    tensor_dtype: np.dtype,
    filterbank_config: dict[str, Any],
) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    rows = load_rows(src_root, split)
    tensor_dir = out_root / "coco" / split / "tensors"
    ann_dir = out_root / "coco" / "annotations"
    meta_dir = out_root / "metadata"
    for path in (tensor_dir, ann_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    channels = channel_names(channel_mode)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_lines: list[str] = []
    valid_category_ids = {int(cat["id"]) for cat in categories}
    category_name_to_id = {normalized_class_name(cat.get("name", "")): int(cat["id"]) for cat in categories}
    channel_sum = np.zeros(len(channels), dtype=np.float64)
    channel_sumsq = np.zeros(len(channels), dtype=np.float64)
    channel_count = 0
    ann_id = 1

    for sample_idx, row in enumerate(rows):
        raw_path = rewrite_path(str(row["raw_path"]))
        with np.load(raw_path) as npz:
            iq = np.asarray(npz["iq"], dtype=np.complex64)

        sample_id = str(row.get("sample_id", f"{split}_{sample_idx:06d}"))
        tensor = compute_filterbank_tensor(iq, filterbank, device=device, channel_mode=channel_mode)
        _, height, width = tensor.shape
        tensor_name = f"{sample_id}.npy"
        tensor_path = tensor_dir / tensor_name
        np.save(tensor_path, tensor.astype(tensor_dtype, copy=False))

        flat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        channel_sum += flat.sum(axis=1)
        channel_sumsq += np.square(flat).sum(axis=1)
        channel_count += flat.shape[1]

        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": tensor_name, "width": width, "height": height})

        sample_rate = float(row.get("sample_rate", 1_000_000.0))
        serial_instances: list[dict[str, Any]] = []
        for inst in row.get("instances", []):
            inst = dict(inst)
            bbox = instance_box_grid(inst, len(iq), sample_rate, width, height)
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
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        metadata_lines.append(
            json.dumps(
                {
                    **row,
                    "tensor_path": portable_path(tensor_path),
                    "iqdet_filterbank": filterbank_config,
                    "filterbank_height": height,
                    "filterbank_width": width,
                    "tensor_channels": channels,
                    "instances": serial_instances,
                },
                ensure_ascii=False,
            )
        )
        if (sample_idx + 1) % 500 == 0 or sample_idx + 1 == len(rows):
            print(f"[export-iqdet-filterbank] {split}: exported {sample_idx + 1}/{len(rows)}")

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    if split == "train" and images:
        first_tensor = tensor_dir / images[0]["file_name"]
        first_anns = [ann for ann in annotations if ann["image_id"] == 1]
        preview_filterbank_tensor(first_tensor, first_anns, out_root / "preview_train_000000.png")

    print(f"[export-iqdet-filterbank] {split}: {len(images)} tensors, {len(annotations)} boxes")
    return len(images), len(annotations), channel_sum, channel_sumsq, channel_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num-bins", type=int, default=512)
    parser.add_argument("--kernel-size", type=int, default=513)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--filterbank-init", choices=["gabor", "fourier", "stft", "fft"], default="fourier")
    parser.add_argument(
        "--filterbank-window",
        choices=["gaussian", "hann", "blackman-harris", "rect"],
        default="blackman-harris",
    )
    parser.add_argument("--channel-mode", choices=["realimag", "realimag_logmag", "logmag2ch"], default="realimag")
    parser.add_argument("--tensor-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = ROOT / args.src_root
    out_root = ROOT / args.out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    source_summary = load_json(src_root / "summary.json")
    categories = source_summary.get("categories", [])
    filterbank = ComplexGaborFilterbank(
        num_bins=args.num_bins,
        kernel_size=args.kernel_size,
        stride=args.stride,
        init=args.filterbank_init,
        window=args.filterbank_window,
    ).to(device=device)
    filterbank.eval()
    filterbank_config = {
        "type": "ComplexGaborFilterbank",
        "num_bins": args.num_bins,
        "kernel_size": args.kernel_size,
        "stride": args.stride,
        "init": args.filterbank_init,
        "window": args.filterbank_window,
        "frequency_axis": "flipped_high_to_low_for_coco_y",
        "trainable_in_export": False,
    }

    channels = channel_names(args.channel_mode)
    split_stats = {}
    total_images = 0
    total_boxes = 0
    train_sum = np.zeros(len(channels), dtype=np.float64)
    train_sumsq = np.zeros(len(channels), dtype=np.float64)
    train_count = 0
    for split in ("train", "val", "test"):
        images, boxes, channel_sum, channel_sumsq, channel_count = write_split(
            src_root,
            out_root,
            split,
            categories,
            filterbank=filterbank,
            device=device,
            channel_mode=args.channel_mode,
            tensor_dtype=np.dtype(args.tensor_dtype),
            filterbank_config=filterbank_config,
        )
        split_stats[split] = {"images": images, "boxes": boxes, "channel_count": channel_count}
        total_images += images
        total_boxes += boxes
        if split == "train":
            train_sum = channel_sum
            train_sumsq = channel_sumsq
            train_count = channel_count

    mean = train_sum / max(train_count, 1)
    var = train_sumsq / max(train_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    summary = {
        **source_summary,
        "out_root": str(out_root),
        "source_root": str(src_root),
        "front_end_representation": "iqdet_complex_filterbank_tensor",
        "stft_representation": "iqdet_complex_filterbank_tensor",
        "iqdet_filterbank": filterbank_config,
        "tensor_dtype": args.tensor_dtype,
        "tensor_layout": "C,F,T",
        "tensor_channels": channels,
        "channel_mode": args.channel_mode,
        "categories": categories,
        "splits": split_stats,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "stft_tensor_stats": {
            "mean": [float(v) for v in mean],
            "std": [float(max(v, 1e-6)) for v in std],
            "computed_on": "train",
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export-iqdet-filterbank] train mean={mean.tolist()} std={std.tolist()}")
    print(f"[export-iqdet-filterbank] wrote {out_root}")


if __name__ == "__main__":
    main()
