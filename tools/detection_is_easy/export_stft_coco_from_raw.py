# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from prepare_torchsig_coco import compute_stft_image, instance_box, render_preview  # noqa: E402


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


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_split(
    src_root: Path,
    out_root: Path,
    split: str,
    categories: list[dict[str, Any]],
    *,
    stft_fft: int,
    stft_hop: int,
    image_size: int,
) -> None:
    rows = load_rows(src_root, split)
    img_dir = out_root / "coco" / split / "images"
    ann_dir = out_root / "coco" / "annotations"
    meta_dir = out_root / "metadata"
    for path in (img_dir, ann_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_lines: list[str] = []
    ann_id = 1

    for sample_idx, row in enumerate(rows):
        raw_path = rewrite_path(str(row["raw_path"]))
        with np.load(raw_path) as npz:
            iq = np.asarray(npz["iq"], dtype=np.complex64)

        sample_id = str(row.get("sample_id", f"{split}_{sample_idx:06d}"))
        image = compute_stft_image(iq, stft_fft, stft_hop, image_size)
        image_name = f"{sample_id}.png"
        image_path = img_dir / image_name
        image.save(image_path)

        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": image_size,
                "height": image_size,
            }
        )

        sample_rate = float(row.get("sample_rate", 1_000_000.0))
        serial_instances: list[dict[str, Any]] = []
        for inst in row.get("instances", []):
            inst = dict(inst)
            bbox = instance_box(inst, len(iq), sample_rate, image_size)
            serial_instances.append(inst)
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(inst.get("category_id", 0)),
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
                    "image_path": portable_path(image_path),
                    "stft_fft": stft_fft,
                    "stft_hop": stft_hop,
                    "image_size": image_size,
                    "instances": serial_instances,
                },
                ensure_ascii=False,
            )
        )
        if (sample_idx + 1) % 500 == 0 or sample_idx + 1 == len(rows):
            print(f"[export-stft] {split}: exported {sample_idx + 1}/{len(rows)}")

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    if split == "train" and images:
        first_image = img_dir / images[0]["file_name"]
        first_anns = [ann for ann in annotations if ann["image_id"] == 1]
        render_preview(first_image, first_anns, out_root / "preview_train_000000.png")

    print(f"[export-stft] {split}: {len(images)} images, {len(annotations)} boxes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stft-fft", type=int, required=True)
    parser.add_argument("--stft-hop", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = ROOT / args.src_root
    out_root = ROOT / args.out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_summary = load_json(src_root / "summary.json")
    categories = source_summary.get("categories", [])
    for split in ("train", "val", "test"):
        write_split(
            src_root,
            out_root,
            split,
            categories,
            stft_fft=args.stft_fft,
            stft_hop=args.stft_hop,
            image_size=args.image_size,
        )

    summary = {
        **source_summary,
        "out_root": str(out_root),
        "source_root": str(src_root),
        "stft_fft": args.stft_fft,
        "stft_hop": args.stft_hop,
        "image_size": args.image_size,
        "categories": categories,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export-stft] wrote {out_root}")


if __name__ == "__main__":
    main()
