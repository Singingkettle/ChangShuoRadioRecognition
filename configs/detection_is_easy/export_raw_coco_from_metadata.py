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

from export_complex_stft_coco_from_raw import (  # noqa: E402
    category_id_for_instance,
    instance_box_grid,
    load_json,
    load_rows,
    normalized_class_name,
    rewrite_path,
)


def infer_num_iq_samples(row: dict[str, Any]) -> int:
    for key in ("num_iq_samples", "num_samples", "n_samples", "sample_count"):
        value = row.get(key)
        if value is not None:
            try:
                return max(int(value), 1)
            except (TypeError, ValueError):
                pass

    max_stop = 0.0
    for inst in row.get("instances", []):
        if inst.get("stop_in_samples") is not None:
            max_stop = max(max_stop, float(inst["stop_in_samples"]))
        elif inst.get("start_in_samples") is not None and inst.get("duration_in_samples") is not None:
            max_stop = max(max_stop, float(inst["start_in_samples"]) + float(inst["duration_in_samples"]))
    if max_stop > 0:
        return max(int(round(max_stop)), 1)

    raw_path = row.get("raw_path")
    if raw_path:
        import numpy as np

        with np.load(rewrite_path(str(raw_path))) as npz:
            return max(int(np.asarray(npz["iq"]).shape[-1]), 1)
    raise ValueError("Could not infer IQ length from metadata row.")


def write_split(
    *,
    src_root: Path,
    out_root: Path,
    split: str,
    categories: list[dict[str, Any]],
    width: int,
    height: int,
    single_class: bool,
) -> tuple[int, int]:
    rows = load_rows(src_root, split)
    ann_dir = out_root / "annotations"
    meta_dir = out_root.parent / "metadata"
    tensor_dir = out_root / split / "tensors"
    ann_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_lines: list[str] = []
    valid_category_ids = {int(cat["id"]) for cat in categories}
    category_name_to_id = {normalized_class_name(cat.get("name", "")): int(cat["id"]) for cat in categories}
    ann_id = 1

    for sample_idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"sample_{sample_idx:06d}"))
        file_name = f"{sample_id}.npy"
        image_id = int(sample_idx) + 1
        images.append({"id": image_id, "file_name": file_name, "width": int(width), "height": int(height)})

        num_iq_samples = infer_num_iq_samples(row)
        sample_rate = float(row.get("sample_rate", 1_000_000.0))
        serial_instances: list[dict[str, Any]] = []
        for inst in row.get("instances", []):
            inst = dict(inst)
            bbox = instance_box_grid(inst, num_iq_samples, sample_rate, int(width), int(height))
            serial_instances.append(inst)
            if bbox is None:
                continue
            category_id = 1 if single_class else category_id_for_instance(
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
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "area": round(float(bbox[2]) * float(bbox[3]), 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        metadata_lines.append(
            json.dumps(
                {
                    **row,
                    "raw_coco_file_name": file_name,
                    "raw_coco_width": int(width),
                    "raw_coco_height": int(height),
                    "instances": serial_instances,
                },
                ensure_ascii=False,
            )
        )

    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (meta_dir / f"{split}.jsonl").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    print(f"[export-raw-coco] {split}: {len(images)} images, {len(annotations)} boxes")
    return len(images), len(annotations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export COCO annotations directly from raw-IQ metadata.")
    parser.add_argument("--src-root", required=True, help="Raw-IQ dataset root containing summary.json and metadata/*.jsonl.")
    parser.add_argument("--out-root", required=True, help="Output dataset root; annotations are written under out-root/coco.")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--single-class", action="store_true", help="Collapse all valid boxes to one 'signal' category.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    if not src_root.is_absolute():
        src_root = ROOT / src_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    coco_root = out_root / "coco"
    coco_root.mkdir(parents=True, exist_ok=True)

    source_summary = load_json(src_root / "summary.json")
    if args.single_class:
        categories = [{"id": 1, "name": "signal"}]
    else:
        categories = source_summary.get("categories", [])
    if not categories:
        raise ValueError("No categories found. Use --single-class or provide categories in summary.json.")

    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if args.splits == "auto":
        requested_splits = sorted(path.stem for path in (src_root / "metadata").glob("*.jsonl"))
    if not requested_splits:
        raise ValueError("No splits requested.")

    split_stats: dict[str, dict[str, int]] = {}
    total_images = 0
    total_boxes = 0
    for split in requested_splits:
        images, boxes = write_split(
            src_root=src_root,
            out_root=coco_root,
            split=split,
            categories=categories,
            width=int(args.width),
            height=int(args.height),
            single_class=bool(args.single_class),
        )
        split_stats[split] = {"images": images, "boxes": boxes}
        total_images += images
        total_boxes += boxes

    summary = {
        **source_summary,
        "out_root": str(out_root),
        "source_root": str(src_root),
        "representation": "raw_iq_metadata_coco",
        "single_class": bool(args.single_class),
        "width": int(args.width),
        "height": int(args.height),
        "categories": categories,
        "splits": split_stats,
        "total_images": total_images,
        "total_boxes": total_boxes,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[export-raw-coco] wrote {out_root}: {total_images} images, {total_boxes} boxes")


if __name__ == "__main__":
    main()
