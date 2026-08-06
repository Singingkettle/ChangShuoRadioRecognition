# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_split(root: Path, split: str) -> tuple[int, int]:
    ann_path = root / "coco" / "annotations" / f"instances_{split}.json"
    meta_path = root / "metadata" / f"{split}.jsonl"
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    assert images, f"{split}: no images"
    assert coco["categories"], f"{split}: no categories"

    for img in coco["images"]:
        assert img["width"] > 0 and img["height"] > 0
        assert img["file_name"]
        assert (root / "coco" / split / "images" / img["file_name"]).exists()

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0, (split, ann)
        assert x + w <= img["width"] + 1e-3, (split, ann)
        assert y + h <= img["height"] + 1e-3, (split, ann)
        assert ann["category_id"] in {cat["id"] for cat in coco["categories"]}

    lines = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(images), f"{split}: metadata/image count mismatch"
    for row in lines:
        raw_path = repo_root() / row["raw_path"]
        assert raw_path.exists(), raw_path
        with np.load(raw_path) as npz:
            assert "iq" in npz
            assert npz["iq"].ndim == 1
        assert len(row["instances"]) >= 1, f"{split}: empty TorchSig instance list"

    return len(images), len(coco["annotations"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/torchsig_mini")
    args = parser.parse_args()
    root = repo_root() / args.root
    total_images = 0
    total_boxes = 0
    for split in ("train", "val", "test"):
        images, boxes = validate_split(root, split)
        total_images += images
        total_boxes += boxes
        print(f"[validate] {split}: {images} images, {boxes} boxes")
    print(f"[validate] ok: {total_images} images, {total_boxes} boxes")


if __name__ == "__main__":
    main()
