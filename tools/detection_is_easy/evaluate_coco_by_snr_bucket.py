# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()


def metric_rows(label: str, stats: list[float]) -> list[dict[str, Any]]:
    names = [
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
        "bbox_AR_1",
        "bbox_AR_10",
        "bbox_AR_100",
        "bbox_AR_s",
        "bbox_AR_m",
        "bbox_AR_l",
    ]
    return [{"snr_bucket": label, "metric": name, "value": float(value)} for name, value in zip(names, stats)]


def load_bucket_image_ids(metadata_root: Path, coco: COCO, split: str) -> dict[str, list[int]]:
    path = metadata_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    stem_to_image_id = {Path(image["file_name"]).stem: int(image["id"]) for image in coco.dataset.get("images", [])}
    buckets: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in stem_to_image_id:
                continue
            bucket = row.get("snr_bucket")
            if bucket is None:
                values = [float(inst["snr_db"]) for inst in row.get("instances", []) if inst.get("snr_db") is not None]
                if not values:
                    label = "unknown"
                else:
                    mean_snr = sum(values) / len(values)
                    label = f"snr_mean={mean_snr:.1f}"
            else:
                lo, hi = float(bucket[0]), float(bucket[1])
                label = f"[{lo:g},{hi:g})"
            buckets.setdefault(label, []).append(stem_to_image_id[sample_id])
    return buckets


def evaluate_subset(coco_gt: COCO, coco_dt: COCO, img_ids: list[int]) -> list[float]:
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(set(int(v) for v in img_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(v) for v in evaluator.stats]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    coco_root = ROOT / args.coco_root
    metadata_root = ROOT / args.metadata_root
    prediction_json = ROOT / args.prediction_json
    out_csv = ROOT / args.out_csv
    ann_file = coco_root / "annotations" / f"instances_{args.split}.json"

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(prediction_json))
    bucket_ids = load_bucket_image_ids(metadata_root, coco_gt, args.split)

    rows: list[dict[str, Any]] = []
    rows.extend(metric_rows("all", evaluate_subset(coco_gt, coco_dt, list(coco_gt.getImgIds()))))
    for label in sorted(bucket_ids):
        rows.extend(metric_rows(label, evaluate_subset(coco_gt, coco_dt, bucket_ids[label])))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_bucket", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[snr-bucket-eval] wrote {out_csv}")


if __name__ == "__main__":
    main()
