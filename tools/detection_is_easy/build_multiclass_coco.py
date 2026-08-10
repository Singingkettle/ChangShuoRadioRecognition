# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build a MULTI-CLASS COCO annotation set from the existing SINGLE-CLASS stft3_memmap
COCO, by label-swapping: reuse the exact rendered boxes (so the detector input/targets
are identical to the 0.948 localizer pipeline) and assign each box its true modulation
category_id from the dataset metadata (matched by max IoU within the image).

This lets us train the BASELINE multi-class RTMDet-M on the SAME memmap input as OURS
localizer -- the only difference is the head (57-class vs 1-class), an apples-to-apples
class-aware comparison. Writes instances_{split}.json with 57 categories + a match report.
Run on the server (data lives there). Verifies the box mapping as a side effect:
if match rate at IoU>0.5 is ~100%, the metadata<->render alignment holds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def meta_boxes_px(meta_jsonl: Path, W: int, H: int):
    """sample_id -> list of (pixel [x,y,w,h], category_id, class_name)."""
    out = {}
    for line in Path(meta_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        nq, fs, sid = r["num_iq_samples"], r["sample_rate"], r["sample_id"]
        boxes = []
        for inst in r["instances"]:
            x0 = _clip(inst["start_in_samples"] / nq, 0.0, 1.0) * W
            x1 = _clip((inst["start_in_samples"] + inst["duration_in_samples"]) / nq, 0.0, 1.0) * W
            f_hi = _clip(inst["_upper_frequency"] / fs, -0.5, 0.5)
            f_lo = _clip(inst["_lower_frequency"] / fs, -0.5, 0.5)
            y0 = (0.5 - f_hi) * H
            y1 = (0.5 - f_lo) * H
            boxes.append(([x0, y0, x1 - x0, y1 - y0], int(inst["category_id"]), inst.get("class_name", str(inst["category_id"]))))
        out[sid] = boxes
    return out


def xwh_dist(a, b):
    """L1 distance on (x, w, h) only -- ALL flip-invariant. The frequency (y) axis is
    mirror-flipped per-image between metadata and the render, but x/w/h are preserved, so
    (x,w,h) uniquely identifies which metadata instance a rendered box is (to read its class)."""
    return abs(a[0] - b[0]) + abs(a[2] - b[2]) + abs(a[3] - b[3])


def link_split_dir(mc_root: Path, coco_root: Path, split: str) -> None:
    """Mirror ``coco/<split>/`` into the multiclass root so the root is directly trainable.

    The training harness decides between the ``tensors/`` and ``images/`` data prefix by
    testing whether ``<root>/<split>/tensors`` exists, and the memmap loader reads the split
    name back out of that path. A multiclass root holding only ``annotations/`` therefore
    resolves the split to "images" and then looks for a non-existent ``images.npy``. Linking
    the split directory across is enough: the per-sample files are never opened, only their
    names are parsed.
    """
    src = coco_root / split
    dst = mc_root / split
    if dst.exists() or not src.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:                      # no symlink privilege (e.g. Windows): mirror the names
        (dst / "tensors").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="stft3_memmap dir with coco/ and metadata/")
    ap.add_argument("--meta-dir", default=None, help="metadata dir (defaults to <dataset-dir>/metadata)")
    ap.add_argument("--out-dir", required=True, help="where to write multi-class instances_{split}.json")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--match-tol", type=float, default=3.0, help="max L1 (x,w,h) px distance to accept a metadata match")
    args = ap.parse_args()

    dd = Path(args.dataset_dir)
    md = Path(args.meta_dir) if args.meta_dir else dd / "metadata"
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    splits = args.splits.split(",")

    # global category map from all splits' metadata
    catset = {}
    for sp in splits:
        for line in (md / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for inst in json.loads(line)["instances"]:
                catset[int(inst["category_id"])] = inst.get("class_name", str(inst["category_id"]))
    categories = [{"id": cid, "name": catset[cid], "supercategory": "signal"} for cid in sorted(catset)]
    print(f"[multiclass] {len(categories)} categories: {[c['name'] for c in categories[:6]]} ...")

    for sp in splits:
        sc = json.loads((dd / "coco" / "annotations" / f"instances_{sp}.json").read_text(encoding="utf-8"))
        W = sc["images"][0].get("width", 512)
        H = sc["images"][0].get("height", 512)
        id2stem = {im["id"]: Path(im["file_name"]).stem for im in sc["images"]}
        mb = meta_boxes_px(md / f"{sp}.jsonl", W, H)
        n_match, n_total, worst = 0, 0, 0.0
        miss = 0
        new_anns = []
        for a in sc["annotations"]:
            stem = id2stem[a["image_id"]]
            cands = mb.get(stem, [])
            n_total += 1
            best_d, best_cat = 1e18, None
            for (pb, cid, _name) in cands:
                d = xwh_dist(a["bbox"], pb)
                if d < best_d:
                    best_d, best_cat = d, cid
            if best_cat is not None and best_d <= args.match_tol:
                n_match += 1
                worst = max(worst, best_d)
                cat = best_cat
            else:
                miss += 1
                cat = categories[0]["id"] if best_cat is None else best_cat  # fallback keep box
            na = dict(a)
            na["category_id"] = cat
            new_anns.append(na)
        out = {"images": sc["images"], "annotations": new_anns, "categories": categories}
        outp = od / f"instances_{sp}.json"
        outp.write_text(json.dumps(out), encoding="utf-8")
        rate = n_match / max(n_total, 1)
        print(f"[{sp}] boxes={n_total} matched@(x,w,h)L1<={args.match_tol}px: {n_match} ({rate:.4f}) "
              f"unmatched={miss} worst-matched-dist={worst:.2f}px -> {outp}")
        if rate < 0.98:
            print(f"[{sp}] WARNING: match rate < 0.98 -- mapping or alignment may be off; inspect before training.")
        link_split_dir(od.parent, dd / "coco", sp)


if __name__ == "__main__":
    main()
