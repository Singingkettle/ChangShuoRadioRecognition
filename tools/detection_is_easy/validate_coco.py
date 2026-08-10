# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Structural + distributional check on a generated dataset.

Answers two different questions:

* *is this dataset well formed?* -- annotations parse, boxes lie inside the canvas, every
  referenced sample exists, metadata and images agree in count, the memmap has one row per
  image and the rows line up with the ids the loader parses out of the file names.
* *is this the benchmark the paper describes?* -- the generator settings recorded in
  ``summary.json`` and the observed per-signal SNR, duration and bandwidth match the
  hardshort-lowsnr configuration. This is the check that catches a dataset built over the
  wrong SNR span, which is otherwise invisible: it trains fine and simply reports different
  numbers.

    python tools/detection_is_easy/validate_coco.py --root data/torchsig_hardshort_lowsnr_stft3_memmap

Exit code 0 = all checks pass, 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# The hardshort-lowsnr configuration the paper is built on.
PAPER_SPEC = {
    "num_iq_samples": 262144,
    "sample_rate": 10_000_000.0,
    "num_signals_min": 1,
    "num_signals_max": 6,
    "impairment_level": 0,
    "stft_fft": 512,
    "stft_hop": 512,
    "duration_min_frac": 0.005,
    "duration_max_frac": 0.25,
    "bandwidth_min_frac": 0.0125,
    "bandwidth_max_frac": 0.49,
    "snr_db_min": -20.0,
    "snr_db_max": 10.0,
    "cochannel_overlap_probability": 0.35,
    "fast_snr_update": True,
}
EXPECTED_SPLIT_SIZES = {"train": 50000, "val": 5000, "test": 10000}

FAILURES: list[str] = []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def sample_dir(coco_root: Path, split: str) -> tuple[Path, str] | tuple[None, None]:
    """The paper's datasets store `.npy` tensors; the smoke datasets store `.png` images."""
    for sub in ("tensors", "images"):
        d = coco_root / split / sub
        if d.exists():
            return d, sub
    return None, None


def resolve_raw(row: dict, raw_root: Path | None, split: str) -> Path | None:
    """metadata `raw_path` is absolute and machine-specific; fall back to reconstruction."""
    candidates = []
    value = row.get("raw_path")
    if value:
        p = Path(value)
        candidates.append(p if p.is_absolute() else repo_root() / p)
    if raw_root is not None:
        sid = row.get("sample_id", "")
        candidates += [raw_root / "raw" / split / f"{sid}.npz",
                       raw_root / "raw_npy_cache" / split / f"{sid}.npy",
                       raw_root / "raw" / split / f"{sid}.npy"]
    for c in candidates:
        if c.exists():
            return c
    return None


def validate_split(root: Path, split: str, raw_root: Path | None, coco_sub: str,
                   sample_scenes: int) -> dict:
    coco_root = root / coco_sub
    coco = json.loads((coco_root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    print(f"\n--- {split} ({coco_sub}) ---")
    check(bool(images), f"{split}: images present", f"{len(images)}")
    check(bool(cat_ids), f"{split}: categories present", f"{len(cat_ids)}")

    expect = EXPECTED_SPLIT_SIZES.get(split)
    if expect:
        check(len(images) == expect, f"{split}: image count", f"{len(images)} (paper {expect})")

    d, kind = sample_dir(coco_root, split)
    check(d is not None, f"{split}: sample directory exists",
          str(d) if d else f"neither {coco_root/split}/tensors nor /images")

    bad_box = 0
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if not (x >= 0 and y >= 0 and w > 0 and h > 0
                and x + w <= img["width"] + 1e-3 and y + h <= img["height"] + 1e-3
                and ann["category_id"] in cat_ids):
            bad_box += 1
    check(bad_box == 0, f"{split}: boxes in range and labelled", f"{bad_box} bad of {len(coco['annotations'])}")

    # The memmap row is parsed out of the file name, not looked up. Contiguity is the
    # invariant that keeps every scene paired with its own spectrogram.
    off = [i for i, im in enumerate(coco["images"])
           if not Path(im["file_name"]).stem.rsplit("_", 1)[-1].isdigit()
           or int(Path(im["file_name"]).stem.rsplit("_", 1)[-1]) != i]
    check(not off, f"{split}: image ids contiguous from 0",
          f"first offender at position {off[0]}" if off else "")

    mm = root / "memmap" / f"{split}.npy"
    if mm.exists():
        arr = np.load(mm, mmap_mode="r")
        check(arr.shape[0] == len(images), f"{split}: memmap rows == images",
              f"{arr.shape[0]} vs {len(images)}")
        check((mm.parent / f"{split}.npy.done").exists(), f"{split}: memmap .done sentinel")
        print(f"  [info] memmap shape {tuple(arr.shape)} dtype {arr.dtype}")

    meta_path = root / "metadata" / f"{split}.jsonl"
    rows = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    check(len(rows) == len(images), f"{split}: metadata lines == images", f"{len(rows)} vs {len(images)}")

    missing = 0
    for row in rows[:sample_scenes]:
        if resolve_raw(row, raw_root, split) is None:
            missing += 1
    check(missing == 0, f"{split}: raw scenes resolvable (first {sample_scenes})", f"{missing} missing")

    snr, dur, bw, per_img, tones = [], [], [], [], 0
    for row in rows:
        nq, fs = row["num_iq_samples"], row["sample_rate"]
        per_img.append(len(row["instances"]))
        for inst in row["instances"]:
            snr.append(float(inst["snr_db"]))
            dur.append(inst["duration_in_samples"] / nq)
            # An unmodulated carrier has no bandwidth, so `tone` legitimately sits below
            # --bandwidth-min-frac. Excluded from the bandwidth bound and counted instead.
            if str(inst.get("class_name", "")).lower() == "tone":
                tones += 1
            else:
                bw.append(inst["bandwidth"] / fs)
    return {"snr": np.array(snr), "dur": np.array(dur), "bw": np.array(bw),
            "per_img": np.array(per_img), "images": len(images), "tones": tones,
            "boxes": len(coco["annotations"]), "cats": sorted(cat_ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/torchsig_mini",
                    help="dataset root holding coco/ (or coco_multiclass/), metadata/, summary.json")
    ap.add_argument("--coco-sub", default="coco", choices=["coco", "coco_multiclass"])
    ap.add_argument("--raw-root", default=None,
                    help="raw-IQ dataset root, when metadata raw_path is from another machine")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--sample-scenes", type=int, default=64)
    ap.add_argument("--skip-spec", action="store_true",
                    help="only run structural checks, do not compare against the paper spec")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = repo_root() / root
    raw_root = Path(args.raw_root) if args.raw_root else None
    if raw_root is not None and not raw_root.is_absolute():
        raw_root = repo_root() / raw_root

    print(f"=== dataset: {root}")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if not args.skip_spec:
        print("\n--- generator settings vs the paper's configuration ---")
        check(bool(summary), "summary.json present", str(summary_path))
        for key, want in PAPER_SPEC.items():
            got = summary.get(key)
            if got is None:
                check(False, f"summary.{key}", "absent")
            elif isinstance(want, bool):
                check(bool(got) == want, f"summary.{key}", f"{got} (paper {want})")
            else:
                check(math.isclose(float(got), float(want), rel_tol=1e-6),
                      f"summary.{key}", f"{got} (paper {want})")
        stats = summary.get("stft_tensor_stats")
        check(bool(stats) and len(stats.get("mean", [])) == 3,
              "summary.stft_tensor_stats has 3 channels",
              json.dumps(stats)[:90] if stats else "absent -- training would use identity "
                                                   "normalisation on sigma~12.8 data")

    agg = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        agg[split] = validate_split(root, split, raw_root, args.coco_sub, args.sample_scenes)

    print("\n--- observed distributions vs the paper's configuration ---")
    snr = np.concatenate([a["snr"] for a in agg.values()])
    dur = np.concatenate([a["dur"] for a in agg.values()])
    bw = np.concatenate([a["bw"] for a in agg.values()])
    per_img = np.concatenate([a["per_img"] for a in agg.values()])
    if not args.skip_spec:
        check(snr.min() >= PAPER_SPEC["snr_db_min"] - 0.5 and snr.max() <= PAPER_SPEC["snr_db_max"] + 0.5,
              "per-signal SNR span",
              f"[{snr.min():.1f}, {snr.max():.1f}] dB (paper [-20, 10]); a [-10, 40] span means "
              "--snr-buckets defaulted -- see the README")
        check(dur.min() >= PAPER_SPEC["duration_min_frac"] * 0.9
              and dur.max() <= PAPER_SPEC["duration_max_frac"] * 1.1,
              "duration fraction span", f"[{dur.min():.4f}, {dur.max():.4f}] (paper [0.005, 0.25])")
        check(bw.min() >= PAPER_SPEC["bandwidth_min_frac"] * 0.9
              and bw.max() <= PAPER_SPEC["bandwidth_max_frac"] * 1.1,
              "bandwidth fraction span (excluding tones)",
              f"[{bw.min():.4f}, {bw.max():.4f}] (paper [0.0125, 0.49])")
        tone_frac = sum(a["tones"] for a in agg.values()) / max(len(snr), 1)
        check(0.005 <= tone_frac <= 0.05, "unmodulated-tone share",
              f"{tone_frac:.2%} of instances (paper 1.90%); tones carry ~0 bandwidth by "
              "definition and are exempt from the bandwidth bound")
        check(per_img.max() <= PAPER_SPEC["num_signals_max"] and per_img.min() >= PAPER_SPEC["num_signals_min"],
              "signals per scene", f"[{per_img.min()}, {per_img.max()}] (paper [1, 6])")
    print(f"  [info] SNR p1/p50/p99 = {np.percentile(snr, [1, 50, 99]).round(1).tolist()}")
    print(f"  [info] boxes per scene mean = {per_img.mean():.2f}")

    cats = [a["cats"] for a in agg.values()]
    check(all(c == cats[0] for c in cats), "category list identical across splits",
          f"{len(cats[0])} categories, ids {cats[0][0]}..{cats[0][-1]}")

    total_i = sum(a["images"] for a in agg.values())
    total_b = sum(a["boxes"] for a in agg.values())
    print(f"\n=== {total_i} images, {total_b} boxes, {len(FAILURES)} failed check(s)")
    if FAILURES:
        for f in FAILURES:
            print(f"  FAILED: {f}")
        raise SystemExit(1)
    print("=== all checks passed")


if __name__ == "__main__":
    main()
