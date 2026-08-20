# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    _p = Path(__file__).resolve()
    for _up in [_p, *_p.parents]:
        if (_up / "tools" / "train.py").exists() and (_up / "csrr").is_dir():
            return _up
    raise RuntimeError("CSRR repo root not found above " + str(_p))


ROOT = repo_root()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_split_limits(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}
    limits: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid split limit {item!r}; expected split:count")
        split, count = item.split(":", 1)
        limits[split.strip()] = int(count)
    return limits


def limit_coco_annotations(ann: dict, limit: int | None) -> dict:
    if limit is None:
        return ann
    images = ann.get("images", [])[: max(0, int(limit))]
    image_ids = {image["id"] for image in images}
    out = dict(ann)
    out["images"] = images
    out["annotations"] = [item for item in ann.get("annotations", []) if item.get("image_id") in image_ids]
    return out


def copy_coco_shell(src_coco: Path, out_coco: Path, splits: list[str], split_limits: dict[str, int] | None = None) -> dict:
    split_limits = split_limits or {}
    split_stats = {}
    for split in splits:
        ann = load_json(src_coco / "annotations" / f"instances_{split}.json")
        ann = limit_coco_annotations(ann, split_limits.get(split))
        write_json(out_coco / "annotations" / f"instances_{split}.json", ann)
        (out_coco / split / "tensors").mkdir(parents=True, exist_ok=True)
        src_meta = src_coco.parent / "metadata" / f"{split}.jsonl"
        if src_meta.exists():
            dst_meta = out_coco.parent / "metadata" / f"{split}.jsonl"
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_meta, dst_meta)
        split_stats[split] = {"images": len(ann.get("images", [])), "boxes": len(ann.get("annotations", []))}
    return split_stats


def _pack_chunk(
    *,
    kind: str,
    split: str,
    names: list[str],
    start: int,
    stop: int,
    src_dir: str,
    out_path: str,
    shape: tuple[int, ...],
    dtype: str,
) -> tuple[int, int]:
    out_dtype = np.dtype(dtype)
    out = np.lib.format.open_memmap(out_path, mode="r+", dtype=out_dtype, shape=shape)
    root = Path(src_dir)
    for index in range(start, stop):
        arr = np.load(root / names[index], mmap_mode="r")
        out[index] = arr.astype(out_dtype, copy=False)
    out.flush()
    return start, stop


def _chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _pack_array_files(
    *,
    kind: str,
    split: str,
    names: list[str],
    src_dir: Path,
    out_path: Path,
    shape: tuple[int, ...],
    dtype: str,
    workers: int,
    chunk_size: int,
) -> None:
    done_path = out_path.with_suffix(out_path.suffix + ".done")
    if done_path.exists():
        print(f"[pack-memmap] skip completed {kind} {split}: {done_path}", flush=True)
        return

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.dtype(dtype), shape=shape)
    out.flush()
    del out

    total = len(names)
    workers = max(1, int(workers))
    chunk_size = max(1, int(chunk_size))
    ranges = _chunk_ranges(total, chunk_size)
    if workers == 1:
        completed = 0
        for start, stop in ranges:
            _pack_chunk(
                kind=kind,
                split=split,
                names=names,
                start=start,
                stop=stop,
                src_dir=str(src_dir),
                out_path=str(out_path),
                shape=shape,
                dtype=dtype,
            )
            completed = stop
            print(f"[pack-memmap] {kind} {split}: {completed}/{total}", flush=True)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _pack_chunk,
                    kind=kind,
                    split=split,
                    names=names,
                    start=start,
                    stop=stop,
                    src_dir=str(src_dir),
                    out_path=str(out_path),
                    shape=shape,
                    dtype=dtype,
                )
                for start, stop in ranges
            ]
            for future in as_completed(futures):
                _, stop = future.result()
                completed += chunk_size
                completed = min(completed, total)
                print(f"[pack-memmap] {kind} {split}: completed chunk ending {stop}; approx {completed}/{total}", flush=True)

    done_path.write_text(json.dumps({"samples": total, "shape": list(shape)}, indent=2), encoding="utf-8")


def pack_tensor(
    src_coco: Path,
    out_root: Path,
    split: str,
    *,
    dtype: str,
    workers: int,
    chunk_size: int,
    limit: int | None = None,
) -> tuple[int, tuple[int, ...]]:
    ann = load_json(src_coco / "annotations" / f"instances_{split}.json")
    ann = limit_coco_annotations(ann, limit)
    images = ann.get("images", [])
    if not images:
        raise ValueError(f"No images for split {split}")
    # The memmap row for a scene is NOT looked up -- LoadTensorMemmapFromCOCOStem parses it
    # out of the trailing integer in the file name (mmdet_plugins._sample_index_from_path),
    # while this packer writes row i from images[i]. The two agree only while the ids run
    # 0..N-1 in order. A filtered, reordered or sharded corpus silently pairs every scene
    # with someone else's spectrogram: training still converges, mAP is just wrong.
    for position, image in enumerate(images):
        stem = Path(image["file_name"]).stem
        try:
            index = int(stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(
                f"{split}: cannot parse a sample index out of '{image['file_name']}'. "
                "The memmap loader needs the trailing integer to find the row."
            ) from exc
        if index != position:
            raise ValueError(
                f"{split}: image at position {position} is '{image['file_name']}' whose "
                f"trailing index is {index}. The memmap loader derives the row from that "
                "index, so a non-contiguous or reordered image list would pair every scene "
                "with the wrong tensor. Re-export with contiguous ids from zero."
            )
    src_dir = src_coco / split / "tensors"
    first = np.load(src_dir / images[0]["file_name"], mmap_mode="r")
    shape = (len(images),) + tuple(first.shape)
    _pack_array_files(
        kind="tensor",
        split=split,
        names=[image["file_name"] for image in images],
        src_dir=src_dir,
        out_path=out_root / "memmap" / f"{split}.npy",
        shape=shape,
        dtype=dtype,
        workers=workers,
        chunk_size=chunk_size,
    )
    return len(images), shape


def pack_raw_complex(
    src_coco: Path,
    raw_cache_root: Path,
    out_root: Path,
    split: str,
    *,
    workers: int,
    chunk_size: int,
    limit: int | None = None,
) -> tuple[int, tuple[int, ...]]:
    ann = load_json(src_coco / "annotations" / f"instances_{split}.json")
    ann = limit_coco_annotations(ann, limit)
    images = ann.get("images", [])
    if not images:
        raise ValueError(f"No images for split {split}")
    first = np.load(raw_cache_root / split / images[0]["file_name"], mmap_mode="r")
    shape = (len(images),) + tuple(first.shape)
    _pack_array_files(
        kind="raw",
        split=split,
        names=[image["file_name"] for image in images],
        src_dir=raw_cache_root / split,
        out_path=out_root / "memmap" / f"{split}.npy",
        shape=shape,
        dtype="complex64",
        workers=workers,
        chunk_size=chunk_size,
    )
    return len(images), shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack COCO tensor datasets into split-level .npy memmaps.")
    parser.add_argument(
        "--kind",
        choices=["tensor", "stft2", "raw_complex"],
        required=True,
        help="Use 'tensor' for any precomputed [C,F,T] COCO tensor dataset. 'stft2' is a backward-compatible alias.",
    )
    parser.add_argument("--src-coco", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--dtype", default="float32", help="STFT memmap dtype; raw_complex always uses complex64")
    parser.add_argument("--raw-cache-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel loader/writer workers per split.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Samples written per worker task.")
    parser.add_argument(
        "--split-limits",
        default=None,
        help="Optional comma-separated split:count limits, e.g. train:10000,val:2048. "
        "The output COCO annotations and memmaps are both truncated consistently.",
    )
    args = parser.parse_args()

    src_coco = (ROOT / args.src_coco).resolve()
    out_root = (ROOT / args.out_root).resolve()
    out_coco = out_root / "coco"
    if args.force and out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "memmap").mkdir(parents=True, exist_ok=True)
    out_coco.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    split_limits = parse_split_limits(args.split_limits)
    split_stats = copy_coco_shell(src_coco, out_coco, splits, split_limits)

    packed = {}
    for split in splits:
        limit = split_limits.get(split)
        if args.kind in {"tensor", "stft2"}:
            count, shape = pack_tensor(
                src_coco,
                out_root,
                split,
                dtype=args.dtype,
                workers=args.workers,
                chunk_size=args.chunk_size,
                limit=limit,
            )
        else:
            if not args.raw_cache_root:
                raise ValueError("--raw-cache-root is required for raw_complex")
            count, shape = pack_raw_complex(
                src_coco,
                (ROOT / args.raw_cache_root).resolve(),
                out_root,
                split,
                workers=args.workers,
                chunk_size=args.chunk_size,
                limit=limit,
            )
        packed[split] = {"samples": count, "shape": list(shape), "path": str(out_root / "memmap" / f"{split}.npy")}

    source_summary = load_json(src_coco.parent / "summary.json") if (src_coco.parent / "summary.json").exists() else {}
    summary = dict(source_summary)
    summary.update(
        {
            "source_coco": str(src_coco),
            "out_root": str(out_root),
            "format": f"{args.kind}_split_memmap_coco",
            "memmap": packed,
            "splits": split_stats,
            "split_limits": split_limits,
        }
    )
    write_json(out_root / "summary.json", summary)
    print(f"[pack-memmap] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
