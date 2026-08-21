# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Collect every finished detector run into one auditable table.

A reproduction is only worth as much as the paper trail behind it, so this walks a tree
of run directories and pulls, per run: the metrics the harness wrote, the literal command
line, the git commit the code was on, the wall clock, and whether the mmcv-lite fallbacks
were active. Runs of the same cell are then pooled across seeds, because a single seed
cannot settle a 0.01 difference on this benchmark.

    python configs/detection_is_easy/collect_repro_results.py --root work_dirs/repro \
        --out reports/repro_cells.csv --markdown reports/repro_cells.md

Cells are matched to the paper by the `paper_cell` string the campaign records in
`job.json`; any "(paper 0.447)" in that string becomes the reference value. `--tolerance`
is the band inside which a cell counts as reproduced, and should be set from a measured
same-seed repeat rather than guessed -- see `--help`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

# Measured on this benchmark by running one identical configuration three times:
# RTMDet-M sd = 0.0076, RTMDet-tiny sd = 0.0033. Three sigma of the larger one is the
# smallest difference a single pair of runs can carry.
DEFAULT_TOLERANCE = 0.023

PAPER_VALUE = re.compile(r"paper\s+([0-9]*\.?[0-9]+)")
# `bu_m_uniform_s17` and `bo_m_own_s20262811_bs4` differ only by seed and by the control
# they carry, so drop the seed wherever it sits and keep the rest of the name.
SEED_TOKEN = re.compile(r"_s\d+")
REPEAT_SUFFIX = re.compile(r"_r\d+$")

# Runs the campaign named separately that are the same cell. The batch-8 deployment run
# (`deploy_m_own`) is NOT pooled into `bo_m_own` any more: the paper now reports medium/own
# as the batch-4 three-seed value, so pooling the batch-8 run would mix two batch sizes.
ALSO_IN = {
    "floor_m_uniform": ["bu_m_uniform"],
    "floor_tiny_uniform": ["bu_tiny_uniform"],
}


def repo_root() -> Path:
    _p = Path(__file__).resolve()
    for _up in [_p, *_p.parents]:
        if (_up / "tools" / "train.py").exists() and (_up / "csrr").is_dir():
            return _up
    raise RuntimeError("CSRR repo root not found above " + str(_p))


def read_summary(path: Path) -> dict[str, float]:
    """`summary.csv` is `split,metric,value`; keep it keyed as `split/metric`."""
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[0] == "split":
                continue
            try:
                out[f"{row[0]}/{row[1]}"] = float(row[2])
            except ValueError:
                continue
    return out


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def hours_between(started: str, ended: str) -> float | None:
    from datetime import datetime

    try:
        return (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def cell_key(job: str, paper_cell: str) -> str:
    """Everything but the seed. `bu_m_uniform_s17` and `bu_m_uniform_s27` are one cell."""
    stripped = REPEAT_SUFFIX.sub("", SEED_TOKEN.sub("", job))
    return stripped or paper_cell


def cells_of(job: str, paper_cell: str) -> list[str]:
    primary = cell_key(job, paper_cell)
    return [primary] + ALSO_IN.get(primary, [])


def read_reference(path: Path) -> dict[str, dict]:
    """Paper values, one row per cell, so the comparison cites a table rather than memory."""
    if not path.exists():
        return {}
    reference = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell = (row.get("cell") or "").strip()
            if not cell:
                continue
            try:
                row["paper_value"] = float(row["paper_value"])
            except (TypeError, ValueError, KeyError):
                continue
            reference[cell] = row
    return reference


def flag(value) -> str:
    return "" if value is None else ("yes" if value else "no")


def collect(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.rglob("summary.csv")):
        run_dir = summary_path.parent
        metrics = read_summary(summary_path)
        job = read_json(run_dir / "job.json")
        info = read_json(run_dir / "run_info.json")
        args = info.get("args", {}) or {}
        name = job.get("job") or run_dir.name
        # The harness reports val during training and test only when asked; take whichever
        # class-aware mAP the run actually produced, and say which split it came from.
        split = "test" if "test/bbox_mAP" in metrics else "val"
        runs.append({
            "job": name,
            "cell": cell_key(name, job.get("paper_cell", "")),
            "group": job.get("group", ""),
            "paper_cell": job.get("paper_cell", ""),
            "seed": job.get("seed", args.get("seed", "")),
            "split": split,
            "bbox_mAP": metrics.get(f"{split}/bbox_mAP"),
            "bbox_mAP_50": metrics.get(f"{split}/bbox_mAP_50"),
            "bbox_mAP_75": metrics.get(f"{split}/bbox_mAP_75"),
            "bbox_mAP_s": metrics.get(f"{split}/bbox_mAP_s"),
            "config": args.get("config", ""),
            "epochs": args.get("epochs", ""),
            "batch_size": args.get("batch_size", ""),
            "optimizer": args.get("optimizer", ""),
            "lr": args.get("lr", ""),
            "hours": hours_between(job.get("started", ""), job.get("ended", "")),
            "git_commit": job.get("git_commit", ""),
            "git_dirty": flag(job.get("git_dirty")),
            "mmcv_lite_stub": flag(info.get("used_mmcv_lite_stub")),
            "pytorch_focal_loss": flag(info.get("used_pytorch_focal_loss")),
            "run_dir": str(run_dir.relative_to(root.parent.parent)) if root.parent.parent in run_dir.parents else str(run_dir),
            "argv": " ".join(info.get("argv", [])),
        })
    return runs


def pool(runs: list[dict], reference: dict[str, dict]) -> list[dict]:
    """One row per cell: the seed mean, its spread, and the paper value to beat.

    Repeats of one seed (the same-seed runs that measure the non-determinism floor) are
    averaged before pooling, so a seed that happens to have been run three times does not
    outvote the other two.
    """
    cells: dict[str, list[dict]] = {}
    for run in runs:
        if run["bbox_mAP"] is None:
            continue
        for cell in cells_of(run["job"], run["paper_cell"]):
            cells.setdefault(cell, []).append(run)

    pooled = []
    for cell, members in sorted(cells.items()):
        by_seed: dict[str, list[float]] = {}
        for member in members:
            by_seed.setdefault(str(member["seed"]), []).append(member["bbox_mAP"])
        values = [statistics.fmean(v) for _, v in sorted(by_seed.items())]
        paper_cell = next((m["paper_cell"] for m in members if m["paper_cell"]), "")
        ref = reference.get(cell)
        originally = ""
        if ref is not None:
            paper = ref["paper_value"]
            source = ref.get("source", "")
            # The manuscript's current value was itself corrected against this reproduction,
            # so the first-submission value is carried alongside to keep the record honest.
            originally = (ref.get("originally_reported") or "").strip()
        else:  # fall back to the "(paper 0.447)" the campaign wrote into the job file
            match = PAPER_VALUE.search(paper_cell)
            paper = float(match.group(1)) if match else None
            source = "job file" if paper is not None else ""
        mean = statistics.fmean(values)
        pooled.append({
            "cell": cell,
            "paper_cell": paper_cell,
            "n_seeds": len(values),
            "n_runs": len(members),
            "seeds": ";".join(sorted(by_seed)),
            "repro_mean": mean,
            "repro_sd": statistics.stdev(values) if len(values) > 1 else None,
            "repro_min": min(values),
            "repro_max": max(values),
            "paper": paper,
            "paper_source": source,
            "originally_reported": originally,
            "delta": (mean - paper) if paper is not None else None,
        })
    return pooled


def verdict(row: dict, tolerance: float) -> str:
    if row["paper"] is None:
        return "no paper value"
    if row["delta"] is None or math.isnan(row["delta"]):
        return "?"
    return "within" if abs(row["delta"]) <= tolerance else "OUTSIDE"


def write_markdown(path: Path, pooled: list[dict], runs: list[dict], tolerance: float) -> None:
    lines = [
        "# Reproduction: measured vs reported",
        "",
        f"{len(runs)} runs pooled into {len(pooled)} cells. A cell counts as reproduced when the",
        f"seed mean sits within {tolerance:.3f} of the paper value -- three times the largest",
        "same-seed standard deviation measured on this benchmark (RTMDet-M, 0.0076).",
        "",
        "| cell | seeds | reproduced | sd | paper (current) | originally reported | delta | verdict | paper source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in pooled:
        sd = f"{row['repro_sd']:.4f}" if row["repro_sd"] is not None else "--"
        paper = f"{row['paper']:.3f}" if row["paper"] is not None else "--"
        delta = f"{row['delta']:+.3f}" if row["delta"] is not None else "--"
        lines.append(
            f"| {row['cell']} | {row['n_seeds']} | {row['repro_mean']:.3f} | {sd} | "
            f"{paper} | {row.get('originally_reported') or '--'} | {delta} | "
            f"{verdict(row, tolerance)} | {row['paper_source'] or '--'} |"
        )
    outside = [r for r in pooled if verdict(r, tolerance) == "OUTSIDE"]
    lines += ["", f"{len(outside)} of {len([r for r in pooled if r['paper'] is not None])} "
                  "cells with a paper value fall outside the band."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="work_dirs/repro", help="tree of run directories to walk")
    ap.add_argument("--out", default="reports/repro_cells.csv", help="per-run CSV")
    ap.add_argument("--pooled-out", default=None, help="per-cell CSV (default: --out with _pooled)")
    ap.add_argument("--markdown", default=None, help="also write a comparison table")
    ap.add_argument("--reference", default="configs/detection_is_easy/paper_values.csv",
                    help="per-cell paper values, each citing the table it came from")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="band for calling a cell reproduced; set it from a measured "
                         f"same-seed repeat, not from taste (default {DEFAULT_TOLERANCE})")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = repo_root() / root
    runs = collect(root)
    if not runs:
        raise SystemExit(f"no summary.csv under {root}")
    ref_path = Path(args.reference)
    if not ref_path.is_absolute():
        ref_path = repo_root() / ref_path
    pooled = pool(runs, read_reference(ref_path))

    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)

    pooled_out = Path(args.pooled_out) if args.pooled_out else out.with_name(out.stem + "_pooled.csv")
    if not pooled_out.is_absolute():
        pooled_out = repo_root() / pooled_out
    with pooled_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled[0].keys()) + ["verdict"])
        writer.writeheader()
        for row in pooled:
            writer.writerow({**row, "verdict": verdict(row, args.tolerance)})

    if args.markdown:
        md = Path(args.markdown)
        if not md.is_absolute():
            md = repo_root() / md
        md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(md, pooled, runs, args.tolerance)

    scored = [r for r in pooled if r["paper"] is not None]
    outside = [r for r in scored if verdict(r, args.tolerance) == "OUTSIDE"]
    print(f"{len(runs)} runs -> {len(pooled)} cells; {len(scored)} carry a paper value, "
          f"{len(outside)} fall outside +-{args.tolerance:.3f}")
    for row in outside:
        print(f"  OUTSIDE {row['cell']}: {row['repro_mean']:.3f} vs paper {row['paper']:.3f} "
              f"({row['delta']:+.3f}, n={row['n_seeds']})")
    print(f"wrote {out}\nwrote {pooled_out}")


if __name__ == "__main__":
    main()
