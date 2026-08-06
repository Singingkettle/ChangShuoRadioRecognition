# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def latest_vis_dir(work_dir: Path) -> Path:
    candidates = sorted([p for p in work_dir.iterdir() if p.is_dir()])
    for candidate in reversed(candidates):
        scalars = candidate / "vis_data" / "scalars.json"
        log_path = candidate / f"{candidate.name}.log"
        if scalars.exists() or log_path.exists():
            return candidate
    raise FileNotFoundError(f"No MMEngine scalars.json or timestamped log found under {work_dir}")


def read_last_eval(scalars_path: Path) -> dict[str, Any]:
    last_eval: dict[str, Any] | None = None
    with scalars_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "coco/bbox_mAP" in row:
                last_eval = row
    if last_eval is None:
        raise RuntimeError(f"No COCO evaluation metrics found in {scalars_path}")
    return last_eval


def read_last_eval_from_log(log_path: Path) -> dict[str, Any]:
    """Parse COCO AP values from a test-only MMEngine log.

    MMEngine training usually writes vis_data/scalars.json, but plain test runs
    may only write the textual log. The COCO "bbox_mAP_copypaste" line is the
    most stable machine-readable record in that case.
    """

    metrics: dict[str, Any] = {}
    value = r"(-?[0-9.]+)"
    copypaste_pattern = re.compile(
        rf"bbox_mAP_copypaste:\s+{value}\s+{value}\s+{value}\s+{value}\s+{value}\s+{value}"
    )
    epoch_pattern = re.compile(
        r"coco/bbox_mAP:\s+([0-9.]+)\s+"
        r"coco/bbox_mAP_50:\s+([0-9.]+)\s+"
        r"coco/bbox_mAP_75:\s+([0-9.]+).*?"
        r"\stime:\s+([0-9.]+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = copypaste_pattern.search(line)
        if match:
            values = [float(value) for value in match.groups()]
            metrics.update(
                {
                    "coco/bbox_mAP": values[0],
                    "coco/bbox_mAP_50": values[1],
                    "coco/bbox_mAP_75": values[2],
                    "coco/bbox_mAP_s": values[3],
                    "coco/bbox_mAP_m": values[4],
                    "coco/bbox_mAP_l": values[5],
                }
            )
        match = epoch_pattern.search(line)
        if match:
            values = [float(value) for value in match.groups()]
            metrics.update(
                {
                    "coco/bbox_mAP": values[0],
                    "coco/bbox_mAP_50": values[1],
                    "coco/bbox_mAP_75": values[2],
                    "time": values[3],
                }
            )
    if not metrics:
        raise RuntimeError(f"No COCO evaluation metrics found in {log_path}")
    return metrics


def read_last_coco_ar(log_path: Path) -> dict[str, float]:
    """Parse COCO AR values that MMEngine prints but does not write to scalars."""

    patterns = {
        "coco/bbox_AR@1": re.compile(r"Average Recall\s+\(AR\).*area=\s*all\s*\|\s*maxDets=\s*1\s*\]\s*=\s*(-?[0-9.]+)"),
        "coco/bbox_AR@10": re.compile(r"Average Recall\s+\(AR\).*area=\s*all\s*\|\s*maxDets=\s*10\s*\]\s*=\s*(-?[0-9.]+)"),
        "coco/bbox_AR@100": re.compile(r"Average Recall\s+\(AR\).*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[0-9.]+)"),
        "coco/bbox_AR_s@100": re.compile(r"Average Recall\s+\(AR\).*area=\s*small\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[0-9.]+)"),
        "coco/bbox_AR_m@100": re.compile(r"Average Recall\s+\(AR\).*area=\s*medium\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[0-9.]+)"),
        "coco/bbox_AR_l@100": re.compile(r"Average Recall\s+\(AR\).*area=\s*large\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[0-9.]+)"),
    }
    values: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                values[key] = float(match.group(1))
    return values


def write_outputs(work_dir: Path, experiment_name: str, *, split_label: str = "val") -> None:
    vis_dir = latest_vis_dir(work_dir)
    scalars_path = vis_dir / "vis_data" / "scalars.json"
    log_path = vis_dir / f"{vis_dir.name}.log"
    metric_note = ""
    metrics: dict[str, Any] = {}
    if scalars_path.exists():
        try:
            metrics = read_last_eval(scalars_path)
        except RuntimeError as exc:
            metric_note = f"Scalars did not contain COCO metrics: {exc}"
    if not metrics and log_path.exists():
        try:
            metrics = read_last_eval_from_log(log_path)
        except RuntimeError as exc:
            metric_note = f"{metric_note}; log did not contain COCO metrics: {exc}".strip("; ")
    if not metrics:
        if not scalars_path.exists() and not log_path.exists():
            raise FileNotFoundError(f"No scalars or log metrics found in {vis_dir}")
        metrics = {
            "coco/bbox_mAP": 0.0,
            "coco/bbox_mAP_50": 0.0,
            "coco/bbox_mAP_75": 0.0,
            "coco/bbox_mAP_s": 0.0,
            "coco/bbox_mAP_m": 0.0,
            "coco/bbox_mAP_l": 0.0,
            "time": 0.0,
        }
        metric_note = metric_note or "No COCO metrics were emitted; treating this smoke run as zero-detection output."
    log_metrics: dict[str, float] = {}
    if log_path.exists():
        log_metrics = read_last_coco_ar(log_path)
    missing = float("nan")
    source_dir = work_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        (split_label, "bbox_mAP", metrics.get("coco/bbox_mAP", 0.0)),
        (split_label, "bbox_mAP_50", metrics.get("coco/bbox_mAP_50", 0.0)),
        (split_label, "bbox_mAP_75", metrics.get("coco/bbox_mAP_75", 0.0)),
        (split_label, "bbox_mAP_s", metrics.get("coco/bbox_mAP_s", 0.0)),
        (split_label, "bbox_mAP_m", metrics.get("coco/bbox_mAP_m", 0.0)),
        (split_label, "bbox_mAP_l", metrics.get("coco/bbox_mAP_l", 0.0)),
        (split_label, "bbox_AR_1", metrics.get("coco/bbox_AR@1", log_metrics.get("coco/bbox_AR@1", missing))),
        (split_label, "bbox_AR_10", metrics.get("coco/bbox_AR@10", log_metrics.get("coco/bbox_AR@10", missing))),
        (split_label, "bbox_AR_100", metrics.get("coco/bbox_AR@100", log_metrics.get("coco/bbox_AR@100", missing))),
        (split_label, "bbox_AR_s_100", metrics.get("coco/bbox_AR_s@100", log_metrics.get("coco/bbox_AR_s@100", missing))),
        (split_label, "bbox_AR_m_100", metrics.get("coco/bbox_AR_m@100", log_metrics.get("coco/bbox_AR_m@100", missing))),
        (split_label, "bbox_AR_l_100", metrics.get("coco/bbox_AR_l@100", log_metrics.get("coco/bbox_AR_l@100", missing))),
        (split_label, "eval_time_per_image", metrics.get("time", 0.0)),
    ]
    for output in (work_dir / "summary.csv", source_dir / "figure2_stft_rtmdet_metrics.csv"):
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "metric", "value"])
            writer.writerows(rows)

    ar_100 = metrics.get("coco/bbox_AR@100", log_metrics.get("coco/bbox_AR@100", missing))
    ar_100_text = "not captured" if ar_100 != ar_100 else f"{ar_100:.4f}"

    report = f"""# {experiment_name} Report

## Purpose

This run is an STFT+RTMDet baseline check on TorchSig scenes exported to COCO
spectrogram annotations. For formal manuscript use, confirm the run used the
same raw IQ scenes as IQDet and a pre-registered training budget rather than the
one-epoch smoke configuration.

## Run

- Work directory: `{work_dir.as_posix()}`
- MMEngine run directory: `{vis_dir.as_posix()}`
- Source data: `{source_dir.as_posix()}`

## Summary Metrics

| Split | bbox mAP | bbox mAP 50 | bbox mAP 75 | bbox AR@100 | eval time/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| {split_label} | {metrics.get("coco/bbox_mAP", 0.0):.4f} | {metrics.get("coco/bbox_mAP_50", 0.0):.4f} | {metrics.get("coco/bbox_mAP_75", 0.0):.4f} | {ar_100_text} | {metrics.get("time", 0.0):.4f} |

## Manuscript Use

Use this report as baseline evidence only when the corresponding run_info.json
confirms the intended training budget and STFT window. Multi-window STFT variants
are required before any manuscript claim about front-end window sensitivity.

Metric note: {metric_note or "none"}.
"""
    report_path = ROOT / "paper" / "iteration_reports" / f"{experiment_name}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[mmdet-summary] wrote {work_dir / 'summary.csv'}")
    print(f"[mmdet-summary] wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--experiment-name", default="stft_rtmdet_pilot")
    parser.add_argument("--split-label", default="val")
    args = parser.parse_args()
    write_outputs((ROOT / args.work_dir).resolve(), args.experiment_name, split_label=args.split_label)


if __name__ == "__main__":
    main()
