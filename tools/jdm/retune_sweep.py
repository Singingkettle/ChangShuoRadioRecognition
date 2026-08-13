"""JDM hyperparameter retune orchestrator.

ARCHITECTURE FREEZE POLICY
--------------------------
Retunes must NOT change JDM network architecture (detector/AMC backbone depth,
channel widths, head layer topology). Must match paper / configs/jdm/ baseline.
Allowed: init, lr/wd, schedulers, ES, batch size, grad clip, loss weights,
anchor_widths (hyperparams), fuse_scores (inference), training epoch budget.
See docs/csrd_jointdet/retune_campaign.md § Architecture freeze.

Permitted --cfg-options keys (whitelist for manifest ``cfg_options`` and CLI):
  model.head.anchor_widths         # e.g. (96,120,146) vs (110,130,150)
  model.head.loss_bw.loss_weight   # bandwidth MSE weight (20 vs 2)
  model.head.loss_center.loss_weight
  model.backbone.init_cfg
  model.fuse_scores                # inference-only score fusion
  optim_wrapper.optimizer.lr
  optim_wrapper.optimizer.weight_decay
  optim_wrapper.clip_grad
  param_scheduler
  train_cfg.max_epochs
  custom_hooks
  train_dataloader.batch_size

Forbidden via --cfg-options: backbone/head layer counts, channel dims, stride,
anchor scale count, conv kernel sizes, replacing submodule types.

Runs detector / AMC experiments from a JSON manifest, optionally audits the
CSRD dataset scale, and appends results to ``docs/csrd_jointdet/retune_results.md``.

Usage::

    # Dataset audit only
    python tools/jdm/retune_sweep.py --audit-dataset

    # Dry-run wave 1
    python tools/jdm/retune_sweep.py \\
        --manifest configs/jdm/experiments/retune/wave1_manifest.json --dry-run

    # Run P0 detector retunes on two GPUs
    python tools/jdm/retune_sweep.py \\
        --manifest configs/jdm/experiments/retune/wave1_manifest.json \\
        --gpu 0,1 --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from goal_mode_helpers import (  # noqa: E402
    evaluate_jdm_goal,
    jdm_goal_checklist,
    load_json,
    parse_jdm_metrics_json,
    print_jdm_goal_status,
    write_goal_status,
)

RESULTS_PATH = _REPO_ROOT / "docs" / "csrd_jointdet" / "retune_results.md"
DEFAULT_RETUNE_ROOT = _REPO_ROOT / "work_dirs" / "jdm" / "retune"
DEFAULT_GOALS_PATH = _REPO_ROOT / "configs" / "jdm" / "retune" / "goals.json"
GOAL_STATUS_PATH = DEFAULT_RETUNE_ROOT / "GOAL_STATUS.json"

DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "ChangShuoTwc2026"

_LOG = logging.getLogger("jdm.retune")


@dataclass
class JDMExperiment:
    experiment_id: str
    module: str
    variant: str
    config: Path
    cfg_options: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    priority: int = 99

    @property
    def label(self) -> str:
        return f"{self.module}/{self.variant}"


def _cfg_options_to_cli(opts: dict[str, Any]) -> list[str]:
    flat: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else k
                walk(key, v)
        else:
            flat.append(f"{prefix}={value}")

    for key, value in opts.items():
        walk(key, value)
    return flat


def _load_manifest(path: Path) -> list[JDMExperiment]:
    data = json.loads(path.read_text())
    exps: list[JDMExperiment] = []
    for raw in data.get("experiments", []):
        cfg = _REPO_ROOT / raw["config"]
        exps.append(
            JDMExperiment(
                experiment_id=raw.get("id", raw["variant"]),
                module=raw.get("module", "detector"),
                variant=raw["variant"],
                config=cfg,
                cfg_options=raw.get("cfg_options", {}),
                notes=raw.get("notes", ""),
                priority=raw.get("priority", 99),
            ))
    exps.sort(key=lambda e: (e.priority, e.module, e.variant))
    return exps


def _find_best_checkpoint(work_dir: Path) -> Path | None:
    cands = sorted(work_dir.glob("best_*.pth"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return cands[0] if cands else None


def _run_cmd(cmd: list[str], env: dict[str, str], log_path: Path,
             dry_run: bool) -> int:
    line = " ".join(shlex.quote(c) for c in cmd)
    _LOG.info("CMD: %s", line)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n")
        fh.write(line + "\n")
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        return proc.returncode


def audit_dataset(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    """Return dataset scale statistics for ``dataset_scale_audit.md`` updates."""
    import random
    from collections import Counter

    import numpy as np

    if not data_root.is_dir():
        raise SystemExit(f"Data root missing: {data_root}")

    versions = sorted(
        (d for d in os.listdir(data_root)
         if d.startswith("v") and (data_root / d).is_dir()),
        key=lambda d: int(d[1:]),
    )
    sig_counts: Counter[int] = Counter()
    bws: list[float] = []
    frame_len = 1200

    for v in versions:
        anno_dir = data_root / v / "anno"
        for fn in os.listdir(anno_dir):
            anno = json.loads((anno_dir / fn).read_text())
            sig_counts[len(anno["modulation"])] += 1
            sr = anno["sample_rate"][0]
            for cf, bw in zip(anno["center_frequency"], anno["bandwidth"]):
                bws.append(((cf + bw / 2) / sr + 0.5 - (cf - bw / 2) / sr -
                            0.5) * frame_len)

    total_frames = sum(sig_counts.values())
    total_signals = sum(k * v for k, v in sig_counts.items())
    bws_arr = np.array(bws)

    splits = {}
    for split in ("train", "validation", "test"):
        nf = 0
        ns = 0
        for v in versions:
            files = sorted(os.listdir(data_root / v / "anno"))
            n = len(files)
            idx = list(range(n))
            random.Random(0).shuffle(idx)
            nt, nv = int(0.5 * n), int(0.1 * n)
            if split == "train":
                sel = idx[:nt]
            elif split == "validation":
                sel = idx[nt:nt + nv]
            else:
                sel = idx[nt + nv:]
            nf += len(sel)
            for i in sel:
                anno = json.loads(
                    (data_root / v / "anno" / files[i]).read_text())
                ns += len(anno["modulation"])
        splits[split] = {"frames": nf, "signals": ns}

    stats = {
        "data_root": str(data_root),
        "versions": len(versions),
        "frames_per_version": 1000,
        "total_frames": total_frames,
        "total_signals": total_signals,
        "avg_signals_per_frame": total_signals / total_frames,
        "signal_count_histogram": dict(sorted(sig_counts.items())),
        "splits_50_10_40": splits,
        "bandwidth_bins_mean": float(bws_arr.mean()),
        "bandwidth_small_pct": float(
            ((bws_arr >= 0) & (bws_arr < 110)).mean() * 100),
        "bandwidth_medium_pct": float(
            ((bws_arr >= 110) & (bws_arr < 130)).mean() * 100),
        "bandwidth_large_pct": float((bws_arr >= 130).mean() * 100),
    }
    return stats


def _print_audit(stats: dict[str, Any]) -> None:
    print(json.dumps(stats, indent=2))


def _run_experiment(exp: JDMExperiment, gpu: int,
                    args: argparse.Namespace) -> dict[str, Any]:
    work_dir = DEFAULT_RETUNE_ROOT / exp.variant
    log_path = work_dir / "retune.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cfg_cli = _cfg_options_to_cli(exp.cfg_options)
    cfg_args = ["--cfg-options", *cfg_cli] if cfg_cli else []

    row: dict[str, Any] = {
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "id": exp.experiment_id,
        "variant": exp.variant,
        "module": exp.module,
        "status": "pending",
        "notes": exp.notes,
        "work_dir": str(work_dir.relative_to(_REPO_ROOT)),
    }

    best = _find_best_checkpoint(work_dir) if work_dir.is_dir() else None
    needs_train = not args.skip_train and (args.force or best is None)
    if best and not args.force and not args.skip_train:
        _LOG.info("[%s] reusing %s", exp.label, best.name)
        needs_train = False

    if needs_train:
        train_cmd = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "train.py"),
            str(exp.config),
            "--work-dir",
            str(work_dir),
            *cfg_args,
        ]
        rc = _run_cmd(train_cmd, env, log_path, args.dry_run)
        if rc != 0 and not args.dry_run:
            row["status"] = f"train_error({rc})"
            return row
        best = _find_best_checkpoint(work_dir)

    if args.dry_run:
        row["status"] = "dry-run"
        return row

    if best is None:
        row["status"] = "no_checkpoint"
        return row

    row["checkpoint"] = str(best.relative_to(_REPO_ROOT))

    if not args.skip_test:
        if exp.module == "detector":
            test_cmd = [
                sys.executable,
                str(_REPO_ROOT / "tools" / "test_det.py"),
                str(exp.config),
                str(best),
                "--work-dir",
                str(work_dir),
                *cfg_args,
            ]
        else:
            test_cmd = [
                sys.executable,
                str(_REPO_ROOT / "tools" / "test.py"),
                str(exp.config),
                str(best),
                "--work-dir",
                str(work_dir),
                *cfg_args,
            ]
        rc = _run_cmd(test_cmd, env, log_path, False)
        row["status"] = "done" if rc == 0 else f"test_error({rc})"
        if rc == 0 and args.goals.is_file():
            metrics = parse_jdm_metrics_json(work_dir, exp.module)
            row["metrics"] = metrics
            goal_eval = evaluate_jdm_goal(load_json(args.goals), exp.module, metrics)
            row["goal_met"] = goal_eval.get("goal_met", False)
            row["goal_checks"] = goal_eval.get("checks", [])
    else:
        row["status"] = "trained"

    return row


def _append_results(rows: list[dict[str, Any]], dry_run: bool,
                    goal_mode: bool = False) -> None:
    if not rows:
        return
    if goal_mode:
        header = "| When | ID | Module | Variant | Status | goal_met | Metrics | Notes |"
        sep = "|---|" * 7
    else:
        header = "| When | ID | Module | Variant | Status | Checkpoint | Notes |"
        sep = "|---|" * 7
    lines = [header, sep]
    for r in rows:
        if goal_mode:
            metrics = r.get("metrics") or {}
            metric_s = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) and v < 10
                                 else f"{k}={v}" for k, v in metrics.items()) or "—"
            lines.append(
                f"| {r['when']} | {r['id']} | {r['module']} | `{r['variant']}` "
                f"| {r['status']} | `{r.get('goal_met', '—')}` | {metric_s} | {r['notes']} |")
        else:
            lines.append(
                f"| {r['when']} | {r['id']} | {r['module']} | `{r['variant']}` "
                f"| {r['status']} | `{r.get('checkpoint', '—')}` | {r['notes']} |")
    block = "\n".join(lines) + "\n"
    if dry_run:
        print("[DRY-RUN] retune_results.md append:\n", block)
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.is_file():
        RESULTS_PATH.write_text(
            "# JDM Retune Results\n\n"
            "Append-only log from ``tools/jdm/retune_sweep.py``.\n\n")
    with RESULTS_PATH.open("a") as fh:
        fh.write(f"\n## {rows[0]['when']}\n\n")
        fh.write(block)


class _GpuPool:
    def __init__(self, gpus: list[int]):
        import threading
        self._gpus = gpus
        self._lock = threading.Lock()
        self._free = list(gpus)

    def acquire(self) -> int:
        import time
        while True:
            with self._lock:
                if self._free:
                    return self._free.pop()
            time.sleep(2)

    def release(self, gpu: int) -> None:
        with self._lock:
            self._free.append(gpu)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, help="JSON experiment manifest.")
    p.add_argument("--audit-dataset", action="store_true",
                   help="Print dataset scale JSON and exit.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--gpu", default="0", help="CUDA device id(s), comma-sep.")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument(
        "--goal-mode",
        action="store_true",
        help="Loop manifest until active goals met (see docs/csrd_jointdet/goal_mode.md).",
    )
    p.add_argument(
        "--until-pass",
        action="store_true",
        help="Stop at first passing variant per module when applicable.",
    )
    p.add_argument(
        "--stop-when-all-pass",
        action="store_true",
        help="Stop when all active goals in goals.json are met.",
    )
    p.add_argument(
        "--goal-status",
        action="store_true",
        help="Print goal checklist; no training.",
    )
    p.add_argument(
        "--goals",
        type=Path,
        default=DEFAULT_GOALS_PATH,
        help="Goal thresholds JSON.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    args.goals = Path(args.goals).resolve()
    if args.until_pass or args.stop_when_all_pass:
        args.goal_mode = True

    if args.goal_status:
        print_jdm_goal_status(args.goals, DEFAULT_RETUNE_ROOT)
        status = jdm_goal_checklist(args.goals, DEFAULT_RETUNE_ROOT)
        write_goal_status(GOAL_STATUS_PATH, status, dry_run=False)
        return 0

    if args.audit_dataset:
        stats = audit_dataset(args.data_root.resolve())
        _print_audit(stats)
        return 0

    if not args.manifest:
        _LOG.error("--manifest required unless --audit-dataset or --goal-status")
        return 2

    experiments = _load_manifest(args.manifest.resolve())
    gpus = [int(g.strip()) for g in args.gpu.split(",") if g.strip()]
    pool = _GpuPool(gpus)
    parallel = min(args.max_parallel, len(gpus), len(experiments))
    _LOG.info("Scheduling %d experiment(s), GPUs=%s, parallel=%d, goal_mode=%s",
              len(experiments), gpus, parallel, args.goal_mode)

    rows: list[dict[str, Any]] = []
    module_passed: dict[str, bool] = {}

    def worker(exp: JDMExperiment) -> dict[str, Any]:
        gpu = pool.acquire()
        try:
            _LOG.info("[%s] GPU %d", exp.label, gpu)
            return _run_experiment(exp, gpu, args)
        finally:
            pool.release(gpu)

    if args.goal_mode:
        for exp in experiments:
            if args.stop_when_all_pass and jdm_goal_checklist(args.goals, DEFAULT_RETUNE_ROOT)["campaign_complete"]:
                _LOG.info("Campaign complete — all active goals met.")
                break
            if args.until_pass and module_passed.get(exp.module):
                _LOG.info("Skipping %s — goal already met for module %s", exp.label, exp.module)
                continue
            row = worker(exp)
            rows.append(row)
            if row.get("goal_met"):
                module_passed[exp.module] = True
                _LOG.info("Goal met for module %s via %s", exp.module, exp.variant)
                if args.until_pass:
                    continue
            if args.stop_when_all_pass:
                status = jdm_goal_checklist(args.goals, DEFAULT_RETUNE_ROOT)
                if status["campaign_complete"]:
                    _LOG.info("Campaign complete — all active goals met.")
                    break
    elif parallel <= 1:
        for exp in experiments:
            rows.append(worker(exp))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(worker, exp) for exp in experiments]
            for fut in as_completed(futs):
                rows.append(fut.result())

    _append_results(rows, args.dry_run, goal_mode=args.goal_mode)

    if args.goal_mode:
        status = jdm_goal_checklist(args.goals, DEFAULT_RETUNE_ROOT)
        if rows:
            status["last_experiment"] = dict(
                id=rows[-1].get("id"),
                variant=rows[-1].get("variant"),
                goal_met=rows[-1].get("goal_met"),
            )
        write_goal_status(GOAL_STATUS_PATH, status, dry_run=args.dry_run)
        _LOG.info(
            "Goal status: %d/%d active goals met, campaign_complete=%s",
            status["goals_met"],
            status["active_goals"],
            status["campaign_complete"],
        )
    errors = [r for r in rows if "error" in r["status"]]
    return 0 if not errors or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
