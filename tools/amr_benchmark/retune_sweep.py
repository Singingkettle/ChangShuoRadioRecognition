"""Hyperparameter retune orchestrator for AMR-Benchmark fail entries.

ARCHITECTURE FREEZE POLICY
--------------------------
Retunes must NOT change model network architecture (layers, topology, channel
counts, backbone structure). Must match AMR-Benchmark Keras reference / paper.
Allowed: hyperparameters, per-layer init, training strategy, documented input
pipeline choices. See docs/amr_benchmark/retune_campaign.md § Architecture freeze.

Permitted --cfg-options keys (whitelist for manifest ``cfg_options`` and CLI):
  model.backbone.init_cfg          # init only — not structural fields
  optim_wrapper.optimizer.lr
  optim_wrapper.optimizer.weight_decay
  optim_wrapper.clip_grad
  param_scheduler                  # lr schedule, warmup, cosine, etc.
  train_cfg.max_epochs
  train_cfg.val_interval
  custom_hooks                     # EarlyStoppingHook patience/min_delta
  train_dataloader.batch_size
  # Input pipeline: swap _base_ dataset config in dedicated .py files instead

Forbidden via --cfg-options: model.backbone.type, channels, kernel sizes,
hidden dims, num_classes, head type, layer add/remove.

Runs one or more retune variants for a (model × dataset) pair: train → test →
parse ``paper.pkl`` → append results to ``docs/amr_benchmark/retune_results.md``.
Optionally promotes a passing variant into the main tracking table via
``run_migration.py`` logic.

Each variant uses its own work directory under ``work_dirs/amr_benchmark_retune/``
so baseline ``work_dirs/amr_benchmark/`` runs are never overwritten.

Usage::

    # Dry-run the wave-1 manifest
    python tools/amr_benchmark/retune_sweep.py \\
        --manifest configs/amr_benchmark/retune/wave1_manifest.json --dry-run

    # Run two variants for CNN1DPF@2018 on GPU 0
    python tools/amr_benchmark/retune_sweep.py \\
        --model cnn1dpf --dataset deepsig201801A \\
        --variants xavier_lr1e3,lr2e4_warmup_clip --gpu 0

    # Run every experiment listed in a manifest
    python tools/amr_benchmark/retune_sweep.py \\
        --manifest configs/amr_benchmark/retune/wave1_manifest.json --gpu 0,1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from amr_benchmark.matrix import MATRIX, TOLERANCES, iter_jobs  # noqa: E402
from amr_benchmark.run_migration import (  # noqa: E402
    GpuPool,
    JobResult,
    JobSpec,
    _classify,
    _find_best_checkpoint,
    _parse_paper_pkl,
    _run_subprocess,
    _update_tracking_md,
)
from goal_mode_helpers import (  # noqa: E402
    amr_job_campaign_goal_met,
    amr_tracking_summary,
    group_manifest_by_pair,
    load_json,
    print_amr_goal_status,
    remaining_queue_from_manifest,
    resolve_amr_paper_exact,
    write_goal_status,
)

_LOG = logging.getLogger("amr_benchmark.retune")

DEFAULT_RETUNE_ROOT = _REPO_ROOT / "work_dirs" / "amr_benchmark_retune"
RETUNE_CONFIG_DIR = _REPO_ROOT / "configs" / "amr_benchmark" / "retune"
DEFAULT_GOALS_PATH = RETUNE_CONFIG_DIR / "goals.json"
GOAL_STATUS_PATH = DEFAULT_RETUNE_ROOT / "GOAL_STATUS.json"
RESULTS_PATH = _REPO_ROOT / "docs" / "amr_benchmark" / "retune_results.md"
CAMPAIGN_PATH = _REPO_ROOT / "docs" / "amr_benchmark" / "retune_campaign.md"


@dataclass
class RetuneExperiment:
    """A single hyperparameter variant to train and evaluate."""

    experiment_id: str
    model: str
    dataset: str
    variant: str
    config: Path
    cfg_options: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    priority: int = 99

    @property
    def label(self) -> str:
        return f"{self.model}/{self.dataset}/{self.variant}"

    def work_dir(self, root: Path) -> Path:
        return root / self.model / self.dataset / self.variant


def _cfg_options_to_cli(opts: dict[str, Any]) -> list[str]:
    """Flatten a nested dict into ``train.py``/``test.py`` --cfg-options args."""
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


def _load_manifest(path: Path) -> list[RetuneExperiment]:
    data = json.loads(path.read_text())
    exps: list[RetuneExperiment] = []
    for raw in data.get("experiments", []):
        cfg = _REPO_ROOT / raw["config"]
        exps.append(
            RetuneExperiment(
                experiment_id=raw.get("id", raw["variant"]),
                model=raw["model"].lower(),
                dataset=raw["dataset"],
                variant=raw["variant"],
                config=cfg,
                cfg_options=raw.get("cfg_options", {}),
                notes=raw.get("notes", ""),
                priority=raw.get("priority", 99),
            )
        )
    exps.sort(key=lambda e: (e.priority, e.model, e.dataset, e.variant))
    return exps


def _resolve_from_matrix(
    model: str,
    dataset: str,
    variants: list[str],
) -> list[RetuneExperiment]:
    model = model.lower()
    entry = MATRIX.get(model, {}).get(dataset)
    if entry is None:
        raise SystemExit(f"Unknown pair {model}/{dataset}")
    exps: list[RetuneExperiment] = []
    for variant in variants:
        cfg = RETUNE_CONFIG_DIR / f"wave1_{model}_{dataset}_{variant}.py"
        if not cfg.is_file():
            raise SystemExit(f"Retune config missing: {cfg}")
        exps.append(
            RetuneExperiment(
                experiment_id=f"{model}_{dataset}_{variant}",
                model=model,
                dataset=dataset,
                variant=variant,
                config=cfg,
            )
        )
    return exps


def _append_results_md(rows: list[dict[str, Any]], dry_run: bool,
                       goal_mode: bool = False) -> None:
    if not rows:
        return
    cols = (
        "| When (UTC) | Experiment | Variant | Overall | Peak | Status "
        "| vs baseline | Notes | Work dir |"
    )
    if goal_mode:
        cols = (
            "| When (UTC) | Experiment | Variant | Overall | Peak | Status "
            "| goal_met | vs baseline | Notes | Work dir |"
        )
    sep = "|" + "---|" * (cols.count("|") - 1)
    lines = [cols, sep]
    for row in rows:
        if goal_mode:
            lines.append(
                f"| {row['when']} | {row['experiment']} | `{row['variant']}` "
                f"| {row['overall']} | {row['peak']} | `{row['status']}` "
                f"| `{row.get('goal_met', '—')}` | {row['delta']} | {row['notes']} "
                f"| `{row['work_dir']}` |"
            )
        else:
            lines.append(
                f"| {row['when']} | {row['experiment']} | `{row['variant']}` "
                f"| {row['overall']} | {row['peak']} | `{row['status']}` "
                f"| {row['delta']} | {row['notes']} "
                f"| `{row['work_dir']}` |"
            )
    block = "\n".join(lines) + "\n"
    if dry_run:
        print("[DRY-RUN] would append to retune_results.md:\n")
        print(block)
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.is_file():
        RESULTS_PATH.write_text(
            "# AMR-Benchmark Retune Results\n\n"
            "Append-only log written by ``tools/amr_benchmark/retune_sweep.py``.\n\n"
        )
    with RESULTS_PATH.open("a") as fh:
        fh.write(f"\n## Run {rows[0]['when']}\n\n")
        fh.write(block)


def _run_variant(
    exp: RetuneExperiment,
    gpu: int,
    args: argparse.Namespace,
) -> tuple[RetuneExperiment, JobResult, dict[str, Any]]:
    entry = MATRIX[exp.model][exp.dataset]
    work_dir = exp.work_dir(args.retune_root)
    work_dir.mkdir(parents=True, exist_ok=True)

    spec = JobSpec(
        model=exp.model,
        dataset=exp.dataset,
        config=exp.config,
        work_dir=work_dir,
        target_overall=entry["target_overall"],
        target_peak=entry["target_peak"],
        target_best_snr=entry["target_best_snr"],
        notes=exp.notes,
    )
    result = JobResult(spec=spec)
    result.log_path = work_dir / "retune.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Keep CUDA logical indices aligned with nvidia-smi physical indices so
    # GPU-idle detection (nvidia-smi) and job placement agree.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    cfg_cli = _cfg_options_to_cli(exp.cfg_options)
    cfg_args = ["--cfg-options", *cfg_cli] if cfg_cli else []

    paper_pkl = work_dir / "res" / "paper.pkl"
    best_ckpt = _find_best_checkpoint(work_dir)
    needs_train = not args.skip_train
    if best_ckpt is not None and not args.force and not args.skip_train:
        _LOG.info("[%s] reusing checkpoint %s", exp.label, best_ckpt)
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
        result.train_cmd = " ".join(shlex.quote(c) for c in train_cmd)
        rc = _run_subprocess(train_cmd, env, result.log_path, args.dry_run)
        if rc != 0 and not args.dry_run:
            result.status = "error"
            result.error = f"train.py exited with code {rc}"
            row = _result_row(exp, result, args)
            return exp, result, row
        best_ckpt = _find_best_checkpoint(work_dir)

    if args.dry_run:
        result.status = "dry-run"
        row = _result_row(exp, result, args)
        return exp, result, row

    if best_ckpt is None:
        result.status = "error"
        result.error = "no checkpoint after train"
        row = _result_row(exp, result, args)
        return exp, result, row

    if not args.skip_test and (not paper_pkl.is_file() or args.force):
        test_cmd = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "test.py"),
            str(exp.config),
            str(best_ckpt),
            "--work-dir",
            str(work_dir),
            *cfg_args,
        ]
        result.test_cmd = " ".join(shlex.quote(c) for c in test_cmd)
        rc = _run_subprocess(test_cmd, env, result.log_path, False)
        if rc != 0:
            result.status = "error"
            result.error = f"test.py exited with code {rc}"
            row = _result_row(exp, result, args)
            return exp, result, row

    if not paper_pkl.is_file():
        result.status = "error"
        result.error = f"missing {paper_pkl}"
        row = _result_row(exp, result, args)
        return exp, result, row

    parsed = _parse_paper_pkl(paper_pkl)
    result.overall_acc = parsed["overall"]
    result.peak_acc = parsed["peak_acc"]
    result.best_snr = parsed["peak_snr"]
    result.extras = parsed
    result.status = _classify(result)
    result.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = _result_row(exp, result, args)
    return exp, result, row


def _baseline_meas(model: str, dataset: str) -> tuple[float | None, float | None]:
    """Best-effort read of baseline overall/peak from accuracy_tracking auto table."""
    tracking = _REPO_ROOT / "docs" / "amr_benchmark" / "accuracy_tracking.md"
    if not tracking.is_file():
        return None, None
    text = tracking.read_text()
    model_u = model.upper()
    ds_map = {
        "deepsig201610A": "RML2016.10A",
        "deepsig201610B": "RML2016.10B",
        "deepsig201801A": "RML2018.01A",
        "hisar2019": "HisarMod",
    }
    ds_label = ds_map.get(dataset, dataset)
    for line in text.splitlines():
        if not line.startswith(f"| {model_u} "):
            continue
        if f"| {ds_label} " not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 10:
            continue
        try:
            return float(parts[6]), float(parts[8])
        except ValueError:
            return None, None
    return None, None


def _result_row(
    exp: RetuneExperiment,
    result: JobResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_o, base_p = _baseline_meas(exp.model, exp.dataset)
    overall = result.overall_acc
    peak = result.peak_acc
    delta_parts: list[str] = []
    if overall is not None and base_o is not None:
        delta_parts.append(f"Δoverall {overall - base_o:+.2f}pp")
    if peak is not None and base_p is not None:
        delta_parts.append(f"Δpeak {peak - base_p:+.2f}pp")
    goals = load_json(args.goals) if getattr(args, "goals", None) else {}
    paper_exact = resolve_amr_paper_exact(
        goals,
        paper_exact=getattr(args, "paper_exact", None),
    )
    if getattr(args, "goal_mode", False):
        goal_met = amr_job_campaign_goal_met(result, goals, paper_exact=paper_exact)
    else:
        goal_met = result.status == "pass" if result.status else False
    return dict(
        when=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        experiment=f"{exp.model}/{exp.dataset}",
        variant=exp.variant,
        overall=f"{overall:.2f}" if overall is not None else "—",
        peak=f"{peak:.2f}" if peak is not None else "—",
        status=result.status,
        goal_met=goal_met,
        delta="; ".join(delta_parts) if delta_parts else "—",
        notes=exp.notes or result.error or "",
        work_dir=str(exp.work_dir(args.retune_root).relative_to(_REPO_ROOT)),
    )


def _expected_tracking_rows() -> int:
    """Full (model × dataset) matrix size used as a wipe-guard for GOAL."""
    try:
        from amr_benchmark.matrix import MATRIX

        return sum(len(ds) for ds in MATRIX.values())
    except Exception:  # pragma: no cover - defensive
        return 72


def _amr_campaign_complete(summary: dict[str, Any]) -> bool:
    """True only when fails=0 on a near-complete tracking matrix.

    A wiped / truncated auto table (e.g. promote wrote 1 row) must never
    report campaign_complete=true just because fail_count happened to be 0.
    """
    expected = _expected_tracking_rows()
    min_rows = max(expected - 5, expected // 2)  # allow a few CSRR-only gaps
    return (
        summary.get("fail_count", 0) == 0
        and summary.get("total_rows", 0) >= min_rows
        and summary.get("pass_count", 0) + summary.get("measured_count", 0)
        >= min_rows
    )


def _build_amr_goal_status(
    args: argparse.Namespace,
    experiments: list[RetuneExperiment],
    result_rows: list[dict[str, Any]],
    exhausted_pairs: list[str],
) -> dict[str, Any]:
    summary = amr_tracking_summary()
    grouped = group_manifest_by_pair(experiments)
    queue = remaining_queue_from_manifest(grouped, summary)
    passes_this_run = sum(1 for r in result_rows if r.get("goal_met") is True)
    goals = load_json(args.goals) if getattr(args, "goals", None) else {}
    return dict(
        total_fails=summary["fail_count"],
        passes_converted=passes_this_run,
        remaining_queue=len(queue),
        campaign_complete=_amr_campaign_complete(summary),
        campaign_mode=goals.get("campaign_mode", "paper_exact"),
        exhausted_pairs=exhausted_pairs,
        queue_preview=queue[:20],
        tracking_rows=summary.get("total_rows", 0),
        tracking_pass=summary.get("pass_count", 0),
        tracking_fail=summary.get("fail_count", 0),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal-status",
        action="store_true",
        help="Print pass/fail counts vs targets; no training.",
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest under configs/amr_benchmark/retune/.",
    )
    src.add_argument("--model", help="Model key (with --dataset and --variants).")
    parser.add_argument("--dataset", help="Dataset key (deepsig201610A, …).")
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names (maps to wave1_<model>_<dataset>_<variant>.py).",
    )
    parser.add_argument(
        "--retune-root",
        type=Path,
        default=DEFAULT_RETUNE_ROOT,
        help="Root work dir for retune variants.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device id(s), comma-separated (default: 0).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Concurrent retune jobs (default: 1).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If a variant passes, copy best ckpt + paper.pkl into the "
        "baseline work_dirs/amr_benchmark/ tree and refresh tracking.",
    )
    parser.add_argument(
        "--goal-mode",
        action="store_true",
        help="Enable goal-driven variant loop (see docs/amr_benchmark/goal_mode.md).",
    )
    parser.add_argument(
        "--until-pass",
        action="store_true",
        help="For each (model×dataset), stop at first passing variant.",
    )
    parser.add_argument(
        "--stop-when-all-pass",
        action="store_true",
        help="Stop campaign when accuracy_tracking.md shows 0 fails.",
    )
    parser.add_argument(
        "--goals",
        type=Path,
        default=DEFAULT_GOALS_PATH,
        help="Goal thresholds JSON (default: configs/amr_benchmark/retune/goals.json).",
    )
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        default=None,
        help="Campaign success = paper target exactly (default: goals.json campaign_mode).",
    )
    parser.add_argument(
        "--no-paper-exact",
        action="store_false",
        dest="paper_exact",
        help="Campaign success = tracking tolerance pass (legacy).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.goal_status and not args.manifest and not args.model:
        parser.error("one of --manifest or --model is required (unless --goal-status)")
    return args


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
        print_amr_goal_status(args.goals, args.manifest.resolve() if args.manifest else None)
        return 0

    args.retune_root = Path(args.retune_root).resolve()
    args.retune_root.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        experiments = _load_manifest(args.manifest.resolve())
    else:
        if not args.model or not args.dataset or not args.variants:
            _LOG.error("--model, --dataset, and --variants are required without --manifest")
            return 2
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        experiments = _resolve_from_matrix(args.model, args.dataset, variants)

    gpus = [int(g.strip()) for g in args.gpu.split(",") if g.strip()]
    gpu_pool = GpuPool(gpus)
    parallelism = min(args.max_parallel, len(gpu_pool), len(experiments))

    result_rows: list[dict[str, Any]] = []
    passed: list[tuple[RetuneExperiment, JobResult]] = []
    exhausted_pairs: list[str] = []

    def worker(exp: RetuneExperiment):
        gpu = gpu_pool.acquire()
        try:
            _LOG.info("[%s] start (GPU %d)", exp.label, gpu)
            return _run_variant(exp, gpu, args)
        finally:
            gpu_pool.release(gpu)

    if args.goal_mode:
        grouped = group_manifest_by_pair(experiments)
        _LOG.info(
            "Goal mode: %d (model×dataset) pair(s), until_pass=%s stop_when_all_pass=%s "
            "paper_exact=%s",
            len(grouped),
            args.until_pass,
            args.stop_when_all_pass,
            resolve_amr_paper_exact(load_json(args.goals), paper_exact=args.paper_exact),
        )
        for (model, dataset), variants in sorted(grouped.items()):
            if args.stop_when_all_pass and _amr_campaign_complete(amr_tracking_summary()):
                _LOG.info("Campaign complete — 0 fails on full tracking matrix.")
                break
            pair_passed = False
            for exp in variants:
                exp_out, res, row = worker(exp)
                result_rows.append(row)
                if row.get("goal_met"):
                    passed.append((exp_out, res))
                    pair_passed = True
                    _LOG.info(
                        "Campaign goal met for %s/%s via %s (tracking status=%s)",
                        model,
                        dataset,
                        exp.variant,
                        res.status,
                    )
                    if args.until_pass:
                        break
            if not pair_passed and variants:
                label = f"{model}/{dataset}"
                exhausted_pairs.append(label)
                _LOG.warning("Goal exhausted for %s — all %d variant(s) failed", label, len(variants))
            if args.stop_when_all_pass and _amr_campaign_complete(amr_tracking_summary()):
                _LOG.info("Campaign complete — 0 fails on full tracking matrix.")
                break
    elif parallelism <= 1:
        _LOG.info(
            "Scheduling %d retune experiment(s) on GPU(s) %s (parallel=%d)",
            len(experiments),
            gpus,
            parallelism,
        )
        for exp in experiments:
            exp_out, res, row = worker(exp)
            result_rows.append(row)
            if res.status == "pass":
                passed.append((exp_out, res))
    else:
        _LOG.info(
            "Scheduling %d retune experiment(s) on GPU(s) %s (parallel=%d)",
            len(experiments),
            gpus,
            parallelism,
        )
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = [pool.submit(worker, exp) for exp in experiments]
            for fut in as_completed(futures):
                exp_out, res, row = fut.result()
                result_rows.append(row)
                if res.status == "pass":
                    passed.append((exp_out, res))

    _append_results_md(result_rows, dry_run=args.dry_run, goal_mode=args.goal_mode)

    if args.goal_mode:
        status = _build_amr_goal_status(args, experiments, result_rows, exhausted_pairs)
        write_goal_status(GOAL_STATUS_PATH, status, dry_run=args.dry_run)
        _LOG.info(
            "Goal status: %d fails, %d queue remaining, campaign_complete=%s",
            status["total_fails"],
            status["remaining_queue"],
            status["campaign_complete"],
        )

    if passed:
        _LOG.info("%d variant(s) met campaign goal:", len(passed))
        for exp, res in passed:
            _LOG.info(
                "  PASS %s overall=%.2f peak=%.2f",
                exp.label,
                res.overall_acc or 0,
                res.peak_acc or 0,
            )

    if args.promote and passed and not args.dry_run:
        import shutil

        promote_results: list[JobResult] = []
        for exp, res in passed:
            baseline_dir = _REPO_ROOT / "work_dirs" / "amr_benchmark" / exp.model / exp.dataset
            variant_dir = exp.work_dir(args.retune_root)
            baseline_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ("best_*.pth", "res/paper.pkl"):
                for src in variant_dir.glob(pattern):
                    dst = baseline_dir / src.name if "res" not in pattern else baseline_dir / "res" / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            promote_results.append(res)
        _update_tracking_md(promote_results, dry_run=False)
        _LOG.info("Promoted %d passing variant(s) to baseline tracking.", len(passed))

    fails = [r for r in result_rows if r["status"] not in ("pass", "dry-run")]
    return 0 if not fails or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
