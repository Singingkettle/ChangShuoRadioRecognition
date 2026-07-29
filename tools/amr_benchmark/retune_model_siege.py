"""Per-model siege orchestrator for AMR-Benchmark retunes.

ARCHITECTURE FREEZE POLICY — same as ``retune_sweep.py``; see
``docs/amr_benchmark/retune_campaign.md`` § Architecture freeze.

**Siege mode** resolves one (model × dataset) at a time:
  1. Launch **all** variants for the current pair **in parallel** (GPU pool).
  2. Stop the pair on first pass (``--until-pass``) or when every variant fails.
  3. Auto-advance to the next entry in ``siege_queue.json``.

Contrast with goal-mode ``retune_sweep.py``: goal mode walks the manifest
serially by pair *and* serially by variant within each pair.

Usage::

    # Dry-run the full siege queue
    python tools/amr_benchmark/retune_model_siege.py \\
        --queue configs/amr_benchmark/retune/siege_queue.json --dry-run

    # Run siege (2 GPUs, 2 parallel variants per pair)
    python tools/amr_benchmark/retune_model_siege.py \\
        --queue configs/amr_benchmark/retune/siege_queue.json \\
        --gpu 0,1 --max-parallel 2 --until-pass --promote

    # Single pair siege (bypass queue)
    python tools/amr_benchmark/retune_model_siege.py \\
        --model fastmldnn --dataset deepsig201610A \\
        --manifest configs/amr_benchmark/retune/siege_fastmldnn_10a.json \\
        --gpu 0,1 --max-parallel 2 --until-pass
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from amr_benchmark.retune_sweep import (  # noqa: E402
    DEFAULT_RETUNE_ROOT,
    GOAL_STATUS_PATH,
    RetuneExperiment,
    _append_results_md,
    _build_amr_goal_status,
    _load_manifest,
    _run_variant,
    DEFAULT_GOALS_PATH,
)
from amr_benchmark.run_migration import (  # noqa: E402
    GpuPool,
    JobResult,
    _update_tracking_md,
)
from goal_mode_helpers import (  # noqa: E402
    amr_job_campaign_goal_met,
    amr_tracking_summary,
    load_json,
    resolve_amr_paper_exact,
    write_goal_status,
)

_LOG = logging.getLogger("amr_benchmark.retune_siege")
DEFAULT_QUEUE_PATH = (
    _REPO_ROOT / "configs" / "amr_benchmark" / "retune" / "siege_queue.json"
)
SIEGE_STATUS_PATH = DEFAULT_RETUNE_ROOT / "SIEGE_STATUS.json"


def _load_siege_queue(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_siege_queue(path: Path, data: dict[str, Any], dry_run: bool) -> None:
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if dry_run:
        _LOG.info("[DRY-RUN] would update siege queue %s", path)
        return
    path.write_text(json.dumps(data, indent=2) + "\n")


def _update_queue_entry(
    path: Path, entry_id: str, updates: dict[str, Any], dry_run: bool
) -> None:
    """Reload-merge-save a SINGLE entry so concurrent orchestrators never
    clobber each other's status writes (or entries added after we started)."""
    queue = _load_siege_queue(path)
    for entry in queue.get("entries", []):
        if entry.get("id") == entry_id:
            entry.update(updates)
            break
    else:
        _LOG.warning("Queue entry %s vanished from %s — not saving", entry_id, path)
        return
    _save_siege_queue(path, queue, dry_run=dry_run)


def _campaign_goal_met(
    exp: RetuneExperiment,
    res: JobResult,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """True when variant meets campaign success (paper-exact by default)."""
    if row.get("goal_met") is not None:
        return bool(row.get("goal_met"))
    goals = load_json(args.goals)
    return amr_job_campaign_goal_met(
        res,
        goals,
        paper_exact=resolve_amr_paper_exact(goals, paper_exact=args.paper_exact),
    )


def _run_pair_siege(
    experiments: list[RetuneExperiment],
    args: argparse.Namespace,
    gpu_pool: GpuPool,
) -> tuple[list[dict[str, Any]], list[tuple[RetuneExperiment, JobResult]], bool]:
    """Run all variants for one pair in parallel; return rows, passes, pair_resolved."""
    if not experiments:
        return [], [], True

    parallelism = min(args.max_parallel, len(gpu_pool), len(experiments))
    result_rows: list[dict[str, Any]] = []
    passed: list[tuple[RetuneExperiment, JobResult]] = []

    def worker(exp: RetuneExperiment):
        gpu = gpu_pool.acquire()
        try:
            _LOG.info("[%s] siege start (GPU %d)", exp.label, gpu)
            return _run_variant(exp, gpu, args)
        finally:
            gpu_pool.release(gpu)

    _LOG.info(
        "Siege pair %s/%s: %d variant(s), parallel=%d",
        experiments[0].model,
        experiments[0].dataset,
        len(experiments),
        parallelism,
    )

    if parallelism <= 1:
        for exp in experiments:
            exp_out, res, row = worker(exp)
            result_rows.append(row)
            if _campaign_goal_met(exp_out, res, row, args):
                passed.append((exp_out, res))
                if args.until_pass:
                    _LOG.info(
                        "Siege campaign goal met for %s/%s via %s (tracking status=%s)",
                        exp.model,
                        exp.dataset,
                        exp.variant,
                        res.status,
                    )
                    return result_rows, passed, True
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {pool.submit(worker, exp): exp for exp in experiments}
            for fut in as_completed(futures):
                exp_out, res, row = fut.result()
                result_rows.append(row)
                if _campaign_goal_met(exp_out, res, row, args):
                    passed.append((exp_out, res))
                    if args.until_pass:
                        _LOG.info(
                            "Siege campaign goal met for %s/%s via %s (tracking status=%s)",
                            exp_out.model,
                            exp_out.dataset,
                            exp_out.variant,
                            res.status,
                        )
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        return result_rows, passed, True

    pair_resolved = bool(passed) or len(result_rows) >= len(experiments)
    return result_rows, passed, pair_resolved


def _promote_passes(
    passed: list[tuple[RetuneExperiment, JobResult]],
    args: argparse.Namespace,
) -> None:
    import shutil

    promote_results: list[JobResult] = []
    for exp, res in passed:
        baseline_dir = (
            _REPO_ROOT / "work_dirs" / "amr_benchmark" / exp.model / exp.dataset
        )
        variant_dir = exp.work_dir(args.retune_root)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("best_*.pth", "res/paper.pkl"):
            for src in variant_dir.glob(pattern):
                dst = (
                    baseline_dir / src.name
                    if "res" not in pattern
                    else baseline_dir / "res" / src.name
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        promote_results.append(res)
    _update_tracking_md(promote_results, dry_run=False)
    _LOG.info("Promoted %d passing variant(s) to baseline tracking.", len(passed))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="Siege queue JSON (default: configs/amr_benchmark/retune/siege_queue.json).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Single-pair manifest (with --model/--dataset, skips queue advance).",
    )
    parser.add_argument("--model", help="Model key for single-pair siege.")
    parser.add_argument("--dataset", help="Dataset key for single-pair siege.")
    parser.add_argument(
        "--retune-root",
        type=Path,
        default=DEFAULT_RETUNE_ROOT,
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument(
        "--until-pass",
        action="store_true",
        default=True,
        help="Stop pair at first passing variant (default: on).",
    )
    parser.add_argument(
        "--no-until-pass",
        action="store_false",
        dest="until_pass",
        help="Run all variants even after a pass.",
    )
    parser.add_argument(
        "--stop-on-pass",
        action="store_true",
        help="Stop entire siege campaign when any pair passes.",
    )
    parser.add_argument(
        "--goals",
        type=Path,
        default=DEFAULT_GOALS_PATH,
        help="Goals JSON for campaign-success rule.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    args.retune_root = Path(args.retune_root).resolve()
    args.retune_root.mkdir(parents=True, exist_ok=True)
    args.goals = Path(args.goals).resolve()
    args.goal_mode = True

    gpus = [int(g.strip()) for g in args.gpu.split(",") if g.strip()]
    gpu_pool = GpuPool(gpus)

    all_rows: list[dict[str, Any]] = []
    all_passed: list[tuple[RetuneExperiment, JobResult]] = []
    siege_outcomes: list[dict[str, Any]] = []
    exhausted_pairs: list[str] = []
    all_experiments: list[RetuneExperiment] = []

    single_pair = args.manifest is not None
    if single_pair:
        manifest_path = args.manifest.resolve()
        experiments = _load_manifest(manifest_path)
        all_experiments = experiments
        if args.model and experiments:
            experiments = [
                e
                for e in experiments
                if e.model == args.model.lower() and e.dataset == args.dataset
            ]
        rows, passed, resolved = _run_pair_siege(experiments, args, gpu_pool)
        all_rows.extend(rows)
        all_passed.extend(passed)
        label = f"{experiments[0].model}/{experiments[0].dataset}" if experiments else "?"
        siege_outcomes.append(
            dict(
                id="single_pair",
                pair=label,
                status="passed" if passed else "exhausted",
                passing_variant=passed[0][0].variant if passed else None,
            )
        )
        if not passed and resolved:
            exhausted_pairs.append(label)
    else:
        queue_path = args.queue.resolve()
        claimed_ids: set[str] = set()

        while True:
            # Reload the queue on EVERY claim so entries added or claimed by
            # concurrent orchestrators after we started are honoured.
            queue = _load_siege_queue(queue_path)
            entries = sorted(
                queue.get("entries", []), key=lambda e: e.get("priority", 99)
            )
            entry = None
            for cand in entries:
                status = cand.get("status", "pending")
                # "running" is skipped so a second orchestrator (launched to
                # fill idle GPUs while an earlier one is mid-entry) never
                # re-claims an entry that is already being processed.
                if status in ("passed", "exhausted", "skipped", "running"):
                    continue
                if cand.get("id") in claimed_ids:
                    continue
                if not cand.get("manifest"):
                    _LOG.error("Entry %s missing manifest", cand.get("id"))
                    claimed_ids.add(cand.get("id", ""))
                    continue
                entry = cand
                break
            if entry is None:
                break

            entry_id = entry.get("id", "")
            claimed_ids.add(entry_id)
            manifest_path = _REPO_ROOT / entry["manifest"]
            experiments = _load_manifest(manifest_path)
            all_experiments.extend(experiments)
            _update_queue_entry(
                queue_path, entry_id, {"status": "running"}, dry_run=args.dry_run
            )

            rows, passed, resolved = _run_pair_siege(experiments, args, gpu_pool)
            all_rows.extend(rows)
            all_passed.extend(passed)

            pair_label = f"{entry['model']}/{entry['dataset']}"
            updates: dict[str, Any] = {}
            if passed:
                updates["status"] = "passed"
                updates["passing_variant"] = passed[0][0].variant
                updates["resolved_at"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                _LOG.info(
                    "Siege WON %s via %s",
                    pair_label,
                    passed[0][0].variant,
                )
                if args.promote and not args.dry_run:
                    _promote_passes(passed, args)
            elif resolved:
                updates["status"] = "exhausted"
                exhausted_pairs.append(pair_label)
                _LOG.warning(
                    "Siege exhausted for %s — all %d variant(s) failed",
                    pair_label,
                    len(experiments),
                )
            else:
                updates["status"] = "pending"

            siege_outcomes.append(
                dict(
                    id=entry_id,
                    pair=pair_label,
                    status=updates["status"],
                    passing_variant=updates.get("passing_variant"),
                    variant_count=len(experiments),
                )
            )
            _update_queue_entry(queue_path, entry_id, updates, dry_run=args.dry_run)

            if passed and args.stop_on_pass:
                _LOG.info("Campaign stop-on-pass — halting siege queue.")
                break

            summary = amr_tracking_summary()
            if summary["fail_count"] == 0:
                _LOG.info("Campaign complete — 0 fails in tracking table.")
                break

    _append_results_md(all_rows, dry_run=args.dry_run, goal_mode=True)

    if all_rows:
        status = _build_amr_goal_status(
            args,
            all_experiments,
            all_rows,
            exhausted_pairs,
        )
        status["siege_mode"] = True
        status["siege_outcomes"] = siege_outcomes
        write_goal_status(SIEGE_STATUS_PATH, status, dry_run=args.dry_run)
        write_goal_status(GOAL_STATUS_PATH, status, dry_run=args.dry_run)

    if all_passed:
        _LOG.info("%d variant(s) met campaign goal:", len(all_passed))
        for exp, res in all_passed:
            _LOG.info(
                "  GOAL %s overall=%.2f peak=%.2f (tracking=%s)",
                exp.label,
                res.overall_acc or 0,
                res.peak_acc or 0,
                res.status,
            )
        if args.promote and not args.dry_run and single_pair:
            _promote_passes(all_passed, args)

    fails = [r for r in all_rows if r["status"] not in ("pass", "dry-run")]
    return 0 if not fails or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
