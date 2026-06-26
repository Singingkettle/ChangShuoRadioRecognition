"""Phase 2 orchestrator for the AMR-Benchmark reproduction sweep.

Iterates the (model × dataset) matrix in :mod:`tools.amr_benchmark.matrix`,
trains and tests each entry on the available GPUs, parses ``paper.pkl`` to
compute overall and per-SNR peak accuracies, and rewrites the auto-managed
section of ``docs/amr_benchmark/accuracy_tracking.md`` with measured vs.
target columns.

Design notes:

* No CSRR/torch imports happen here. ``train.py`` and ``test.py`` run in
  subprocesses so the orchestrator stays light-weight and resilient.
* GPUs are round-robin assigned via ``CUDA_VISIBLE_DEVICES``. With the
  default ``--gpus 0,1 --max-parallel 2`` two jobs run concurrently.
* The script is fully resumable: re-running with the same ``--results-dir``
  reuses any existing ``paper.pkl`` and only re-launches missing or failed
  jobs. Pass ``--force`` to retrain regardless.
* ``--dry-run`` prints the planned commands without executing them, which
  is the recommended pre-flight check before the full Phase 2 sweep.

Usage examples::

    # Dry-run the entire matrix on GPUs 0 and 1
    python tools/amr_benchmark/run_migration.py --dry-run

    # Train just MCLDNN on the two RML 2016 datasets
    python tools/amr_benchmark/run_migration.py \\
        --models mcldnn \\
        --datasets deepsig201610A deepsig201610B

    # Re-parse paper.pkl files without retraining (e.g. after editing
    # tolerances) -- useful when refreshing the tracking table only.
    python tools/amr_benchmark/run_migration.py --skip-train --skip-test
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repository / sibling-module imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from amr_benchmark.matrix import (  # noqa: E402  (after sys.path tweak)
    MATRIX,
    TOLERANCES,
    iter_jobs,
    known_datasets,
    known_models,
)


_LOG = logging.getLogger("amr_benchmark.orchestrator")

DEFAULT_RESULTS_DIR = _REPO_ROOT / "work_dirs" / "amr_benchmark"
TRACKING_PATH = _REPO_ROOT / "docs" / "amr_benchmark" / "accuracy_tracking.md"
AUTO_BEGIN = "<!-- AMR_BENCHMARK_AUTO_TABLE_BEGIN -->"
AUTO_END = "<!-- AMR_BENCHMARK_AUTO_TABLE_END -->"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    """A single (model, dataset) work item."""

    model: str
    dataset: str
    config: Path
    work_dir: Path
    target_overall: float | None
    target_peak: float | None
    target_best_snr: float | None
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.model}/{self.dataset}"


@dataclass
class JobResult:
    spec: JobSpec
    status: str = "pending"
    overall_acc: float | None = None
    peak_acc: float | None = None
    best_snr: float | None = None
    error: str | None = None
    train_cmd: str | None = None
    test_cmd: str | None = None
    finished_at: str | None = None
    log_path: Path | None = None
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPU bookkeeping
# ---------------------------------------------------------------------------


class GpuPool:
    """Simple semaphore-like pool that hands out CUDA device IDs."""

    def __init__(self, gpus: list[int]):
        self._gpus = list(gpus)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._available = list(gpus)

    def acquire(self) -> int:
        with self._cond:
            while not self._available:
                self._cond.wait()
            return self._available.pop(0)

    def release(self, gpu_id: int) -> None:
        with self._cond:
            self._available.append(gpu_id)
            self._cond.notify()

    def __len__(self) -> int:
        return len(self._gpus)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jobs(args: argparse.Namespace) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for model, dataset, entry in iter_jobs(args.models, args.datasets):
        config = _REPO_ROOT / entry["config"]
        if not config.is_file():
            _LOG.warning("Config missing for %s/%s -> %s", model, dataset, config)
            continue
        work_dir = args.results_dir / model / dataset
        jobs.append(
            JobSpec(
                model=model,
                dataset=dataset,
                config=config,
                work_dir=work_dir,
                target_overall=entry["target_overall"],
                target_peak=entry["target_peak"],
                target_best_snr=entry["target_best_snr"],
                notes=entry.get("notes", ""),
            )
        )
    return jobs


def _find_best_checkpoint(work_dir: Path) -> Path | None:
    """Return the path to the best-validation checkpoint, or None."""
    candidates: list[Path] = []
    candidates.extend(sorted(work_dir.glob("best_*.pth")))
    candidates.extend(sorted(work_dir.glob("**/best_*.pth")))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    last = work_dir / "last_checkpoint"
    if last.is_file():
        text = last.read_text().strip()
        if text:
            ckpt = Path(text)
            if ckpt.is_file():
                return ckpt
    epoch = sorted(work_dir.glob("epoch_*.pth"))
    if epoch:
        epoch.sort(key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)))
        return epoch[-1]
    return None


def _run_subprocess(cmd: list[str], env: dict[str, str], log_path: Path,
                    dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(f"[DRY-RUN] {pretty} > {log_path}")
        return 0
    _LOG.info("Running: %s", pretty)
    with open(log_path, "ab") as fh:
        fh.write(f"\n$ {pretty}\n".encode())
        fh.flush()
        proc = subprocess.run(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    return proc.returncode


def _parse_paper_pkl(pkl_path: Path) -> dict[str, float | dict]:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    pps = data["pps"]
    gts = np.asarray(data["gts"]).astype(np.int64)
    snrs = np.asarray(data["snrs"])
    preds = np.argmax(pps, axis=1)
    correct = (preds == gts).astype(np.float64)
    overall = float(correct.mean() * 100.0)

    per_snr: dict[float, float] = {}
    for snr in np.unique(snrs):
        mask = snrs == snr
        if mask.any():
            per_snr[float(snr)] = float(correct[mask].mean() * 100.0)
    if per_snr:
        peak_snr, peak_acc = max(per_snr.items(), key=lambda kv: kv[1])
    else:
        peak_snr, peak_acc = float("nan"), float("nan")
    return dict(
        overall=overall,
        peak_acc=peak_acc,
        peak_snr=peak_snr,
        per_snr=per_snr,
    )


def _classify(result: JobResult) -> str:
    if result.error is not None:
        return "error"
    if result.overall_acc is None:
        return "pending"
    spec = result.spec
    if spec.target_overall is None or spec.target_peak is None:
        return "measured"
    # One-sided tolerance: matching OR exceeding the reference accuracy is a
    # pass. We only fail when the measured value falls more than the tolerance
    # *below* the reference. The best-SNR location is informational only --
    # accuracy saturates on a high-SNR plateau, so the per-SNR argmax (commonly
    # 14-18 dB) is not a meaningful discriminator and must not fail a run that
    # otherwise reproduces the reference accuracy.
    overall_ok = result.overall_acc >= spec.target_overall - TOLERANCES["overall"]
    peak_ok = (result.peak_acc is not None and
               result.peak_acc >= spec.target_peak - TOLERANCES["peak"])
    return "pass" if (overall_ok and peak_ok) else "fail"


# ---------------------------------------------------------------------------
# Pipeline (per-job)
# ---------------------------------------------------------------------------


def _run_job(spec: JobSpec, gpu: int, args: argparse.Namespace) -> JobResult:
    result = JobResult(spec=spec)
    result.log_path = spec.work_dir / "orchestrator.log"
    spec.work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    paper_pkl = spec.work_dir / "res" / "paper.pkl"
    needs_train = not args.skip_train
    needs_test = not args.skip_test

    # ---------------- TRAIN ----------------
    best_ckpt = _find_best_checkpoint(spec.work_dir)
    if best_ckpt is not None and not args.force and not args.skip_train:
        _LOG.info("[%s] Reusing existing checkpoint %s", spec.label, best_ckpt)
        needs_train = False
    if needs_train:
        train_cmd = [
            sys.executable, str(_REPO_ROOT / "tools" / "train.py"),
            str(spec.config), "--work-dir", str(spec.work_dir),
        ]
        if args.train_extra:
            train_cmd.extend(shlex.split(args.train_extra))
        result.train_cmd = " ".join(shlex.quote(c) for c in train_cmd)
        rc = _run_subprocess(train_cmd, env, result.log_path, args.dry_run)
        if rc != 0 and not args.dry_run:
            result.status = "error"
            result.error = f"train.py exited with code {rc}"
            return result
        best_ckpt = _find_best_checkpoint(spec.work_dir)

    if args.dry_run:
        result.status = "dry-run"
        result.test_cmd = "(dry-run, test command will be assembled after train)"
        return result

    if best_ckpt is None:
        result.status = "error"
        result.error = "no checkpoint produced after train.py"
        return result

    # ---------------- TEST ----------------
    if needs_test and (not paper_pkl.is_file() or args.force):
        test_cmd = [
            sys.executable, str(_REPO_ROOT / "tools" / "test.py"),
            str(spec.config), str(best_ckpt),
            "--work-dir", str(spec.work_dir),
        ]
        if args.test_extra:
            test_cmd.extend(shlex.split(args.test_extra))
        result.test_cmd = " ".join(shlex.quote(c) for c in test_cmd)
        rc = _run_subprocess(test_cmd, env, result.log_path, False)
        if rc != 0:
            result.status = "error"
            result.error = f"test.py exited with code {rc}"
            return result

    # ---------------- PARSE ----------------
    if not paper_pkl.is_file():
        result.status = "error"
        result.error = f"missing {paper_pkl}"
        return result
    try:
        parsed = _parse_paper_pkl(paper_pkl)
    except Exception as exc:  # pragma: no cover - defensive
        result.status = "error"
        result.error = f"failed to parse paper.pkl: {exc}"
        return result
    result.overall_acc = parsed["overall"]
    result.peak_acc = parsed["peak_acc"]
    result.best_snr = parsed["peak_snr"]
    result.extras = parsed
    result.status = _classify(result)
    result.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


# ---------------------------------------------------------------------------
# Tracking table rendering
# ---------------------------------------------------------------------------


_DATASET_LABELS = {
    "deepsig201610A": "RML2016.10A",
    "deepsig201610B": "RML2016.10B",
    "deepsig201801A": "RML2018.01A",
    "hisar2019": "HisarMod",
}


def _fmt_pp(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _fmt_db(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:g} dB"


def _render_row(result: JobResult) -> str:
    spec = result.spec
    target_o = _fmt_pp(spec.target_overall) if spec.target_overall is not None else "(CSRR-only)"
    target_p = _fmt_pp(spec.target_peak) if spec.target_peak is not None else "(CSRR-only)"
    target_s = _fmt_db(spec.target_best_snr)
    measured_o = _fmt_pp(result.overall_acc)
    measured_p = _fmt_pp(result.peak_acc)
    measured_s = _fmt_db(result.best_snr)
    return (
        f"| {spec.model.upper()} "
        f"| {_DATASET_LABELS.get(spec.dataset, spec.dataset)} "
        f"| `{spec.config.relative_to(_REPO_ROOT)}` "
        f"| `{spec.work_dir.relative_to(_REPO_ROOT)}` "
        f"| {target_o} | {measured_o} "
        f"| {target_p} | {measured_p} "
        f"| {target_s} | {measured_s} "
        f"| `{result.status}` "
        f"| {result.finished_at or '—'} |"
    )


def _render_table(results: list[JobResult]) -> str:
    header = (
        "| Model | Dataset | Config | Work dir "
        "| Overall (target %) | Overall (meas %) "
        "| Peak (target %) | Peak (meas %) "
        "| Best SNR (target) | Best SNR (meas) "
        "| Status | Updated |"
    )
    sep = "|" + "---|" * 12
    lines = [header, sep]
    by_model: dict[str, list[JobResult]] = {}
    for res in results:
        by_model.setdefault(res.spec.model, []).append(res)
    for model in MATRIX.keys():
        for res in by_model.get(model, []):
            lines.append(_render_row(res))
    return "\n".join(lines)


def _update_tracking_md(results: list[JobResult], dry_run: bool) -> None:
    if not TRACKING_PATH.is_file():
        _LOG.warning("Tracking file %s missing; not updating", TRACKING_PATH)
        return
    text = TRACKING_PATH.read_text()
    new_table = _render_table(results)
    block = (
        f"{AUTO_BEGIN}\n"
        f"_Last orchestrator run: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_\n\n"
        f"{new_table}\n"
        f"{AUTO_END}"
    )
    if AUTO_BEGIN in text and AUTO_END in text:
        before = text.split(AUTO_BEGIN)[0]
        after = text.split(AUTO_END)[1]
        new_text = before + block + after
    else:
        new_text = text.rstrip() + "\n\n## Auto-generated results\n\n" + block + "\n"
    if dry_run:
        print("[DRY-RUN] would write the following tracking section:\n")
        print(block)
        return
    TRACKING_PATH.write_text(new_text)
    _LOG.info("Wrote %d rows to %s", len(results), TRACKING_PATH)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _parse_gpus(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid GPU id {p!r}; expected comma-separated integers") from exc
    if not out:
        raise argparse.ArgumentTypeError("--gpus needs at least one id")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate AMR-Benchmark reproduction (train + test + parse).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", nargs="*", choices=known_models(),
                        help="Restrict to a subset of models (default: all).")
    parser.add_argument("--datasets", "--dataset", nargs="*",
                        choices=known_datasets(),
                        help="Restrict to a subset of datasets (default: all).")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Root work directory for all (model, dataset) "
                             "runs (default: work_dirs/amr_benchmark).")
    parser.add_argument("--gpus", type=_parse_gpus, default="0,1",
                        help="Comma-separated CUDA device ids to round-robin "
                             "over (default: 0,1).")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Maximum concurrent (train+test) jobs "
                             "(default: 2; capped by len(--gpus)).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned commands without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-train and re-test even when checkpoints / "
                             "paper.pkl already exist.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train.py and reuse existing checkpoints.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test.py and only re-parse existing paper.pkl.")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Do not update docs/amr_benchmark/accuracy_tracking.md.")
    parser.add_argument("--train-extra", default="",
                        help="Extra args appended verbatim to train.py "
                             "(e.g. --train-extra '--amp').")
    parser.add_argument("--test-extra", default="",
                        help="Extra args appended verbatim to test.py.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Increase logging verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    args.results_dir = Path(args.results_dir).resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    jobs = _make_jobs(args)
    if not jobs:
        _LOG.error("No jobs match the requested filters.")
        return 2

    _LOG.info("Scheduling %d job(s) across GPUs %s "
              "(max-parallel=%d, dry-run=%s, force=%s)",
              len(jobs), args.gpus, args.max_parallel, args.dry_run, args.force)

    if args.dry_run:
        for job in jobs:
            print(f"  - {job.label:35s} cfg={job.config.relative_to(_REPO_ROOT)} "
                  f"target_overall={job.target_overall} target_peak={job.target_peak} "
                  f"target_snr={job.target_best_snr}")

    gpu_pool = GpuPool(args.gpus)
    parallelism = min(args.max_parallel, len(gpu_pool))
    if parallelism < 1:
        _LOG.error("max_parallel resolved to %d -- nothing to do.", parallelism)
        return 2

    results: list[JobResult] = []
    results_lock = threading.Lock()

    def worker(job: JobSpec) -> JobResult:
        gpu = gpu_pool.acquire()
        try:
            _LOG.info("[%s] start (GPU %d)", job.label, gpu)
            res = _run_job(job, gpu, args)
            _LOG.info("[%s] done (status=%s, overall=%s, peak=%s @ %s dB)",
                      job.label, res.status,
                      f"{res.overall_acc:.2f}" if res.overall_acc is not None else "—",
                      f"{res.peak_acc:.2f}" if res.peak_acc is not None else "—",
                      f"{res.best_snr:g}" if res.best_snr is not None else "—")
            return res
        finally:
            gpu_pool.release(gpu)

    if args.dry_run:
        for job in jobs:
            results.append(_run_job(job, args.gpus[0], args))
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for fut in as_completed(futures):
                res = fut.result()
                with results_lock:
                    results.append(res)
                    if not args.no_tracking:
                        _update_tracking_md(results, dry_run=False)

    if args.dry_run:
        _LOG.info("Dry-run complete; %d planned jobs.", len(results))
        if not args.no_tracking:
            _update_tracking_md(results, dry_run=True)
        return 0

    # Final summary
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    _LOG.info("Run complete. Status counts: %s", by_status)
    failed = [r for r in results if r.status in ("error", "fail")]
    if failed:
        _LOG.warning("%d job(s) did not pass:", len(failed))
        for r in failed:
            _LOG.warning("  - %s : %s %s", r.spec.label, r.status,
                         r.error or "")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
