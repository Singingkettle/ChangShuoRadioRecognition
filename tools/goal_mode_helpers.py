"""Shared helpers for AMR / JDM retune goal mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

AMR_DATASET_LABELS: dict[str, str] = {
    "deepsig201610A": "RML2016.10A",
    "deepsig201610B": "RML2016.10B",
    "deepsig201801A": "RML2018.01A",
    "hisar2019": "HisarMod",
}
AMR_LABEL_TO_DATASET = {v: k for k, v in AMR_DATASET_LABELS.items()}

DEFAULT_AMR_GOALS_PATH = _REPO_ROOT / "configs" / "amr_benchmark" / "retune" / "goals.json"


def resolve_pair_campaign_mode(
    goals: dict[str, Any] | None,
    model: str | None = None,
    dataset: str | None = None,
) -> str:
    """Resolve ``paper_exact`` vs ``approximate`` for a (model, dataset) pair.

    Priority: pair_campaign_modes → tier A (own methods) → campaign_mode /
    default_campaign_mode. Tier B AMR-Benchmark ports default to approximate
    when ``campaign_mode`` is ``tiered``.
    """
    goals = goals or {}
    if model and dataset:
        key = f"{model.lower()}/{dataset}"
        override = goals.get("pair_campaign_modes", {}).get(key)
        if override in ("paper_exact", "approximate"):
            return override
        own = {
            m.lower()
            for m in goals.get("tiers", {})
            .get("A_own_methods", {})
            .get("models", ["mldnn", "fastmldnn", "hcgdnn"])
        }
        if model.lower() in own:
            return str(
                goals.get("tiers", {})
                .get("A_own_methods", {})
                .get("campaign_mode", "paper_exact")
            )
        if goals.get("campaign_mode") == "tiered":
            return str(
                goals.get("tiers", {})
                .get("B_amr_benchmark_ports", {})
                .get(
                    "campaign_mode",
                    goals.get("default_campaign_mode", "approximate"),
                )
            )
    mode = goals.get("campaign_mode", "paper_exact")
    if mode == "tiered":
        return str(goals.get("default_campaign_mode", "approximate"))
    return str(mode)


def resolve_amr_paper_exact(
    goals: dict[str, Any] | None = None,
    *,
    paper_exact: bool | None = None,
    model: str | None = None,
    dataset: str | None = None,
) -> bool:
    """Return whether campaign success uses paper-exact targets (no tolerance)."""
    if paper_exact is not None:
        return paper_exact
    return resolve_pair_campaign_mode(goals, model, dataset) == "paper_exact"


def amr_paper_targets_for_pair(
    model: str,
    dataset: str,
    goals: dict[str, Any],
    *,
    fallback_overall: float | None = None,
    fallback_peak: float | None = None,
) -> tuple[float | None, float | None]:
    """Paper-exact targets for (model, dataset); matrix fallbacks when unset."""
    key = f"{model.lower()}/{dataset}"
    override = goals.get("paper_targets", {}).get(key, {})
    overall = override.get("overall", fallback_overall)
    peak = override.get("peak", fallback_peak)
    return (
        float(overall) if overall is not None else None,
        float(peak) if peak is not None else None,
    )


def amr_campaign_goal_met(
    overall: float | None,
    peak: float | None,
    target_overall: float | None,
    target_peak: float | None,
    *,
    paper_exact: bool = True,
    tolerances: dict[str, Any] | None = None,
) -> bool:
    """Campaign success check; paper_exact=True requires measured >= target."""
    if overall is None or peak is None:
        return False
    if target_overall is None or target_peak is None:
        return False
    tol = tolerances or {}
    if paper_exact:
        return overall >= target_overall and peak >= target_peak
    overall_ok = overall >= target_overall - float(tol.get("overall_pp", 1.5))
    peak_ok = peak >= target_peak - float(tol.get("peak_pp", 1.0))
    return overall_ok and peak_ok


def amr_job_campaign_goal_met(
    result: Any,
    goals: dict[str, Any],
    *,
    paper_exact: bool | None = None,
) -> bool:
    """Campaign goal for a finished JobResult (uses paper_targets overrides)."""
    spec = result.spec
    paper_exact = resolve_amr_paper_exact(
        goals,
        paper_exact=paper_exact,
        model=spec.model,
        dataset=spec.dataset,
    )
    tgt_o, tgt_p = amr_paper_targets_for_pair(
        spec.model,
        spec.dataset,
        goals,
        fallback_overall=spec.target_overall,
        fallback_peak=spec.target_peak,
    )
    return amr_campaign_goal_met(
        result.overall_acc,
        result.peak_acc,
        tgt_o,
        tgt_p,
        paper_exact=paper_exact,
        tolerances=goals.get("tolerances"),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_goal_status(path: Path, payload: dict[str, Any], dry_run: bool = False) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if dry_run:
        print(f"[DRY-RUN] would write {path}:\n{json.dumps(payload, indent=2)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_amr_tracking_table(
    tracking_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Parse rows from ``accuracy_tracking.md`` auto table."""
    tracking_path = tracking_path or (
        _REPO_ROOT / "docs" / "amr_benchmark" / "accuracy_tracking.md"
    )
    if not tracking_path.is_file():
        return []

    text = tracking_path.read_text()
    begin = "<!-- AMR_BENCHMARK_AUTO_TABLE_BEGIN -->"
    end = "<!-- AMR_BENCHMARK_AUTO_TABLE_END -->"
    if begin in text and end in text:
        text = text.split(begin, 1)[1].split(end, 1)[0]

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 13 or parts[1] == "Model":
            continue
        status = parts[11].strip("`").strip()
        try:
            rows.append(
                dict(
                    model=parts[1].lower(),
                    dataset_label=parts[2],
                    dataset=AMR_LABEL_TO_DATASET.get(parts[2], parts[2]),
                    target_overall=float(parts[5]),
                    overall=float(parts[6]),
                    target_peak=float(parts[7]),
                    peak=float(parts[8]),
                    status=status,
                )
            )
        except ValueError:
            continue
    return rows


def amr_tracking_summary(tracking_path: Path | None = None) -> dict[str, Any]:
    rows = parse_amr_tracking_table(tracking_path)
    by_status: dict[str, int] = {}
    fails: list[dict[str, Any]] = []
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        if row["status"] == "fail":
            fails.append(
                dict(
                    model=row["model"],
                    dataset=row["dataset"],
                    overall=f"{row['overall']:.2f}",
                    peak=f"{row['peak']:.2f}",
                    target_overall=f"{row['target_overall']:.2f}",
                    target_peak=f"{row['target_peak']:.2f}",
                )
            )
    return dict(
        total_rows=len(rows),
        pass_count=by_status.get("pass", 0),
        fail_count=by_status.get("fail", 0),
        measured_count=by_status.get("measured", 0),
        error_count=by_status.get("error", 0),
        fails=fails,
    )


def group_manifest_by_pair(
    experiments: list[Any],
) -> dict[tuple[str, str], list[Any]]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for exp in experiments:
        key = (exp.model.lower(), exp.dataset)
        grouped.setdefault(key, []).append(exp)
    for variants in grouped.values():
        variants.sort(key=lambda e: (e.priority, e.variant))
    return grouped


def remaining_queue_from_manifest(
    grouped: dict[tuple[str, str], list[Any]],
    tracking_summary: dict[str, Any],
) -> list[dict[str, str]]:
    """Pairs still failing in tracking with variants not yet run."""
    fail_pairs = {(f["model"], f["dataset"]) for f in tracking_summary.get("fails", [])}
    queue: list[dict[str, str]] = []
    for (model, dataset), variants in sorted(grouped.items()):
        if (model, dataset) not in fail_pairs:
            continue
        for exp in variants:
            queue.append(
                dict(
                    model=model,
                    dataset=dataset,
                    variant=exp.variant,
                    experiment_id=exp.experiment_id,
                    priority=str(exp.priority),
                )
            )
    return queue


def _amc_top1_from_paper_pkl(work_dir: Path) -> float | None:
    """Compute OA% from ``tools/test.py`` output ``res/paper.pkl``."""
    import pickle

    pkl = work_dir / "res" / "paper.pkl"
    if not pkl.is_file():
        return None
    try:
        with pkl.open("rb") as fh:
            data = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    if not isinstance(data, dict) or "pps" not in data or "gts" not in data:
        return None
    try:
        import numpy as np

        pps = np.asarray(data["pps"])
        gts = np.asarray(data["gts"])
        if pps.size == 0 or gts.size == 0:
            return None
        return float(np.mean(np.argmax(pps, axis=1) == gts) * 100.0)
    except (ImportError, ValueError, TypeError, IndexError):
        return None


def _amc_top1_from_logs(work_dir: Path) -> float | None:
    """Parse ``overall accuracy: XX.XX%`` printed by ``tools/test.py``."""
    import re

    pat = re.compile(r"overall accuracy:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)
    best: tuple[float, float] | None = None  # (mtime, top1)
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".log", ".txt"} and path.name != "retune.log":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        matches = pat.findall(text)
        if not matches:
            continue
        top1 = float(matches[-1])
        mtime = path.stat().st_mtime
        if best is None or mtime >= best[0]:
            best = (mtime, top1)
    return None if best is None else best[1]


def parse_jdm_metrics_json(work_dir: Path, module: str) -> dict[str, float]:
    """Best-effort read of latest mmengine test JSON under *work_dir*.

    AMC ``tools/test.py`` does not emit mmengine metric JSON; it writes
    ``res/paper.pkl`` and prints ``overall accuracy``. Fall back to those
    when no ``accuracy/top1`` JSON is present.
    """
    if not work_dir.is_dir():
        return {}

    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in work_dir.rglob("*.json"):
        if path.name in ("snr_curve.json", "GOAL_STATUS.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        candidates.append((path.stat().st_mtime, data))

    for _, data in sorted(candidates, key=lambda x: x[0], reverse=True):
        if module == "detector" and "detection/mAP" in data:
            out = {"map": float(data["detection/mAP"])}
            if "detection/AP75" in data:
                out["ap75"] = float(data["detection/AP75"])
            return out
        if module == "amc" and "accuracy/top1" in data:
            top1 = float(data["accuracy/top1"])
            if top1 <= 1.0:
                top1 *= 100.0
            return {"top1_pct": top1}
        if module == "joint" and "detection/mAP" in data:
            return {"map": float(data["detection/mAP"])}

    if module == "amc":
        top1 = _amc_top1_from_paper_pkl(work_dir)
        if top1 is None:
            top1 = _amc_top1_from_logs(work_dir)
        if top1 is not None:
            return {"top1_pct": top1}
    return {}


def evaluate_jdm_goal(
    goals: dict[str, Any],
    module: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Return per-goal pass/fail for a single experiment module."""
    targets = goals.get("targets", {})
    baseline = goals.get("baseline_best", {})
    checks: list[dict[str, Any]] = []

    if module == "detector":
        spec = targets.get("detector", {})
        if not spec.get("active", True):
            return dict(module=module, skipped=True, checks=[])
        measured_map = metrics.get("map")
        measured_ap75 = metrics.get("ap75")
        map_min = float(spec.get("map_min", baseline.get("detector_map", 0.0)))
        ap75_min = float(spec.get("ap75_min", baseline.get("detector_ap75", 0.0)))
        checks.append(
            dict(
                metric="detector_map",
                target=map_min,
                measured=measured_map,
                met=measured_map is not None and measured_map >= map_min,
            )
        )
        if ap75_min > 0:
            checks.append(
                dict(
                    metric="detector_ap75",
                    target=ap75_min,
                    measured=measured_ap75,
                    met=measured_ap75 is not None and measured_ap75 >= ap75_min,
                )
            )
    elif module == "amc":
        spec = targets.get("amc_proposal", {})
        if not spec.get("active", True):
            return dict(module=module, skipped=True, checks=[])
        top1 = metrics.get("top1_pct")
        top1_min = float(spec.get("top1_min_pct", 80.0))
        checks.append(
            dict(
                metric="amc_proposal_top1_pct",
                target=top1_min,
                measured=top1,
                met=top1 is not None and top1 >= top1_min,
            )
        )
    elif module == "joint":
        spec = targets.get("joint", {})
        if not spec.get("active", False):
            return dict(module=module, skipped=True, checks=[])
        measured = metrics.get("map")
        map_min = float(spec.get("map_min", 0.60))
        checks.append(
            dict(
                metric="joint_map_fuse",
                target=map_min,
                measured=measured,
                met=measured is not None and measured >= map_min,
            )
        )

    goal_met = bool(checks) and all(c["met"] for c in checks)
    return dict(module=module, skipped=not checks, checks=checks, goal_met=goal_met)


def _jdm_best_metric_over_globs(
    retune_root: Path,
    module: str,
    metric_key: str,
    source_globs: list[str],
) -> tuple[float | None, str | None]:
    """Best (max) value of *metric_key* for *module* across dirs matching globs.

    Used for dual-protocol scoring: ideal globs point at v1 test-only eval dirs,
    simulate globs at Real/Real_awgn (v104+v105–v124) eval dirs — not mixed-all.
    """
    best: float | None = None
    best_dir: str | None = None
    seen: set[Path] = set()
    for pattern in source_globs:
        for variant_dir in sorted(retune_root.glob(pattern)):
            if not variant_dir.is_dir() or variant_dir in seen:
                continue
            seen.add(variant_dir)
            m = parse_jdm_metrics_json(variant_dir, module)
            val = m.get(metric_key)
            if val is not None and (best is None or val > best):
                best = val
                try:
                    best_dir = str(variant_dir.relative_to(_REPO_ROOT))
                except ValueError:
                    best_dir = str(variant_dir)
    return best, best_dir


def _jdm_dual_protocol_items(
    goal_prefix: str,
    module: str,
    metric_key: str,
    spec: dict[str, Any],
    retune_root: Path,
    baseline_value: float | None,
) -> list[dict[str, Any]]:
    """Build one checklist item per declared protocol (ideal / simulate).

    Falls back to a single legacy item when no ``protocols`` block is present.
    """
    protocols = spec.get("protocols")
    if not protocols:
        return []
    items: list[dict[str, Any]] = []
    for proto_name, proto in protocols.items():
        target = proto.get(f"{metric_key}_min")
        if target is None:
            continue
        globs = proto.get("source_globs", [])
        measured, source = _jdm_best_metric_over_globs(
            retune_root, module, metric_key, globs
        )
        # Never fall back to mixed-test baseline for simulate: after the
        # 2026-07-24 protocol tighten, baseline mixed numbers would falsely
        # mark Real/Real_awgn simulate goals as met.
        _ = baseline_value  # kept for call-site compat; unused by design
        items.append(
            dict(
                goal=f"{goal_prefix}_{proto_name}",
                priority=spec.get("priority", "P0"),
                protocol=proto.get("protocol", proto_name),
                target=target,
                best_measured=measured,
                source=source or "unmeasured",
                met=measured is not None and measured >= float(target),
                active=True,
            )
        )
    return items


def jdm_goal_checklist(
    goals_path: Path,
    retune_root: Path | None = None,
) -> dict[str, Any]:
    goals = load_json(goals_path)
    retune_root = retune_root or (_REPO_ROOT / "work_dirs" / "jdm" / "retune")
    baseline = goals.get("baseline_best", {})
    targets = goals.get("targets", {})
    checklist: list[dict[str, Any]] = []

    # --- Dual-protocol detector + joint (2026-07-23) ---
    det_spec_dp = targets.get("detector", {})
    joint_spec_dp = targets.get("joint", {})
    dual_used = False
    if det_spec_dp.get("active", True) and det_spec_dp.get("protocols"):
        dual_used = True
        checklist.extend(
            _jdm_dual_protocol_items(
                "detector_map", "detector", "map", det_spec_dp, retune_root,
                baseline.get("detector_map"),
            )
        )
        checklist.extend(
            _jdm_dual_protocol_items(
                "detector_ap75", "detector", "ap75", det_spec_dp, retune_root,
                baseline.get("detector_ap75"),
            )
        )
    if joint_spec_dp.get("active", False) and joint_spec_dp.get("protocols"):
        dual_used = True
        checklist.extend(
            _jdm_dual_protocol_items(
                "joint_map", "joint", "map", joint_spec_dp, retune_root,
                baseline.get("joint_map_fuse"),
            )
        )

    det_spec = {} if dual_used else targets.get("detector", {})
    if det_spec and det_spec.get("active", True):
        best_map = baseline.get("detector_map")
        best_ap75 = baseline.get("detector_ap75")
        best_dir = None
        best_ap75_dir = None
        # Include wave1 30-ep, wave2/3/3b, and any future det_* retune dirs.
        det_dirs: set[Path] = set()
        for pattern in ("det_30ep_*", "det_wave*", "det_*"):
            det_dirs.update(p for p in retune_root.glob(pattern) if p.is_dir())
        for variant_dir in sorted(det_dirs, key=lambda p: p.name):
            m = parse_jdm_metrics_json(variant_dir, "detector")
            if m.get("map") is not None and (best_map is None or m["map"] > best_map):
                best_map = m["map"]
                best_dir = str(variant_dir.relative_to(_REPO_ROOT))
            if m.get("ap75") is not None and (
                    best_ap75 is None or m["ap75"] > best_ap75):
                best_ap75 = m["ap75"]
                best_ap75_dir = str(variant_dir.relative_to(_REPO_ROOT))
        checklist.append(
            dict(
                goal="detector_map",
                priority=det_spec.get("priority", "P0"),
                target=det_spec.get("map_min"),
                best_measured=best_map,
                source=best_dir or "baseline",
                met=best_map is not None and best_map >= float(det_spec.get("map_min", 0)),
                active=True,
            )
        )
        if float(det_spec.get("ap75_min", 0) or 0) > 0:
            checklist.append(
                dict(
                    goal="detector_ap75",
                    priority=det_spec.get("priority", "P0"),
                    target=det_spec.get("ap75_min"),
                    best_measured=best_ap75,
                    source=best_ap75_dir or "baseline",
                    met=best_ap75 is not None
                    and best_ap75 >= float(det_spec.get("ap75_min", 0)),
                    active=True,
                )
            )

    amc_spec = targets.get("amc_proposal", {})
    if amc_spec.get("active", True):
        best_top1 = baseline.get("amc_proposal_top1_pct")
        best_amc_dir: Path | None = None
        amc_dirs: set[Path] = set()
        for pattern in ("amc_detprops*", "amc_*"):
            amc_dirs.update(p for p in retune_root.glob(pattern) if p.is_dir())
        for variant_dir in sorted(amc_dirs, key=lambda p: p.name):
            m = parse_jdm_metrics_json(variant_dir, "amc")
            if m.get("top1_pct") is not None and (
                best_top1 is None or m["top1_pct"] > best_top1
            ):
                best_top1 = m["top1_pct"]
                best_amc_dir = variant_dir
        checklist.append(
            dict(
                goal="amc_proposal_top1_pct",
                priority=amc_spec.get("priority", "P1"),
                target=amc_spec.get("top1_min_pct"),
                best_measured=best_top1,
                source=(
                    str(best_amc_dir.relative_to(_REPO_ROOT))
                    if best_amc_dir is not None
                    else "baseline"
                ),
                met=best_top1 is not None and best_top1 >= float(amc_spec.get("top1_min_pct", 80)),
                active=True,
            )
        )

    joint_spec = {} if dual_used else targets.get("joint", {})
    if joint_spec.get("active", False):
        best_joint = baseline.get("joint_map_fuse")
        best_joint_dir = None
        joint_dirs: set[Path] = set()
        for pattern in ("joint_*", "jdm_joint_*"):
            joint_dirs.update(p for p in retune_root.glob(pattern) if p.is_dir())
        # Also scan sibling fuse workdirs under work_dirs/jdm/
        jdm_root = retune_root.parent
        for pattern in ("jdm-joint*", "*fuse*"):
            joint_dirs.update(p for p in jdm_root.glob(pattern) if p.is_dir())
        for variant_dir in sorted(joint_dirs, key=lambda p: p.name):
            m = parse_jdm_metrics_json(variant_dir, "joint")
            if m.get("map") is not None and (
                    best_joint is None or m["map"] > best_joint):
                best_joint = m["map"]
                try:
                    best_joint_dir = str(variant_dir.relative_to(_REPO_ROOT))
                except ValueError:
                    best_joint_dir = str(variant_dir)
        checklist.append(
            dict(
                goal="joint_map_fuse",
                priority=joint_spec.get("priority", "P2"),
                target=joint_spec.get("map_min"),
                best_measured=best_joint,
                source=best_joint_dir
                or "work_dirs/jdm/retune/joint_wave3b_amc",
                met=best_joint is not None
                and best_joint >= float(joint_spec.get("map_min", 0.60)),
                active=True,
            )
        )

    active = [c for c in checklist if c.get("active")]
    met_count = sum(1 for c in active if c["met"])
    return dict(
        goals_file=str(goals_path.relative_to(_REPO_ROOT)),
        active_goals=len(active),
        goals_met=met_count,
        campaign_complete=len(active) > 0 and met_count == len(active),
        checklist=checklist,
        paper_notes=goals.get("paper_notes", []),
    )


@dataclass
class _ManifestPair:
    model: str
    dataset: str
    variant: str
    experiment_id: str
    priority: int


def _load_amr_manifest_pairs(path: Path) -> list[_ManifestPair]:
    data = load_json(path)
    pairs: list[_ManifestPair] = []
    for raw in data.get("experiments", []):
        pairs.append(
            _ManifestPair(
                model=raw["model"].lower(),
                dataset=raw["dataset"],
                variant=raw["variant"],
                experiment_id=raw.get("id", raw["variant"]),
                priority=raw.get("priority", 99),
            )
        )
    return pairs


def print_amr_goal_status(
    goals_path: Path,
    manifest_path: Path | None = None,
) -> None:
    goals = load_json(goals_path)
    summary = amr_tracking_summary()
    tolerances = goals.get("tolerances", {})
    paper_exact = resolve_amr_paper_exact(goals)
    print("=== AMR Goal Status ===")
    print(f"Tracking: {summary['pass_count']} pass, {summary['fail_count']} fail, "
          f"{summary['measured_count']} measured, {summary['error_count']} error "
          f"(total {summary['total_rows']})")
    print(f"Tracking pass rule: overall −{tolerances.get('overall_pp', 1.5)} pp, "
          f"peak −{tolerances.get('peak_pp', 1.0)} pp "
          f"(accuracy_tracking.md / run_migration.py::_classify)")
    mode = goals.get("campaign_mode", "paper_exact")
    print(
        f"Campaign mode: {mode} "
        f"(global default success = "
        f"{'paper target exactly' if paper_exact else 'tracking tolerance pass'})"
    )
    tiers = goals.get("tiers", {})
    if tiers:
        print("Tiers:")
        for tid, tspec in tiers.items():
            models = tspec.get("models", ["(all other)"])
            print(
                f"  {tid}: campaign_mode={tspec.get('campaign_mode')} "
                f"models={models}"
            )
    paper_targets = goals.get("paper_targets", {})
    if paper_targets:
        print("Paper-exact overrides:")
        for pair, spec in sorted(paper_targets.items()):
            print(f"  {pair}: overall ≥ {spec.get('overall')}, peak ≥ {spec.get('peak')}")
    print(
        "Campaign target: Tier A own methods → paper-exact; "
        "Tier B ports → approximate (tracking −1.5/−1.0); "
        f"{summary['fail_count']} tracking fails remain"
    )
    if manifest_path and manifest_path.is_file():
        pairs = _load_amr_manifest_pairs(manifest_path)
        grouped: dict[tuple[str, str], list[_ManifestPair]] = {}
        for p in pairs:
            grouped.setdefault((p.model, p.dataset), []).append(p)
        for variants in grouped.values():
            variants.sort(key=lambda e: (e.priority, e.variant))
        queue = remaining_queue_from_manifest(grouped, summary)
        print(f"Manifest queue for failing pairs: {len(queue)} variant(s)")
        for item in queue[:10]:
            print(f"  P{item['priority']} {item['model']}/{item['dataset']} "
                  f"→ {item['variant']}")
        if len(queue) > 10:
            print(f"  … and {len(queue) - 10} more")
    if summary["fail_count"] == 0:
        print("Campaign complete: no remaining fails in tracking table.")
    print(f"Goals config: {goals_path.relative_to(_REPO_ROOT)}")


def print_jdm_goal_status(
    goals_path: Path,
    retune_root: Path | None = None,
) -> None:
    status = jdm_goal_checklist(goals_path, retune_root)
    print("=== JDM Goal Status ===")
    print(f"Active goals: {status['goals_met']}/{status['active_goals']} met")
    for item in status["checklist"]:
        flag = "PASS" if item["met"] else "FAIL"
        measured = item["best_measured"]
        meas_s = f"{measured:.4f}" if isinstance(measured, float) and measured < 10 else measured
        print(f"  [{flag}] {item['goal']} ({item['priority']}): "
              f"target={item['target']} best={meas_s} ({item['source']})")
    for note in status.get("paper_notes", []):
        print(f"  note: {note}")
    if status["campaign_complete"]:
        print("Campaign complete: all active P0/P1 goals met.")
    else:
        pending = [c["goal"] for c in status["checklist"] if c.get("active") and not c["met"]]
        print(f"Pending: {', '.join(pending) or 'none'}")
    print(f"Goals config: {status['goals_file']}")
