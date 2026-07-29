#!/usr/bin/env python3
"""Close phantom pending in siege_queue_full.json.

Stall class ``phantom_pending_force_blocked``:
  status stays ``pending``/stale ``running`` while identical fail re-logs (≥3)
  make selectors skip every entry → ``full_pending>0`` but nothing launches.

Reconcile rules (per model/dataset entry):
  - pass in retune_results.md → ``passed``
  - all **this entry's** manifest variants already recorded → ``exhausted``

Pair-level ≥3 identical fail re-logs must NOT close pending: that incorrectly
exhausts new wave recipes (e.g. wave6 FastMLDNN) on pairs that already have
old fail clusters. ``identical_fail_ge3`` remains available for force-block
selectors only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def pair_has_pass(text: str, model: str, dataset: str) -> bool:
    label = f"{model}/{dataset}"
    return bool(
        re.search(
            r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
            + re.escape(label)
            + r"\s*\|\s*`[^`]+`\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*`pass`\s*\|",
            text,
            re.I,
        )
    )


def identical_fail_ge3(text: str, model: str, dataset: str) -> bool:
    label = f"{model}/{dataset}"
    clusters: dict[tuple[str, int, int], int] = defaultdict(int)
    row_re = re.compile(
        r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
        + re.escape(label)
        + r"\s*\|\s*`([^`]+)`\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*`fail`\s*\|\s*`False`",
        re.I,
    )
    for line in text.splitlines():
        m = row_re.search(line)
        if not m:
            continue
        variant, overall_s, peak_s = m.group(1), m.group(2), m.group(3)
        try:
            clusters[
                (variant, int(round(float(overall_s) * 4)), int(round(float(peak_s) * 4)))
            ] += 1
        except ValueError:
            continue
    return any(n >= 3 for n in clusters.values())


def manifest_variants_have_results(
    repo: Path, text: str, manifest: str | None, model: str, dataset: str
) -> bool:
    if not manifest:
        return False
    mp = Path(manifest)
    if not mp.is_file():
        mp = repo / manifest
    if not mp.is_file():
        return False
    try:
        man = json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    exps = [
        e
        for e in man.get("experiments", [])
        if (e.get("model") or "").lower() == model.lower() and e.get("dataset") == dataset
    ]
    if not exps:
        exps = man.get("experiments", [])
    if not exps:
        return False
    label = f"{model}/{dataset}"
    for e in exps:
        variant = e.get("variant") or ""
        if not variant:
            continue
        if not re.search(
            r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
            + re.escape(label)
            + r"\s*\|\s*`"
            + re.escape(variant)
            + r"`\s*\|",
            text,
            re.I,
        ):
            return False
    return True


def _entry_protected_from_phantom_close(e: dict) -> bool:
    """Skip phantom-close for careful / capped waves (auto_seed_wave_spam_phantom)."""
    notes = e.get("notes") or ""
    if any(
        k in notes
        for k in (
            "manual_careful",
            "auto_seed_capped",
            "reopened_careful",
            "careful_running",
        )
    ):
        return True
    # Wave-unique variant suffixes (_wN) are intentional new work — never
    # exhaust them just because a bare-name ancestor variant already has rows.
    man = e.get("manifest") or ""
    if not man:
        return False
    mp = Path(man)
    if not mp.is_file():
        mp = Path("/home/citybuster/Projects/ChangShuoRadioRecognition") / man
    if not mp.is_file():
        return False
    try:
        exps = json.loads(mp.read_text()).get("experiments") or []
    except (OSError, json.JSONDecodeError):
        return False
    return bool(exps) and all("_w" in str(x.get("variant") or "") for x in exps)


def reconcile(queue_path: Path, results_md: Path, repo: Path) -> list[str]:
    data = json.loads(queue_path.read_text())
    text = results_md.read_text(errors="replace") if results_md.is_file() else ""
    changed: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for e in data.get("entries", []):
        model = (e.get("model") or "").lower()
        dataset = e.get("dataset") or ""
        status = e.get("status", "pending")
        if not model or not dataset:
            continue
        if status == "running":
            if _entry_protected_from_phantom_close(e):
                continue
            if pair_has_pass(text, model, dataset):
                e["status"] = "passed"
                e["notes"] = (e.get("notes") or "") + " | reconcile: pass → passed"
                changed.append(f"{e.get('id')}:running->passed")
            elif manifest_variants_have_results(
                repo, text, e.get("manifest"), model, dataset
            ):
                # Only close when THIS entry's manifest variants are all recorded.
                # Do NOT use pair-level identical_fail_ge3 — that wrongly exhausts
                # new wave recipes (wave6) on pairs that already have ≥3 old fails.
                e["status"] = "exhausted"
                e["exhausted_at"] = now
                e["notes"] = (e.get("notes") or "") + " | reconcile: stale running → exhausted"
                changed.append(f"{e.get('id')}:running->exhausted")
            continue
        if status != "pending":
            continue
        if _entry_protected_from_phantom_close(e):
            continue
        if pair_has_pass(text, model, dataset):
            e["status"] = "passed"
            e["notes"] = (e.get("notes") or "") + " | reconcile: pass → passed"
            changed.append(f"{e.get('id')}:pending->passed")
        elif manifest_variants_have_results(
            repo, text, e.get("manifest"), model, dataset
        ):
            e["status"] = "exhausted"
            e["exhausted_at"] = now
            e["notes"] = (e.get("notes") or "") + " | reconcile: phantom pending → exhausted"
            changed.append(f"{e.get('id')}:pending->exhausted")

    if changed:
        data["updated_at"] = now
        data["notes"] = (data.get("notes") or "") + " | reconcile_phantom_pending " + ",".join(
            changed
        )
        queue_path.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--queue-full",
        type=Path,
        default=Path("configs/amr_benchmark/retune/siege_queue_full.json"),
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("docs/amr_benchmark/retune_results.md"),
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("/home/citybuster/Projects/ChangShuoRadioRecognition"),
    )
    args = ap.parse_args()
    changed = reconcile(args.queue_full, args.results, args.repo)
    print(";".join(changed) if changed else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
