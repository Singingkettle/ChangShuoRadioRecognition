#!/usr/bin/env python
"""Hourly two-machine steward for the ChangShuo reproduction campaign.

Runs unattended via cron on the LOCAL box (which has passwordless SSH to the
remote 4xH100 at 10.161.4.55). Its job, every hour:

  1. Keep every daemon alive (local gpu_keepalive + health_watchdog; remote
     gpu_pool_keepalive) — restart only when actually dead.
  2. ANTI-IDLE: if either box has an idle GPU while work remains, force the
     responsible daemon to re-fill (the exact "GPU idle but reproduction not
     advancing" failure the user keeps hitting).
  3. PROGRESS-DRIVEN STRATEGY: record the four JDM metrics + AMR pass/pending
     each hour. When a metric plateaus for several hours WHILE GPUs are busy,
     the current recipe has converged, so escalate strategy within the
     architecture freeze:
        - AMR queue empty  -> trigger the capped auto-seed (next Tier-A wave).
        - JDM ideal-det below target and 30ep done -> launch the longer
          det_full_60ep rung on the remote JDM GPU (once, guarded).
  4. Write STEWARD_STATUS.json + append steward_history.jsonl + log a verdict.

Every corrective action is idempotent and rate-limited (stamps) so the steward
can never spam waves/relaunches — the class of bug that caused prior stalls.

Cron (installed by tools/amr_benchmark/install_steward_cron.sh):
    0 * * * * cd <REPO> && <PY> tools/amr_benchmark/hourly_steward.py >> \
        work_dirs/amr_benchmark_retune/steward_cron.log 2>&1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", "/home/citybuster/Projects/ChangShuoRadioRecognition"))
PY = os.environ.get(
    "PYTHON", "/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python"
)
LOGDIR = REPO / "work_dirs" / "amr_benchmark_retune"
JDM_RETUNE = REPO / "work_dirs" / "jdm" / "retune"
LOG = LOGDIR / "steward.log"
STATUS = LOGDIR / "STEWARD_STATUS.json"
HIST = LOGDIR / "steward_history.jsonl"
STAMPS = LOGDIR / "steward_stamps.json"

REMOTE = os.environ.get("REMOTE_HOST", "citybuster@10.161.4.55")
REMOTE_REPO = os.environ.get("REMOTE_REPO", "/home/citybuster/Projects/ChangShuoRadioRecognition")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", REMOTE]

# Plateau = no improvement in any JDM metric across this many consecutive hourly
# samples while GPUs were busy.
PLATEAU_HOURS = 3
# A given escalation action fires at most once per this many seconds.
NUDGE_MIN_INTERVAL_SEC = 6 * 3600
# The idempotent pool one-shot kick (fills idle GPUs with the next runnable rung)
# is guarded but cheap; allow it more often than heavy escalations so an
# idle-with-work stall is corrected promptly without spamming.
POOL_ONESHOT_MIN_INTERVAL_SEC = 20 * 60


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(f"[{now_iso()}] {msg}\n")
    print(f"[{now_iso()}] {msg}", flush=True)


def sh(cmd: list[str] | str, timeout: int = 60) -> tuple[int, str]:
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as e:
        return 1, str(e)


def ssh(remote_cmd: str, timeout: int = 60) -> tuple[int, str]:
    return sh(SSH + [remote_cmd], timeout=timeout)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def stamp_ok(key: str, min_interval_sec: int = NUDGE_MIN_INTERVAL_SEC) -> bool:
    """True if this nudge may fire now (respecting its rate limit)."""
    stamps = load_json(STAMPS, {})
    prev = stamps.get(key)
    if not prev:
        return True
    try:
        prev_t = datetime.fromisoformat(prev.replace("Z", "+00:00"))
    except ValueError:
        return True
    age = (datetime.now(timezone.utc) - prev_t).total_seconds()
    return age >= min_interval_sec


def stamp_set(key: str) -> None:
    stamps = load_json(STAMPS, {})
    stamps[key] = now_iso()
    save_json(STAMPS, stamps)


# --- Daemon liveness ---------------------------------------------------------

def ensure_local_daemons(actions: list[str]) -> None:
    for d in ("gpu_keepalive.sh", "health_watchdog.sh"):
        rc, _ = sh(f"pgrep -f 'tools/amr_benchmark/{d}' >/dev/null")
        if rc != 0:
            log(f"RESTART local daemon {d} (was dead)")
            sh(
                f"cd {REPO} && setsid nohup bash tools/amr_benchmark/{d} "
                f">> work_dirs/amr_benchmark_retune/{d.replace('.sh', '')}.log 2>&1 < /dev/null &"
            )
            actions.append(f"restart_local_{d}")


def ensure_remote_pool(actions: list[str]) -> bool:
    rc, _ = ssh("pgrep -f gpu_pool_keepalive.sh >/dev/null")
    if rc != 0:
        log("RESTART remote pool daemon (was dead)")
        ssh(
            f"cd {REMOTE_REPO} && setsid nohup bash tools/amr_benchmark/gpu_pool_keepalive.sh "
            ">> work_dirs/amr_benchmark_retune/pool.log 2>&1 < /dev/null &"
        )
        actions.append("restart_remote_pool")
        return False
    return True


# --- Metrics ------------------------------------------------------------------

def jdm_metrics() -> dict:
    code = (
        "import json,sys;sys.path.insert(0,'tools');"
        "from goal_mode_helpers import jdm_goal_checklist;"
        "from pathlib import Path;"
        "st=jdm_goal_checklist(Path('configs/jdm/retune/goals.json').resolve());"
        "print(json.dumps({c['goal']:c['best_measured'] for c in st['checklist']}|"
        "{'_complete':st['campaign_complete'],'_met':st['goals_met'],'_active':st['active_goals']}))"
    )
    rc, out = sh([PY, "-c", code], timeout=60)
    line = [l for l in out.splitlines() if l.strip().startswith("{")]
    return json.loads(line[-1]) if line else {}


def remote_pool_status() -> dict:
    rc, out = ssh(
        f"cat {REMOTE_REPO}/work_dirs/amr_benchmark_retune/POOL_STATUS.json 2>/dev/null"
    )
    return load_json_str(out)


def load_json_str(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def remote_gpu_idle() -> tuple[int, int]:
    """Return (idle_count, total) via nvidia-smi util<=1%."""
    rc, out = ssh(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null"
    )
    utils = [int(x) for x in out.split() if x.strip().isdigit()]
    idle = sum(1 for u in utils if u <= 1)
    return idle, len(utils)


def local_gpu_idle() -> tuple[int, int]:
    rc, out = sh(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null"
    )
    utils = [int(x) for x in out.split() if x.strip().isdigit()]
    idle = sum(1 for u in utils if u <= 1)
    return idle, len(utils)


# --- Anti-idle + strategy -----------------------------------------------------

def remote_amr_pending() -> int:
    rc, out = ssh(
        f"cd {REMOTE_REPO} && {PY} - <<'PY'\n"
        "import json;from pathlib import Path\n"
        "d=json.loads(Path('configs/amr_benchmark/retune/siege_queue_remote.json').read_text())\n"
        "skip={'passed','exhausted','skipped','running'}\n"
        "print(sum(1 for e in d.get('entries',[]) if e.get('status','pending') not in skip))\n"
        "PY"
    )
    for tok in out.split():
        if tok.strip().isdigit():
            return int(tok)
    return -1


def kick_pool_once(reason: str, actions: list[str]) -> None:
    """Run ONE idempotent pool tick on the remote so idle GPUs are immediately
    filled with the next runnable JDM escalation rung (or AMR entry).

    This reuses the hardened gpu_pool_keepalive.sh ladder logic (single source
    of truth) rather than duplicating rung selection here. The tick guards every
    launch by physical GPU occupancy + checkpoint/proc existence, so it can
    neither double-launch nor spam. Additionally rate-limited by a stamp so a
    burst of triggers in one steward run collapses to a single kick.
    """
    if not stamp_ok("pool_oneshot", POOL_ONESHOT_MIN_INTERVAL_SEC):
        return
    log(f"KICK remote pool one-shot tick (reason={reason})")
    ensure_remote_pool(actions)  # make sure the persistent daemon is alive too
    ssh(
        f"cd {REMOTE_REPO} && POOL_ONCE=1 bash tools/amr_benchmark/gpu_pool_keepalive.sh "
        ">> work_dirs/amr_benchmark_retune/pool.log 2>&1"
    )
    stamp_set("pool_oneshot")
    actions.append(f"kick_pool_once:{reason}")


def anti_idle(pool: dict, r_idle: int, metrics: dict, actions: list[str]) -> None:
    """Take REAL corrective action on an idle-with-work stall.

    Two independent idle signals are honored so a wedged / alive-but-childless
    pool daemon cannot silently strand GPUs:
      * pool POOL_STATUS.idle_gpus (derived from compute-proc occupancy), and
      * nvidia-smi utilization idle count (r_idle), which catches the case where
        the daemon is alive but has no child train/test proc on a GPU.
    When either fires with work remaining we kick a one-shot pool tick, which
    launches the next runnable rung directly (not just a log/nudge).
    """
    idle_gpus = pool.get("idle_gpus") or []
    amr_pending = pool.get("amr_pending", 0)
    jdm_work = pool.get("jdm_has_work", False)
    jdm_unmet = jdm_work or (metrics.get("_complete") is False)
    idle = bool(idle_gpus) or r_idle >= 1
    if idle and (amr_pending > 0 or jdm_unmet):
        log(
            f"ANTI-IDLE remote: pool_idle={idle_gpus} nvsmi_idle={r_idle} "
            f"amr_pending={amr_pending} jdm_work={jdm_work} — kicking pool to fill GPUs"
        )
        kick_pool_once("anti_idle", actions)


def escalate_amr(actions: list[str]) -> None:
    if remote_amr_pending() != 0:
        return
    if not stamp_ok("amr_autoseed"):
        return
    log("STRATEGY: remote AMR queue empty — triggering capped auto-seed of next Tier-A wave")
    ssh(
        f"cd {REMOTE_REPO} && GPU_KEEPALIVE_AUTO_SEED_ONCE=1 "
        f"QUEUE_FULL={REMOTE_REPO}/configs/amr_benchmark/retune/siege_queue_remote.json "
        "bash tools/amr_benchmark/gpu_keepalive.sh 2>&1 | tail -3"
    )
    stamp_set("amr_autoseed")
    actions.append("escalate_amr_autoseed")


def escalate_jdm(metrics: dict, actions: list[str]) -> None:
    """On plateau with the ideal-det goal unmet, advance the JDM escalation
    ladder. The ladder (JDM_LADDER in gpu_pool_keepalive.sh) is the single
    source of truth for which architecture-frozen rung comes next and which GPU
    is free, so we delegate to the idempotent pool tick instead of launching a
    specific rung here (which risked co-locating two trainers on one GPU)."""
    ideal = metrics.get("detector_map_ideal")
    if ideal is None or ideal >= 0.91:
        return
    log(
        f"STRATEGY: JDM ideal-det {ideal:.4f} < 0.91 on plateau — advancing "
        "escalation ladder via pool tick (architecture freeze)"
    )
    kick_pool_once("plateau_jdm", actions)


# --- Wave-12 follow-ups --------------------------------------------------------

# Historical Tier-A bests; a wave-12 test result above these is a NEW BEST and
# gets logged loudly so the next strategy review starts from it.
HIST_BEST = {"fastmldnn": 61.301, "hcgdnn": 63.39}


def _log_w12_report(host: str, rep: dict, actions: list[str]) -> None:
    for r in rep.get("tested", []):
        log(f"W12[{host}] launched post-train TEST: {r['variant']} ckpt={r['ckpt']} gpu={r['gpu']}")
        actions.append(f"w12_test:{host}:{r['variant']}")
    for r in rep.get("launched", []):
        log(f"W12[{host}] CHAINED stage-2 from stage-1 best: {r}")
        actions.append(f"w12_chain:{host}:{r['variant']}")
    for r in rep.get("done", []):
        top1, model = r.get("top1"), r.get("model", "?")
        if isinstance(top1, (int, float)) and top1 > HIST_BEST.get(model, 1e9):
            log(f"W12[{host}] *** NEW BEST {model}: {r['variant']} top1={top1:.2f} "
                f"(prev {HIST_BEST[model]}) ***")
    for w in rep.get("warnings", []):
        log(f"W12[{host}] warn: {w}")


def wave12_followups(actions: list[str]) -> None:
    """Run the wave-12 follow-up executor on BOTH boxes: test finished variants,
    chain the author stage-1 -> stage-2 pipeline, and pull remote test metrics
    back to the local tree so goal tracking sees them."""
    # Local box.
    rc, out = sh([PY, str(REPO / "tools/amr_benchmark/wave12_followup.py")], timeout=120)
    lines = [l for l in out.splitlines() if l.strip().startswith("{")]
    if lines:
        _log_w12_report("local", load_json_str(lines[-1]), actions)

    # Remote box: sync the executor (cheap, keeps it current), run, pull metrics.
    sh(f"scp -o BatchMode=yes {REPO}/tools/amr_benchmark/wave12_followup.py "
       f"{REMOTE}:{REMOTE_REPO}/tools/amr_benchmark/ >/dev/null 2>&1")
    rc, out = ssh(f"cd {REMOTE_REPO} && {PY} tools/amr_benchmark/wave12_followup.py", timeout=180)
    lines = [l for l in out.splitlines() if l.strip().startswith("{")]
    if not lines:
        log(f"W12[remote] executor produced no report (rc={rc})")
        return
    rep = load_json_str(lines[-1])
    _log_w12_report("remote", rep, actions)
    for r in rep.get("done", []):
        model = r.get("model")
        if not model:
            continue
        rel = f"work_dirs/amr_benchmark_retune/{model}/deepsig201610A/{r['variant']}/amc_test_metrics.json"
        local_path = REPO / rel
        if not local_path.is_file():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sh(f"scp -o BatchMode=yes {REMOTE}:{REMOTE_REPO}/{rel} {local_path} >/dev/null 2>&1")
            if local_path.is_file():
                log(f"W12 pulled remote metrics -> {rel}")


# --- Collapse detection ---------------------------------------------------------

# A run is "collapsed" when it has trained well past warm-up yet validation
# accuracy is still at chance level (e.g. the w24 lstm2-Hisar run that sat at
# 9.6% for 150 epochs while occupying a GPU). Alert loudly: occupied-but-
# useless GPUs are invisible to pure busy/idle checks.
COLLAPSE_MIN_EPOCH = 15
COLLAPSE_TOP1_PCT = 12.0

_COLLAPSE_SNIPPET = (
    "cd {repo} && find work_dirs/amr_benchmark_retune work_dirs/jdm/retune "
    "-name '*.log' -mmin -40 2>/dev/null | while read -r f; do "
    "tail -c 400000 \"$f\" | grep -oE 'Epoch\\(val\\) *\\[[0-9]+\\].*accuracy/top1: [0-9.]+' "
    "| tail -n 1 | sed \"s|^|$f\\||\"; done"
)


def _parse_collapse_lines(host: str, out: str) -> list[dict]:
    import re

    bad = []
    for ln in out.splitlines():
        if "|" not in ln:
            continue
        path, rest = ln.split("|", 1)
        m = re.search(r"Epoch\(val\) *\[(\d+)\].*accuracy/top1: ([0-9.]+)", rest)
        if not m:
            continue
        epoch, top1 = int(m.group(1)), float(m.group(2))
        if epoch >= COLLAPSE_MIN_EPOCH and top1 < COLLAPSE_TOP1_PCT:
            bad.append(dict(host=host, log=path, epoch=epoch, top1=top1))
    return bad


def collapse_check(actions: list[str]) -> list[dict]:
    """Scan actively-written training logs on both boxes for collapsed runs."""
    bad = []
    _, out = sh(_COLLAPSE_SNIPPET.format(repo=REPO), timeout=60)
    bad += _parse_collapse_lines("local", out)
    _, out = ssh(_COLLAPSE_SNIPPET.format(repo=REMOTE_REPO), timeout=60)
    bad += _parse_collapse_lines("remote", out)
    for b in bad:
        log(
            f"*** COLLAPSED RUN [{b['host']}] {b['log']}: epoch {b['epoch']} "
            f"val top1 {b['top1']:.2f}% (chance-level; GPU busy but wasted) ***"
        )
        actions.append(f"collapse_alert:{b['host']}:{b['log']}")
    return bad


def detect_plateau(history: list[dict], current: dict) -> bool:
    metric_keys = [
        "detector_map_ideal", "detector_ap75_ideal", "joint_map_ideal",
        "amc_proposal_top1_pct",
    ]
    samples = (history + [current])[-(PLATEAU_HOURS + 1):]
    if len(samples) < PLATEAU_HOURS + 1:
        return False
    # Busy the whole window?
    if not all(s.get("remote_busy", 0) >= 1 for s in samples):
        return False
    for k in metric_keys:
        vals = [s.get("metrics", {}).get(k) for s in samples]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) >= PLATEAU_HOURS + 1 and (max(vals) - min(vals)) > 1e-4:
            return False  # something moved -> not plateaued
    return True


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    log("=== steward tick start ===")

    ensure_local_daemons(actions)
    pool_alive = ensure_remote_pool(actions)

    metrics = jdm_metrics()
    pool = remote_pool_status()
    r_idle, r_total = remote_gpu_idle()
    l_idle, l_total = local_gpu_idle()
    remote_busy = max(0, r_total - r_idle)
    local_busy = max(0, l_total - l_idle)

    # Anti-idle: real corrective action (kick the ladder) on an idle-with-work
    # stall, using BOTH the pool status and nvidia-smi idle signals so an
    # alive-but-childless daemon cannot strand GPUs.
    anti_idle(pool, r_idle, metrics, actions)

    # Idle-with-unmet-goal trigger: even when no plateau is detected (the old
    # escalation was gated behind a busy-plateau, a catch-22 when GPUs sat
    # idle), if remote GPUs are idle AND JDM goals are unmet, advance the ladder
    # directly. Rate-limited + idempotent via the pool one-shot kick.
    jdm_unmet = bool(metrics) and metrics.get("_complete") is False
    if r_idle >= 1 and jdm_unmet:
        log(
            f"IDLE-WITH-UNMET-GOAL: remote_idle={r_idle}/{r_total} jdm_complete="
            f"{metrics.get('_complete')} — advancing JDM ladder"
        )
        kick_pool_once("idle_unmet_goal", actions)

    history = []
    if HIST.is_file():
        for ln in HIST.read_text().splitlines()[-24:]:
            try:
                history.append(json.loads(ln))
            except json.JSONDecodeError:
                pass

    current = dict(
        ts=now_iso(), metrics=metrics,
        remote_busy=remote_busy, remote_total=r_total, remote_idle=r_idle,
        local_busy=local_busy, local_total=l_total,
        amr_pending=pool.get("amr_pending"),
    )

    # Wave-12 follow-ups: post-train tests, stage-1 -> stage-2 chaining, and
    # pulling remote test metrics into the local tree (idempotent; file-gated).
    try:
        wave12_followups(actions)
    except Exception as e:  # never let follow-ups break the core steward loop
        log(f"W12 followups error: {e}")

    try:
        collapsed = collapse_check(actions)
    except Exception as e:
        log(f"collapse check error: {e}")
        collapsed = []

    plateau = detect_plateau(history, current)
    if plateau:
        log("PLATEAU detected (no JDM metric moved for %dh while busy) — escalating strategy" % PLATEAU_HOURS)
        escalate_jdm(metrics, actions)
    # AMR auto-seed is safe to check every hour (its own rate limit + queue cap).
    escalate_amr(actions)

    # Verdict
    verdict = "OK"
    if collapsed:
        verdict = f"COLLAPSED_RUNS({len(collapsed)})"
    elif (r_idle >= 1 and (pool.get("amr_pending", 0) or pool.get("jdm_has_work"))):
        verdict = "REMOTE_IDLE_WITH_WORK (nudged)"
    elif r_total and remote_busy == 0:
        verdict = "REMOTE_ALL_IDLE"
    elif plateau:
        verdict = "PLATEAU (escalated)"

    status = dict(
        updated_at=now_iso(),
        verdict=verdict,
        actions=actions,
        remote=dict(busy=remote_busy, total=r_total, idle=r_idle,
                    pool_alive=pool_alive, pool_status=pool),
        local=dict(busy=local_busy, total=l_total, idle=l_idle),
        jdm_metrics=metrics,
        plateau=plateau,
        collapsed_runs=collapsed,
    )
    save_json(STATUS, status)
    with HIST.open("a") as fh:
        fh.write(json.dumps(current) + "\n")
    log(
        f"verdict={verdict} remote_busy={remote_busy}/{r_total} local_busy={local_busy}/{l_total} "
        f"amr_pending={pool.get('amr_pending')} jdm_met={metrics.get('_met')}/{metrics.get('_active')} "
        f"ideal_det={metrics.get('detector_map_ideal')} actions={actions}"
    )
    log("=== steward tick end ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
