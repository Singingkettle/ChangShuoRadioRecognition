#!/usr/bin/env bash
# N-GPU pool keepalive daemon for the 4xH100 workhorse (two-machine goal mode).
#
# Design rationale: the local gpu_keepalive.sh is a ~1900-line, deeply
# 2-GPU-asymmetric (GPU0-primary / GPU1-JDM) scheduler with a large JDM fallback
# ladder and AMR auto-seed. Rather than contort that (high stall-regression risk),
# this is a SMALL, N-GPU-native pool daemon that keeps two long-lived worker
# orchestrators alive over disjoint GPU lanes and expands one lane onto the
# other's freed GPUs when a lane's work is exhausted. Zero-idle guarantee: no GPU
# sits idle while EITHER lane has work.
#
#   AMR lane  (default GPUs 0,1,2): retune_model_siege.py over a partitioned queue
#   JDM lane  (default GPU 3):       escalation ladder of architecture-frozen
#                                    TRAINING rungs (JDM_LADDER), launched
#                                    directly on idle JDM GPUs (NOT via the
#                                    eval-idempotent ideal_fair_ladder, which
#                                    caused the "has work but nothing runnable"
#                                    idle stall). Evals are refreshed once per
#                                    new checkpoint when the ladder is exhausted.
#
# NOTE on the split: the JDM ideal-fair ladder is inherently SEQUENTIAL on one
# GPU (train det -> train AMC -> merge -> eval), so a 2-GPU JDM lane would leave
# one GPU idle. Defaulting AMR to 3 GPUs + JDM to 1 keeps all four H100s busy
# and still runs BOTH Tier-A AMR and JDM v1-fair on the workhorse. Set
# AMR_GPUS=0,1 JDM_GPUS=2,3 to force the plan's literal 2+2 split if desired.
#
# Env overrides:
#   REPO_ROOT   PYTHON   GPUS(=0,1,2,3)   AMR_GPUS(=0,1,2)   JDM_GPUS(=3)
#   QUEUE_REMOTE   POOL_INTERVAL(=120)
#
# Deploy (disconnect-safe) on the remote box:
#   cd $REPO_ROOT && setsid nohup bash tools/amr_benchmark/gpu_pool_keepalive.sh \
#     >> work_dirs/amr_benchmark_retune/pool.log 2>&1 < /dev/null &

set -uo pipefail

# Make CUDA logical indices match nvidia-smi physical indices. Without this,
# CUDA's default FASTEST_FIRST ordering can map CUDA_VISIBLE_DEVICES=0 to
# physical GPU 1 etc., so busy_gpus() (nvidia-smi based) and job placement
# (CUDA based) disagree — jobs pack onto already-busy GPUs while others idle.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

REPO="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
PY="${PYTHON:-/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
AMR_GPUS="${AMR_GPUS:-0,1,2}"
JDM_GPUS="${JDM_GPUS:-3}"
INTERVAL="${POOL_INTERVAL:-120}"
LOGDIR="${REPO}/work_dirs/amr_benchmark_retune"
POOL_LOG="${LOGDIR}/pool.log"
POOL_STATUS="${LOGDIR}/POOL_STATUS.json"
QUEUE_REMOTE="${QUEUE_REMOTE:-${REPO}/configs/amr_benchmark/retune/siege_queue_remote.json}"

# JDM escalation ladder (architecture-frozen TRAINING rungs, ordered). Each
# entry: <config>|<work_dir>|<done_glob>|<prereq_glob>. A rung is treated as:
#   DONE     if <work_dir>/<done_glob> matches (final-epoch checkpoint present),
#   BLOCKED  if <prereq_glob> is non-empty and matches nothing (SKIP, no churn),
#   RUNNING  if a tools/train.py proc references its <config>,
#   RUNNABLE otherwise.
# ROOT-CAUSE FIX: the JDM lane no longer relies on ideal_fair_ladder liveness
# (which is EVAL-idempotent and instantly exits once evals are cached, leaving
# GPUs idle-but-"has work"). Instead the daemon launches the NEXT runnable rung
# directly on each idle JDM GPU while JDM goals are unmet. Only when NO rung is
# runnable does the lane yield its GPUs and refresh evals once (stamped).
JDM_LADDER=(
    "configs/jdm/jdm-det_fft-csrd.py|work_dirs/jdm/retune/det_full_30ep|epoch_30.pth|"
    "configs/jdm/experiments/retune/det_full_60ep_lr1e3.py|work_dirs/jdm/retune/det_full_60ep_lr1e3|epoch_60.pth|"
    "configs/jdm/experiments/retune/det_full_90ep_lr1e3.py|work_dirs/jdm/retune/det_full_90ep_lr1e3|epoch_90.pth|"
    "configs/jdm/experiments/retune/det_full_120ep_lr1e3.py|work_dirs/jdm/retune/det_full_120ep_lr1e3|epoch_120.pth|"
    "configs/jdm/experiments/retune/det_full_90ep_lr5e4.py|work_dirs/jdm/retune/det_full_90ep_lr5e4|epoch_90.pth|"
    "configs/jdm/experiments/retune/amc_wave3b_detprops_60ep.py|work_dirs/jdm/retune/amc_wave3b_detprops_60ep|epoch_60.pth|work_dirs/jdm/amc_proposals/wave3b_5ep_lr1e3.json"
)
JDM_EVAL_STAMP="${REPO}/work_dirs/jdm/retune/.pool_last_eval_ckpt"

cd "${REPO}" || exit 1
mkdir -p "${LOGDIR}" "${REPO}/work_dirs/jdm/retune"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" >> "${POOL_LOG}"; }

_count_gpus() { echo "$1" | tr ',' '\n' | grep -c '[0-9]'; }

# Physical GPU ids (0-based, in nvidia-smi enumeration order) that have a live
# tools/train.py or tools/test_det.py compute process. Prints space-separated.
busy_gpus() {
    "${PY}" - <<'PY'
import subprocess
from pathlib import Path
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"], text=True)
except Exception:
    print("")
    raise SystemExit(0)
order, pid_gpu = [], {}
for ln in out.splitlines():
    ln = ln.strip()
    if not ln:
        continue
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) < 2:
        continue
    uuid, pid = parts[0], parts[1]
    if uuid not in order:
        order.append(uuid)
    try:
        pid_gpu[int(pid)] = order.index(uuid)
    except ValueError:
        pass
busy = set()
for pid, gpu in pid_gpu.items():
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
    except OSError:
        continue
    if "tools/train.py" in cmd or "tools/test_det.py" in cmd or "tools/test.py" in cmd:
        busy.add(gpu)
print(" ".join(str(g) for g in sorted(busy)))
PY
}

# True if a python orchestrator with cmdline matching $1 is alive (skip shells).
orchestrator_alive() {
    local pattern="$1" pid cmdline base exe
    while read -r pid; do
        [[ -z "${pid}" || "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ -z "${cmdline}" ]] && continue
        case "${cmdline}" in *pgrep*|*"gpu_pool_keepalive"*) continue ;; esac
        [[ "${cmdline}" == *"${pattern}"* ]] || continue
        exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
        base="$(basename "${exe}" 2>/dev/null || true)"
        case "${base}" in python|python3|python3.*) return 0 ;; esac
        case "${cmdline}" in python*" ${pattern}"*|python3*" ${pattern}"*|*"/python ${pattern}"*|*"/python3 ${pattern}"*) return 0 ;; esac
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
    return 1
}

amr_pending() {
    [[ -f "${QUEUE_REMOTE}" ]] || { echo 0; return; }
    "${PY}" - "${QUEUE_REMOTE}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
skip = {"passed", "exhausted", "skipped", "running"}
print(sum(1 for e in data.get("entries", []) if e.get("status", "pending") not in skip))
PY
}

# Next pending remote queue entry: id<TAB>manifest<TAB>model<TAB>dataset
amr_next() {
    "${PY}" - "${QUEUE_REMOTE}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
skip = {"passed", "exhausted", "skipped", "running"}
for e in sorted(data.get("entries", []), key=lambda e: e.get("priority", 99)):
    if e.get("status", "pending") in skip:
        continue
    print("\t".join([e.get("id", ""), e.get("manifest", ""),
                      e.get("model", ""), e.get("dataset", "")]))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

# JDM has more work if the goal checklist is not campaign_complete.
jdm_has_work() {
    "${PY}" - "${REPO}" <<'PY'
import sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "tools"))
try:
    from goal_mode_helpers import jdm_goal_checklist
    st = jdm_goal_checklist((repo / "configs/jdm/retune/goals.json"))
    print("no" if st.get("campaign_complete") else "yes")
except Exception:
    print("yes")
PY
}

launch_amr() {
    local gpus="$1" ncount pending line eid manifest model dataset
    pending="$(amr_pending)"
    [[ "${pending}" -gt 0 ]] || return 1
    line="$(amr_next 2>/dev/null || true)"
    [[ -n "${line}" ]] || return 1
    IFS=$'\t' read -r eid manifest model dataset <<< "${line}"
    ncount="$(_count_gpus "${gpus}")"
    log "ACTION: AMR siege lane on GPUs ${gpus} (max-parallel ${ncount}) queue=${QUEUE_REMOTE} next=${eid} ${model}/${dataset}"
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --queue "${QUEUE_REMOTE}" \
        --gpu "${gpus}" --max-parallel "${ncount}" \
        --until-pass --paper-exact --promote \
        >> "${LOGDIR}/siege_remote.log" 2>&1 &
    log "Launched AMR siege PID=$! gpus=${gpus}"
    return 0
}

# Physical GPU ids (space-separated) in lane $1 (csv) that are NOT in busy $2.
lane_idle_gpus() {
    local lane="$1" busy="$2" g
    for g in ${lane//,/ }; do
        [[ -n "${g}" ]] || continue
        case " ${busy} " in *" ${g} "*) ;; *) printf '%s ' "${g}" ;; esac
    done
}

# Split a ladder entry "cfg|wd|glob|prereq" into _R_CFG _R_WD _R_GLOB _R_PRE.
_parse_rung() {
    local IFS='|'
    read -r _R_CFG _R_WD _R_GLOB _R_PRE <<< "$1"
}

# Print the next runnable JDM training rung entry (or nothing). A rung is
# skipped if DONE (final ckpt present), BLOCKED (declared prereq missing), or
# RUNNING (a train proc already references its config).
jdm_next_rung() {
    local entry
    for entry in "${JDM_LADDER[@]}"; do
        _parse_rung "${entry}"
        compgen -G "${REPO}/${_R_WD}/${_R_GLOB}" >/dev/null 2>&1 && continue
        if [[ -n "${_R_PRE}" ]]; then
            compgen -G "${REPO}/${_R_PRE}" >/dev/null 2>&1 || continue
        fi
        pgrep -f "tools/train.py .*${_R_CFG}" >/dev/null 2>&1 && continue
        printf '%s\n' "${entry}"
        return 0
    done
    return 1
}

# Launch a single escalation rung (disconnect-safe) on physical GPU $1.
launch_jdm_rung() {
    local gpu="$1" entry="$2"
    _parse_rung "${entry}"
    log "ACTION: JDM escalation rung on GPU ${gpu}: ${_R_CFG} -> ${_R_WD}"
    CUDA_VISIBLE_DEVICES="${gpu}" setsid nohup "${PY}" tools/train.py "${_R_CFG}" \
        --work-dir "${_R_WD}" >> "${REPO}/${_R_WD}.log" 2>&1 < /dev/null &
    log "Launched JDM rung PID=$! gpu=${gpu} cfg=${_R_CFG}"
}

# One-shot (per new checkpoint) eval refresh so goal metrics track the newest
# full-data detector. Runs test_det.py into timestamped dirs matched by
# goal_mode_helpers source_globs. Stamped by ckpt path -> never churns.
jdm_eval_refresh() {
    local gpu="$1" newest ts ideal_wd sim_wd
    newest="$(REPO_ROOT="${REPO}" "${PY}" - <<'PY'
import glob, os
repo = os.environ.get("REPO_ROOT")
cands = [p for p in glob.glob(os.path.join(
    repo, "work_dirs/jdm/retune/det_full_*/best_detection_mAP_epoch_*.pth"))
    if os.path.isfile(p)]
print(max(cands, key=os.path.getmtime) if cands else "")
PY
)"
    [[ -n "${newest}" ]] || return 0
    [[ -f "${JDM_EVAL_STAMP}" && "$(cat "${JDM_EVAL_STAMP}" 2>/dev/null)" == "${newest}" ]] && return 0
    pgrep -f "tools/test_det.py" >/dev/null 2>&1 && return 0
    ts="$(date -u +%Y%m%d_%H%M%S)"
    ideal_wd="work_dirs/jdm/retune/eval_ideal_v1_det_testonly_${ts}"
    sim_wd="work_dirs/jdm/retune/det_simulate_eval_${ts}"
    log "ACTION: JDM eval refresh on GPU ${gpu} for newest ckpt ${newest}"
    CUDA_VISIBLE_DEVICES="${gpu}" setsid nohup bash -c "\
${PY} tools/test_det.py configs/jdm/experiments/retune/eval_ideal_v1_det_testonly.py '${newest}' --work-dir ${ideal_wd} >> ${ideal_wd}.log 2>&1; \
${PY} tools/test_det.py configs/jdm/jdm-det_fft-csrd.py '${newest}' --work-dir ${sim_wd} >> ${sim_wd}.log 2>&1" < /dev/null &
    printf '%s\n' "${newest}" > "${JDM_EVAL_STAMP}"
    log "Launched JDM eval refresh PID=$! gpu=${gpu} ideal=${ideal_wd} sim=${sim_wd}"
}

write_status() {
    local busy="$1" amr_alive="$2" jdm_alive="$3" pend="$4" jwork="$5"
    "${PY}" - "${POOL_STATUS}" "${GPUS}" "${AMR_GPUS}" "${JDM_GPUS}" \
        "${busy}" "${amr_alive}" "${jdm_alive}" "${pend}" "${jwork}" <<'PY'
import json, sys
from datetime import datetime, timezone
(path, gpus, amr_gpus, jdm_gpus, busy, amr_alive, jdm_alive, pend, jwork) = sys.argv[1:10]
all_g = [g for g in gpus.split(",") if g]
busy_set = set(busy.split())
idle = [g for g in all_g if g not in busy_set]
payload = dict(
    updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    gpus=all_g, amr_gpus=amr_gpus.split(","), jdm_gpus=jdm_gpus.split(","),
    busy_gpus=sorted(busy_set), idle_gpus=idle,
    amr_orchestrator_alive=(amr_alive == "1"),
    jdm_orchestrator_alive=(jdm_alive == "1"),
    amr_pending=int(pend), jdm_has_work=(jwork == "yes"),
)
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
PY
}

tick() {
    local busy amr_alive=0 jdm_alive=0 pend jwork amr_g="${AMR_GPUS}" jdm_g="${JDM_GPUS}"
    busy="$(busy_gpus)"
    orchestrator_alive "tools/amr_benchmark/retune_model_siege.py" && amr_alive=1
    orchestrator_alive "tools/jdm/ideal_fair_ladder.py" && jdm_alive=1
    pend="$(amr_pending)"
    jwork="$(jdm_has_work)"

    log "tick busy=[${busy}] amr_alive=${amr_alive} jdm_alive=${jdm_alive} amr_pending=${pend} jdm_work=${jwork}"
    write_status "${busy}" "${amr_alive}" "${jdm_alive}" "${pend}" "${jwork}"

    # Lane expansion: if one lane is out of work, the other may claim all GPUs.
    if [[ "${pend}" -eq 0 && "${jwork}" == "yes" ]]; then
        jdm_g="${GPUS}"
    fi
    if [[ "${jwork}" != "yes" && "${pend}" -gt 0 ]]; then
        amr_g="${GPUS}"
    fi

    # Ensure AMR lane alive when it has work.
    if [[ "${amr_alive}" -eq 0 && "${pend}" -gt 0 ]]; then
        launch_amr "${amr_g}" || log "AMR launch skipped (no launchable entry)"
    elif [[ "${pend}" -eq 0 ]]; then
        log "AMR queue exhausted (amr_pending=0) — no churn; JDM lane may claim AMR GPUs"
    fi

    # JDM lane: fill EACH idle JDM GPU with the next runnable escalation rung.
    # This is the anti-idle root-cause fix: we key off physical compute
    # occupancy (busy_gpus), not orchestrator liveness, so an "alive-but-
    # childless" orchestrator can no longer leave a GPU idle while goals unmet.
    if [[ "${jwork}" == "yes" ]]; then
        local idle_jdm g rung
        idle_jdm="$(lane_idle_gpus "${jdm_g}" "${busy}")"
        if [[ -n "${idle_jdm}" ]]; then
            for g in ${idle_jdm}; do
                rung="$(jdm_next_rung || true)"
                if [[ -n "${rung}" ]]; then
                    launch_jdm_rung "${g}" "${rung}"
                    sleep 3  # let the train proc register so next rung isn't double-assigned
                else
                    # No runnable training rung remains (all done/running/blocked).
                    # Use this idle GPU to refresh evals once, then stop (no churn).
                    jdm_eval_refresh "${g}"
                    log "JDM ladder exhausted (no runnable rung); GPUs freed for AMR/auto-seed"
                    break
                fi
            done
        fi
    fi

    if [[ "${pend}" -eq 0 && "${jwork}" != "yes" ]]; then
        log "IDLE: both lanes exhausted (amr_pending=0, jdm campaign complete) — nothing to launch"
    fi
}

# POOL_ONCE=1 runs a single tick and exits (used for testing and for the
# steward's anti-idle to kick one runnable rung without spawning a daemon).
if [[ "${POOL_ONCE:-0}" == "1" ]]; then
    log "gpu_pool_keepalive one-shot tick PID=$$ GPUS=${GPUS} queue=${QUEUE_REMOTE}"
    tick
    exit 0
fi

log "gpu_pool_keepalive started PID=$$ GPUS=${GPUS} AMR_GPUS=${AMR_GPUS} JDM_GPUS=${JDM_GPUS} interval=${INTERVAL}s queue=${QUEUE_REMOTE}"
tick
while true; do
    sleep "${INTERVAL}"
    tick
done
