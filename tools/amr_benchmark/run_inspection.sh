#!/usr/bin/env bash
# One-shot AMR/JDM/GPU health inspection for cron or manual use.
# Prints a human-readable report to stdout and appends the same to health.log.
#
# Usage:
#   bash tools/amr_benchmark/run_inspection.sh
#
# Cron (every 10 min, optional):
#   */10 * * * * cd /home/citybuster/Projects/ChangShuoRadioRecognition && \
#     bash tools/amr_benchmark/run_inspection.sh \
#     >> work_dirs/amr_benchmark_retune/inspection.log 2>&1

set -uo pipefail

REPO="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
PY="${PYTHON:-/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python}"
LOGDIR="${REPO}/work_dirs/amr_benchmark_retune"
HEALTH_LOG="${LOGDIR}/health.log"
HEALTH_STATUS="${LOGDIR}/HEALTH_STATUS.json"
QUEUE="${REPO}/configs/amr_benchmark/retune/siege_queue.json"
AMR_MANIFEST="${REPO}/configs/amr_benchmark/retune/wave1_manifest.json"

cd "${REPO}" || exit 1
mkdir -p "${LOGDIR}"

CRITICAL=0
REPORT=""

append() {
    REPORT+="$1"$'\n'
}

# --- Probes (shared with health_watchdog.sh) ---

count_trains_per_gpu() {
    "${PY}" - <<'PY'
import subprocess, sys
from collections import defaultdict

repo = "/home/citybuster/Projects/ChangShuoRadioRecognition"
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
except subprocess.CalledProcessError:
    print("0 0")
    sys.exit(0)

uuid_order, pid_to_gpu = [], {}
for ln in out.splitlines():
    ln = ln.strip()
    if not ln:
        continue
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) < 2:
        continue
    uuid, pid_s = parts[0], parts[1]
    if uuid not in uuid_order:
        uuid_order.append(uuid)
    try:
        pid_to_gpu[int(pid_s)] = uuid_order.index(uuid)
    except ValueError:
        pass

amr = defaultdict(int)
for pid, gpu in pid_to_gpu.items():
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").replace("\x00", " ")
    except OSError:
        continue
    if "tools/train.py" not in cmd or repo not in cmd:
        continue
    try:
        ppid = int(open(f"/proc/{pid}/status").read().split("PPid:")[1].split()[0])
        ppcmd = open(f"/proc/{ppid}/cmdline", "rb").read().decode(errors="replace")
    except (OSError, IndexError, ValueError):
        ppcmd = ""
    if "tools/train.py" in ppcmd:
        continue
    if "amr_benchmark_retune" in cmd or "/configs/amr_benchmark/" in cmd:
        amr[gpu] += 1

print(amr.get(0, 0), amr.get(1, 0))
PY
}

gpu_utilization() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '{gsub(/ %/,"",$2); print $1,$2,$3}' | sort -n
}

daemon_pid() {
    # Prefer real daemon cmdline; skip this inspection script / pgrep self-match.
    local pid cmdline
    while read -r pid; do
        [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "${cmdline}" == *pgrep* || "${cmdline}" == *run_inspection* ]] && continue
        [[ "${cmdline}" == *"$1"* ]] || continue
        echo "${pid}"
        return 0
    done < <(pgrep -f "$1" 2>/dev/null || true)
    true
}

orchestrator_running() {
    pgrep -f "tools/amr_benchmark/retune_model_siege.py" >/dev/null 2>&1
}

queue_summary() {
    "${PY}" - "${QUEUE}" <<'PY'
import json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
entries = data.get("entries", [])
by_status = {}
for e in entries:
    st = e.get("status", "pending")
    by_status.setdefault(st, []).append(e)

pending = [e for e in entries if e.get("status", "pending") not in {"passed", "exhausted", "skipped", "running"}]
running = [e for e in entries if e.get("status") == "running"]
stuck = []
for e in running:
    stuck.append(e)

print(f"total={len(entries)} pending={len(pending)} running={len(running)} exhausted={len(by_status.get('exhausted', []))}")
for e in pending:
    print(f"  pending: {e.get('id')} ({e.get('model')}/{e.get('dataset')})")
for e in running:
    print(f"  running: {e.get('id')} ({e.get('model')}/{e.get('dataset')})")
PY
}

stuck_queue_entries() {
    local orch_alive=false
    orchestrator_running && orch_alive=true
    "${PY}" - "${QUEUE}" "${orch_alive}" <<'PY'
import json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
orch_alive = sys.argv[2] == "true"
for e in data.get("entries", []):
    st = e.get("status", "pending")
    eid = e.get("id", "")
    if st == "running" and not orch_alive:
        print(f"stuck_running_no_orch: {eid}")
    elif st in ("pending",) and not orch_alive:
        print(f"pending_no_orch: {eid}")
PY
}

goal_status_summary() {
    local tool="$1"
    shift
    "${PY}" "${tool}" --goal-status "$@" 2>&1 \
        | "${PY}" -c "
import sys
lines = sys.stdin.read().splitlines()
for ln in lines[:12]:
    print(ln)
if len(lines) > 12:
    print(f'  … ({len(lines) - 12} more lines)')
"
}

# --- Build report ---

TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
append "=== AMR/JDM Health Inspection @ ${TS} ==="
append ""

# GPU + trains
append "--- GPU ---"
while read -r idx util mem; do
    append "  GPU${idx}: util=${util}% mem=${mem}MiB"
done < <(gpu_utilization)

counts="$(count_trains_per_gpu)"
read -r amr0 amr1 <<< "${counts}"
append "  AMR train.py parent jobs: gpu0=${amr0} gpu1=${amr1}"
append ""

# Daemons
KEEPALIVE_PID="$(daemon_pid 'tools/amr_benchmark/gpu_keepalive.sh')"
WATCHDOG_PID="$(daemon_pid 'tools/amr_benchmark/health_watchdog.sh')"
SIEGE_PID="$(daemon_pid 'tools/amr_benchmark/retune_model_siege.py')"

append "--- Daemons ---"
if [[ -n "${KEEPALIVE_PID}" ]]; then
    append "  gpu_keepalive.sh: alive PID=${KEEPALIVE_PID}"
else
    append "  gpu_keepalive.sh: DEAD"
    CRITICAL=1
fi
if [[ -n "${WATCHDOG_PID}" ]]; then
    append "  health_watchdog.sh: alive PID=${WATCHDOG_PID}"
else
    append "  health_watchdog.sh: DEAD"
    CRITICAL=1
fi
if [[ -n "${SIEGE_PID}" ]]; then
    append "  retune_model_siege.py: alive PID=${SIEGE_PID}"
else
    append "  retune_model_siege.py: not running"
fi
append ""

# Queue
append "--- Siege queue (siege_queue.json) ---"
while IFS= read -r line; do
    append "  ${line}"
done < <(queue_summary)
stuck_lines="$(stuck_queue_entries)"
if [[ -n "${stuck_lines}" ]]; then
    append "  STUCK:"
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        append "    ${line}"
    done <<< "${stuck_lines}"
fi
append ""

# HEALTH_STATUS.json
append "--- HEALTH_STATUS.json ---"
if [[ -f "${HEALTH_STATUS}" ]]; then
    while IFS= read -r line; do
        append "  ${line}"
    done < <("${PY}" - "${HEALTH_STATUS}" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
print(f"last_check: {d.get('last_check')}")
print(f"gpu_util: {d.get('gpu_util')}")
print(f"amr_jobs: {d.get('amr_jobs')}")
issues = d.get("issues") or []
actions = d.get("auto_actions_taken") or []
print(f"issues ({len(issues)}): {issues if issues else 'none'}")
print(f"auto_actions ({len(actions)}): {actions if actions else 'none'}")
PY
)
else
    append "  (missing)"
fi
append ""

# health.log tail
append "--- health.log (last 5 lines) ---"
if [[ -f "${HEALTH_LOG}" ]]; then
    while IFS= read -r line; do
        append "  ${line}"
    done < <(tail -5 "${HEALTH_LOG}")
else
    append "  (missing)"
fi
append ""

# Goal status
append "--- AMR goal-status ---"
while IFS= read -r line; do
    append "  ${line}"
done < <(goal_status_summary tools/amr_benchmark/retune_sweep.py --manifest "${AMR_MANIFEST}")
append ""

append "--- JDM goal-status ---"
while IFS= read -r line; do
    append "  ${line}"
done < <(goal_status_summary tools/jdm/retune_sweep.py)
append ""

# Critical: both GPUs idle with pending queue
gpu_utils="$(gpu_utilization | awk '{print $2}')"
read -r gpu0_util gpu1_util <<< "$(echo "${gpu_utils}" | tr '\n' ' ')"
gpu0_util="${gpu0_util:-0}"
gpu1_util="${gpu1_util:-0}"
pending_n="$("${PY}" - "${QUEUE}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
skip = {"passed", "exhausted", "skipped", "running"}
print(sum(1 for e in data.get("entries", []) if e.get("status", "pending") not in skip))
PY
)"

both_idle=false
if [[ "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${gpu0_util}" -le 1 && "${gpu1_util}" -le 1 ]]; then
    both_idle=true
fi

# --- Remote H100 box (two-machine goal mode, read-only SSH) ---
# Enable by exporting REMOTE_SSH (passwordless). Best-effort: never fails the
# local report. Surfaces remote idle GPUs so neither box idles unnoticed.
REMOTE_SSH="${REMOTE_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=5 citybuster@10.161.4.55}"
REMOTE_REPO="${REMOTE_REPO:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
REMOTE_ENABLE="${REMOTE_ENABLE:-1}"
append "--- Remote H100 (10.161.4.55) ---"
if [[ "${REMOTE_ENABLE}" == "1" ]] && command -v ssh >/dev/null 2>&1; then
    remote_out="$(${REMOTE_SSH} "
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | sed 's/^/GPU /';
        echo '---POOL---';
        cat ${REMOTE_REPO}/work_dirs/amr_benchmark_retune/POOL_STATUS.json 2>/dev/null || echo 'no POOL_STATUS.json';
    " 2>/dev/null)"
    if [[ -n "${remote_out}" ]]; then
        remote_idle=0
        while IFS= read -r line; do
            [[ -z "${line}" ]] && continue
            append "  ${line}"
            if [[ "${line}" == GPU* ]]; then
                util="$(echo "${line}" | awk -F', ' '{gsub(/ %/,"",$2); print $2}' | tr -d ' ')"
                [[ -n "${util}" && "${util}" -le 1 ]] 2>/dev/null && remote_idle=$((remote_idle + 1))
            fi
        done <<< "${remote_out}"
        if [[ "${remote_idle}" -ge 1 ]]; then
            append "  WARN: ${remote_idle} remote GPU(s) at util≤1% — remote pool daemon should auto-fill; check pool.log"
        fi
    else
        append "  (remote unreachable via passwordless SSH — set REMOTE_SSH / ssh-copy-id to enable unified status)"
    fi
else
    append "  (remote status disabled: REMOTE_ENABLE=0 or ssh missing)"
fi
append ""

append "--- Verdict (local box) ---"
if [[ "${both_idle}" == "true" && "${pending_n}" -gt 0 ]]; then
    append "  CRITICAL: both local GPUs idle (util≤1%, no AMR trains) with ${pending_n} pending queue entr(y/ies)"
    CRITICAL=1
elif [[ ${CRITICAL} -eq 1 ]]; then
    append "  CRITICAL: daemon or GPU policy violation detected"
else
    append "  OK: local GPUs active or no pending queue work"
fi

# Emit report
printf '%s' "${REPORT}"
{
    echo "[${TS}] INSPECTION REPORT"
    printf '%s' "${REPORT}"
    echo "[${TS}] INSPECTION complete exit=${CRITICAL} keepalive=${KEEPALIVE_PID:-none} watchdog=${WATCHDOG_PID:-none}"
} >> "${HEALTH_LOG}"

exit "${CRITICAL}"
