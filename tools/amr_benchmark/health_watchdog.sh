#!/usr/bin/env bash
# Proactive health monitor for AMR siege + GPU keepalive.
# Detects idle GPUs, dead daemons, stale queue state, and crashed sieges;
# auto-remediates where safe and writes HEALTH_STATUS.json every tick.
#
# Usage:
#   nohup bash tools/amr_benchmark/health_watchdog.sh >> work_dirs/amr_benchmark_retune/health.log 2>&1 &

set -uo pipefail

REPO="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
PY="${PYTHON:-/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python}"
LOGDIR="${REPO}/work_dirs/amr_benchmark_retune"
HEALTH_LOG="${LOGDIR}/health.log"
HEALTH_STATUS="${LOGDIR}/HEALTH_STATUS.json"
QUEUE="${REPO}/configs/amr_benchmark/retune/siege_queue.json"
QUEUE_FULL="${REPO}/configs/amr_benchmark/retune/siege_queue_full.json"
WAVE1_MANIFEST="${REPO}/configs/amr_benchmark/retune/wave1_manifest.json"
WAVE1_LOG="${LOGDIR}/wave1_active.log"
WAVE4_MANIFEST="${REPO}/configs/amr_benchmark/retune/wave4_marginal_manifest.json"
WAVE4_LOG="${LOGDIR}/wave4_marginal.log"
GOAL_STATUS="${LOGDIR}/GOAL_STATUS.json"
KEEPALIVE_SCRIPT="${REPO}/tools/amr_benchmark/gpu_keepalive.sh"
SIEGE_R3_MANIFEST="${REPO}/configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json"
SIEGE_R2_MANIFEST="${REPO}/configs/amr_benchmark/retune/siege_fastmldnn_10a_r2.json"

INTERVAL="${HEALTH_WATCHDOG_INTERVAL:-120}"
# Both-GPU idle escalation (was 15 min). Single-GPU IDLE_GPU path uses 300s below.
GPU_IDLE_THRESHOLD="${GPU_IDLE_THRESHOLD_SEC:-600}"
# GPU1 alone idle with full_pending → auto-dispatch (do not wait for user ask).
GPU1_IDLE_FILL_THRESHOLD="${GPU1_IDLE_FILL_THRESHOLD_SEC:-300}"
SIEGE_STALE_THRESHOLD="${SIEGE_STALE_THRESHOLD_SEC:-7200}"

cd "${REPO}" || exit 1
mkdir -p "${LOGDIR}"

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" >> "${HEALTH_LOG}"
}

log_issue() {
    log "ISSUE: $*"
}

log_action() {
    log "AUTO_ACTION: $*"
}

# --- Shared probes (mirror gpu_keepalive.sh) ---

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
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '{gsub(/ %/,"",$2); print $1,$2}' | sort -n | awk '{print $2}' | tr '\n' ' '
}

# True only for a live python siege/sweep (exclude bash/pgrep self-matches).
_python_orch_running() {
    local pattern="$1"
    local pid cmdline
    while read -r pid; do
        [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "${cmdline}" == *pgrep* || "${cmdline}" == *bash* ]] && continue
        case "${cmdline}" in
            *python*"${pattern}"*|*python3*"${pattern}"*) return 0 ;;
        esac
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
    return 1
}

orchestrator_running() {
    _python_orch_running "tools/amr_benchmark/retune_model_siege.py"
}

amr_sweep_running() {
    _python_orch_running "tools/amr_benchmark/retune_sweep.py"
}

count_pending_entries_full() {
    "${PY}" - "${QUEUE_FULL}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
print(sum(1 for e in data.get("entries", []) if e.get("status", "pending") not in skip))
PY
}

next_pending_manifest_entry_full() {
    # Skip only queue-terminal statuses. Phantom pending closed by reconcile.
    "${PY}" - "${QUEUE_FULL}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
for entry in sorted(data.get("entries", []), key=lambda e: e.get("priority", 99)):
    if entry.get("status", "pending") in skip:
        continue
    model = entry.get("model", "") or ""
    dataset = entry.get("dataset", "") or ""
    manifest = entry.get("manifest")
    if not manifest:
        if model and dataset:
            print("\t".join([entry.get("id", ""), f"__synthesize__:{model}:{dataset}", model, dataset]))
            raise SystemExit(0)
        continue
    print("\t".join([entry.get("id", ""), manifest, model, dataset]))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

wave1_goal_exhausted() {
    "${PY}" - "${WAVE1_MANIFEST}" "${GOAL_STATUS}" <<'PY'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
goal_path = Path(sys.argv[2])
if not manifest_path.is_file() or not goal_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
pairs = {(e["model"].lower(), e["dataset"]) for e in manifest.get("experiments", [])}
goal = json.loads(goal_path.read_text())
exhausted = set()
for label in goal.get("exhausted_pairs", []):
    model, dataset = label.split("/", 1)
    exhausted.add((model.lower(), dataset))
print("yes" if pairs <= exhausted else "no")
PY
}

launch_wave1_goal_sweep() {
    nohup "${PY}" tools/amr_benchmark/retune_sweep.py \
        --manifest "${WAVE1_MANIFEST}" \
        --gpu 0,1 --max-parallel 2 \
        --goal-mode --stop-when-all-pass --paper-exact \
        >> "${WAVE1_LOG}" 2>&1 &
    log_action "launched_wave1_goal_sweep PID=$! log=${WAVE1_LOG}"
}

wave4_marginal_exhausted() {
    "${PY}" - "${WAVE4_MANIFEST}" "${LOGDIR}" <<'PY'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
retune_root = Path(sys.argv[2])
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
experiments = manifest.get("experiments", [])
if not experiments:
    raise SystemExit(1)
for exp in experiments:
    wd = retune_root / exp["model"] / exp["dataset"] / exp["variant"]
    if not (wd / "last_checkpoint").is_file():
        print("no")
        raise SystemExit(0)
print("yes")
PY
}

launch_wave4_marginal_siege() {
    local force_flag="$1"
    local extra=()
    if [[ "${force_flag}" == "force" ]]; then
        local blocked=0
        for pair in "icamcnet:hisar2019" "hcgdnn:deepsig201610A" "lstm2:deepsig201610A" "resnetamr:deepsig201610B"; do
            IFS=':' read -r m d <<< "${pair}"
            if [[ "$(pair_force_blocked "${m}" "${d}" 2>/dev/null || echo no)" == "yes" ]]; then
                blocked=$((blocked + 1))
            fi
        done
        if [[ "${blocked}" -ge 3 ]]; then
            log_action "skip_wave4_force blocked=${blocked}/4"
            force_flag=""
        else
            extra+=(--force)
        fi
    fi
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest "${WAVE4_MANIFEST}" \
        --gpu 0,1 --max-parallel 2 \
        --until-pass --paper-exact --promote \
        "${extra[@]}" \
        >> "${WAVE4_LOG}" 2>&1 &
    log_action "launched_wave4_marginal PID=$! log=${WAVE4_LOG} force=${force_flag:-no}"
}

launch_full_queue_marginal_siege() {
    local force_flag="$1"
    local gpu_arg="${2:-0}"
    local entry_id manifest model dataset pending_line
    pending_line="$(next_pending_manifest_entry_full 2>/dev/null || true)"
    if [[ -z "${pending_line}" ]]; then
        return 1
    fi
    IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
    if [[ "${manifest}" == __synthesize__:* ]]; then
        # Defer synthesize to keepalive ensure_marginal — watchdog only launches known manifests.
        log_action "skip_full_queue_synthesize entry=${entry_id} (keepalive owns synthesize)"
        return 1
    fi
    local extra=()
    if [[ "${force_flag}" == "force" ]]; then
        if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
            log_action "skip_force entry=${entry_id} reason=force_blocked"
            force_flag=""
        else
            extra+=(--force)
        fi
    fi
    # Prefer exclusive free GPU; never steal GPU0 work when caller asks for GPU1.
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest "${REPO}/${manifest}" \
        --gpu "${gpu_arg}" --max-parallel 1 \
        --until-pass --paper-exact --promote \
        "${extra[@]}" \
        >> "${LOGDIR}/siege_${model}.log" 2>&1 &
    log_action "launched_full_queue_marginal entry=${entry_id} manifest=${manifest} gpu=${gpu_arg} PID=$! force=${force_flag:-no}"
    return 0
}

launch_post_siege_work() {
    local force_flag="$1"
    local reconciled
    reconciled="$(reconcile_phantom_pending_full 2>/dev/null || true)"
    if [[ -n "${reconciled}" ]]; then
        log_action "reconcile_phantom_pending ${reconciled}"
    fi
    # Prefer full-queue marginals before wave1 (same order as keepalive).
    if launch_full_queue_marginal_siege "${force_flag}"; then
        echo "full_queue"
        return 0
    fi
    if [[ "$(wave1_goal_exhausted 2>/dev/null || echo no)" != "yes" ]]; then
        launch_wave1_goal_sweep
        echo "wave1"
        return 0
    fi
    if [[ "$(wave4_marginal_exhausted 2>/dev/null || echo no)" != "yes" && -f "${WAVE4_MANIFEST}" ]]; then
        launch_wave4_marginal_siege "${force_flag}"
        echo "wave4"
        return 0
    fi
    return 1
}

keepalive_running() {
    pgrep -f "tools/amr_benchmark/gpu_keepalive.sh" >/dev/null 2>&1
}

next_pending_entry() {
    "${PY}" - "${QUEUE}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
for entry in sorted(data.get("entries", []), key=lambda e: e.get("priority", 99)):
    if entry.get("status", "pending") in skip:
        continue
    print(
        "\t".join(
            [
                entry.get("id", ""),
                entry.get("manifest", ""),
                entry.get("model", ""),
                entry.get("dataset", ""),
            ]
        )
    )
    raise SystemExit(0)
raise SystemExit(1)
PY
}

count_pending_entries() {
    "${PY}" - "${QUEUE}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
n = sum(1 for e in data.get("entries", []) if e.get("status", "pending") not in skip)
print(n)
PY
}

# Mirror gpu_keepalive: --force refused for queue-terminal OR ≥3 identical fails
# (unless pair already has a pass). Selectors do NOT use identical-fails alone.
pair_force_blocked() {
    local model="$1" dataset="$2"
    "${PY}" - "${QUEUE}" "${QUEUE_FULL}" "${REPO}/docs/amr_benchmark/retune_results.md" "${model}" "${dataset}" <<'PY'
import json, re, sys
from collections import defaultdict
from pathlib import Path

queue_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
results_md = Path(sys.argv[3])
model = sys.argv[4].lower()
dataset = sys.argv[5]
label = f"{model}/{dataset}"

for qp in queue_paths:
    if not qp.is_file():
        continue
    data = json.loads(qp.read_text())
    for e in data.get("entries", []):
        if (e.get("model") or "").lower() != model or e.get("dataset") != dataset:
            continue
        status = e.get("status", "pending")
        # Open pending/running entries (newer waves) keep the pair launchable.
        if status in {"pending", "running"}:
            print("no")
            raise SystemExit(0)
        if status.startswith("waived") or e.get("waiver"):
            print("yes")
            raise SystemExit(0)
# If we saw only passed/exhausted/skipped (no open entry), fall through to
# identical-fail / pass checks below. Do not treat a single exhausted entry
# as blocking when a newer pending wave exists (handled above).

text = results_md.read_text(errors="replace") if results_md.is_file() else ""
if re.search(
    r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
    + re.escape(label)
    + r"\s*\|\s*`[^`]+`\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*`pass`\s*\|",
    text,
    re.I,
):
    print("no")
    raise SystemExit(0)

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
        clusters[(variant, int(round(float(overall_s) * 4)), int(round(float(peak_s) * 4)))] += 1
    except ValueError:
        continue
print("yes" if any(n >= 3 for n in clusters.values()) else "no")
PY
}

reconcile_phantom_pending_full() {
    "${PY}" "${REPO}/tools/amr_benchmark/reconcile_phantom_pending.py" \
        --queue-full "${QUEUE_FULL}" \
        --results "${REPO}/docs/amr_benchmark/retune_results.md" \
        --repo "${REPO}"
}

reset_false_exhausted_queue() {
    local changed orch_alive
    orch_alive=false
    orchestrator_running && orch_alive=true
    changed="$("${PY}" - "${QUEUE}" "${LOGDIR}" "${orch_alive}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

queue_path = Path(sys.argv[1])
logdir = Path(sys.argv[2])
orch_alive = sys.argv[3] == "true"
data = json.loads(queue_path.read_text())
changed = []

def has_siege_evidence(model: str, entry: dict) -> bool:
    if entry.get("best_variant"):
        return True
    primary = logdir / f"siege_{model}.log"
    if primary.exists() and primary.stat().st_size > 200:
        return True
    if model == "fastmldnn":
        for alt in ("siege.log", "siege_r2.log", "siege_r3.log"):
            p = logdir / alt
            if p.exists() and p.stat().st_size > 200:
                return True
    return False

for entry in data.get("entries", []):
    model = entry.get("model", "")
    status = entry.get("status", "pending")
    if status == "exhausted" and not has_siege_evidence(model, entry):
        entry["status"] = "pending"
        entry.pop("notes", None)
        changed.append(f"{entry.get('id')}:{model}->pending(no_evidence)")
    elif status == "running" and not orch_alive:
        entry["status"] = "pending"
        changed.append(f"{entry.get('id')}:{model}->pending(stale_running)")
if changed:
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    queue_path.write_text(json.dumps(data, indent=2) + "\n")
print(";".join(changed) if changed else "")
PY
)"
    if [[ -n "${changed}" ]]; then
        log_action "reset_queue_entries ${changed}"
    fi
}

launch_siege_queue() {
    local force_flag="$1"
    local entry_id manifest model dataset pending_line
    pending_line="$(next_pending_entry 2>/dev/null || true)"
    if [[ -z "${pending_line}" ]]; then
        return 1
    fi
    IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
    local extra=()
    [[ "${force_flag}" == "force" ]] && extra+=(--force)
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --queue "${QUEUE}" \
        --gpu 0,1 --max-parallel 2 \
        --until-pass --paper-exact --promote \
        "${extra[@]}" \
        >> "${LOGDIR}/siege_${model}.log" 2>&1 &
    log_action "launched_siege_queue entry=${entry_id} manifest=${manifest} PID=$! force=${force_flag}"
    return 0
}

launch_siege_manifest() {
    local manifest="$1"
    local model="$2"
    local log_name="${3:-siege_${model}}"
    local dataset="${4:-}"
    if [[ -z "${dataset}" ]]; then
        case "${model}" in
            icamcnet) dataset="hisar2019" ;;
            *) dataset="deepsig201610A" ;;
        esac
    fi
    if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
        log_action "skip_stale_relaunch model=${model} dataset=${dataset} reason=exhausted_or_identical_paper_exact_fails_ge3"
        return 1
    fi
    # Prefer advancing siege_queue_full pending work over --force looping a dead pair.
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest "${manifest}" \
        --gpu 0 --max-parallel 1 \
        --until-pass --paper-exact --promote \
        >> "${LOGDIR}/${log_name}.log" 2>&1 &
    log_action "relaunched_siege manifest=${manifest} PID=$! (no --force; GPU0-only)"
}

restart_keepalive() {
    # Kill prior keepalive instances first — restart-without-kill spawned ~870
    # zombies and caused pgrep false-positives (stall 2026-07-20 ~14.7h idle).
    local oldpid
    while read -r oldpid; do
        [[ -z "${oldpid}" ]] && continue
        # Only the real script process (not this watchdog / inspector shells).
        if tr '\0' ' ' <"/proc/${oldpid}/cmdline" 2>/dev/null | grep -q 'tools/amr_benchmark/gpu_keepalive.sh'; then
            kill "${oldpid}" 2>/dev/null || true
        fi
    done < <(pgrep -f 'tools/amr_benchmark/gpu_keepalive.sh' 2>/dev/null || true)
    sleep 1
    # Force leftover
    while read -r oldpid; do
        [[ -z "${oldpid}" ]] && continue
        if tr '\0' ' ' <"/proc/${oldpid}/cmdline" 2>/dev/null | grep -q 'tools/amr_benchmark/gpu_keepalive.sh'; then
            kill -9 "${oldpid}" 2>/dev/null || true
        fi
    done < <(pgrep -f 'tools/amr_benchmark/gpu_keepalive.sh' 2>/dev/null || true)
    nohup bash "${KEEPALIVE_SCRIPT}" >> "${LOGDIR}/scheduler.log" 2>&1 &
    log_action "restarted_gpu_keepalive PID=$! (replaced prior instances)"
}

read_idle_since() {
    local state_file="${LOGDIR}/health_idle_state.json"
    if [[ -f "${state_file}" ]]; then
        cat "${state_file}"
    else
        echo '{"gpu_idle_since": null}'
    fi
}

write_idle_since() {
    echo "$1" > "${LOGDIR}/health_idle_state.json"
}

stale_siege_logs() {
    "${PY}" - "${LOGDIR}" "${SIEGE_STALE_THRESHOLD}" <<'PY'
import json, sys, time
from pathlib import Path

logdir = Path(sys.argv[1])
threshold = int(sys.argv[2])
now = time.time()
results = []
for log_file in sorted(logdir.glob("siege_*.log")):
    mtime = log_file.stat().st_mtime
    age = now - mtime
    if age < threshold:
        continue
    model = log_file.stem.replace("siege_", "")
    results.append(f"{model}\t{log_file}\t{int(age)}")
print("\n".join(results))
PY
}

disconnect_recovery_check() {
    "${PY}" - "${REPO}" "${LOGDIR}" <<'PY'
import json, sys
from pathlib import Path

repo = Path(sys.argv[1])
logdir = Path(sys.argv[2])
checks = [
    ("siege_r2", repo / "configs/amr_benchmark/retune/siege_fastmldnn_10a_r2.json", logdir / "siege_r2.log"),
    ("siege_r3", repo / "configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json", logdir / "siege_r3.log"),
]
for label, manifest, log_file in checks:
    if not manifest.exists():
        continue
    data = json.loads(manifest.read_text())
    exps = data.get("experiments", [])
    if not exps:
        continue
    model = exps[0].get("model", "fastmldnn")
    dataset = exps[0].get("dataset", "")
    root = repo / "work_dirs/amr_benchmark_retune" / model / dataset
    incomplete = False
    for exp in exps:
        variant = exp.get("variant", "")
        paper = root / variant / "res" / "paper.pkl"
        if not paper.exists():
            incomplete = True
            break
    if incomplete and not log_file.exists():
        print(f"ACTION_NEEDED {label} manifest={manifest} reason=manifest_exists_no_log")
    elif incomplete and log_file.exists() and log_file.stat().st_size < 200:
        print(f"ACTION_NEEDED {label} manifest={manifest} reason=log_trivially_small")
PY
}

write_health_status() {
    local last_check="$1"
    local gpu0_util="$2"
    local gpu1_util="$3"
    local amr0="$4"
    local amr1="$5"
    local issues_json="$6"
    local actions_json="$7"
    "${PY}" - "${HEALTH_STATUS}" "${last_check}" "${gpu0_util}" "${gpu1_util}" "${amr0}" "${amr1}" "${issues_json}" "${actions_json}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "last_check": sys.argv[2],
    "gpu_util": [int(sys.argv[3]), int(sys.argv[4])],
    "amr_jobs": {"gpu0": int(sys.argv[5]), "gpu1": int(sys.argv[6])},
    "issues": json.loads(sys.argv[7]),
    "auto_actions_taken": json.loads(sys.argv[8]),
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

tick() {
    local now counts amr0 amr1 gpu_utils gpu0_util gpu1_util
    local issues=() actions=()
    now="$(date -u +%s)"
    counts="$(count_trains_per_gpu)"
    read -r amr0 amr1 <<< "${counts}"
    gpu_utils="$(gpu_utilization)"
    read -r gpu0_util gpu1_util <<< "${gpu_utils}"
    gpu0_util="${gpu0_util:-0}"
    gpu1_util="${gpu1_util:-0}"

    local siege_orch amr_sweep keepalive pending_n pending_full_n
    siege_orch=false
    amr_sweep=false
    keepalive_running && keepalive=true || keepalive=false
    orchestrator_running && siege_orch=true || siege_orch=false
    amr_sweep_running && amr_sweep=true || amr_sweep=false
    pending_n="$(count_pending_entries)"
    pending_full_n="$(count_pending_entries_full)"

    log "check gpu_util=[${gpu0_util},${gpu1_util}] amr_jobs=[${amr0},${amr1}] siege_orch=${siege_orch} amr_sweep=${amr_sweep} keepalive=${keepalive} pending=${pending_n} full_pending=${pending_full_n}"

    # Reset falsely exhausted queue entries (no siege log).
    reset_false_exhausted_queue
    local reconciled
    reconciled="$(reconcile_phantom_pending_full 2>/dev/null || true)"
    if [[ -n "${reconciled}" ]]; then
        log_action "reconcile_phantom_pending ${reconciled}"
    fi
    pending_n="$(count_pending_entries)"
    pending_full_n="$(count_pending_entries_full)"

    # Track GPU idle duration (both util ~0 and no AMR trains).
    local idle_state gpu_idle_since idle_sec
    idle_state="$(read_idle_since)"
    if [[ "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${gpu0_util}" -le 1 && "${gpu1_util}" -le 1 ]]; then
        gpu_idle_since="$(echo "${idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('gpu_idle_since') or '')")"
        if [[ -z "${gpu_idle_since}" ]]; then
            gpu_idle_since="${now}"
            write_idle_since "$(echo "${idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu_idle_since']=${now}; print(json.dumps(d))")"
            log "GPU idle timer started"
        fi
        idle_sec=$((now - gpu_idle_since))
    else
        # Clear both-idle timer only; preserve gpu1_alone_idle_since for single-GPU fill.
        write_idle_since "$(echo "${idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu_idle_since']=None; print(json.dumps(d))")"
        idle_sec=0
    fi

    # 1. Keepalive dead → restart
    if [[ "${keepalive}" == "false" ]]; then
        issues+=("keepalive_dead")
        restart_keepalive
        actions+=("restarted_keepalive")
    fi

    # 2. GPU idle >15min with pending queue and no orchestrator
    if [[ "${idle_sec}" -ge "${GPU_IDLE_THRESHOLD}" && "${pending_n}" -gt 0 && "${siege_orch}" == "false" ]]; then
        issues+=("gpu_idle_${idle_sec}s_pending_queue_${pending_n}")
        if launch_siege_queue "normal"; then
            actions+=("launched_pending_siege_idle_gpus")
        fi
    fi

    # 2b. Deadlock: primary queue pending (Tier-A) while GPU0 busy / GPU1 idle.
    # Matches gpu_keepalive fix — do not wait for both GPUs to drain.
    if [[ "${pending_n}" -gt 0 && "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${gpu1_util}" -le 1 ]]; then
        local entry_id manifest model dataset pending_line
        pending_line="$(next_pending_entry 2>/dev/null || true)"
        if [[ -n "${pending_line}" ]]; then
            IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
            if [[ -n "${entry_id}" && -n "${manifest}" ]] \
                && ! pgrep -af 'tools/amr_benchmark/retune_model_siege.py' 2>/dev/null \
                    | grep -E -- '--gpu[= ]1([[:space:]]|$)' >/dev/null; then
                issues+=("IDLE_GPU1_primary_pending_${pending_n}")
                nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                    --queue "${QUEUE}" \
                    --gpu 1 --max-parallel 1 \
                    --until-pass --paper-exact --promote \
                    >> "${LOGDIR}/siege_${model}.log" 2>&1 &
                log_action "launched_primary_pending_on_gpu1 entry=${entry_id} PID=$! (break Tier-B deadlock)"
                actions+=("launched_primary_pending_gpu1")
            fi
        fi
    fi

    # 2c. IDLE_GPU1: primary pending=0 but siege_queue_full still has work, GPU0 busy,
    # GPU1 has no AMR train. Do NOT wait for both GPUs idle — this is the stall class
    # that left GPU1 empty for hours behind force-blocked cgdnet.
    if [[ "${pending_n}" -eq 0 && "${pending_full_n}" -gt 0 && "${amr0}" -ge 1 && "${amr1}" -eq 0 ]]; then
        local gpu1_idle_state gpu1_alone_since gpu1_alone_sec
        gpu1_idle_state="$(read_idle_since)"
        gpu1_alone_since="$(echo "${gpu1_idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('gpu1_alone_idle_since') or '')" 2>/dev/null || true)"
        if [[ -z "${gpu1_alone_since}" ]]; then
            write_idle_since "$(echo "${gpu1_idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_alone_idle_since']=${now}; print(json.dumps(d))")"
            gpu1_alone_since="${now}"
            log "GPU1 alone-idle timer started (full_pending=${pending_full_n})"
        fi
        gpu1_alone_sec=$((now - gpu1_alone_since))
        issues+=("IDLE_GPU1_full_pending_${pending_full_n}_${gpu1_alone_sec}s")
        if [[ "${gpu1_alone_sec}" -ge "${GPU1_IDLE_FILL_THRESHOLD}" ]] \
            && ! pgrep -af 'tools/amr_benchmark/retune_model_siege.py' 2>/dev/null \
                | grep -E -- '--gpu[= ]1([[:space:]]|$)' >/dev/null; then
            log_issue "IDLE_GPU1 ${gpu1_alone_sec}s with full_pending=${pending_full_n} — auto-dispatch"
            if launch_full_queue_marginal_siege "normal" "1"; then
                actions+=("auto_filled_IDLE_GPU1_full_queue")
                write_idle_since "$(read_idle_since | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_alone_idle_since']=None; print(json.dumps(d))")"
            else
                # Nudge keepalive: restart so patched skip-force-blocked logic is live.
                log_action "IDLE_GPU1 fill failed (no launchable target) — restarting keepalive for skip-ahead"
                restart_keepalive
                actions+=("restarted_keepalive_for_IDLE_GPU1")
            fi
        fi
    elif [[ "${pending_n}" -eq 0 && "${pending_full_n}" -eq 0 && "${amr1}" -eq 0 ]]; then
        # Stall class amr_queue_empty_no_jdm_fallback (GPU1 idle ~94min 2026-07-18)
        # and jdm_fallback_false_exhausted_both_idle (BOTH idle ~14.6h 2026-07-19):
        # full queue empty + GPU1 empty (GPU0 busy OR also idle) — keepalive must
        # fall through to JDM ideal. Prior bug required amr0>=1 so BOTH-idle never
        # raised this issue and only hit silent siege_exhausted_sweep.
        local gpu1_idle_state gpu1_alone_since gpu1_alone_sec jdm_live=0
        gpu1_idle_state="$(read_idle_since)"
        gpu1_alone_since="$(echo "${gpu1_idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('gpu1_alone_idle_since') or '')" 2>/dev/null || true)"
        if [[ -z "${gpu1_alone_since}" ]]; then
            write_idle_since "$(echo "${gpu1_idle_state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_alone_idle_since']=${now}; print(json.dumps(d))")"
            gpu1_alone_since="${now}"
            log "GPU1 alone-idle timer started (full_pending=0 — expect JDM ideal fallback; amr0=${amr0})"
        fi
        gpu1_alone_sec=$((now - gpu1_alone_since))
        # jdm_on_gpu1 only — JDM on GPU0 must NOT clear the GPU1 fill timer.
        if pgrep -af 'tools/train.py' 2>/dev/null | grep -E 'configs/jdm|work_dirs/jdm|det_ideal_v1' >/dev/null \
            && [[ "${gpu1_util}" -ge 5 ]]; then
            jdm_live=1
        fi
        if pgrep -af 'tools/test_det.py' 2>/dev/null | grep -E 'eval_ideal_v1|eval_awgn|jdm' >/dev/null \
            && [[ "${gpu1_util}" -ge 5 ]]; then
            jdm_live=1
        fi
        if [[ "${gpu1_alone_sec}" -ge "${GPU1_IDLE_FILL_THRESHOLD}" && "${jdm_live}" -eq 0 ]]; then
            if [[ "${amr0}" -eq 0 && "${gpu0_util}" -le 1 ]]; then
                issues+=("jdm_fallback_false_exhausted_both_idle_${gpu1_alone_sec}s")
                log_issue "jdm_fallback_false_exhausted_both_idle both GPUs idle ${gpu1_alone_sec}s with full_pending=0 — restart keepalive for JDM ideal/AWGN"
                actions+=("restarted_keepalive_jdm_fallback_false_exhausted_both_idle")
            else
                issues+=("amr_queue_empty_no_jdm_fallback_${gpu1_alone_sec}s")
                log_issue "amr_queue_empty_no_jdm_fallback GPU1 idle ${gpu1_alone_sec}s with full_pending=0 — restart keepalive for JDM ideal"
                actions+=("restarted_keepalive_amr_queue_empty_no_jdm_fallback")
            fi
            restart_keepalive
        elif [[ "${jdm_live}" -eq 1 ]]; then
            write_idle_since "$(read_idle_since | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d.pop('gpu1_alone_idle_since', None); print(json.dumps(d))" 2>/dev/null || echo '{"gpu_idle_since": null}')"
        fi
    else
        # Clear alone-idle timer when GPU1 has work or both idle path owns state.
        if [[ "${amr1}" -ge 1 ]]; then
            write_idle_since "$(read_idle_since | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d.pop('gpu1_alone_idle_since', None); print(json.dumps(d))" 2>/dev/null || echo '{"gpu_idle_since": null}')"
        fi
    fi

    # 3. Queue stuck: pending but orch dead and GPUs idle >threshold → force
    if [[ "${pending_n}" -gt 0 && "${siege_orch}" == "false" && "${idle_sec}" -ge "${GPU_IDLE_THRESHOLD}" ]]; then
        issues+=("queue_stuck_pending_${pending_n}")
        if launch_siege_queue "force"; then
            actions+=("force_launched_stuck_queue")
        fi
    fi

    # 3b. Siege exhausted, sweep not running, GPUs idle → wave1 or full-queue marginals
    if [[ "${pending_n}" -eq 0 && "${siege_orch}" == "false" && "${amr_sweep}" == "false"
        && "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${gpu0_util}" -le 1 && "${gpu1_util}" -le 1
        && "${idle_sec}" -ge 300 ]]; then
        issues+=("siege_exhausted_sweep_idle_gpus_${idle_sec}s")
        local post_force="normal" launched_post=""
        if [[ "${idle_sec}" -ge "${GPU_IDLE_THRESHOLD}" ]]; then
            post_force="force"
        fi
        launched_post="$(launch_post_siege_work "${post_force}" || true)"
        case "${launched_post}" in
            wave1) actions+=("launched_wave1_post_siege") ;;
            wave4) actions+=("launched_wave4_marginal_post_siege") ;;
            full_queue) actions+=("launched_full_queue_marginal_post_siege") ;;
            *)
                # post-siege selectors empty — keepalive JDM fallback must fill.
                # Stall classes: jdm_fallback_false_exhausted_both_idle (2026-07-19);
                # all_waves_exhausted_no_next_seed (2026-07-20 ~14.7h; again 2026-07-22 ~36h).
                if [[ "${pending_full_n}" -eq 0 && "${idle_sec}" -ge "${GPU_IDLE_THRESHOLD}" ]]; then
                    if grep -q 'all_waves_exhausted_no_next_seed' "${LOGDIR}/health.log" 2>/dev/null \
                        && [[ "$(tail -n 50 "${LOGDIR}/scheduler.log" 2>/dev/null | grep -c 'all_waves_exhausted_no_next_seed' || true)" -ge 1 ]]; then
                        issues+=("all_waves_exhausted_no_next_seed_${idle_sec}s")
                        log_issue "all_waves_exhausted_no_next_seed ${idle_sec}s — invoking auto_seed generate+queue (not restart-spam)"
                        # Rate-limit KA restarts: prior path restarted every tick (~2min)
                        # for 36h without creating wave manifests. Seed via keepalive
                        # helper once, then restart at most every 30 min.
                        if GPU_KEEPALIVE_AUTO_SEED_ONCE=1 bash "${KEEPALIVE_SCRIPT}" >> "${LOGDIR}/health.log" 2>&1; then
                            actions+=("auto_seeded_next_wave_from_watchdog")
                        fi
                        local last_rst rst_age=999999
                        last_rst="$(stat -c '%Y' "${LOGDIR}/.last_ka_restart_exhausted" 2>/dev/null || echo 0)"
                        rst_age=$(( $(date +%s) - last_rst ))
                        if [[ "${rst_age}" -ge 1800 ]]; then
                            date +%s > "${LOGDIR}/.last_ka_restart_exhausted"
                            restart_keepalive
                            actions+=("restarted_keepalive_exhausted_fallback_rate_limited")
                        else
                            log_action "skip_ka_restart_exhausted (last ${rst_age}s ago; rate-limit 1800s)"
                            actions+=("skipped_ka_restart_rate_limited")
                        fi
                    else
                        issues+=("jdm_fallback_false_exhausted_both_idle_${idle_sec}s")
                        log_issue "jdm_fallback_false_exhausted_both_idle ${idle_sec}s — post-siege empty, restart keepalive for JDM"
                        restart_keepalive
                        actions+=("restarted_keepalive_exhausted_fallback")
                    fi
                fi
                ;;
        esac
    fi

    # 4. Siege crashed: stale log + dead orchestrator.
    # Never --force-loop exhausted/waived pairs (icamcnet Hisar peak-100 near-miss).
    if [[ "${siege_orch}" == "false" ]]; then
        local stale_line model manifest_path dataset=""
        while IFS= read -r stale_line; do
            [[ -z "${stale_line}" ]] && continue
            model="${stale_line%%$'\t'*}"
            issues+=("siege_stale_log_${model}")
            manifest_path="${REPO}/configs/amr_benchmark/retune/siege_${model}_10a.json"
            dataset="deepsig201610A"
            if [[ "${model}" == "icamcnet" ]]; then
                manifest_path="${REPO}/configs/amr_benchmark/retune/siege_icamcnet_hisar.json"
                dataset="hisar2019"
            elif [[ "${model}" == "resnetamr" ]]; then
                # Prefer 2018 marginal if pending; else skip generic stale relaunch.
                if [[ -f "${REPO}/configs/amr_benchmark/retune/siege_resnetamr_2018.json" ]]; then
                    manifest_path="${REPO}/configs/amr_benchmark/retune/siege_resnetamr_2018.json"
                    dataset="deepsig201801A"
                fi
            fi
            if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
                log_action "skip_stale_relaunch model=${model} dataset=${dataset} reason=force_blocked"
                actions+=("skipped_stale_${model}_force_blocked")
                continue
            fi
            if [[ -f "${manifest_path}" ]]; then
                if launch_siege_manifest "${manifest_path}" "${model}" "siege_${model}" "${dataset}"; then
                    actions+=("relaunched_stale_${model}")
                else
                    actions+=("skipped_stale_${model}")
                fi
            fi
        done < <(stale_siege_logs)
    fi

    # 5. Disconnect recovery
    local disc_line
    while IFS= read -r disc_line; do
        [[ -z "${disc_line}" ]] && continue
        issues+=("${disc_line// /_}")
        log_issue "${disc_line}"
    done < <(disconnect_recovery_check)

    # Serialize issues/actions for JSON
    local issues_json actions_json
    if [[ ${#issues[@]} -eq 0 ]]; then
        issues_json='[]'
    else
        issues_json="$("${PY}" -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${issues[@]}")"
    fi
    if [[ ${#actions[@]} -eq 0 ]]; then
        actions_json='[]'
    else
        actions_json="$("${PY}" -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${actions[@]}")"
    fi
    write_health_status "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${gpu0_util}" "${gpu1_util}" "${amr0}" "${amr1}" "${issues_json}" "${actions_json}"

    if [[ ${#issues[@]} -gt 0 ]]; then
        log "STATUS issues=${#issues[@]} actions=${#actions[@]}"
    else
        log "STATUS ok"
    fi
}

log "health_watchdog started PID=$$ interval=${INTERVAL}s gpu_idle_threshold=${GPU_IDLE_THRESHOLD}s"
tick
while true; do
    sleep "${INTERVAL}"
    tick
done
