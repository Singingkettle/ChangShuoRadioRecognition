#!/usr/bin/env bash
# Continuous GPU utilization daemon for AMR siege + JDM secondary slot.
# Policy: zero idle GPUs — never kill running jobs; auto-launch next queue work.
#
# Usage:
#   nohup bash tools/amr_benchmark/gpu_keepalive.sh >> work_dirs/amr_benchmark_retune/scheduler.log 2>&1 &

set -uo pipefail

# Align CUDA logical indices with nvidia-smi physical indices (see
# gpu_pool_keepalive.sh for the mispacking failure mode this prevents).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# REPO_ROOT / PYTHON overridable for multi-host deploy (paths identical on the
# local 2-GPU box and the remote 4xH100 box, but env override keeps it portable).
REPO="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
PY="${PYTHON:-/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python}"
LOGDIR="${REPO}/work_dirs/amr_benchmark_retune"
SCHED_LOG="${LOGDIR}/scheduler.log"
STATE_FILE="${LOGDIR}/scheduler_state.json"
QUEUE="${REPO}/configs/amr_benchmark/retune/siege_queue.json"
QUEUE_FULL="${REPO}/configs/amr_benchmark/retune/siege_queue_full.json"
WAVE1_MANIFEST="${REPO}/configs/amr_benchmark/retune/wave1_manifest.json"
WAVE1_LOG="${LOGDIR}/wave1_active.log"
WAVE4_MANIFEST="${REPO}/configs/amr_benchmark/retune/wave4_marginal_manifest.json"
WAVE4_LOG="${LOGDIR}/wave4_marginal.log"
GOAL_STATUS="${LOGDIR}/GOAL_STATUS.json"
SIEGE_R3_MANIFEST="${REPO}/configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json"
JDM_TRACKB_MANIFEST="${REPO}/configs/jdm/experiments/retune/wave3_trackb_manifest.json"
JDM_AMC_MANIFEST="${REPO}/configs/jdm/experiments/retune/wave_p1_amc_manifest.json"

# Tick every 2 min by default — idle GPUs must not wait 5+ min for the next chance.
INTERVAL="${GPU_KEEPALIVE_INTERVAL:-120}"
JDM_IDLE_THRESHOLD="${JDM_IDLE_THRESHOLD_SEC:-600}"
# Single-GPU idle backfill: act within ~5 min (was 10). Immediate path already
# fires when primary pending=0; this is the safety net during siege_orch races.
GPU1_BACKFILL_THRESHOLD="${GPU1_BACKFILL_THRESHOLD_SEC:-300}"
STREAK_IDLE_MAX="${STREAK_IDLE_MAX:-2}"

cd "${REPO}" || exit 1
mkdir -p "${LOGDIR}" "${REPO}/work_dirs/jdm/retune"

STREAK_IDLE=0

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" >> "${SCHED_LOG}"
}

log_error() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ERROR: $*" >> "${SCHED_LOG}"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ERROR: $*" >> "${LOGDIR}/health.log"
}

count_pending_entries() {
    "${PY}" - "${QUEUE}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
print(sum(1 for e in data.get("entries", []) if e.get("status", "pending") not in skip))
PY
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
        log "ACTION: reset queue entries: ${changed}"
    fi
}

startup_self_test() {
    local failures=0 pending_line entry_id manifest model dataset
    log "self-test: parsing siege queue with tab-delimited fields"
    pending_line="$(next_pending_entry "${QUEUE}" 2>/dev/null || true)"
    if [[ -n "${pending_line}" ]]; then
        IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
        if [[ -z "${entry_id}" || -z "${manifest}" ]]; then
            log_error "self-test: queue parse produced empty id/manifest from: ${pending_line}"
            failures=$((failures + 1))
        elif [[ ! -f "${REPO}/${manifest}" ]]; then
            log_error "self-test: manifest missing for ${entry_id}: ${REPO}/${manifest}"
            failures=$((failures + 1))
        else
            log "self-test: next pending=${entry_id} manifest_ok=${manifest}"
        fi
    else
        log "self-test: no pending queue entries (may be expected if campaign done)"
    fi

    for mf in "${SIEGE_R3_MANIFEST}" "${WAVE4_MANIFEST}" "${QUEUE}"; do
        if [[ ! -f "${mf}" ]]; then
            log_error "self-test: required path missing: ${mf}"
            failures=$((failures + 1))
        fi
    done

    reset_false_exhausted_queue

    if [[ "${failures}" -gt 0 ]]; then
        log_error "self-test FAILED (${failures} issue(s)) — watchdog should escalate"
        return 1
    fi
    log "self-test passed"
    return 0
}

# Return "amr0 amr1 jdm0 jdm1" counts of top-level train.py jobs per GPU.
count_trains_per_gpu() {
    "${PY}" - <<'PY'
import subprocess, re, sys
from collections import defaultdict

repo = "/home/citybuster/Projects/ChangShuoRadioRecognition"
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
except subprocess.CalledProcessError:
    print("0 0 0 0")
    sys.exit(0)

uuid_lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
uuid_order = []
pid_to_gpu = {}
for ln in uuid_lines:
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
jdm = defaultdict(int)
for pid, gpu in pid_to_gpu.items():
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").replace("\x00", " ")
    except OSError:
        continue
    if "tools/train.py" not in cmd:
        continue
    # Accept relative or absolute cmdlines (manual launches often omit abs repo path).
    if repo not in cmd and "ChangShuoRadioRecognition" not in cmd \
            and "configs/" not in cmd and "work_dirs/" not in cmd:
        continue
    # Parent train.py only (skip dataloader workers inheriting GPU context)
    try:
        ppid = int(open(f"/proc/{pid}/status").read().split("PPid:")[1].split()[0])
        ppcmd = open(f"/proc/{ppid}/cmdline", "rb").read().decode(errors="replace")
    except (OSError, IndexError, ValueError):
        ppcmd = ""
    if "tools/train.py" in ppcmd:
        continue
    if "configs/jdm" in cmd or "work_dirs/jdm" in cmd:
        jdm[gpu] += 1
    elif "amr_benchmark_retune" in cmd or "configs/amr_benchmark" in cmd:
        amr[gpu] += 1
    else:
        amr[gpu] += 1

for gpu in (0, 1):
    print(amr.get(gpu, 0), end=" ")
for gpu in (0, 1):
    print(jdm.get(gpu, 0), end=" " if gpu == 0 else "\n")
PY
}

# True if a live python orchestrator matches pattern (exclude bash waiters /
# scripts whose cmdline merely *mentions* the pattern — self-match trap).
# Stall class auto_seed_wave_spam_phantom / orphan waiter: bash -c that embeds
# "$PY … retune_model_siege.py" matched *python*$pattern* and parked siege_orch=true.
orchestrator_running() {
    local pattern="${1:-tools/amr_benchmark/retune_model_siege.py}"
    local pid cmdline exe base
    while read -r pid; do
        [[ -z "${pid}" || "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ -z "${cmdline}" ]] && continue
        # Bash/sh waiters and pgrep noise must never count as the orchestrator.
        case "${cmdline}" in
            *pgrep*|*bash*|*"/bin/sh "*|*" sh "*) continue ;;
        esac
        [[ "${cmdline}" == *"${pattern}"* ]] || continue
        # Prefer /proc/exe → real python interpreter (not a shell wrapping python path).
        exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
        base="$(basename "${exe}" 2>/dev/null || true)"
        case "${base}" in
            python|python3|python3.*) return 0 ;;
        esac
        # Fallback: argv0 looks like python and pattern is a script arg.
        case "${cmdline}" in
            python*" ${pattern}"*|python3*" ${pattern}"*|*"/python ${pattern}"*|*"/python3 ${pattern}"*)
                return 0 ;;
        esac
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
    return 1
}

# Live JDM work that should own GPU1 (train / test_det / AMC / python sweep).
# Bash waiters (launch_paper_exact_keepalive) must NOT count — a self-matching
# waiter previously kept jdm_amc_launched parked forever with GPU1 at 0%.
jdm_gpu1_live() {
    local pid cmdline
    while read -r pid; do
        [[ -z "${pid}" || "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
        cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
        # Skip patterns that only *mention* JDM (bash waiters / pgrep / shells).
        [[ "${cmdline}" == *pgrep* ]] && continue
        [[ "${cmdline}" == *launch_paper_exact_keepalive* ]] && continue
        case "${cmdline}" in
            *bash*|*"/bin/sh "*|*" sh "*) continue ;;
        esac
        # Real GPU work / python AMC pipeline.
        if [[ "${cmdline}" == *tools/precompute_amc_proposals.py* ]]; then
            return 0
        fi
        if [[ "${cmdline}" == *tools/train.py* || "${cmdline}" == *tools/test_det.py* ]] \
           && { [[ "${cmdline}" == *configs/jdm/* ]] || [[ "${cmdline}" == *work_dirs/jdm/* ]]; }; then
            return 0
        fi
        if [[ "${cmdline}" == *python*tools/jdm/retune_sweep.py* \
           || "${cmdline}" == *python3*tools/jdm/retune_sweep.py* \
           || "${cmdline}" == *python*tools/jdm/launch_wave* \
           || "${cmdline}" == *python3*tools/jdm/launch_wave* ]]; then
            return 0
        fi
        # launch_wave_p1_amc.sh only counts when it has already spawned python children
        # (detected above). The bare bash wrapper alone must not park GPU1.
    done < <(pgrep -f 'tools/train.py|tools/test_det.py|tools/precompute_amc_proposals|tools/jdm/retune_sweep' 2>/dev/null || true)
    return 1
}

# Clear jdm_amc_launched when no JDM process remains (don't park GPU1 forever).
clear_stale_jdm_amc_flag() {
    local state launched
    state="$(read_state)"
    launched="$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('true' if d.get('jdm_amc_launched') else 'false')" 2>/dev/null || echo false)"
    if [[ "${launched}" != "true" ]]; then
        return 0
    fi
    if jdm_gpu1_live; then
        return 0
    fi
    write_state "$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['jdm_amc_launched']=False; print(json.dumps(d))")"
    log "ACTION: cleared stale jdm_amc_launched — no live JDM process"
}

siege_r3_incomplete() {
    "${PY}" - <<'PY'
import json
from pathlib import Path

repo = Path("/home/citybuster/Projects/ChangShuoRadioRecognition")
manifest = json.loads(
    (repo / "configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json").read_text()
)
root = repo / "work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A"
for exp in manifest.get("experiments", []):
    variant = exp["variant"]
    paper = root / variant / "res" / "paper.pkl"
    if not paper.exists():
        print("yes")
        raise SystemExit(0)
print("no")
PY
}

next_pending_entry() {
    local qpath="${1:-${QUEUE}}"
    "${PY}" - "${qpath}" <<'PY'
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

# Queue-terminal only when there is NO pending/running entry for this pair.
# Older exhausted entries (e.g. siege_fastmldnn_10a) must NOT block a newer
# pending wave (siege_fastmldnn_10a_wave6) for the same model/dataset.
pair_queue_terminal() {
    local model="$1" dataset="$2"
    "${PY}" - "${QUEUE}" "${QUEUE_FULL}" "${model}" "${dataset}" <<'PY'
import json, sys
from pathlib import Path

queue_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
model = sys.argv[3].lower()
dataset = sys.argv[4]
saw_pair = False
has_open = False
has_waiver = False
for qp in queue_paths:
    if not qp.is_file():
        continue
    data = json.loads(qp.read_text())
    for e in data.get("entries", []):
        if (e.get("model") or "").lower() != model or e.get("dataset") != dataset:
            continue
        saw_pair = True
        status = e.get("status", "pending")
        if e.get("waiver") or status.startswith("waived"):
            has_waiver = True
            continue
        if status in {"pending", "running"}:
            has_open = True
if not saw_pair:
    print("no")
elif has_open:
    print("no")
elif has_waiver:
    print("yes")
else:
    # All entries for this pair are passed/exhausted/skipped — terminal.
    print("yes")
PY
}

# True when --force must be refused: queue-terminal OR ≥3 identical fail re-logs
# (unless the pair already has a pass in retune_results — never block a pass).
pair_force_blocked() {
    local model="$1" dataset="$2"
    if [[ "$(pair_queue_terminal "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
        echo yes
        return 0
    fi
    "${PY}" - "${REPO}/docs/amr_benchmark/retune_results.md" "${model}" "${dataset}" <<'PY'
import re, sys
from collections import defaultdict
from pathlib import Path

results_md = Path(sys.argv[1])
model = sys.argv[2].lower()
dataset = sys.argv[3]
label = f"{model}/{dataset}"
if not results_md.is_file():
    print("no")
    raise SystemExit(0)
text = results_md.read_text(errors="replace")
# Any pass for this pair ⇒ not force-blocked (fail re-logs of other variants OK).
pass_re = re.compile(
    r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
    + re.escape(label)
    + r"\s*\|\s*`[^`]+`\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*`pass`\s*\|",
    re.I,
)
if pass_re.search(text):
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
        overall_b = int(round(float(overall_s) * 4))
        peak_b = int(round(float(peak_s) * 4))
    except ValueError:
        continue
    clusters[(variant, overall_b, peak_b)] += 1
print("yes" if any(n >= 3 for n in clusters.values()) else "no")
PY
}

# Stall class: phantom_pending_force_blocked — status=pending but every entry
# skipped by identical-fail heuristic while full_pending>0. Reconcile so counts
# match launchability (pass→passed, recipe-exhausted→exhausted, stale running→fix).
reconcile_phantom_pending_full() {
    "${PY}" "${REPO}/tools/amr_benchmark/reconcile_phantom_pending.py" \
        --queue-full "${QUEUE_FULL}" \
        --results "${REPO}/docs/amr_benchmark/retune_results.md" \
        --repo "${REPO}"
}

next_pending_manifest_entry_full() {
    # Prefer entries with a siege manifest; synthesize for null-manifest.
    # Skip only queue-terminal statuses (not identical-fail re-logs). Phantom
    # pending is closed by reconcile_phantom_pending_full before select.
    "${PY}" - "${QUEUE_FULL}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
pending = sorted(
    [e for e in data.get("entries", []) if e.get("status", "pending") not in skip],
    key=lambda e: e.get("priority", 99),
)
for entry in pending:
    manifest = entry.get("manifest")
    model = entry.get("model", "") or ""
    dataset = entry.get("dataset", "") or ""
    if manifest:
        print("\t".join([entry.get("id", ""), manifest, model, dataset]))
        raise SystemExit(0)
for entry in pending:
    model = entry.get("model")
    dataset = entry.get("dataset")
    if not model or not dataset:
        continue
    print("\t".join([entry.get("id", ""), f"__synthesize__:{model}:{dataset}", model, dataset]))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_marginal_siege_manifest() {
    # Build ES-patience30 (+ optional warmup) configs + mini-manifest for a model/dataset pair.
    # Prints relative manifest path on success.
    local model="$1" dataset="$2" entry_id="${3:-}"
    "${PY}" - "${REPO}" "${model}" "${dataset}" "${entry_id}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(sys.argv[1])
model, dataset, eid = sys.argv[2], sys.argv[3], sys.argv[4]
retune = repo / "configs/amr_benchmark/retune"
# Prefer MATRIX config path (handles petcgdnn shape-L-F, lstm2 ap, etc.)
sys.path.insert(0, str(repo / "tools"))
try:
    from amr_benchmark.matrix import MATRIX
    matrix_cfg = MATRIX.get(model, {}).get(dataset, {}).get("config")
except Exception:
    matrix_cfg = None
if matrix_cfg:
    # configs/foo/bar.py → ../../foo/bar.py from retune/
    parts = matrix_cfg.split("/", 1)[1] if matrix_cfg.startswith("configs/") else matrix_cfg
    base_rel = "../../" + parts
    base_abs = repo / matrix_cfg
else:
    if dataset.startswith("deepsig"):
        slug = "deepsig-" + dataset[len("deepsig"):]
    elif dataset.startswith("hisar"):
        slug = "hisar-" + dataset[len("hisar"):]
    else:
        slug = dataset
    base_rel = f"../../{model}/{model}_iq-{slug}.py"
    base_abs = repo / "configs" / model / f"{model}_iq-{slug}.py"
if not base_abs.is_file():
    print(f"missing base config: {base_abs}", file=sys.stderr)
    raise SystemExit(1)

def write_es(path: Path) -> None:
    if path.is_file():
        return
    path.write_text(
        f'"""Auto marginal retune: {model} @ {dataset} — relaxed early stopping."""\n\n'
        f"_base_ = ['{base_rel}']\n\n"
        "custom_hooks = [\n"
        "    dict(type='EarlyStoppingHook', monitor='accuracy/top1',\n"
        "         min_delta=0.05, patience=30, rule='greater'),\n"
        "]\n"
    )

def write_warmup(path: Path) -> None:
    if path.is_file():
        return
    path.write_text(
        f'"""Auto marginal retune: {model} @ {dataset} — lower LR + warmup."""\n\n'
        f"_base_ = ['{base_rel}']\n\n"
        "optim_wrapper = dict(\n"
        "    optimizer=dict(type='Adam', lr=2e-4),\n"
        "    clip_grad=dict(max_norm=5.0, norm_type=2),\n"
        ")\n\n"
        "param_scheduler = [\n"
        "    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5,\n"
        "         convert_to_iter_based=True),\n"
        "    dict(type='CosineAnnealingLR', by_epoch=True, T_max=145, begin=5, end=150,\n"
        "         eta_min=1e-6),\n"
        "]\n"
    )

es_name = f"wave4_{model}_{dataset}_es_patience30.py"
wu_name = f"wave4_{model}_{dataset}_lr2e4_warmup.py"
write_es(retune / es_name)
write_warmup(retune / wu_name)
manifest_name = f"siege_{model}_{dataset}.json"
# Prefer short alias for known ids
if eid:
    alias = retune / f"{eid.replace('siege_', 'siege_')}.json"
else:
    alias = retune / manifest_name
# Use siege_<model>_<short> when possible
short_map = {
    ("resnetamr", "deepsig201801A"): "siege_resnetamr_2018.json",
    ("mcnet", "deepsig201610A"): "siege_mcnet_10a.json",
    ("cnn1dpf", "deepsig201610A"): "siege_cnn1dpf_10a.json",
}
manifest_rel = f"configs/amr_benchmark/retune/{short_map.get((model, dataset), manifest_name)}"
manifest_path = repo / manifest_rel
payload = {
    "description": f"Auto siege — {model} @ {dataset} (marginal paper-exact)",
    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "experiments": [
        {
            "id": f"W4_{model}_{dataset}_es30",
            "priority": 0,
            "model": model,
            "dataset": dataset,
            "variant": "es_patience30",
            "config": f"configs/amr_benchmark/retune/{es_name}",
            "notes": "Auto ES patience 30",
        },
        {
            "id": f"W4_{model}_{dataset}_warmup",
            "priority": 0,
            "model": model,
            "dataset": dataset,
            "variant": "lr2e4_warmup",
            "config": f"configs/amr_benchmark/retune/{wu_name}",
            "notes": "Auto lr=2e-4 warmup",
        },
    ],
}
if not manifest_path.is_file():
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
# Patch queue_full entry if present
qpath = repo / "configs/amr_benchmark/retune/siege_queue_full.json"
if qpath.is_file():
    q = json.loads(qpath.read_text())
    for e in q.get("entries", []):
        if e.get("model") == model and e.get("dataset") == dataset and not e.get("manifest"):
            e["manifest"] = manifest_rel
            q["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            qpath.write_text(json.dumps(q, indent=2) + "\n")
            break
print(manifest_rel)
PY
}

wave1_goal_exhausted() {
    "${PY}" - "${WAVE1_MANIFEST}" "${GOAL_STATUS}" <<'PY'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
goal_path = Path(sys.argv[2])
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
pairs = {(e["model"].lower(), e["dataset"]) for e in manifest.get("experiments", [])}
if not pairs:
    raise SystemExit(1)
if not goal_path.is_file():
    raise SystemExit(1)
goal = json.loads(goal_path.read_text())
exhausted = set()
for label in goal.get("exhausted_pairs", []):
    model, dataset = label.split("/", 1)
    exhausted.add((model.lower(), dataset))
print("yes" if pairs <= exhausted else "no")
PY
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
    local now_ts extra=()
    now_ts="$(date -u +%s)"
    # Wave4 includes icamcnet@hisar etc. — never --force-loop pairs already
    # exhausted / waived / stuck on identical paper-exact fails.
    if [[ "${force_flag}" == "force" ]]; then
        local blocked=0
        for pair in "icamcnet:hisar2019" "hcgdnn:deepsig201610A" "lstm2:deepsig201610A" "resnetamr:deepsig201610B"; do
            IFS=':' read -r m d <<< "${pair}"
            if [[ "$(pair_force_blocked "${m}" "${d}" 2>/dev/null || echo no)" == "yes" ]]; then
                blocked=$((blocked + 1))
            fi
        done
        if [[ "${blocked}" -ge 3 ]]; then
            log "SKIP wave4 --force — ${blocked}/4 known pairs force-blocked (exhausted or ≥3 identical fails)"
            force_flag=""
        else
            extra+=(--force)
        fi
    fi
    log "ACTION: wave1 exhausted — launching wave4 marginal manifest on both GPUs"
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest "${WAVE4_MANIFEST}" \
        --gpu 0,1 --max-parallel 2 \
        --until-pass --paper-exact --promote \
        "${extra[@]}" \
        >> "${WAVE4_LOG}" 2>&1 &
    local pid=$!
    log "Launched wave4 marginal siege PID=${pid} log=${WAVE4_LOG} force=${force_flag:-no}"
    write_state "$("${PY}" -c "
import json
from pathlib import Path
state_path = Path('${STATE_FILE}')
state = json.loads(state_path.read_text()) if state_path.is_file() else {}
state['wave4_last_launch'] = ${now_ts}
state['wave4_last_pid'] = ${pid}
print(json.dumps(state))
")"
}

launch_wave1_goal_sweep() {
    local now_ts
    now_ts="$(date -u +%s)"
    log "ACTION: siege_queue exhausted — launching goal-mode wave1 on both GPUs"
    nohup "${PY}" tools/amr_benchmark/retune_sweep.py \
        --manifest "${WAVE1_MANIFEST}" \
        --gpu 0,1 --max-parallel 2 \
        --goal-mode --stop-when-all-pass --paper-exact \
        >> "${WAVE1_LOG}" 2>&1 &
    local pid=$!
    log "Launched AMR goal-mode wave1 PID=${pid} log=${WAVE1_LOG}"
    write_state "$("${PY}" -c "
import json
from pathlib import Path
state_path = Path('${STATE_FILE}')
state = json.loads(state_path.read_text()) if state_path.is_file() else {}
state['wave1_last_launch'] = ${now_ts}
state['wave1_last_pid'] = ${pid}
print(json.dumps(state))
")"
}

launch_full_queue_marginal_siege() {
    local force_flag="$1"
    local entry_id manifest model dataset pending_line gpu_arg max_par
    local amr0_now="${2:-0}" amr1_now="${3:-0}"
    pending_line="$(next_pending_manifest_entry_full 2>/dev/null || true)"
    if [[ -z "${pending_line}" ]]; then
        return 1
    fi
    IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
    if [[ "${manifest}" == __synthesize__:* ]]; then
        manifest="$(ensure_marginal_siege_manifest "${model}" "${dataset}" "${entry_id}" 2>>"${SCHED_LOG}" || true)"
        if [[ -z "${manifest}" || ! -f "${REPO}/${manifest}" ]]; then
            log_error "failed to synthesize siege manifest for ${entry_id} (${model}/${dataset})"
            return 1
        fi
        log "ACTION: synthesized marginal siege manifest ${manifest} for ${entry_id}"
    fi
    local extra=()
    if [[ "${force_flag}" == "force" ]]; then
        if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
            log "SKIP force for ${model}/${dataset} (${entry_id}) — exhausted/waived or ≥3 identical paper-exact fails; not looping"
            force_flag=""
        else
            extra+=(--force)
        fi
    fi
    if [[ "${amr0_now}" -ge 1 && "${amr1_now}" -eq 0 ]]; then
        gpu_arg="1"; max_par=1
    elif [[ "${amr0_now}" -eq 0 && "${amr1_now}" -ge 1 ]]; then
        gpu_arg="0"; max_par=1
    else
        gpu_arg="0,1"; max_par=2
    fi
    log "ACTION: launching marginal siege from siege_queue_full (${entry_id}) on gpu=${gpu_arg}"
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest "${REPO}/${manifest}" \
        --gpu "${gpu_arg}" --max-parallel "${max_par}" \
        --until-pass --paper-exact --promote \
        "${extra[@]}" \
        >> "${LOGDIR}/siege_${model}.log" 2>&1 &
    log "Launched full-queue marginal siege PID=$! entry=${entry_id} manifest=${manifest} force=${force_flag:-no}"
    return 0
}

launch_post_siege_work() {
    local force_flag="${1:-}"
    local amr0_now="${2:-0}" amr1_now="${3:-0}"
    if [[ "${STREAK_IDLE}" -ge "${STREAK_IDLE_MAX}" ]]; then
        force_flag="force"
        log_error "STREAK_IDLE=${STREAK_IDLE} — post-siege escalation with force"
    fi
    # Close phantom pending (status=pending but recipes already done / force-blocked
    # by identical re-logs) so full_pending matches launchable work.
    local reconciled
    reconciled="$(reconcile_phantom_pending_full 2>/dev/null || true)"
    if [[ -n "${reconciled}" ]]; then
        log "ACTION: reconcile_phantom_pending ${reconciled}"
    fi
    # Prefer siege_queue_full pending before wave1/wave4 — avoids idle wave1
    # relaunch loops when GOAL_STATUS is falsely campaign_complete / empty
    # exhausted_pairs after a tracking wipe, while marginals still remain.
    log "post-siege: trying siege_queue_full marginals first"
    if launch_full_queue_marginal_siege "${force_flag}" "${amr0_now}" "${amr1_now}"; then
        STREAK_IDLE=0
        return 0
    fi
    local pending_full_n
    pending_full_n="$(count_pending_entries_full)"
    if [[ "${pending_full_n}" -gt 0 ]]; then
        # Should not happen after reconcile: pending>0 but selector empty.
        log_error "phantom_pending_force_blocked stall: full_pending=${pending_full_n} but no launchable entry after reconcile — clearing via re-reconcile"
        reconcile_phantom_pending_full >/dev/null 2>&1 || true
        if launch_full_queue_marginal_siege "${force_flag}" "${amr0_now}" "${amr1_now}"; then
            STREAK_IDLE=0
            return 0
        fi
    fi
    if [[ "$(wave1_goal_exhausted 2>/dev/null || echo no)" != "yes" ]]; then
        launch_wave1_goal_sweep
        STREAK_IDLE=0
        return 0
    fi
    if [[ "$(wave4_marginal_exhausted 2>/dev/null || echo no)" != "yes" && -f "${WAVE4_MANIFEST}" ]]; then
        launch_wave4_marginal_siege "${force_flag}"
        STREAK_IDLE=0
        return 0
    fi
    # Generate+seed next near-miss wave BEFORE JDM fallback so AMR Tier-A
    # never waits on exhausted JDM done.flags (2026-07-22 ~36h idle root cause:
    # auto_seed only ran after JDM and refused to create missing manifests).
    local seeded_early
    seeded_early="$(auto_seed_next_near_miss_wave 2>/dev/null || true)"
    if [[ -n "${seeded_early}" && "${seeded_early}" != "all_waves_exhausted_no_next_seed" ]]; then
        log "ACTION: auto-seeded next near-miss wave (pre-JDM): ${seeded_early}"
        if launch_full_queue_marginal_siege "${force_flag}" "${amr0_now}" "${amr1_now}"; then
            STREAK_IDLE=0
            return 0
        fi
    fi
    # AMR queues truly exhausted — keep GPUs busy with JDM ideal (v1) next step.
    if launch_jdm_ideal_fallback "${amr0_now}" "${amr1_now}"; then
        STREAK_IDLE=0
        return 0
    fi
    log "no post-siege work available (full queue + wave1 + wave4 + jdm ideal exhausted)"
    return 1
}

# Newest best_detection_mAP_epoch_*.pth under a work dir (by mtime).
_jdm_newest_best_ckpt() {
    local dir="$1"
    "${PY}" - "${dir}" <<'PY'
import sys
from pathlib import Path
d = Path(sys.argv[1])
cands = sorted(d.glob("best_detection_mAP_epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
print(cands[0] if cands else "")
PY
}

# True if a real python tools/test_det.py ideal/AWGN eval is live on gpu_arg.
# Avoids classic pgrep self-match (pattern in argv) that false-skipped fallback
# under ~870 concurrent keepalive zombies (2026-07-20 ~14.7h idle).
_jdm_eval_live_on_gpu() {
    local gpu_arg="$1"
    "${PY}" - "${gpu_arg}" <<'PY'
import os, sys
from pathlib import Path
gpu = sys.argv[1]
needle = ("tools/test_det.py",)
cfgs = ("eval_ideal_v1", "eval_awgn_snr12_30")
for pid in Path("/proc").iterdir():
    if not pid.name.isdigit():
        continue
    try:
        cmd = (pid / "cmdline").read_bytes().decode("utf-8", "ignore")
    except OSError:
        continue
    if "python" not in cmd or "tools/test_det.py" not in cmd:
        continue
    if not any(c in cmd for c in cfgs):
        continue
    # Prefer CUDA_VISIBLE_DEVICES match; also accept nvidia-smi occupancy via env.
    envp = pid / "environ"
    try:
        env = envp.read_bytes().decode("utf-8", "ignore")
    except OSError:
        env = ""
    cvd = ""
    for part in env.split("\0"):
        if part.startswith("CUDA_VISIBLE_DEVICES="):
            cvd = part.split("=", 1)[1]
            break
    if cvd == gpu or (cvd == "" and gpu == "0"):
        sys.exit(0)
sys.exit(1)
PY
}

# When AMR queues + listed JDM steps are exhausted, auto-seed the next near-miss
# wave into siege_queue_full (stall class all_waves_exhausted_no_next_seed).
# CRITICAL (2026-07-22 ~36h idle): must GENERATE configs+manifests when missing —
# merely looking for siege_*_wave{N}.json and exiting 2 left both GPUs idle forever.
# Returns 0 if new pending work was added.
auto_seed_next_near_miss_wave() {
    # Cap/rate-limit auto-seed (stall class auto_seed_wave_spam_phantom):
    # - at most ONE new wave ahead of the highest non-junk wave
    # - refuse while pending OR stale running remains
    # - require distinct wave-suffixed variants that lack retune_results rows
    # - never reopen exhausted auto-seed entries; never phantom-exhaust empty waves
    "${PY}" - "${QUEUE_FULL}" "${REPO}" <<'PY'
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

queue_path = Path(sys.argv[1])
repo = Path(sys.argv[2])
cfg = repo / "configs/amr_benchmark/retune"
stamp = repo / "work_dirs/amr_benchmark_retune/auto_seed_stamp.json"
results_md = repo / "docs/amr_benchmark/retune_results.md"
data = json.loads(queue_path.read_text())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
MAX_AHEAD = 1
MIN_SEED_INTERVAL_SEC = 6 * 3600  # rate-limit: at most one auto-seed per 6h

def wave_num(eid: str):
    m = re.search(r"wave(\d+)", eid or "")
    return int(m.group(1)) if m else None

entries = data.get("entries", [])
active = [e for e in entries if e.get("status", "pending") in {"pending", "running"}]
if active:
    ids = ",".join(e.get("id", "?") for e in active[:8])
    print(f"auto_seed_blocked_active:{ids}")
    sys.exit(1)

# Highest "real" wave = hand-seeded / non-auto-junk.
real_waves = []
junk_waves = []
for e in entries:
    n = wave_num(e.get("id", ""))
    if n is None:
        continue
    notes = e.get("notes") or ""
    is_auto_junk = ("auto_seed_next_near_miss" in notes) and e.get("status") == "exhausted"
    if is_auto_junk:
        junk_waves.append(n)
    else:
        real_waves.append(n)

anchor = max(real_waves) if real_waves else 9
nxt = anchor + MAX_AHEAD
# If a non-junk entry already exists at/above nxt, do not advance further.
if any(n >= nxt for n in real_waves if n > anchor):
    print(f"auto_seed_capped_anchor{anchor}_already_ahead")
    sys.exit(3)
# Ignore junk wave numbers (wave10–81 spam) when choosing nxt; never chase max(junk).

# Rate-limit stamp
if stamp.is_file():
    try:
        prev = json.loads(stamp.read_text())
        prev_t = datetime.fromisoformat(
            prev.get("at", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")
        )
        age = (datetime.now(timezone.utc) - prev_t).total_seconds()
        if age < MIN_SEED_INTERVAL_SEC and int(prev.get("wave", -1)) == nxt:
            print(f"auto_seed_rate_limited_age{int(age)}s_wave{nxt}")
            sys.exit(4)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass

results_text = results_md.read_text(errors="replace") if results_md.is_file() else ""

def variant_has_result(model: str, dataset: str, variant: str) -> bool:
    label = f"{model}/{dataset}"
    return bool(re.search(
        r"\|\s*[0-9-]+\s+[0-9:]+\s*\|\s*"
        + re.escape(label)
        + r"\s*\|\s*`"
        + re.escape(variant)
        + r"`\s*\|",
        results_text,
        re.I,
    ))

def write_new(path: Path, body: str) -> None:
    path.write_text(body)

def generate_wave_artifacts(n: int):
    """Create DISTINCT wave-suffixed Tier-A FT recipes (never reuse bare variant names)."""
    fast_ckpt = repo / (
        "work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A/"
        "paper_fixedlr_l2_ft80_from_w8best/best_accuracy_top1_epoch_37.pth"
    )
    if not fast_ckpt.is_file():
        cands = sorted(
            (repo / "work_dirs/amr_benchmark_retune/fastmldnn/deepsig201610A").glob(
                "*/best_accuracy_top1_epoch_*.pth"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        fast_ckpt = cands[0] if cands else None
    hcg_ckpt = repo / (
        "work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A/"
        "paper_multistep_exact800_esoff1600/best_accuracy_top1_epoch_968.pth"
    )
    if not hcg_ckpt.is_file():
        cands = sorted(
            (repo / "work_dirs/amr_benchmark_retune/hcgdnn/deepsig201610A").glob(
                "*/best_accuracy_top1_epoch_*.pth"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        hcg_ckpt = cands[0] if cands else None

    fast_specs = []
    if fast_ckpt and fast_ckpt.is_file():
        rel = fast_ckpt.relative_to(repo).as_posix()
        v = f"paper_fixedlr_l2_ft120_lr5e5_from_w9best_w{n}"
        write_new(cfg / f"wave{n}_fastmldnn_deepsig201610A_{v}.py", f"""\"\"\"Wave-{n} auto: FastMLDNN gentle FT 120ep @ lr5e-5 from W9 best (arch freeze).\"\"\"
_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']
load_from = '{rel}'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
""")
        fast_specs.append((v, 0, f"gentle FT from W9 best ckpt wave{n}"))

    hcg_specs = []
    if hcg_ckpt and hcg_ckpt.is_file():
        rel = hcg_ckpt.relative_to(repo).as_posix()
        v = f"paper_multistep_l2_ft120_lr5e5_from_w9best_w{n}"
        write_new(cfg / f"wave{n}_hcgdnn_deepsig201610A_{v}.py", f"""\"\"\"Wave-{n} auto: HCGDNN gentle FT 120ep @ lr5e-5 from W9 best 63.30 (arch freeze).\"\"\"
_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']
load_from = '{rel}'
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-5, weight_decay=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
param_scheduler = dict(_delete_=True, type='CosineAnnealingLR', by_epoch=True, T_max=120, eta_min=1e-6)
""")
        hcg_specs.append((v, 0, f"gentle FT from W9 HCG 63.30 wave{n}"))

    fast_specs = [s for s in fast_specs if not variant_has_result("fastmldnn", "deepsig201610A", s[0])]
    hcg_specs = [s for s in hcg_specs if not variant_has_result("hcgdnn", "deepsig201610A", s[0])]
    if not fast_specs and not hcg_specs:
        return [], []

    def man_exps(model, specs):
        exps = []
        for i, (variant, pri, notes) in enumerate(specs):
            conf = f"configs/amr_benchmark/retune/wave{n}_{model}_deepsig201610A_{variant}.py"
            exps.append({
                "id": f"W{n}_{model}_10A_{variant}"[:80],
                "priority": pri if pri is not None else i,
                "model": model, "dataset": "deepsig201610A", "variant": variant,
                "config": conf, "notes": notes,
            })
        return exps

    if fast_specs:
        (cfg / f"siege_fastmldnn_10a_wave{n}.json").write_text(json.dumps({
            "description": f"Wave-{n} Tier-A FastMLDNN auto (capped; distinct FT only)",
            "updated_at": now, "campaign_mode": "paper_exact",
            "paper_target_overall": 63.24, "paper_target_peak": 92.0,
            "experiments": man_exps("fastmldnn", fast_specs),
        }, indent=2) + "\n")
    if hcg_specs:
        (cfg / f"siege_hcgdnn_10a_wave{n}.json").write_text(json.dumps({
            "description": f"Wave-{n} Tier-A HCGDNN auto (capped; distinct FT only)",
            "updated_at": now, "campaign_mode": "paper_exact",
            "paper_target_overall": 64.9, "paper_target_peak": 93.0,
            "experiments": man_exps("hcgdnn", hcg_specs),
        }, indent=2) + "\n")
    return fast_specs, hcg_specs

fast_man = cfg / f"siege_fastmldnn_10a_wave{nxt}.json"
hcg_man = cfg / f"siege_hcgdnn_10a_wave{nxt}.json"
need_gen = True
if fast_man.is_file() and hcg_man.is_file():
    try:
        fe = json.loads(fast_man.read_text()).get("experiments") or []
        he = json.loads(hcg_man.read_text()).get("experiments") or []

        def useful(exps, model):
            for e in exps:
                v = e.get("variant") or ""
                if v.endswith(f"_w{nxt}") and not variant_has_result(model, "deepsig201610A", v):
                    return True
            return False

        if useful(fe, "fastmldnn") or useful(he, "hcgdnn"):
            need_gen = False
    except (OSError, json.JSONDecodeError):
        need_gen = True

if need_gen:
    for p in (fast_man, hcg_man, cfg / f"siege_tierb_wave{nxt}.json"):
        if p.is_file():
            p.unlink()
    fast_specs, hcg_specs = generate_wave_artifacts(nxt)
    if not fast_specs and not hcg_specs:
        print("auto_seed_no_distinct_configs")
        sys.exit(5)

candidates = []
if (cfg / f"siege_fastmldnn_10a_wave{nxt}.json").is_file():
    candidates.append((
        f"siege_fastmldnn_10a_wave{nxt}", f"siege_fastmldnn_10a_wave{nxt}.json",
        "fastmldnn", "deepsig201610A", 0, 0.35,
    ))
if (cfg / f"siege_hcgdnn_10a_wave{nxt}.json").is_file():
    candidates.append((
        f"siege_hcgdnn_10a_wave{nxt}", f"siege_hcgdnn_10a_wave{nxt}.json",
        "hcgdnn", "deepsig201610A", 0, 1.6,
    ))

added = []
ids = {e.get("id") for e in entries}
for eid, fname, model, dataset, pri, gap in candidates:
    man = cfg / fname
    if not man.is_file() or eid in ids:
        continue
    try:
        man_data = json.loads(man.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    exps = man_data.get("experiments") or []
    if not any(not variant_has_result(model, dataset, e.get("variant") or "") for e in exps):
        continue
    data.setdefault("entries", []).append({
        "id": eid,
        "priority": pri,
        "model": model,
        "dataset": dataset,
        "manifest": f"configs/amr_benchmark/retune/{fname}",
        "status": "pending",
        "gap_pp": gap,
        "failing_metric": "overall",
        "notes": f"auto_seed_capped_wave{nxt} {now} max_ahead={MAX_AHEAD}",
    })
    added.append(eid)

if not added:
    print("auto_seed_no_new_pending")
    sys.exit(6)

data["updated_at"] = now
note = data.get("notes") or ""
data["notes"] = note + f" | auto_seed_capped_wave{nxt}_{now} " + ",".join(added)
queue_path.write_text(json.dumps(data, indent=2) + "\n")
stamp.parent.mkdir(parents=True, exist_ok=True)
stamp.write_text(json.dumps({"at": now, "wave": nxt, "added": added}, indent=2) + "\n")
print(",".join(added))
sys.exit(0)
PY
}

# Per-ckpt eval dir — never share one done.flag across recipes (stall class
# jdm_fallback_false_exhausted_both_idle: improved ep4 never re-eval'd because
# shared eval_ideal_v1_det/done.flag was stamped after base ideal train).
_jdm_eval_dir_for_ckpt() {
    local jdm_dir="$1" ckpt="$2"
    local recipe ep
    recipe="$(basename "$(dirname "${ckpt}")")"
    ep="$(basename "${ckpt}" .pth)"
    echo "${jdm_dir}/eval_ideal_v1_det_${recipe}_${ep}"
}

# True if done.flag missing or older than ckpt (needs re-eval).
_jdm_eval_stale_vs_ckpt() {
    local done_flag="$1" ckpt="$2"
    [[ -z "${ckpt}" || ! -f "${ckpt}" ]] && return 1
    [[ ! -f "${done_flag}" ]] && return 0
    local done_m ckpt_m
    done_m="$(stat -c '%Y' "${done_flag}" 2>/dev/null || echo 0)"
    ckpt_m="$(stat -c '%Y' "${ckpt}" 2>/dev/null || echo 0)"
    [[ "${ckpt_m}" -gt "${done_m}" ]]
}

# Parse test mAP from an eval log (last detection/mAP in Epoch(test) line).
_jdm_test_map_from_log() {
    local logf="$1"
    [[ -f "${logf}" ]] || { echo ""; return; }
    "${PY}" - "${logf}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="ignore")
maps = re.findall(r"Epoch\(test\).*?detection/mAP:\s*([0-9.]+)", text)
print(maps[-1] if maps else "")
PY
}

launch_jdm_ideal_fallback() {
    # Fair ideal comparison: versions=['v1'] (no random impairments / infdB).
    # Skip known-failed 5ep AP75 FT / 5ep+AMC AWGN merge paths.
    # Called when AMR queues empty OR GPU1 idle >threshold with nothing launchable.
    local amr0_now="${1:-0}" amr1_now="${2:-0}"
    local jdm_dir="${REPO}/work_dirs/jdm/retune"
    local det_cfg="${REPO}/configs/jdm/experiments/retune/eval_ideal_v1_det.py"
    local joint_cfg="${REPO}/configs/jdm/experiments/retune/eval_ideal_v1_joint.py"
    local awgn_det_cfg="${REPO}/configs/jdm/experiments/retune/eval_awgn_snr12_30_det.py"
    local train_cfg="${REPO}/configs/jdm/experiments/retune/det_ideal_v1_30ep.py"
    local train_impr_cfg="${REPO}/configs/jdm/experiments/retune/det_ideal_v1_anchor110130150_30ep.py"
    local train_60_cfg="${REPO}/configs/jdm/experiments/retune/det_ideal_v1_60ep_lr5e4.py"
    local train_dir="${jdm_dir}/det_ideal_v1_30ep"
    local train_impr_dir="${jdm_dir}/det_ideal_v1_anchor110130150_30ep"
    local train_60_dir="${jdm_dir}/det_ideal_v1_60ep_lr5e4"
    local baseline_test_map="0.3850"  # ideal ep7 test mAP (2026-07-18)
    mkdir -p "${jdm_dir}"
    # Pick a free GPU (AMR+JDM occupancy). Prefer GPU1 when GPU0 already busy so
    # ideal-on-0 does not block GPU1 fill (prior bug: both-amr-idle always chose 0).
    local _tc _a0 _a1 _j0 _j1 busy0=0 busy1=0
    _tc="$(count_trains_per_gpu 2>/dev/null || echo '0 0 0 0')"
    read -r _a0 _a1 _j0 _j1 <<< "${_tc}"
    [[ "${_a0:-0}" -ge 1 || "${_j0:-0}" -ge 1 ]] && busy0=1
    [[ "${_a1:-0}" -ge 1 || "${_j1:-0}" -ge 1 ]] && busy1=1
    local gpu_arg=""
    if [[ "${busy0}" -eq 1 && "${busy1}" -eq 0 ]]; then
        gpu_arg="1"
    elif [[ "${busy0}" -eq 0 && "${busy1}" -eq 1 ]]; then
        gpu_arg="0"
    elif [[ "${busy0}" -eq 0 && "${busy1}" -eq 0 ]]; then
        gpu_arg="0"
    else
        return 1
    fi
    # Skip only when the *target* GPU already has ideal/AWGN work (allow parallel
    # ideal on the other GPU).
    if [[ "${gpu_arg}" == "1" && "${_j1:-0}" -ge 1 ]] \
        || [[ "${gpu_arg}" == "0" && "${_j0:-0}" -ge 1 ]]; then
        log "JDM ideal fallback skip — target GPU${gpu_arg} already has JDM"
        return 1
    fi
    # Require a real python tools/test_det.py (not pgrep self-match / shell argv spam).
    if _jdm_eval_live_on_gpu "${gpu_arg}"; then
        log "JDM ideal fallback skip — ideal/AWGN eval already live on GPU${gpu_arg}"
        return 1
    fi
    # 1) Base ideal 30ep train until epoch_30.pth exists.
    if [[ -f "${train_cfg}" && ! -f "${train_dir}/epoch_30.pth" ]]; then
        log "ACTION: AMR exhausted — launching JDM ideal v1 det train on GPU${gpu_arg}"
        mkdir -p "${train_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_arg}" nohup "${PY}" tools/train.py "${train_cfg}" \
            --work-dir "${train_dir}" \
            >> "${jdm_dir}/det_ideal_v1_30ep.log" 2>&1 &
        log "Launched JDM ideal v1 train PID=$! gpu=${gpu_arg}"
        return 0
    fi
    # 2) Per-recipe ideal det re-eval (each best ckpt gets its own eval dir).
    local recipe_dir ckpt eval_dir
    for recipe_dir in "${train_dir}" "${train_impr_dir}" "${train_60_dir}"; do
        ckpt="$(_jdm_newest_best_ckpt "${recipe_dir}")"
        [[ -z "${ckpt}" || ! -f "${det_cfg}" ]] && continue
        eval_dir="$(_jdm_eval_dir_for_ckpt "${jdm_dir}" "${ckpt}")"
        if _jdm_eval_stale_vs_ckpt "${eval_dir}/done.flag" "${ckpt}"; then
            log "ACTION: AMR exhausted — launching JDM ideal det eval on GPU${gpu_arg} ckpt=${ckpt} dir=${eval_dir}"
            mkdir -p "${eval_dir}"
            (
              CUDA_VISIBLE_DEVICES="${gpu_arg}" "${PY}" tools/test_det.py "${det_cfg}" "${ckpt}" \
                --work-dir "${eval_dir}" \
                >> "${eval_dir}/eval.log" 2>&1
              date -Is > "${eval_dir}/done.flag" || true
            ) &
            log "Launched JDM ideal det eval PID=$! gpu=${gpu_arg}"
            return 0
        fi
    done
    # 3) Improved ideal recipe (paper anchors 110/130/150) toward ~0.91.
    if [[ -f "${train_impr_cfg}" && ! -f "${train_impr_dir}/epoch_30.pth" ]]; then
        log "ACTION: AMR exhausted — launching JDM ideal v1 improved anchors train on GPU${gpu_arg}"
        mkdir -p "${train_impr_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_arg}" nohup "${PY}" tools/train.py "${train_impr_cfg}" \
            --work-dir "${train_impr_dir}" \
            >> "${jdm_dir}/det_ideal_v1_anchor110130150_30ep.log" 2>&1 &
        log "Launched JDM ideal improved train PID=$! gpu=${gpu_arg}"
        return 0
    fi
    # 4) Longer ideal 60ep @ lr5e4 (architecture freeze; next useful train after 30ep done).
    if [[ -f "${train_60_cfg}" && ! -f "${train_60_dir}/epoch_60.pth" ]]; then
        log "ACTION: AMR exhausted — launching JDM ideal v1 60ep lr5e4 train on GPU${gpu_arg}"
        mkdir -p "${train_60_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_arg}" nohup "${PY}" tools/train.py "${train_60_cfg}" \
            --work-dir "${train_60_dir}" \
            >> "${jdm_dir}/det_ideal_v1_60ep_lr5e4.log" 2>&1 &
        log "Launched JDM ideal 60ep train PID=$! gpu=${gpu_arg}"
        return 0
    fi
    # 5) Ideal joint measurement with best available det+AMC (even below target).
    # Prefer improved-anchors if available; else baseline ideal ep7 (mAP~0.385).
    # Skip known-failed 5ep AP75 FT and 5ep+AMC AWGN merge.
    local amc_ckpt="${jdm_dir}/amc_wave3b_detprops_30ep/best_accuracy_top1_epoch_23.pth"
    local best_impr_ckpt impr_eval best_base_ckpt
    best_impr_ckpt="$(_jdm_newest_best_ckpt "${train_impr_dir}")"
    best_base_ckpt="$(_jdm_newest_best_ckpt "${train_dir}")"
    if [[ -f "${amc_ckpt}" && -f "${joint_cfg}" ]]; then
        local det_for_joint joint_tag joint_out joint_eval_dir
        if [[ -n "${best_impr_ckpt}" ]]; then
            det_for_joint="${best_impr_ckpt}"
            joint_tag="impr"
        elif [[ -n "${best_base_ckpt}" ]]; then
            det_for_joint="${best_base_ckpt}"
            joint_tag="ep7_baseline"
        else
            det_for_joint=""
        fi
        if [[ -n "${det_for_joint}" ]]; then
            joint_out="${jdm_dir}/jdm_joint_ideal_${joint_tag}_amc.pth"
            joint_eval_dir="${jdm_dir}/eval_ideal_v1_joint_${joint_tag}"
            if [[ ! -f "${joint_out}" ]] || _jdm_eval_stale_vs_ckpt "${joint_eval_dir}/done.flag" "${joint_out}"; then
                log "ACTION: AMR exhausted — merge+joint ideal (${joint_tag}) measure on GPU${gpu_arg} det=${det_for_joint}"
                "${PY}" tools/merge_jdm_checkpoints.py "${det_for_joint}" "${amc_ckpt}" "${joint_out}" \
                    >> "${jdm_dir}/merge_ideal_${joint_tag}.log" 2>&1 || true
                if [[ -f "${joint_out}" ]]; then
                    mkdir -p "${joint_eval_dir}"
                    (
                      CUDA_VISIBLE_DEVICES="${gpu_arg}" "${PY}" tools/test_det.py "${joint_cfg}" \
                        "${joint_out}" \
                        --work-dir "${joint_eval_dir}" \
                        >> "${joint_eval_dir}/eval.log" 2>&1
                      date -Is > "${joint_eval_dir}/done.flag" || true
                    ) &
                    log "Launched JDM ideal ${joint_tag} joint eval PID=$! gpu=${gpu_arg}"
                    return 0
                fi
            fi
        fi
    fi
    # 6) Legacy joint ideal eval (wave3b mixed det+AMC that hit AWGN mAP 0.762).
    local joint_ckpt="${jdm_dir}/jdm_joint_wave3b_amc.pth"
    if [[ -f "${joint_ckpt}" && -f "${joint_cfg}" ]] \
        && _jdm_eval_stale_vs_ckpt "${jdm_dir}/eval_ideal_v1_joint/done.flag" "${joint_ckpt}"; then
        log "ACTION: AMR exhausted — launching JDM ideal v1 joint eval on GPU${gpu_arg}"
        mkdir -p "${jdm_dir}/eval_ideal_v1_joint"
        (
          CUDA_VISIBLE_DEVICES="${gpu_arg}" "${PY}" tools/test_det.py "${joint_cfg}" \
            "${joint_ckpt}" \
            --work-dir "${jdm_dir}/eval_ideal_v1_joint" \
            >> "${jdm_dir}/eval_ideal_v1_joint.log" 2>&1
          date -Is > "${jdm_dir}/eval_ideal_v1_joint/done.flag" || true
        ) &
        log "Launched JDM ideal v1 joint eval PID=$! gpu=${gpu_arg}"
        return 0
    fi
    # 7) AWGN Table-I det re-eval with best ideal ckpt (skip failed 5ep+AMC paths).
    local ideal_best awgn_dir
    ideal_best="$(_jdm_newest_best_ckpt "${train_dir}")"
    [[ -z "${ideal_best}" ]] && ideal_best="$(_jdm_newest_best_ckpt "${train_60_dir}")"
    if [[ -n "${ideal_best}" && -f "${awgn_det_cfg}" ]]; then
        awgn_dir="${jdm_dir}/eval_awgn_snr12_30_det_$(basename "$(dirname "${ideal_best}")")_$(basename "${ideal_best}" .pth)"
        if _jdm_eval_stale_vs_ckpt "${awgn_dir}/done.flag" "${ideal_best}"; then
            log "ACTION: AMR exhausted — launching AWGN det re-eval with ideal ckpt on GPU${gpu_arg} ckpt=${ideal_best}"
            mkdir -p "${awgn_dir}"
            (
              CUDA_VISIBLE_DEVICES="${gpu_arg}" "${PY}" tools/test_det.py "${awgn_det_cfg}" "${ideal_best}" \
                --work-dir "${awgn_dir}" \
                >> "${awgn_dir}/eval.log" 2>&1
              date -Is > "${awgn_dir}/done.flag" || true
            ) &
            log "Launched JDM AWGN-ideal det eval PID=$! gpu=${gpu_arg}"
            return 0
        fi
    fi
    # 8) Auto-seed next AMR near-miss wave so fallback never 14h-spins empty.
    local seeded
    seeded="$(auto_seed_next_near_miss_wave 2>/dev/null || true)"
    if [[ -n "${seeded}" && "${seeded}" != "all_waves_exhausted_no_next_seed" ]]; then
        log "ACTION: auto-seeded next near-miss wave: ${seeded} — dispatch on next tick"
        return 0
    fi
    if [[ "${seeded}" == "all_waves_exhausted_no_next_seed" ]]; then
        log_error "all_waves_exhausted_no_next_seed — no AMR wave manifests left to seed; JDM steps also done"
    else
        log_error "amr_queue_empty_no_jdm_fallback — no ideal train/eval/joint/AWGN work left"
    fi
    return 1
}

best_gpu1_backfill_target() {
    # Skip force-blocked pairs (exhausted/waived / ≥3 identical paper-exact fails)
    # so head-of-queue cgdnet-style stalls cannot starve mcnet@10B etc.
    "${PY}" - "${QUEUE}" "${QUEUE_FULL}" "${WAVE4_MANIFEST}" "${WAVE1_MANIFEST}" "${GOAL_STATUS}" \
        "${REPO}/docs/amr_benchmark/retune_results.md" <<'PY'
import json, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

repo = Path("/home/citybuster/Projects/ChangShuoRadioRecognition")
queue_path = Path(sys.argv[1])
queue_full = Path(sys.argv[2])
wave4_path = Path(sys.argv[3])
wave1_path = Path(sys.argv[4])
goal_path = Path(sys.argv[5])
results_md = Path(sys.argv[6])
queue_paths = [queue_path, queue_full]

def pair_is_force_blocked(model: str, dataset: str) -> bool:
    # Selector path: pair is blocked only when there is no pending/running entry
    # for this model/dataset. An older exhausted entry must not hide a newer
    # pending wave (wave6 FastMLDNN after siege_fastmldnn_10a exhausted).
    model = (model or "").lower()
    dataset = dataset or ""
    if not model or not dataset:
        return False
    saw_pair = False
    has_open = False
    has_waiver_only = True
    for qp in queue_paths:
        if not qp.is_file():
            continue
        data = json.loads(qp.read_text())
        for e in data.get("entries", []):
            if (e.get("model") or "").lower() != model or e.get("dataset") != dataset:
                continue
            saw_pair = True
            status = e.get("status", "pending")
            if status in {"pending", "running"}:
                has_open = True
                has_waiver_only = False
            elif not (e.get("waiver") or status.startswith("waived") or status in {"passed", "exhausted", "skipped"}):
                has_waiver_only = False
            if e.get("waiver") or status.startswith("waived"):
                pass
            else:
                if status not in {"passed", "exhausted", "skipped"}:
                    has_waiver_only = False
    if not saw_pair:
        return False
    if has_open:
        return False
    return True

active_gpu0 = set()
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_order = []
    pid_to_gpu = {}
    for ln in smi.splitlines():
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
    for pid, gpu in pid_to_gpu.items():
        if gpu != 0:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").replace("\x00", " ")
        except OSError:
            continue
        if "tools/train.py" not in cmd or str(repo) not in cmd:
            continue
        try:
            ppid = int(open(f"/proc/{pid}/status").read().split("PPid:")[1].split()[0])
            ppcmd = open(f"/proc/{ppid}/cmdline", "rb").read().decode(errors="replace")
        except (OSError, IndexError, ValueError):
            ppcmd = ""
        if "tools/train.py" in ppcmd:
            continue
        for part in cmd.split():
            if part.startswith(str(repo / "work_dirs/amr_benchmark_retune/")):
                rel = part[len(str(repo / "work_dirs/amr_benchmark_retune/")) :].strip("/")
                bits = rel.split("/")
                if len(bits) >= 2:
                    active_gpu0.add((bits[0].lower(), bits[1]))
except subprocess.CalledProcessError:
    pass

data = json.loads(queue_full.read_text())
skip = {"passed", "exhausted", "skipped", "running"}
pending = sorted(
    [
        e
        for e in data.get("entries", [])
        if e.get("status", "pending") not in skip
        and e.get("gap_pp") is not None
        and e.get("model")
        and e.get("dataset")
    ],
    key=lambda e: float(e.get("gap_pp", 999)),
)

for entry in pending:
    model = entry["model"].lower()
    dataset = entry["dataset"]
    if (model, dataset) in active_gpu0:
        continue
    if pair_is_force_blocked(model, dataset):
        continue
    manifest = entry.get("manifest")
    gap = entry.get("gap_pp")
    eid = entry.get("id", "")
    if manifest:
        print("\t".join(["siege_manifest", manifest, model, dataset, str(gap), eid]))
        raise SystemExit(0)
    print("\t".join(["synthesize", model, dataset, str(gap), eid]))
    raise SystemExit(0)

if wave1_path.is_file():
    exhausted = set()
    if goal_path.is_file():
        goal = json.loads(goal_path.read_text())
        for label in goal.get("exhausted_pairs", []):
            m, d = label.split("/", 1)
            exhausted.add((m.lower(), d))
    manifest = json.loads(wave1_path.read_text())
    remaining = [
        e
        for e in manifest.get("experiments", [])
        if (e.get("model", "").lower(), e.get("dataset")) not in exhausted
        and (e.get("model", "").lower(), e.get("dataset")) not in active_gpu0
    ]
    if remaining:
        print("\t".join(["wave1_goal", str(wave1_path.relative_to(repo))]))
        raise SystemExit(0)

raise SystemExit(1)
PY
}
launch_gpu1_amr_backfill() {
    # JDM P1 AMC / paper-exact owns GPU1 — do not steal for AMR.
    # Flag alone is insufficient: clear_stale_jdm_amc_flag drops it when idle;
    # still skip while any live JDM train/test/amc/sweep is present.
    local amc_flag
    amc_flag="$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('true' if d.get('jdm_amc_launched') else 'false')" 2>/dev/null || echo false)"
    if [[ "${amc_flag}" == "true" ]]; then
        log "GPU1 backfill skipped — jdm_amc_launched (JDM owns GPU1)"
        return 1
    fi
    if jdm_gpu1_live; then
        log "GPU1 backfill skipped — live JDM process owns GPU1"
        return 1
    fi
    # Avoid racing a manual/other exclusive GPU1 siege that has not yet registered train.py.
    if pgrep -af 'tools/amr_benchmark/retune_model_siege.py' 2>/dev/null | grep -E -- '--gpu[= ]1([[:space:]]|$)' >/dev/null; then
        log "GPU1 backfill skipped — exclusive --gpu 1 siege already running"
        return 1
    fi
    local target_line mode model dataset variant gap entry_id manifest log_name pid
    target_line="$(best_gpu1_backfill_target 2>>"${SCHED_LOG}" || true)"
    if [[ -z "${target_line}" ]]; then
        log "GPU1 backfill: no pending full-queue / wave1 target (force-blocked heads skipped)"
        return 1
    fi
    IFS=$'\t' read -r mode _rest <<< "${target_line}"
    case "${mode}" in
        synthesize)
            IFS=$'\t' read -r _mode model dataset gap entry_id <<< "${target_line}"
            # Belt-and-suspenders: selector already skips force-blocked; refuse if race.
            if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
                log "GPU1 backfill skip ${model}/${dataset} — force-blocked after select (race); no launch"
                return 1
            fi
            manifest="$(ensure_marginal_siege_manifest "${model}" "${dataset}" "${entry_id}" 2>>"${SCHED_LOG}" || true)"
            if [[ -z "${manifest}" || ! -f "${REPO}/${manifest}" ]]; then
                log_error "GPU1 backfill synthesize failed for ${model}/${dataset}"
                return 1
            fi
            log_name="${LOGDIR}/siege_${model}.log"
            log "ACTION: GPU1 AMR backfill — synthesized siege ${model}/${dataset} gap=${gap}pp entry=${entry_id} manifest=${manifest}"
            # First attempt: no --force (reuse cache). Cached-failed only gets --force
            # when pair_force_blocked is still no (fewer than 3 identical paper-exact fails).
            nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                --manifest "${REPO}/${manifest}" \
                --gpu 1 --max-parallel 1 \
                --until-pass --paper-exact --promote \
                >> "${log_name}" 2>&1 &
            pid=$!
            log "Launched GPU1 AMR backfill siege PID=${pid} log=${log_name}"
            ;;
        siege_manifest)
            IFS=$'\t' read -r _mode manifest model dataset gap entry_id <<< "${target_line}"
            if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
                log "GPU1 backfill skip ${model}/${dataset} — force-blocked after select (race); no launch"
                return 1
            fi
            log_name="${LOGDIR}/siege_${model}.log"
            log "ACTION: GPU1 AMR backfill — siege ${model}/${dataset} gap=${gap}pp entry=${entry_id} manifest=${manifest}"
            nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                --manifest "${REPO}/${manifest}" \
                --gpu 1 --max-parallel 1 \
                --until-pass --paper-exact --promote \
                >> "${log_name}" 2>&1 &
            pid=$!
            log "Launched GPU1 AMR backfill siege PID=${pid} log=${log_name}"
            ;;
        wave1_goal)
            IFS=$'\t' read -r _mode manifest <<< "${target_line}"
            log_name="${WAVE1_LOG}"
            log "ACTION: GPU1 AMR backfill — wave1 goal-mode sweep manifest=${manifest}"
            nohup "${PY}" tools/amr_benchmark/retune_sweep.py \
                --manifest "${REPO}/${manifest}" \
                --gpu 1 --max-parallel 1 \
                --goal-mode --stop-when-all-pass --paper-exact \
                >> "${log_name}" 2>&1 &
            pid=$!
            log "Launched GPU1 AMR backfill wave1 PID=${pid} log=${log_name}"
            ;;
        *)
            log_error "GPU1 backfill unknown mode: ${mode}"
            return 1
            ;;
    esac
    write_state "$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_backfill_last_pid']=${pid}; d['gpu1_backfill_last_launch']='$(date -u +%s)'; print(json.dumps(d))")"
    return 0
}

read_state() {
    if [[ -f "${STATE_FILE}" ]]; then
        cat "${STATE_FILE}"
    else
        echo '{"gpu1_idle_since": null, "jdm_trackb_launched": false, "jdm_amc_launched": false}'
    fi
}

write_state() {
    echo "$1" > "${STATE_FILE}"
}

tick() {
    local counts amr0 amr1 jdm0 jdm1
    reset_false_exhausted_queue
    # Close phantom pending before any dispatch decision (stall class).
    local _reconciled
    _reconciled="$(reconcile_phantom_pending_full 2>/dev/null || true)"
    if [[ -n "${_reconciled}" ]]; then
        log "ACTION: reconcile_phantom_pending ${_reconciled}"
    fi
    clear_stale_jdm_amc_flag
    counts="$(count_trains_per_gpu)"
    read -r amr0 amr1 jdm0 jdm1 <<< "${counts}"

    local siege_orchestrator amr_sweep jdm_sweep
    siege_orchestrator=false
    amr_sweep=false
    jdm_sweep=false
    orchestrator_running "tools/amr_benchmark/retune_model_siege.py" && siege_orchestrator=true
    orchestrator_running "tools/amr_benchmark/retune_sweep.py" && amr_sweep=true
    orchestrator_running "tools/jdm/retune_sweep.py" && jdm_sweep=true

    log "tick amr_gpu0=${amr0} amr_gpu1=${amr1} jdm_gpu0=${jdm0} jdm_gpu1=${jdm1} siege_orch=${siege_orchestrator} amr_sweep=${amr_sweep} jdm_sweep=${jdm_sweep}"

    local state gpu1_idle_since now launched_jdm
    state="$(read_state)"
    now="$(date -u +%s)"
    launched_jdm="$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('true' if d.get('jdm_trackb_launched') else 'false')")"

    # Track GPU1 AMR idle duration for JDM secondary slot.
    if [[ "${amr1}" -eq 0 ]]; then
        gpu1_idle_since="$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('gpu1_idle_since') or '')")"
        if [[ -z "${gpu1_idle_since}" ]]; then
            gpu1_idle_since="${now}"
            write_state "$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_idle_since']='${now}'; print(json.dumps(d))")"
            log "GPU1 AMR idle timer started"
        fi
    else
        write_state "$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['gpu1_idle_since']=None; print(json.dumps(d))")"
        gpu1_idle_since=""
    fi

    # --- Launch decisions (never kill running jobs) ---
    # Flag location: work_dirs/amr_benchmark_retune/scheduler_state.json → jdm_amc_launched
    # Auto-cleared by clear_stale_jdm_amc_flag when no live train/test_det/AMC python remains.

    # Both GPUs free, siege r3 still has variants → resume r3.
    if [[ "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${siege_orchestrator}" == "false" ]]; then
        if [[ "$(siege_r3_incomplete)" == "yes" ]]; then
            log "ACTION: both GPUs idle, siege r3 incomplete — launching r3 manifest"
            nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                --manifest "${SIEGE_R3_MANIFEST}" \
                --gpu 0,1 --max-parallel 2 --until-pass --paper-exact --promote \
                >> "${LOGDIR}/siege_r3.log" 2>&1 &
            log "Launched siege r3 orchestrator PID=$!"
            return
        fi
    fi

    # Deadlock breaker: primary siege_queue still has pending (e.g. Tier-A HCGDNN)
    # while another siege/train already owns GPU0 (often Tier-B ResNetAMR).
    # Old logic required siege_orch=false OR pending_n==0 → GPU1 idled forever.
    # Prefer dispatching the next primary-queue entry onto free GPU1.
    local jdm1_own=0
    if jdm_gpu1_live \
        || [[ "$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('1' if d.get('jdm_amc_launched') else '0')" 2>/dev/null || echo 0)" == "1" ]]; then
        jdm1_own=1
    fi
    if [[ "${jdm1}" -eq 0 && "${jdm1_own}" -eq 1 ]]; then
        jdm1=1
    fi
    if [[ "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${jdm1}" -eq 0 && "${amr_sweep}" == "false" ]]; then
        local entry_id manifest model dataset pending_line pending_n
        pending_n="$(count_pending_entries)"
        pending_line="$(next_pending_entry "${QUEUE}" 2>/dev/null || true)"
        if [[ "${pending_n}" -gt 0 && -n "${pending_line}" ]]; then
            IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
            if [[ -n "${entry_id}" && -n "${manifest}" ]]; then
                # Avoid stacking a second exclusive GPU1 siege.
                if ! pgrep -af 'tools/amr_benchmark/retune_model_siege.py' 2>/dev/null | grep -E -- '--gpu[= ]1([[:space:]]|$)' >/dev/null; then
                    log "ACTION: GPU0 busy + primary pending=${pending_n} — Tier-A/queue ${model}/${dataset} (${entry_id}) on GPU1 (break Tier-B deadlock)"
                    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                        --queue "${QUEUE}" \
                        --gpu 1 --max-parallel 1 \
                        --until-pass --paper-exact --promote \
                        >> "${LOGDIR}/siege_${model}.log" 2>&1 &
                    log "Launched siege queue orchestrator PID=$! entry=${entry_id} manifest=${manifest} gpu=1"
                    STREAK_IDLE=0
                    return
                fi
            fi
        fi
    fi

    # AMR primary idle slot(s) and no siege orchestrator → advance siege queue.
    if [[ "${siege_orchestrator}" == "false" && ( "${amr0}" -lt 1 || "${amr1}" -lt 1 ) ]]; then
        local entry_id manifest model dataset pending_line pending_n force_flag=""
        pending_n="$(count_pending_entries)"
        pending_line="$(next_pending_entry "${QUEUE}" 2>/dev/null || true)"
        if [[ -n "${pending_line}" ]]; then
            IFS=$'\t' read -r entry_id manifest model dataset <<< "${pending_line}"
            if [[ -n "${entry_id}" && -n "${manifest}" ]]; then
                local gpu_arg max_par log_suffix
                if [[ "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${jdm1}" -eq 0 ]]; then
                    gpu_arg="1"
                    max_par=1
                    log_suffix="gpu1_parallel"
                    log "ACTION: GPU0 busy, GPU1 free — single-pair siege ${model}/${dataset} on GPU1 manifest=${manifest}"
                elif [[ "${amr0}" -eq 0 && ( "${amr1}" -ge 1 || "${jdm1}" -eq 1 ) ]]; then
                    gpu_arg="0"
                    max_par=1
                    log_suffix="gpu0_parallel"
                    log "ACTION: GPU1 busy (AMR or JDM), GPU0 free — single-pair siege ${model}/${dataset} on GPU0 manifest=${manifest}"
                elif [[ "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${jdm1}" -eq 0 ]]; then
                    gpu_arg="0,1"
                    max_par=2
                    log_suffix="full"
                    log "ACTION: launching siege queue continuation (${entry_id}) manifest=${manifest} on both GPUs"
                else
                    log "ACTION skipped — no free AMR GPU for ${entry_id} (amr0=${amr0} amr1=${amr1} jdm1=${jdm1})"
                    return
                fi
                if [[ "${STREAK_IDLE}" -ge "${STREAK_IDLE_MAX}" && "${pending_n}" -gt 0 ]]; then
                    if [[ "$(pair_force_blocked "${model}" "${dataset}" 2>/dev/null || echo no)" == "yes" ]]; then
                        log_error "STREAK_IDLE=${STREAK_IDLE} but ${model}/${dataset} force-blocked — launching without --force"
                    else
                        force_flag="--force"
                        log_error "STREAK_IDLE=${STREAK_IDLE} with pending=${pending_n} — mandatory force dispatch"
                    fi
                fi
                nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
                    --queue "${QUEUE}" \
                    --gpu "${gpu_arg}" --max-parallel "${max_par}" \
                    --until-pass --paper-exact --promote \
                    ${force_flag} \
                    >> "${LOGDIR}/siege_${model}.log" 2>&1 &
                log "Launched siege queue orchestrator PID=$! entry=${entry_id} manifest=${manifest} force=${force_flag:+yes}"
                STREAK_IDLE=0
                return
            fi
        else
            # Primary queue exhausted — wave1 / wave4 / siege_queue_full on free GPU(s).
            if [[ "${amr_sweep}" == "false" ]]; then
                if launch_post_siege_work "" "${amr0}" "${amr1}"; then
                    return
                fi
            fi
        fi
    fi

    # Siege orchestrator owns one GPU but the other is empty — still fill the free slot.
    # Prior bug: siege_orch=true blocked the entire primary-idle block → GPU0 idle for
    # hours while FastMLDNN siege ran exclusively on GPU1.
    if [[ "${siege_orchestrator}" == "true" && "${amr_sweep}" == "false" ]]; then
        if [[ "${amr0}" -eq 0 && "${jdm0}" -eq 0 && "${amr1}" -ge 1 ]]; then
            log "ACTION: siege_orch on GPU1, GPU0 idle — JDM ideal/AWGN fallback on GPU0"
            if launch_jdm_ideal_fallback 0 1; then
                STREAK_IDLE=0
                return
            fi
            if [[ "$(count_pending_entries_full)" -gt 0 ]]; then
                if launch_full_queue_marginal_siege "" 0 1; then
                    STREAK_IDLE=0
                    return
                fi
            fi
        elif [[ "${amr1}" -eq 0 && "${jdm1}" -eq 0 && "${amr0}" -ge 1 ]]; then
            log "ACTION: siege_orch on GPU0, GPU1 idle — JDM ideal fallback on GPU1"
            if launch_jdm_ideal_fallback 1 0; then
                STREAK_IDLE=0
                return
            fi
        fi
    fi

    # Primary siege_queue pending=0 + GPU1 idle → fill from siege_queue_full / wave1
    # remaining (including while another orchestrator owns GPU0). Do not wait 10 min.
    # When full_pending=0 AND waves exhausted, MUST fall through to JDM ideal
    # (stall class amr_queue_empty_no_jdm_fallback — GPU1 idle ~94min 2026-07-18).
    local pending_n pending_full_n
    pending_n="$(count_pending_entries)"
    pending_full_n="$(count_pending_entries_full)"
    if [[ "${pending_n}" -eq 0 && "${amr1}" -eq 0 && "${jdm1}" -eq 0 && "${amr_sweep}" == "false" ]]; then
        local wave1_done wave4_done need_amr_fill=0
        wave1_done="$(wave1_goal_exhausted 2>/dev/null || echo no)"
        wave4_done="$(wave4_marginal_exhausted 2>/dev/null || echo no)"
        if [[ "${pending_full_n}" -gt 0 || "${wave1_done}" != "yes" || "${wave4_done}" != "yes" ]]; then
            need_amr_fill=1
        fi
        if [[ "${need_amr_fill}" -eq 1 ]]; then
            if [[ "${amr0}" -eq 0 && "${siege_orchestrator}" == "false" ]]; then
                log "ACTION: siege_queue pending=0, GPUs idle — dispatching post-siege / full-queue work"
                if launch_post_siege_work "" "${amr0}" "${amr1}"; then
                    STREAK_IDLE=0
                    return
                fi
            elif [[ "${amr0}" -ge 1 ]]; then
                log "ACTION: siege_queue pending=0, GPU0 busy / GPU1 idle — immediate AMR backfill (full_pending=${pending_full_n})"
                if launch_gpu1_amr_backfill; then
                    STREAK_IDLE=0
                    return
                fi
                # AMR has nothing launchable for GPU1 — keep it busy with JDM ideal.
                if launch_jdm_ideal_fallback "${amr0}" 0; then
                    STREAK_IDLE=0
                    return
                fi
                log "GPU1 backfill deferred (no target or exclusive GPU1 siege already running); full_pending=${pending_full_n}"
            fi
        elif [[ "${amr0}" -ge 1 ]]; then
            # full_pending=0 and waves exhausted — immediate JDM ideal (do not wait 10 min / AMC skip).
            log "ACTION: full_pending=0 waves exhausted, GPU0 busy / GPU1 idle — JDM ideal fallback"
            if launch_jdm_ideal_fallback "${amr0}" 0; then
                STREAK_IDLE=0
                return
            fi
        elif [[ "${amr0}" -eq 0 && "${siege_orchestrator}" == "false" ]]; then
            log "ACTION: all AMR queues empty, GPUs idle — JDM ideal fallback"
            if launch_jdm_ideal_fallback 0 0; then
                STREAK_IDLE=0
                return
            fi
        fi
    fi

    # GPU1 AMR backfill: GPU0 busy, GPU1 idle >5 min (including during siege orchestrator).
    # Skip when JDM already owns GPU1, or when P1 AMC has been reserved/launched
    # (Track B complete → AMC owns the secondary slot; do not steal for AMR).
    local launched_amc
    launched_amc="$(echo "${state}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('true' if d.get('jdm_amc_launched') else 'false')")"
    if [[ "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${jdm1}" -eq 0 && "${jdm_sweep}" == "false" && "${amr_sweep}" == "false" && "${launched_amc}" == "false" ]]; then
        gpu1_idle_since="$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); v=d.get('gpu1_idle_since'); print(v if v else '')")"
        if [[ -n "${gpu1_idle_since}" ]]; then
            local idle_sec=$((now - gpu1_idle_since))
            if [[ "${idle_sec}" -ge "${GPU1_BACKFILL_THRESHOLD}" ]]; then
                if launch_gpu1_amr_backfill; then
                    STREAK_IDLE=0
                    return
                fi
                # Threshold hit + nothing AMR launchable → JDM ideal (harden 5min guarantee).
                if [[ "${pending_full_n}" -eq 0 ]]; then
                    log "ACTION: GPU1 idle ${idle_sec}s with full_pending=0 — JDM ideal fallback"
                    if launch_jdm_ideal_fallback "${amr0}" 0; then
                        STREAK_IDLE=0
                        return
                    fi
                fi
            fi
        fi
    fi

    # JDM secondary on GPU1: staged Track B → P1 AMC. Flag jdm_trackb_launched alone
    # must NOT block AMC — watchdog advances to wave_p1_amc_manifest after Track B.
    # Only when AMR backfill has nothing left (or AMR trains occupy GPU1).
    if [[ "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${jdm0}" -eq 0 && "${jdm1}" -eq 0 && "${jdm_sweep}" == "false" && "${amr_sweep}" == "false" ]]; then
        gpu1_idle_since="$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); v=d.get('gpu1_idle_since'); print(v if v else '')")"
        if [[ -n "${gpu1_idle_since}" ]]; then
            local idle_sec=$((now - gpu1_idle_since))
            if [[ "${idle_sec}" -ge "${JDM_IDLE_THRESHOLD}" ]]; then
                if [[ "${launched_jdm}" == "false" && -f "${JDM_TRACKB_MANIFEST}" ]]; then
                    log "ACTION: GPU1 AMR idle ${idle_sec}s (>${JDM_IDLE_THRESHOLD}) — launching JDM wave3 Track B on GPU1"
                    nohup "${PY}" tools/jdm/retune_sweep.py \
                        --manifest "${JDM_TRACKB_MANIFEST}" \
                        --goal-mode --gpu 1 --max-parallel 1 \
                        >> "${REPO}/work_dirs/jdm/retune/wave3_trackb.log" 2>&1 &
                    log "Launched JDM wave3 Track B PID=$!"
                    write_state "$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['jdm_trackb_launched']=True; print(json.dumps(d))")"
                    return
                elif [[ "${launched_jdm}" == "true" && "${launched_amc}" == "false" && -f "${JDM_AMC_MANIFEST}" ]]; then
                    # Do not re-arm jdm_amc_launched / re-run AMC when P1 already produced a best ckpt
                    # (clearing the flag after eval previously caused infinite AMC relaunch → GPU1 thrash).
                    if compgen -G "${REPO}/work_dirs/jdm/retune/amc_wave3b_detprops_30ep/best_accuracy_*.pth" > /dev/null; then
                        log "SKIP JDM P1 AMC relaunch — amc_wave3b best ckpt already exists; fall through to ideal"
                        write_state "$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['jdm_amc_launched']=False; d['jdm_amc_complete']=True; print(json.dumps(d))")"
                        if launch_jdm_ideal_fallback "${amr0}" 0; then
                            STREAK_IDLE=0
                            return
                        fi
                    else
                        log "ACTION: GPU1 AMR idle ${idle_sec}s — Track B done; launching JDM P1 AMC on GPU1"
                        nohup bash "${REPO}/tools/jdm/launch_wave_p1_amc.sh" \
                            >> "${REPO}/work_dirs/jdm/retune/wave_p1_amc.log" 2>&1 &
                        log "Launched JDM P1 AMC wrapper PID=$!"
                        write_state "$(read_state | "${PY}" -c "import json,sys; d=json.load(sys.stdin); d['jdm_amc_launched']=True; print(json.dumps(d))")"
                        return
                    fi
                elif [[ "${launched_jdm}" == "true" ]]; then
                    # Track B + AMC done path — still fill GPU1 with ideal when idle.
                    if launch_jdm_ideal_fallback "${amr0}" 0; then
                        STREAK_IDLE=0
                        return
                    fi
                fi
            fi
        fi
    fi

    pending_n="$(count_pending_entries)"
    pending_full_n="$(count_pending_entries_full)"
    if [[ "${amr0}" -eq 0 && "${amr1}" -eq 0 && "${siege_orchestrator}" == "false" && "${amr_sweep}" == "false" ]]; then
        if [[ "${pending_n}" -gt 0 ]]; then
            STREAK_IDLE=$((STREAK_IDLE + 1))
            log "no action required (STREAK_IDLE=${STREAK_IDLE}/${STREAK_IDLE_MAX} pending=${pending_n})"
            if [[ "${STREAK_IDLE}" -ge "${STREAK_IDLE_MAX}" ]]; then
                log_error "STREAK_IDLE=${STREAK_IDLE} — GPUs idle with pending queue; escalating on next tick"
            fi
        elif [[ "${pending_full_n}" -gt 0 || "$(wave1_goal_exhausted 2>/dev/null || echo no)" != "yes" || "$(wave4_marginal_exhausted 2>/dev/null || echo no)" != "yes" ]]; then
            STREAK_IDLE=$((STREAK_IDLE + 1))
            log "IDLE WITH WORK remaining (STREAK_IDLE=${STREAK_IDLE}/${STREAK_IDLE_MAX} full_pending=${pending_full_n}) — force-dispatch next tick"
            if [[ "${STREAK_IDLE}" -ge "${STREAK_IDLE_MAX}" ]]; then
                log_error "STREAK_IDLE=${STREAK_IDLE} — siege_queue exhausted, GPUs idle with full_pending=${pending_full_n}; force post-siege"
                if launch_post_siege_work "force" "${amr0}" "${amr1}"; then
                    STREAK_IDLE=0
                    return
                fi
            fi
        else
            # All AMR queues exhausted — still try JDM ideal before declaring idle.
            if launch_jdm_ideal_fallback 0 0; then
                STREAK_IDLE=0
                return
            fi
            STREAK_IDLE=0
            log "no action required (all queues + jdm ideal exhausted)"
        fi
    elif [[ "${amr0}" -ge 1 && "${amr1}" -eq 0 && "${pending_n}" -eq 0 && "${pending_full_n}" -gt 0 ]]; then
        log "GPU1 idle with full_pending=${pending_full_n} (siege_orch=${siege_orchestrator}); waiting for next backfill opportunity"
    else
        STREAK_IDLE=0
        log "no action required (gpus busy or waiting)"
    fi
}

# One-shot mode for watchdog: generate+seed next wave without starting the daemon.
if [[ "${GPU_KEEPALIVE_AUTO_SEED_ONCE:-}" == "1" ]]; then
    auto_seed_next_near_miss_wave
    exit $?
fi

log "gpu_keepalive daemon started PID=$$ interval=${INTERVAL}s gpu1_backfill_threshold=${GPU1_BACKFILL_THRESHOLD}s jdm_idle_threshold=${JDM_IDLE_THRESHOLD}s streak_idle_max=${STREAK_IDLE_MAX}"
startup_self_test || true
tick
while true; do
    sleep "${INTERVAL}"
    tick
done
