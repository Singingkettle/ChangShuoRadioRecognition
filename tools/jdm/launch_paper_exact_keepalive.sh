#!/usr/bin/env bash
# Keep GPU1 busy for paper-exact campaign.
# Prefer AP75 FT from 5-ep baseline (AP75 0.9182) over wave3b path (failed).
#
# Stall hardening (2026-07-15):
# - Never treat this script / waiters as the train job (pgrep self-match).
# - Require tools/train.py in cmdline; optional PID file.
# - Wait loops have max-wait + heartbeat; exit if target never existed.
set -uo pipefail
REPO=/home/citybuster/Projects/ChangShuoRadioRecognition
PY=/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LOG=work_dirs/jdm/retune/paper_exact_keepalive.log
PID_DIR=work_dirs/jdm/retune
FT_PID_FILE="${PID_DIR}/det_paper_exact_ap75_ft_from_5ep_baseline.train.pid"
MAX_WAIT_SEC="${PAPER_EXACT_MAX_WAIT_SEC:-7200}"  # 2h hard cap on wait loops
HEARTBEAT_SEC=60
mkdir -p work_dirs/jdm/retune
exec >>"$LOG" 2>&1

echo "=== keepalive start $(date -Is) PID=$$ ==="

# True if a real train.py job matches pattern (never match this script's cmdline).
train_running() {
  local needle="$1"
  local pid cmdline
  # Prefer PID file when present and process still alive.
  if [[ -f "${FT_PID_FILE}" ]]; then
    pid="$(cat "${FT_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" && -d "/proc/${pid}" ]]; then
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      if [[ "${cmdline}" == *"tools/train.py"* && "${cmdline}" == *"${needle}"* ]]; then
        return 0
      fi
    fi
  fi
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    # Exclude bash/shell waiters and pgrep noise.
    [[ "${cmdline}" == *bash* || "${cmdline}" == */bin/sh* || "${cmdline}" == *pgrep* ]] && continue
    [[ "${cmdline}" == *"launch_paper_exact_keepalive"* ]] && continue
    [[ "${cmdline}" == *"tools/train.py"* && "${cmdline}" == *"${needle}"* ]] || continue
    echo "${pid}" > "${FT_PID_FILE}" 2>/dev/null || true
    return 0
  done < <(pgrep -f "tools/train.py.*${needle}" 2>/dev/null || true)
  return 1
}

# Wait until train exits, or MAX_WAIT_SEC. If never saw a live train, exit early
# (do not spin for hours matching self / zombies).
wait_for_train() {
  local needle="$1"
  local label="${2:-train}"
  local waited=0
  local saw_live=0
  while true; do
    if train_running "${needle}"; then
      saw_live=1
      echo "$(date -Is) waiting ${label} (elapsed=${waited}s)..."
      sleep "${HEARTBEAT_SEC}"
      waited=$((waited + HEARTBEAT_SEC))
    else
      if [[ "${saw_live}" -eq 0 ]]; then
        echo "$(date -Is) ERROR: ${label} not running (never observed live tools/train.py); exit wait"
        return 1
      fi
      echo "$(date -Is) ${label} finished after ${waited}s"
      rm -f "${FT_PID_FILE}" 2>/dev/null || true
      return 0
    fi
    if [[ "${waited}" -ge "${MAX_WAIT_SEC}" ]]; then
      echo "$(date -Is) ERROR: ${label} wait exceeded MAX_WAIT_SEC=${MAX_WAIT_SEC}; aborting waiter"
      return 2
    fi
  done
}

# Do not continue failed wave3b AP75 FT if somehow still alive.
if train_running 'det_paper_exact_ap75_ft_from_wave3b'; then
  echo "$(date -Is) killing residual wave3b AP75 FT"
  pkill -f 'tools/train.py.*det_paper_exact_ap75_ft_from_wave3b' || true
  sleep 2
fi

BASE_DONE=0
shopt -s nullglob
for f in work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_5ep_baseline/best_detection_*.pth; do
  BASE_DONE=1
done

if [[ $BASE_DONE -eq 0 ]]; then
  if ! train_running 'det_paper_exact_ap75_ft_from_5ep_baseline'; then
    echo "$(date -Is) launch AP75 FT from 5-ep baseline"
    $PY tools/train.py \
      configs/jdm/experiments/retune/det_paper_exact_ap75_ft_from_5ep_baseline.py \
      --work-dir work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_5ep_baseline &
    echo $! > "${FT_PID_FILE}"
    wait_for_train 'det_paper_exact_ap75_ft_from_5ep_baseline' '5ep baseline AP75 FT' || true
  else
    echo "$(date -Is) waiting for existing 5-ep baseline AP75 FT..."
    wait_for_train 'det_paper_exact_ap75_ft_from_5ep_baseline' '5ep baseline AP75 FT' || true
  fi
else
  echo "$(date -Is) 5-ep baseline FT already has best ckpt; skip train"
  rm -f "${FT_PID_FILE}" 2>/dev/null || true
fi

echo "$(date -Is) running goal-status"
$PY tools/jdm/retune_sweep.py --goal-status || true

# Prefer newest 5-ep FT best, else production 5-ep baseline, else wave3b.
DET_CKPT=work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth
for cand in \
  work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_5ep_baseline/best_detection_AP75_*.pth \
  work_dirs/jdm/retune/det_paper_exact_ap75_ft_from_5ep_baseline/best_detection_mAP_*.pth
do
  DET_CKPT=$cand
done
echo "DET_CKPT=$DET_CKPT"

if [[ ! -f work_dirs/jdm/retune/eval_awgn_snr12_30_det/done.flag ]]; then
  echo "$(date -Is) AWGN SNR12-30 detector eval (v89-v98)"
  $PY tools/test_det.py \
    configs/jdm/experiments/retune/eval_awgn_snr12_30_det.py \
    "$DET_CKPT" \
    --work-dir work_dirs/jdm/retune/eval_awgn_snr12_30_det || true
  date -Is > work_dirs/jdm/retune/eval_awgn_snr12_30_det/done.flag || true
fi

if [[ -f work_dirs/jdm/retune/jdm_joint_wave3b_amc.pth ]] \
  && [[ ! -f work_dirs/jdm/retune/eval_awgn_snr12_30_joint/done.flag ]]; then
  echo "$(date -Is) AWGN SNR12-30 joint eval (v89-v98)"
  $PY tools/test_det.py \
    configs/jdm/experiments/retune/eval_awgn_snr12_30_joint.py \
    work_dirs/jdm/retune/jdm_joint_wave3b_amc.pth \
    --work-dir work_dirs/jdm/retune/eval_awgn_snr12_30_joint || true
  date -Is > work_dirs/jdm/retune/eval_awgn_snr12_30_joint/done.flag || true
fi

$PY tools/jdm/retune_sweep.py --goal-status || true
echo "=== keepalive done $(date -Is) ==="
