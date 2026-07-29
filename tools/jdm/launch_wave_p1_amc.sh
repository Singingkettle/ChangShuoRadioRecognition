#!/usr/bin/env bash
# JDM P1 AMC on GPU1: Track B proposals → 30-ep proposal-crop fine-tune.
# Does NOT touch GPU0 (AMR / icamcnet). Intended log: work_dirs/jdm/retune/wave_p1_amc.log
set -euo pipefail

REPO="/home/citybuster/Projects/ChangShuoRadioRecognition"
PY="/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python"
cd "${REPO}"

DET_CFG="configs/jdm/experiments/retune/det_wave3b_5ep_lr1e3.py"
DET_CKPT="work_dirs/jdm/retune/det_wave3b_5ep_lr1e3/best_detection_mAP_epoch_5.pth"
PROP_OUT="work_dirs/jdm/amc_proposals/wave3b_5ep_lr1e3.json"
MANIFEST="configs/jdm/experiments/retune/wave_p1_amc_manifest.json"

mkdir -p work_dirs/jdm/amc_proposals work_dirs/jdm/retune

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === wave_p1_amc start (GPU1 only) ==="
if [[ ! -f "${PROP_OUT}" ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Precomputing Track B proposals → ${PROP_OUT}"
  CUDA_VISIBLE_DEVICES=1 "${PY}" tools/precompute_amc_proposals.py \
    "${DET_CFG}" "${DET_CKPT}" \
    --out "${PROP_OUT}" --device cuda:0
else
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Reusing existing proposals ${PROP_OUT}"
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting AMC retune_sweep (manifest=${MANIFEST})"
"${PY}" tools/jdm/retune_sweep.py \
  --manifest "${MANIFEST}" \
  --goal-mode --gpu 1 --max-parallel 1
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] === wave_p1_amc done ==="
