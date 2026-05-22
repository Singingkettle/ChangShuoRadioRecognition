#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
DATA_ROOT="${DATA_ROOT:-/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1}"
WORK_ROOT="${WORK_ROOT:-/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/citybuster/Applications/conda/bin/activate}"
ENV_NAME="${ENV_NAME:-ChangShuoRadioRecognition}"

cd "$REPO_ROOT"
source "$CONDA_ACTIVATE" "$ENV_NAME"

CONFIG="configs/rcps/synthetic_awgn/petcgdnn_hard-ce_iq-snr-synthetic-awgn-v1.py"
SMOKE_WORK="$WORK_ROOT/smoke_petcgdnn_hard_ce"

python tools/train.py "$CONFIG" \
  --work-dir "$SMOKE_WORK" \
  --no-persistent-workers \
  --cfg-options \
    train_cfg.max_epochs=1 \
    train_dataloader.dataset.data_root="$DATA_ROOT" \
    val_dataloader.dataset.data_root="$DATA_ROOT" \
    test_dataloader.dataset.data_root="$DATA_ROOT" \
    train_dataloader.num_workers=0 \
    val_dataloader.num_workers=0 \
    test_dataloader.num_workers=0

echo "Smoke training finished in $SMOKE_WORK"
