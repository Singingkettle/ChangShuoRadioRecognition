#!/usr/bin/env bash
set -u

REPO="${REPO:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
CONDA="${CONDA:-/home/citybuster/Applications/conda/bin/conda}"
ENV_NAME="${ENV_NAME:-ChangShuoRadioRecognition}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-/home/citybuster/Data/RCPS/work_dirs/logs}"

mkdir -p "$LOG_DIR"

while true; do
  (
    cd "$REPO" || exit 1
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
      python tools/rcps/monitor_experiments.py
  ) >> "$LOG_DIR/rcps_monitor_loop.log" 2>&1
  sleep "$INTERVAL_SECONDS"
done
