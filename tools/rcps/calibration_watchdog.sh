#!/usr/bin/env bash
set -u

REPO="$HOME/Projects/ChangShuoRadioRecognition"
CONDA="/home/citybuster/Applications/conda/bin/conda"
ENV_NAME="ChangShuoRadioRecognition"
WORK="/home/citybuster/Data/RCPS/work_dirs"
CAL="$WORK/calibration_10ep"
CONF="$WORK/calibration_10ep_confusion"
LOGDIR="$WORK/logs"
DATASET="deepsig201610A"
ANN="/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A/train.json"
PRIOR="$WORK/priors/deepsig201610A.npy"
CONFBASE="$WORK/confusion_bases/deepsig201610A.npy"

mkdir -p "$LOGDIR" "$CAL" "$CONF"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_py() {
  "$CONDA" run --no-capture-output -n "$ENV_NAME" python "$@"
}

count_csv() {
  find "$1/metrics/calibration" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l
}

wait_for_grid() {
  local root="$1"
  local expected="$2"
  local label="$3"
  log "Waiting for $label grids under $root"
  while pgrep -af "run_calibration_grid.py .*--work-root $root" | grep -v grep >/dev/null; do
    log "$label still running; metrics_csv=$(count_csv "$root")"
    sleep 300
  done

  local csv_count
  csv_count=$(count_csv "$root")
  log "$label finished process check; metrics_csv=$csv_count expected_min=$expected"
  if [ "$csv_count" -lt "$expected" ]; then
    log "ERROR: $label produced too few metrics CSV files."
    return 1
  fi
}

check_logs_clean() {
  local pattern="$1"
  if grep -E "Traceback|CalledProcessError|ERROR conda" $pattern >/dev/null 2>&1; then
    log "ERROR: failure pattern found in logs: $pattern"
    grep -n -E "Traceback|CalledProcessError|ERROR conda" $pattern | tail -20
    return 1
  fi
}

select_hard_teacher() {
  run_py - <<'PY'
import csv
import glob
import os
from pathlib import Path

metric_paths = sorted(glob.glob('/home/citybuster/Data/RCPS/work_dirs/calibration_10ep/metrics/calibration/*_hard-ce_hard_seed2026_validation.csv'))
best = None
for path in metric_paths:
    with open(path, newline='', encoding='utf-8') as f:
        all_row = next(row for row in csv.DictReader(f) if row['reliability_bin'] == 'all')
    nll = float(all_row['nll'])
    if best is None or nll < best[0]:
        best = (nll, path)
if best is None:
    raise SystemExit('No hard-ce teacher metrics found.')
name = os.path.basename(best[1])
rest = name[len('deepsig201610A_'):-len('_hard-ce_hard_seed2026_validation.csv')]
pred = Path('/home/citybuster/Data/RCPS/work_dirs/calibration_10ep/amc/deepsig201610A') / f'{rest}_hard-ce_hard' / 'seed_2026' / 'predictions' / 'validation.pkl'
if not pred.exists():
    raise SystemExit(f'Teacher prediction file missing: {pred}')
print(pred)
PY
}

launch_confusion_grid() {
  if pgrep -af "run_calibration_grid.py .*--work-root $CONF" | grep -v grep >/dev/null; then
    log "Confusion grids already running; not launching duplicates."
    return 0
  fi

  log "Launching RCPS-Confusion grids."
  (
    cd "$REPO" || exit 1
    nohup setsid bash -lc "CUDA_VISIBLE_DEVICES=0 $CONDA run --no-capture-output -n $ENV_NAME python tools/rcps/run_calibration_grid.py --models cgdnet mcformer --methods rcps-confusion --seeds 2026 --dataset $DATASET --work-root $CONF --max-epochs 10 --epsilon-max 0.3 0.5 0.7 1.0 --epsilon-gamma 0.5 1.0 2.0 --num-workers 0 --collect-splits validation --analyze --execute" \
      > "$LOGDIR/calibration_10ep_confusion_gpu0_cgdnet_mcformer.log" 2>&1 < /dev/null &
    echo $! > "$LOGDIR/calibration_10ep_confusion_gpu0.pid"
    nohup setsid bash -lc "CUDA_VISIBLE_DEVICES=1 $CONDA run --no-capture-output -n $ENV_NAME python tools/rcps/run_calibration_grid.py --models mcldnn petcgdnn --methods rcps-confusion --seeds 2026 --dataset $DATASET --work-root $CONF --max-epochs 10 --epsilon-max 0.3 0.5 0.7 1.0 --epsilon-gamma 0.5 1.0 2.0 --num-workers 0 --collect-splits validation --analyze --execute" \
      > "$LOGDIR/calibration_10ep_confusion_gpu1_mcldnn_petcgdnn.log" 2>&1 < /dev/null &
    echo $! > "$LOGDIR/calibration_10ep_confusion_gpu1.pid"
  )
}

main() {
  cd "$REPO" || exit 1
  log "RCPS calibration watchdog started."

  wait_for_grid "$CAL" 68 "static/uniform calibration" || exit 1
  check_logs_clean "$LOGDIR/calibration_10ep_gpu*.log" || exit 1

  log "Selecting Static-LS and RCPS-Uniform candidates."
  run_py tools/rcps/select_calibration.py \
    --metrics "$CAL"/metrics/calibration/*.csv \
    --out-csv "$CAL/metrics/calibration/selected_static_uniform.csv" \
    --baseline-method hard-ce_hard \
    --high-min 10 \
    --max-high-drop 1.0 || exit 1

  log "Building class prior."
  run_py tools/rcps/build_prior.py "$ANN" --out "$PRIOR" || exit 1

  local teacher
  teacher=$(select_hard_teacher) || exit 1
  log "Building confusion base from teacher: $teacher"
  run_py tools/rcps/build_confusion_base.py "$teacher" --out "$CONFBASE" || exit 1

  launch_confusion_grid
  wait_for_grid "$CONF" 48 "confusion calibration" || exit 1
  check_logs_clean "$LOGDIR/calibration_10ep_confusion_gpu*.log" || exit 1

  log "Selecting RCPS-Confusion candidates."
  run_py tools/rcps/select_calibration.py \
    --metrics "$CAL"/metrics/calibration/*hard-ce_hard*.csv "$CONF"/metrics/calibration/*.csv \
    --out-csv "$CONF/metrics/calibration/selected_confusion.csv" \
    --baseline-method hard-ce_hard \
    --high-min 10 \
    --max-high-drop 1.0 || exit 1

  log "RCPS calibration watchdog completed successfully."
}

main "$@"
