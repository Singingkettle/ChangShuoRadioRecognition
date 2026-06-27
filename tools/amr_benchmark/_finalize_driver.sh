#!/bin/bash
# Durable, idempotent finalize driver for the AMR-Benchmark migration tail.
#
# Runs detached via nohup. Safe to re-run: every training step delegates to
# tools/amr_benchmark/run_migration.py, which skips any (model,dataset) whose
# work_dirs/amr_benchmark/<model>/<dataset>/res/paper.pkl already exists, and
# reuses existing checkpoints. No results are ever deleted.
#
# Steps, in order:
#   a. Wait until the existing sweep driver (_sweep_driver.sh / run_migration.py)
#      has fully finished RML2018.01A + HisarMod.
#   b. Train leftover jobs the main sweep does not cover (lstm2@10B; own methods
#      on 10A/10B; own methods on 2018/Hisar if missed), then a final full
#      idempotent sweep to guarantee complete 18x4 coverage.
#   c. Render every plot config with tools/analyze.py (each wrapped so one
#      failure cannot abort the rest, and analyze errors never abort training).
#   d. Write the work_dirs/amr_benchmark/FINALIZE_DONE sentinel.
#
# This script deliberately does NOT do final adjudication, bounded re-tuning,
# own_methods_results.md, README, or accuracy_tracking.md finalization. Those
# are left for a short follow-up agent run after FINALIZE_DONE appears.

set -u

REPO=/home/citybuster/Projects/ChangShuoRadioRecognition
cd "$REPO" || exit 1
PY=/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python
LOGDIR="$REPO/work_dirs/amr_benchmark"
LOG="$LOGDIR/finalize_driver.log"
LOCK="$LOGDIR/finalize_driver.lock"
SENTINEL="$LOGDIR/FINALIZE_DONE"
mkdir -p "$LOGDIR"

# ----- single-instance lock -----
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    echo "[finalize] another instance holds the lock; exiting" >> "$LOG"
    exit 0
fi

log() { echo "[finalize $(date '+%F %T')] $*" >> "$LOG"; }

ALL_MODELS="cnn2 cnn4 mcnet icamcnet resnetamr denscnn gru2 lstm2 dae mcldnn cldnnw cldnnl cgdnet petcgdnn cnn1dpf mldnn hcgdnn fastmldnn"
ALL_DATASETS="deepsig201610A deepsig201610B deepsig201801A hisar2019"

count_coverage() {
    local n=0
    local m d
    for m in $ALL_MODELS; do
        for d in $ALL_DATASETS; do
            [ -f "$LOGDIR/$m/$d/res/paper.pkl" ] && n=$((n+1))
        done
    done
    echo "$n"
}

log "=============================================================="
log "finalize driver started (pid $$)"
log "initial paper.pkl coverage: $(count_coverage)/72"

# ----- step a: wait for the existing sweep to finish -----
log "step a: waiting for existing sweep driver to finish (2018.01A + HisarMod)"
while pgrep -f 'run_migration.py' >/dev/null 2>&1 || pgrep -f '_sweep_driver.sh' >/dev/null 2>&1; do
    sleep 180
done
log "step a: existing sweep driver has exited. coverage now: $(count_coverage)/72"

# ----- step b: fill leftover training jobs (idempotent) -----
run_mig() {
    # $1 = label ; rest = args
    local label="$1"; shift
    log "step b: $label -> run_migration $*"
    "$PY" tools/amr_benchmark/run_migration.py --gpus 0,1 --max-parallel 2 -v "$@" \
        >> "$LOGDIR/finalize_train.log" 2>&1
    log "step b: $label finished (rc=$?), coverage: $(count_coverage)/72"
}

# lstm2@10B (config fixed) + own methods on the two RML2016 datasets the main
# sweep's 10A/10B phases did not cover.
run_mig "lstm2+own @ 2016" --models lstm2 mldnn hcgdnn fastmldnn \
        --datasets deepsig201610A deepsig201610B
# own methods on the large datasets (skipped automatically if main sweep got them)
run_mig "own @ 2018+hisar" --models mldnn hcgdnn fastmldnn \
        --datasets deepsig201801A hisar2019
# final safety net: full idempotent sweep so any straggler (any model, any
# dataset) gets trained/tested exactly once. Existing paper.pkl are skipped.
run_mig "full idempotent sweep" --datasets deepsig201610A deepsig201610B deepsig201801A hisar2019

log "step b complete. final coverage: $(count_coverage)/72"

# ----- step c: render all plot configs (defensive) -----
log "step c: rendering plot configs into work_dirs/performance/"
PLOT_CFGS="
configs/amr_benchmark/plot_deepsig201610A.py
configs/amr_benchmark/plot_deepsig201610B.py
configs/amr_benchmark/plot_deepsig201801A.py
configs/amr_benchmark/plot_hisar2019.py
configs/amr_benchmark/plot_own_deepsig201610A.py
configs/amr_benchmark/plot_own_deepsig201610B.py
configs/amr_benchmark/plot_own_deepsig201801A.py
configs/amr_benchmark/plot_own_hisar2019.py
"
for cfg in $PLOT_CFGS; do
    if [ ! -f "$cfg" ]; then
        log "step c: MISSING plot config $cfg (skipped)"
        continue
    fi
    log "step c: analyze.py $cfg"
    if "$PY" tools/analyze.py "$cfg" >> "$LOGDIR/finalize_analyze.log" 2>&1; then
        log "step c: OK $cfg"
    else
        log "step c: FAILED $cfg (rc=$?, continuing)"
    fi
done

# ----- step d: sentinel -----
COV=$(count_coverage)
echo "$(date '+%F %T') FINALIZE_DONE coverage=${COV}/72" > "$SENTINEL"
log "step d: wrote sentinel -> $SENTINEL (coverage ${COV}/72)"
log "finalize driver done."
