#!/bin/bash
# One-shot recovery driver for the two RML2018.01A jobs that were killed by
# the kernel OOM killer on 2026-07-04 (host RAM exhaustion, not a model or
# config problem):
#   * mldnn@deepsig201801A     - DataLoader worker OOM-killed at epoch 64;
#                                val acc had plateaued (best 58.06% @ epoch
#                                62), so we only re-run the test step on the
#                                existing best checkpoint.
#   * fastmldnn@deepsig201801A - main train process OOM-killed in epoch 1
#                                (fc8c10c warmup config was applied and
#                                healthy); clean retrain + test.
#
# Runs on GPU 0 ONLY; never touches GPU 1 (fastmldnn@hisar2019 sweep job).
# Idempotent: each step is skipped when its res/paper.pkl already exists.
# Coordination with _finalize_driver.sh: once this retrain has produced its
# first best_*.pth, the finalize sweep will skip training for these jobs and
# at most re-run a cheap test with the then-current best checkpoint; the
# final test below overwrites paper.pkl with the fully-trained result.

set -u

REPO=/home/citybuster/Projects/ChangShuoRadioRecognition
PY=/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python
cd "$REPO" || exit 1
LOGDIR="$REPO/work_dirs/amr_benchmark"
LOG="$LOGDIR/recover_2018.log"
LOCK="$LOGDIR/recover_2018.lock"

exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    echo "[recover] another instance holds the lock; exiting" >> "$LOG"
    exit 0
fi

log() { echo "[recover $(date '+%F %T')] $*" >> "$LOG"; }

export CUDA_VISIBLE_DEVICES=0

log "=============================================================="
log "recovery driver started (pid $$, GPU 0)"

# ---- mldnn@2018: test-only on the converged best checkpoint --------------
mldnn_recover() {
    local mw="$LOGDIR/mldnn/deepsig201801A"
    local ckpt="$mw/best_accuracy_top1_epoch_62.pth"
    if [ -f "$mw/res/paper.pkl" ]; then
        log "mldnn: res/paper.pkl already exists, skipping"
        return 0
    fi
    if [ ! -f "$ckpt" ]; then
        log "mldnn: ERROR checkpoint $ckpt missing"
        return 1
    fi
    log "mldnn: test start (ckpt=$ckpt)"
    "$PY" tools/test.py configs/mldnn/mldnn_iq-ap-deepsig201801A.py \
        "$ckpt" --work-dir "$mw" >> "$mw/orchestrator.log" 2>&1
    local rc=$?
    log "mldnn: test finished rc=$rc paper.pkl=$([ -f "$mw/res/paper.pkl" ] && echo yes || echo no)"
    return $rc
}

# ---- fastmldnn@2018: clean retrain then test ------------------------------
fastmldnn_recover() {
    local fw="$LOGDIR/fastmldnn/deepsig201801A"
    if [ -f "$fw/res/paper.pkl" ] && ls "$fw"/best_*.pth >/dev/null 2>&1; then
        log "fastmldnn: paper.pkl + best ckpt already exist, skipping"
        return 0
    fi
    log "fastmldnn: train start"
    "$PY" tools/train.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201801A.py \
        --work-dir "$fw" >> "$fw/orchestrator.log" 2>&1
    local rc=$?
    log "fastmldnn: train finished rc=$rc"
    if [ $rc -ne 0 ]; then
        return $rc
    fi
    local best
    best=$(ls -t "$fw"/best_*.pth 2>/dev/null | head -n 1)
    if [ -z "$best" ]; then
        log "fastmldnn: ERROR no best checkpoint after training"
        return 1
    fi
    log "fastmldnn: test start (ckpt=$best)"
    "$PY" tools/test.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201801A.py \
        "$best" --work-dir "$fw" >> "$fw/orchestrator.log" 2>&1
    rc=$?
    log "fastmldnn: test finished rc=$rc paper.pkl=$([ -f "$fw/res/paper.pkl" ] && echo yes || echo no)"
    return $rc
}

mldnn_recover &
MLDNN_JOB=$!
fastmldnn_recover
FRC=$?
wait "$MLDNN_JOB"
MRC=$?
log "recovery driver done (mldnn rc=$MRC, fastmldnn rc=$FRC)"
