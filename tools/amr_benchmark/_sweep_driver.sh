#!/bin/bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition || exit 1
PY=/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python
LOGDIR=work_dirs/amr_benchmark

until grep -q "Run complete" "$LOGDIR/sweep_10b.log" 2>/dev/null; do
    sleep 180
done
echo "[driver] 10B complete, starting 2018.01A $(date)" >> "$LOGDIR/driver.log"

$PY tools/amr_benchmark/run_migration.py --datasets deepsig201610A deepsig201610B deepsig201801A --gpus 0,1 --max-parallel 2 -v >> "$LOGDIR/sweep_2018.log" 2>&1
echo "[driver] 2018.01A complete, starting HisarMod $(date)" >> "$LOGDIR/driver.log"

$PY tools/amr_benchmark/run_migration.py --datasets deepsig201610A deepsig201610B deepsig201801A hisar2019 --gpus 0,1 --max-parallel 2 -v >> "$LOGDIR/sweep_hisar.log" 2>&1
echo "[driver] HisarMod complete, rendering full table $(date)" >> "$LOGDIR/driver.log"

$PY tools/amr_benchmark/run_migration.py --datasets deepsig201610A deepsig201610B deepsig201801A hisar2019 --skip-train --skip-test -v >> "$LOGDIR/sweep_final.log" 2>&1
echo "[driver] ALL DONE $(date)" >> "$LOGDIR/driver.log"
