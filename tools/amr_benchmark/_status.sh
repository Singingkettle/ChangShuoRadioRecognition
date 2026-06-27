#!/bin/bash
# Compact sweep status snapshot for monitoring the AMR-Benchmark driver.
cd /home/citybuster/Projects/ChangShuoRadioRecognition || exit 1
LOGDIR=work_dirs/amr_benchmark
echo "=== $(date '+%F %T') ==="
echo "--- driver/orchestrator procs ---"
ps -o pid,etime,%cpu,cmd -C python,bash 2>/dev/null | grep -E "_sweep_driver|run_migration" | grep -v grep
echo "--- paper.pkl per dataset ---"
for d in deepsig201610A deepsig201610B deepsig201801A hisar2019; do
  n=$(ls $LOGDIR/*/$d/res/paper.pkl 2>/dev/null | wc -l)
  echo "  $d: $n/15"
done
echo "--- driver.log ---"
cat "$LOGDIR/driver.log" 2>/dev/null
echo "--- active train cfg (epoch progress) ---"
for wd in $(ps aux | grep 'tools/train.py' | grep -v grep | grep -oE '\-\-work-dir [^ ]+' | awk '{print $2}' | sort -u); do
  last=$(ls -t "$wd"/epoch_*.pth 2>/dev/null | head -1)
  echo "  ${wd#*amr_benchmark/}: ${last##*/}"
done
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "--- tail current sweep log ---"
for f in sweep_hisar.log sweep_2018.log sweep_10b.log; do
  if [ -f "$LOGDIR/$f" ]; then echo "## $f"; tail -n 3 "$LOGDIR/$f"; break; fi
done
