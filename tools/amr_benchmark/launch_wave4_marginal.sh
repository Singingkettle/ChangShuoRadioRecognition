#!/usr/bin/env bash
# One-shot launcher for wave4 marginal retune (both GPUs, --force).
set -euo pipefail
REPO="/home/citybuster/Projects/ChangShuoRadioRecognition"
PY="/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python"
LOGDIR="${REPO}/work_dirs/amr_benchmark_retune"
mkdir -p "${LOGDIR}"

cd "${REPO}"

if pgrep -f "retune_model_siege.py.*wave4_marginal_manifest" >/dev/null 2>&1; then
    echo "wave4 siege already running: $(pgrep -af 'retune_model_siege.py.*wave4_marginal_manifest')"
else
    nohup "${PY}" tools/amr_benchmark/retune_model_siege.py \
        --manifest configs/amr_benchmark/retune/wave4_marginal_manifest.json \
        --gpu 0,1 --max-parallel 2 --until-pass --paper-exact --promote --force \
        >> "${LOGDIR}/wave4_marginal.log" 2>&1 &
    echo "WAVE4_PID=$!"
fi

if ! pgrep -f "tools/amr_benchmark/gpu_keepalive.sh" >/dev/null 2>&1; then
    nohup bash tools/amr_benchmark/gpu_keepalive.sh >> "${LOGDIR}/scheduler.log" 2>&1 &
    echo "KEEPALIVE_PID=$!"
fi

if ! pgrep -f "tools/amr_benchmark/health_watchdog.sh" >/dev/null 2>&1; then
    nohup bash tools/amr_benchmark/health_watchdog.sh >> "${LOGDIR}/health.log" 2>&1 &
    echo "WATCHDOG_PID=$!"
fi

sleep 30
echo "--- GPU util after 30s ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 || true
echo "--- processes ---"
pgrep -af "retune_model_siege|gpu_keepalive|health_watchdog|train.py" 2>&1 || true
echo "--- wave4 log tail ---"
tail -15 "${LOGDIR}/wave4_marginal.log" 2>&1 || true
