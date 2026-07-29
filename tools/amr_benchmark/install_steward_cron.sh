#!/usr/bin/env bash
# Install (idempotently) the hourly two-machine steward cron entry on the LOCAL
# box. The steward keeps both boxes' daemons alive, kills sustained GPU idle,
# and escalates strategy on plateau. Safe to re-run; it never duplicates the
# entry and never touches other crontab lines.
#
#   bash tools/amr_benchmark/install_steward_cron.sh
#   bash tools/amr_benchmark/install_steward_cron.sh --remove   # uninstall
set -uo pipefail

REPO="${REPO_ROOT:-/home/citybuster/Projects/ChangShuoRadioRecognition}"
PY="${PYTHON:-/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python}"
LOG="${REPO}/work_dirs/amr_benchmark_retune/steward_cron.log"
# Every 30 minutes: tight enough to catch stalls/finished-runs promptly, and
# every corrective action inside the steward is stamp-rate-limited anyway.
ENTRY="*/30 * * * * cd ${REPO} && ${PY} tools/amr_benchmark/hourly_steward.py >> ${LOG} 2>&1"

if [[ "${1:-}" == "--remove" ]]; then
    crontab -l 2>/dev/null | grep -v 'hourly_steward.py' | crontab -
    echo "steward cron removed"
    exit 0
fi

# Drop any prior steward line, then append the current one.
( crontab -l 2>/dev/null | grep -v 'hourly_steward.py'; echo "${ENTRY}" ) | crontab -
echo "steward cron installed:"
crontab -l 2>/dev/null | grep steward
