#!/usr/bin/env bash
#
# Cron entry point for the JACI_01 live pre-stage (E216), run on deepsix BEFORE
# the 09:05 crypto round:
#
#   bash src/live/cron_jaci.sh            # what cron runs
#
# prices -> momentum3 panel -> GPU states (ssh to deepfive) -> forecast, all in
# `python3 -m src.live.jaci all`. It leaves models/live/jaci/forecasts.parquet,
# which run_round reads for CU_SA_IA_BO_MA_JA_01. If this job fails or is late,
# the slot still submits -- CU_SA_IA_BO_MA_R_01's ranks alone, never a stand-in
# labelled JACI -- and the round email carries a warning naming the cause. That
# warning is the alert; this script does not mail.
#
# Cron supplies almost nothing -- no conda, PATH=/usr/bin:/bin, no cwd -- so all
# three are named here.
set -uo pipefail

REPO="/home/danielco/Downloads/my-numerai-agent"
PYTHON="/home/danielco/miniconda3/bin/python3"
LOCK="/tmp/numerai_jaci_prestage.lock"
LOG_DIR="${REPO}/logs/jaci"
LOG="${LOG_DIR}/prestage_$(date +%Y%m%d).log"

cd "${REPO}" || { echo "$(date -Is) FATAL: cannot cd to ${REPO}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

echo "$(date -Is) START jaci pre-stage on $(hostname)" >> "${LOG}"
flock -n -E 99 "${LOCK}" timeout 50m "${PYTHON}" -u -m src.live.jaci all >> "${LOG}" 2>&1
STATUS=$?
if [ "${STATUS}" -eq 99 ]; then
    echo "$(date -Is) SKIPPED: a previous pre-stage is still running" >> "${LOG}"
else
    echo "$(date -Is) END jaci pre-stage (exit ${STATUS})" >> "${LOG}"
fi
exit "${STATUS}"
