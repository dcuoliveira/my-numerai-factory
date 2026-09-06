#!/usr/bin/env bash
#
# Cron entry point for the daily Numerai Crypto stake report.
#
#   bash src/analysis/cron_stake_report.sh          # what cron runs (mails)
#   NO_MAIL=1 bash src/analysis/cron_stake_report.sh
#
# Runs at 10:05 local (13:05 UTC), which is chosen and not arbitrary:
#
#   12:00Z  round opens
#   12:05Z / 12:25Z / 12:45Z  the three submission passes (cron_round.sh)
#   ~13:05Z closeStakingTime -- the real deadline
#   13:05Z  this report
#
# It has to be AFTER the last pass, so the round it reports on is the round that
# was actually submitted. It also has to be at least an hour after the open:
# Numerai does not serve `atRisk` for a round in its first hour, and a report
# built before that reads the newest round as unstaked, which is not the same as
# stake zero.
#
# Cron supplies almost nothing -- no conda, PATH=/usr/bin:/bin, no working
# directory -- so all three are named here. MAILTO is empty in this crontab, so
# cron will not report a failure: the job mails its own result via --email, and
# a missing mail is the signal that it did not run.
#
set -uo pipefail

REPO="/home/danielco/Downloads/my-numerai-agent"
PYTHON="/home/danielco/miniconda3/bin/python3"
LOCK="/tmp/numerai_stake_report.lock"
LOG_DIR="${REPO}/logs"
LOG="${LOG_DIR}/stake_report.log"

cd "${REPO}" || { echo "$(date -Is) FATAL: cannot cd to ${REPO}" >> "${LOG}"; exit 1; }
mkdir -p "${LOG_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "$(date -Is) FATAL: interpreter ${PYTHON} is missing" >> "${LOG}"
    exit 1
fi

MAIL_FLAG="--email"
[ "${NO_MAIL:-0}" = "1" ] && MAIL_FLAG=""

echo "$(date -Is) START stake report" >> "${LOG}"
flock -n -E 99 "${LOCK}" \
    "${PYTHON}" -m src.analysis.crypto_stake_report ${MAIL_FLAG} >> "${LOG}" 2>&1
STATUS=$?
if [ "${STATUS}" -eq 99 ]; then
    echo "$(date -Is) SKIPPED: a previous stake report is still running" >> "${LOG}"
else
    echo "$(date -Is) END stake report (exit ${STATUS})" >> "${LOG}"
fi
exit "${STATUS}"
