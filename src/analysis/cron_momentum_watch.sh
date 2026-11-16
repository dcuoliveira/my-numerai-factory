#!/usr/bin/env bash
#
# Cron entry point for the E55 factor-momentum stake monitor.
#
#   bash src/analysis/cron_momentum_watch.sh          # what cron runs (mails on CHANGE)
#   NO_MAIL=1 bash src/analysis/cron_momentum_watch.sh
#
# Reruns `top1 roll250` -- the best single model by 250-day trailing payout on
# RESOLVED rounds -- and e-mails ONLY when that pick changes from the stored one
# (results/live/momentum_top1.json), so you know to move the stake. Runs at
# 10:20 local (13:20 UTC), after the three crypto submission passes
# (cron_round.sh at :05/:25/:45) and off the Classic iara pattern (:15/:35/:55)
# and its 10:30 watchdog.
#
# NOTE it will stay SILENT until the models build RESOLVED live history: today
# the crypto performance API returns only ~18 recent, unresolved rounds, so the
# 250-day signal is pinned to the backtest and cannot move. The monitor is
# correct and auto-activates as live rounds resolve; near-term "no mail" is the
# expected state, not a fault.
#
# Cron supplies almost nothing -- no conda, PATH=/usr/bin:/bin, no cwd -- so all
# three are named here. MAILTO is empty in this crontab, so cron reports no
# failure; the job mails its own result on a change, and the log is the record.
#
set -uo pipefail

REPO="/home/danielco/Downloads/my-numerai-agent"
PYTHON="/home/danielco/miniconda3/bin/python3"
LOCK="/tmp/numerai_momentum_watch.lock"
LOG_DIR="${REPO}/logs"
LOG="${LOG_DIR}/momentum_watch.log"

cd "${REPO}" || { echo "$(date -Is) FATAL: cannot cd to ${REPO}" >> "${LOG}"; exit 1; }
mkdir -p "${LOG_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "$(date -Is) FATAL: interpreter ${PYTHON} is missing" >> "${LOG}"
    exit 1
fi

MAIL_ARG=""
[ "${NO_MAIL:-0}" = "1" ] && MAIL_ARG="--no-mail"

echo "$(date -Is) START momentum watch" >> "${LOG}"
flock -n -E 99 "${LOCK}" \
    "${PYTHON}" -m src.analysis.factor_momentum_stake --watch ${MAIL_ARG} >> "${LOG}" 2>&1
STATUS=$?
if [ "${STATUS}" -eq 99 ]; then
    echo "$(date -Is) SKIPPED: a previous momentum watch is still running" >> "${LOG}"
else
    echo "$(date -Is) END momentum watch (exit ${STATUS})" >> "${LOG}"
fi
exit "${STATUS}"
