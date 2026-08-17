#!/usr/bin/env bash
#
# Cron entry point for one live Numerai Crypto round.
#
#   bash src/live/cron_round.sh            # what cron runs
#   DRY_RUN=1 bash src/live/cron_round.sh  # rehearse
#
# Cron gives a process almost no environment: no conda, a PATH of /usr/bin:/bin,
# no working directory, and no shell profile. Every one of those has to be
# supplied here rather than assumed, because a cron job that fails on a missing
# import fails silently and weekly.
#
# Three things this adds over calling the module directly:
#
#   1. The absolute interpreter. `python3` on PATH under cron is /usr/bin/python3,
#      which does not have numerapi, sklearn or pandas installed. The conda
#      interpreter is named in full.
#   2. A lock. Two overlapping rounds would race on the model cache and the
#      submission CSVs. flock makes a second start a no-op rather than a corruption.
#   3. A log line either side, so an empty log is distinguishable from a log that
#      never ran.
#
set -uo pipefail

REPO="/home/danielco/Downloads/my-numerai-agent"
PYTHON="/home/danielco/miniconda3/bin/python3"
LOCK="/tmp/numerai_live_round.lock"
LOG_DIR="${REPO}/logs"
LOG="${LOG_DIR}/live_round.log"

DEADLINE="${DEADLINE:-60}"

cd "${REPO}" || { echo "$(date -Is) FATAL: cannot cd to ${REPO}" >> "${LOG}"; exit 1; }
mkdir -p "${LOG_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "$(date -Is) FATAL: interpreter ${PYTHON} is missing" >> "${LOG}"
    exit 1
fi

ARGS=(--deadline-minutes "${DEADLINE}")
[ "${DRY_RUN:-0}" = "1" ] && ARGS+=(--dry-run)
# Set on the retry passes: slots that already submitted from their own model are
# skipped, so a retry after a clean run costs one API call instead of re-uploading.
[ "${SKIP_IF_SUBMITTED:-0}" = "1" ] && ARGS+=(--skip-if-submitted)

{
    echo
    echo "=============================================================="
    echo "$(date -Is) starting round (deadline ${DEADLINE} min, host $(hostname))"
    echo "=============================================================="
} >> "${LOG}"

# -n: if another round is still running, do not queue behind it -- by the time
# it finished this one's data would be stale anyway.
#
# -E 99: flock's default "could not acquire" exit code is 1, which is exactly
# what run_round returns when a slot fails. Without a distinct code, "another
# round is running" and "a submission failed" look identical in the log, and
# only one of them needs waking up for.
flock -n -E 99 "${LOCK}" "${PYTHON}" -u -m src.live.run_round "${ARGS[@]}" >> "${LOG}" 2>&1
status=$?

if [ "${status}" -eq 99 ]; then
    echo "$(date -Is) SKIPPED: another round is still holding ${LOCK}" >> "${LOG}"
    exit 0
fi

echo "$(date -Is) finished with exit ${status}" >> "${LOG}"
exit "${status}"
