#!/usr/bin/env bash
#
# Experiment: "refit cadence for crypto"
#
# How often should the model be retuned? Five runs, identical in every respect
# except how frequently the walk-forward refits:
#
#   month     22 periods  -> 24 steps
#   3month    65          ->  8
#   6month   130          ->  4
#   9month   195          ->  3
#   12month  260          ->  2
#
# All five predict the same 510 out-of-sample dates, so they are directly
# comparable: the only thing that varies is retuning frequency. The four longer
# cadences are much cheaper than the monthly one, so the whole sweep costs about
# 1.7x a single monthly run.
#
# The cadence is passed as "Nmonth" rather than a period count on purpose. A
# month is 22 periods on crypto but 4 on numerai, so a raw number silently means
# something different per tournament; the month form converts using the
# tournament's own period length.
#
# Note what each cadence can and cannot tell you. The monthly run produces 24
# rows in the hyperparameter history and is the one that can show whether alpha
# actually drifts; 12month produces 2 rows and cannot show drift at all. The
# longer cadences answer the other half of the question -- whether *not*
# retuning costs you anything.
#
# Usage:
#   bash experiments/refit_cadence_crypto.sh            # run all 5, in order
#   DRY_RUN=1 bash experiments/refit_cadence_crypto.sh  # print, run nothing
#   ONLY=6month bash experiments/refit_cadence_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---------------------------------------------------------------------------
# Knobs. Everything except --refit-every matches the plain
#   python3 -m src.run_forecast crypto ridge --refit-every month
# so the five runs differ in exactly one thing.
#
# SEARCH_JOBS/N_JOBS only affect speed, never results: the pipeline is
# deterministic apart from the seeded candidate sampling, so parallelism cannot
# change what gets selected.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-ridge}"
TASK="${TASK:-regression}"
METRIC="${METRIC:-corr}"
N_ITER="${N_ITER:-50}"
N_SPLITS="${N_SPLITS:-5}"
ERA_STEP="${ERA_STEP:-1}"
VAL_FRAC="${VAL_FRAC:-0.3}"
SEED="${SEED:-42}"
SEARCH_JOBS="${SEARCH_JOBS:-8}"
N_JOBS="${N_JOBS:-4}"
BMC_REFERENCE="${BMC_REFERENCE:-benchmark}"

CADENCES=(month 3month 6month 9month 12month)

total="${#CADENCES[@]}"
index=0
failed=0
started_all=$(date +%s)
mkdir -p logs

for cadence in "${CADENCES[@]}"; do
    index=$((index + 1))

    if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${cadence}" ]; then
        continue
    fi

    short_task="clf"; [ "$TASK" = "regression" ] && short_task="reg"
    run_name="${short_task}_${METRIC}_${cadence}"

    cmd=(python3 -u -m src.run_forecast crypto "${MODEL}"
         --task "${TASK}" --metric "${METRIC}"
         --refit-every "${cadence}"
         --bmc-reference "${BMC_REFERENCE}"
         --n-iter "${N_ITER}" --n-splits "${N_SPLITS}"
         --n-jobs "${N_JOBS}" --search-jobs "${SEARCH_JOBS}" --seed "${SEED}"
         --era-step "${ERA_STEP}" --val-frac "${VAL_FRAC}"
         --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-9s %s\n' "$index" "$total" "$cadence" "${cmd[*]}"
        continue
    fi

    echo
    echo "=============================================================="
    echo "[${index}/${total}] refit every ${cadence}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # One bad cadence must not cost you the other four.
    "${cmd[@]}" 2>&1 | tee "logs/cadence_${cadence}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${cadence} FAILED (exit ${status}) after ${elapsed}s -- continuing"
    else
        echo ">>> ${cadence} done in ${elapsed}s"
    fi
done

echo
echo "=============================================================="
echo "finished ${total} runs in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo "results in results/{forecast,per_period,hyperparams,metrics}/crypto/${MODEL}/${short_task:-reg}_${METRIC}_*month"
echo "=============================================================="
