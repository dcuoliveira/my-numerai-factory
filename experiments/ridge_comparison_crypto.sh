#!/usr/bin/env bash
#
# Experiment: "ridge comparison for crypto"
#
# 2 models x 2 tasks x 4 losses = 16 runs, executed ONE AT A TIME.
#
#   models  ridge_glmnet   data-driven glmnet lambda path, rebuilt per refit
#           ridge          fixed logspace(-4, 2, 100) grid
#   tasks   clf            classification on the binarised target, predict_proba
#           reg            regression on the [0, 1] binned target
#   losses  corr           mean per-period numerai_corr
#           sharpe         corr_sharpe: mean / std of the per-period corr
#           bmc            mean per-period contribution over the reference
#           blend          0.1 * corr + 1 * bmc   (see WEIGHT below)
#
# Sequential, not parallel. Because only one run is live at a time it gets the
# whole machine, so the search is parallelised across candidates and folds
# instead of across runs -- which is the better trade here anyway, since the
# bottleneck is the per-period scoring loop rather than the ridge fit itself.
#
# Two deliberate choices, both load-bearing:
#
# 1. Every run uses --bmc-reference meta_proxy.
#    Crypto's benchmark IS the published meta model, which starts 2024-05 and so
#    does not exist anywhere in the training block the inner CV folds live in.
#    A BMC objective against it is unscorable and the search would raise. The
#    proxy is defined over every period. Using it for all 16 also keeps the
#    reported BMC comparable across the whole experiment -- mixing references
#    would make half the table meaningless.
#
# 2. WEIGHT=0.0909 encodes "0.1 * corr + 1 * bmc".
#    The runner takes a convex weight: w * corr + (1 - w) * bmc. Setting
#    w = 0.1 / (0.1 + 1.0) = 0.0909 gives exactly (1 / 1.1) * (0.1*corr + 1*bmc).
#    Scaling an objective by a positive constant does not move its argmax, so
#    the selected model is identical; only the reported cv_score is 1/1.1 of the
#    literal sum.
#
# Usage:
#   bash experiments/ridge_comparison_crypto.sh            # run all 16, in order
#   DRY_RUN=1 bash experiments/ridge_comparison_crypto.sh  # print, run nothing
#   ONLY=glmnet_clf_corr bash experiments/...              # run a single config
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---------------------------------------------------------------------------
# Knobs.
#
# SEARCH_JOBS x N_JOBS should be about the core count. With one run at a time
# the search can fan out across processes; the runner refuses SEARCH_JOBS > 1
# together with N_JOBS=-1, because that would give every worker every core.
#
# ERA_STEP=1 matters more than it looks: the meta proxy is fit on the overlap of
# "in training" and "has a meta model", only ~37 periods at full resolution.
# Downsampling shrinks it below the point where the proxy is worth trusting.
# ---------------------------------------------------------------------------
N_ITER="${N_ITER:-25}"              # hyperparameter draws per refit
N_SPLITS="${N_SPLITS:-5}"           # inner CV folds
ERA_STEP="${ERA_STEP:-1}"           # keep every Nth date; 1 = full resolution
# One month of crypto rounds. Measured from the data, not assumed: the median
# calendar month holds 22 dates (mean 21.5, p10 20, p90 23).
#
# This is 24 walk-forward steps over the 510 validation dates, against 3 at the
# old yearly cadence -- so roughly 8x the hyperparameter searches per run. Two
# things make that worth it: the model re-tunes monthly, which is how you would
# actually operate it, and the hyperparameter history becomes 24 points instead
# of 3, which is the difference between a time series and an anecdote.
#
# If it proves too slow, the cheap lever is N_ITER rather than this. The other
# option is CV_SCHEME=rolling with a TRAIN_WINDOW, which keeps every refit the
# same size instead of letting the expanding window grow the cost step by step.
REFIT_EVERY="${REFIT_EVERY:-month}"
VAL_FRAC="${VAL_FRAC:-0.3}"
SEED="${SEED:-42}"
SEARCH_JOBS="${SEARCH_JOBS:-8}"     # candidate/fold evaluations in parallel
N_JOBS="${N_JOBS:-4}"               # threads inside each of those
WEIGHT="${WEIGHT:-0.0909}"          # 0.1*corr + 1*bmc, normalised

EXPERIMENT="ridgecmp"
mkdir -p logs

# model | task | loss | extra flags
CONFIGS=(
  "ridge_glmnet|classification|corr|--metric corr"
  "ridge_glmnet|classification|sharpe|--metric corr_sharpe"
  "ridge_glmnet|classification|bmc|--metric bmc"
  "ridge_glmnet|classification|blend|--metric corr --metric2 bmc --metric-weight ${WEIGHT}"
  "ridge_glmnet|regression|corr|--metric corr"
  "ridge_glmnet|regression|sharpe|--metric corr_sharpe"
  "ridge_glmnet|regression|bmc|--metric bmc"
  "ridge_glmnet|regression|blend|--metric corr --metric2 bmc --metric-weight ${WEIGHT}"
  "ridge|classification|corr|--metric corr"
  "ridge|classification|sharpe|--metric corr_sharpe"
  "ridge|classification|bmc|--metric bmc"
  "ridge|classification|blend|--metric corr --metric2 bmc --metric-weight ${WEIGHT}"
  "ridge|regression|corr|--metric corr"
  "ridge|regression|sharpe|--metric corr_sharpe"
  "ridge|regression|bmc|--metric bmc"
  "ridge|regression|blend|--metric corr --metric2 bmc --metric-weight ${WEIGHT}"
)

total="${#CONFIGS[@]}"
index=0
failed=0
started_all=$(date +%s)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r model task loss extra <<< "$entry"
    index=$((index + 1))

    short_task="clf"; [ "$task" = "regression" ] && short_task="reg"
    short_model="glmnet"; [ "$model" = "ridge" ] && short_model="classical"

    run_name="${EXPERIMENT}_${short_task}_${loss}"
    label="${short_model}_${short_task}_${loss}"

    # ONLY=<label> runs just that one, useful for retrying a single failure.
    if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${label}" ]; then
        continue
    fi

    cmd=(python3 -u -m src.run_forecast crypto "${model}"
         --task "${task}"
         --bmc-reference meta_proxy
         --n-iter "${N_ITER}" --n-splits "${N_SPLITS}"
         --n-jobs "${N_JOBS}" --search-jobs "${SEARCH_JOBS}" --seed "${SEED}"
         --era-step "${ERA_STEP}" --refit-every "${REFIT_EVERY}"
         --val-frac "${VAL_FRAC}" --run-name "${run_name}" ${extra})

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-26s %s\n' "$index" "$total" "$label" "${cmd[*]}"
        continue
    fi

    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${label}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # Do not let one failure abort the sequence -- record it and carry on, so a
    # single bad config does not cost you the other fifteen.
    "${cmd[@]}" 2>&1 | tee "logs/${EXPERIMENT}_${label}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${label} FAILED (exit ${status}) after ${elapsed}s -- continuing"
    else
        echo ">>> ${label} done in ${elapsed}s"
    fi
done

echo
echo "=============================================================="
echo "finished ${total} runs in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo "results in results/{forecast,per_period,metrics}/crypto/{ridge,ridge_glmnet}/${EXPERIMENT}_*"
echo "=============================================================="
