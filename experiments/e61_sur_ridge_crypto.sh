#!/usr/bin/env bash
#
# Experiment E61: multi-symbol multivariate ridge (SUR-style feasible GLS)
# against the stacked panel-data ridge, on every book dataset, reg vs clf.
#
# E61 is the reference handle; every artefact is e61_*.
#
# THE QUESTION
#   The deployed linear slots fit ONE shared beta on the STACKED panel -- every
#   (symbol,date) row an independent observation, errors i.i.d. That is
#   `ridge_glmnet`. `sur_ridge` (src/prediction/sur_ridge.py) keeps the shared
#   beta but partials out k common CROSS-SYMBOL factors from the residuals -- the
#   feasible-GLS adjustment under a factor covariance -- so it uses the fact that
#   coins' errors co-move within a date. n_factors=0 is an EXACT stacked-ridge
#   control, so the whole value of the method is the searched k>0 delta.
#
# WHAT THIS RUNS
#   Two models x two tasks x 9 datasets = 36 from-scratch tunes, SAME protocol
#   (rolling 130/130, embargo auto=20, 5 folds, 50 draws, seed 42) so every
#   sur_ridge arm pairs cleanly against its matched ridge_glmnet on the same
#   dataset and task, and reg pairs against clf.
#
#     sur_ridge : searches alpha (50 log levels) x n_factors {0,1,2,3,5}
#     ridge_glmnet : the glmnet lambda path (the stacked baseline "we already have")
#
#   9 datasets (the linear book's, same as E29):
#     raw            v2.0                                      22
#     tenc           v2.0-tenc                                 23
#     interact       v2.0-interact                             36
#     fenccausal     v2.0-fenccausal                           44
#     cumlag3        v2.0-cumlag3                              88
#     tenc_onehot    v2.0-tenc-onehot                         763
#     onehot_rollsel v2.0-onehot-rollsel                      786
#     allfeat        v2.0-tenc-onehot-spread-interact         788
#     allfeatpp      v2.0-tenc-onehot-rollsel-spread-interact 812
#
# WHAT WOULD COUNT AS A RESULT
#   sur_ridge beating its matched ridge_glmnet on numerai_corr at paired |t|>=2
#   on any dataset -- i.e. the cross-symbol factor adjustment helps the ranking.
#   The prior (E27/E30 and every fitted-vs-1/N result here) is a null: modelling
#   the error covariance sharpens beta in MSE terms but need not move rank corr.
#   The n_factors the search KEEPS is the headline diagnostic -- if it sits at 0
#   the method reduces to the stacked ridge on its own.
#
# Usage:  bash experiments/e61_sur_ridge_crypto.sh   [DRY_RUN=1] [NO_MAIL=1]
#         ONLY="sur_reg_raw glm_reg_raw"   SKIP=...   SHARD=1 OF=4
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E61}"
E61_WINDOW="${E61_WINDOW:-130}"
E61_REFIT="${E61_REFIT:-month}"
E61_N_ITER="${E61_N_ITER:-50}"
E61_N_SPLITS="${E61_N_SPLITS:-5}"
E61_N_JOBS="${E61_N_JOBS:-2}"
E61_SEARCH_JOBS="${E61_SEARCH_JOBS:-16}"
E61_SEED="${E61_SEED:-42}"

DSNAME=(raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp)
DSVER=("crypto/v2.0" "crypto/v2.0-tenc" "crypto/v2.0-interact" "crypto/v2.0-fenccausal" \
       "crypto/v2.0-cumlag3" "crypto/v2.0-tenc-onehot" "crypto/v2.0-onehot-rollsel" \
       "crypto/v2.0-tenc-onehot-spread-interact" "crypto/v2.0-tenc-onehot-rollsel-spread-interact")

ARM=(); DATA=(); MODEL=(); TASK=()
aadd() { ARM+=("$1"); MODEL+=("$2"); TASK+=("$3"); DATA+=("$4"); }
for i in "${!DSNAME[@]}"; do
    d="${DSNAME[$i]}"; v="${DSVER[$i]}"
    aadd "sur_reg_${d}" sur_ridge    regression     "$v"
    aadd "sur_clf_${d}" sur_ridge    classification "$v"
    aadd "glm_reg_${d}" ridge_glmnet regression     "$v"
    aadd "glm_clf_${d}" ridge_glmnet classification "$v"
done

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: sur_ridge vs ridge_glmnet, reg vs clf, x 9 datasets
  model     sur_ridge = cross-symbol factor-GLS; ridge_glmnet = stacked baseline
  budget    --n-iter ${E61_N_ITER}, ${E61_N_SPLITS} folds, rolling ${E61_WINDOW}/${E61_REFIT}, seed ${E61_SEED}

Where to look while it runs:
  runner    experiments/e61_sur_ridge_crypto.sh
  progress  logs/e61_*.log
  results   results/metrics/crypto/{sur_ridge,ridge_glmnet}/e61_*.json
  report    logs/e61_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e61_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: sur_ridge vs ridge_glmnet, reg vs clf, x9 datasets | ${total} arms"
echo "n_iter=${E61_N_ITER} folds=${E61_N_SPLITS} window=${E61_WINDOW}/${E61_REFIT} seed=${E61_SEED} shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="e61_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${MODEL[$i]}" --task "${TASK[$i]}" --metric corr
         --cv-scheme rolling --train-window "${E61_WINDOW}" --refit-every "${E61_REFIT}"
         --data-version "${DATA[$i]}"
         --n-iter "${E61_N_ITER}" --n-splits "${E61_N_SPLITS}"
         --n-jobs "${E61_N_JOBS}" --search-jobs "${E61_SEARCH_JOBS}"
         --era-step 1 --seed "${E61_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-14s %s\n' "$index" "$total" "$run_name" "${MODEL[$i]}" "${TASK[$i]}" "${DATA[$i]}"; continue; fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${MODEL[$i]} ${TASK[$i]}  ${DATA[$i]}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e61_report.txt"
{
echo "${EXPERIMENT} -- sur_ridge (cross-symbol factor-GLS) vs ridge_glmnet, reg vs clf"
echo "  results  results/metrics/crypto/{sur_ridge,ridge_glmnet}/e61_*.json"
echo "  runner   experiments/e61_sur_ridge_crypto.sh"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
} | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
sys.argv[1], open(sys.argv[2]).read())" "[crypto] ${EXPERIMENT} done" "${REPORT}" \
      || echo "note: completion email failed"
fi
