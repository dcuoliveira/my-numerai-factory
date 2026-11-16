#!/usr/bin/env bash
#
# Experiment E62: the E61 cross-symbol factor-GLS (SUR) treatment generalised to
# the tree families -- lgb, rf, catboost -- against their stacked baselines,
# every book dataset, reg vs clf.
#
# E62 is the reference handle; every artefact is e62_*.
#
# THE APPROACH (generalised from E61 / sur_ridge)
#   A tree has no coefficients to partial, so SurAugment (src/prediction/sur_ridge.py)
#   adds the k common CROSS-SYMBOL factor scores as EXTRA FEATURES during fit (the
#   tree explains away common shocks in-sample) and predicts with them ZEROED --
#   future factor scores are unknowable, exactly as sur_ridge predicts x'beta
#   without F. n_factors=0 is an EXACT plain-tree control, so the value is the
#   searched k>0 delta. Factors come from a cheap stage-1 ridge's residuals, so
#   every family shares one model-agnostic notion of a common shock.
#
# WHAT THIS RUNS  (108 arms = 3 families x 2 tasks x 9 datasets x {sur, stacked})
#   For each (family, task, dataset): a sur_<fam> arm and its MATCHED stacked
#   baseline, same protocol (rolling 130/130, embargo 20, 5 folds, 50 draws,
#   seed 42), so sur pairs cleanly against stacked and reg against clf.
#     lgb       : sur_lgb        vs lgb_ktz          (both tasks)
#     rf        : sur_rf         vs rf_ktz           (both tasks)
#     catboost  : sur_catboost   vs catboost_ktz(reg) / catboost(clf)
#       (catboost_ktz is regression-only; the plain catboost carries the clf, and
#        sur_catboost switches its base the same way, so each cell stays matched.)
#
#   9 datasets (the linear book's, same as E29/E61):
#     raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp
#
# WHAT WOULD COUNT AS A RESULT
#   sur_<fam> beating its matched stacked tree on numerai_corr at paired |t|>=2.
#   Prior (E61 ridge null, E27/E30): the factor is selected in CV but does not
#   survive to out-of-sample rank corr. The n_factors kept is the diagnostic.
#
# Usage:  bash experiments/e62_sur_trees_crypto.sh   [DRY_RUN=1] [NO_MAIL=1]
#         ONLY="sur_lgb_reg_raw stk_lgb_reg_raw"   SKIP=...   SHARD=1 OF=4
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E62}"
E62_WINDOW="${E62_WINDOW:-130}"
E62_REFIT="${E62_REFIT:-month}"
E62_N_ITER="${E62_N_ITER:-50}"
E62_N_SPLITS="${E62_N_SPLITS:-5}"
E62_N_JOBS="${E62_N_JOBS:-2}"
E62_SEARCH_JOBS="${E62_SEARCH_JOBS:-16}"
E62_SEED="${E62_SEED:-42}"

DSNAME=(raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp)
DSVER=("crypto/v2.0" "crypto/v2.0-tenc" "crypto/v2.0-interact" "crypto/v2.0-fenccausal" \
       "crypto/v2.0-cumlag3" "crypto/v2.0-tenc-onehot" "crypto/v2.0-onehot-rollsel" \
       "crypto/v2.0-tenc-onehot-spread-interact" "crypto/v2.0-tenc-onehot-rollsel-spread-interact")

# family -> (sur model, stacked-reg model, stacked-clf model)
FAM=(lgb rf catboost)
SUR=(sur_lgb sur_rf sur_catboost)
STK_REG=(lgb_ktz rf_ktz catboost_ktz)
STK_CLF=(lgb_ktz rf_ktz catboost)

ARM=(); DATA=(); MODEL=(); TASK=()
aadd() { ARM+=("$1"); MODEL+=("$2"); TASK+=("$3"); DATA+=("$4"); }
for di in "${!DSNAME[@]}"; do
    d="${DSNAME[$di]}"; v="${DSVER[$di]}"
    for fi in "${!FAM[@]}"; do
        f="${FAM[$fi]}"
        aadd "sur_${f}_reg_${d}" "${SUR[$fi]}"     regression     "$v"
        aadd "stk_${f}_reg_${d}" "${STK_REG[$fi]}" regression     "$v"
        aadd "sur_${f}_clf_${d}" "${SUR[$fi]}"     classification "$v"
        aadd "stk_${f}_clf_${d}" "${STK_CLF[$fi]}" classification "$v"
    done
done

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: sur_{lgb,rf,catboost} vs stacked, reg vs clf, x 9 datasets
  approach  cross-symbol factor-augmented features, k=0 = plain-tree control
  budget    --n-iter ${E62_N_ITER}, ${E62_N_SPLITS} folds, rolling ${E62_WINDOW}/${E62_REFIT}, seed ${E62_SEED}

Where to look while it runs:
  runner    experiments/e62_sur_trees_crypto.sh
  progress  logs/e62_*.log
  results   results/metrics/crypto/{sur_lgb,sur_rf,sur_catboost,lgb_ktz,rf_ktz,catboost_ktz,catboost}/e62_*.json
  report    logs/e62_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e62_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: sur trees vs stacked, reg vs clf, x9 datasets | ${total} arms"
echo "n_iter=${E62_N_ITER} folds=${E62_N_SPLITS} window=${E62_WINDOW}/${E62_REFIT} seed=${E62_SEED} shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="e62_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${MODEL[$i]}" --task "${TASK[$i]}" --metric corr
         --cv-scheme rolling --train-window "${E62_WINDOW}" --refit-every "${E62_REFIT}"
         --data-version "${DATA[$i]}"
         --n-iter "${E62_N_ITER}" --n-splits "${E62_N_SPLITS}"
         --n-jobs "${E62_N_JOBS}" --search-jobs "${E62_SEARCH_JOBS}"
         --era-step 1 --seed "${E62_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-24s %-14s %-14s %s\n' "$index" "$total" "$run_name" "${MODEL[$i]}" "${TASK[$i]}" "${DATA[$i]}"; continue; fi
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

REPORT="logs/e62_report.txt"
{
echo "${EXPERIMENT} -- sur trees (factor-augmented) vs stacked, reg vs clf"
echo "  runner   experiments/e62_sur_trees_crypto.sh"
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
