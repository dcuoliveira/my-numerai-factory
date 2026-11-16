#!/usr/bin/env bash
#
# Experiment E63: do the literature crypto-momentum features help?
#
# E63 is the reference handle; every artefact is e63_*.
#
# THE QUESTION
#   The 22 shipped features hide the raw price path (they are rank-binned). We
#   rebuilt 17 momentum features straight from spot prices (src.data_mani.
#   build_momentum -> data/crypto/v2.0-litmom), each drawn from a crypto momentum
#   paper (Liu & Tsyvinski 2021; Grobys & Sapkota 2019; Tzouvanas et al. 2019;
#   Li et al. 2021; Liu, Tsyvinski & Wu 2022; Begusic & Kostanjcar 2019), put on
#   Numerai's exact {0,.25,.5,.75,1} scale and merged POINT-IN-TIME (row date D
#   uses spot closes <= D-1, the last completed daily bar known at submission).
#   Does adding that block to the 22 raw features beat raw alone on numerai_corr?
#
# WHAT THIS RUNS
#   Two models (lgb, ridge_glmnet) x three feature views, matched protocol
#   (rolling 130/130, embargo auto=20, 5 folds, 50 draws, seed 42) so base pairs
#   cleanly against base+mom on the same model, and mom-only shows standalone
#   signal:
#     base     crypto/v2.0        the 22 shipped features           (the baseline)
#     mom      crypto/v2.0-litmom    22 + 17 momentum                  (the test)
#     momonly  crypto/v2.0-litmom    only ^feature_mom_ (10)           (diagnostic)
#
# WHAT WOULD COUNT AS A RESULT
#   base+mom beating its matched base on numerai_corr at paired Newey-West
#   |t|>=2 (lag 20) on either model -- i.e. the momentum block adds ranking
#   information the binned features do not already carry. momonly is not expected
#   to beat base; it only says whether the block stands up at all. Report with
#   src.analysis.report_html/report_pdf, base vs base+mom, paired, same seed.
#
# Usage:  bash experiments/e63_momentum_features_crypto.sh  [DRY_RUN=1] [NO_MAIL=1]
#         [SMOKE=1]  ONLY="lgb_mom glm_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
PYTHON="${PYTHON:-python3}"

EXPERIMENT="${EXPERIMENT:-E63}"
E63_WINDOW="${E63_WINDOW:-130}"
E63_REFIT="${E63_REFIT:-month}"
E63_N_ITER="${E63_N_ITER:-50}"
E63_N_SPLITS="${E63_N_SPLITS:-5}"
E63_N_JOBS="${E63_N_JOBS:-2}"
E63_SEARCH_JOBS="${E63_SEARCH_JOBS:-16}"
E63_SEED="${E63_SEED:-42}"
E63_ERA_STEP="${E63_ERA_STEP:-1}"

# SMOKE: a fast wiring check, not a result -- downsample hard and shrink search.
if [ "${SMOKE:-0}" = "1" ]; then
    E63_WINDOW=520; E63_REFIT=6month; E63_N_ITER=3; E63_N_SPLITS=3; E63_ERA_STEP=5
fi

# Build the momentum variant if it is not already on disk (idempotent).
if [ ! -f "data/crypto/v2.0-litmom/train.parquet" ]; then
    echo "building data/crypto/v2.0-litmom ..."
    [ "${DRY_RUN:-0}" = "1" ] || "${PYTHON}" -m src.data_mani.build_momentum
fi

#            arm         model         view      data-version        include
ARM=(); MODEL=(); DATA=(); INCLUDE=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); INCLUDE+=("$4"); }
aadd lgb_base    lgb          "crypto/v2.0"     ""
aadd lgb_mom     lgb          "crypto/v2.0-litmom" ""
aadd lgb_momonly lgb          "crypto/v2.0-litmom" "^feature_mom_"
aadd glm_base    ridge_glmnet "crypto/v2.0"     ""
aadd glm_mom     ridge_glmnet "crypto/v2.0-litmom" ""
aadd glm_momonly ridge_glmnet "crypto/v2.0-litmom" "^feature_mom_"

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: base vs base+momentum vs momentum-only, lgb & ridge_glmnet
  data      crypto/v2.0 vs crypto/v2.0-litmom (22 + 17 literature momentum proxies)
  budget    --n-iter ${E63_N_ITER}, ${E63_N_SPLITS} folds, rolling ${E63_WINDOW}/${E63_REFIT}, seed ${E63_SEED}

Where to look while it runs:
  runner    experiments/e63_momentum_features_crypto.sh
  progress  logs/e63_*.log
  results   results/metrics/crypto/{lgb,ridge_glmnet}/e63_*.json
  report    logs/e63_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e63_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: crypto momentum features vs raw | ${total} arms"
echo "n_iter=${E63_N_ITER} folds=${E63_N_SPLITS} window=${E63_WINDOW}/${E63_REFIT} seed=${E63_SEED} shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="e63_${arm}"
    cmd=("${PYTHON}" -u -m src.run_forecast crypto "${MODEL[$i]}" --task regression --metric corr
         --cv-scheme rolling --train-window "${E63_WINDOW}" --refit-every "${E63_REFIT}"
         --data-version "${DATA[$i]}"
         --n-iter "${E63_N_ITER}" --n-splits "${E63_N_SPLITS}"
         --n-jobs "${E63_N_JOBS}" --search-jobs "${E63_SEARCH_JOBS}"
         --era-step "${E63_ERA_STEP}" --seed "${E63_SEED}" --run-name "${run_name}")
    [ -n "${INCLUDE[$i]}" ] && cmd+=(--feature-include "${INCLUDE[$i]}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-16s %-14s %-16s %s\n' "$index" "$total" "$run_name" "${MODEL[$i]}" "${DATA[$i]}" "${INCLUDE[$i]:-(all)}"; continue; fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${MODEL[$i]}  ${DATA[$i]}  ${INCLUDE[$i]:-(all)}  $(date '+%H:%M:%S')"
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

REPORT="logs/e63_report.txt"
{
echo "${EXPERIMENT} -- crypto momentum features (base vs base+mom vs mom-only)"
echo "  results  results/metrics/crypto/{lgb,ridge_glmnet}/e63_*.json"
echo "  runner   experiments/e63_momentum_features_crypto.sh"
echo "  report   python3 -m src.analysis.report_html --model lgb --runs e63_lgb_base e63_lgb_mom"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
} | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
sys.argv[1], open(sys.argv[2]).read())" "[crypto] ${EXPERIMENT} done" "${REPORT}" \
      || echo "note: completion email failed"
fi
