#!/usr/bin/env bash
#
# E129 -- does the 6-month RF's leaf structure help the models inside the two
#         meta-residualised/neural slots CU_SA_IA_BO_MA_R_01 and CU_SA_IA_BO_02?
#
# The extra block is feature_rfleaf_* (src/data_mani/build_rfleaf.py): for every
# row, the distribution of the 100 tree leaf VALUES of E128's w130 random forest
# (rf_ktz, tenc-onehot, fit 20, rolling 130 / refit 130 / embargo 20) -- mean, std,
# q10/25/50/75/90 -- plus the forest's prediction residualised on the 5-day-lagged
# meta model, all binned to {0,.25,.5,.75,1}. Causal: a row's leaves come from the
# forest of its own walk-forward block, trained only on rows that resolved before
# it; no row is ever scored by a forest that trained on it. Leaf IDS are not used:
# the forest is refit every 130 periods, so an ID means something different in
# every refit, while a leaf's VALUE does not.
#
# Arms: each of the 8 base models inside the two slots, run on its EXACT recorded
# protocol twice -- control (its own dataset) and +rfleaf (the same dataset plus the
# 8 columns). The two ridge block-blends (CURUPIRA_05, CURUPIRA_05_60) get the
# leaves as a SIXTH block member, a ridge_glmnet on feature_rfleaf_* alone, at the
# blend's own label; their 5 stored members are unchanged, so they need no control.
#
#   e129_ctl_<m> / e129_lf_<m>   m in ia01 ia02 saci04 mafenc maallfeat c04
#   e129_lf_c05blk  (t20)        e129_lf_c0560blk (t60)
#
# Measured against: each arm vs its control, paired Newey-West lag 20 (60 for the
# t60 block), and the two SLOTS rebuilt by src.analysis.e129_report from the
# controls vs from the +rfleaf arms with one assembly, which first has to reproduce
# the registry's stored slot series. A result is a slot or member beating its
# control on payout at paired |t| >= 2.
#
#   bash experiments/e129_rfleaf_members_crypto.sh            # all 14, sequential
#   ONLY="ctl_ia01 lf_ia01" / SKIP=... / DRY_RUN=1 / NO_MAIL=1
#   E129_SJ / E129_NJ   --search-jobs / --n-jobs (size to the slot: product = cores)
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="/home/danielco/miniconda3/bin:${PATH}"
EXPERIMENT="E129"
SJ="${E129_SJ:-3}"; NJ="${E129_NJ:-1}"
mkdir -p logs

COMMON=(--task regression --metric corr --cv-scheme rolling --train-window 130 --refit-every 130
        --era-step 1 --seed 42 --search-jobs "${SJ}" --n-jobs "${NJ}")
IA02_FIXED='{"mlp__activation": "relu", "mlp__alpha": 10, "mlp__batch_size": 1024, "mlp__hidden_layer_sizes": [128, 64], "mlp__learning_rate_init": 0.001}'
SACI04_FIXED='{"colsample_bytree": 0.070213, "learning_rate": 0.002572, "max_depth": 8, "min_child_samples": 313, "n_estimators": 4327, "num_leaves": 255, "reg_lambda": 1.757511, "subsample": 0.816153, "subsample_freq": 1}'

declare -a ARM MODEL DATA EXTRA
add() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); EXTRA+=("$4"); }
for kind in ctl lf; do
    sfx=""; [ "$kind" = "lf" ] && sfx="-rfleaf"
    add "${kind}_ia01"      mlp          "crypto/v2.0-fenccausal${sfx}"                          "--n-iter 50 --n-splits 5"
    add "${kind}_ia02"      mlp          "crypto/v2.0-fenccausal${sfx}"                          "--n-iter 1 --n-splits 2 --inner-cv purged --fixed-params IA02"
    add "${kind}_saci04"    lgb          "crypto/v2.0-asof${sfx}"                                     "--n-iter 50 --n-splits 2 --fixed-params SACI04"
    add "${kind}_mafenc"    rf_ktz       "crypto/v2.0-fenccausal${sfx}"                          "--n-iter 40 --n-splits 5 --inner-cv purged"
    add "${kind}_maallfeat" rf_ktz       "crypto/v2.0-tenc-onehot-spread-interact${sfx}"         "--n-iter 40 --n-splits 5 --inner-cv purged"
    add "${kind}_c04"       ridge_glmnet "crypto/v2.0-tenc-onehot-rollsel-spread-interact${sfx}" "--n-iter 50 --n-splits 5"
done
add "lf_c05blk"   ridge_glmnet "crypto/v2.0-asof-rfleaf" "--n-iter 50 --n-splits 5 --feature-include ^feature_rfleaf_"
add "lf_c0560blk" ridge_glmnet "crypto/v2.0-asof-rfleaf" "--n-iter 50 --n-splits 5 --feature-include ^feature_rfleaf_ --target target_binned_return_60"

total=${#ARM[@]}; ran=0; failed=0; FAILED_ARMS=()
started_all=$(date +%s)
if [ "${DRY_RUN:-0}" != "1" ] && [ "${NO_MAIL:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2])" \
      "[crypto] ${EXPERIMENT} started" "${EXPERIMENT} rfleaf members started $(date '+%F %T') on $(hostname)
  arms      ${total} (8 members x ctl/+rfleaf, 2 block members)
  runner    experiments/e129_rfleaf_members_crypto.sh
  progress  logs/e129_*.log
  results   results/metrics/crypto/*/e129_*.json
  report    python3 -m src.analysis.e129_report
  handle    E129; every artefact is e129_*" || echo "note: start email failed, run continues"
fi

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    run_name="e129_${arm}"
    read -r -a extra <<< "${EXTRA[$i]}"
    cmd=(python3 -u -m src.run_forecast crypto "${MODEL[$i]}" "${COMMON[@]}" --data-version "${DATA[$i]}" --run-name "${run_name}")
    for ((j = 0; j < ${#extra[@]}; j++)); do
        case "${extra[$j]}" in
            IA02) cmd+=("${IA02_FIXED}") ;;
            SACI04) cmd+=("${SACI04_FIXED}") ;;
            *) cmd+=("${extra[$j]}") ;;
        esac
    done
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%d] %s\n' "$((i + 1))" "$total" "${cmd[*]}"; continue
    fi
    ran=$((ran + 1)); started=$(date +%s)
    echo "=== [$((i + 1))/${total}] ${run_name}  $(date '+%H:%M:%S')"
    "${cmd[@]}" > "logs/${run_name}.log" 2>&1
    status=$?; el=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el / 60))m -- see logs/${run_name}.log"
    else
        echo ">>> ${run_name} done in $((el / 60))m$((el % 60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo "ran ${ran}, ${failed} failed in $(( ($(date +%s) - started_all) / 60 )) min${FAILED_ARMS:+: ${FAILED_ARMS[*]}}"
[ "${failed}" -eq 0 ]
