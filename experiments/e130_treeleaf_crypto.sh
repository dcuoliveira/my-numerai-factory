#!/usr/bin/env bash
#
# E130 -- do two of the book's best tree models carry structure that the best
#         ridge and the best MLP miss?
#
# The extra blocks come from src/data_mani/build_treeleaf.py: for every row, the
# distribution of its trees' leaf VALUES (mean, std, q10/25/50/75/90), binned to
# {0,.25,.5,.75,1}, from two sources re-run walk-forward and searched from scratch
# at every refit (fit 20, rolling 130 / refit 130 / embargo 20, seed 42):
#   feature_lfcat_*  catboost_ktz on fenccausal + E118's x29 block (e118_catboost_x29)
#   feature_lfrf_*   rf_ktz on allfeat (MA_ALLFEAT, e52_allfeat)
# Causal as in E129: a row's leaves come from its own walk-forward block's model,
# trained only on rows that had resolved, so no row is scored by a model that
# trained on it. Leaf IDs are not used; they change meaning at every refit.
#
# Arms -- every one a 50-draw purged search FROM SCRATCH, rolling 130 / refit 130:
#   e130_r_{ctl,cat,rf,both}  ridge_glmnet, allfeat_pp - tenc + x29 (e118_ridge_x29's protocol)
#   e130_m_{ctl,cat,rf,both}  mlp, fenccausal (IA_01's protocol)
#   e130_m_{ctl,both}_s43     the MLP control and +both again at seed 43 (noise floor)
# Each base dataset carries both blocks; the arms differ only in --feature-exclude.
#
# Measured against: each arm vs its same-seed in-study control, paired Newey-West
# lag 20 on the 20-day payout, rebuilt per period (src.analysis.e130_report). The
# controls should reproduce e118_ridge_x29 exactly and land near e15_mlp_fenccausal
# (IA_01's search does not reproduce, E129). A result is |t| >= 2 on payout.
#
#   bash experiments/e130_treeleaf_crypto.sh          # everything, sequential
#   STAGE=leaf_cat | leaf_rf | merge | arms | report  one stage only
#   ONLY="r_ctl m_both" / SKIP=... / DRY_RUN=1 / NO_MAIL=1
#   SMOKE=1     arms only: 2-draw search, 2 splits, run names e130scout_*
#   E130_SJ / E130_NJ            --search-jobs / --n-jobs for the arms (product = cores)
#   E130_LEAF_SJ / E130_LEAF_NJ  the same for the two leaf fits
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="/home/danielco/miniconda3/bin:${PATH}"
EXPERIMENT="E130"
STAGE="${STAGE:-all}"
SJ="${E130_SJ:-8}"; NJ="${E130_NJ:-1}"
LSJ="${E130_LEAF_SJ:-4}"; LNJ="${E130_LEAF_NJ:-2}"
mkdir -p logs

RIDGE_DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx-treeleaf"
MLP_DATA="crypto/v2.0-fenccausal-treeleaf"
X29='^feature_sym_tenc$|^feature_x(?!29_)\d\d_'     # e118_ridge_x29's exclude

declare -a ARM MODEL DATA SEED EXCL
add() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SEED+=("$4"); EXCL+=("$5"); }
add r_ctl   ridge_glmnet "$RIDGE_DATA" 42 "${X29}|^feature_lf"
add r_cat   ridge_glmnet "$RIDGE_DATA" 42 "${X29}|^feature_lfrf_"
add r_rf    ridge_glmnet "$RIDGE_DATA" 42 "${X29}|^feature_lfcat_"
add r_both  ridge_glmnet "$RIDGE_DATA" 42 "${X29}"
add m_ctl   mlp          "$MLP_DATA"   42 "^feature_lf"
add m_cat   mlp          "$MLP_DATA"   42 "^feature_lfrf_"
add m_rf    mlp          "$MLP_DATA"   42 "^feature_lfcat_"
add m_both  mlp          "$MLP_DATA"   42 ""
add m_ctl_s43  mlp       "$MLP_DATA"   43 "^feature_lf"
add m_both_s43 mlp       "$MLP_DATA"   43 ""
total=${#ARM[@]}

mail() {
    [ "${DRY_RUN:-0}" = "1" ] || [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2])" \
        "$1" "$2" || echo "note: email failed, run continues"
}

run_logged() {   # name, then the command
    local name="$1"; shift
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "${*}"; return 0; fi
    local started; started=$(date +%s)
    echo "=== ${name}  $(date '+%F %T') on $(hostname)"
    "$@" > "logs/${name}.log" 2>&1
    local status=$? el=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        echo ">>> ${name} FAILED (exit ${status}) after $((el / 60))m -- see logs/${name}.log"
    else
        echo ">>> ${name} done in $((el / 60))m$((el % 60))s"
    fi
    return "$status"
}

leaf() {
    run_logged "e130_leaf_$1" python3 -u -m src.data_mani.build_treeleaf --source "$1" \
        --search-jobs "${LSJ}" --n-jobs "${LNJ}"
}

merge() {
    run_logged e130_merge python3 -u -m src.data_mani.build_treeleaf --merge || return 1
    [ "${DRY_RUN:-0}" = "1" ] && return 0
    # The two variants must load as a drop-in: neutral-filled, leaf columns present.
    python3 - "$RIDGE_DATA" "$MLP_DATA" <<'X'
import sys
from src.data_mani.utils import load_tournament_data
for version in sys.argv[1:]:
    df, feats = load_tournament_data("crypto", version=version)
    leaf = [f for f in feats if f.startswith("feature_lf") and not f.endswith("_missing")]
    flags = [f for f in feats if f.startswith("feature_lf") and f.endswith("_missing")]
    assert len(leaf) == 14 and len(flags) == 2 and not df[feats].isna().any().any(), (version, len(leaf), flags)
    # Where the arms read: from the first training window (130 + 20 embargo
    # periods before the 30% validation start) to the end.
    dates = sorted(df["date"].unique()); first = dates[len(dates) - int(round(len(dates) * 0.3)) - 150]
    read = df["date"] >= first
    print("%s: %d features (%d leaf + %d indicators), loads clean; missing share overall %s, "
          "inside the arms' windows %s" % (version, len(feats), len(leaf), len(flags),
          {f: round(float(df[f].mean()), 4) for f in flags},
          {f: round(float(df.loc[read, f].mean()), 4) for f in flags}))
X
}

arms() {
    local tag="e130" ran=0 failed=0
    local -a iter=(--n-iter 50 --n-splits 5) FAILED_ARMS=()
    if [ "${SMOKE:-0}" = "1" ]; then tag="e130scout"; iter=(--n-iter 2 --n-splits 2); fi
    for i in "${!ARM[@]}"; do
        local arm="${ARM[$i]}"
        [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
        [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
        local cmd=(python3 -u -m src.run_forecast crypto "${MODEL[$i]}" --task regression --metric corr
                   --cv-scheme rolling --train-window 130 --refit-every 130 --era-step 1
                   --inner-cv purged --seed "${SEED[$i]}" --search-jobs "${SJ}" --n-jobs "${NJ}"
                   "${iter[@]}" --data-version "${DATA[$i]}" --run-name "${tag}_${arm}")
        [ -n "${EXCL[$i]}" ] && cmd+=(--feature-exclude "${EXCL[$i]}")
        ran=$((ran + 1))
        run_logged "${tag}_${arm}" "${cmd[@]}" || { failed=$((failed + 1)); FAILED_ARMS+=("${tag}_${arm}"); }
    done
    echo "arms: ran ${ran}, ${failed} failed${FAILED_ARMS:+: ${FAILED_ARMS[*]}}"
    [ "${failed}" -eq 0 ]
}

report() {
    [ "${DRY_RUN:-0}" = "1" ] && { echo "python3 -m src.analysis.e130_report"; return 0; }
    python3 -m src.analysis.e130_report | tee logs/e130_report.txt
}

started_all=$(date +%s); status=0
case "$STAGE" in
    leaf_cat) leaf cat || status=1 ;;
    leaf_rf)  leaf rf || status=1 ;;
    merge)    merge || status=1 ;;
    arms)     arms || status=1 ;;
    report)   report || status=1 ;;
    all)
        mail "[crypto] ${EXPERIMENT} started" "${EXPERIMENT} tree-leaf features started $(date '+%F %T') on $(hostname)
  stages    2 leaf fits (catboost fenccausal+x29, rf allfeat) -> merge -> ${total} arms
  runner    experiments/e130_treeleaf_crypto.sh
  progress  logs/e130_*.log
  results   results/metrics/crypto/{ridge_glmnet,mlp}/e130_*.json
  report    python3 -m src.analysis.e130_report"
        { leaf cat && leaf rf && merge && arms; } || status=1
        report || status=1
        mail "[crypto] ${EXPERIMENT} finished (status ${status})" "$(cat logs/e130_report.txt 2>/dev/null)" ;;
    *) echo "unknown STAGE=${STAGE}"; exit 2 ;;
esac
echo "${EXPERIMENT} ${STAGE}: status ${status} in $(( ($(date +%s) - started_all) / 60 )) min"
exit "$status"
