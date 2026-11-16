#!/usr/bin/env bash
#
# E119 -- E117's question asked of the best standalone MLP: how much HISTORY
#         should each refit of IA_01 use, from one week to six months?
#
# The model is the payout-best standalone MLP of the 2,515 20-day MLP runs on
# the shared 510-day window: IA_01, run mlp/a2_mlp_fenccausal -- sklearn's
# MLPRegressor (relu, adam, early stopping off, max_iter 200) through the repo's
# `mlp` wrapper, on crypto/v2.0-fenccausal (22 raw features + their 22 causal
# per-symbol means, 44 columns). Payout 0.0987, CORR 0.1617, MMC 0.0825.
# Everything about it is held fixed -- data, features, the 50-draw random search
# with 5 inner splits, 130-period refit cadence, embargo 20, seed 42 -- and ONLY
# the rolling training window moves, through the same seven windows as E117:
#
#     1 week  5    2 weeks 10    3 weeks 15    4 weeks 20
#     1 month 22   3 months 65   6 months 130 (IA_01's own window)
#
# ---------------------------------------------------------------------------
# The hyperparameters are SEARCHED FROM SCRATCH at every refit, for every window
# ---------------------------------------------------------------------------
#
# As in E117: a purged inner fold needs train + embargo 20 + validation periods
# inside the searched block, which a short window cannot hold. Every window
# below 130 is therefore searched with --tune-window 130 -- on the 130 periods
# ending where the window ends, with rolling inner folds that each TRAIN ON THE
# WINDOW'S LENGTH, so each candidate is scored exactly as it will be used -- and
# the winner is refitted on the window alone. Every arm picks its architecture,
# alpha, learning rate and batch size on the same validation slices.
#
# Nothing has to be rescaled to the window, unlike glmnet's lambda x N: the MLP
# grid is fixed (MLPWrapper.param_grid, no fit_grid), and sklearn's L2 term is
# divided by the rows in the batch, so alpha means the same thing at any size.
#
# ---------------------------------------------------------------------------
# What the literal recipe does to a short window, and what is NOT varied
# ---------------------------------------------------------------------------
#
# The batch grid starts at 1,024 rows (measured, see MLPWrapper), and sklearn
# clips a batch to the training rows. A 1-week window is 1,500 rows, so its fit
# takes one or two gradient steps per epoch where the 6-month window takes 19+,
# under the same 200-epoch ceiling and the same stop-on-flat-training-loss rule.
# That is part of what "the same recipe on less history" means, and it is said
# here so a short window's loss is not read as data alone.
#
# The REFIT CADENCE stays at 130, so a 1-week window is fitted once every six
# months: how much history, not freshness (E116's axis). Expect the short
# windows to lose: 5 periods at ~95% target overlap is about one effective time
# observation, against an MLP of 1,000-20,000 weights.
#
# The 130 arm needs no tuning block and must reproduce a2_mlp_fenccausal: the
# report prints the per-period difference. BLAS threading can move an MLP fit by
# rounding, so the arms pin every BLAS pool to E119_THREADS (default 1) and get
# their parallelism from --search-jobs instead.
#
# ---------------------------------------------------------------------------
# E119_TARGET=60 -- the same sweep FITTED on the 60-day label (study E125)
# ---------------------------------------------------------------------------
#
# Identical arms, search and --tune-window 130, but every fit and every inner
# fold uses target_binned_return_60, with the embargo it needs (60, not 20).
# Runs are named e125_w<N>. Their own metrics score them on the 60-day
# label, which nobody pays for; the report RE-SCORES every arm of both halves on
# target_binned_return_20 through src.analysis.cross_target, the scorer
# CURUPIRA_05_60 and E60 were graded with, and puts the 14 rows on the 470
# dates they share (the 60-day label is NaN over the last 40). The 130 arm
# must reproduce mlp/e60_mlp_fenccausal, IA_01's setup fitted on 60 days.
#
# With embargo 60 a 130-period block holds only the last two of the five inner
# folds, and a 65-period fold is cut short by the start of the block. That is
# what E60 and E122 searched with too; it is said here so a thin search is not
# mistaken for a bad label.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash experiments/e119_ia01_train_window_crypto.sh            # all arms, then the report
#   ONLY=e119_w5 bash experiments/e119_ia01_train_window_crypto.sh   # one arm, no report (cluster)
#   REPORT_ONLY=1 bash experiments/e119_ia01_train_window_crypto.sh  # report + mail once all are done
#   E119_SCOUT=1 ONLY=e119_w5 bash ...                           # --era-step 4 wiring check, e119scout_
#   E119_TARGET=60 [ONLY=e119_w5] bash ...                       # fit on 60 days: e125_w<N> (E125)
#   DRY_RUN=1 bash experiments/e119_ia01_train_window_crypto.sh
#   NO_MAIL=1 ...
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/danielco/miniconda3/bin:${PATH}"   # bare python3 can be the 3.8 sandbox
cd "$REPO"
mkdir -p logs

E119_SEARCH_JOBS="${E119_SEARCH_JOBS:-8}"
E119_THREADS="${E119_THREADS:-1}"
export OMP_NUM_THREADS="${E119_THREADS}" MKL_NUM_THREADS="${E119_THREADS}" \
       OPENBLAS_NUM_THREADS="${E119_THREADS}" NUMEXPR_NUM_THREADS="${E119_THREADS}"
DATA="crypto/v2.0-fenccausal"
TUNE_WINDOW="${E119_TUNE_WINDOW:-130}"
E119_TARGET="${E119_TARGET:-20}"
case "${E119_TARGET}" in
    20) TARGET_COL="target_binned_return_20"; EMBARGO=20; BASE="e119";    TARGET_ARGS=() ;;
    60) TARGET_COL="target_binned_return_60"; EMBARGO=60; BASE="e125"
        TARGET_ARGS=(--target "${TARGET_COL}") ;;
    *)  echo "E119_TARGET must be 20 or 60, got '${E119_TARGET}'" >&2; exit 1 ;;
esac

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add e119_w5      5 "1 week"
add e119_w10    10 "2 weeks"
add e119_w15    15 "3 weeks"
add e119_w20    20 "4 weeks"
add e119_w22    22 "1 month"
add e119_w65    65 "3 months"
add e119_w130  130 "6 months (control = a2_mlp_fenccausal, IA_01)"
[ "${E119_TARGET}" = "60" ] && WHY[6]="6 months (control = e60_mlp_fenccausal, IA_01's setup on 60d)"

REPORT="logs/e119_report.txt"
notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}

if [ "${REPORT_ONLY:-0}" = "1" ]; then
    python3 -u -m src.analysis.e119_report > "${REPORT}" 2>&1 || echo "(report generation failed)" >> "${REPORT}"
    cat "${REPORT}"
    notify "E119 IA_01 training window -- report"
    exit 0
fi

# Gate: 44 features, the fitted label, and the 300-row cross-section the batch
# arithmetic above assumes.
python3 - "${DATA}" "${TARGET_COL}" <<'PY' || exit 1
import sys
import pandas as pd, pyarrow.parquet as pq
path = "data/%s/train.parquet" % sys.argv[1]
schema = pq.ParquetFile(path).schema.names
names = [c for c in schema if c.startswith("feature_")]
if len(names) != 44:
    sys.exit("expected 44 features in %s, found %d" % (path, len(names)))
if sys.argv[2] not in schema:
    sys.exit("%s has no %s" % (path, sys.argv[2]))
per = pd.read_parquet(path, columns=["date"]).groupby("date").size().iloc[-800:]
if per.min() != 300 or per.max() != 300:
    sys.exit("rows per period over the last 800 dates span %d..%d, not 300" % (per.min(), per.max()))
print("gate OK: 44 features; 300 rows on every one of the last 800 periods")
PY

single="${ONLY:-}"
total="${#NAMES[@]}"; index=0; failed=0; ran=0
started_all=$(date +%s)
for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${single}" ] && [ "${single}" != "${name}" ] && [ "${single}" != "${BASE}_w${window}" ] && continue
    run="${BASE}_w${window}"
    extra=()
    if [ "${E119_SCOUT:-0}" = "1" ]; then
        run="${BASE}scout_w${window}"; extra+=(--era-step 4)
    fi
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}"
         --refit-every 130 --embargo "${EMBARGO}" "${TARGET_ARGS[@]}"
         --n-splits 5 --n-iter 50 --seed 42 --val-frac 0.3
         --n-jobs 1 --search-jobs "${E119_SEARCH_JOBS}"
         --data-version "${DATA}" --run-name "${run}" "${extra[@]}")
    if [ "${window}" -lt "${TUNE_WINDOW}" ]; then
        cmd+=(--tune-window "${TUNE_WINDOW}")
        why="${why}, tuned on ${TUNE_WINDOW}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-14s %s\n    %s\n' "$index" "$total" "$run" "$why" "${cmd[*]}"
        continue
    fi
    ran=$((ran + 1)); started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run}.log" >/dev/null
    status="${PIPESTATUS[0]}"; elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        printf '[%d/%d] %-18s FAILED (exit %d) after %5ds  -- %s\n' "$index" "$total" "$run" "$status" "$elapsed" "$why"
    else
        printf '[%d/%d] %-18s done in %5ds  -- %s\n' "$index" "$total" "$run" "$elapsed" "$why"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && exit 0
if [ -n "${single}" ]; then                 # one arm of a spread batch: the report waits for all
    exit "$(( failed > 0 ? 1 : 0 ))"
fi
echo "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
python3 -u -m src.analysis.e119_report > "${REPORT}" 2>&1 || echo "(report generation failed)" >> "${REPORT}"
cat "${REPORT}"
if [ "${failed}" -eq 0 ]; then notify "E119 IA_01 training window, fit ${E119_TARGET}d -- OK (${ran}/${total})"
else notify "E119 IA_01 training window, fit ${E119_TARGET}d -- ${failed} arm(s) failed"; fi
