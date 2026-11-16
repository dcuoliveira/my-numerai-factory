#!/usr/bin/env bash
#
# E117 -- how much HISTORY should each refit use? The training window of the
#         best single ridge, from one week to six months.
#
# The model is the payout-best single ridge found across 360 clean runs:
# ridge_glmnet on allfeat_pp with the one tenc column dropped (run
# w6r6_lobo_tenc, 811 features, payout 0.08779). Everything about it is held
# fixed -- data, features, 130-period refit cadence, embargo 20, seed -- and
# ONLY the rolling training window moves.
#
# ---------------------------------------------------------------------------
# The windows, in PERIODS. The panel is weekday-only: 5.01 periods per week,
# 21.8 per month, measured on the last 260 dates.
# ---------------------------------------------------------------------------
#
#     1 week  5    2 weeks 10    3 weeks 15    4 weeks 20
#     1 month 22   3 months 65   6 months 130 (the deployed window)
#
# 4 weeks and 1 month are two periods apart and are both run because both were
# asked for; expect them to be indistinguishable.
#
# ---------------------------------------------------------------------------
# Lambda is SEARCHED FROM SCRATCH at every refit, for every window
# ---------------------------------------------------------------------------
#
# The obstacle: a purged inner fold needs training periods + the 20-period
# embargo + validation periods INSIDE the block it searches. purged_period_splits
# RAISES "no usable CV folds" for every window of 22 periods or fewer -- a month
# does not contain a held-out fold. So the windows cannot be tuned on themselves.
#
# The fix is --tune-window 130: at each refit the search runs on the 130 periods
# ending at the SAME date the training window ends, with rolling inner folds that
# each TRAIN ON THE WINDOW'S LENGTH -- 5 periods for the 1-week arm -- so every
# candidate lambda is scored exactly as it will be used. The winner is then
# refitted on the training window alone. Every arm therefore selects lambda on
# the SAME four validation slices; only how much history each fold (and the
# final fit) sees differs. The glmnet lambda path is built from the TRAINING
# window, because its alphas are lambda x N and a path built on the 39,000-row
# tuning block would be 26x too strong for a 1,500-row fit.
#
# The 130 arm needs no tuning block: it searches its own window exactly as
# production does, and reproduces w6r6_lobo_tenc.
#
# An earlier version of this study pinned lambda instead; it was superseded at
# the user's request before reporting. Its artifacts are kept out of results/.
#
# ---------------------------------------------------------------------------
# What this does NOT vary, and why that matters for reading it
# ---------------------------------------------------------------------------
#
# The REFIT CADENCE stays at 130. A 1-week window is therefore fitted once every
# six months and predicts the next 130 periods from one week of data. That is
# the literal question -- how much history should a refit use, at the cadence
# the account runs -- and it is NOT the freshness question. Short windows only
# pay through freshness, which needs a short cadence too; that is E116's axis
# (cadence at fixed window), and the coupled design is the natural follow-up.
#
# Expect the short windows to lose badly. 5 periods at ~95% target overlap is
# about ONE effective time observation, 811 columns against 1,500 rows.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash experiments/e117_train_window_crypto.sh
#   DRY_RUN=1 bash experiments/e117_train_window_crypto.sh
#   ONLY=e117_w5 bash experiments/e117_train_window_crypto.sh
#   NO_MAIL=1 bash experiments/e117_train_window_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

TW_SEARCH_JOBS="${TW_SEARCH_JOBS:-8}"
TW_N_JOBS="${TW_N_JOBS:-1}"
TW_ROWS_PER_PERIOD="${TW_ROWS_PER_PERIOD:-300}"
DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact"
EXCLUDE='^feature_sym_tenc$'

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add e117_w5      5 "1 week"
add e117_w10    10 "2 weeks"
add e117_w15    15 "3 weeks"
add e117_w20    20 "4 weeks"
add e117_w22    22 "1 month"
add e117_w65    65 "3 months"
add e117_w130  130 "6 months (control = w6r6_lobo_tenc)"
TUNE_WINDOW="${TW_TUNE_WINDOW:-130}"

REPORT="logs/e117_report.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E117 training window -- ABORTED (exit ${code})"
      fi' EXIT

# Gate: the dataset still has 812 columns, the exclusion still removes exactly
# one, and the rows-per-period assumption behind alpha_W still holds.
python3 - "${DATA}" "${TW_ROWS_PER_PERIOD}" <<'PY' || exit 1
import re, sys
import pandas as pd, pyarrow.parquet as pq
data, rows = sys.argv[1], int(sys.argv[2])
path = "data/%s/train.parquet" % data
names = [c for c in pq.ParquetFile(path).schema.names if c.startswith("feature_")]
kept = [c for c in names if not re.search(r"^feature_sym_tenc$", c)]
if (len(names), len(kept)) != (812, 811):
    sys.exit("expected 812 -> 811 features, found %d -> %d" % (len(names), len(kept)))
d = pd.read_parquet(path, columns=["date"]).reset_index()
per = d.groupby("date").size()
recent = per.iloc[-800:]
if recent.min() != rows or recent.max() != rows:
    sys.exit("rows per period over the last 800 dates span %d..%d, not %d -- the "
             "study assumes a fixed 300-row cross-section" % (recent.min(), recent.max(), rows))
print("gate OK: 812 -> 811 features; %d rows on every one of the last 800 periods"
      % rows)
PY

total="${#NAMES[@]}"; index=0; failed=0; ran=0
started_all=$(date +%s)
say "=============================================================="
say "E117 training window, 1 week to 6 months | ${total} arms"
say "ridge_glmnet, allfeat_pp minus tenc (811 feat), refit 130, embargo 20"
say "lambda SEARCHED at every refit: tuned on ${TUNE_WINDOW} periods, folds train on the window"
say "search-jobs=${TW_SEARCH_JOBS}   started $(date -Is)"
say "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    cmd=(python3 -u -m src.run_forecast crypto ridge_glmnet
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}"
         --refit-every 130 --embargo 20
         --n-splits 5 --n-iter 50 --seed 42 --val-frac 0.3
         --n-jobs "${TW_N_JOBS}" --search-jobs "${TW_SEARCH_JOBS}"
         --data-version "${DATA}" --feature-exclude "${EXCLUDE}"
         --run-name "${name}")
    if [ "${window}" -lt "${TUNE_WINDOW}" ]; then
        cmd+=(--tune-window "${TUNE_WINDOW}")
        why="${why}, tuned on ${TUNE_WINDOW}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s %s\n' "$index" "$total" "$name" "$why"
        continue
    fi
    ran=$((ran + 1)); started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log" >/dev/null
    status="${PIPESTATUS[0]}"; elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%d/%d] %-18s FAILED (exit %d) after %5ds  -- %s' \
             "$index" "$total" "$name" "$status" "$elapsed" "$why")"
    else
        say "$(printf '[%d/%d] %-18s done in %5ds  -- %s' "$index" "$total" "$name" "$elapsed" "$why")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }
say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""
python3 -u -m src.analysis.e117_report >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"
trap - EXIT
if [ "${failed}" -eq 0 ]; then notify "E117 training window -- OK (${ran}/${total})"
else notify "E117 training window -- ${failed} arm(s) failed"; fi
echo
echo "report: ${REPORT}"
