#!/usr/bin/env bash
#
# E126 -- the 60-DAY-label twin of E123. Same from-scratch-searched LightGBM
#         training-window sweep, but each arm TRAINS ON target_binned_return_60
#         instead of the 20-day label. The predictions are then RE-SCORED on the
#         payable 20-day target for consolidation (see src/analysis/e126_report.py),
#         because crypto pays on target_binned_return_20 -- the 60-day label is a
#         different signal, not a different payout.
#
# The question: does training a LightGBM on the smoother 60-day horizon and
# deploying it on the 20-day payable label beat training directly on 20 days,
# and does the answer depend on the training window? E123 is the 20-day control;
# this is the 60-day arm; src/analysis/e126_report.py puts both in ONE table, every row
# scored on the 20-day target.
#
# Base, identical to E123 except the target: model class lgb, the group-CONCAT
# subset of crypto/v2.0-onehot (748 feat), 130-period refit cadence, seed 42,
# windows 5/10/15/20/22/65/130, a --n-iter 60 --n-splits 5 from-scratch search.
#
# Two changes from E123, both consequences of the 60-day target:
#   --target target_binned_return_60   train on the 60-day label
#   --embargo auto                     derives 60, not 20 (a 60-day forward
#                                      target leaks three 20-day blocks otherwise)
# At embargo 60 a block under ~70 periods cannot host its own purged inner fold,
# so EVERY window < 130 uses --tune-window 130 (the E117/E122 mechanism); w130
# searches its own 130-window. The 60-day label is NaN over its last ~40 dates,
# so these arms score ~470 periods on their own label (and the consolidation
# inner-joins to whatever the 20-day target covers on the same dates).
#
# COST  a 60-day from-scratch search, same shape as E123 but embargo 60 -> fewer
# usable fold placements, similar per-arm wall-clock. One arm per cluster job.
#
# Usage (mirror E123):
#   ONLY=e126_w5 NO_MAIL=1 bash experiments/e126_train_window_lgb_search_t60_crypto.sh
#   bash experiments/e126_train_window_lgb_search_t60_crypto.sh          # serial, slow
#   REPORT_ONLY=1 bash experiments/e126_train_window_lgb_search_t60_crypto.sh  # -> src/analysis/e126_report.py
#   DRY_RUN=1 bash experiments/e126_train_window_lgb_search_t60_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT="${EXPERIMENT:-E126}"
DATA="crypto/v2.0-onehot"
INCLUDE='(^feature_market_cap_(avg|ewa)_)|(^feature_momentum_)|(^feature_volatility_)|(^feature_sym_)'
TARGET="${E126_TARGET:-target_binned_return_60}"
EMBARGO="${E126_EMBARGO:-auto}"
E126_N_ITER="${E126_N_ITER:-60}"
E126_N_SPLITS="${E126_N_SPLITS:-5}"
E126_SEARCH_JOBS="${E126_SEARCH_JOBS:-4}"
E126_N_JOBS="${E126_N_JOBS:-2}"
TUNE_WINDOW="${E126_TUNE_WINDOW:-130}"

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add e126_w5     5  "1 week"
add e126_w10   10  "2 weeks"
add e126_w15   15  "3 weeks"
add e126_w20   20  "4 weeks"
add e126_w22   22  "1 month"
add e126_w65   65  "3 months"
add e126_w130 130  "6 months"

REPORT="logs/e126_report.txt"

# The consolidation (cross-score to 20-day + merge with E123) is a separate
# step, because it needs both studies' forecasts and the 20-day scoring frame.
if [ "${REPORT_ONLY:-0}" = "1" ]; then
  "${PYTHON}" -u -m src.analysis.e126_report 2>&1 | tee "${REPORT}"
  if [ "${NO_MAIL:-0}" != "1" ]; then
    "${PYTHON}" - "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s report (fit20/fit60 x raw/resid, scored on 20d)" % sys.argv[2], open(sys.argv[1]).read())
print("emailed the report")
PY
  fi
  exit 0
fi

# Gate: same base as E123 (762 -> 748 features, 300 rows/period), plus the
# 60-day target column must exist and actually be NaN over its tail.
"${PYTHON}" - "${DATA}" "${INCLUDE}" "${TARGET}" <<'PY' || exit 1
import re, sys
import pandas as pd, pyarrow.parquet as pq
data, inc, tgt = sys.argv[1], sys.argv[2], sys.argv[3]
path = "data/%s/train.parquet" % data
schema = pq.ParquetFile(path).schema.names
names = [c for c in schema if c.startswith("feature_")]
kept = [c for c in names if re.search(inc, c)]
if (len(names), len(kept)) != (762, 748):
    sys.exit("expected 762 -> 748 features, found %d -> %d" % (len(names), len(kept)))
if tgt not in schema:
    sys.exit("target column %s not in %s" % (tgt, path))
d = pd.read_parquet(path, columns=["date"]).reset_index()
per = d.groupby("date").size().iloc[-800:]
if per.min() != 300 or per.max() != 300:
    sys.exit("rows/period over last 800 dates span %d..%d, not a fixed 300"
             % (per.min(), per.max()))
print("gate OK: 762 -> 748 features; 300 rows/period; target %s present" % tgt)
PY

total="${#NAMES[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ -z "${ONLY:-}" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: training window 1 week -> 6 months, booster searched from scratch per window
  target    ${TARGET} (embargo ${EMBARGO}); predictions RE-SCORED on the 20-day target to consolidate
  base      lgb, concat_trio feature set (748 feat on crypto/v2.0-onehot); refit 130, seed 42
  search    --n-iter ${E126_N_ITER} --n-splits ${E126_N_SPLITS}; tune-window ${TUNE_WINDOW} for windows < ${TUNE_WINDOW}
  twin of   E123 (same search, 20-day target) -- report: python3 -m src.analysis.e126_report (raw + resid, 28 rows)
  results   results/metrics/crypto/lgb/e126_*.json   report logs/e126_report.txt" \
|| echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: 60-day-label twin of E123 (from-scratch-searched lgb) | ${total} arms"
echo "target ${TARGET} embargo ${EMBARGO}; windows 5..130; refit 130; search n_iter=${E126_N_ITER} splits=${E126_N_SPLITS}"
echo "tune-window ${TUNE_WINDOW} for windows < ${TUNE_WINDOW}; search-jobs=${E126_SEARCH_JOBS} n-jobs=${E126_N_JOBS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
echo "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index+1))
    name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    cmd=("${PYTHON}" -u -m src.run_forecast crypto lgb
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}" --refit-every 130
         --target "${TARGET}" --embargo "${EMBARGO}"
         --n-iter "${E126_N_ITER}" --n-splits "${E126_N_SPLITS}"
         --search-jobs "${E126_SEARCH_JOBS}" --n-jobs "${E126_N_JOBS}"
         --val-frac 0.3 --seed 42
         --data-version "${DATA}" --feature-include "${INCLUDE}" --run-name "${name}")
    if [ "${window}" -lt "${TUNE_WINDOW}" ]; then
        cmd+=(--tune-window "${TUNE_WINDOW}")
        why="${why}, tuned on ${TUNE_WINDOW}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s window=%-3s  %s\n' "$index" "$total" "$name" "$window" "$why"
        printf '        %s\n' "${cmd[*]}"; continue; fi
    ran=$((ran+1)); started=$(date +%s)
    echo; echo "[${index}/${total}] ${name}  window=${window}  ${why}  $(date '+%H:%M:%S')"
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log" >/dev/null
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${name}")
        echo ">>> ${name} FAILED (exit ${status}) after $((el))s -- continuing"
    else
        echo ">>> ${name} done in $((el))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

echo
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"

# One arm per job (ONLY set) skips consolidation; render it with REPORT_ONLY=1
# once both E123 and E126 have landed.
if [ -z "${ONLY:-}" ]; then
    "${PYTHON}" -u -m src.analysis.e126_report 2>&1 | tee "${REPORT}"
    if [ "${NO_MAIL:-0}" != "1" ]; then
      "${PYTHON}" - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
    fi
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
