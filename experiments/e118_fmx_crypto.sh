#!/usr/bin/env bash
#
# E118 -- 34 causal representation blocks, ONE at a time, on the best clean ridge,
#         LightGBM and CatBoost.
#
# Bases (the payout-best clean standalone run of each family on the 510 scored
# periods; "clean" = regression, era-step 1, 20-day-scored, on a data version some
# live slot was backtested on):
#   ridge     ridge_glmnet  allfeat_pp minus tenc   811f   w6r6_lobo_tenc          0.08779
#   lgb       lgb_ktz       tenc_onehot             763f   e29_tenc_onehot         0.08999
#   catboost  catboost_ktz  fenccausal               44f   e109_catboost_ktz_fenc  0.09303
#
# Blocks feature_xNN_ (built by src/data_mani/build_fmx.py from the caches of
# foundation_repr, fmx_tabpfn, fmx_models and foundation_repr_ts): x01-x20 TabPFN /
# TimesFM / Chronos-2 representations, x22-x35 the book's RF, CatBoost, LightGBM,
# MLP and block ridges refit walk-forward from 2023-05. 21 does not exist (rows,
# not columns).
#
# Arms: e118_<family>_xNN = base + block NN, selected on the -fmx dataset with
#         --feature-exclude '^feature_x(?!NN_)\d\d_'   (ridge also drops ^feature_sym_tenc$)
#       e118_<family>_x00 = base alone on the -fmx dataset: the in-study "without",
#         which must reproduce the reused baseline (build_fmx asserts the data is identical).
# 3 x 35 = 105 arms, each family at ITS baseline's protocol: rolling 130/130,
# embargo auto=20, 5-fold, val_frac 0.3, seed 42, target_binned_return_20; n_iter
# 50 / 60 / 50; search-jobs 1 / 3 / 1.
#
# Measured against: e118_<family>_x00, paired Newey-West at lag 20 over 510 periods,
# CORR / MMC / payout with calmar and burn. 34 blocks x 3 families is 102 tests, so
# a single |t| ~ 2 is about what chance alone produces: read the pattern across
# families, not one cell.
#
# Knobs: DRY_RUN=1  ONLY=<regex on run name>  SHARD=k OF=n  SKIP_DONE=0  NO_MAIL=1  REPORT_ONLY=1
#        CLAIM=1  work-queue mode: every lane walks ALL selected arms and runs the next one
#                 nobody has claimed (atomic mkdir under logs/e118_claims/), so lanes can be
#                 added whenever the cluster has room. Clear a dead lane's claim to retry it.
# Slots (src.cluster, 8 cores): search-jobs x n-jobs = 1x8 ridge, 4x2 lgb, 1x6 catboost.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E118; PREFIX=e118
SHARD="${SHARD:-1}"; OF="${OF:-1}"
BLOCKS=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35)

# family | model | data | base exclude | n_iter | search_jobs | n_jobs | base features
FAMILIES=(
  "ridge|ridge_glmnet|crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx|^feature_sym_tenc\$|50|1|8|811"
  "lgb|lgb_ktz|crypto/v2.0-tenc-onehot-fmx||60|4|2|763"
  "catboost|catboost_ktz|crypto/v2.0-fenccausal-fmx||50|1|6|44"
)

ARMS=()
for nn in "${BLOCKS[@]}"; do for f in "${FAMILIES[@]}"; do ARMS+=("$nn|$f"); done; done
SELECTED=()
for a in "${ARMS[@]}"; do
  IFS='|' read -r nn fam _ <<< "$a"; run="${PREFIX}_${fam}_x${nn}"
  if [ -n "${ONLY:-}" ] && ! [[ "$run" =~ ${ONLY} ]]; then continue; fi
  SELECTED+=("$a")
done
MINE=(); for k in "${!SELECTED[@]}"; do
  if [ "${CLAIM:-0}" = "1" ] || [ $(( k % OF + 1 )) -eq "$SHARD" ]; then MINE+=("${SELECTED[$k]}"); fi
done
mkdir -p logs/e118_claims

exclude_for() {   # block, base exclude -> the arm's --feature-exclude
  local nn="$1" base="$2" blk
  if [ "$nn" = 00 ]; then blk='^feature_x\d\d_'; else blk="^feature_x(?!${nn}_)\\d\\d_"; fi
  if [ -n "$base" ]; then echo "${base}|${blk}"; else echo "$blk"; fi
}

if [ "${REPORT_ONLY:-0}" = "1" ]; then
  "$PYTHON" -m src.analysis.e118_report; st=$?
  if [ "${NO_MAIL:-0}" != "1" ] && [ -s logs/e118_report.txt ]; then
    "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text('[crypto] E118 report',open('logs/e118_report.txt').read())" || echo "note: email failed"
  fi
  exit $st
fi

# Preflight: each dataset present with meta, and each arm's filter keeping exactly
# base + its block.
"$PYTHON" - "${MINE[@]}" <<'PY' || exit 1
import os, sys, re
import pyarrow.parquet as pq
from src.prediction.functions import filter_features
bad, seen = [], {}
for arm in sys.argv[1:]:
    nn, fam, model, data, bexcl, n_iter, sj, nj, nbase = arm.split("|")
    if data not in seen:
        p = "data/%s/train.parquet" % data
        if not os.path.exists(p): bad.append("%s MISSING" % data); seen[data] = None; continue
        seen[data] = [c for c in pq.ParquetFile(p).schema.names if c.startswith("feature_")]
        if not os.path.exists("data/%s/historical_meta_models.parquet" % data): bad.append("%s lacks meta" % data)
    feats = seen[data]
    if feats is None: continue
    blk = r"^feature_x\d\d_" if nn == "00" else r"^feature_x(?!%s_)\d\d_" % nn
    excl = "%s|%s" % (bexcl, blk) if bexcl else blk
    kept = filter_features(feats, None, excl, verbose=False)
    block = [c for c in kept if re.match(r"^feature_x\d\d_", c)]
    if len(kept) - len(block) != int(nbase): bad.append("%s x%s keeps %d base features, expected %s" % (fam, nn, len(kept) - len(block), nbase))
    if nn != "00" and (not block or any(not c.startswith("feature_x%s_" % nn) for c in block)):
        bad.append("%s x%s block selection wrong: %s" % (fam, nn, block[:4]))
if bad: sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d arms" % len(sys.argv[1:]))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ "$SHARD" = 1 ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname): ${#SELECTED[@]} arms in ${OF} shard(s). 34 causal representation blocks, one at a time, on the best clean ridge / LightGBM / CatBoost." 2>/dev/null || echo "note: start email failed"
fi

echo "=== ${EXPERIMENT} shard ${SHARD}/${OF} on $(hostname): ${#MINE[@]} of ${#SELECTED[@]} arms ==="
ran=0; failed=0; FAILED=()
for arm in "${MINE[@]}"; do
  IFS='|' read -r nn fam model data bexcl n_iter sj nj nbase <<< "$arm"
  run="${PREFIX}_${fam}_x${nn}"; excl="$(exclude_for "$nn" "$bexcl")"
  cmd=("$PYTHON" -u -m src.run_forecast crypto "$model" --task regression --metric corr --bmc-reference benchmark
       --cv-scheme rolling --train-window 130 --refit-every 130 --embargo auto --n-splits 5 --n-iter "$n_iter"
       --val-frac 0.3 --era-step 1 --seed 42 --search-jobs "$sj" --n-jobs "$nj" --data-version "$data"
       --target target_binned_return_20 --feature-exclude "$excl" --run-name "$run")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%-22s %-13s %s\n' "$run" "$model" "$excl"; continue; fi
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
  if [ "${CLAIM:-0}" = "1" ] && ! mkdir "logs/e118_claims/${run}" 2>/dev/null; then continue; fi
  ran=$((ran+1)); echo "[${run}] $(date '+%H:%M:%S')"; s=$(date +%s)
  "${cmd[@]}" > "logs/${run}.log" 2>&1; st=$?; el=$(( $(date +%s)-s ))
  if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$run"); echo ">>> ${run} FAILED ($st) after $((el/60))m"
  else echo ">>> ${run} done $((el/60))m$((el%60))s"; fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo "shard ${SHARD}/${OF}: ran ${ran}, failed ${failed} ${FAILED[*]:-}"
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
