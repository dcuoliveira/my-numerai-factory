#!/usr/bin/env bash
#
# E122 -- how much HISTORY should each refit use? The training window of the best
#         standalone RANDOM FOREST, from one week to six months. The rf sibling
#         of E117 (best ridge) and E119 (best MLP, IA_01).
#
# The base model is e57_tenc_onehot: the payout-best standalone rf_ktz (the E48
# priors), on crypto/v2.0-tenc-onehot (763f), the 60-DAY label -- the strongest
# single RF we have measured (payout 0.0988, MMC 0.0826 at the 130 window).
# Everything about it is held fixed -- data, features, the 40-draw search, 5
# inner folds, 130-period refit cadence, embargo auto=60 (the target horizon),
# seed -- and ONLY the rolling training window moves. (The user asked for this as
# "the equivalent E117"; E117-E121 are taken (ridge/mlp/lgb/catboost), so the RF one is filed E122.)
#
# THE WINDOWS, in PERIODS (weekday-only panel: ~5/week, ~22/month):
#     1 week 5   2 weeks 10   3 weeks 15   4 weeks 20
#     1 month 22   3 months 65   6 months 130 (the deployed window = e57)
#
# TUNE-WINDOW 130, exactly as E117. A purged inner fold needs training + the
# 60-period embargo + validation INSIDE the block it searches, so any window
# <=~70 periods has "no usable CV folds" on itself. --tune-window 130 runs the
# search on the 130 periods ending at the same date, with rolling inner folds
# that each TRAIN ON THE WINDOW'S LENGTH, then refits the winner on the training
# window alone. Every arm selects on the same slices; only how much history each
# fold and the final fit see differs. The 130 arm needs none (it searches its own
# window, reproducing e57_tenc_onehot) -- run_forecast rejects a tune-window that
# does not exceed the train-window, so it is omitted there.
#
# WHAT IT DOES NOT VARY  the refit cadence stays 130 (as E117): a 1-week window is
# fitted once every six months and predicts the next 130 periods from one week of
# data -- the "how much history per refit" question, NOT the freshness question
# (that is the cadence axis). Expect short windows to lose: 5 periods at a 60-day
# target is well under one effective observation, and the rf_ktz leaf floor
# {300..4000} is larger than a 5-period block has rows, so the search is pushed to
# its lightest leaves -- a finding in itself.
#
# All arms score the SAME ~470 60-day periods (the scored block depends on the
# data/cadence/embargo, not the training window), so they are directly paired.
#
# Usage:
#   bash experiments/e122_rf_window_crypto.sh            # all 7 arms (pooled), then report
#   ONLY_W="5 130" / DRY_RUN=1 / NO_MAIL=1 / RFK_POOL=7
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e122_rf_window_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
# TW_TARGET picks the FITTED label. 60 is E122 (e57_tenc_onehot's label); 20 is
# E128, the same ladder fitted on the payable label, so the two halves come from
# one runner and cannot drift apart. --embargo auto follows the label (60 / 20).
TW_TARGET="${TW_TARGET:-60}"
case "${TW_TARGET}" in
    60) EXPERIMENT="${EXPERIMENT:-E122}"; PREFIX_BASE="e122" ;;
    20) EXPERIMENT="${EXPERIMENT:-E128}"; PREFIX_BASE="e128" ;;
    *)  echo "TW_TARGET must be 20 or 60, got '${TW_TARGET}'" >&2; exit 1 ;;
esac
DATA="crypto/v2.0-tenc-onehot"
TARGET="target_binned_return_${TW_TARGET}"
REFIT=130; NSPLITS=5; NITER="${RFK_NITER:-40}"; SEED=42; VALFRAC=0.3; TUNE=130
RFK_JOBS="${RFK_JOBS:-4}"; RFK_THREADS="${RFK_THREADS:-4}"; RFK_POOL="${RFK_POOL:-7}"
export OMP_NUM_THREADS="$RFK_THREADS" MKL_NUM_THREADS="$RFK_THREADS" OPENBLAS_NUM_THREADS="$RFK_THREADS"

PREFIX="${PREFIX_BASE}"
ERA=1
if [ "${SMOKE:-0}" = "1" ]; then PREFIX="${PREFIX_BASE}scout"; ERA=4; NITER=6; echo "SMOKE: era 4, wiring only"; fi

# name window
NAMES=(); WINDOWS=()
add(){ NAMES+=("$1"); WINDOWS+=("$2"); }
add w5 5; add w10 10; add w15 15; add w20 20; add w22 22; add w65 65; add w130 130

# Preflight: dataset + t60 column + rf_ktz grid intact.
"$PYTHON" - "$DATA" "$TARGET" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
from src.prediction.models import build_wrapper
ver, tgt = sys.argv[1], sys.argv[2]; bad=[]
cols = pq.ParquetFile("data/%s/train.parquet"%ver).schema.names
n = sum(c.startswith("feature_") for c in cols)
if n != 763: bad.append("%s has %d features, expected 763"%(ver,n))
if tgt not in cols: bad.append("%s lacks %s"%(ver,tgt))
if not os.path.exists("data/%s/historical_meta_models.parquet"%ver): bad.append("no meta file")
g = build_wrapper("rf_ktz","regression",seed=42,n_jobs=1).param_grid
if g.get("n_estimators")!=[100] or max(g["min_samples_leaf"])<=1000: bad.append("rf_ktz grid changed")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %s (763f) has %s; rf_ktz grid intact"%(ver,tgt))
PY

SEL_N=(); SEL_W=()
for i in "${!NAMES[@]}"; do
  w="${WINDOWS[$i]}"
  if [ -n "${ONLY_W:-}" ]; then case " ${ONLY_W} " in *" $w "*) ;; *) continue;; esac; fi
  SEL_N+=("${NAMES[$i]}"); SEL_W+=("$w")
done
total="${#SEL_N[@]}"; [ "$total" -eq 0 ] && { echo "no windows selected">&2; exit 1; }

[ "${NO_MAIL:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ] && \
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text('[crypto] %s started'%sys.argv[1],sys.argv[2])" \
    "$EXPERIMENT" "$EXPERIMENT training-window sweep of the best RF (e57_tenc_onehot)
  arms ${total}: ${SEL_N[*]} | base rf_ktz/tenc_onehot/t${TW_TARGET}, refit 130, tune-window 130, n_iter $NITER
  results results/metrics/crypto/rf_ktz/${PREFIX}_w*.json" 2>/dev/null && echo "  start mail sent" || true

echo "=== $EXPERIMENT: RF training-window sweep | ${total} arm(s): ${SEL_N[*]} | python=$PYTHON ==="
started=$(date +%s)
for i in "${!SEL_N[@]}"; do
  name="${PREFIX}_${SEL_N[$i]}"; w="${SEL_W[$i]}"
  if [ "${SKIP_DONE:-1}" = 1 ] && [ -s "results/metrics/crypto/rf_ktz/${name}.json" ]; then echo "[skip] $name"; continue; fi
  cmd=("$PYTHON" -u -m src.run_forecast crypto rf_ktz --task regression --metric corr
       --bmc-reference benchmark --data-version "$DATA" --target "$TARGET" --embargo auto
       --cv-scheme rolling --train-window "$w" --refit-every "$REFIT"
       --n-splits "$NSPLITS" --n-iter "$NITER" --seed "$SEED" --val-frac "$VALFRAC"
       --era-step "$ERA" --search-jobs 1 --n-jobs "$RFK_JOBS" --run-name "$name")
  # small windows need the search run on a 130-period tune block; the 130 arm
  # searches its own window (tune-window must exceed train-window).
  [ "$w" -lt "$TUNE" ] && cmd+=(--tune-window "$TUNE")
  if [ "${DRY_RUN:-0}" = 1 ]; then printf '[%d/%d] %-6s window=%s %s\n' "$((i+1))" "$total" "$name" "$w" "$([ "$w" -lt "$TUNE" ] && echo "(tune $TUNE)")"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$RFK_POOL" ]; do sleep 5; done
  echo "[launch $((i+1))/$total] $name window=$w $(date '+%H:%M:%S')"
  ( t0=$(date +%s); "${cmd[@]}" >"logs/${name}.log" 2>&1; echo ">>> $name w=$w exit=$? in $(($(date +%s)-t0))s" >>"logs/${PREFIX}_pool.log" ) &
done
[ "${DRY_RUN:-0}" = 1 ] && exit 0
wait

REPORT="logs/${PREFIX}_report.txt"
{
echo "$EXPERIMENT -- RF (rf_ktz/tenc_onehot/t${TW_TARGET}) training-window sweep, all arms paired vs w130"
[ "${TW_TARGET}" = "20" ] && echo "(the fit-60 half is E122; both, raw and meta-residualised, on the 20-day label: python3 -m src.analysis.e128_report)"
echo "wall $(( ($(date +%s)-started)/60 ))m"
echo
runs=$(for n in "${SEL_N[@]}"; do echo -n "${PREFIX}_${n} "; done)
"$PYTHON" -m src.analysis.results_table --model rf_ktz --runs $runs --sort-by payout 2>&1 || echo "(results_table failed)"
echo
echo "paired Newey-West (lag 20) vs w130 (the deployed window):"
for n in "${SEL_N[@]}"; do
  [ "$n" = "w130" ] && continue
  echo "--- ${PREFIX}_${n} vs ${PREFIX}_w130 ---"
  "$PYTHON" .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
     --model rf_ktz --baseline "${PREFIX}_w130" --arms "${PREFIX}_${n}" 2>&1 | grep -E "corr|mmc|payout|verdict|periods" || true
done
} 2>&1 | tee "$REPORT"

[ "${NO_MAIL:-0}" != 1 ] && "$PYTHON" - "$REPORT" "$EXPERIMENT" <<'PY' || true
import sys;from src.live.notify import send_text
send_text("[crypto] %s finished"%sys.argv[2], open(sys.argv[1]).read())
PY
