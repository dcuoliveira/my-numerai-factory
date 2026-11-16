#!/usr/bin/env bash
#
# E110 -- all-momentum dataset vs raw, on the 60-DAY target.
#
# The 60-day twin of E108. Two lanes, each 5 models tuned from scratch on
# target_binned_return_60 (embargo auto -> 60), run concurrently:
#   raw60     crypto/v2.0                   (22f)  -- the matched 60-day baseline
#   allmom60  crypto/v2.0-allmom-fillmask   (57f)  -- raw + literature momentum 17
#                                                     + xsmom 3 + Carver trend 15
# No matched raw@60 baseline existed for 4 of the 5 model classes, and pairing a
# 60-day arm against a 20-day one would rank the LABEL (CLAUDE.md: never mix
# targets), so the baseline is trained here at the same protocol.
#
# Protocol: rolling 130/130, embargo auto(=60), 5-fold, val_frac 0.3, seed 42,
# n_iter 50. Report per model: raw60 and allmom60 rows (CORR MMC payout c.sharpe
# calmar n, payout + calmar formed per period), then allmom60 vs raw60 paired
# (Newey-West lag 20). Feeds E111 (meta residualisation + ensembles), which also
# rescores these 60-day forecasts on the payable 20-day label.
#
# Knobs: DRY_RUN=1, NO_MAIL=1, SKIP_DONE=1 (default), ONLY_LANE=raw60|allmom60.
# INTERPRETER: PYTHON=/home/danielco/miniconda3/bin/python3.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E110; PREFIX=e110
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; NITER=50; ERA=1
TGT=target_binned_return_60
MODELS=(ridge lgb_ktz catboost_ktz rf_ktz mlp_ktz)
SJOBS=(8 4 1 8 8); NJOBS=(1 2 6 2 1); CELLS=(100 432 12 60 120)
LNAME=(raw60 allmom60); LDATA=(crypto/v2.0 crypto/v2.0-allmom-fillmask); LFEATS=(22 57)

mail() {  # subject body -- send() never raises, it returns False; retry on that
  local i
  for i in 1 2 3; do
    "$PYTHON" -c "import sys;from src.live.notify import send_text;sys.exit(0 if send_text(sys.argv[1],sys.argv[2]) else 1)" "$1" "$2" && return 0
    sleep 30
  done
  echo "note: email '$1' failed after 3 tries"
}

# ---- preflight: datasets carry the right feature count, the 60d label, the meta file ----
"$PYTHON" - "$TGT" "${LDATA[@]}" -- "${LFEATS[@]}" -- "${MODELS[@]}" -- "${CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
a = sys.argv[1:]; tgt = a[0]; rest = a[1:]
def seg(x):
    i = x.index("--"); return x[:i], x[i + 1:]
data, rest = seg(rest); feats, rest = seg(rest); models, cells = seg(rest)
bad = []
for d, n in zip(data, feats):
    names = pq.ParquetFile("data/%s/train.parquet" % d).schema.names
    got = sum(c.startswith("feature_") for c in names)
    if got != int(n): bad.append("%s has %d features, expected %s" % (d, got, n))
    if tgt not in names: bad.append("%s lacks %s" % (d, tgt))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % d): bad.append("%s lacks meta" % d)
for m, wc in zip(models, cells):
    try:
        w = build_wrapper(m, "regression", seed=42, n_jobs=1)
        gc = len(list(ParameterGrid(w.param_grid)))
    except Exception as e:
        bad.append("%s: %s" % (m, e)); continue
    if gc != int(wc): bad.append("%s grid %d != %s" % (m, gc, wc))
if bad: sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s on %s; 5 wrappers construct." % (tgt, ", ".join(data)))
PY

if [ "${DRY_RUN:-0}" = "1" ]; then
  for li in "${!LNAME[@]}"; do for m in "${MODELS[@]}"; do
    printf '%-28s %-14s %s  %s\n' "${PREFIX}_${m}_${LNAME[$li]}" "$m" "${LDATA[$li]}" "$TGT"; done; done
  exit 0
fi

t0=$(date +%s)
[ "${NO_MAIL:-0}" = "1" ] || mail "[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  allmom vs raw on ${TGT}: 2 lanes x 5 models from scratch (raw60 on crypto/v2.0, allmom60 on crypto/v2.0-allmom-fillmask).
  n_iter=${NITER}, rolling ${WIN}/${REFIT}, embargo ${EMB}, ${NSPL} folds, seed ${SEED}. report logs/${PREFIX}_report.txt"

run_lane() {
  local li="$1" lane="${LNAME[$1]}" data="${LDATA[$1]}" mi mdl run st
  for mi in "${!MODELS[@]}"; do
    mdl="${MODELS[$mi]}"; run="${PREFIX}_${mdl}_${lane}"
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${mdl}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
    echo "[${lane} $((mi+1))/5] ${run} $(date '+%H:%M:%S')"
    "$PYTHON" -u -m src.run_forecast crypto "$mdl" --task regression --metric corr --bmc-reference benchmark \
      --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL" \
      --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED" \
      --search-jobs "${SJOBS[$mi]}" --n-jobs "${NJOBS[$mi]}" --data-version "$data" --target "$TGT" \
      --run-name "$run" > "logs/${run}.log" 2>&1
    st=$?; [ "$st" -eq 0 ] && echo ">>> ${run} done" || echo ">>> ${run} FAILED ($st)"
  done
}

echo "=== ${EXPERIMENT}: allmom vs raw on ${TGT} | 2 lanes x 5 models ==="
for li in "${!LNAME[@]}"; do
  [ -n "${ONLY_LANE:-}" ] && [ "${ONLY_LANE}" != "${LNAME[$li]}" ] && continue
  run_lane "$li" &
done
wait

REPORT="logs/${PREFIX}_report.txt"; PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- allmom vs raw on ${TGT} (60-day label; NOT comparable to 20-day tables)"
echo "finished in $(( ($(date +%s)-t0)/60 ))m. payout and calmar formed per period; calmar comparable only at equal n."
echo "=============================================================="
for mdl in "${MODELS[@]}"; do
  echo "=== ${mdl} ==="
  "$PYTHON" - "$mdl" "$PREFIX" <<'PY' || echo "  (unreadable)"
import sys
import pandas as pd
from src.prediction.metrics import summarize
mdl, pref = sys.argv[1], sys.argv[2]
for lane in ("raw60", "allmom60"):
    p = "results/per_period/crypto/%s/%s_%s_%s.csv" % (mdl, pref, mdl, lane)
    try: f = pd.read_csv(p)
    except Exception: print("  %-9s (missing)" % lane); continue
    s = summarize(0.1 * f["corr"] + f["mmc"]); cs = summarize(f["corr"])
    dd = s.get("max_drawdown") or 0.0
    cal = s["mean"] / dd if dd else float("nan")
    print("  %-9s CORR %.5f MMC %.5f payout %.5f c.sharpe %.3f calmar %.3f n=%d"
          % (lane, f["corr"].mean(), f["mmc"].mean(), s["mean"], cs.get("sharpe") or 0.0, cal, len(f)))
PY
  echo "  -- paired allmom60 vs raw60 --"
  "$PYTHON" "$PD" --model "$mdl" --baseline "${PREFIX}_${mdl}_raw60" --arms "${PREFIX}_${mdl}_allmom60" 2>/dev/null \
    | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
  echo
done
} 2>&1 | tee "$REPORT"

fails=$(grep -c "FAILED" "$REPORT" 2>/dev/null || true)
[ "${NO_MAIL:-0}" = "1" ] || mail "[crypto] ${EXPERIMENT} finished" "$(cat "$REPORT")"
