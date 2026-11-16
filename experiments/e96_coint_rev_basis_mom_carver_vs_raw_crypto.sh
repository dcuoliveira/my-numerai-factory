#!/usr/bin/env bash
#
# E96 -- the widest stat-arb/trend stack: does coint + reversal + basis/funding +
# momentum + Carver trend, all together, beat the raw panel? Tune 5 model classes
# FROM SCRATCH on crypto/v2.0-coint-rev-basis-mom-carver-fillmask and compare each,
# PAIRED, against its raw v2.0 twin.
#
# The dataset is a STRICT SUPERSET of crypto/v2.0: the raw 22 plus 60 features in
# six families --
#   coint   (4):  feature_coint_{adf_120d,halflife_120d,resid_z_60d,resid_z_120d}
#   rev    (14):  feature_rev_*   (short-term reversal)
#   basis   (6):  feature_basis_* (futures-spot basis)
#   funding (4):  feature_funding_* (perp funding)
#   mom    (17):  feature_mom_*   (literature momentum incl. Tzotchev robust trend)
#   tf     (15):  feature_tf_*    (Carver/Clenow trend-following: ewmac/breakout/
#                 bollinger-z/clenow slope)
# fillmask = missing cells pre-filled to 0.5 (== the loader's own fill), so as
# loaded it is a clean raw+60 superset. The ONLY difference between an
# e96_<m>_crbmc arm and its reused raw baseline e72_<m>_base is those 60 columns.
# This is E95 (coint+rev+basis, 50f) extended with the momentum (E63/E66/E67) and
# Carver (E72) blocks -- the full microstructure + trend kitchen sink on raw.
#
# 5 models (ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz) at the shared protocol
# (rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, n_iter 50,
# target_binned_return_20, era 1) -- IDENTICAL to e72_*_base, so the pairing is
# valid. Raw side REUSED (e72_*_base), not re-run; each reported with a paired
# Newey-West (lag 20) delta crbmc - raw.
#
# Env knobs: DRY_RUN=1, SCOUT=1 (era 6 / n_iter 4 / 2 folds, not comparable),
# NO_MAIL=1, SKIP_DONE=1 (default).
# INTERPRETER: PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E96; PREFIX=e96; VARIANT=crbmc
DATA=crypto/v2.0-coint-rev-basis-mom-carver-fillmask; RAW=crypto/v2.0; V0_PREFIX=e72; V0_SUFFIX=base
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; TGT=target_binned_return_20; ERA=1; NITER=50
if [ "${SCOUT:-0}" = "1" ]; then PREFIX="${PREFIX}scout"; ERA=6; NSPL=2; NITER=4; echo "SCOUT MODE (not comparable)"; fi

MODELS=(ridge lgb_ktz catboost_ktz rf_ktz mlp_ktz)
SJOBS=(8 4 1 8 8); NJOBS=(1 2 6 2 1); CELLS=(100 432 12 60 120)

# ---- preflight: superset of raw, exactly 60 new in the 6 expected families; target + raw baselines; wrappers ----
"$PYTHON" - "$DATA" "$RAW" "$TGT" "$V0_PREFIX" "$V0_SUFFIX" "${MODELS[@]}" -- "${CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
a=sys.argv[1:]; c=a.index("--")
data,raw,tgt,v0p,v0s=a[0],a[1],a[2],a[3],a[4]; models=a[5:c]; cells=[int(x) for x in a[c+1:]]
nb=pq.ParquetFile("data/%s/train.parquet"%data).schema.names
fb=set(x for x in nb if x.startswith("feature_"))
fr=set(x for x in pq.ParquetFile("data/%s/train.parquet"%raw).schema.names if x.startswith("feature_"))
bad=[]
if tgt not in nb: bad.append("%s lacks %s"%(data,tgt))
if not fr.issubset(fb): bad.append("%s is NOT a superset of %s (missing %s)"%(data,raw,sorted(fr-fb)[:3]))
new=sorted(fb-fr)
exp={"coint":4,"rev":14,"basis":6,"funding":4,"mom":17,"tf":15}
fam={p:sum(1 for f in new if f.startswith("feature_"+p+"_")) for p in exp}
unexp=[f for f in new if not any(f.startswith("feature_"+p+"_") for p in exp)]
if len(new)!=60: bad.append("expected 60 new feats, got %d"%len(new))
if unexp: bad.append("new feats outside the 6 families: %s"%unexp[:4])
if fam!=exp: bad.append("family counts %s != expected %s"%(fam,exp))
miss=[m for m in models if not os.path.exists("results/metrics/crypto/%s/%s_%s_%s.json"%(m,v0p,m,v0s))]
if miss: bad.append("missing reused raw baselines %s_*_%s for: %s"%(v0p,v0s,miss))
for m,wc in zip(models,cells):
    try: gc=len(list(ParameterGrid(build_wrapper(m,"regression",seed=42,n_jobs=1).param_grid)))
    except Exception as e: bad.append("%s: %s"%(m,e)); continue
    if gc!=wc: bad.append("%s grid %d != %d"%(m,gc,wc))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %s = raw %d + 60 (coint4/rev14/basis6/funding4/mom17/tf15) = %d feats; 5 raw baselines present; 5 wrappers build."%(data,len(fr),len(fb)))
PY

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in "${!MODELS[@]}"; do printf '[%d/5] %-28s %-14s %s (paired vs %s_%s_%s)\n' "$((i+1))" "${PREFIX}_${MODELS[$i]}_${VARIANT}" "${MODELS[$i]}" "$DATA" "$V0_PREFIX" "${MODELS[$i]}" "$V0_SUFFIX"; done
  exit 0
fi

t0=$(date +%s); ran=0; failed=0; declare -a FAILED=()
if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} coint+rev+basis+mom+carver vs raw started $(date '+%F %T') on $(hostname)
  5 models from scratch on ${DATA} (raw 22 + 60); paired vs raw v2.0 (${V0_PREFIX}_*_${V0_SUFFIX}), reused.
  n_iter=${NITER}, rolling ${WIN}/${REFIT}, embargo ${EMB}, ${NSPL} folds, seed ${SEED}. report logs/${PREFIX}_report.txt" 2>/dev/null || echo "note: start email failed"
fi

echo "=== ${EXPERIMENT}: ${DATA} (raw+60) vs raw v2.0, 5 models from scratch | n_iter=${NITER} ==="
for i in "${!MODELS[@]}"; do
  mdl="${MODELS[$i]}"; run="${PREFIX}_${mdl}_${VARIANT}"
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${mdl}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
  ran=$((ran+1)); echo; echo "[$((i+1))/5] ${run} $(date '+%H:%M:%S')"; s=$(date +%s)
  "$PYTHON" -u -m src.run_forecast crypto "$mdl" --task regression --metric corr --bmc-reference benchmark \
    --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL" \
    --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED" \
    --search-jobs "${SJOBS[$i]}" --n-jobs "${NJOBS[$i]}" --data-version "$DATA" --target "$TGT" --run-name "$run" \
    > "logs/${run}.log" 2>&1; st=$?; el=$(( $(date +%s)-s ))
  if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$run"); echo ">>> ${run} FAILED ($st) after $((el/60))m -- continue"; else echo ">>> ${run} done $((el/60))m$((el%60))s"; fi
done

REPORT="logs/${PREFIX}_report.txt"; PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- coint+rev+basis+mom+carver (raw 22 + 60) vs raw v2.0, 5 models from scratch"
echo "ran ${ran} of 5 in $(( ($(date +%s)-t0)/60 ))m, ${failed} failed"; [ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"
echo "=============================================================="
for mdl in "${MODELS[@]}"; do
  v0="${V0_PREFIX}_${mdl}_${V0_SUFFIX}"; mk="${PREFIX}_${mdl}_${VARIANT}"
  echo "=== ${mdl}: raw v2.0 vs crbmc (raw+60) ==="
  "$PYTHON" - "results/metrics/crypto/${mdl}/${v0}.json" "results/metrics/crypto/${mdl}/${mk}.json" <<'PY' || echo "  (unreadable)"
import json,sys
def row(t,p):
  try:d=json.load(open(p))
  except Exception:print("  %-6s (missing)"%t);return
  c=d.get("corr") or {};m=d.get("mmc") or {};cm=c.get("mean") or 0.;mm=m.get("mean") or 0.
  print("  %-6s CORR %.5f MMC %.5f payout %.5f c.sharpe %.3f n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.,d.get("n_periods")))
row("raw",sys.argv[1]); row("crbmc",sys.argv[2])
PY
  echo "  -- paired crbmc vs raw (NW lag 20) --"; "$PYTHON" "$PD" --model "$mdl" --baseline "$v0" --arms "$mk" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
  echo
done
} 2>&1 | tee "$REPORT"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "$([ "${failed}" -gt 0 ] && echo "WITH ${failed} FAILED" || echo SUCCESS)" "$REPORT" "$EXPERIMENT" <<'PY' || echo "note: email failed"
import sys;from src.live.notify import send_text
send_text("[crypto] %s finished -- %s"%(sys.argv[3],sys.argv[1]),open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
