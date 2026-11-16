#!/usr/bin/env bash
#
# Shared core for the mommask/fillmask sweep (E73/E74/E75/E76/E77/E78/E79): run 5
# models from scratch on a <dataset>-{mommask,fillmask} variant, then compare
# THREE-WAY against v2.0 (raw) and a parent dataset, both REUSED from earlier
# studies. Parameterised by env; the e7x wrappers set these and exec this file.
#
#   EXPERIMENT       e.g. E74           (handle)
#   PREFIX           e.g. e74           (artefact prefix)
#   VARIANT          mommask (default) | fillmask   (run-name suffix + report label)
#   MM_DATA          crypto/v2.0-...-{mommask,fillmask}   (the dataset to run)
#   MM_FEATS         expected feature count of MM_DATA (preflight gate)
#   PARENT_PREFIX    prefix of the reused parent runs   (e.g. e70/e71/e72/e76)
#   PARENT_SUFFIX    suffix of the reused parent runs   (e.g. base/xint/carver/mommask)
#   PARENT_LABEL     display label for the parent        (e.g. momswap-macro)
#   V0_PREFIX/V0_SUFFIX  the reused raw-v2.0 runs (e72/base)
#
# Protocol is fixed to match the reused arms: rolling 130/130, embargo auto=20,
# 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20, era-step 1.
# INTERPRETER: PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
: "${EXPERIMENT:?set EXPERIMENT}"; : "${PREFIX:?set PREFIX}"; : "${MM_DATA:?set MM_DATA}"
: "${MM_FEATS:?set MM_FEATS}"; : "${PARENT_PREFIX:?}"; : "${PARENT_SUFFIX:?}"; : "${PARENT_LABEL:?}"
V0_PREFIX="${V0_PREFIX:-e72}"; V0_SUFFIX="${V0_SUFFIX:-base}"
VARIANT="${VARIANT:-mommask}"
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; TGT=target_binned_return_20; ERA=1; NITER=50
SHARD="${SHARD:-1}"; OF="${OF:-1}"
if [ "${SCOUT:-0}" = "1" ]; then PREFIX="${PREFIX}scout"; ERA=6; NSPL=2; NITER=4; echo "SCOUT MODE (not comparable)"; fi

MODELS=(ridge lgb_ktz catboost_ktz rf_ktz mlp_ktz)
SJOBS=(8 4 1 8 8); NJOBS=(1 2 6 2 1); CELLS=(100 432 12 60 120)

# Preflight: MM_DATA present with MM_FEATS + label + meta; parent+v2.0 reuse arms
# exist (warn, don't fail, on a missing parent -- the report degrades gracefully);
# 5 wrappers construct.
"$PYTHON" - "$MM_DATA" "$MM_FEATS" "$TGT" "$PARENT_PREFIX" "$PARENT_SUFFIX" "$V0_PREFIX" "$V0_SUFFIX" "${MODELS[@]}" -- "${CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
a=sys.argv[1:]; c=a.index("--")
mm,feats,tgt,pp,ps,v0p,v0s=a[0],int(a[1]),a[2],a[3],a[4],a[5],a[6]
models=a[7:c]; cells=[int(x) for x in a[c+1:]]
bad=[]; warn=[]
p="data/%s/train.parquet"%mm
try: names=pq.ParquetFile(p).schema.names
except OSError: sys.exit("preflight FAILED: %s MISSING"%mm)
got=sum(cc.startswith("feature_") for cc in names)
if got!=feats: bad.append("%s has %d features, expected %d"%(mm,got,feats))
if tgt not in names: bad.append("%s lacks %s"%(mm,tgt))
if not os.path.exists("data/%s/historical_meta_models.parquet"%mm): bad.append("%s lacks meta"%mm)
for mdl in models:
    for pref,suf,lbl in [(pp,ps,"parent"),(v0p,v0s,"v2.0")]:
        if not os.path.exists("results/metrics/crypto/%s/%s_%s_%s.json"%(mdl,pref,mdl,suf)):
            warn.append("%s_%s_%s (%s) missing -> that cell will be blank"%(pref,mdl,suf,lbl))
for name,wc in zip(models,cells):
    try: w=build_wrapper(name,"regression",seed=42,n_jobs=1); gc=len(list(ParameterGrid(w.param_grid)))
    except Exception as e: bad.append("%s: %s"%(name,e)); continue
    if gc!=wc: bad.append("%s grid %d != %d"%(name,gc,wc))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
if warn: print("preflight WARN (reused arms):\n  "+"\n  ".join(warn))
print("preflight OK: %s (%df) present; 5 wrappers construct."%(mm,feats))
PY

ran=0; failed=0; declare -a FAILED=(); t0=$(date +%s)
if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  5 models on ${MM_DATA} (${VARIANT}); 3-way vs v2.0 (${V0_PREFIX}_*_${V0_SUFFIX}) and ${PARENT_LABEL} (${PARENT_PREFIX}_*_${PARENT_SUFFIX}), reused.
  n_iter=${NITER}, rolling ${WIN}/${REFIT}, embargo ${EMB}, ${NSPL} folds, seed ${SEED}. runner mommask_core.sh; report logs/${PREFIX}_report.txt" 2>/dev/null || echo "note: start email failed"
fi

echo "=== ${EXPERIMENT}: ${MM_DATA} ${VARIANT}, 3-way vs v2.0 + ${PARENT_LABEL} | n_iter=${NITER} ==="
for i in "${!MODELS[@]}"; do
  mdl="${MODELS[$i]}"; run="${PREFIX}_${mdl}_${VARIANT}"
  cmd=("$PYTHON" -u -m src.run_forecast crypto "$mdl" --task regression --metric corr --bmc-reference benchmark
       --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL"
       --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED"
       --search-jobs "${SJOBS[$i]}" --n-jobs "${NJOBS[$i]}" --data-version "$MM_DATA" --target "$TGT" --run-name "$run")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '[%d/5] %-24s %-14s %s\n' "$((i+1))" "$run" "$mdl" "$MM_DATA"; continue; fi
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${mdl}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
  ran=$((ran+1)); echo; echo "[$((i+1))/5] ${run} $(date '+%H:%M:%S')"; s=$(date +%s)
  "${cmd[@]}" > "logs/${run}.log" 2>&1; st=$?; el=$(( $(date +%s)-s ))
  if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$run"); echo ">>> ${run} FAILED ($st) after $((el/60))m -- continue"; else echo ">>> ${run} done $((el/60))m$((el%60))s"; fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="${REPORT:-logs/${PREFIX}_report.txt}"; PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- ${MM_DATA} ${VARIANT}, 3-way vs v2.0 (raw) and ${PARENT_LABEL}"
echo "ran ${ran} of 5 in $(( ($(date +%s)-t0)/60 ))m, ${failed} failed"; [ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"
echo "=============================================================="
for mdl in "${MODELS[@]}"; do
  v0="${V0_PREFIX}_${mdl}_${V0_SUFFIX}"; par="${PARENT_PREFIX}_${mdl}_${PARENT_SUFFIX}"; mk="${PREFIX}_${mdl}_${VARIANT}"
  echo "=== ${mdl}: v2.0 vs ${PARENT_LABEL} vs ${VARIANT} ==="
  "$PYTHON" - "results/metrics/crypto/${mdl}/${v0}.json" "results/metrics/crypto/${mdl}/${par}.json" "results/metrics/crypto/${mdl}/${mk}.json" "$PARENT_LABEL" "$VARIANT" <<'PY' || echo "  (unreadable)"
import sys
import pandas as pd
from src.prediction.metrics import summarize
def row(t,mp):
  # read the per-period series so payout + prop calmar are formed per period
  # (not rebuilt from stored means); calmar = payout mean / payout max-drawdown
  # (positive dd), comparable only across equal n -- so n is printed alongside.
  pp=mp.replace("/metrics/","/per_period/").replace(".json",".csv")
  try: f=pd.read_csv(pp)
  except Exception: print("  %-12s (missing)"%t); return
  cm=float(f["corr"].mean()); mm=float(f["mmc"].mean())
  s=summarize(0.1*f["corr"]+f["mmc"]); cs=summarize(f["corr"])
  dd=s.get("max_drawdown") or 0.0
  calmar=(s["mean"]/dd) if dd else float("nan")
  print("  %-12s CORR %.5f MMC %.5f payout %.5f c.sharpe %.3f calmar %.3f n=%d"
        %(t,cm,mm,s["mean"],cs.get("sharpe") or 0.0,calmar,len(f)))
row("v2.0",sys.argv[1]); row(sys.argv[4],sys.argv[2]); row(sys.argv[5],sys.argv[3])
PY
  echo "  -- paired ${VARIANT} vs ${PARENT_LABEL} --"; "$PYTHON" "$PD" --model "$mdl" --baseline "$par" --arms "$mk" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
  echo "  -- paired ${VARIANT} vs v2.0 --"; "$PYTHON" "$PD" --model "$mdl" --baseline "$v0" --arms "$mk" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
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
