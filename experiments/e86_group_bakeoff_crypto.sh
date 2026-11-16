#!/usr/bin/env bash
#
# E86 -- diverse-primitives bake-off: tune every model class FROM SCRATCH on each
# primitive feature GROUP of the raw 22, then rank the (model x group) cells and
# screen ensembles.
#
# The g1_diverse_primitives.json the user referenced never reached disk, so the
# groups are the raw-22 primitive families, which partition the 22 exactly:
#   momentum(2) volatility(4=volatility+bollinger) rsi(2) sharpe(2) volume(4)
#   size(4=market_cap) price_ma(4=close_)
# Each group is a --feature-include regex on crypto/v2.0 (one dataset -> one clean
# common window for a fair rank + ensemble screen). 7 groups x 5 models = 35 runs,
# from scratch at the shared protocol (rolling 130/130, embargo auto=20, 5-fold,
# val_frac 0.3, seed 42, n_iter 50, target_binned_return_20). Each cell is paired
# (NW lag 20) vs the FULL raw 22 (e72_*_base, reused) to show how much of raw's
# signal the group carries. Ensemble screen (promising blends across cells + the
# book) is run afterwards with report_html on the e86_* arms.
#
# Parallelism: the 7 groups run CONCURRENTLY (each a subshell looping its 5 models
# sequentially); ~7 x 8 cores = ~56 of 64. Wall-clock ~ 5 models x ~28 min ~ 2.5h.
# Knobs: DRY_RUN=1 print+exit; SCOUT=1 era-step 6 / n_iter 4 (wiring only, NOT
# comparable); NO_MAIL=1; GROUP_JOBS caps concurrent groups (default 7); SKIP_DONE=1.
# INTERPRETER: PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E86; PREFIX=e86
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; TGT=target_binned_return_20; ERA=1; NITER=50
DATA=crypto/v2.0
if [ "${SCOUT:-0}" = "1" ]; then PREFIX="${PREFIX}scout"; ERA=6; NSPL=2; NITER=4; echo "SCOUT MODE (not comparable)"; fi
GROUP_JOBS="${GROUP_JOBS:-7}"

# parallel arrays: groups (name / include-regex / expected feature count)
GNAME=(momentum volatility rsi sharpe volume size price_ma)
GRX=(momentum 'volatility|bollinger' relative_strength sharpe volume market_cap 'close_')
GCNT=(2 4 2 2 4 4 4)
# models with their search/fit job split and grid-cell gate
MODELS=(ridge lgb_ktz catboost_ktz rf_ktz mlp_ktz)
SJOBS=(4 2 1 4 4); NJOBS=(1 1 2 1 1); CELLS=(100 432 12 60 120)

# ---- preflight: dataset present, every group regex hits its count, wrappers build ----
"$PYTHON" - "$DATA" "$TGT" "${GNAME[@]}" -- "${GRX[@]}" -- "${GCNT[@]}" -- "${MODELS[@]}" -- "${CELLS[@]}" <<'PY' || exit 1
import sys, re
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
a=sys.argv[1:]; data,tgt=a[0],a[1]; rest=a[2:]
def seg(x):
    i=x.index("--"); return x[:i], x[i+1:]
gname,rest=seg(rest); grx,rest=seg(rest); gcnt,rest=seg(rest); models,cells=seg(rest)
gcnt=[int(x) for x in gcnt]; cells=[int(x) for x in cells]
names=pq.ParquetFile("data/%s/train.parquet"%data).schema.names
feats=[c for c in names if c.startswith("feature_")]
bad=[]
if tgt not in names: bad.append("%s lacks %s"%(data,tgt))
for n,rx,c in zip(gname,grx,gcnt):
    got=len([f for f in feats if re.search(rx,f)])
    if got!=c: bad.append("group %s regex /%s/ matched %d, expected %d"%(n,rx,got,c))
for m,wc in zip(models,cells):
    try: w=build_wrapper(m,"regression",seed=42,n_jobs=1); gc=len(list(ParameterGrid(w.param_grid)))
    except Exception as e: bad.append("%s: %s"%(m,e)); continue
    if gc!=wc: bad.append("%s grid %d != %d"%(m,gc,wc))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %s, 7 groups partition raw (%d feats), 5 wrappers construct."%(data,len(feats)))
PY

[ "${DRY_RUN:-0}" = "1" ] && { for gi in "${!GNAME[@]}"; do for mi in "${!MODELS[@]}"; do
  printf '%-28s %-14s include=/%s/\n' "${PREFIX}_${MODELS[$mi]}_${GNAME[$gi]}" "${MODELS[$mi]}" "${GRX[$gi]}"; done; done; exit 0; }

t0=$(date +%s)
if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} diverse-primitives bake-off started $(date '+%F %T') on $(hostname)
  7 raw-22 groups x 5 models = 35 from-scratch runs on ${DATA}, n_iter=${NITER}, rolling ${WIN}/${REFIT}.
  groups run concurrently (${GROUP_JOBS} at a time). report logs/${PREFIX}_report.txt" 2>/dev/null || echo "note: start email failed"
fi

# ---- run one group: its 5 models sequentially ----
run_group() {
  local gi="$1"; local g="${GNAME[$gi]}"; local rx="${GRX[$gi]}"
  for mi in "${!MODELS[@]}"; do
    local mdl="${MODELS[$mi]}"; local run="${PREFIX}_${mdl}_${g}"
    [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${mdl}/${run}.json" ] && { echo "  ${run} on disk -- skip"; continue; }
    "$PYTHON" -u -m src.run_forecast crypto "$mdl" --task regression --metric corr --bmc-reference benchmark \
      --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL" \
      --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED" \
      --search-jobs "${SJOBS[$mi]}" --n-jobs "${NJOBS[$mi]}" --data-version "$DATA" --feature-include "$rx" \
      --target "$TGT" --run-name "$run" > "logs/${run}.log" 2>&1 \
      && echo ">>> ${run} done" || echo ">>> ${run} FAILED ($?)"
  done
}

# ---- launch groups with a concurrency cap ----
echo "=== ${EXPERIMENT}: 7 groups x 5 models from scratch on ${DATA} | n_iter=${NITER} ==="
active=0
for gi in "${!GNAME[@]}"; do
  run_group "$gi" &
  active=$((active+1))
  if [ "$active" -ge "$GROUP_JOBS" ]; then wait -n 2>/dev/null || wait; active=$((active-1)); fi
done
wait

# ---- rank every (model x group) cell + paired vs full raw 22 (e72_*_base) ----
REPORT="logs/${PREFIX}_report.txt"; PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- diverse-primitives bake-off: 7 raw groups x 5 models, from scratch on ${DATA}"
echo "finished in $(( ($(date +%s)-t0)/60 ))m. Each cell paired (NW lag 20) vs FULL raw 22 (e72_<model>_base)."
echo "=============================================================="
"$PYTHON" - "$PREFIX" "${GNAME[@]}" -- "${MODELS[@]}" <<'PY'
import json, sys, os
a=sys.argv[1:]; pref=a[0]; i=a.index("--"); groups=a[1:i]; models=a[i+1:]
def load(m,run):
    p="results/metrics/crypto/%s/%s.json"%(m,run)
    try: d=json.load(open(p))
    except Exception: return None
    c=d.get("corr") or {}; mm=d.get("mmc") or {}
    return c.get("mean") or 0., mm.get("mean") or 0., (c.get("sharpe") or 0.)
rows=[]
for g in groups:
    for m in models:
        r=load(m,"%s_%s_%s"%(pref,m,g))
        if r: rows.append((g,m,r[0],r[1],0.1*r[0]+r[1],r[2]))
rows.sort(key=lambda x:-x[4])
print("rank  %-10s %-13s %8s %8s %8s %7s"%("group","model","CORR","MMC","payout","c.shrp"))
for k,(g,m,c,mm,p,s) in enumerate(rows,1):
    print("%4d  %-10s %-13s %8.5f %8.5f %8.5f %7.3f"%(k,g,m,c,mm,p,s))
print("\nFULL raw-22 reference (e72_*_base):")
for m in models:
    r=load(m,"e72_%s_base"%m)
    if r: print("  %-13s CORR %.5f MMC %.5f payout %.5f"%(m,r[0],r[1],0.1*r[0]+r[1]))
PY
echo
echo "Ensemble screen: run_html on the 35 e86_* arms + book, e.g.:"
echo "  $PYTHON -m src.analysis.report_html --model lgb_ktz --runs \$(for g in ${GNAME[*]}; do for m in ${MODELS[*]}; do echo \$m/${PREFIX}_\${m}_\$g; done; done)"
} 2>&1 | tee "$REPORT"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "$REPORT" "$EXPERIMENT" <<'PY' || echo "note: email failed"
import sys;from src.live.notify import send_text
send_text("[crypto] %s finished"%sys.argv[2], open(sys.argv[1]).read())
PY
fi
