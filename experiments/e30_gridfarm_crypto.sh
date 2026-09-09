#!/usr/bin/env bash
# E30 -- FARM over the full (model x feature-group) grid.
#
# The (model x feature-group) singles grid is already ~complete: every one of the
# 7 semantic raw-groups (momentum/volatility/oscillator/bollinger/price/size/
# liquidity) and the derived blocks (tenc/spread/interact/fenccausal/rollsel) has
# a from-scratch pseudo-OOS search for all three families -- from the grp_*
# studies, E14 (lgb) and E15 (mlp), plus E29's e28_glmnet_rollsel. So no new
# tuning is needed here (onehot is EXCLUDED per the scope decision; its singles
# live in E29). What was missing is a FARM combination over that FULL grid: E29's
# FARM used only the 7 TOP-LEVEL blocks and never the 7 raw-groups.
#
# POOL: best searched run per (family x feature-group), 12 non-onehot groups x 3
# families = 36, plus the 3 GRR arms (E29's e28_grr_{pos,neg,symfe}) = 39 base
# models. Reuses the existing from-scratch runs -- "for the models we have not
# already done" -- rather than re-tuning what is already tuned.
#
# Usage:  bash experiments/e30_gridfarm_crypto.sh   [DRY_RUN=1]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
PREFIX="e30"; TREE="ridge_glmnet"

build_pool() {
  python3 - <<'PY'
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.analysis.results_table import target_of
FAM=["ridge_glmnet","mlp","lgb"]
GROUPKW={"momentum":"momentum","volatility":"volatility","bollinger":"bollinger",
         "relative_strength":"oscillator","sharpe":"oscillator","oscillator":"oscillator",
         "close_":"price","market_cap":"size","volume":"liquidity"}
BLOCKS={"crypto/v2.0-tenc":"tenc","crypto/v2.0-spread":"spread","crypto/v2.0-interact":"interact",
        "crypto/v2.0-fenccausal":"fenccausal","crypto/v2.0-rollsel":"rollsel"}
SUBSETS=["momentum","volatility","oscillator","bollinger","price","size","liquidity",
         "tenc","spread","interact","fenccausal","rollsel"]
def is_search(m,c):
    if c.get("fixed_params"): return m=="ridge_glmnet"
    if c.get("model_options"): return False
    return int(c.get("n_iter") or c.get("bayes_init_points") or 0)>=10 or m=="ridge_glmnet"
def sub(c):
    ds=str(c.get("feature_set") or c.get("data_version")); fi=str(c.get("feature_include") or "")
    if ds in BLOCKS and not fi.strip(): return BLOCKS[ds]
    if ds=="crypto/v2.0" and fi:
        for kw,g in GROUPKW.items():
            if kw in fi: return g
    return None
best={}
for f in FAM:
    for p in glob.glob("results/metrics/crypto/%s/*.json"%f):
        run=os.path.basename(p)[:-5]
        if "clf" in run: continue
        c=json.load(open(p)).get("config",{})
        if c.get("members"): continue
        s=sub(c)
        if s not in SUBSETS or not is_search(f,c): continue
        if target_of("crypto",f,run)!="target_binned_return_20": continue
        pp="results/per_period/crypto/%s/%s.csv"%(f,run)
        if not os.path.exists(pp): continue
        d=pd.read_csv(pp); d=d.set_index("date") if "date" in d.columns else d
        if "corr" not in d.columns or len(d)<470: continue
        if any(k in run.lower() for k in ("leak","fencglobal")): continue
        cc=float(d["corr"].mean()); key=(f,s)
        if key not in best or cc>best[key][1]: best[key]=("%s/%s"%(f,run),cc)
mem=[v[0] for v in best.values()]
for g in ("e28_grr_pos","e28_grr_neg","e28_grr_symfe"):
    if os.path.exists("results/metrics/crypto/gen_ridge/%s.json"%g): mem.append("gen_ridge/%s"%g)
print(" ".join(mem))
PY
}

MEMBERS="$(build_pool)"; NMEM=$(echo ${MEMBERS} | wc -w)
echo "E30 pool: ${NMEM} base models (12 non-onehot feature-groups x 3 families + 3 GRR)"
[ "${NMEM}" -lt 30 ] && { echo ">>> pool too small (${NMEM})"; exit 1; }

ARMS="equal farm1:ridge farm1:l2boost farm1:postalasso farm2:ridge farm2:l2boost farm2:postalasso farm3:ridge farm3:l2boost farm3:postalasso gr ds"
for arm in ${ARMS}; do
  method="${arm%%:*}"; beta="${arm#*:}"; [ "${beta}" = "${arm}" ] && beta="ridge"
  run_name="${PREFIX}_farm_${arm/:/_}"
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '  would run  %-24s %s\n' "${run_name}" "${method}/${beta}"; continue; fi
  echo; echo "--- ${run_name}"
  python3 -u -m src.run_farm_ensemble crypto "${TREE}" ${MEMBERS} \
      --method "${method}" --beta-method "${beta}" --run-name "${run_name}" 2>&1 | tee "logs/${run_name}.log"
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{ echo "E30 -- FARM over the full (model x feature-group) grid, ${NMEM} base models"; echo
  python3 - "${TREE}" "${PREFIX}" <<'PY'
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from src.analysis.results_table import payout_series
tree, prefix = sys.argv[1], sys.argv[2]
def pp(r):
    p="results/per_period/crypto/%s/%s.csv"%(tree,r)
    return pd.read_csv(p).set_index("date") if os.path.exists(p) else None
def nw(x,lag=20):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
base=pp("%s_farm_equal"%prefix)
print("paired vs equal (1/N), Newey-West lag 20")
print("%-22s %8s %8s %10s %10s"%("arm","CORR","payout","dPay(t)","dCORR(t)"))
rows=[]
for path in sorted(glob.glob("results/metrics/crypto/%s/%s_farm_*.json"%(tree,prefix))):
    run=os.path.basename(path)[:-5]; arm=run[len("%s_farm_"%prefix):]
    f=pp(run)
    if f is None: continue
    c=f.index.intersection(base.index)
    dp,tp=nw((payout_series(f).reindex(c)-payout_series(base).reindex(c)).dropna())
    dc,tc=nw((f["corr"].reindex(c)-base["corr"].reindex(c)).dropna())
    rows.append((arm,f["corr"].mean(),float(payout_series(f).mean()),dp,tp,dc,tc,arm=="equal"))
for arm,corr,pay,dp,tp,dc,tc,eq in sorted(rows,key=lambda r:-r[2]):
    print("%-22s %8.4f %8.4f %+8.4f(%+.2f) %+.4f(%+.2f)%s"%(arm,corr,pay,dp,tp,dc,tc,"  <- baseline" if eq else ""))
PY
} | tee "${REPORT}"
echo "wrote ${REPORT}"
