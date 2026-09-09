#!/usr/bin/env bash
#
# Experiments E05 / E06: cumulative feature lags on LightGBM
#
#   SCOPE=all         E05  raw 22 + every lag 1..p of all 22 features
#   SCOPE=volatility  E06  the volatility group and its lags only
#
# The datasets already exist -- crypto/v2.0-cumlag1 .. -cumlag10, built by
# src.data_mani.build_lagged --through p -- so each arm is a --data-version and,
# for E06, one --feature-include away. Nothing is rebuilt here.
#
#     cumlag p   features (all)   features (volatility only)
#     1          44               4
#     5          132              12
#     10         242              22
#
# ---------------------------------------------------------------------------
# What a lag adds, and why LightGBM is the interesting test
# ---------------------------------------------------------------------------
#
# The builder's argument is a linear one: a lag lets RIDGE form a difference it
# cannot otherwise express, b1*f_t + b2*f_{t-p} with b2 negative. That is the
# same trick `spread` used across horizons and `interact` used across
# smoothings, and those are the only blocks that ever cleared |t| > 2 here.
#
# A tree does not need the lag to express a difference -- it can split on both
# columns directly. So the ridge and LightGBM answers should differ, and if
# lags help a booster it is for a different reason: more history to condition
# on, not a representational fix. The ridge cumlag arms already exist
# (w6r6_glmnet_reg_cumlag1..10) for exactly this comparison.
#
# The cost is dilution, and it grows with p. colsample_bytree=0.0702 draws ~1.5
# of 22 columns at p=0 but ~17 of 242 at p=10, so the 22 columns that carry the
# signal appear in a steadily smaller share of trees. Every "more features" arm
# in this study has lost for that reason. E06 exists to separate the two: the
# volatility-only arms grow from 2 columns to 22, so dilution stays mild and any
# effect is closer to the lags themselves.
#
# volatility is the right group to isolate: alone it scored payout 0.067425 at
# t = -0.96 against all 22 -- the only group not significantly worse than the
# full set, on 2 of the 22 columns.
#
# The model is PINNED, so the lag count is the only thing varying. On ~26
# effective independent windows |t| < 2 is the expected outcome.
#
# Usage:
#   SCOPE=all        bash experiments/e05_cumlags_lgb_reg_crypto.sh
#   SCOPE=volatility bash experiments/e05_cumlags_lgb_reg_crypto.sh
#   DRY_RUN=1 SCOPE=all bash experiments/e05_cumlags_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

SCOPE="${SCOPE:-all}"
case "${SCOPE}" in
  all)        EXPERIMENT="${EXPERIMENT:-E05}"; PREFIX="e05_cumlag"; INCLUDE="" ;;
  volatility) EXPERIMENT="${EXPERIMENT:-E06}"; PREFIX="e06_volcumlag"
              INCLUDE='^feature_volatility_' ;;
  *) echo "SCOPE must be 'all' or 'volatility', got '${SCOPE}'" >&2; exit 2 ;;
esac
LAGS="${LAGS:-0 1 2 3 4 5 6 7 8 9 10}"     # 0 = the raw baseline, no lags
C_WINDOW="${C_WINDOW:-6month}"; C_REFIT="${C_REFIT:-6month}"
C_N_SPLITS="${C_N_SPLITS:-2}"; C_N_JOBS="${C_N_JOBS:-8}"; C_SEED="${C_SEED:-42}"
case "${C_WINDOW}" in ""|0|[0-9]) echo "C_WINDOW looks wrong" >&2; exit 1;; esac
PARAMS='{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}'

# Refuse to run on a dataset whose lag depth is not what its name claims.
python3 - "${SCOPE}" ${LAGS} <<'PY' || exit 1
import sys, os, re
import pyarrow.parquet as pq
scope=sys.argv[1]; lags=[int(x) for x in sys.argv[2:]]
bad=[]
for p in lags:
    v="crypto/v2.0" if p==0 else "crypto/v2.0-cumlag%d"%p
    path="data/%s/train.parquet"%v
    if not os.path.exists(path): bad.append("%s MISSING (build_lagged --through %d)"%(v,p)); continue
    cols=[c for c in pq.ParquetFile(path).schema.names if c.startswith("feature_")]
    if scope=="volatility": cols=[c for c in cols if "volatility" in c]
    base=2 if scope=="volatility" else 22
    want=base*(p+1)
    depths={int(m.group(1)) for c in cols for m in [re.search(r"_lag(\d+)$",c)] if m}
    if len(cols)!=want: bad.append("%s: %d columns in scope, expected %d"%(v,len(cols),want))
    if p and depths!=set(range(1,p+1)):
        bad.append("%s: lag depths %s, expected 1..%d"%(v,sorted(depths),p))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d datasets, lag depths and %s column counts as claimed"
      %(len(lags),"volatility" if scope=="volatility" else "full"))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  python3 -c "import sys;from src.live.notify import send_text;send_text('[crypto] %s started'%sys.argv[1],sys.argv[2])" \
    "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  cumulative feature lags, scope=${SCOPE}, LightGBM regression, estimator pinned
  lags: ${LAGS}   (0 = raw baseline)

Where to look:
  runner   experiments/e05_cumlags_lgb_reg_crypto.sh
  results  results/metrics/crypto/lgb/${PREFIX}*.json
  report   logs/${PREFIX}_report.txt
  handle   ${EXPERIMENT}" || echo "note: start email failed"
fi

started=$(date +%s); failed=0
echo "=============================================================="
echo "${EXPERIMENT}: cumulative lags, scope=${SCOPE}, model pinned"
echo "=============================================================="
for p in ${LAGS}; do
  if [ "$p" = "0" ]; then data="crypto/v2.0"; else data="crypto/v2.0-cumlag${p}"; fi
  n="${PREFIX}${p}"
  cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
       --cv-scheme rolling --train-window "${C_WINDOW}" --refit-every "${C_REFIT}"
       --data-version "${data}" --fixed-params "${PARAMS}"
       --n-splits "${C_N_SPLITS}" --n-jobs "${C_N_JOBS}" --era-step 1
       --seed "${C_SEED}" --run-name "$n")
  [ -n "${INCLUDE}" ] && cmd+=(--feature-include "${INCLUDE}")
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  would run $n on ${data}"; continue; fi
  echo; echo "--- $n  ${data}  $(date '+%H:%M:%S')"
  "${cmd[@]}" 2>&1 | tee "logs/${n}.log" >/dev/null
  [ "${PIPESTATUS[0]}" -ne 0 ] && { failed=$((failed+1)); echo "  FAILED $n"; } || echo "  done $n"
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- cumulative lags, scope=${SCOPE}, LightGBM regression"
echo "  results results/metrics/crypto/lgb/${PREFIX}*.json"
echo "  ran in $(( ($(date +%s)-started)/60 )) min, ${failed} failed"
echo
python3 - "${PREFIX}" <<'PY'
import json, glob, os, re, sys
import numpy as np, pandas as pd
pre=sys.argv[1]
def pp(r):
    f=pd.read_csv("results/per_period/crypto/lgb/%s.csv"%r)
    return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb/%s*.json"%pre)):
    n=os.path.basename(p)[:-5]; lag=int(re.search(r"(\d+)$",n).group(1))
    d=json.load(open(p))
    rows.append((lag,n,d["config"]["n_features"],d["corr"]["mean"],d["mmc"]["mean"],
                 d["corr"]["sharpe"],d["corr"]["max_drawdown"],
                 0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
if not rows: print("no results"); raise SystemExit
rows.sort()
base=pp("%s0"%pre)
print("%-5s %5s %10s %10s %8s %8s %10s %11s %7s"%(
  "lag","feat","CORR","MMC","sharpe","maxdd","payout","d vs lag0","t"))
print("-"*88)
for lag,n,f,c,m,s,dd,pay in rows:
    if lag==0: d=t=None
    else:
        j=pd.concat([payout(base),payout(pp(n))],axis=1,join="inner",keys=["b","a"])
        d,t=nw(j["a"]-j["b"])
    print("%-5d %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s"%(
      lag,f,c,m,s,dd,pay,"baseline" if d is None else "%+.6f"%d,
      "-" if t is None else "%+.2f"%t))
print("-"*88)
best=max(rows,key=lambda r:r[7])
print("best lag depth by payout: %d (%.6f)"%(best[0],best[7]))
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s"%(sys.argv[3],sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
