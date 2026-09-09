#!/usr/bin/env bash
#
# Experiment E02: group BLEND vs group CONCAT, on LightGBM
#
# The ridge group-blend study, re-asked of a model class whose answer should
# differ. Two structurally different ways to use several feature groups:
#
#   BLEND   fit each group as its own model, rank its predictions within each
#           period, average the ranks. N estimators, each tuned to its own block.
#           This is also exactly what splitting a stake across N slots does.
#   CONCAT  one model on the union of those groups' columns. One estimator.
#
# WHY THE ANSWER SHOULD DIFFER FROM RIDGE. glmnet applies ONE alpha across every
# column, so a 4-column price block and a 2-column momentum block cannot be
# regularised differently inside one fit -- the blend can do that and the concat
# cannot. That constraint is why blending was worth testing for ridge. A tree has
# no such constraint: it already splits on whichever column helps, at whatever
# depth, so CONCAT should lose much less here. If the blend still wins, the
# advantage is not about shrinkage and the ridge explanation was incomplete.
#
# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
#
#   7 group arms, pinned          each of the 7 raw groups fitted alone
#   best trio  BLEND / CONCAT     from screening every 3-subset
#   best quintet BLEND / CONCAT   from screening every 5-subset
#   raw 22, pinned                the baseline everything is measured against
#
# The 7 group arms are RE-RUN PINNED rather than reusing w6r6_lgb_grp_*_reg,
# which searched 50 draws each. Mixing searched members into a blend that is
# then compared against a pinned concat would move two things at once. Pinning
# all of it costs ~2 min per arm instead of ~100.
#
# Baseline: e02_raw (pinned, 22 features). The group arms individually scored
# 0.0099-0.0675 payout in the searched study against raw's 0.0833, so the bar is
# high -- 12 of the 22 raw features carry roughly three features' worth of
# information, and splitting them up throws away the rest.
#
# On ~26 effective independent windows |t| < 2 is the expected outcome.
#
# Usage:
#   bash experiments/e02_group_blends_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e02_group_blends_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e02_group_blends_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
EXPERIMENT="${EXPERIMENT:-E02}"
E2_WINDOW="${E2_WINDOW:-6month}"; E2_REFIT="${E2_REFIT:-6month}"
E2_N_SPLITS="${E2_N_SPLITS:-2}"; E2_N_JOBS="${E2_N_JOBS:-8}"; E2_SEED="${E2_SEED:-42}"
case "${E2_WINDOW}" in ""|0|[0-9]) echo "E2_WINDOW looks wrong" >&2; exit 1;; esac
PARAMS='{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}'

GNAME=(); GRE=()
g() { GNAME+=("$1"); GRE+=("$2"); }
g price      '^feature_close_(avg|ewa)_'
g size       '^feature_market_cap_(avg|ewa)_'
g liquidity  '^feature_volume_(avg|ewa)_'
g oscillator '^feature_(relative_strength_index|sharpe_ratio)_'
g bollinger  '^feature_bollinger_'
g momentum   '^feature_momentum_'
g volatility '^feature_volatility_'

# The groups must exactly partition the 22 raw features, or every number here
# describes a different experiment than the header claims.
python3 - "${GRE[@]}" <<'PY' || exit 1
import re, sys, pyarrow.parquet as pq
cols=[c for c in pq.ParquetFile("data/crypto/v2.0/train.parquet").schema.names if c.startswith("feature_")]
seen=set(); total=0
for pat in sys.argv[1:]:
    hit={c for c in cols if re.search(pat,c)}
    if not hit: sys.exit("pattern %r matched nothing"%pat)
    if hit & seen: sys.exit("patterns overlap on %s"%sorted(hit&seen))
    seen|=hit; total+=len(hit)
miss=set(cols)-seen
if miss: sys.exit("groups do not cover %d feature(s): %s"%(len(miss),sorted(miss)))
print("preflight OK: %d groups partition all %d raw features"%(len(sys.argv)-1,total))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  python3 -c "import sys;from src.live.notify import send_text;send_text('[crypto] %s started'%sys.argv[1],sys.argv[2])" \
    "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  group BLEND vs group CONCAT on LightGBM, estimator pinned
  7 group arms + raw baseline, then the best trio and quintet each way
  expect ~25 min

Where to look:
  runner    experiments/e02_group_blends_lgb_reg_crypto.sh
  progress  logs/e02_*.log
  results   results/metrics/crypto/lgb/e02_*.json
  report    logs/e02_report.txt
  handle    E02; every artefact is e02_*

A second mail follows on completion." || echo "note: start email failed"
fi

started_all=$(date +%s); failed=0
one() { # run_name, extra args...
  local n="$1"; shift
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  would run $n"; return; fi
  echo; echo "--- $n  $(date '+%H:%M:%S')"
  python3 -u -m src.run_forecast crypto lgb --task regression --metric corr \
    --cv-scheme rolling --train-window "${E2_WINDOW}" --refit-every "${E2_REFIT}" \
    --data-version crypto/v2.0 --fixed-params "${PARAMS}" \
    --n-splits "${E2_N_SPLITS}" --n-jobs "${E2_N_JOBS}" --era-step 1 --seed "${E2_SEED}" \
    --run-name "$n" "$@" 2>&1 | tee "logs/${n}.log" >/dev/null
  [ "${PIPESTATUS[0]}" -ne 0 ] && { failed=$((failed+1)); echo "  FAILED $n"; } || echo "  done $n"
}

echo "=============================================================="
echo "${EXPERIMENT}: group blend vs concat, LightGBM, model pinned"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
one e02_raw
for i in "${!GNAME[@]}"; do one "e02_grp_${GNAME[$i]}" --feature-include "${GRE[$i]}"; done

[ "${DRY_RUN:-0}" = "1" ] && { echo "  (blends and concats are chosen from the screen at runtime)"; exit 0; }

# Screen every 3- and 5-subset of the seven group arms, then build only the
# winners. Screening is a RANK, not a price -- the shortlist is run properly.
python3 - > /tmp/e02_pick.txt <<'PY'
import itertools, os
import numpy as np, pandas as pd
G=["price","size","liquidity","oscillator","bollinger","momentum","volatility"]
S={}
for g in G:
    p="results/per_period/crypto/lgb/e02_grp_%s.csv"%g
    if os.path.exists(p):
        f=pd.read_csv(p); f=f.set_index("date") if "date" in f.columns else f
        S[g]=0.1*f["corr"]+f["mmc"]
F=pd.DataFrame(S).dropna()
best={}
for r in (3,5):
    cand=[(F[list(c)].mean(axis=1).mean()/F[list(c)].mean(axis=1).std(ddof=0), c)
          for c in itertools.combinations(F.columns,r)]
    sh,c=max(cand); best[r]=c
    print(" ".join(c))
PY
TRIO=$(sed -n 1p /tmp/e02_pick.txt); QUINT=$(sed -n 2p /tmp/e02_pick.txt)
echo; echo "screen picked  trio: ${TRIO}   quintet: ${QUINT}"

mem() { for g in $1; do printf "e02_grp_%s " "$g"; done; }
re()  { local out=""; for g in $1; do for i in "${!GNAME[@]}"; do
          [ "${GNAME[$i]}" = "$g" ] && out="${out}|${GRE[$i]#^}"; done; done
        echo "^(${out#|})"; }

python3 -m src.run_ensemble crypto lgb $(mem "${TRIO}")  --run-name e02_blend_trio    -q >/dev/null && echo "  done e02_blend_trio"
python3 -m src.run_ensemble crypto lgb $(mem "${QUINT}") --run-name e02_blend_quintet -q >/dev/null && echo "  done e02_blend_quintet"
one e02_concat_trio    --feature-include "$(re "${TRIO}")"
one e02_concat_quintet --feature-include "$(re "${QUINT}")"

REPORT="logs/e02_report.txt"
{
echo "${EXPERIMENT} -- group blend vs concat, LightGBM"
echo "  trio    ${TRIO}"
echo "  quintet ${QUINT}"
echo "  results results/metrics/crypto/lgb/e02_*.json"
echo "=============================================================="
echo "ran in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
echo
python3 - <<'PY'
import json, glob, os
import numpy as np, pandas as pd
def pp(r):
    f=pd.read_csv("results/per_period/crypto/lgb/%s.csv"%r)
    return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb/e02_*.json")):
    d=json.load(open(p)); n=os.path.basename(p)[:-5]
    rows.append(dict(run=n,feat=d["config"]["n_features"],corr=d["corr"]["mean"],
        mmc=d["mmc"]["mean"],sharpe=d["corr"]["sharpe"],maxdd=d["corr"]["max_drawdown"],
        payout=0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
t=pd.DataFrame(rows); base=pp("e02_raw")
print("%-24s %5s %10s %8s %8s %10s %11s %7s"%("run","feat","CORR","sharpe","maxdd","payout","d vs raw","t"))
print("-"*94)
for _,r in t.sort_values("payout",ascending=False).iterrows():
    if r["run"]=="e02_raw": d=tt=None
    else:
        j=pd.concat([payout(base),payout(pp(r["run"]))],axis=1,join="inner",keys=["b","a"])
        d,tt=nw(j["a"]-j["b"])
    print("%-24s %5d %10.6f %8.4f %8.4f %10.6f %11s %7s"%(r["run"],r["feat"],r["corr"],
        r["sharpe"],r["maxdd"],r["payout"],"baseline" if d is None else "%+.6f"%d,
        "-" if tt is None else "%+.2f"%tt))
print("-"*94)
for k in ("trio","quintet"):
    b,c=t[t.run=="e02_blend_%s"%k],t[t.run=="e02_concat_%s"%k]
    if len(b) and len(c):
        j=pd.concat([payout(pp("e02_concat_%s"%k)),payout(pp("e02_blend_%s"%k))],
                    axis=1,join="inner",keys=["c","b"])
        d,tt=nw(j["b"]-j["c"])
        print("%-8s BLEND %.6f vs CONCAT %.6f  ->  blend %+.6f, t %+.2f"
              %(k,b["payout"].iloc[0],c["payout"].iloc[0],d,tt))
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
