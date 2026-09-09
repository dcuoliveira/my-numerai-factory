#!/usr/bin/env bash
#
# Experiment E46: FARM over the from-scratch ktz SEARCH-PICK models
#
# E46 is the reference handle; every artefact is e46_*.
#
# THE IDEA. Combine the 27 best-tuned (model x dataset) forecasters -- the ones
# each family's from-scratch KTZ-prior search actually PICKED per dataset -- with
# the Lee & Lee FARM combiner, and ask whether a fitted factor-adjusted
# combination beats the plain 1/N rank-average of the same 27. This is the
# E27/E29/E36 FARM programme over the cleanest possible pool: one tuned model per
# (family, dataset), no grid junk, every forecast already on disk.
#
# THE POOL (27 base forecasters, all on target_binned_return_20, all on disk):
#   lgb   lgb_ktz/e29_<ds>        E29  from-scratch RandomizedSearch (60 draws) over the
#                                 E27-prior lgb_ktz grid, CV-best kept, 9 datasets
#   cat   catboost_ktz/e32_sel_<ds>  E32  exhaustive search over the 12-cell catboost_ktz
#                                 grid, CV-best kept, 9 datasets
#   mlp   mlp_ktz2/e34_<ds>       E34  from-scratch search (80 draws) over the E33-reprior
#                                 WIDER mlp_ktz2 grid (width 32-128, alpha {1,10}), 9 datasets
#   datasets: raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp
#
# POOLS: per-family (lgb 9, cat 9, mlp 9) AND the pooled union (all 27, the
# cross-model/cross-dataset regime the book itself blends over). ARMS: equal (the
# 1/N baseline every arm is measured against), farm1/2/3 x {ridge,l2boost,
# postalasso}, gr, ds. Paired Newey-West lag 20 vs the pool's own equal.
#
# WHAT WOULD COUNT AS A RESULT. Prior (E27/E29/E36) is that 1/N wins on this
# panel -- a fitted combiner rarely clears |t|>=2 on ~26 effective windows. The
# honest read is FARM-vs-1/N per pool, and each pool's best arm ALSO ranked
# against the deployed book (report_html) -- does combining tuned singles reach
# the slotblends. Raw scores, no payout factor/clip: ratios, not NMR.
#
# Usage:
#   bash experiments/e46_ktz_searchpick_farm_crypto.sh
#   DRY_RUN=1 ...                 print the plan, run nothing
#   POOLS="all" ...               subset of {lgb cat mlp all}
#   ARMS="equal farm1:l2boost" ...  subset of the FARM arms
#   NO_MAIL=1 ...                 no start/finish email
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E46}"
POOLS="${POOLS:-lgb cat mlp all}"
ARMS="${ARMS:-equal farm1:ridge farm1:l2boost farm1:postalasso farm2:ridge farm2:l2boost farm2:postalasso farm3:ridge farm3:l2boost farm3:postalasso gr ds}"
TREE="ktz_farm"
DS9="raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp"

lgb_members() { for d in ${DS9}; do echo "lgb_ktz/e29_${d}"; done; }
cat_members() { for d in ${DS9}; do echo "catboost_ktz/e32_sel_${d}"; done; }
mlp_members() { for d in ${DS9}; do echo "mlp_ktz2/e34_${d}"; done; }
members_for() { case "$1" in
  lgb) lgb_members;; cat) cat_members;; mlp) mlp_members;;
  all) lgb_members; cat_members; mlp_members;; esac; }

started_all=$(date +%s)
if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  python3 -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  idea      FARM over the 27 from-scratch ktz search-pick models vs 1/N
  pool      lgb E29 (9) + catboost E32-sel (9) + mlp_ktz2 E34 (9) = 27 tuned singles
  pools     ${POOLS}
  arms      equal, farm1/2/3 x {ridge,l2boost,postalasso}, gr, ds
  runner    experiments/e46_ktz_searchpick_farm_crypto.sh
  results   results/{forecast,per_period,metrics}/crypto/${TREE}/e46_*
  report    logs/e46_report.txt
  handle    ${EXPERIMENT}; every artefact is e46_*

A second mail follows on completion." || echo "note: start email failed, continuing"
fi

for pool in ${POOLS}; do
  MEM="$(members_for "$pool" | tr '\n' ' ')"; NMEM=$(echo $MEM | wc -w)
  echo "=== FARM ${pool}: ${NMEM} members ==="
  for arm in ${ARMS}; do
    method="${arm%%:*}"; beta="${arm#*:}"; [ "${beta}" = "${arm}" ] && beta="ridge"
    run_name="e46_${pool}_${arm/:/_}"
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  %-24s %s/%s  (%d members)\n' "${run_name}" "${method}" "${beta}" "${NMEM}"; continue; fi
    [ -f "results/metrics/crypto/${TREE}/${run_name}.json" ] && { echo "  ${run_name} on disk"; continue; }
    python3 -u -m src.run_farm_ensemble crypto "${TREE}" ${MEM} \
        --method "${method}" --beta-method "${beta}" --run-name "${run_name}" -q \
        >> "logs/e46_farm.log" 2>&1 || echo "  FARM FAILED ${run_name}"
  done
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e46_report.txt"
{ echo "${EXPERIMENT} -- FARM over the 27 ktz search-pick models, paired vs equal (1/N), NW lag 20"
  echo "  pools: ${POOLS}"; echo
  POOLS="${POOLS}" TREE="${TREE}" DS9="${DS9}" python3 - <<'PY'
import os, glob
import numpy as np, pandas as pd
from src.analysis.results_table import payout_series
POOLS=os.environ["POOLS"].split(); TREE=os.environ["TREE"]; DS9=os.environ["DS9"].split()
def pp(tree,r):
    p="results/per_period/crypto/%s/%s.csv"%(tree,r)
    return pd.read_csv(p).set_index("date") if os.path.exists(p) else None
def nw(x,lag=20):
    x=np.asarray(x,float); n=len(x)
    if n<3: return float("nan"),float("nan")
    m=x.mean(); e=x-m; v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
# best single member payout per pool (oracle, needs test scores -- context only)
POOLSRC={"lgb":[("lgb_ktz","e29_%s"%d) for d in DS9],
         "cat":[("catboost_ktz","e32_sel_%s"%d) for d in DS9],
         "mlp":[("mlp_ktz2","e34_%s"%d) for d in DS9]}
POOLSRC["all"]=POOLSRC["lgb"]+POOLSRC["cat"]+POOLSRC["mlp"]
for pool in POOLS:
    base=pp(TREE,"e46_%s_equal"%pool)
    if base is None: continue
    best=max((payout_series(f).mean() for f in
              (pp(t,r) for t,r in POOLSRC.get(pool,[])) if f is not None), default=float("nan"))
    print("== %s ==  1/N: CORR %.4f payout %.4f  | best single member payout %.4f"%(
        pool, base["corr"].mean(), payout_series(base).mean(), best))
    print("   %-16s %8s %8s %8s %10s"%("arm","CORR","MMC","payout","dPay(t)"))
    rows=[]
    for path in glob.glob("results/metrics/crypto/%s/e46_%s_*.json"%(TREE,pool)):
        run=os.path.basename(path)[:-5]; arm=run[len("e46_%s_"%pool):]
        f=pp(TREE,run)
        if f is None: continue
        c=f.index.intersection(base.index)
        dp,tp=nw((payout_series(f).reindex(c)-payout_series(base).reindex(c)).dropna())
        rows.append((arm,f["corr"].mean(),f["mmc"].mean(),float(payout_series(f).mean()),dp,tp))
    for arm,corr,mmc,pay,dp,tp in sorted(rows,key=lambda r:-r[3]):
        tag="  <- 1/N baseline" if arm=="equal" else ""
        print("   %-16s %8.4f %8.4f %8.4f %+8.4f(%+.2f)%s"%(arm,corr,mmc,pay,dp,tp,tag))
    print()
PY
} 2>&1 | tee "${REPORT}"
echo "wrote ${REPORT}"

el=$(( ($(date +%s)-started_all)/60 ))
if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "${EXPERIMENT}" "${el}" "${REPORT}" <<'PY' || echo "note: finish email failed"
import sys
from src.live.notify import send_text
exp,el,body=sys.argv[1:4]
try: txt=open(body).read()
except Exception: txt=""
send_text("[crypto] %s finished (%s min)"%(exp,el), txt)
print("emailed the report")
PY
fi
exit 0
