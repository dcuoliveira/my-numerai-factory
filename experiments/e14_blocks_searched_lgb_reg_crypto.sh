#!/usr/bin/env bash
#
# Experiment E14: every feature block, LightGBM trained FROM SCRATCH
#
# E14 is the reference handle. Everything this study writes is e14_*.
#
# Six arms, each a full 50-draw randomised search at every refit -- the block
# gets hyperparameters chosen for its own width, not the raw 22's:
#
#     arm         dataset                feat
#     ------------------------------------------
#     tenc        crypto/v2.0-tenc         23
#     spread      crypto/v2.0-spread       33
#     interact    crypto/v2.0-interact     36
#     cumlag1     crypto/v2.0-cumlag1      44
#     fenccausal  crypto/v2.0-fenccausal   44
#     rollsel     crypto/v2.0-rollsel      46
#
# ---------------------------------------------------------------------------
# What is NOT here, and why
# ---------------------------------------------------------------------------
#
# THE 7 RAW FEATURE GROUPS ARE ALREADY DONE FROM SCRATCH. price, size,
# liquidity, oscillator, bollinger, momentum and volatility each ran a 50-draw
# search in the lgb_6m6m_reg study (w6r6_lgb_grp_*_reg, 97-151 min each). They
# are not re-run; the report pulls them in for the ranking.
#
# THE BASELINE IS NOT RE-RUN EITHER. e13_raw_search is raw 22, searched, seed
# 42, same cadence, on the CURRENT panel -- identical protocol to these arms.
# Re-running it would cost ~5 hours to reproduce a number already on disk.
#
# rollsel had no LightGBM run at all, searched or pinned. cumlag1 is the arm
# E05 favoured: payout falls monotonically with cumlag depth (cumlag1 0.079724,
# cumlag2 0.076960, ... cumlag10 0.073777), so 1 is the least-diluted lag depth
# and the fairest single representative of the block.
#
# ---------------------------------------------------------------------------
# Why from scratch, when every one of these has a pinned twin
# ---------------------------------------------------------------------------
#
# The pinned config was selected on the raw 22 features, where
# colsample_bytree=0.0702 draws ~1.5 columns per tree. On a 46-column block it
# draws ~3, of which only ~1.5 are raw. A pinned null on a wide block is
# therefore ambiguous between "the block carries nothing" and "the sampling
# rate buried it".
#
# E13 showed that ambiguity is worth resolving but does not always resolve in
# the block's favour: given 50 draws on the 89-column ADL set the search raised
# colsample_bytree to 0.2846, four times the pinned rate -- and still scored
# WORSE than pinned (-0.002369). So the honest prior here is that searching
# will not rescue a block either. What it does is remove the excuse.
#
# ---------------------------------------------------------------------------
# Runtime: price this before launching
# ---------------------------------------------------------------------------
#
# A searched arm is ~100-320 min (50 draws x 5 folds x 4 refits). Measured on
# this panel: raw 22 took 159 and 321 min on two occasions, the 2-4 feature
# groups 97-151 min. Six arms is roughly 12-18 HOURS. The pinned twins of these
# same arms took 1-2 min each.
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Baseline is e13_raw_search, paired, Newey-West lag 20. On ~26 effective
# independent windows |t| < 2 is expected and a null is the finding.
#
# Eleven blocks have now been tried pinned and none beat raw by a margin the
# sample supports. A block clearing |t| > 2 HERE would say those nulls were an
# artefact of the pinned sampling rate rather than a property of the columns --
# which would be the first such result in the programme, and would mean every
# pinned feature-block conclusion needs revisiting.
#
# Usage:
#   bash experiments/e14_blocks_searched_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e14_blocks_searched_lgb_reg_crypto.sh
#   ONLY="interact rollsel" bash experiments/e14_blocks_searched_lgb_reg_crypto.sh
#   SHARD=1 OF=2 bash experiments/e14_blocks_searched_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e14_blocks_searched_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E14}"
E14_WINDOW="${E14_WINDOW:-6month}"
E14_REFIT="${E14_REFIT:-6month}"
E14_N_ITER="${E14_N_ITER:-50}"
E14_N_SPLITS="${E14_N_SPLITS:-5}"
E14_N_JOBS="${E14_N_JOBS:-8}"
E14_SEARCH_JOBS="${E14_SEARCH_JOBS:-4}"
E14_SEED="${E14_SEED:-42}"
BASELINE="${BASELINE:-e13_raw_search}"
case "${E14_WINDOW}" in ""|0|[0-9]) echo "E14_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add tenc       "crypto/v2.0-tenc"        23
add spread     "crypto/v2.0-spread"      33
add interact   "crypto/v2.0-interact"    36
add cumlag1    "crypto/v2.0-cumlag1"     44
add fenccausal "crypto/v2.0-fenccausal"  44
add rollsel    "crypto/v2.0-rollsel"     46

python3 - "${BASELINE}" "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, json, pyarrow.parquet as pq
baseline = sys.argv[1]; cut = sys.argv.index("--")
versions, wants = sys.argv[2:cut], [int(x) for x in sys.argv[cut+1:]]
bad = []
for v, w in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    if not os.path.exists(p):
        bad.append("%s MISSING" % v); continue
    g = sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if g != w:
        bad.append("%s has %d features, expected %d" % (v, g, w))
# The baseline must exist AND have been searched, or the comparison crosses
# protocols as well as feature sets.
bp = "results/metrics/crypto/lgb/%s.json" % baseline
if not os.path.exists(bp):
    bad.append("baseline %s has no metrics file" % baseline)
else:
    c = json.load(open(bp))["config"]
    if c.get("fixed_params"):
        bad.append("baseline %s was PINNED, not searched -- these arms are searched, "
                   "so the comparison would move two things at once" % baseline)
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d datasets, counts match, baseline %s is searched" % (len(versions), baseline))
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}, each a full ${E14_N_ITER}-draw search at every refit
            tenc 23, spread 33, interact 36, cumlag1 44, fenccausal 44, rollsel 46
  asks      does any feature block beat raw once it gets its OWN hyperparameters?
  baseline  ${BASELINE} (raw 22, searched, same panel and protocol)
  expect    ~12-18 HOURS -- a searched arm is ~100-320 min

  not re-run: the 7 raw feature groups are already searched
              (w6r6_lgb_grp_*_reg); the report pulls them in.

Where to look while it runs:
  runner    experiments/e14_blocks_searched_lgb_reg_crypto.sh
  progress  logs/e14_*.log
  results   results/metrics/crypto/lgb/e14_*.json
  report    logs/e14_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e14_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: feature blocks, LightGBM from scratch  |  ${total} arms"
echo "baseline ${BASELINE}   ${E14_N_ITER} draws x ${E14_N_SPLITS} folds per refit"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  shard ${SHARD}/${OF}"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="e14_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
         --cv-scheme rolling --train-window "${E14_WINDOW}" --refit-every "${E14_REFIT}"
         --data-version "${DATA[$i]}"
         --n-iter "${E14_N_ITER}" --n-splits "${E14_N_SPLITS}"
         --n-jobs "${E14_N_JOBS}" --search-jobs "${E14_SEARCH_JOBS}"
         --era-step 1 --seed "${E14_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-16s %-26s %3s feat\n' "$index" "$total" "$run_name" "${DATA[$i]}" "${NFEAT[$i]}"; continue; fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${DATA[$i]}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e14_report.txt"
{
echo "${EXPERIMENT} -- feature blocks, LightGBM trained from scratch"
echo "  results  results/metrics/crypto/lgb/e14_*.json"
echo "  runner   experiments/e14_blocks_searched_lgb_reg_crypto.sh"
echo "  logs     logs/e14_*.log"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${BASELINE}" <<'PY'
import json, glob, os, sys
import numpy as np, pandas as pd
baseline = sys.argv[1]
def pp(r):
    p="results/per_period/crypto/lgb/%s.csv"%r
    if not os.path.exists(p): return None
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
runs=[(os.path.basename(p)[:-5], "block") for p in sorted(glob.glob("results/metrics/crypto/lgb/e14_*.json"))]
runs += [(os.path.basename(p)[:-5], "group") for p in
         sorted(glob.glob("results/metrics/crypto/lgb/w6r6_lgb_grp_*_reg.json"))]
runs += [(baseline, "baseline")]
b = pp(baseline)
rows=[]
for r,kind in runs:
    f=pp(r)
    if f is None: continue
    d=json.load(open("results/metrics/crypto/lgb/%s.json"%r))
    p=payout(f); c=p.cumsum()
    delta=t=None; n=len(f)
    if r!=baseline and b is not None:
        j=pd.concat([payout(b),p],axis=1,join="inner",keys=["b","a"])
        delta,t=nw(j["a"]-j["b"]); n=len(j)
    rows.append((p.mean(), r.replace("e14_","").replace("w6r6_lgb_grp_","").replace("_reg",""),
                 kind, d["config"]["n_features"], f["corr"].mean(), f["mmc"].mean(),
                 p.mean()/p.std(ddof=0), (np.maximum.accumulate(c)-c).max(), delta, t, n))
print("%-14s %-9s %5s %10s %10s %8s %8s %10s %11s %7s %5s"%(
  "arm","kind","feat","CORR","MMC","sharpe","maxdd","payout","d vs raw","t","n"))
print("-"*104)
for pay,name,kind,feat,corr,mmc,sh,dd,delta,t,n in sorted(rows,reverse=True):
    print("%-14s %-9s %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s %5d"%(
      name,kind,feat,corr,mmc,sh,dd,pay,
      "baseline" if delta is None else "%+.6f"%delta,
      "-" if t is None else "%+.2f"%t, n))
print("-"*104)
sig=[r for r in rows if r[9] is not None and abs(r[9])>2]
print("\narms clearing |t| > 2 against %s: %d"%(baseline,len(sig)))
for r in sorted(sig,key=lambda x:-abs(x[9])): print("   %-14s t = %+.2f"%(r[1],r[9]))
if not sig: print("   none -- on ~26 effective independent windows that is the expected result")
print("\nNOTE the groups were searched on the PRE-REFRESH panel; deltas above are")
print("paired on shared dates only, and the n column shows how many.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
