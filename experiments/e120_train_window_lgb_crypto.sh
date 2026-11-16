#!/usr/bin/env bash
#
# E120 -- how much HISTORY should each refit use? The training window of the
#         best single LightGBM, from one week to six months. The LightGBM twin
#         of E117 (which asked this of the best single ridge).
#
# The model is the payout-best standalone LightGBM: e03_concat_trio -- the
# canonical pinned booster (num_leaves 255, 4327 trees, lr 0.00257, colsample
# 0.07) on the group-CONCAT feature subset of crypto/v2.0-onehot: market-cap +
# momentum + volatility + the per-symbol one-hot block, 748 features. On the
# 20-day label it pays 0.09074 (CORR 0.14321, MMC 0.07642), the best standalone
# LightGBM in the book. Everything about it is held fixed -- data, features,
# the pinned hyperparameters, 130-period refit cadence, embargo 20, seed -- and
# ONLY the rolling training window moves.
#
# ---------------------------------------------------------------------------
# The windows, in PERIODS -- identical to E117. The panel is weekday-only:
# 5.01 periods/week, 21.8/month, 300 rows on every recent period.
# ---------------------------------------------------------------------------
#
#     1 week  5    2 weeks 10    3 weeks 15    4 weeks 20
#     1 month 22   3 months 65   6 months 130 (the deployed window, = e03_concat_trio)
#
# 4 weeks (20) and 1 month (22) are two periods apart and both run because E117
# ran both; expect them indistinguishable.
#
# ---------------------------------------------------------------------------
# The one simplification vs E117: NO lambda search, so NO --tune-window
# ---------------------------------------------------------------------------
#
# E117 had to search ridge's lambda from scratch at every refit, and a purged
# inner fold cannot fit inside a <=22-period window, so it leaned on
# --tune-window 130 to host the validation slices. This model is PINNED
# (--fixed-params), so run_forecast skips the search entirely (run_forecast.py
# `if not args.fixed_params`) -- every window is just the same booster refitted
# on that much history. There is no fold to place, so no tune-window is needed
# and every arm is a clean, directly-paired variant of the 130 control.
#
# ---------------------------------------------------------------------------
# What this does NOT vary, and why that matters for reading it
# ---------------------------------------------------------------------------
#
# The REFIT CADENCE stays at 130 and the TREE COUNT stays at 4327. A 1-week
# window therefore fits a 4327-tree booster once every six months on ~1,500
# rows of 748 columns and predicts the next 130 periods from it. That is the
# literal question -- how much history a refit should use, at the cadence the
# account runs -- NOT the freshness question (short window AND short cadence),
# which is E116's axis. Holding the tree count fixed is deliberate: varying it
# would change the model, not the window. Expect the short windows to overfit
# and lose badly -- 5 periods at ~95% target overlap is ~one effective
# observation under 4327 trees -- exactly as E117 expected for ridge.
#
# WHAT WOULD COUNT AS A RESULT  any window beating the deployed 130 on payout,
# paired Newey-West lag 20 (every arm shares dates/cadence/seed with w130, so
# the delta series is tightly paired). A null -- 130 is already right -- is the
# expected and reportable outcome. The w130 arm must reproduce e03_concat_trio.
#
# COST  e03_concat_trio ran in 1.6 min; short windows are faster -> whole study
# ~10-15 min serial on one box.
#
# Usage:
#   bash experiments/e120_train_window_lgb_crypto.sh
#   DRY_RUN=1 / ONLY=e120_w5 / NO_MAIL=1 bash experiments/e120_train_window_lgb_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT="${EXPERIMENT:-E120}"
DATA="crypto/v2.0-onehot"
INCLUDE='(^feature_market_cap_(avg|ewa)_)|(^feature_momentum_)|(^feature_volatility_)|(^feature_sym_)'
# e03_concat_trio's pinned booster, verbatim (the w130 arm must reproduce it).
FIXED='{"colsample_bytree": 0.070213, "learning_rate": 0.002572, "max_depth": 8, "min_child_samples": 50, "n_estimators": 4327, "num_leaves": 255, "reg_lambda": 1.757511, "subsample": 0.816153, "subsample_freq": 1}'
E120_N_JOBS="${E120_N_JOBS:-6}"

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add e120_w5     5  "1 week"
add e120_w10   10  "2 weeks"
add e120_w15   15  "3 weeks"
add e120_w20   20  "4 weeks"
add e120_w22   22  "1 month"
add e120_w65   65  "3 months"
add e120_w130 130  "6 months (control = e03_concat_trio)"

# Gate: the dataset still carries 762 feature columns, the include regex still
# keeps exactly 748, and the 300-rows-per-period cross-section still holds.
"${PYTHON}" - "${DATA}" "${INCLUDE}" <<'PY' || exit 1
import re, sys
import pandas as pd, pyarrow.parquet as pq
data, inc = sys.argv[1], sys.argv[2]
path = "data/%s/train.parquet" % data
names = [c for c in pq.ParquetFile(path).schema.names if c.startswith("feature_")]
kept = [c for c in names if re.search(inc, c)]
if (len(names), len(kept)) != (762, 748):
    sys.exit("expected 762 -> 748 features, found %d -> %d" % (len(names), len(kept)))
d = pd.read_parquet(path, columns=["date"]).reset_index()
per = d.groupby("date").size().iloc[-800:]
if per.min() != 300 or per.max() != 300:
    sys.exit("rows/period over last 800 dates span %d..%d, not a fixed 300"
             % (per.min(), per.max()))
print("gate OK: 762 -> 748 features; 300 rows on every one of the last 800 periods")
PY

total="${#NAMES[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: training window 1 week -> 6 months of the best standalone LightGBM
  base      e03_concat_trio (pinned 4327-tree booster, 748 feat on crypto/v2.0-onehot)
  windows   5 10 15 20 22 65 130 periods; refit 130, embargo 20, seed 42; ONLY the window moves
  twin of   E117 (same sweep on the best single ridge); pinned, so no lambda search / tune-window
  results   results/metrics/crypto/lgb/e120_*.json   report logs/e120_report.txt

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: training window of the best standalone LightGBM | ${total} arms"
echo "base e03_concat_trio (pinned); windows 5..130; refit 130; ONLY the window moves"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index+1))
    name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    cmd=("${PYTHON}" -u -m src.run_forecast crypto lgb
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}" --refit-every 130 --embargo 20
         --val-frac 0.3 --seed 42 --n-jobs "${E120_N_JOBS}"
         --data-version "${DATA}" --feature-include "${INCLUDE}"
         --fixed-params "${FIXED}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s window=%-3s  %s\n' "$index" "$total" "$name" "$window" "$why"; continue; fi
    ran=$((ran+1)); started=$(date +%s)
    echo; echo "[${index}/${total}] ${name}  window=${window}  ${why}  $(date '+%H:%M:%S')"
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log" >/dev/null
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${name}")
        echo ">>> ${name} FAILED (exit ${status}) after $((el))s -- continuing"
    else
        echo ">>> ${name} done in $((el))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e120_report.txt"
{
echo "${EXPERIMENT} -- training window of the best standalone LightGBM (e03_concat_trio)"
echo "  results  results/metrics/crypto/lgb/e120_*.json ; runner e120_train_window_lgb_crypto.sh"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
"${PYTHON}" - <<'PY'
import os, glob
import numpy as np, pandas as pd
def ser(run):
    p="results/per_period/crypto/lgb/%s.csv"%run
    if not os.path.exists(p): return None
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def summ(p):
    c=p.cumsum(); dd=(np.maximum.accumulate(c)-c).max()
    return p.mean(), (p.mean()/p.std(ddof=0) if p.std(ddof=0)>0 else float("nan")), dd
def nw(x,lag=20):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
WIN=[("e120_w5",5),("e120_w10",10),("e120_w15",15),("e120_w20",20),
     ("e120_w22",22),("e120_w65",65),("e120_w130",130)]
base=ser("e120_w130"); bp=payout(base) if base is not None else None
# reproduction check against e03_concat_trio
ct=ser("e03_concat_trio")
if base is not None and ct is not None:
    cp=payout(ct)
    print("w130 vs e03_concat_trio (reproduction): payout %.6f vs %.6f  (dmean %.2e)"
          %(bp.mean(), cp.mean(), bp.mean()-cp.mean()))
print()
print("%-10s %4s %10s %10s %10s %8s %8s %6s   dpay vs w130 (t)"%(
    "arm","win","CORR","MMC","payout","p.sh","dd(pay)","n"))
print("-"*104)
rows=[]
for name,w in WIN:
    f=ser(name)
    if f is None: continue
    p=payout(f); pm,psh,pdd=summ(p); cm=f["corr"].mean(); mm=f["mmc"].mean()
    d=t=None
    if bp is not None and name!="e120_w130":
        j=pd.concat([p,bp],axis=1,join="inner",keys=["a","b"]); d,t=nw(j["a"]-j["b"])
    rows.append((w,name,cm,mm,pm,psh,pdd,len(p),d,t))
for w,name,cm,mm,pm,psh,pdd,n,d,t in rows:
    ds="-" if d is None else "%+.6f (t %+.2f)"%(d,t)
    print("%-10s %4d %10.6f %10.6f %10.6f %8.3f %8.4f %6d   %s"%(name,w,cm,mm,pm,psh,pdd,n,ds))
print("-"*104)
print("Paired Newey-West lag 20 vs w130 (the deployed 6-month window). +t => the")
print("shorter/longer window beats the deployed one. payout = 0.1*CORR + MMC.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "${PYTHON}" - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
