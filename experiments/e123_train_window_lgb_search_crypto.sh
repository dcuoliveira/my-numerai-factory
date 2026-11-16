#!/usr/bin/env bash
#
# E123 -- how much HISTORY should each refit use, when the booster is SEARCHED
#         FROM SCRATCH at every window? The from-scratch-search twin of E120.
#
# E120 asked the window question of the single payout-best LightGBM with its
# hyperparameters PINNED (e03_concat_trio, verbatim) -- so a short window was
# the exact 4327-tree booster refitted on less data, and it lost monotonically.
# E123 asks the SAME window question but lets a full LightGBM search pick the
# booster afresh at every window. This is the real analogue of E117, which
# searched ridge's lambda from scratch per window; E120 was the pinned
# simplification. The question E123 answers that E120 could not:
#
#     when the model is allowed to get SIMPLER as its history shrinks, does a
#     short window still lose -- or does a re-tuned booster recover the payout
#     that pinning the big booster threw away?
#
# Base, held fixed and identical to E120: model class lgb (the class
# concat_trio belongs to), data crypto/v2.0-onehot, the group-CONCAT feature
# subset (market-cap + momentum + volatility + per-symbol one-hot, 748 feat),
# 130-period refit cadence, embargo 20, seed 42. ONLY the training window moves,
# and -- the one change from E120 -- the hyperparameters are searched, not given.
#
# ---------------------------------------------------------------------------
# The windows, in PERIODS -- identical to E117 and E120. Weekday-only panel:
# ~5 periods/week, ~22/month, 300 rows on every recent period.
# ---------------------------------------------------------------------------
#
#     1 week  5    2 weeks 10    3 weeks 15    4 weeks 20
#     1 month 22   3 months 65   6 months 130 (the deployed window)
#
# ---------------------------------------------------------------------------
# The search, and why the short windows need a tune-window (E117's mechanism)
# ---------------------------------------------------------------------------
#
# A from-scratch random search of the plain `lgb` grid: --n-iter 60 candidates,
# --n-splits 5 purged inner folds, --metric corr. The grid spans concat_trio's
# region on every axis -- n_estimators 2000..30000, lr 0.001..0.01, colsample
# 0.05..0.30, subsample 0.5..1.0, num_leaves 31..1023 (max_depth 5..10),
# reg_lambda 1..1000 -- with ONE deliberate exception: min_child_samples floors
# at 1000, where concat_trio sits at 50. So the search explores the standard,
# more-regularised leaf regime and is NOT expected to reproduce the pinned
# concat_trio at w130. That is the point of "from scratch": it uses the
# pipeline's current lgb search space, not the old bespoke pin.
#
# A purged inner fold needs train + 20-period embargo + validation INSIDE the
# block it searches, so purged_period_splits raises "no usable CV folds" for any
# window of 22 periods or fewer -- a month holds no held-out fold. The fix is
# --tune-window 130 (exactly as E117): at each refit the search runs on the 130
# periods ending where the training window ends, with rolling inner folds that
# each TRAIN ON THE WINDOW'S LENGTH, so every candidate is scored as it will be
# used; the winner is then refitted on the training window alone. The 130 arm
# needs no tuning block -- it searches its own window exactly as production
# would. Unlike E117's glmnet, lgb's grid is scale-free (num_leaves, lr, ...),
# so there is no lambda-path-strength correction to make.
#
# ---------------------------------------------------------------------------
# What this does NOT vary
# ---------------------------------------------------------------------------
#
# The REFIT CADENCE stays at 130. A 1-week window is fitted once every six
# months on one week of data and predicts the next 130 periods -- the literal
# "how much history per refit" question at the cadence the account runs, NOT the
# freshness question (short window AND short cadence), which is E116's axis.
#
# WHAT WOULD COUNT AS A RESULT  any searched window beating the deployed pinned
# concat_trio (e120_w130 = 0.09074) on payout, paired Newey-West lag 20; OR the
# searched arms beating their pinned E120 twins window-for-window (searching
# recovers loss). The expected null -- 130 is still right and searching cannot
# rescue a starved window -- is reportable.
#
# COST  a from-scratch search is ~60x5 fits per refit, ~4 refits per arm; the
# w65/w130 arms fit folds on 65-130 periods and are the slow ones. Budget a few
# hours per arm on an 8-core slot -> launch one arm per cluster job, not serial.
#
# Usage:
#   # one arm (what the cluster launches):
#   ONLY=e123_w5 NO_MAIL=1 bash experiments/e123_train_window_lgb_search_crypto.sh
#   # whole study serially on one box (slow):
#   bash experiments/e123_train_window_lgb_search_crypto.sh
#   # render the report once every arm has landed:
#   REPORT_ONLY=1 bash experiments/e123_train_window_lgb_search_crypto.sh
#   DRY_RUN=1 bash experiments/e123_train_window_lgb_search_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT="${EXPERIMENT:-E123}"
DATA="crypto/v2.0-onehot"
INCLUDE='(^feature_market_cap_(avg|ewa)_)|(^feature_momentum_)|(^feature_volatility_)|(^feature_sym_)'
# Search budget and slot sizing. Defaults size an 8-core slot (lgb: search x
# n_jobs = 4 x 2), the cluster-safe shape for a 30%-of-32 box. n_iter/n_splits
# can be shrunk for a smoke test.
E123_N_ITER="${E123_N_ITER:-60}"
E123_N_SPLITS="${E123_N_SPLITS:-5}"
E123_SEARCH_JOBS="${E123_SEARCH_JOBS:-4}"
E123_N_JOBS="${E123_N_JOBS:-2}"
TUNE_WINDOW="${E123_TUNE_WINDOW:-130}"

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add e123_w5     5  "1 week"
add e123_w10   10  "2 weeks"
add e123_w15   15  "3 weeks"
add e123_w20   20  "4 weeks"
add e123_w22   22  "1 month"
add e123_w65   65  "3 months"
add e123_w130 130  "6 months (deployed window; searches its own window)"

REPORT="logs/e123_report.txt"

# --------------------------------------------------------------------------
# Report block (shared by REPORT_ONLY and the end of a full serial run).
# --------------------------------------------------------------------------
run_report() {
  {
  echo "${EXPERIMENT} -- training window of a FROM-SCRATCH-SEARCHED LightGBM (lgb, concat_trio feature set)"
  echo "  results  results/metrics/crypto/lgb/e123_*.json ; runner e123_train_window_lgb_search_crypto.sh"
  echo "  budget   --n-iter ${E123_N_ITER} --n-splits ${E123_N_SPLITS}; tune-window ${TUNE_WINDOW} for windows < ${TUNE_WINDOW}"
  echo "=============================================================="
  echo
  "${PYTHON}" - <<'PY'
import os, json, glob
import numpy as np, pandas as pd
def ser(run, model="lgb"):
    p="results/per_period/crypto/%s/%s.csv"%(model,run)
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
def paired(a,b):
    j=pd.concat([a,b],axis=1,join="inner",keys=["a","b"]); return nw(j["a"]-j["b"])
def hp(run):
    for p in ("results/hyperparams/crypto/lgb/%s.json"%run,):
        if os.path.exists(p):
            try: d=json.load(open(p))
            except Exception: return None
            # hyperparams file may store a list (per-refit) or a dict
            if isinstance(d,list) and d: d=d[-1]
            if isinstance(d,dict): return d.get("params", d)
    return None

WIN=[("e123_w5",5),("e123_w10",10),("e123_w15",15),("e123_w20",20),
     ("e123_w22",22),("e123_w65",65),("e123_w130",130)]
base=ser("e123_w130"); bp=payout(base) if base is not None else None

print("== E123 arms: from-scratch search per window, paired vs the searched w130 ==")
print("%-10s %4s %10s %10s %10s %8s %8s %6s   dpay vs e123_w130 (t)"%(
    "arm","win","CORR","MMC","payout","p.sh","dd(pay)","n"))
print("-"*104)
for name,w in WIN:
    f=ser(name)
    if f is None:
        print("%-10s %4d   (missing)"%(name,w)); continue
    p=payout(f); pm,psh,pdd=summ(p); cm=f["corr"].mean(); mm=f["mmc"].mean()
    ds="-"
    if bp is not None and name!="e123_w130":
        d,t=paired(p,bp); ds="%+.6f (t %+.2f)"%(d,t)
    print("%-10s %4d %10.6f %10.6f %10.6f %8.3f %8.4f %6d   %s"%(name,w,cm,mm,pm,psh,pdd,len(p),ds))
print("-"*104)

# Cross-study: searched (E123) vs pinned (E120) at each window, paired NW.
print()
print("== Searching vs pinning, window for window: E123 (searched) - E120 (pinned concat_trio) ==")
print("%-6s %12s %12s   %-24s"%("win","searched pay","pinned pay","dpay searched-pinned (t)"))
print("-"*72)
for name,w in WIN:
    fs=ser(name); fp=ser("e120_w%d"%w)
    if fs is None or fp is None:
        print("%-6d %12s %12s   (need both e123 and e120 arms)"%(
            w, "-" if fs is None else "%.6f"%payout(fs).mean(),
               "-" if fp is None else "%.6f"%payout(fp).mean())); continue
    ps, pp = payout(fs), payout(fp); d,t=paired(ps,pp)
    print("%-6d %12.6f %12.6f   %+.6f (t %+.2f)"%(w, ps.mean(), pp.mean(), d, t))
print("-"*72)
print("+t => the from-scratch search beats the pinned concat_trio at that window.")

# Headline: does any searched window beat the DEPLOYED pinned concat_trio?
dep=ser("e120_w130")
if dep is not None:
    dp=payout(dep)
    print()
    print("== Headline: each searched arm vs the DEPLOYED pinned concat_trio (e120_w130 = %.6f) =="%dp.mean())
    best=None
    for name,w in WIN:
        f=ser(name)
        if f is None: continue
        d,t=paired(payout(f),dp)
        print("  %-10s (w%-3d)  dpay %+.6f (t %+.2f)"%(name,w,d,t))
        if best is None or payout(f).mean()>best[1]: best=(name,payout(f).mean())
    if best: print("  best searched arm by payout: %s at %.6f"%best)

# Selected booster per window -- does the best model get simpler as history shrinks?
print()
print("== Selected hyperparameters per window (last refit) -- how the booster changes ==")
keys=["num_leaves","max_depth","n_estimators","learning_rate","colsample_bytree",
      "subsample","min_child_samples","reg_lambda"]
print("%-10s %4s "%("arm","win")+" ".join("%12s"%k[:12] for k in keys))
for name,w in WIN:
    d=hp(name)
    if not d: print("%-10s %4d   (no hyperparams file)"%(name,w)); continue
    def g(k):
        v=d.get(k)
        return "%.5g"%v if isinstance(v,(int,float)) else str(v)
    print("%-10s %4d "%(name,w)+" ".join("%12s"%g(k) for k in keys))
print()
print("payout = 0.1*CORR + MMC, rebuilt per period. Paired Newey-West, lag 20.")
PY
  } 2>&1 | tee "${REPORT}"
}

if [ "${REPORT_ONLY:-0}" = "1" ]; then
  run_report
  if [ "${NO_MAIL:-0}" != "1" ]; then
    "${PYTHON}" - "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s report" % sys.argv[2], open(sys.argv[1]).read()); print("emailed the report")
PY
  fi
  exit 0
fi

# Gate: the dataset still carries 762 feature columns, the include regex still
# keeps exactly 748, and the 300-rows-per-period cross-section still holds.
# (identical to E120's gate -- same base.)
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

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ -z "${ONLY:-}" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: training window 1 week -> 6 months, booster SEARCHED from scratch per window
  base      lgb, concat_trio feature set (748 feat on crypto/v2.0-onehot); refit 130, embargo 20, seed 42
  search    --n-iter ${E123_N_ITER} --n-splits ${E123_N_SPLITS}; tune-window ${TUNE_WINDOW} for windows < ${TUNE_WINDOW}
  twin of   E120 (same windows, booster PINNED) and E117 (the ridge lambda search)
  results   results/metrics/crypto/lgb/e123_*.json   report logs/e123_report.txt

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: training window of a from-scratch-searched LightGBM | ${total} arms"
echo "base lgb concat_trio (748 feat); windows 5..130; refit 130; search n_iter=${E123_N_ITER} splits=${E123_N_SPLITS}"
echo "tune-window ${TUNE_WINDOW} for short windows; search-jobs=${E123_SEARCH_JOBS} n-jobs=${E123_N_JOBS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
echo "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index+1))
    name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    cmd=("${PYTHON}" -u -m src.run_forecast crypto lgb
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}" --refit-every 130 --embargo 20
         --n-iter "${E123_N_ITER}" --n-splits "${E123_N_SPLITS}"
         --search-jobs "${E123_SEARCH_JOBS}" --n-jobs "${E123_N_JOBS}"
         --val-frac 0.3 --seed 42
         --data-version "${DATA}" --feature-include "${INCLUDE}" --run-name "${name}")
    if [ "${window}" -lt "${TUNE_WINDOW}" ]; then
        cmd+=(--tune-window "${TUNE_WINDOW}")
        why="${why}, tuned on ${TUNE_WINDOW}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s window=%-3s  %s\n' "$index" "$total" "$name" "$window" "$why"
        printf '        %s\n' "${cmd[*]}"; continue; fi
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

echo
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"

# When launched one-arm-per-job (ONLY set), skip the report -- it needs every
# arm. Render it with REPORT_ONLY=1 once all seven have landed.
if [ -z "${ONLY:-}" ]; then
    run_report
    if [ "${NO_MAIL:-0}" != "1" ]; then
      "${PYTHON}" - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
    fi
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)

# CORRECTION (2026-10-24), appended rather than edited in place because arms were
# executing this file (bash reads scripts incrementally; rewriting the header
# would shift every byte offset under them): the header's "min_child_samples
# floors at 1000" is WRONG. fit_grid rewrites that range from the training block
# on every refit (models.scaled_leaf_range): 20..780 at 130 periods / 39,000 rows,
# 20..60 at 10 periods. concat_trio's 50 is inside the grid. Fold this into the
# header once no arm is running.
