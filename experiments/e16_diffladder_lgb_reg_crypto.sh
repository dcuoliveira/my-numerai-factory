#!/usr/bin/env bash
#
# Experiment E16: cumulative per-symbol DIFFERENCES on a ladder reaching a year
#
# E16 is the reference handle. Everything this study writes is e16_*, and its
# datasets are v2.0-e16d* -- namespaced because a concurrent study in this same
# checkout owns build_cumdiff.py and the v2.0-cumdiff* names.
#
# Twelve arms: six ladder rungs x {differences alone, raw + differences}.
#
#     rung  horizon      orders included        only  raw+
#     ------------------------------------------------------
#        1  one day      d1                       23    45
#        5  one week     d1 d5                    46    68
#       22  one month    d1 d5 d22                69    91
#       66  one quarter  d1 d5 d22 d66            92   114
#      132  half a year  d1 ... d132             115   137
#      264  one year     d1 ... d264             138   160
#
# Rungs are log-spaced, not every integer: cumulative through all 264 orders
# would add 5,808 feature columns, more than the panel has rows per date, and
# would bury the raw 22 entirely under colsample_bytree. Skipping intermediate
# lags is what keeps the column count -- not the disk, which is 87 MB total --
# in a range a booster can actually sample from.
#
# ---------------------------------------------------------------------------
# Coverage falls with the rung, and that is a real limit on the long end
# ---------------------------------------------------------------------------
#
# A difference at lag q is undefined for a symbol's first q dates, and most
# tokens are short-lived. Rows with enough history, measured at build time:
#
#     d1 100%   d5 99%   d22 95%   d66 87%   d132 76%   d264 60%
#
# So the year rung carries real signal on 60% of the panel and the neutral 0.5
# on the other 40%. A null at d264 is therefore ambiguous between "a year-long
# change says nothing" and "most tokens never lived long enough to have one".
# The short rungs do not have this problem.
#
# ---------------------------------------------------------------------------
# The target differences are anchored; the feature differences are not
# ---------------------------------------------------------------------------
#
# target_binned_return_20 at t is the return over [t, t+20] and is not
# observable at t, so every target difference uses only resolved values:
#
#     feature_target_e16d<q> = y_{t-20} - y_{t-20-q}
#
# Feature differences need no anchor -- a feature describes what was observable
# at its own date. The preflight re-derives the horizon from the target name and
# proves the anchor on real rows before any arm runs.
#
# ---------------------------------------------------------------------------
# Protocol, and what it costs
# ---------------------------------------------------------------------------
#
# Default is the PINNED best LightGBM (regression -- payout 0.082061 against
# 0.043929 for classification on raw 22, so the class is settled). Twelve pinned
# arms are ~40 min total.
#
# SEARCHED=1 re-runs with a full 50-draw search per refit, which is what "from
# scratch" means. That is ~100-320 min PER ARM on a free machine and far worse
# on a contended one -- E14's single searched arm has taken 24 h against five
# concurrent jobs. Twelve searched arms is days. Run the pinned ladder first to
# find which rungs are worth that.
#
# Usage:
#   bash experiments/e16_diffladder_lgb_reg_crypto.sh
#   SEARCHED=1 ONLY="e16d22 e16d22only" bash experiments/e16_diffladder_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e16_diffladder_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e16_diffladder_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E16}"
E16_WINDOW="${E16_WINDOW:-6month}"
E16_REFIT="${E16_REFIT:-6month}"
E16_N_JOBS="${E16_N_JOBS:-8}"
E16_SEED="${E16_SEED:-42}"
SEARCHED="${SEARCHED:-0}"
E16_N_ITER="${E16_N_ITER:-50}"
case "${E16_WINDOW}" in ""|0|[0-9]) echo "E16_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

PARAMS='{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}'

ARM=(); DATA=()
for k in 1 5 22 66 132 264; do
    ARM+=("e16d${k}only"); DATA+=("crypto/v2.0-e16d${k}only")
    ARM+=("e16d${k}");     DATA+=("crypto/v2.0-e16d${k}")
done

# Preflight: datasets exist, and the target anchor is PROVEN on real rows rather
# than trusted from the builder's docstring.
python3 - "${DATA[@]}" <<'PY' || exit 1
import sys, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, ".")
from src.data_mani.utils import default_embargo, get_spec
bad = [v for v in sys.argv[1:] if not os.path.exists("data/%s/train.parquet" % v)]
if bad:
    sys.exit("preflight FAILED, missing: %s" % bad)
target = get_spec("crypto")["target"]
H = default_embargo("crypto", target)
df = pq.read_table("data/crypto/v2.0-e16d1/train.parquet").to_pandas().reset_index()
df = df[["symbol", "date", "feature_target_e16d1", target]].sort_values(["symbol", "date"])
g = df.groupby("symbol", sort=False)
anchored = ((g[target].shift(H) - g[target].shift(H + 1)) + 1.0) / 2.0
leaky    = ((df[target] - g[target].shift(1)) + 1.0) / 2.0
ok  = np.isclose(df["feature_target_e16d1"], anchored, equal_nan=True).mean()
bad_ = np.isclose(df["feature_target_e16d1"], leaky, equal_nan=True).mean()
print("preflight OK: %d datasets" % len(sys.argv[1:]))
print("  target diff == y_(t-%d) - y_(t-%d) [anchored, causal] : %.2f%%" % (H, H+1, 100*ok))
print("  target diff == y_t - y_(t-1)      [leaky]             : %.2f%%" % (100*bad_))
if ok < 0.99:
    sys.exit("the target difference is NOT anchored at the horizon -- refusing to run")
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
MODE=$([ "${SEARCHED}" = "1" ] && echo "searched (${E16_N_ITER} draws/refit)" || echo "pinned")

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: six ladder rungs (1,5,22,66,132,264 dates) x {diff-only, raw+diff}
  asks      do cumulative per-symbol differences, out to a year, beat raw?
  protocol  ${MODE}
  coverage  d1 100%, d5 99%, d22 95%, d66 87%, d132 76%, d264 60%

Where to look while it runs:
  runner    experiments/e16_diffladder_lgb_reg_crypto.sh
  progress  logs/e16_*.log
  results   results/metrics/crypto/lgb/e16_*.json
  report    logs/e16_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e16_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: cumulative difference ladder  |  ${total} arms, ${MODE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    run_name="e16_${arm}$([ "${SEARCHED}" = "1" ] && echo "_search")"
    cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
         --cv-scheme rolling --train-window "${E16_WINDOW}" --refit-every "${E16_REFIT}"
         --data-version "${DATA[$i]}" --era-step 1 --seed "${E16_SEED}"
         --run-name "${run_name}")
    if [ "${SEARCHED}" = "1" ]; then
        cmd+=(--n-iter "${E16_N_ITER}" --n-splits 5 --n-jobs "${E16_N_JOBS}" --search-jobs 4)
    else
        cmd+=(--fixed-params "${PARAMS}" --n-splits 2 --n-jobs "${E16_N_JOBS}")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-24s %s\n' "$index" "$total" "$run_name" "${DATA[$i]}"; continue; fi
    # Re-check the dataset IMMEDIATELY before the arm runs, not just in the
    # preflight. These are locally built files: if one vanishes mid-study
    # run_forecast falls through to download_if_missing and asks Numerai for a
    # version that exists only on this disk, and the arm dies with "the
    # requested file does not exist" -- a network error for a local problem.
    # That is exactly how seven arms were lost on the first pass.
    if [ ! -f "data/${DATA[$i]}/train.parquet" ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} SKIPPED: data/${DATA[$i]}/train.parquet is missing."
        echo "    Rebuild: python3 -m src.data_mani.build_e16_diffladder --through ${arm//[^0-9]/}$([[ "$arm" == *only ]] && echo ' --drop-originals')"
        continue
    fi

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

REPORT="logs/e16_report.txt"
{
echo "${EXPERIMENT} -- cumulative difference ladder (${MODE})"
echo "  results  results/metrics/crypto/lgb/e16_*.json"
echo "  runner   experiments/e16_diffladder_lgb_reg_crypto.sh"
echo "=============================================================="
echo "completed $(( ran - failed )) of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min ($ran attempted, ${failed} failed)"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - <<'PY'
import json, glob, os, re
import numpy as np, pandas as pd
COV={1:1.00,5:0.99,22:0.95,66:0.87,132:0.76,264:0.60}
def pp(r):
    p="results/per_period/crypto/lgb/%s.csv"%r
    if not os.path.exists(p): return None
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
# baseline: raw 22 under the same protocol, whichever exists
base=None
for cand in ("e13_raw","cad_lgb_reg_w6m_r6m"):
    if pp(cand) is not None: base=cand; break
b=pp(base) if base else None
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb/e16_*.json")):
    r=os.path.basename(p)[:-5]; d=json.load(open(p)); f=pp(r)
    m=re.match(r"e16_e16d(\d+)(only)?",r)
    rung=int(m.group(1)); kind="diff only" if m.group(2) else "raw + diff"
    pay=payout(f); c=pay.cumsum(); delta=t=None
    if b is not None:
        j=pd.concat([payout(b),pay],axis=1,join="inner",keys=["b","a"])
        delta,t=nw(j["a"]-j["b"])
    rows.append((pay.mean(),rung,kind,d["config"]["n_features"],f["corr"].mean(),
                 pay.mean()/pay.std(ddof=0),(np.maximum.accumulate(c)-c).max(),delta,t))
if not rows: print("no results"); raise SystemExit
print("%-6s %-11s %5s %5s %10s %8s %8s %10s %11s %7s"%(
  "rung","block","cov","feat","CORR","sharpe","maxdd","payout","d vs raw","t"))
print("-"*92)
for pay,rung,kind,feat,corr,sh,dd,delta,t in sorted(rows,key=lambda x:(-x[0])):
    print("%-6s %-11s %4.0f%% %5d %10.6f %8.4f %8.4f %10.6f %11s %7s"%(
      "d%d"%rung,kind,100*COV.get(rung,1),feat,corr,sh,dd,pay,
      "-" if delta is None else "%+.6f"%delta, "-" if t is None else "%+.2f"%t))
print("-"*92)
print("baseline: %s (raw 22, same protocol)"%base)
print("cov = share of rows with enough history for the longest rung; the rest fill neutral.")
print("Newey-West lag 20; ~26 effective independent windows, so |t| < 2 is expected.")
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
