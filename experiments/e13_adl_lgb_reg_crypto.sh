#!/usr/bin/env bash
#
# Experiment E13: autoregressive and distributed lags, on the best LightGBM
#
# E13 is the reference handle. Everything this study writes is e13_*.
#
# Four arms, decomposed so the answer is attributable rather than a single
# yes/no on a combined block:
#
#     arm    dataset               feat  what it adds
#     -------------------------------------------------------------------
#     raw    crypto/v2.0             22  nothing -- the baseline
#     dl     crypto/v2.0-e13dl       88  + 66 distributed lags: every raw
#                                        feature at t-1, t-2, t-3
#     ar     crypto/v2.0-e13ar       23  + 1 autoregressive lag: the target
#                                        at t-20
#     adl    crypto/v2.0-e13adl1     89  + both
#
# Running dl and ar separately is the point. If adl helps, "which half" is a
# different question from "does it help", and a combined arm cannot answer it.
#
# ---------------------------------------------------------------------------
# The leakage rule, which is the whole reason the AR lag is 20 and not 1
# ---------------------------------------------------------------------------
#
# target_binned_return_20 at date t is the return over [t, t+20]. It is NOT
# observable at t -- it resolves at t+20. So a row at t may only condition on a
# target observed at t-q for q >= 20.
#
# y_{t-1} is the return over [t-1, t+19]: it overlaps the label being predicted
# by 19 of its 20 days. Regressing on it is very nearly regressing y on itself.
# It would score spectacularly and be worth exactly nothing live -- the same
# trap the mean-target encoding hit (whole-sample 0.2536, past-only-no-lag
# 0.2397, causal-with-lag 0.0557).
#
# build_adl.py refuses a target lag below the horizon; --allow-leaky-ar is the
# only way past it and exists so the leak can be MEASURED, not stumbled into.
# This runner never passes it. The preflight below re-derives the floor from the
# target name and re-checks every AR column in the dataset against it, because a
# dataset built earlier with a different flag would otherwise pass unnoticed.
#
# The distributed lags need no such floor: features describe what was observable
# at their own date, so x_{t-1} is known at t. Feature rule and target rule are
# deliberately different -- conflating them is exactly how the leak gets in.
#
# ---------------------------------------------------------------------------
# The model is the pinned best LightGBM, unchanged
# ---------------------------------------------------------------------------
#
#     n_estimators 4327  learning_rate 0.002572  max_depth 8  num_leaves 255
#     colsample_bytree 0.070213  subsample 0.816153  min_child_samples 313
#     reg_lambda 1.757511          rolling 6m window / 6m refit, seed 42
#
# One caveat that follows from colsample_bytree=0.0702: on the 88-column dl arm
# a tree draws ~6 columns, of which ~1.5 are raw. The lags do not merely ADD
# information, they DILUTE the 22 features that already work -- the mechanism
# that sank every wide arm in the feature-block study. A null on dl is therefore
# ambiguous between "lags carry nothing" and "the sampling rate buried them",
# and the ar arm (23 columns, essentially undiluted) is the clean test.
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Baseline is e13_raw, run inside the study, which must reproduce
# cad_lgb_reg_w6m_r6m (payout 0.083272) -- the report checks it.
#
# On ~26 effective independent windows |t| < 2 is the expected outcome. What
# would be genuinely new: an AR term clearing |t| > 2 would say this panel has
# exploitable target autocorrelation at a 20-day remove, which nothing in the
# programme has tested. Eleven feature blocks have failed to beat raw; none of
# them was a lag.
#
# Usage:
#   bash experiments/e13_adl_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e13_adl_lgb_reg_crypto.sh
#   ONLY="raw ar" bash experiments/e13_adl_lgb_reg_crypto.sh
#   SEARCHED=1 bash experiments/e13_adl_lgb_reg_crypto.sh   # raw + adl, from scratch
#   NO_MAIL=1 bash experiments/e13_adl_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E13}"
E13_WINDOW="${E13_WINDOW:-6month}"
E13_REFIT="${E13_REFIT:-6month}"
E13_N_SPLITS="${E13_N_SPLITS:-2}"
E13_N_JOBS="${E13_N_JOBS:-8}"
E13_SEED="${E13_SEED:-42}"
# SEARCHED=1 re-runs raw and adl with a FULL 50-draw search per refit instead of
# the pinned config. The pin was selected on the raw 22 features, where
# colsample_bytree=0.0702 draws ~1.5 columns per tree; on the 89-column adl arm
# it draws ~6, of which ~1.5 are raw. That rate was never chosen for this width,
# so a pinned null on adl is ambiguous between "lags carry nothing" and "the
# model was never allowed to use them". These arms remove the ambiguity: each
# dataset gets its own hyperparameters, trained from scratch at every refit.
#
# Cost: a searched arm is ~160 min against ~2 for a pinned one -- 50 draws x 5
# folds x 4 refits instead of one configuration. Two arms is roughly 5-6 hours.
SEARCHED="${SEARCHED:-0}"
E13_N_ITER="${E13_N_ITER:-50}"
E13_SEARCH_JOBS="${E13_SEARCH_JOBS:-4}"
case "${E13_WINDOW}" in ""|0|[0-9]) echo "E13_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

PARAMS='{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}'

ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
if [ "${SEARCHED}" = "1" ]; then
    add raw_search "crypto/v2.0"         22
    add adl_search "crypto/v2.0-e13adl1" 89
else
    add raw "crypto/v2.0"            22
    add dl  "crypto/v2.0-e13dl"      88
    add ar  "crypto/v2.0-e13ar"      23
    add adl "crypto/v2.0-e13adl1"    89
fi

# Preflight: feature counts, AND every AR column re-checked against the horizon
# derived from the target name. A dataset built earlier with --allow-leaky-ar
# would otherwise sail through with a plausible, worthless score.
python3 - "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, re, pyarrow.parquet as pq
sys.path.insert(0, ".")
from src.data_mani.utils import default_embargo, get_spec
cut = sys.argv.index("--")
versions, wants = sys.argv[1:cut], [int(x) for x in sys.argv[cut+1:]]
target = get_spec("crypto")["target"]
floor = default_embargo("crypto", target)
bad = []
for v, w in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    if not os.path.exists(p):
        bad.append("%s MISSING" % v); continue
    cols = [c for c in pq.ParquetFile(p).schema.names if c.startswith("feature_")]
    if len(cols) != w:
        bad.append("%s has %d features, expected %d" % (v, len(cols), w))
    for c in cols:
        m = re.search(r"_arlag(\d+)$", c)
        if m and int(m.group(1)) < floor:
            bad.append("%s: %s lags the target by %s < the %d-period horizon -- LOOKAHEAD"
                       % (v, c, m.group(1), floor))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d datasets, feature counts match, every AR lag >= the "
      "%d-period horizon of %s" % (len(versions), floor, target))
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: raw (22), dl (88), ar (23), adl (89)
  asks      do autoregressive and/or distributed lags beat raw?
  model     pinned best LightGBM, ${E13_WINDOW}/${E13_REFIT} rolling, seed ${E13_SEED}
  AR lag    20 periods, the target horizon -- anything shorter is lookahead
  expect    ~15 min pinned / ~5-6 h searched (SEARCHED=1)

Where to look while it runs:
  runner    experiments/e13_adl_lgb_reg_crypto.sh
  progress  logs/e13_*.log
  results   results/metrics/crypto/lgb/e13_*.json
  report    logs/e13_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e13_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: autoregressive / distributed lags  |  ${total} arms"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    run_name="e13_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
         --cv-scheme rolling --train-window "${E13_WINDOW}" --refit-every "${E13_REFIT}"
         --data-version "${DATA[$i]}" --era-step 1
         --seed "${E13_SEED}" --run-name "${run_name}")
    if [ "${SEARCHED}" = "1" ]; then
        # trained from scratch: every refit runs its own randomised search
        cmd+=(--n-iter "${E13_N_ITER}" --n-splits 5
              --n-jobs "${E13_N_JOBS}" --search-jobs "${E13_SEARCH_JOBS}")
    else
        cmd+=(--fixed-params "${PARAMS}"
              --n-splits "${E13_N_SPLITS}" --n-jobs "${E13_N_JOBS}")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s %-24s %3s feat\n' "$index" "$total" "$run_name" "${DATA[$i]}" "${NFEAT[$i]}"; continue; fi
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

REPORT="logs/e13_report.txt"
{
echo "${EXPERIMENT} -- autoregressive / distributed lags on the pinned LightGBM"
echo "  results  results/metrics/crypto/lgb/e13_*.json"
echo "  runner   experiments/e13_adl_lgb_reg_crypto.sh"
echo "  logs     logs/e13_*.log"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
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
for p in sorted(glob.glob("results/metrics/crypto/lgb/e13_*.json")):
    d=json.load(open(p)); c=d["config"]; n=os.path.basename(p)[:-5]
    rows.append(dict(run=n,arm=n.replace("e13_",""),feat=c["n_features"],
        corr=d["corr"]["mean"],mmc=d["mmc"]["mean"],sharpe=d["corr"]["sharpe"],
        maxdd=d["corr"]["max_drawdown"],payout=0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
if not rows: print("no results"); raise SystemExit
t=pd.DataFrame(rows)
base=t[t.arm=="raw"]
print("%-6s %5s %10s %10s %8s %8s %10s %11s %7s"%(
  "arm","feat","CORR","MMC","sharpe","maxdd","payout","d vs raw","t"))
print("-"*84)
for _,r in t.sort_values("payout",ascending=False).iterrows():
    d=tt=None
    if len(base) and base["run"].iloc[0]!=r["run"]:
        j=pd.concat([payout(pp(base["run"].iloc[0])),payout(pp(r["run"]))],
                    axis=1,join="inner",keys=["b","a"])
        d,tt=nw(j["a"]-j["b"])
    print("%-6s %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s"%(
      r["arm"],r["feat"],r["corr"],r["mmc"],r["sharpe"],r["maxdd"],r["payout"],
      "baseline" if d is None else "%+.6f"%d, "-" if tt is None else "%+.2f"%tt))
print("-"*84)
ref="results/metrics/crypto/lgb/cad_lgb_reg_w6m_r6m.json"
if len(base) and os.path.exists(ref):
    d=json.load(open(ref)); want=0.1*d["corr"]["mean"]+d["mmc"]["mean"]
    got=float(base["payout"].iloc[0])
    print("baseline check: e13_raw payout %.6f vs cad_lgb_reg_w6m_r6m %.6f -> %s"
          % (got,want,"REPRODUCES" if abs(got-want)<1e-9 else "*** MISMATCH ***"))
print("Newey-West lag 20; 510 periods ~ 26 effective independent windows.")
print("AR lag is 20 periods = the target horizon. Shorter would be lookahead.")
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
