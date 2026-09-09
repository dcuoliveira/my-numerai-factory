#!/usr/bin/env bash
#
# Experiment E29: one from-scratch LightGBM tune per submitted-model dataset,
# using the E27 (Kalfa-Timmermann-van der Zwan) priors (model class lgb_ktz).
#
# E29 is the reference handle; every artefact is e29_*. This is the booster twin
# of E28 (mlp_ktz): same 9 datasets, same "one search per dataset" shape.
#
# WHAT THIS RUNS
#   For each of the 9 datasets the deployed book trains on, ONE full
#   hyperparameter search from scratch over the lgb_ktz grid -- the E27 priors
#   compiled into a search space, with the effort ordering
#   learning_rate >= colsample_bytree > num_leaves >= n_estimators made true by
#   the LEVEL COUNTS (a random search explores a many-level axis more per draw):
#
#     learning_rate     12 log values in [0.001, 0.01]   the tuning focus
#     colsample_bytree   6 values in [0.05, 0.20]         second focus
#     num_leaves         3 values {2, 4, 8}               coarse, stump-favoured
#     n_estimators       2 values {2000, 4000}            fewest
#     FIXED: max_depth=-1, min_child_samples=100, subsample=0.8, reg_lambda=10
#            = 12 x 6 x 3 x 2 = 432 cells, min_child_samples PINNED (not rescaled)
#
#   The two known bad corners are UNREACHABLE by construction, not merely
#   discouraged: learning_rate stops at 0.01 and n_estimators at 4000, so the
#   product L = lr*n_est never exceeds 40 -- the good side of E27's monotone-bad
#   L curve -- and colsample stops at 0.20, excluding the 1.0 value that locked
#   in a bad median. The search cannot wander into a corner E27 already ruled out.
#
#   9 datasets x 1 tune = 9 arms (dataset -> slots that train on it):
#     e29_raw            v2.0                                      22   CURUPIRA_02, SACI_04
#     e29_tenc           v2.0-tenc                                 23   CURUPIRA_01
#     e29_interact       v2.0-interact                             36   SACI_01
#     e29_fenccausal     v2.0-fenccausal                           44   IA_01
#     e29_cumlag3        v2.0-cumlag3                              88   SACI_05
#     e29_tenc_onehot    v2.0-tenc-onehot                         763   SACI_03
#     e29_onehot_rollsel v2.0-onehot-rollsel                      786   CURUPIRA_03
#     e29_allfeat        v2.0-tenc-onehot-spread-interact         788   SACI_02
#     e29_allfeatpp      v2.0-tenc-onehot-rollsel-spread-interact 812   CURUPIRA_04
#
# WHAT WOULD COUNT AS A RESULT
#   Two questions. (1) Does a prior-guided from-scratch tune beat the deployed
#   pin (the 4327-tree canonical config) on each dataset? (2) Does the search
#   LAND where E27 says the good models are -- small num_leaves, low learning
#   rate, low colsample? The report prints each arm's chosen hyperparameters so
#   the second question is answered, not assumed.
#
# Usage:
#   bash experiments/e29_lgb_ktz_tune_crypto.sh
#   DRY_RUN=1 bash experiments/e29_lgb_ktz_tune_crypto.sh
#   ONLY="raw tenc" bash experiments/e29_lgb_ktz_tune_crypto.sh
#   SHARD=1 OF=2 bash experiments/e29_lgb_ktz_tune_crypto.sh
#   NO_MAIL=1 bash experiments/e29_lgb_ktz_tune_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E29}"
E29_WINDOW="${E29_WINDOW:-6month}"
E29_REFIT="${E29_REFIT:-6month}"
E29_N_ITER="${E29_N_ITER:-60}"
E29_N_SPLITS="${E29_N_SPLITS:-5}"
E29_N_JOBS="${E29_N_JOBS:-6}"
E29_SEARCH_JOBS="${E29_SEARCH_JOBS:-3}"
E29_SEED="${E29_SEED:-42}"
case "${E29_WINDOW}" in ""|0|[0-9]) echo "E29_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add raw            "crypto/v2.0"                                      22
add tenc           "crypto/v2.0-tenc"                                23
add interact       "crypto/v2.0-interact"                            36
add fenccausal     "crypto/v2.0-fenccausal"                          44
add cumlag3        "crypto/v2.0-cumlag3"                             88
add tenc_onehot    "crypto/v2.0-tenc-onehot"                        763
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                     786
add allfeat        "crypto/v2.0-tenc-onehot-spread-interact"        788
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812

python3 - "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,w in zip(versions,wants):
    p="data/%s/train.parquet"%v
    if not os.path.exists(p): bad.append("%s MISSING"%v); continue
    g=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if g!=w: bad.append("%s has %d features, expected %d"%(v,g,w))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
# and the wrapper's grid must satisfy the requested effort ordering
sys.path.insert(0,".")
from src.prediction.models import build_wrapper
g=build_wrapper("lgb_ktz",task="regression",seed=42).param_grid
lv=lambda k: len(g[k])
if not (lv("learning_rate")>=lv("colsample_bytree")>lv("num_leaves")>=lv("n_estimators")):
    sys.exit("lgb_ktz grid violates lr>=cs>nl>=n_est")
Lmax=max(g["learning_rate"])*max(g["n_estimators"])
if Lmax>40+1e-6: sys.exit("lgb_ktz L=%.1f exceeds the 40 safe ceiling"%Lmax)
print("preflight OK: %d datasets; grid lr(%d)>=cs(%d)>nl(%d)>=ne(%d), L<=%.0f"
      % (len(versions),lv("learning_rate"),lv("colsample_bytree"),lv("num_leaves"),lv("n_estimators"),Lmax))
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: one from-scratch lgb_ktz tune per book dataset
  priors    E27: lr[0.001,0.01] dense, cs[0.05,0.20], num_leaves{2,4,8}, n_est{2000,4000}
  effort    lr(12) >= cs(6) > nl(3) >= n_est(2) levels; L=lr*n_est <= 40
  budget    --n-iter ${E29_N_ITER}, ${E29_N_SPLITS} folds per refit

Where to look while it runs:
  runner    experiments/e29_lgb_ktz_tune_crypto.sh
  progress  logs/e29_*.log
  results   results/metrics/crypto/lgb_ktz/e29_*.json
  report    logs/e29_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e29_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: one lgb_ktz tune per book dataset  |  ${total} arms"
echo "n_iter=${E29_N_ITER} folds=${E29_N_SPLITS}  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="e29_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto lgb_ktz --task regression --metric corr
         --cv-scheme rolling --train-window "${E29_WINDOW}" --refit-every "${E29_REFIT}"
         --data-version "${DATA[$i]}"
         --n-iter "${E29_N_ITER}" --n-splits "${E29_N_SPLITS}"
         --n-jobs "${E29_N_JOBS}" --search-jobs "${E29_SEARCH_JOBS}"
         --era-step 1 --seed "${E29_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-46s %3s feat\n' "$index" "$total" "$run_name" "${DATA[$i]}" "${NFEAT[$i]}"; continue; fi
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

REPORT="logs/e29_report.txt"
{
echo "${EXPERIMENT} -- one lgb_ktz tune per book dataset (E27 priors)"
echo "  results  results/metrics/crypto/lgb_ktz/e29_*.json"
echo "  runner   experiments/e29_lgb_ktz_tune_crypto.sh"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - <<'PY'
import json, glob, os
import pandas as pd
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb_ktz/e29_*.json")):
    r=os.path.basename(p)[:-5]; d=json.load(open(p)); c=d.get("config",{})
    # the chosen hyperparameters from the last refit
    hp="results/hyperparams/crypto/lgb_ktz/%s.csv"%r
    chosen={}
    if os.path.exists(hp):
        h=pd.read_csv(hp)
        for k in ("num_leaves","learning_rate","colsample_bytree","n_estimators"):
            if k in h.columns: chosen[k]=h[k].iloc[-1]
    rows.append((0.1*d["corr"]["mean"]+d["mmc"]["mean"], r.replace("e29_",""),
                 c["n_features"], d["corr"]["mean"], d["corr"]["sharpe"],
                 d["corr"]["max_drawdown"], chosen))
if not rows: print("no results"); raise SystemExit
print("%-16s %5s %10s %8s %8s %10s   chosen (last refit)"%(
  "dataset","feat","CORR","sharpe","maxdd","payout"))
print("-"*104)
for pay,name,feat,corr,sh,dd,ch in sorted(rows,reverse=True):
    cs=" nl=%s lr=%.4f cs=%.2f ne=%s"%(ch.get("num_leaves","?"),
        float(ch.get("learning_rate",0)),float(ch.get("colsample_bytree",0)),
        ch.get("n_estimators","?")) if ch else ""
    print("%-16s %5d %10.6f %8.4f %8.4f %10.6f  %s"%(name,feat,corr,sh,dd,pay,cs))
print("-"*104)
print("Did the search land in E27's good corner? Expect small num_leaves (2-4),")
print("low learning_rate, low colsample. payout = 0.1*CORR + MMC.")
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
