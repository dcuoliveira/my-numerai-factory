#!/usr/bin/env bash
#
# Experiment E58: lgb_ktz tuned FROM SCRATCH on the 60-day label, for the five
# datasets that scored highest on MMC in E29 (the 20-day lgb_ktz tune per book
# dataset). The 60-day twin of E29's top-MMC arms -- and the LightGBM sibling of
# E57, which did exactly this for rf_ktz.
#
# E58 is the reference handle; every artefact it writes is e58_*.
#
# THE IDEA
#   E29 tuned lgb_ktz on all 9 book datasets on target_binned_return_20 and the
#   arms rank by MMC as below. MMC is what a slot pays over the meta model, so
#   E58 asks the natural follow-up: do the FIVE best-MMC LightGBM datasets keep
#   that strength when the label is the 60-DAY return instead of the 20-day one?
#   The 60-day book (SACI_03, CURUPIRA_05_60) pays ~0.079 MMC / ~0.093 payout;
#   this measures whether a from-scratch tuned booster reaches it on that label.
#
#   Top-5 by E29 MMC (t20): tenc_onehot 0.07427, allfeat 0.07383, interact
#   0.07365, allfeatpp 0.07330, fenccausal 0.07314. (These are the SAME five
#   feature sets E57 selected from the rf_ktz t20 tune -- the MMC ranking agrees
#   across model classes.) Excluded by construction: the leaky non-causal global
#   target encoding (fencglobal, whose ~0.14 MMC is an artefact), and every blend
#   / grid-ensemble arm (e30_*_ens, e47_*, e03_blend_*) -- a rank-average has no
#   hyperparameters to tune from scratch.
#
# THE ONLY CHANGE FROM E29  the label. Same lgb_ktz grid (the E27 priors: lr
#   [0.001,0.01] dense, colsample [0.05,0.20], num_leaves {2,4,8}, n_estimators
#   {2000,4000}; max_depth -1, min_child_samples 100, subsample 0.8, reg_lambda
#   10 pinned; L = lr*n_est <= 40 by construction), same --n-iter 60 from-scratch
#   RandomizedSearchCV, same rolling 6month/6month cadence, n_splits 5, seed 42,
#   same default python3 E29 used. Two consequences of --target
#   target_binned_return_60 with --embargo auto:
#     * embargo auto-derives 60 from the target horizon (not 20), so the
#       walk-forward stays leak-free for the longer label.
#     * 60-day runs score on ~470 periods, not 510 (consecutive periods overlap
#       ~98%), so treat arm counts and any t-stat with the usual care.
#
# WHAT WOULD COUNT AS A RESULT
#   (1) Does a from-scratch 60-day tune reach the 60-day book on MMC / payout on
#   each of these five feature sets? (2) Does the search still land in E27's good
#   corner (small num_leaves, low lr, low colsample) on the longer label? The
#   report prints each arm's chosen hyperparameters so (2) is answered, not
#   assumed. A secondary paired read exists against E01's 60d FIXED-param arms
#   (same feature set, same label): does searching beat E01's m50 pin?
#
# COST  five from-scratch searches; E29's 20-day arms ran 32-50 min each, so this
#   is ~3 h serial on an idle box, less when sharded (SHARD/OF) across machines.
#
# Usage:
#   bash experiments/e58_lgb_ktz_60d_tune_crypto.sh
#   DRY_RUN=1 bash experiments/e58_lgb_ktz_60d_tune_crypto.sh
#   ONLY="allfeat fenccausal" bash experiments/e58_lgb_ktz_60d_tune_crypto.sh
#   SHARD=1 OF=2 bash experiments/e58_lgb_ktz_60d_tune_crypto.sh
#   NO_MAIL=1 / E58_SCOUT=1 (era-step 4, 6 draws -- wiring only, not comparable)
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E58}"
E58_TARGET="${E58_TARGET:-target_binned_return_60}"
E58_WINDOW="${E58_WINDOW:-6month}"
E58_REFIT="${E58_REFIT:-6month}"
E58_EMBARGO="${E58_EMBARGO:-auto}"      # auto -> 60 for the 60-day target
E58_N_ITER="${E58_N_ITER:-60}"
E58_N_SPLITS="${E58_N_SPLITS:-5}"
E58_N_JOBS="${E58_N_JOBS:-6}"
E58_SEARCH_JOBS="${E58_SEARCH_JOBS:-3}"
E58_SEED="${E58_SEED:-42}"
E58_ERA_STEP=1
case "${E58_WINDOW}" in ""|0|[0-9]) echo "E58_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
[ "${E58_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E58_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e58"
if [ "${E58_SCOUT:-0}" = "1" ]; then
    PREFIX="e58scout"; E58_ERA_STEP=4; E58_N_ITER=6; E58_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${E58_N_ITER} draws -- wiring only, not comparable."
fi

# The five top-MMC datasets from E29 (tag, version, expected feature count).
ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add tenc_onehot "crypto/v2.0-tenc-onehot"                         763   # E29 MMC #1
add allfeat     "crypto/v2.0-tenc-onehot-spread-interact"         788   # E29 MMC #2
add interact    "crypto/v2.0-interact"                            36    # E29 MMC #3
add allfeatpp   "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812   # E29 MMC #4
add fenccausal  "crypto/v2.0-fenccausal"                          44    # E29 MMC #5

"${PYTHON}" - "${E58_TARGET}" "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
target=sys.argv[1]
cut=sys.argv.index("--"); versions=sys.argv[2:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,w in zip(versions,wants):
    p="data/%s/train.parquet"%v
    if not os.path.exists(p): bad.append("%s MISSING"%v); continue
    names=pq.ParquetFile(p).schema.names
    g=sum(c.startswith("feature_") for c in names)
    if g!=w: bad.append("%s has %d features, expected %d"%(v,g,w))
    if target not in names: bad.append("%s lacks the %s column"%(v,target))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
# the wrapper's grid must still satisfy E27's effort ordering and the L<=40 ceiling
sys.path.insert(0,".")
from src.prediction.models import build_wrapper
g=build_wrapper("lgb_ktz",task="regression",seed=42).param_grid
lv=lambda k: len(g[k])
if not (lv("learning_rate")>=lv("colsample_bytree")>lv("num_leaves")>=lv("n_estimators")):
    sys.exit("lgb_ktz grid violates lr>=cs>nl>=n_est")
Lmax=max(g["learning_rate"])*max(g["n_estimators"])
if Lmax>40+1e-6: sys.exit("lgb_ktz L=%.1f exceeds the 40 safe ceiling"%Lmax)
print("preflight OK: %d datasets carry the %s column and expected feature counts;"
      " lgb_ktz grid lr(%d)>=cs(%d)>nl(%d)>=ne(%d), L<=%.0f"
      % (len(versions),target,lv("learning_rate"),lv("colsample_bytree"),
         lv("num_leaves"),lv("n_estimators"),Lmax))
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ "${E58_SCOUT:-0}" != "1" ]; then
    "${PYTHON}" -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: from-scratch lgb_ktz 60-day tune of E29's top-5-MMC datasets
  datasets  ${ARM[*]}
  label     ${E58_TARGET} (embargo ${E58_EMBARGO} -> 60), the ONLY change from E29
  priors    E27: lr[0.001,0.01] dense, cs[0.05,0.20], num_leaves{2,4,8}, n_est{2000,4000}; L<=40
  budget    --n-iter ${E58_N_ITER}, ${E58_N_SPLITS} folds per refit, rolling ${E58_WINDOW}/${E58_REFIT}

Where to look while it runs:
  runner    experiments/e58_lgb_ktz_60d_tune_crypto.sh
  progress  logs/e58_*.log
  results   results/metrics/crypto/lgb_ktz/e58_*.json
  report    logs/e58_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is e58_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: lgb_ktz 60-day tune of E29's top-5-MMC datasets | ${total} arms"
echo "target=${E58_TARGET} n_iter=${E58_N_ITER} folds=${E58_N_SPLITS}  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    run_name="${PREFIX}_${arm}"
    cmd=("${PYTHON}" -u -m src.run_forecast crypto lgb_ktz --task regression --metric corr
         --cv-scheme rolling --train-window "${E58_WINDOW}" --refit-every "${E58_REFIT}"
         --embargo "${E58_EMBARGO}" --target "${E58_TARGET}"
         --data-version "${DATA[$i]}"
         --n-iter "${E58_N_ITER}" --n-splits "${E58_N_SPLITS}"
         --n-jobs "${E58_N_JOBS}" --search-jobs "${E58_SEARCH_JOBS}"
         --era-step "${E58_ERA_STEP}" --seed "${E58_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-22s %-46s %3s feat  tgt=%s\n' "$index" "$total" "$run_name" "${DATA[$i]}" "${NFEAT[$i]}" "${E58_TARGET}"; continue; fi
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

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- lgb_ktz 60-day tune of E29's top-5-MMC datasets"
echo "  results  results/metrics/crypto/lgb_ktz/${PREFIX}_*.json"
echo "  runner   experiments/e58_lgb_ktz_60d_tune_crypto.sh"
echo "  label    ${E58_TARGET} (the only change from E29's 20-day tune)"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
"${PYTHON}" - "${PREFIX}" <<'PY'
import json, glob, os, sys
import pandas as pd
prefix=sys.argv[1]
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb_ktz/%s_*.json"%prefix)):
    r=os.path.basename(p)[:-5]; d=json.load(open(p)); c=d.get("config",{})
    hp="results/hyperparams/crypto/lgb_ktz/%s.csv"%r
    chosen={}
    if os.path.exists(hp):
        h=pd.read_csv(hp)
        for k in ("num_leaves","learning_rate","colsample_bytree","n_estimators"):
            if k in h.columns: chosen[k]=h[k].iloc[-1]
    rows.append((d["mmc"]["mean"], r.replace(prefix+"_",""), c.get("n_features"),
                 d["corr"]["mean"], d["mmc"]["mean"],
                 0.1*d["corr"]["mean"]+d["mmc"]["mean"],
                 d["corr"]["sharpe"], d["corr"]["max_drawdown"],
                 d.get("n_periods"), chosen))
if not rows: print("no results"); raise SystemExit
print("%-14s %5s %10s %10s %10s %8s %8s %5s   chosen (last refit)"%(
  "dataset","feat","CORR","MMC","payout","sharpe","maxdd","n"))
print("-"*112)
for _,name,feat,corr,mmc,pay,sh,dd,n,ch in sorted(rows,reverse=True):
    cs=" nl=%s lr=%.4f cs=%.2f ne=%s"%(ch.get("num_leaves","?"),
        float(ch.get("learning_rate",0)),float(ch.get("colsample_bytree",0)),
        ch.get("n_estimators","?")) if ch else ""
    print("%-14s %5s %10.6f %10.6f %10.6f %8.4f %8.4f %5s  %s"%(
        name,feat,corr,mmc,pay,sh,dd,n,cs))
print("-"*112)
print("Ranked by MMC. payout = 0.1*CORR + MMC. The 60-day book pays ~0.079 MMC /")
print("~0.093 payout (SACI_03, CURUPIRA_05_60); does a from-scratch tune reach it?")
print("Did the search land in E27's good corner? Expect small num_leaves (2-4),")
print("low learning_rate, low colsample -- same as the 20-day tune, or not.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ] && [ "${E58_SCOUT:-0}" != "1" ]; then
  "${PYTHON}" - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
