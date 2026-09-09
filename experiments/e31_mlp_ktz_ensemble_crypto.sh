#!/usr/bin/env bash
#
# Experiment E31: ENSEMBLE the mlp_ktz grid per book dataset, instead of
# SEARCHING it. The MLP counterpart of E30 (which does the same for lgb_ktz).
#
# E31 is the reference handle; every artefact is e31_*.
#
# WHAT THIS RUNS
#   For each of the 9 datasets the deployed book trains on, ONE walk-forward of
#   the model class mlp_ktz_ens: at every refit it fits ALL 120 cells of the
#   E17-prior mlp_ktz grid on the training block and RANK-AVERAGES their
#   out-of-sample predictions -- no inner-CV selection. The 120 cells are
#
#     learning rate  20 dense levels, ~2e-4..5e-2
#     architecture   {[32],[32,16],[32,16,8]}   -- shallow/deep gate
#     alpha          {1e-2, 1e-1}               -- near-pinned
#     activation     identity                    -- pinned
#                = 20 x 3 x 2 x 1 = 120 cells, each a fitted member.
#
#   The combiner is a per-member rank-to-[0,1] then average -- the same operator
#   run_ensemble.py and the deployed CURUPIRA_SACI/CU_SA blends use. It is done
#   INSIDE the estimator (GridEnsembleRegressor), which never sees period labels
#   at predict time, so it ranks over the whole predicted block rather than per
#   period. test_pipeline asserts that block-vs-per-period difference is
#   negligible at scoring time (numerai_corr re-ranks per period anyway).
#
#   9 datasets x 1 ensemble = 9 arms, the SAME datasets/cadence/seed as E28 so
#   each e31_<ds> is paired against its e28_<ds> single-pick counterpart:
#     e31_raw            v2.0                                      22   CURUPIRA_02,SACI_04
#     e31_tenc           v2.0-tenc                                 23   CURUPIRA_01
#     e31_interact       v2.0-interact                             36   SACI_01
#     e31_fenccausal     v2.0-fenccausal                           44   IA_01
#     e31_cumlag3        v2.0-cumlag3                              88   SACI_05
#     e31_tenc_onehot    v2.0-tenc-onehot                         763   SACI_03
#     e31_onehot_rollsel v2.0-onehot-rollsel                      786   CURUPIRA_03
#     e31_allfeat        v2.0-tenc-onehot-spread-interact         788   SACI_02
#     e31_allfeatpp      v2.0-tenc-onehot-rollsel-spread-interact 812   CURUPIRA_04
#
# WHAT IT ANSWERS
#   Does averaging the WHOLE mlp_ktz grid beat PICKING the search-selected best
#   (E28)? E28 showed a single searched mlp_ktz loses to the book, whose strength
#   is blending; this asks whether the same rank-average move applied to a
#   model's own grid recovers the gap. The clean comparison is the paired
#   ensemble-vs-pick delta per dataset (both mlp on identical data/cadence/seed).
#   Secondary: does any ensemble reach the deployed book (IA_01, CU_SA_IA_BO_01)?
#
#   Read with care: equal-weighting ALL 120 cells includes the lr-tail members
#   (2e-4 underfit, 5e-2 noisy) at full weight. If the ensemble underperforms,
#   the natural follow-up is a CV-weighted or lr-trimmed blend -- not evidence
#   that ensembling fails. On clean synthetic data the equal-weight ensemble
#   lands at the MEAN member, below the best; the case for it is that on a noisy
#   26-effective-window CV the "best member" is a noisy pick, and averaging is
#   immune to that selection variance.
#
#   TWO CAVEATS the report must carry (from the pre-launch review):
#   * ENSEMBLE-vs-BEST-MEMBER is the arm-board skill's mandated question and is
#     NOT in these artifacts -- the single meta-estimator discards per-cell OOS
#     predictions. It IS recoverable: the 120 cells are a deterministic function
#     of (dataset, seed, cadence), so a --fixed-params sweep (the E30 pattern)
#     reconstructs them for the oracle/best-member number on any dataset. Compute
#     it for the headline dataset (fenccausal, the IA_01 pair) at report time.
#   * The block-vs-per-period combiner equivalence is validated only on narrow
#     feature sets. The 4 WIDE one-hot arms (tenc_onehot, onehot_rollsel,
#     allfeat, allfeatpp) rest on it -- identity ticker intercepts can give
#     members period-correlated levels that do not cancel. The LIVE artifact is
#     unaffected (one period/round => block==period); the backtest blocks are the
#     exposure. Cross-check one wide arm's block-rank against a per-period
#     run_ensemble of its reconstructed members before trusting those 4.
#
# COST
#   The machinery has NO "one candidate => skip CV" path: a selection-free
#   estimator is still fit (usable folds) + 1 times per refit. n_splits=1 skips
#   its only fold (nothing precedes period 0), so this runs at --n-splits 2 --
#   1 usable fold + 1 refit = 2 ensemble fits per refit, the floor. Each fit trains 120
#   tiny nets (max [32,16,8]); MLP cost is its layers, not its width, so the
#   763-812-column one-hot sets cost little more than raw. SCOUT first to price
#   it -- the per-member cost on a real block is unmeasured in this repo.
#
# Usage:
#   ENS_SCOUT=1 bash experiments/e31_mlp_ktz_ensemble_crypto.sh
#   SHARD=1 OF=4 bash experiments/e31_mlp_ktz_ensemble_crypto.sh
#   ONLY="raw fenccausal" / DRY_RUN=1 / NO_MAIL=1 / ENS_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E31}"
ENS_WINDOW="${ENS_WINDOW:-130}"
ENS_REFIT="${ENS_REFIT:-130}"
ENS_EMBARGO="${ENS_EMBARGO:-20}"
ENS_N_SPLITS="${ENS_N_SPLITS:-2}"     # the CV floor: n_splits=1 skips its only fold
                                       # (no history before period 0), so 2 is the
                                       # minimum -> 1 usable fold + 1 refit = 2 fits
ENS_N_ITER="${ENS_N_ITER:-1}"          # empty grid => one candidate, nothing to search
ENS_VAL_FRAC="${ENS_VAL_FRAC:-0.3}"
ENS_SEED="${ENS_SEED:-42}"
ENS_THREADS="${ENS_THREADS:-4}"
ENS_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${ENS_WINDOW}" in ""|0|[0-9]) echo "ENS_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${ENS_REFIT}"  in ""|0|[0-9]) echo "ENS_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
# n_splits=1 skips its only fold (nothing precedes period 0) -> 0 usable folds
# -> run_forecast errors. 2 is the floor; guard an override before any work.
[ "${ENS_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "ENS_N_SPLITS must be >= 2 (1 yields no usable CV folds)" >&2; exit 1; }

PREFIX="e31"
if [ "${ENS_SCOUT:-0}" = "1" ]; then
    PREFIX="e31scout"; ENS_ERA_STEP=4
    echo "SCOUT MODE: era-step 4 -- wiring + rough per-fit cost only, not comparable."
fi

export OMP_NUM_THREADS="${ENS_THREADS}"
export MKL_NUM_THREADS="${ENS_THREADS}"
export OPENBLAS_NUM_THREADS="${ENS_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
NAME=(); DATA=(); NFEAT=(); SLOT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); SLOT+=("$4"); }
add raw            "crypto/v2.0"                                     22 CURUPIRA_02
add tenc           "crypto/v2.0-tenc"                                23 CURUPIRA_01
add interact       "crypto/v2.0-interact"                            36 SACI_01
add fenccausal     "crypto/v2.0-fenccausal"                          44 IA_01
add cumlag3        "crypto/v2.0-cumlag3"                             88 SACI_05
add tenc_onehot    "crypto/v2.0-tenc-onehot"                        763 SACI_03
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                     786 CURUPIRA_03
add allfeat        "crypto/v2.0-tenc-onehot-spread-interact"        788 SACI_02
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 CURUPIRA_04

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=(); SEL_SLOT=(); picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}"); SEL_SLOT+=("${SLOT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

# Preflight: datasets present with meta files; mlp_ktz_ens constructs; its member
# grid is exactly the 120-cell mlp_ktz grid within the E17 corners; the estimator
# is the GridEnsembleRegressor and carries the dummy-safe scaler on its members.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import os, sys
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper, GridEnsembleRegressor, ScaleContinuousOnly

cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    p="data/%s/train.parquet"%v
    try: got=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    except OSError: bad.append("%s MISSING"%v); continue
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
    if not os.path.exists("data/%s/historical_meta_models.parquet"%v):
        bad.append("%s lacks historical_meta_models.parquet"%v)

w=build_wrapper("mlp_ktz_ens","regression",seed=42,n_jobs=1)
if not isinstance(w.ModelClass, GridEnsembleRegressor):
    bad.append("mlp_ktz_ens ModelClass is not GridEnsembleRegressor")
if w.param_grid != {}:
    bad.append("mlp_ktz_ens grid is not empty -- it would relocate selection, not remove it")

base=build_wrapper("mlp_ktz","regression",seed=42,n_jobs=1)
cells=list(ParameterGrid(base.param_grid)); lr=np.asarray(base.param_grid["mlp__learning_rate_init"])
if len(cells)!=120:
    bad.append("mlp_ktz grid is %d cells, expected 120 members"%len(cells))
if lr.min()<=1e-4 or lr.max()>=1e-1:
    bad.append("mlp_ktz lr grid %.2e..%.2e escapes the E17 corners (1e-4, 1e-1)"%(lr.min(),lr.max()))
if any(len(a)>3 for a in base.param_grid["mlp__hidden_layer_sizes"]):
    bad.append("mlp_ktz has a >3-layer architecture -- those collapse")
if not isinstance(base.ModelClass.steps[0][1], ScaleContinuousOnly):
    bad.append("mlp_ktz members are not using ScaleContinuousOnly -- one-hot datasets would break")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files; mlp_ktz_ens fits the "
      "120-cell mlp_ktz grid within the E17 corners; dummy-safe scaler in place"%len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (of 9; shard ${SHARD}/${OF})
  model     mlp_ktz_ens -- fit all 120 E17-prior cells, rank-average (no select)
  each      ONE walk-forward per dataset; 120 members refit per block
  datasets  the 9 the submitted book trains on (paired vs the E28 single-pick)
  cadence   rolling ${ENS_WINDOW}/${ENS_REFIT}, embargo ${ENS_EMBARGO}, ${ENS_N_SPLITS} fold(s)
  expect    unmeasured per-fit; SCOUT priced it before launch

Where to look:
  runner    experiments/e31_mlp_ktz_ensemble_crypto.sh
  results   results/metrics/crypto/mlp_ktz_ens/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp_ktz grid ENSEMBLE per book dataset  |  ${total} arms  |  shard ${SHARD}/${OF}"
echo "120 members (E17 grid), rank-averaged | n-splits ${ENS_N_SPLITS} | ${ENS_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp_ktz_ens
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${ENS_WINDOW}" --refit-every "${ENS_REFIT}"
         --embargo "${ENS_EMBARGO}" --n-splits "${ENS_N_SPLITS}" --n-iter "${ENS_N_ITER}"
         --val-frac "${ENS_VAL_FRAC}" --era-step "${ENS_ERA_STEP}" --seed "${ENS_SEED}"
         --search-jobs 1 --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %4s feat  %-46s slot %s\n' "$index" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${SEL_DATA[$i]}" "${SEL_SLOT[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${run_name}  ${SEL_NFEAT[$i]} feat  (slot ${SEL_SLOT[$i]})  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${run_name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing (logs/${run_name}.log)"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- mlp_ktz grid ENSEMBLE per submitted-model dataset"
echo "  results  results/metrics/crypto/mlp_ktz_ens/${PREFIX}_*.json"
echo "  runner   experiments/e31_mlp_ktz_ensemble_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arms in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; other shards hold the rest"
echo
python3 - "${PREFIX}" "${ENS_SCOUT:-0}" <<'PY'
import glob, json, os, sys
import numpy as np

prefix, scout = sys.argv[1], sys.argv[2] == "1"
ARMS = ["raw","tenc","interact","fenccausal","cumlag3",
        "tenc_onehot","onehot_rollsel","allfeat","allfeatpp"]

def load(model, run):
    p="results/metrics/crypto/%s/%s.json"%(model,run)
    return json.load(open(p)) if os.path.exists(p) else None

def pay(d): return 0.1*d["corr"]["mean"]+d["mmc"]["mean"]

print("ensemble (e31) vs the E28 single-pick, per dataset -- quick look at the means")
print("%-16s %5s %6s | %9s %8s %8s | %9s | %9s" % (
    "dataset","feat","n","e31 pay","c.sharpe","dd(corr)","e28 pay","dpay(mean)"))
print("-"*92)
for arm in ARMS:
    e=load("mlp_ktz_ens","%s_%s"%(prefix,arm))
    if e is None:
        print("%-16s  NOT ON DISK -- not run (or another shard)"%arm); continue
    ep=pay(e)
    # A scout is era-step 4; the E28 pick is era-step 1 -- they share neither the
    # sample nor the split, so differencing them is meaningless. Pair only full runs.
    s=None if scout else load("mlp_ktz","e28_%s"%arm)
    sp="%9.6f"%pay(s) if s else "        -"
    dp="%+9.6f"%(ep-pay(s)) if s else "        -"
    print("%-16s %5d %6d | %9.6f %8.3f %8.3f | %9s | %9s" % (
        arm, e["config"]["n_features"], e["n_periods"], ep,
        e["corr"]["sharpe"], e["corr"]["max_drawdown"], sp, dp))
print("-"*92)
if scout:
    print("\nSCOUT RUN -- wiring + rough cost only, not comparable.")
    raise SystemExit
print()
print("dpay(mean) is a QUICK LOOK, not a claim: it differences two summary means.")
print("The defensible ensemble-vs-pick test is paired Newey-West at lag 20 on the")
print("common periods -- both arms share dataset, cadence and seed, so they pair:")
print("  python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \\")
print("    --model mlp_ktz_ens --baseline mlp_ktz/e28_fenccausal --arms %s_fenccausal"%prefix)
print()
print("And rank the ensembles against the deployed book (IA_01, CU_SA_IA_BO_01)")
print("with the full report -- split by target, paired deltas, prediction corr:")
print("  python3 -m src.analysis.report_html --model mlp_ktz_ens \\")
print("    --runs %s_raw %s_tenc %s_interact %s_fenccausal %s_cumlag3 \\"%((prefix,)*5))
print("           %s_tenc_onehot %s_onehot_rollsel %s_allfeat %s_allfeatpp \\"%((prefix,)*4))
print("    --out results/reports/crypto/%s_ensemble_vs_book.html"%prefix)
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
