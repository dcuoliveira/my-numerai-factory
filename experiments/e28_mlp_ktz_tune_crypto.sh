#!/usr/bin/env bash
#
# Experiment E28: one from-scratch MLP tune per submitted-model dataset, using
# the E17 hyperparameter priors (model class mlp_ktz).
#
# E28 is the reference handle; every artefact is e28_*.
#
# WHAT THIS RUNS
#   For each of the 9 datasets the deployed book trains on, ONE full
#   hyperparameter search from scratch over the mlp_ktz grid -- the E17 priors
#   compiled into a search space:
#
#     learning rate  DENSE, ~2e-4..5e-2 (20 values)  -- the tuning focus
#     architecture   {[32],[32,16],[32,16,8]}        -- a coarse shallow/deep gate
#     alpha          {1e-2, 1e-1}                     -- near-pinned, least important
#     activation     PINNED identity                  -- E17's dominant safe default
#                = 20 x 3 x 2 x 1 = 120 cells.
#
#   activation is PINNED, not searched, and that is what concentrates the budget
#   on lr. RandomizedSearchCV samples the discrete grid uniformly, so a free
#   activation would take ~half the draws no matter how many lr levels exist --
#   giving lr 20 levels does not down-weight it. With activation pinned and alpha
#   at 2 values, --n-iter 80 over 120 cells gives each of the 20 lr levels ~4
#   draws. That is "spend the budget on the learning rate", made true by the
#   grid shape rather than asserted.
#
#   The two known bad corners are UNREACHABLE by construction, not merely
#   discouraged: lr stops short of 1e-1 (which fails without high alpha), and the
#   architecture gate excludes the 4-5 layer nets that collapse to a constant.
#
#   9 datasets x 1 tune = 9 arms:
#     e28_raw            v2.0                                      22   CURUPIRA_02,SACI_04
#     e28_tenc           v2.0-tenc                                 23   CURUPIRA_01
#     e28_interact       v2.0-interact                             36   SACI_01
#     e28_fenccausal     v2.0-fenccausal                           44   IA_01
#     e28_cumlag3        v2.0-cumlag3                              88   SACI_05
#     e28_tenc_onehot    v2.0-tenc-onehot                         763   SACI_03
#     e28_onehot_rollsel v2.0-onehot-rollsel                      786   CURUPIRA_03
#     e28_allfeat        v2.0-tenc-onehot-spread-interact         788   SACI_02
#     e28_allfeatpp      v2.0-tenc-onehot-rollsel-spread-interact 812   CURUPIRA_04
#
# WHAT IT ANSWERS
#   Primarily: what learning rate does an evidence-primed tune SELECT on each
#   dataset? If all 9 land in ~1e-3..1e-2, the E17 lr prior transfers across
#   every feature set the book uses. That is the clean, unconfounded result.
#
#   The payout-vs-earlier-MLP comparison (E15/E08) is SECONDARY and CONFOUNDED --
#   read it with care, do not headline it:
#     * activation: E28 is identity-only; E15/E08's broad grid used relu. A
#       difference partly reflects that, not the priors.
#     * width: E28 caps the first hidden layer at 32; E08 went to (256,128),
#       sized for the 763-812-column one-hot sets. E17 never varied first-layer
#       width and never ran above 22 features, so the 32 cap on the wide datasets
#       is an UNSTATED EXTRAPOLATION -- an E28 < E08 there may be a width
#       bottleneck, not a failed prior.
#     * budget: E28 draws 80, E15/E08 drew 50 -- E28 has MORE search budget, not
#       less, so "no worse" is not "cheaper".
#   The 9 tunes also share the raw-22 core, one seed, one walk-forward, so their
#   lr optima are correlated: concordance is corroboration, not 9 independent
#   confirmations.
#
# WHAT WOULD COUNT AS A RESULT
#   This is not a single hypothesis test; it is 9 independent tunes. Read each
#   arm against its own dataset's deployed slot (paired, Newey-West lag 20). A
#   consistent learning-rate choice across datasets (all landing in ~1e-3..1e-2)
#   is itself a confirmation that the E17 prior transfers beyond raw. The report
#   prints the selected lr per refit so drift is visible.
#
# COST
#   Each tune is a full --n-iter search x 5 folds x ~4 refits. MLP fits are cheap
#   (~1-4 s) but there are many; budget ~40-70 min per dataset, ~8 h sequential,
#   ~2 h across 4 shards. The 763-812 column one-hot datasets cost no more per
#   fit than the narrow ones (an MLP's cost is its hidden layers, not its width).
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e28_mlp_ktz_tune_crypto.sh
#   SHARD=1 OF=4 bash experiments/e28_mlp_ktz_tune_crypto.sh
#   ONLY="raw fenccausal" / KTZ_N_ITER=120 / DRY_RUN=1 / NO_MAIL=1 / KTZ_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E28}"
KTZ_WINDOW="${KTZ_WINDOW:-130}"
KTZ_REFIT="${KTZ_REFIT:-130}"
KTZ_EMBARGO="${KTZ_EMBARGO:-20}"
KTZ_N_SPLITS="${KTZ_N_SPLITS:-5}"
KTZ_N_ITER="${KTZ_N_ITER:-80}"
KTZ_VAL_FRAC="${KTZ_VAL_FRAC:-0.3}"
KTZ_SEED="${KTZ_SEED:-42}"
KTZ_THREADS="${KTZ_THREADS:-4}"
KTZ_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${KTZ_WINDOW}" in ""|0|[0-9]) echo "KTZ_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${KTZ_REFIT}"  in ""|0|[0-9]) echo "KTZ_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e28"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e28scout"; KTZ_ERA_STEP=4; KTZ_N_ITER=6; KTZ_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${KTZ_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${KTZ_THREADS}"
export MKL_NUM_THREADS="${KTZ_THREADS}"
export OPENBLAS_NUM_THREADS="${KTZ_THREADS}"

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

# Preflight: datasets present with meta files, mlp_ktz constructs and its grid
# stays within the E17 corners, and the dummy-safe scaler is in place.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import os, sys
import numpy as np
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper, ScaleContinuousOnly

cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    p="data/%s/train.parquet"%v
    try: got=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    except OSError: bad.append("%s MISSING"%v); continue
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
    if not os.path.exists("data/%s/historical_meta_models.parquet"%v):
        bad.append("%s lacks historical_meta_models.parquet"%v)
w=build_wrapper("mlp_ktz","regression",seed=42,n_jobs=1)
g=w.param_grid; lr=np.asarray(g["mlp__learning_rate_init"])
if lr.min()<=1e-4 or lr.max()>=1e-1:
    bad.append("mlp_ktz lr grid %.2e..%.2e escapes the E17 corners (1e-4, 1e-1)"%(lr.min(),lr.max()))
if any(len(a)>3 for a in g["mlp__hidden_layer_sizes"]):
    bad.append("mlp_ktz has a >3-layer architecture -- those collapse")
if not isinstance(w.ModelClass.steps[0][1], ScaleContinuousOnly):
    bad.append("mlp_ktz is not using ScaleContinuousOnly -- the one-hot datasets would break")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files; mlp_ktz grid within "
      "the E17 corners; dummy-safe scaler in place"%len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  tunes     ${total}  (of 9; shard ${SHARD}/${OF})
  model     mlp_ktz -- the E17 hyperparameter priors as a search grid
  each      ONE full search from scratch per dataset, ${KTZ_N_ITER} draws,
            budget concentrated on the learning rate (20 dense levels)
  datasets  the 9 the submitted book trains on
  cadence   rolling ${KTZ_WINDOW}/${KTZ_REFIT}, embargo ${KTZ_EMBARGO}, ${KTZ_N_SPLITS} folds
  expect    ~40-70 min per tune; ~2 h across 4 shards

Where to look:
  runner    experiments/e28_mlp_ktz_tune_crypto.sh
  results   results/metrics/crypto/mlp_ktz/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp_ktz tune per book dataset  |  ${total} tunes  |  shard ${SHARD}/${OF}"
echo "grid: E17 priors (lr dense, arch gate, alpha near-pinned) | ${KTZ_N_ITER} draws | ${KTZ_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp_ktz
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${KTZ_WINDOW}" --refit-every "${KTZ_REFIT}"
         --embargo "${KTZ_EMBARGO}" --n-splits "${KTZ_N_SPLITS}" --n-iter "${KTZ_N_ITER}"
         --val-frac "${KTZ_VAL_FRAC}" --era-step "${KTZ_ERA_STEP}" --seed "${KTZ_SEED}"
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
echo "${EXPERIMENT} -- mlp_ktz tune per submitted-model dataset (E17 priors)"
echo "  results  results/metrics/crypto/mlp_ktz/${PREFIX}_*.json"
echo "  runner   experiments/e28_mlp_ktz_tune_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} tunes in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; other shards hold the rest"
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import glob, json, os, sys
import numpy as np, pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
ARMS = [("raw","CURUPIRA_02","ridge/w6r6_fixed_reg"),
        ("tenc","CURUPIRA_01","ridge_glmnet/w6r6_glmnet_reg_tenc"),
        ("interact","SACI_01",None), ("fenccausal","IA_01",None),
        ("cumlag3","SACI_05",None), ("tenc_onehot","SACI_03",None),
        ("onehot_rollsel","CURUPIRA_03",None), ("allfeat","SACI_02",None),
        ("allfeatpp","CURUPIRA_04",None)]

def load(run):
    p="results/metrics/crypto/mlp_ktz/%s.json"%run
    return json.load(open(p)) if os.path.exists(p) else None

def chosen_lrs(d):
    return [r["best_params"]["mlp__learning_rate_init"] for r in d.get("refits",[])]

print("%-20s %5s %6s %10s %8s %8s   %s" % (
    "tune","feat","n","payout","c.sharpe","dd(corr)","lr chosen per refit"))
print("-"*104)
for arm,slot,_ in ARMS:
    d=load("%s_%s"%(prefix,arm))
    if d is None:
        print("%-20s  NOT ON DISK -- not run (or another shard)"%("%s_%s"%(prefix,arm)))
        continue
    lrs=chosen_lrs(d)
    print("%-20s %5d %6d %10.6f %8.3f %8.3f   %s" % (
        "%s_%s"%(prefix,arm), d["config"]["n_features"], d["n_periods"],
        0.1*d["corr"]["mean"]+d["mmc"]["mean"], d["corr"]["sharpe"],
        d["corr"]["max_drawdown"], " ".join("%.1e"%x for x in lrs)))
print("-"*104)
if scout:
    print("\nSCOUT RUN -- wiring only, not comparable.")
    raise SystemExit
print()
print("The learning rate the tune SELECTS is the headline: if every dataset lands")
print("in ~1e-3..1e-2, the E17 prior transfers beyond raw and the dense lr sweep")
print("earned its budget. A dataset drifting to the grid edge (2e-4 or 5e-2) is a")
print("sign its optimum is outside the E17 window -- worth noting, not alarming.")
print()
print("Each tune's payout is one full from-scratch search on that dataset. Compare")
print("it to the deployed slot's own backtest and to the earlier broad-grid MLP")
print("(E15 raw/tenc/interact/fenccausal, E08 the one-hot sets) with paired_delta:")
print("  python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \\")
print("    --model mlp_ktz --baseline mlp/e15_mlp_fenccausal --arms %s_fenccausal"%prefix)
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
