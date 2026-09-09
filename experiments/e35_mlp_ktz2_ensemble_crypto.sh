#!/usr/bin/env bash
#
# Experiment E35: ENSEMBLE the mlp_ktz2 (E33-reprior) grid per book dataset,
# instead of SEARCHING it. E31 redone on the corrected box.
#
# E35 is the reference handle; every artefact is e35_*.
#
# WHY
#   E31 ensembled the mlp_ktz grid (E17 priors: alpha<=0.1, identity) and found
#   averaging beat the E28 pick but could not reach IA_01 -- because IA_01's
#   region (alpha ~10, relu) was OUTSIDE that grid. E33 corrected the box; E34
#   searches it; E35 ensembles it. The ensemble-vs-pick question (E28/E31)
#   re-asked where the grid now CONTAINS the good region.
#
# WHAT THIS RUNS
#   For each of the 9 book datasets, model class mlp_ktz2_ens: at every refit fit
#   ALL 120 cells of the mlp_ktz2 grid (alpha {1,10}, {identity,relu}, lr
#   1e-4..1e-2, width 32-128, depth 1-2) and RANK-AVERAGE their OOS predictions
#   -- no selection. Same combiner and cost notes as E31 (GridEnsembleRegressor,
#   block-level rank-average, --n-splits 2 floor). 9 datasets x 1 ensemble = 9
#   arms, same datasets/cadence/seed as E34 so each e35_<ds> pairs against its
#   e34_<ds> single-pick counterpart.
#
# WHAT IT ANSWERS
#   Paired ensemble-vs-pick per dataset (e35 vs e34, both on the corrected grid),
#   and ranked against the book (IA_01, CU_SA_IA_BO_01). Also compare to E31 (the
#   old-prior ensemble) to see how much the alpha correction moved the ensemble.
#
# COST  120 members x 2 fits/refit x ~4 refits x 9 datasets; ~E31's footprint
#   (~13 min narrow, ~40-70 min wide per dataset); ~2-3 h across 4 shards.
#
# Usage:
#   ENS_SCOUT=1 bash experiments/e35_mlp_ktz2_ensemble_crypto.sh
#   SHARD=1 OF=4 bash experiments/e35_mlp_ktz2_ensemble_crypto.sh
#   ONLY="raw fenccausal" / DRY_RUN=1 / NO_MAIL=1 / ENS_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E35}"
ENS_WINDOW="${ENS_WINDOW:-130}"
ENS_REFIT="${ENS_REFIT:-130}"
ENS_EMBARGO="${ENS_EMBARGO:-20}"
ENS_N_SPLITS="${ENS_N_SPLITS:-2}"      # floor: n_splits=1 yields 0 usable folds
ENS_N_ITER="${ENS_N_ITER:-1}"           # empty grid => one candidate
ENS_VAL_FRAC="${ENS_VAL_FRAC:-0.3}"
ENS_SEED="${ENS_SEED:-42}"
ENS_THREADS="${ENS_THREADS:-4}"
ENS_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${ENS_WINDOW}" in ""|0|[0-9]) echo "ENS_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${ENS_REFIT}"  in ""|0|[0-9]) echo "ENS_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${ENS_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "ENS_N_SPLITS must be >= 2 (1 yields no usable CV folds)" >&2; exit 1; }

PREFIX="e35"
if [ "${ENS_SCOUT:-0}" = "1" ]; then
    PREFIX="e35scout"; ENS_ERA_STEP=4
    echo "SCOUT MODE: era-step 4 -- wiring + rough cost only, not comparable."
fi

export OMP_NUM_THREADS="${ENS_THREADS}"
export MKL_NUM_THREADS="${ENS_THREADS}"
export OPENBLAS_NUM_THREADS="${ENS_THREADS}"

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

# Preflight: datasets present; mlp_ktz2_ens is the GridEnsembleRegressor pointed
# at the mlp_ktz2 grid (120 members, alpha reaches 10, both activations); scaler.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import os, sys
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
w=build_wrapper("mlp_ktz2_ens","regression",seed=42,n_jobs=1)
if not isinstance(w.ModelClass, GridEnsembleRegressor): bad.append("mlp_ktz2_ens ModelClass is not GridEnsembleRegressor")
if getattr(w.ModelClass,"base",None)!="mlp_ktz2": bad.append("mlp_ktz2_ens base is %r, expected 'mlp_ktz2'"%getattr(w.ModelClass,"base",None))
if w.param_grid != {}: bad.append("mlp_ktz2_ens grid is not empty")
base=build_wrapper("mlp_ktz2","regression",seed=42,n_jobs=1); cells=list(ParameterGrid(base.param_grid))
if len(cells)!=120: bad.append("mlp_ktz2 grid is %d cells, expected 120"%len(cells))
if sorted(base.param_grid["mlp__alpha"])!=[1.0,10.0]: bad.append("mlp_ktz2 alpha != {1,10} -- E33 correction missing")
if not isinstance(base.ModelClass.steps[0][1], ScaleContinuousOnly): bad.append("mlp_ktz2 members not using ScaleContinuousOnly")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present; mlp_ktz2_ens fits the 120-cell "
      "mlp_ktz2 grid (alpha{1,10}, both activations); dummy-safe scaler"%len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (of 9; shard ${SHARD}/${OF})
  model     mlp_ktz2_ens -- fit all 120 E33-reprior cells, rank-average
  vs        the E34 single-pick over the SAME grid, paired per dataset
  cadence   rolling ${ENS_WINDOW}/${ENS_REFIT}, embargo ${ENS_EMBARGO}, ${ENS_N_SPLITS} fold(s)

Where to look:
  runner    experiments/e35_mlp_ktz2_ensemble_crypto.sh
  results   results/metrics/crypto/mlp_ktz2_ens/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp_ktz2 grid ENSEMBLE per book dataset (E33 priors)  |  ${total} arms  |  shard ${SHARD}/${OF}"
echo "120 members (alpha{1,10} both-activations), rank-averaged | n-splits ${ENS_N_SPLITS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp_ktz2_ens
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
echo "${EXPERIMENT} -- mlp_ktz2 grid ENSEMBLE per submitted-model dataset (E33 priors)"
echo "  results  results/metrics/crypto/mlp_ktz2_ens/${PREFIX}_*.json"
echo "  runner   experiments/e35_mlp_ktz2_ensemble_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arms in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; other shards hold the rest"
echo
python3 - "${PREFIX}" "${ENS_SCOUT:-0}" <<'PY'
import json, os, sys
ARMS=["raw","tenc","interact","fenccausal","cumlag3","tenc_onehot","onehot_rollsel","allfeat","allfeatpp"]
prefix, scout = sys.argv[1], sys.argv[2]=="1"
def load(model,run):
    p="results/metrics/crypto/%s/%s.json"%(model,run); return json.load(open(p)) if os.path.exists(p) else None
def pay(d): return 0.1*d["corr"]["mean"]+d["mmc"]["mean"]
print("mlp_ktz2 ENSEMBLE (e35) vs the E34 pick vs the E31 old-prior ensemble")
print("%-16s %5s %6s | %9s %8s | %9s | %9s"%("dataset","feat","n","e35 pay","c.sharpe","e34 pay","e31 pay"))
print("-"*80)
for a in ARMS:
    e=load("mlp_ktz2_ens","%s_%s"%(prefix,a))
    if e is None: print("%-16s NOT ON DISK (other shard / not run)"%a); continue
    s=None if scout else load("mlp_ktz2","e34_%s"%a); o=None if scout else load("mlp_ktz_ens","e31_%s"%a)
    sp="%9.6f"%pay(s) if s else "        -"; op="%9.6f"%pay(o) if o else "        -"
    print("%-16s %5d %6d | %9.6f %8.3f | %9s | %9s"%(a,e["config"]["n_features"],e["n_periods"],pay(e),e["corr"]["sharpe"],sp,op))
print("-"*80)
if scout: print("\nSCOUT -- wiring only."); raise SystemExit
print("\nDefensible tests (paired NW lag 20): ensemble-vs-pick on the SAME grid,")
print("and the corrected ensemble vs the old-prior ensemble:")
print("  paired_delta.py --model mlp_ktz2_ens --baseline mlp_ktz2/e34_fenccausal --arms %s_fenccausal"%prefix)
print("  paired_delta.py --model mlp_ktz2_ens --baseline mlp_ktz_ens/e31_fenccausal --arms %s_fenccausal"%prefix)
print("Rank vs the book: report_html --model mlp_ktz2_ens --runs %s_raw ..."%prefix)
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
