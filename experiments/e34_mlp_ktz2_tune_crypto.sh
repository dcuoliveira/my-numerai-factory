#!/usr/bin/env bash
#
# Experiment E34: one from-scratch MLP tune per submitted-model dataset, using
# the E33 RE-SWEEP priors (model class mlp_ktz2). E28 redone on the corrected box.
#
# E34 is the reference handle; every artefact is e34_*.
#
# WHY
#   E28 tuned mlp_ktz -- the E17 priors, which capped alpha at 0.1 and pinned
#   identity. E33 proved that box excludes the region the best MLP slot (IA_01)
#   actually occupies: with alpha opened, all three swept datasets agreed the
#   peak is alpha ~10, relu reachable, width a weak axis. mlp_ktz2 is that
#   agreement compiled into a grid. E34 asks: does a from-scratch tune on the
#   CORRECTED box beat the E28 tune, and does it now reach IA_01?
#
# THE GRID (mlp_ktz2, 120 cells)
#   architecture  6   width {32,64,128} x depth {1,2}, halving
#   activation    2   identity relu        (un-pinned; relu holds the peak)
#   learning_rate 5   1e-4 .. 1e-2 log     (peak 1e-3)
#   alpha         2   {1, 10}              (~100x E17's 0.1 ceiling -- THE change)
#              = 6 x 2 x 5 x 2 = 120 cells, same footprint as mlp_ktz.
#
#   Unlike E28, activation is a FREE axis: E33 shows identity has the higher
#   median but relu the higher peak, so the search should see both. n_iter 80
#   over 120 cells samples every axis.
#
#   9 datasets x 1 tune = 9 arms:
#     e34_raw            v2.0                                      22   CURUPIRA_02,SACI_04
#     e34_tenc           v2.0-tenc                                 23   CURUPIRA_01
#     e34_interact       v2.0-interact                             36   SACI_01
#     e34_fenccausal     v2.0-fenccausal                           44   IA_01
#     e34_cumlag3        v2.0-cumlag3                              88   SACI_05
#     e34_tenc_onehot    v2.0-tenc-onehot                         763   SACI_03
#     e34_onehot_rollsel v2.0-onehot-rollsel                      786   CURUPIRA_03
#     e34_allfeat        v2.0-tenc-onehot-spread-interact         788   SACI_02
#     e34_allfeatpp      v2.0-tenc-onehot-rollsel-spread-interact 812   CURUPIRA_04
#
# WHAT IT ANSWERS
#   Paired vs the E28 old-prior pick per dataset (same data/cadence/seed), and
#   ranked against the book. The clean question: does correcting the alpha
#   ceiling move the single searched MLP up toward IA_01?
#
# COST  each tune is --n-iter 80 x inner CV x ~4 refits; ~40-70 min/dataset,
#   ~2 h across 4 shards, as E28.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e34_mlp_ktz2_tune_crypto.sh
#   SHARD=1 OF=4 bash experiments/e34_mlp_ktz2_tune_crypto.sh
#   ONLY="raw fenccausal" / KTZ_N_ITER=120 / DRY_RUN=1 / NO_MAIL=1 / KTZ_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E34}"
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

PREFIX="e34"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e34scout"; KTZ_ERA_STEP=4; KTZ_N_ITER=6; KTZ_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${KTZ_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${KTZ_THREADS}"
export MKL_NUM_THREADS="${KTZ_THREADS}"
export OPENBLAS_NUM_THREADS="${KTZ_THREADS}"

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

# Preflight: datasets present with meta files; mlp_ktz2 constructs and its grid
# encodes the E33 priors (alpha reaches 10, both activations, 120 cells); scaler.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import os, sys
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
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
w=build_wrapper("mlp_ktz2","regression",seed=42,n_jobs=1); g=w.param_grid
if sorted(g["mlp__alpha"])!=[1.0,10.0]:
    bad.append("mlp_ktz2 alpha %s != {1,10} -- the E33 correction is missing"%g["mlp__alpha"])
if sorted(g["mlp__activation"])!=["identity","relu"]:
    bad.append("mlp_ktz2 activation %s != {identity,relu}"%g["mlp__activation"])
if len(list(ParameterGrid(g)))!=120:
    bad.append("mlp_ktz2 grid is %d cells, expected 120"%len(list(ParameterGrid(g))))
if not isinstance(w.ModelClass.steps[0][1], ScaleContinuousOnly):
    bad.append("mlp_ktz2 is not using ScaleContinuousOnly")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present; mlp_ktz2 encodes the E33 priors "
      "(alpha{1,10}, both activations, 120 cells); dummy-safe scaler in place"%len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  tunes     ${total}  (of 9; shard ${SHARD}/${OF})
  model     mlp_ktz2 -- the E33 re-sweep priors (alpha{1,10}, both activations)
  each      ONE full search from scratch per dataset, ${KTZ_N_ITER} draws
  vs        the E28 old-prior (alpha<=0.1, identity) pick, paired per dataset
  cadence   rolling ${KTZ_WINDOW}/${KTZ_REFIT}, embargo ${KTZ_EMBARGO}, ${KTZ_N_SPLITS} folds
  expect    ~40-70 min per tune; ~2 h across 4 shards

Where to look:
  runner    experiments/e34_mlp_ktz2_tune_crypto.sh
  results   results/metrics/crypto/mlp_ktz2/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp_ktz2 tune per book dataset (E33 priors)  |  ${total} tunes  |  shard ${SHARD}/${OF}"
echo "grid: alpha{1,10} both-activations lr~1e-3 width32-128 depth1-2 | ${KTZ_N_ITER} draws"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp_ktz2
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
echo "${EXPERIMENT} -- mlp_ktz2 tune per submitted-model dataset (E33 priors)"
echo "  results  results/metrics/crypto/mlp_ktz2/${PREFIX}_*.json"
echo "  runner   experiments/e34_mlp_ktz2_tune_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} tunes in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; other shards hold the rest"
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import json, os, sys
ARMS=["raw","tenc","interact","fenccausal","cumlag3","tenc_onehot","onehot_rollsel","allfeat","allfeatpp"]
prefix, scout = sys.argv[1], sys.argv[2]=="1"
def load(model,run):
    p="results/metrics/crypto/%s/%s.json"%(model,run); return json.load(open(p)) if os.path.exists(p) else None
def pay(d): return 0.1*d["corr"]["mean"]+d["mmc"]["mean"]
print("mlp_ktz2 tune (e34) vs the E28 old-prior pick -- quick look at the means")
print("%-16s %5s %6s | %9s %8s | %9s | %9s"%("dataset","feat","n","e34 pay","c.sharpe","e28 pay","dpay(mean)"))
print("-"*80)
for a in ARMS:
    e=load("mlp_ktz2","%s_%s"%(prefix,a))
    if e is None: print("%-16s NOT ON DISK (other shard / not run)"%a); continue
    s=None if scout else load("mlp_ktz","e28_%s"%a)
    sp="%9.6f"%pay(s) if s else "        -"; dp="%+9.6f"%(pay(e)-pay(s)) if s else "        -"
    print("%-16s %5d %6d | %9.6f %8.3f | %9s | %9s"%(a,e["config"]["n_features"],e["n_periods"],pay(e),e["corr"]["sharpe"],sp,dp))
print("-"*80)
if scout: print("\nSCOUT -- wiring only."); raise SystemExit
print("\ndpay(mean) is a quick look. The defensible test is paired NW lag 20:")
print("  python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \\")
print("    --model mlp_ktz2 --baseline mlp_ktz/e28_fenccausal --arms %s_fenccausal"%prefix)
print("And rank vs the book (does the corrected box reach IA_01?):")
print("  python3 -m src.analysis.report_html --model mlp_ktz2 --runs %s_raw ... \\"%prefix)
print("    --out results/reports/crypto/%s_tune_vs_book.html"%prefix)
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
