#!/usr/bin/env bash
#
# Experiment E53: Gaussian Process VIABILITY TEST -- one gp arm per (book dataset
# x task), regression and classification, before the expensive E54 KTZ sweep.
#
# E53 is the reference handle; every artefact it writes is e53_*.
#
# THE IDEA
#   The GP is the first kernel model on the board. Before spending the E54
#   sensitivity budget, confirm the wrapper runs end to end on both tasks across
#   the whole book, that the subsample keeps a fit tractable, and -- the real
#   decision -- whether regression or classification carries a signal worth
#   carrying forward. A dead E53 (as E18 was for TabNet) stops the line cheaply.
#
# THE MODEL  src.prediction.models.GPWrapper ("gp"). A subsampled exact GP
#   (src/prediction/gp.py): sklearn's GaussianProcessRegressor (posterior mean)
#   for regression, GaussianProcessClassifier (positive-class probability, the
#   target binarised at 0.5) for classification. The kernel is FROZEN and the
#   outer search tunes the family + length-scale (+ alpha for regression); an
#   exact GP is O(n^3), so `subsample` caps the training rows -- passed via
#   --model-options so it is recorded in the metrics json, not searched.
#
# THE CADENCE  the deployable-slot one, identical to E52: rolling 130/130,
#   embargo 20, n_splits 5, seed 42, val_frac 0.3, target_binned_return_20,
#   metric corr, MMC over the real meta model. Both tasks are SCORED on the same
#   ordinal target (classification only changes the training loss), so all arms
#   rank in one report block against the deployed book.
#
# PARALLELISM  E52's pattern. A GP fit is one BLAS Cholesky, so --search-jobs 1
#   and --n-jobs are effectively single-threaded per arm; throughput comes from
#   running GP_POOL arms CONCURRENTLY as independent processes, and BLAS threads
#   are bounded by OMP/MKL/OPENBLAS_NUM_THREADS. The four wide one-hot datasets
#   (763-812 feat) are the long pole.
#
# Usage:
#   GP_SCOUT=1 bash experiments/e53_gp_test_crypto.sh                 # wiring only
#   ONLY_DS="raw fenccausal" / SKIP="tenc_onehot allfeatpp"
#   GP_TASKS="reg" / GP_TASKS="clf"
#   DRY_RUN=1 / NO_MAIL=1 / GP_POOL=6 / GP_SUBSAMPLE=4000
#   SHARD=1 OF=4 bash experiments/e53_gp_test_crypto.sh
#   PYTHON=/path/to/venv/bin/python bash experiments/e53_gp_test_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E53}"
GP_WINDOW="${GP_WINDOW:-130}"
GP_REFIT="${GP_REFIT:-130}"
GP_EMBARGO="${GP_EMBARGO:-20}"
GP_N_SPLITS="${GP_N_SPLITS:-5}"
GP_N_ITER="${GP_N_ITER:-12}"
GP_VAL_FRAC="${GP_VAL_FRAC:-0.3}"
GP_SEED="${GP_SEED:-42}"
GP_JOBS="${GP_JOBS:-1}"             # gp drops n_jobs; BLAS threads are the knob
GP_THREADS="${GP_THREADS:-4}"
GP_SUBSAMPLE="${GP_SUBSAMPLE:-6000}"     # regression row cap (O(n^3))
GP_CLF_SUBSAMPLE="${GP_CLF_SUBSAMPLE:-1500}"  # clf: Laplace is O(n^3) PER step
GP_POOL="${GP_POOL:-4}"            # arms run concurrently (independent procs)
GP_ERA_STEP=1
GP_TASKS="${GP_TASKS:-reg clf}"
SHARD="${SHARD:-0}"; OF="${OF:-0}"
# GNU screen exports WINDOW=0 into every session; guard the two that come from it.
case "${GP_WINDOW}" in ""|0|[0-9]) echo "GP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${GP_REFIT}"  in ""|0|[0-9]) echo "GP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${GP_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "GP_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e53"
if [ "${GP_SCOUT:-0}" = "1" ]; then
    PREFIX="e53scout"; GP_ERA_STEP=4; GP_N_ITER=3; GP_N_SPLITS=2
    GP_SUBSAMPLE=1500; GP_CLF_SUBSAMPLE=800
    echo "SCOUT MODE: era-step 4, ${GP_N_ITER} draws, subsample ${GP_SUBSAMPLE} -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${GP_THREADS}"
export MKL_NUM_THREADS="${GP_THREADS}"
export OPENBLAS_NUM_THREADS="${GP_THREADS}"

# The nine book datasets, parallel arrays (tag, version, expected feature count,
# the slot each feature set backs -- for orientation only). Same set E28/E50/E52
# use.
DSTAG=(); DSVER=(); DSFEAT=()
add() { DSTAG+=("$1"); DSVER+=("$2"); DSFEAT+=("$3"); }
add raw            "crypto/v2.0"                                      22   # CURUPIRA_02
add tenc           "crypto/v2.0-tenc"                                 23   # CURUPIRA_01
add interact       "crypto/v2.0-interact"                             36   # SACI_01
add fenccausal     "crypto/v2.0-fenccausal"                          44   # IA_01
add cumlag3        "crypto/v2.0-cumlag3"                             88   # SACI_05
add tenc_onehot    "crypto/v2.0-tenc-onehot"                        763   # SACI_03
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                     786   # CURUPIRA_03
add allfeat        "crypto/v2.0-tenc-onehot-spread-interact"        788   # SACI_02
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812  # CURUPIRA_04

# Task arrays: the CLI arg and the run-name key.
TKEY=(); TARG=()
for t in ${GP_TASKS}; do
    case "$t" in
        reg) TKEY+=(reg); TARG+=(regression);;
        clf) TKEY+=(clf); TARG+=(classification);;
        *) echo "unknown task '$t' (want reg or clf)" >&2; exit 1;;
    esac
done

# Preflight: every selected dataset present with meta + expected feature count,
# and the gp wrapper builds for BOTH tasks and accepts the subsample option.
"$PYTHON" - "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper
cut = sys.argv.index("--"); versions = sys.argv[1:cut]; wants = [int(x) for x in sys.argv[cut+1:]]
bad = []
for v, want in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    try:
        n = sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
        if n != want:
            bad.append("%s has %d features, expected %d" % (v, n, want))
    except OSError:
        bad.append("%s MISSING" % v)
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)
for t in ("regression", "classification"):
    try:
        build_wrapper("gp", t, seed=42, n_jobs=1, subsample=6000)
    except Exception as error:
        bad.append("gp %s build failed: %s" % (t, error))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files and expected feature "
      "counts; gp builds for reg+clf and takes the subsample option" % len(versions))
PY

# Build the work list: (name, dsver, taskarg, subsample), one per dataset x task,
# after ONLY_DS/SKIP filters; then SHARD/OF round-robin across machines.
WL_NAME=(); WL_VER=(); WL_TARG=(); WL_SUB=(); picked=0
for d in "${!DSTAG[@]}"; do
    tag="${DSTAG[$d]}"
    if [ -n "${ONLY_DS:-}" ]; then case " ${ONLY_DS} " in *" $tag "*) ;; *) continue;; esac; fi
    if [ -n "${SKIP:-}" ];    then case " ${SKIP} "    in *" $tag "*) continue;; esac; fi
    for k in "${!TKEY[@]}"; do
        tkey="${TKEY[$k]}"; targ="${TARG[$k]}"
        [ "$tkey" = "clf" ] && sub="${GP_CLF_SUBSAMPLE}" || sub="${GP_SUBSAMPLE}"
        if [ "${OF}" -gt 0 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
        WL_NAME+=("${PREFIX}_gp_${tag}_${tkey}"); WL_VER+=("${DSVER[$d]}")
        WL_TARG+=("$targ"); WL_SUB+=("$sub")
    done
done
total="${#WL_NAME[@]}"
[ "$total" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started' % sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  arms      ${total} gp arms (dataset x task): ${GP_TASKS} over ${#DSTAG[@]} datasets
  method    light --n-iter ${GP_N_ITER} search of the base gp grid; subsample reg=${GP_SUBSAMPLE} clf=${GP_CLF_SUBSAMPLE}
            SERIAL search x n_jobs ${GP_JOBS}; ${GP_POOL} arms concurrent
  results   results/metrics/crypto/gp/${PREFIX}_*.json
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*" 2>/dev/null \
        && echo "   emailed the start notice" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: gp viability test  |  ${total} arm(s)  |  pool ${GP_POOL}  |  shard ${SHARD}/${OF}"
echo "started $(date '+%F %T')  |  python=${PYTHON}  |  n_iter=${GP_N_ITER} search-jobs=1 n-jobs=${GP_JOBS}"
echo "=============================================================="

# Arm-level pool: keep <= GP_POOL run_forecast processes alive at once.
started_all=$(date +%s)
for i in "${!WL_NAME[@]}"; do
    name="${WL_NAME[$i]}"; dsver="${WL_VER[$i]}"; targ="${WL_TARG[$i]}"; sub="${WL_SUB[$i]}"
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/gp/${name}.json" ]; then
        echo "[skip] ${name} already on disk"; continue
    fi
    cmd=("$PYTHON" -u -m src.run_forecast crypto gp
         --task "${targ}" --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${GP_WINDOW}" --refit-every "${GP_REFIT}"
         --embargo "${GP_EMBARGO}" --n-splits "${GP_N_SPLITS}" --n-iter "${GP_N_ITER}"
         --val-frac "${GP_VAL_FRAC}" --era-step "${GP_ERA_STEP}" --seed "${GP_SEED}"
         --search-jobs 1 --n-jobs "${GP_JOBS}"
         --model-options "{\"subsample\": ${sub}}"
         --data-version "${dsver}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-24s %-42s %s sub=%s\n' "$((i+1))" "$total" "$name" "$dsver" "$targ" "$sub"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${GP_POOL}" ]; do sleep 5; done
    echo "[launch $((i+1))/${total}] ${name}  ${dsver}  ${targ}  $(date '+%H:%M:%S')"
    ( t0=$(date +%s); "${cmd[@]}" > "logs/${name}.log" 2>&1
      echo ">>> ${name} exit=$? in $(( $(date +%s)-t0 ))s" >> "logs/${PREFIX}_pool.log" ) &
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
wait   # all arms done

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- gp viability test (dataset x task)"
el_all=$(( $(date +%s)-started_all ))
echo "wall $((el_all/60))m$((el_all%60))s over ${total} arm(s), pool ${GP_POOL}"
echo
"$PYTHON" - "${WL_NAME[@]}" <<'PY'
import json, os, sys
names = sys.argv[1:]
rows = []
for run in names:
    p = "results/metrics/crypto/gp/%s.json" % run
    if not os.path.exists(p):
        print("  %-26s MISSING (see logs/%s.log)" % (run, run)); continue
    d = json.load(open(p)); c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    rows.append((0.1*cm+mm, run, cm, mm, c.get("sharpe") or 0.0, d.get("n_periods")))
rows.sort(reverse=True)
print("  %-26s %8s %8s %8s %8s %5s" % ("arm","CORR","MMC","payout","c.shrp","n"))
for pay, run, cm, mm, sh, n in rows:
    print("  %-26s %8.5f %8.5f %8.5f %8.3f %5s" % (run, cm, mm, pay, sh, n))
PY
echo
echo "NEXT -- paired comparison of the gp arms vs the deployed book"
echo "(both tasks share the scored target, so they rank in one block):"
echo "  ${PYTHON} -m src.analysis.report_html --model gp --runs $(printf '%s ' "${WL_NAME[@]}")"
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished" % sys.argv[2], open(sys.argv[1]).read())
print("emailed the report")
PY
fi
