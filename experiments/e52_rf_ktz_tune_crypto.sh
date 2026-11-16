#!/usr/bin/env bash
#
# Experiment E52: one rf_ktz tune FROM SCRATCH per book dataset, with the E48
# priors. The rf sibling of E28 (mlp_ktz, 9 datasets) and E50 (lgb_ktz, 9
# datasets), and the multi-dataset scale-up of E51 (which tuned raw + fenc only).
#
# E52 is the reference handle; every artefact it writes is e52_*.
#
# THE IDEA
#   E48 mapped the RF hyperparameter surface and gave the rf_ktz grid; E51 showed
#   a from-scratch tune of it works on raw and fenccausal. E52 runs that same
#   from-scratch RandomizedSearchCV on ALL NINE datasets the deployed book trains
#   on, so every book slot has an RF counterpart tuned on its own feature set --
#   the pool a later paired-vs-book comparison and any RF blend would draw from.
#
# THE GRID  src.prediction.models.RFKtzWrapper (see its docstring) -- the E48
#   priors: n_estimators PINNED 100 (fANOVA red herring), max_depth {3,4,6},
#   max_features {0.05..0.30} (dense-low colsample; 0.6/1.0 dropped), and
#   min_samples_leaf {300,1000,2000,4000} (heavy; E48's binding 1000 ceiling
#   opened). 60 cells, all in E48's good region; searched at --n-iter 40.
#
# THE CADENCE  the deployable-slot one: rolling 130/130, embargo 20, n_splits 5
#   (a REAL inner CV that selects), seed 42, val_frac 0.3, target_binned_return_20,
#   metric corr, MMC over the real meta model.
#
# PARALLELISM  robust by construction. E51's fenc arm HUNG on the nested
#   search-jobs>1 x n_jobs>1 loky path, so each arm here runs a SERIAL candidate
#   search (--search-jobs 1) with RF using cores per fit (--n-jobs), i.e. ONE
#   well-tested joblib level. Throughput comes from running RFK_POOL datasets
#   CONCURRENTLY as independent processes -- dataset-level parallelism, not
#   nested joblib.
#
# COST  narrow datasets (22-88 feat) minutes; the four wide one-hot sets
#   (763-812 feat) are the long pole. On an idle box with RFK_POOL=5 x n_jobs 4,
#   ~2-4 h total. Run it where there is CPU.
#
# Usage:
#   RFK_SCOUT=1 bash experiments/e52_rf_ktz_tune_crypto.sh              # wiring only
#   ONLY_DS="raw fenccausal" / SKIP="tenc_onehot allfeatpp"
#   DRY_RUN=1 / NO_MAIL=1 / RFK_POOL=6 / RFK_JOBS=4
#   PYTHON=/path/to/venv/bin/python bash experiments/e52_rf_ktz_tune_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E52}"
RFK_WINDOW="${RFK_WINDOW:-130}"
RFK_REFIT="${RFK_REFIT:-130}"
RFK_EMBARGO="${RFK_EMBARGO:-20}"
RFK_N_SPLITS="${RFK_N_SPLITS:-5}"
RFK_N_ITER="${RFK_N_ITER:-40}"
RFK_VAL_FRAC="${RFK_VAL_FRAC:-0.3}"
RFK_SEED="${RFK_SEED:-42}"
RFK_JOBS="${RFK_JOBS:-4}"          # RF n_jobs per fit -- the ONE joblib level
RFK_THREADS="${RFK_THREADS:-4}"
RFK_POOL="${RFK_POOL:-5}"          # datasets run concurrently (independent procs)
RFK_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${RFK_WINDOW}" in ""|0|[0-9]) echo "RFK_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${RFK_REFIT}"  in ""|0|[0-9]) echo "RFK_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${RFK_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "RFK_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e52"
if [ "${RFK_SCOUT:-0}" = "1" ]; then
    PREFIX="e52scout"; RFK_ERA_STEP=4; RFK_N_ITER=6; RFK_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${RFK_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${RFK_THREADS}"
export MKL_NUM_THREADS="${RFK_THREADS}"
export OPENBLAS_NUM_THREADS="${RFK_THREADS}"

# The nine book datasets, parallel arrays (tag, version, expected feature count,
# the slot each feature set backs -- for orientation only). Same set E28/E50 use.
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

# Preflight: every selected dataset present with meta + expected feature count,
# and the rf_ktz grid still encodes the E48 priors.
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
g = build_wrapper("rf_ktz", "regression", seed=42, n_jobs=1).param_grid
if g.get("n_estimators") != [100]:
    bad.append("rf_ktz n_estimators not pinned at 100")
if max(g["max_features"]) > 0.3 or max(g["min_samples_leaf"]) <= 1000:
    bad.append("rf_ktz grid no longer matches the E48 prior")
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files and expected feature "
      "counts; rf_ktz grid encodes the E48 priors" % len(versions))
PY

# Selection: ONLY_DS / SKIP filters, then SHARD/OF round-robin (for splitting
# across machines if ever needed).
SEL_TAG=(); SEL_VER=(); picked=0
for d in "${!DSTAG[@]}"; do
    tag="${DSTAG[$d]}"
    if [ -n "${ONLY_DS:-}" ]; then case " ${ONLY_DS} " in *" $tag "*) ;; *) continue;; esac; fi
    if [ -n "${SKIP:-}" ];    then case " ${SKIP} "    in *" $tag "*) continue;; esac; fi
    if [ "${OF}" -gt 0 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
    SEL_TAG+=("$tag"); SEL_VER+=("${DSVER[$d]}")
done
total="${#SEL_TAG[@]}"
[ "$total" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started' % sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  arms      ${total} rf_ktz tunes: ${SEL_TAG[*]}
  method    from-scratch RandomizedSearchCV over the E48-prior grid, ${RFK_N_ITER} draws,
            ${RFK_N_SPLITS}-fold CV; SERIAL search x n_jobs ${RFK_JOBS}; ${RFK_POOL} datasets concurrent
  results   results/metrics/crypto/rf_ktz/${PREFIX}_*.json
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*" 2>/dev/null \
        && echo "   emailed the start notice" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: rf_ktz from-scratch tune  |  ${total} dataset(s)  |  pool ${RFK_POOL}  |  shard ${SHARD}/${OF}"
echo "started $(date '+%F %T')  |  python=${PYTHON}  |  n_iter=${RFK_N_ITER} search-jobs=1 n-jobs=${RFK_JOBS}"
echo "=============================================================="

# Dataset-level pool: keep <= RFK_POOL run_forecast processes alive at once.
started_all=$(date +%s)
for i in "${!SEL_TAG[@]}"; do
    tag="${SEL_TAG[$i]}"; dsver="${SEL_VER[$i]}"; name="${PREFIX}_${tag}"
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/rf_ktz/${name}.json" ]; then
        echo "[skip] ${name} already on disk"; continue
    fi
    cmd=("$PYTHON" -u -m src.run_forecast crypto rf_ktz
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${RFK_WINDOW}" --refit-every "${RFK_REFIT}"
         --embargo "${RFK_EMBARGO}" --n-splits "${RFK_N_SPLITS}" --n-iter "${RFK_N_ITER}"
         --val-frac "${RFK_VAL_FRAC}" --era-step "${RFK_ERA_STEP}" --seed "${RFK_SEED}"
         --search-jobs 1 --n-jobs "${RFK_JOBS}"
         --data-version "${dsver}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-14s %s\n' "$((i+1))" "$total" "$name" "$dsver"; continue; fi
    # gate on concurrency
    while [ "$(jobs -rp | wc -l)" -ge "${RFK_POOL}" ]; do sleep 5; done
    echo "[launch $((i+1))/${total}] ${name}  ${dsver}  $(date '+%H:%M:%S')"
    ( t0=$(date +%s); "${cmd[@]}" > "logs/${name}.log" 2>&1
      echo ">>> ${name} exit=$? in $(( $(date +%s)-t0 ))s" >> "logs/${PREFIX}_pool.log" ) &
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
wait   # all dataset searches done

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- rf_ktz from-scratch tune per book dataset (E48 priors)"
el_all=$(( $(date +%s)-started_all ))
echo "wall $((el_all/60))m$((el_all%60))s over ${total} dataset(s), pool ${RFK_POOL}"
echo
"$PYTHON" - "${PREFIX}" "${SEL_TAG[@]}" <<'PY'
import json, os, sys, csv
prefix = sys.argv[1]; tags = sys.argv[2:]
rows = []
for tag in tags:
    run = "%s_%s" % (prefix, tag)
    p = "results/metrics/crypto/rf_ktz/%s.json" % run
    if not os.path.exists(p):
        print("  %-14s MISSING (see logs/%s.log)" % (tag, run)); continue
    d = json.load(open(p)); c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    rows.append((0.1*cm+mm, tag, cm, mm, c.get("sharpe") or 0.0, d.get("n_periods")))
rows.sort(reverse=True)
print("  %-14s %8s %8s %8s %8s %5s" % ("dataset","CORR","MMC","payout","c.shrp","n"))
for pay, tag, cm, mm, sh, n in rows:
    print("  %-14s %8.5f %8.5f %8.5f %8.3f %5s" % (tag, cm, mm, pay, sh, n))
PY
echo
echo "NEXT -- paired comparison of the tuned RFs vs the deployed book:"
echo "  ${PYTHON} -m src.analysis.report_html --model rf_ktz --runs $(printf '%s ' "${SEL_TAG[@]/#/${PREFIX}_}")"
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished" % sys.argv[2], open(sys.argv[1]).read())
print("emailed the report")
PY
fi
