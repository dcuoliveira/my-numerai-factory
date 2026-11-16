#!/usr/bin/env bash
#
# Experiment E57: rf_ktz tuned FROM SCRATCH on the 60-day label, for the five
# datasets that scored highest on MMC in E52 (the 20-day rf_ktz tune per book
# dataset). The 60-day twin of E52's top-MMC arms.
#
# E57 is the reference handle; every artefact it writes is e57_*.
#
# THE IDEA
#   E52 tuned rf_ktz on all 9 book datasets on target_binned_return_20 and ranked
#   them by MMC. RF's MMC is what a slot pays over the meta model, so E57 asks the
#   natural follow-up: do the FIVE best-MMC RF datasets keep that strength when
#   the label is the 60-DAY return instead of the 20-day one? The 60-day book
#   (SACI_03, CURUPIRA_05_60) pays ~0.079 MMC / ~0.093 payout; this measures
#   whether a from-scratch RF reaches it on that label.
#
#   Top-5 by E52 MMC (t20): allfeat 0.07919, fenccausal 0.07806, tenc_onehot
#   0.07635, allfeatpp 0.07576, interact 0.07570.
#
# THE ONLY CHANGE FROM E52  the label. Same rf_ktz grid (the E48 priors: n_est
#   pinned 100, depth {3,4,6}, max_features {0.05..0.30}, min_samples_leaf
#   {300..4000}), same --n-iter 40 from-scratch RandomizedSearchCV, same rolling
#   130/130 cadence, n_splits 5, seed 42. Two consequences of --target
#   target_binned_return_60:
#     * --embargo auto derives 60 from the target horizon (not 20), so the
#       walk-forward is leak-free for the longer label.
#     * 60-day runs score on ~470 periods, not 510 (consecutive periods overlap
#       ~98%), so treat the arm counts and any t-stat with the usual care.
#
# MATCHED ENV  run on the SAME sklearn (0.24, the .e48venv) E52 used, so the only
#   thing separating E57's numbers from E52's is the label, not the RF version.
#
# COST  five from-scratch searches; ~35-45 min each, pooled -> under an hour on an
#   idle box. RFK_POOL datasets run concurrently as independent processes.
#
# Usage:
#   RFK_SCOUT=1 bash experiments/e57_rf_ktz_60d_tune_crypto.sh          # wiring only
#   ONLY_DS="allfeat fenccausal" / DRY_RUN=1 / NO_MAIL=1 / RFK_POOL=5
#   PYTHON=/home/danielco/.e48venv/bin/python bash experiments/e57_rf_ktz_60d_tune_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E57}"
RFK_TARGET="${RFK_TARGET:-target_binned_return_60}"
RFK_WINDOW="${RFK_WINDOW:-130}"
RFK_REFIT="${RFK_REFIT:-130}"
RFK_EMBARGO="${RFK_EMBARGO:-auto}"      # auto -> 60 for the 60-day target
RFK_N_SPLITS="${RFK_N_SPLITS:-5}"
RFK_N_ITER="${RFK_N_ITER:-40}"
RFK_VAL_FRAC="${RFK_VAL_FRAC:-0.3}"
RFK_SEED="${RFK_SEED:-42}"
RFK_JOBS="${RFK_JOBS:-4}"
RFK_THREADS="${RFK_THREADS:-4}"
RFK_POOL="${RFK_POOL:-5}"
RFK_ERA_STEP=1
case "${RFK_WINDOW}" in ""|0|[0-9]) echo "RFK_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
[ "${RFK_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "RFK_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e57"
if [ "${RFK_SCOUT:-0}" = "1" ]; then
    PREFIX="e57scout"; RFK_ERA_STEP=4; RFK_N_ITER=6; RFK_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${RFK_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${RFK_THREADS}"
export MKL_NUM_THREADS="${RFK_THREADS}"
export OPENBLAS_NUM_THREADS="${RFK_THREADS}"

# The five top-MMC datasets from E52 (tag, version, expected feature count).
DSTAG=(); DSVER=(); DSFEAT=()
add() { DSTAG+=("$1"); DSVER+=("$2"); DSFEAT+=("$3"); }
add allfeat     "crypto/v2.0-tenc-onehot-spread-interact"         788   # E52 MMC #1
add fenccausal  "crypto/v2.0-fenccausal"                          44    # E52 MMC #2
add tenc_onehot "crypto/v2.0-tenc-onehot"                         763   # E52 MMC #3
add allfeatpp   "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812   # E52 MMC #4
add interact    "crypto/v2.0-interact"                            36    # E52 MMC #5

# Preflight: datasets present with meta + expected feature count + the 60-day
# target column, and the rf_ktz grid still encodes the E48 priors.
"$PYTHON" - "${RFK_TARGET}" "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper
target = sys.argv[1]
cut = sys.argv.index("--"); versions = sys.argv[2:cut]; wants = [int(x) for x in sys.argv[cut+1:]]
bad = []
for v, want in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    try:
        cols = pq.ParquetFile(p).schema.names
        n = sum(c.startswith("feature_") for c in cols)
        if n != want:
            bad.append("%s has %d features, expected %d" % (v, n, want))
        if target not in cols:
            bad.append("%s lacks the %s column" % (v, target))
    except OSError:
        bad.append("%s MISSING" % v)
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)
g = build_wrapper("rf_ktz", "regression", seed=42, n_jobs=1).param_grid
if g.get("n_estimators") != [100] or max(g["max_features"]) > 0.3 or max(g["min_samples_leaf"]) <= 1000:
    bad.append("rf_ktz grid no longer matches the E48 prior")
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files, expected feature "
      "counts and the %s column; rf_ktz grid encodes the E48 priors" % (len(versions), target))
PY

# Selection.
SEL_TAG=(); SEL_VER=()
for d in "${!DSTAG[@]}"; do
    tag="${DSTAG[$d]}"
    if [ -n "${ONLY_DS:-}" ]; then case " ${ONLY_DS} " in *" $tag "*) ;; *) continue;; esac; fi
    SEL_TAG+=("$tag"); SEL_VER+=("${DSVER[$d]}")
done
total="${#SEL_TAG[@]}"
[ "$total" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started' % sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  arms      ${total} rf_ktz 60-day tunes: ${SEL_TAG[*]}
  method    from-scratch RandomizedSearchCV over the E48-prior grid on
            ${RFK_TARGET} (embargo auto=60), ${RFK_N_ITER} draws, ${RFK_N_SPLITS}-fold CV
  results   results/metrics/crypto/rf_ktz/${PREFIX}_*.json
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*" 2>/dev/null \
        && echo "   emailed the start notice" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: rf_ktz 60-day tune (top-5 MMC datasets)  |  ${total} arm(s): ${SEL_TAG[*]}"
echo "target=${RFK_TARGET} embargo=${RFK_EMBARGO} | python=${PYTHON} | n_iter=${RFK_N_ITER} pool=${RFK_POOL}"
echo "=============================================================="

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
         --search-jobs 1 --n-jobs "${RFK_JOBS}" --target "${RFK_TARGET}"
         --data-version "${dsver}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-14s %s (%s)\n' "$((i+1))" "$total" "$name" "$dsver" "$RFK_TARGET"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${RFK_POOL}" ]; do sleep 5; done
    echo "[launch $((i+1))/${total}] ${name}  ${dsver}  $(date '+%H:%M:%S')"
    ( t0=$(date +%s); "${cmd[@]}" > "logs/${name}.log" 2>&1
      echo ">>> ${name} exit=$? in $(( $(date +%s)-t0 ))s" >> "logs/${PREFIX}_pool.log" ) &
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
wait

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- rf_ktz 60-day tune, top-5 MMC datasets (matched to E52 on the label)"
el_all=$(( $(date +%s)-started_all ))
echo "wall $((el_all/60))m$((el_all%60))s over ${total} dataset(s)"
echo
"$PYTHON" - "${PREFIX}" "${SEL_TAG[@]}" <<'PY'
import json, os, sys
prefix = sys.argv[1]; tags = sys.argv[2:]
# E52 t20 MMC for the reference column
e52 = {}
for tag in tags:
    p = "results/metrics/crypto/rf_ktz/e52_%s.json" % tag
    if os.path.exists(p):
        e52[tag] = (json.load(open(p))["mmc"]["mean"])
rows = []
for tag in tags:
    p = "results/metrics/crypto/rf_ktz/%s_%s.json" % (prefix, tag)
    if not os.path.exists(p):
        print("  %-14s MISSING" % tag); continue
    d = json.load(open(p)); c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    rows.append((mm, tag, cm, 0.1*cm+mm, c.get("sharpe") or 0.0, d.get("n_periods")))
rows.sort(reverse=True)
print("  %-14s %8s %8s %8s %7s %5s | %10s" % ("dataset(t60)","CORR","MMC","payout","c.shrp","n","E52 MMC(t20)"))
for mm, tag, cm, pay, sh, n in rows:
    ref = ("%.5f" % e52[tag]) if tag in e52 else "-"
    print("  %-14s %8.5f %8.5f %8.5f %7.3f %5s | %10s" % (tag, cm, mm, pay, sh, n, ref))
print()
print("  60-day book to beat: SACI_03 payout 0.09302 (MMC 0.07875) | CURUPIRA_05_60 payout 0.09220 (MMC 0.08035), both n=470")
PY
echo
echo "NEXT -- paired vs the 60-day book:"
echo "  ${PYTHON} -m src.analysis.results_table --model rf_ktz --runs $(printf '%s ' "${SEL_TAG[@]/#/${PREFIX}_}") --sort-by payout"
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished" % sys.argv[2], open(sys.argv[1]).read())
print("emailed the report")
PY
fi
