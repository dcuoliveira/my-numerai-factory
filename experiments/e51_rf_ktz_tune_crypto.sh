#!/usr/bin/env bash
#
# Experiment E51: an RF regression model TUNED FROM SCRATCH under the E48 priors.
# The rf sibling of E28 (mlp_ktz) and E50 (lgb_ktz): take the Kalfa-Timmermann-
# van der Zwan sensitivity findings from E48 (experiments/e48_rf_..._crypto.sh),
# compiled into the rf_ktz search grid, and run ONE from-scratch RandomizedSearchCV
# per dataset -- a real search that SELECTS a config per refit, not the pinned
# cells E48 swept.
#
# E51 is the reference handle; every artefact it writes is e51_*.
#
# WHY THIS, AFTER E48
#   E48 mapped the whole 192-cell surface and read off where the good forests
#   live. This spends that knowledge: instead of re-searching a naive grid, the
#   search only ever draws from E48's GOOD REGION, so every candidate is already
#   a sensible forest and the budget fine-tunes rather than explores.
#
# THE GRID  (src.prediction.models.RFKtzWrapper -- see its docstring)
#   n_estimators      [100]                 PINNED. E48 fANOVA share 0.0002 -- a
#                                           forest averages out variance by ~100
#                                           trees, so tuning it wastes budget.
#   max_depth         {3, 4, 6}             shallow gate (E48: depth 3 best,
#                                           monotone worse to None).
#   max_features      {0.05,0.1,0.15,0.2,0.3}  dense-low colsample idiom (E48:
#                                           0.1 best, 1.0 worst, 0/48 good-cell
#                                           support at 1.0 -- dropped).
#   min_samples_leaf  {300,1000,2000,4000}  heavy, and the ceiling OPENED past
#                                           E48's max of 1000: its marginal rose
#                                           monotonically to the grid EDGE, the
#                                           E07/E17 alpha-ceiling trap, so E51
#                                           tests beyond it (4000 ~ 10% of a 39k
#                                           block, still splittable at these depths).
#   = 60 cells; searched at --n-iter 40 (2/3 coverage of an all-good grid).
#
# THE CADENCE  the deployable-slot one: rolling 130/130, embargo 20, n_splits 5
#   (a REAL inner CV that selects, unlike E48's pinned n_splits 2), seed 42,
#   val_frac 0.3. Scored on the payable target_binned_return_20, metric corr,
#   MMC over the real meta model.
#
# TWO ARMS  raw 22 (crypto/v2.0, where the priors were derived) and fenccausal
#   (crypto/v2.0-fenccausal, 44 -- the feature set the book's best neural slot
#   uses, so the interesting deployment candidate). ONLY_DS picks one.
#
# WHAT COUNTS  a deployable RF plus its OOS payout/CORR. The honest test of "does
#   RF add to the book" is a PAIRED comparison, printed at the end -- not run here.
#
# COST  a 40-draw x 5-fold search of shallow 100-tree forests is cheap; on an idle
#   box ~20-40 min per arm. On a loaded box, far more -- run it where there is CPU.
#
# Usage:
#   RFK_SCOUT=1 bash experiments/e51_rf_ktz_tune_crypto.sh          # wiring only
#   ONLY_DS=raw / ONLY_DS=fenc / DRY_RUN=1 / NO_MAIL=1
#   PYTHON=/path/to/venv/bin/python bash experiments/e51_rf_ktz_tune_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E51}"
RFK_WINDOW="${RFK_WINDOW:-130}"
RFK_REFIT="${RFK_REFIT:-130}"
RFK_EMBARGO="${RFK_EMBARGO:-20}"
RFK_N_SPLITS="${RFK_N_SPLITS:-5}"
RFK_N_ITER="${RFK_N_ITER:-40}"
RFK_VAL_FRAC="${RFK_VAL_FRAC:-0.3}"
RFK_SEED="${RFK_SEED:-42}"
RFK_JOBS="${RFK_JOBS:-2}"
RFK_SEARCH_JOBS="${RFK_SEARCH_JOBS:-8}"
RFK_THREADS="${RFK_THREADS:-2}"
RFK_ERA_STEP=1
case "${RFK_WINDOW}" in ""|0|[0-9]) echo "RFK_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${RFK_REFIT}"  in ""|0|[0-9]) echo "RFK_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${RFK_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "RFK_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e51"
if [ "${RFK_SCOUT:-0}" = "1" ]; then
    PREFIX="e51scout"; RFK_ERA_STEP=4; RFK_N_ITER=6; RFK_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${RFK_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${RFK_THREADS}"
export MKL_NUM_THREADS="${RFK_THREADS}"
export OPENBLAS_NUM_THREADS="${RFK_THREADS}"

# Datasets: parallel arrays (tag, version, expected feature count).
DSTAG=(raw fenc)
DSVER=("crypto/v2.0" "crypto/v2.0-fenccausal")
DSFEAT=(22 44)

# Preflight: datasets present with meta files + expected feature counts, and the
# rf_ktz grid actually encodes the E48 priors (n_estimators pinned at 100, the
# colsample idiom capped low, the leaf ceiling opened past 1000).
"$PYTHON" - "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import os, sys
import numpy as np
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper

cut = sys.argv.index("--"); versions = sys.argv[1:cut]; wants = [int(x) for x in sys.argv[cut+1:]]
bad = []
for v, want in zip(versions, wants):
    path = "data/%s/train.parquet" % v
    try:
        n = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
        if n != want:
            bad.append("%s has %d features, expected %d" % (v, n, want))
    except OSError:
        bad.append("%s MISSING" % v)
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)
g = build_wrapper("rf_ktz", "regression", seed=42, n_jobs=1).param_grid
if g.get("n_estimators") != [100]:
    bad.append("rf_ktz n_estimators is not pinned at 100 (got %s)" % g.get("n_estimators"))
if max(g["max_features"]) > 0.3:
    bad.append("rf_ktz max_features escapes the E48 low region (max %g)" % max(g["max_features"]))
if max(g["min_samples_leaf"]) <= 1000:
    bad.append("rf_ktz leaf ceiling not opened past E48's binding 1000 (max %d)" % max(g["min_samples_leaf"]))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files; rf_ktz grid encodes "
      "the E48 priors (n_est pinned 100, max_features<=0.3, leaf opened to %d)"
      % (len(versions), max(g["min_samples_leaf"])))
PY

# Build the work list.
SEL_TAG=(); SEL_VER=()
for d in "${!DSTAG[@]}"; do
    tag="${DSTAG[$d]}"
    [ -n "${ONLY_DS:-}" ] && [ "${ONLY_DS}" != "$tag" ] && continue
    SEL_TAG+=("$tag"); SEL_VER+=("${DSVER[$d]}")
done
total="${#SEL_TAG[@]}"
[ "$total" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (rf_ktz from-scratch tune: ${SEL_TAG[*]})
  method    RandomizedSearchCV over the E48-prior grid (rf_ktz), ${RFK_N_ITER} draws,
            ${RFK_N_SPLITS}-fold inner CV -- a real search that SELECTS per refit
  grid      n_est pinned 100 | depth {3,4,6} | max_features {.05..0.3} |
            min_samples_leaf {300,1000,2000,4000} (E48 ceiling opened)
  cadence   rolling ${RFK_WINDOW}/${RFK_REFIT}, embargo ${RFK_EMBARGO}, seed ${RFK_SEED}
  results   results/metrics/crypto/rf_ktz/${PREFIX}_*.json
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: rf_ktz from-scratch tune  |  ${total} arm(s): ${SEL_TAG[*]}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}  |  n_iter=${RFK_N_ITER} search-jobs=${RFK_SEARCH_JOBS} n-jobs=${RFK_JOBS}"
echo "=============================================================="

for i in "${!SEL_TAG[@]}"; do
    tag="${SEL_TAG[$i]}"; dsver="${SEL_VER[$i]}"; name="${PREFIX}_${tag}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto rf_ktz
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${RFK_WINDOW}" --refit-every "${RFK_REFIT}"
         --embargo "${RFK_EMBARGO}" --n-splits "${RFK_N_SPLITS}" --n-iter "${RFK_N_ITER}"
         --val-frac "${RFK_VAL_FRAC}" --era-step "${RFK_ERA_STEP}" --seed "${RFK_SEED}"
         --search-jobs "${RFK_SEARCH_JOBS}" --n-jobs "${RFK_JOBS}"
         --data-version "${dsver}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-10s %s\n' "$((i+1))" "$total" "$name" "$dsver"; printf '       %s\n' "${cmd[*]}"; continue; fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/rf_ktz/${name}.json" ]; then
        echo "[$((i+1))/$total] ${name} already on disk -- skipping"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${name}  ${dsver}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${name}")
        echo ">>> ${name} FAILED (exit ${status}) after ${el}s -- continuing (logs/${name}.log)"
    else
        echo ">>> ${name} done in ${el}s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- rf_ktz from-scratch tune (E48 priors)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/60))m$((el_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo
# Print each arm's OOS metrics + the config the search settled on per refit.
for tag in "${SEL_TAG[@]}"; do
    name="${PREFIX}_${tag}"
    echo "=== ${name} ==="
    "$PYTHON" - "$name" <<'PY' || echo "  (metrics not readable)"
import json, sys, os
run = sys.argv[1]
p = "results/metrics/crypto/rf_ktz/%s.json" % run
if not os.path.exists(p):
    print("  no metrics json"); raise SystemExit(0)
d = json.load(open(p))
# summaries are stored nested: d["corr"] = {mean,std,sharpe,max_drawdown}, etc.
corr = d.get("corr") or {}; mmc = d.get("mmc") or {}
cm = corr.get("mean") or 0.0; mm = mmc.get("mean") or 0.0
print("  CORR %.5f  MMC %.5f  payout(0.1c+mmc, summary) %.5f  c.sharpe %.3f  c.max_dd %.4f  n=%s"
      % (cm, mm, 0.1*cm + mm, corr.get("sharpe") or 0.0,
         corr.get("max_drawdown") or 0.0, d.get("n_periods")))
hp = "results/hyperparams/crypto/rf_ktz/%s.csv" % run
if os.path.exists(hp):
    import csv
    rows = list(csv.DictReader(open(hp)))
    print("  per-refit picks (%d refits):" % len(rows))
    for r in rows:
        cols = {k: r[k] for k in r if k in ("max_depth","max_features","min_samples_leaf","n_estimators")}
        print("    " + ", ".join("%s=%s" % (k, cols[k]) for k in sorted(cols)))
PY
done
echo
echo "NEXT -- the honest test is a PAIRED comparison vs the deployed book:"
echo "  $PYTHON -m src.analysis.report_html --model rf_ktz --runs ${PREFIX}_raw ${PREFIX}_fenc"
echo "  (table only) $PYTHON -m src.analysis.results_table --model rf_ktz --runs ${PREFIX}_raw ${PREFIX}_fenc --sort-by payout"
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  "$PYTHON" - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
