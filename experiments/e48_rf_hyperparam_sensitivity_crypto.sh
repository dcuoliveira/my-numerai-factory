#!/usr/bin/env bash
#
# Experiment E48: hyperparameter sensitivity of the RANDOM FOREST, after Kalfa,
# Timmermann & van der Zwan, "Overhyped? Can ML Models Reliably Predict Stock
# Returns?" (2025), papers/Hyped_ML_paper-21.pdf, Section 2.5 and Table 2 -- the
# framework E17 ran on the MLP and E27/E29 on LightGBM, now on a bagged forest.
#
# E48 is the reference handle; every artefact it writes is e48_*.
#
# THE PAPER'S METHOD, IN ONE SENTENCE
#   To find which hyperparameters matter, pin each cell of a full factorial as
#   its own walk-forward, then study the DISTRIBUTION of out-of-sample
#   performance conditional on each value: a value that barely moves the
#   distribution is not worth tuning; one whose WORST case is still good is a
#   "corner" that fixes performance regardless of the rest. The reading --
#   marginals, functional-ANOVA importance, corners, priors -- is
#   src/analysis/hyperparam_sensitivity_rf.py, invoked at the end.
#
# WHY A FOREST NEEDS ITS OWN SWEEP
#   A random forest is bagging, not boosting: trees are fit independently and
#   AVERAGED, so there is no learning rate and no n_estimators/learning_rate
#   coupling. E17 (MLP) and E27 (LightGBM) therefore do not transfer -- the axes
#   are different. This maps the forest's own surface once, so its grid can be
#   set from evidence rather than the sklearn defaults.
#
# THE FACTORIAL  (the three Numerai idioms mapped onto a forest, plus its two
#   native complexity controls)
#   n_estimators      3   100 300 1000        variance knob (no lr to pair it to)
#   max_depth         4   3 6 12 None         complexity; None = grow out fully
#   max_features      4   0.1 0.3 0.6 1.0     THE colsample idiom -- low starves
#                                             each split of columns for diversity
#   min_samples_leaf  4   1 10 100 1000       heavy-leaf idiom (= min_data_in_leaf)
#                = 3 x 4 x 4 x 4 = 192 cells, each a pinned walk-forward on the
#                  raw 22 features of crypto/v2.0.
#
#   THREE FIXED SETTINGS, stated so the sweep is not mistaken for full coverage:
#   * NO learning-rate axis. A forest has none; n_estimators + max_depth stand
#     in for the boosters' lr + num_leaves. This is the one structural departure
#     from the E17/E27 sweeps and it is not optional.
#   * bootstrap=True, max_samples=None (each tree sees a full-size bootstrap).
#     Row subsampling is the least-evidenced idiom; a fifth axis would double the
#     cost for it, so it is pinned at the textbook forest default rather than
#     swept. A follow-up can open it.
#   * criterion is left at the sklearn default (squared error in every version;
#     the label changed at 1.0, so it is not named explicitly).
#
#   n_splits is 2, not 5: the cells are PINNED (--fixed-params), so the inner CV
#   only records a cv_score the sensitivity analysis never reads -- the OOS
#   payout it DOES read comes from the walk-forward and is identical at any
#   n_splits. 2 is the floor (1 yields 0 folds).
#
# THE METRIC  this repo's OOS payout (0.1*corr + mmc), the mean over the
#   walk-forward periods -- one number per cell, plus the per-period series for
#   the half-sample robustness split. Same axis as the whole board. All 192
#   cells share one window (same data, cadence, seed), so they are directly
#   comparable and no window caveat applies within the grid.
#
# COST  the per-period BMC scoring is a ~2-3 min floor per cell on the current
#   213-period dataset (one refit, 64 scored periods), and --era-step barely
#   reduces it here, so a scout saves little. The expensive corner (1000
#   full-depth trees, max_features 1.0) adds the forest fit on top. Budget
#   ~192 x 2-5 min ~= 8-16 h sequential; SHARD=/OF= round-robins across machines.
#   The box is often already loaded -- keep RF_JOBS modest.
#
# Usage:
#   RF_SCOUT=1 bash experiments/e48_rf_hyperparam_sensitivity_crypto.sh   # wiring only
#   SHARD=1 OF=8 bash experiments/e48_rf_hyperparam_sensitivity_crypto.sh
#   ONLY_NE=1000 / ONLY_DEPTH=None / ONLY_MF=0.1 / ONLY_ML=1000 / DRY_RUN=1 / NO_MAIL=1
#   PYTHON=/path/to/venv/bin/python bash experiments/e48_rf_hyperparam_sensitivity_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E48}"
RF_WINDOW="${RF_WINDOW:-130}"
RF_REFIT="${RF_REFIT:-130}"
RF_EMBARGO="${RF_EMBARGO:-20}"
RF_N_SPLITS="${RF_N_SPLITS:-2}"
RF_VAL_FRAC="${RF_VAL_FRAC:-0.3}"
RF_SEED="${RF_SEED:-42}"
RF_DATA="${RF_DATA:-crypto/v2.0}"
RF_JOBS="${RF_JOBS:-4}"
RF_THREADS="${RF_THREADS:-4}"
RF_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
# GNU screen exports WINDOW=0 into every session; a bare WINDOW default would
# silently zero the training block. Guard the two that come from it.
case "${RF_WINDOW}" in ""|0|[0-9]) echo "RF_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${RF_REFIT}"  in ""|0|[0-9]) echo "RF_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${RF_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "RF_N_SPLITS must be >= 2 (1 yields no usable CV folds)" >&2; exit 1; }

PREFIX="e48"
if [ "${RF_SCOUT:-0}" = "1" ]; then
    PREFIX="e48scout"; RF_ERA_STEP=4
    echo "SCOUT MODE: era-step 4 -- wiring only; on this dataset it saves little, arms compare to nothing."
fi

# A forest parallelises over TREES via n_jobs (joblib), not BLAS, so RF_JOBS is
# the real knob; the OMP vars only bound any stray BLAS in scoring.
export OMP_NUM_THREADS="${RF_THREADS}"
export MKL_NUM_THREADS="${RF_THREADS}"
export OPENBLAS_NUM_THREADS="${RF_THREADS}"

# The factorial. Scalar tokens, no '|' -- plain space-lists. max_depth carries
# the literal token "None" for grow-out-fully; it becomes JSON null below.
NES="100 300 1000"
DEPTHS="3 6 12 None"
MFS="0.1 0.3 0.6 1.0"
MLS="1 10 100 1000"

# Preflight: the raw-22 dataset is present with its meta file, and the rf wrapper
# constructs and pins all four axes -- including the two easy-to-break values,
# max_depth=None (JSON null) and max_features=1.0.
"$PYTHON" - "${RF_DATA}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.base import clone
from src.prediction.models import build_wrapper

version = sys.argv[1]; bad = []
path = "data/%s/train.parquet" % version
try:
    n = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if n != 22:
        bad.append("%s has %d features, expected the raw 22" % (version, n))
except OSError:
    bad.append("%s MISSING" % version)
if not os.path.exists("data/%s/historical_meta_models.parquet" % version):
    bad.append("%s lacks historical_meta_models.parquet -- the loader would try "
               "to download it and fail after the run starts" % version)
try:
    e = clone(build_wrapper("rf", "regression", seed=42, n_jobs=1).ModelClass)
    e.set_params(n_estimators=1000, max_depth=None, max_features=1.0, min_samples_leaf=1)
    e.set_params(n_estimators=100, max_depth=3, max_features=0.1, min_samples_leaf=1000)
except Exception as error:
    bad.append("pinning the four axes (incl. max_depth=None / max_features=1.0) failed: %s" % error)
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s has the raw 22 and its meta file; all four axes pin "
      "(incl. max_depth=None and max_features=1.0)" % version)
PY

# Build the work list.
CFG_NAME=(); CFG_PARAMS=()
for ne in ${NES}; do
    [ -n "${ONLY_NE:-}" ] && [ "${ONLY_NE}" != "$ne" ] && continue
    for d in ${DEPTHS}; do
        [ -n "${ONLY_DEPTH:-}" ] && [ "${ONLY_DEPTH}" != "$d" ] && continue
        if [ "$d" = "None" ]; then djson=null; dtag=dnone; else djson="$d"; dtag="d${d}"; fi
        for mf in ${MFS}; do
            [ -n "${ONLY_MF:-}" ] && [ "${ONLY_MF}" != "$mf" ] && continue
            # DISTINCT tags per value, so no two cells collide on a name and
            # clobber each other's output (the bug the E17 header warns about).
            case "$mf" in 0.1) mftag=mf01;; 0.3) mftag=mf03;; 0.6) mftag=mf06;; 1.0) mftag=mf10;; *) echo "unmapped max_features $mf" >&2; exit 1;; esac
            for ml in ${MLS}; do
                [ -n "${ONLY_ML:-}" ] && [ "${ONLY_ML}" != "$ml" ] && continue
                name="${PREFIX}_ne${ne}_${dtag}_${mftag}_ml${ml}"
                params="{\"n_estimators\":${ne},\"max_depth\":${djson},\"max_features\":${mf},\"min_samples_leaf\":${ml}}"
                CFG_NAME+=("$name"); CFG_PARAMS+=("$params")
            done
        done
    done
done

# Shard round-robin.
SEL_NAME=(); SEL_PARAMS=(); picked=0
for i in "${!CFG_NAME[@]}"; do
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${CFG_NAME[$i]}"); SEL_PARAMS+=("${CFG_PARAMS[$i]}")
done
total="${#SEL_NAME[@]}"
[ "$total" -eq 0 ] && { echo "no configs selected" >&2; exit 1; }

# Unique run-name gate: a collision means two cells write the same file and one
# clobbers the other. Fail loudly here, not after hours of compute.
dupes="$(printf '%s\n' "${SEL_NAME[@]}" | sort | uniq -d)"
if [ -n "$dupes" ]; then
    echo "FATAL: run-name collision -- these names map >1 config:" >&2
    printf '  %s\n' $dupes >&2; exit 1
fi

index=0; ran=0; failed=0; skipped=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  cells     ${total}  (of 192; shard ${SHARD}/${OF})
  method    Kalfa-Timmermann-van der Zwan sensitivity on the RANDOM FOREST: one
            pinned walk-forward per cell of a 3x4x4x4 factorial (n_estimators x
            max_depth x max_features x min_samples_leaf), then marginals,
            functional-ANOVA importance, corners and priors
  data      ${RF_DATA} (raw 22)   cadence rolling ${RF_WINDOW}/${RF_REFIT}, embargo ${RF_EMBARGO}
  expect    ~2-5 min per cell; the per-period scoring is the floor

Where to look:
  runner    experiments/e48_rf_hyperparam_sensitivity_crypto.sh
  analysis  src/analysis/hyperparam_sensitivity_rf.py
  results   results/metrics/crypto/rf/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: random-forest hyperparameter sensitivity  |  ${total} cells  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}  |  n_jobs=${RF_JOBS}"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    name="${SEL_NAME[$i]}"; params="${SEL_PARAMS[$i]}"; index=$((index+1))
    cmd=("$PYTHON" -u -m src.run_forecast crypto rf
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${RF_WINDOW}" --refit-every "${RF_REFIT}"
         --embargo "${RF_EMBARGO}" --n-splits "${RF_N_SPLITS}" --val-frac "${RF_VAL_FRAC}"
         --era-step "${RF_ERA_STEP}" --seed "${RF_SEED}" --search-jobs 1 --n-jobs "${RF_JOBS}"
         --fixed-params "${params}" --data-version "${RF_DATA}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%3d/%3d] %-32s %s\n' "$index" "$total" "$name" "$params"; continue; fi
    # Resume: a cell already on disk with a valid metrics json is skipped, so a
    # restart or a re-shard never recomputes finished work.
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/rf/${name}.json" ]; then
        skipped=$((skipped+1)); continue
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${name}")
        echo ">>> ${name} FAILED (exit ${status}) after ${el}s -- continuing (see logs/${name}.log)"
    else
        echo ">>> ${name} done in ${el}s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- random-forest hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} cells in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed, ${skipped} already-done skipped"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL. Run the analysis only once ALL shards are done."
echo
# The full framework analysis: marginals, importance (fANOVA), corners, priors.
# Skipped for a scout or a partial shard -- it needs the whole 192-cell grid.
if [ "${RF_SCOUT:-0}" != "1" ] && [ "${OF}" -le 1 ]; then
    "$PYTHON" -m src.analysis.hyperparam_sensitivity_rf --prefix "${PREFIX}" || \
        echo "analysis failed -- run: $PYTHON -m src.analysis.hyperparam_sensitivity_rf --prefix ${PREFIX}"
else
    echo "partial/scout run: once every shard's cells are on disk, read the"
    echo "framework analysis with"
    echo "    $PYTHON -m src.analysis.hyperparam_sensitivity_rf --prefix e48"
fi
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
