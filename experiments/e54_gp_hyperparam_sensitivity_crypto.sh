#!/usr/bin/env bash
#
# Experiment E54: hyperparameter sensitivity of the GAUSSIAN PROCESS, after
# Kalfa, Timmermann & van der Zwan, "Overhyped? Can ML Models Reliably Predict
# Stock Returns?" (2025), papers/Hyped_ML_paper-21.pdf, Section 2.5 and Table 2 --
# the framework E17 ran on the MLP, E27/E29 on LightGBM and E48 on the random
# forest, now on the board's first KERNEL model.
#
# E54 is the reference handle; every artefact it writes is e54_*.
#
# THE PAPER'S METHOD, IN ONE SENTENCE
#   To find which hyperparameters matter, pin each cell of a full factorial as
#   its own walk-forward, then study the DISTRIBUTION of out-of-sample
#   performance conditional on each value: a value that barely moves the
#   distribution is not worth tuning; one whose WORST case is still good is a
#   "corner" that fixes performance regardless of the rest. The reading --
#   marginals, functional-ANOVA importance, corners, priors -- is
#   src/analysis/hyperparam_sensitivity_gp.py, invoked at the end.
#
# WHY A GP NEEDS ITS OWN SWEEP
#   A GP has no trees, no learning rate and no architecture -- its behaviour is
#   set by the KERNEL. E17 (MLP), E27 (LightGBM) and E48 (forest) do not transfer;
#   the axes are the kernel family, the length-scale, ARD, and the noise floor.
#   This maps the GP's own surface once so its grid -- and its kernel prior -- can
#   be set from evidence rather than sklearn defaults.
#
# THE FACTORIAL
#   kernel        4   matern12 matern32 matern52 rbf   smoothness (rough -> smooth)
#   ard           2   iso ard                          shared vs per-feature length-scale
#   length_scale  3   0.25 1.0 4.0                     on [0,1]-scaled features
#   alpha         4   1e-3 1e-2 1e-1 1.0               observation-noise floor
#                = 4 x 2 x 3 x 4 = 96 cells, each a pinned walk-forward on the
#                  raw 22 features of crypto/v2.0.
#
#   FIXED SETTINGS, stated so the sweep is not mistaken for full coverage:
#   * The kernel is FROZEN (length_scale_bounds="fixed" -> optimizer=None). A
#     pinned cell must measure the pinned length-scale, NOT the marginal-
#     likelihood basin the optimiser would fall into, or the fANOVA is meaningless.
#     gp_ktz (E55) may re-optimise WITHIN the E54-derived bounds; E54 must freeze.
#   * subsample=6000 and normalize_y=true are pinned for every cell. subsample is
#     a real fidelity/cost trade, so holding it constant keeps "kernel shape
#     matters" from being confounded with "I subsampled differently"; a separate
#     ladder (GP_SUB_LADDER=1) measures its cost and is kept OUT of the fANOVA.
#   * REGRESSION only. GaussianProcessClassifier has no alpha (Laplace, no noise
#     diagonal), so the alpha axis is undefined for it; the clf side is carried by
#     E53 and E55.
#
#   n_splits is 2, not 5: the cells are PINNED (--fixed-params), so the inner CV
#   only records a cv_score the sensitivity analysis never reads -- the OOS payout
#   it DOES read comes from the walk-forward and is identical at any n_splits.
#
# THE METRIC  this repo's OOS payout (0.1*corr + mmc), the mean over the
#   walk-forward periods. All 96 cells share one window (same data, cadence,
#   seed), so they are directly comparable and no window caveat applies.
#
# COST  each cell is one frozen-kernel Cholesky at subsample 6000 plus the
#   per-period scoring floor -- seconds to low minutes. Budget ~96 x 1-5 min;
#   SHARD=/OF= round-robins across machines. The ard cells (44 length-scales) and
#   the alpha=1e-3 cells (near-singular kernel) are the slow/fragile pole -- the
#   estimator falls back to a constant there rather than crashing, and the
#   analysis flags those cells.
#
# Usage:
#   GP_SCOUT=1 bash experiments/e54_gp_hyperparam_sensitivity_crypto.sh   # wiring only
#   ONLY_KERNEL=matern32 ONLY_ARD=iso  (narrow a wiring check to 12 cells)
#   SHARD=1 OF=8 bash experiments/e54_gp_hyperparam_sensitivity_crypto.sh
#   ONLY_LS=1.0 / ONLY_AL=0.01 / DRY_RUN=1 / NO_MAIL=1
#   PYTHON=/path/to/venv/bin/python bash experiments/e54_gp_hyperparam_sensitivity_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E54}"
GP_WINDOW="${GP_WINDOW:-130}"
GP_REFIT="${GP_REFIT:-130}"
GP_EMBARGO="${GP_EMBARGO:-20}"
GP_N_SPLITS="${GP_N_SPLITS:-2}"
GP_VAL_FRAC="${GP_VAL_FRAC:-0.3}"
GP_SEED="${GP_SEED:-42}"
GP_DATA="${GP_DATA:-crypto/v2.0}"
GP_JOBS="${GP_JOBS:-1}"             # gp drops n_jobs; BLAS threads are the knob
GP_THREADS="${GP_THREADS:-4}"
GP_SUBSAMPLE="${GP_SUBSAMPLE:-6000}"
GP_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${GP_WINDOW}" in ""|0|[0-9]) echo "GP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${GP_REFIT}"  in ""|0|[0-9]) echo "GP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${GP_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "GP_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e54"
if [ "${GP_SCOUT:-0}" = "1" ]; then
    PREFIX="e54scout"; GP_ERA_STEP=4; GP_SUBSAMPLE=1500
    echo "SCOUT MODE: era-step 4, subsample ${GP_SUBSAMPLE} -- wiring only; narrow with ONLY_KERNEL=/ONLY_ARD=."
fi

export OMP_NUM_THREADS="${GP_THREADS}"
export MKL_NUM_THREADS="${GP_THREADS}"
export OPENBLAS_NUM_THREADS="${GP_THREADS}"

# The factorial. Plain space-lists; ard carries "iso"/"ard" (-> JSON bool below).
KERNELS="matern12 matern32 matern52 rbf"
ARDS="iso ard"
LENGTHSCALES="0.25 1.0 4.0"
ALPHAS="0.001 0.01 0.1 1.0"

# Preflight: raw-22 present with its meta file, and the gp wrapper pins a full
# cell -- including the two easy-to-break values, ard=true (JSON) and a frozen
# kernel via length_scale_bounds="fixed".
"$PYTHON" - "${GP_DATA}" "${GP_SUBSAMPLE}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.base import clone
from src.prediction.models import build_wrapper

version = sys.argv[1]; subsample = int(sys.argv[2]); bad = []
path = "data/%s/train.parquet" % version
try:
    n = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if n != 22:
        bad.append("%s has %d features, expected the raw 22" % (version, n))
except OSError:
    bad.append("%s MISSING" % version)
if not os.path.exists("data/%s/historical_meta_models.parquet" % version):
    bad.append("%s lacks historical_meta_models.parquet" % version)
try:
    e = clone(build_wrapper("gp", "regression", seed=42, n_jobs=1,
                            subsample=subsample).ModelClass)
    e.set_params(kernel="rbf", ard=True, length_scale=4.0, alpha=1e-3,
                 normalize_y=True, length_scale_bounds="fixed")
    e.set_params(kernel="matern12", ard=False, length_scale=0.25, alpha=1.0)
except Exception as error:
    bad.append("pinning a gp cell (incl. ard=True, frozen kernel) failed: %s" % error)
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s has the raw 22 and its meta file; a full gp cell pins "
      "(incl. ard=True and a frozen kernel)" % version)
PY

# Build the work list.
CFG_NAME=(); CFG_PARAMS=()
for k in ${KERNELS}; do
    [ -n "${ONLY_KERNEL:-}" ] && [ "${ONLY_KERNEL}" != "$k" ] && continue
    for a in ${ARDS}; do
        [ -n "${ONLY_ARD:-}" ] && [ "${ONLY_ARD}" != "$a" ] && continue
        [ "$a" = "ard" ] && ardjson=true || ardjson=false
        for ls in ${LENGTHSCALES}; do
            [ -n "${ONLY_LS:-}" ] && [ "${ONLY_LS}" != "$ls" ] && continue
            # DISTINCT tags per value so no two cells collide on a name.
            case "$ls" in 0.25) lstag=ls025;; 1.0) lstag=ls1;; 4.0) lstag=ls4;; *) echo "unmapped length_scale $ls" >&2; exit 1;; esac
            for al in ${ALPHAS}; do
                [ -n "${ONLY_AL:-}" ] && [ "${ONLY_AL}" != "$al" ] && continue
                case "$al" in 0.001) altag=al0001;; 0.01) altag=al001;; 0.1) altag=al01;; 1.0) altag=al1;; *) echo "unmapped alpha $al" >&2; exit 1;; esac
                name="${PREFIX}_${k}_${a}_${lstag}_${altag}"
                params="{\"kernel\":\"${k}\",\"ard\":${ardjson},\"length_scale\":${ls},\"alpha\":${al},\"normalize_y\":true,\"length_scale_bounds\":\"fixed\"}"
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

# Unique run-name gate: a collision means two cells write the same file.
dupes="$(printf '%s\n' "${SEL_NAME[@]}" | sort | uniq -d)"
if [ -n "$dupes" ]; then
    echo "FATAL: run-name collision -- these names map >1 config:" >&2
    printf '  %s\n' $dupes >&2; exit 1
fi

index=0; ran=0; failed=0; skipped=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%F %T') on $(hostname)

  cells     ${total}  (of 96; shard ${SHARD}/${OF})
  method    Kalfa-Timmermann-van der Zwan sensitivity on the GAUSSIAN PROCESS: one
            pinned walk-forward per cell of a 4x2x3x4 factorial (kernel x ard x
            length_scale x alpha), kernel FROZEN, then marginals, fANOVA, corners
            and priors
  data      ${GP_DATA} (raw 22)   cadence rolling ${GP_WINDOW}/${GP_REFIT}, embargo ${GP_EMBARGO}
            subsample ${GP_SUBSAMPLE}, normalize_y=true (both pinned)

Where to look:
  runner    experiments/e54_gp_hyperparam_sensitivity_crypto.sh
  analysis  src/analysis/hyperparam_sensitivity_gp.py
  results   results/metrics/crypto/gp/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: gaussian-process hyperparameter sensitivity  |  ${total} cells  |  shard ${SHARD}/${OF}"
echo "started $(date '+%F %T')  |  python=${PYTHON}  |  subsample=${GP_SUBSAMPLE}"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    name="${SEL_NAME[$i]}"; params="${SEL_PARAMS[$i]}"; index=$((index+1))
    cmd=("$PYTHON" -u -m src.run_forecast crypto gp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${GP_WINDOW}" --refit-every "${GP_REFIT}"
         --embargo "${GP_EMBARGO}" --n-splits "${GP_N_SPLITS}" --val-frac "${GP_VAL_FRAC}"
         --era-step "${GP_ERA_STEP}" --seed "${GP_SEED}" --search-jobs 1 --n-jobs "${GP_JOBS}"
         --model-options "{\"subsample\": ${GP_SUBSAMPLE}}"
         --fixed-params "${params}" --data-version "${GP_DATA}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%3d/%3d] %-40s %s\n' "$index" "$total" "$name" "$params"; continue; fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/gp/${name}.json" ]; then
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
echo "${EXPERIMENT} -- gaussian-process hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} cells in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed, ${skipped} already-done skipped"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL. Run the analysis only once ALL shards are done."
echo
# The full framework analysis: marginals, importance (fANOVA), corners, priors.
# Skipped for a scout or a partial shard -- it needs the whole 96-cell grid.
if [ "${GP_SCOUT:-0}" != "1" ] && [ "${OF}" -le 1 ]; then
    "$PYTHON" -m src.analysis.hyperparam_sensitivity_gp --prefix "${PREFIX}" || \
        echo "analysis failed -- run: $PYTHON -m src.analysis.hyperparam_sensitivity_gp --prefix ${PREFIX}"
else
    echo "partial/scout run: once every shard's cells are on disk, read the"
    echo "framework analysis with"
    echo "    $PYTHON -m src.analysis.hyperparam_sensitivity_gp --prefix e54"
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
