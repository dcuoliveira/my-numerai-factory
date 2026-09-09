#!/usr/bin/env bash
#
# Experiment E17: hyperparameter sensitivity of the MLP, after Kalfa,
# Timmermann & van der Zwan, "Overhyped? Can ML Models Reliably Predict Stock
# Returns?" (2025), papers/Hyped_ML_paper-21.pdf, Section 2.5 and Table 2.
#
# E17 is the reference handle; every artefact it writes is e17_*.
#
# THE PAPER'S METHOD, IN ONE SENTENCE
#   To find which hyperparameters matter, fix ONE at a value and evaluate the
#   FULL grid of all the others, then study the DISTRIBUTION of out-of-sample
#   performance conditional on that fixed value (paper eq. around p.11). A
#   hyperparameter whose fixed value barely moves the distribution is not worth
#   tuning; one that moves it a lot is. A fixed value whose WORST case is still
#   good is a "corner" that guarantees good performance regardless of the rest;
#   one whose BEST case is still bad locks in failure.
#
#   That analysis needs the full factorial on disk. This runner produces it: one
#   pinned walk-forward per cell. The reading -- marginals, importance, corners,
#   priors -- is src/analysis/hyperparam_sensitivity.py, invoked at the end.
#
# THE FACTORIAL  (paper Table 2, NN row, mapped to sklearn MLPRegressor)
#   architecture   5   [32] [32,16] [32,16,8] [32,16,8,4] [32,16,8,4,2]  (NN1-5)
#   activation     2   identity (=paper 'linear'), relu
#   learning_rate  4   1e-4 1e-3 1e-2 1e-1        (paper's exact set)
#   alpha          4   1e-4 1e-3 1e-2 1e-1        (paper's exact set)
#                = 160 configs, each a full pinned walk-forward on crypto/v2.0.
#
#   THREE DEPARTURES FROM THE PAPER, none of them optional:
#   * LOSS is fixed at squared_error. The paper varies {MSE, Huber}; sklearn's
#     MLPRegressor offers only {squared_error, poisson} and poisson is wrong for
#     a bounded target. Huber would need the torch MLP; the ask was sklearn. The
#     paper found loss "relatively less consequential", so this costs little.
#   * The paper's THIRD activation, leaky-ReLU, is dropped, not substituted.
#     sklearn's only non-{identity,relu} options are tanh and logistic, and the
#     paper (footnote 14) explicitly avoids both for low signal-to-noise return
#     data -- they saturate and the gradient vanishes -- so a poor result would
#     be expected a priori and could not stand in for leaky-ReLU. identity and
#     relu are exact paper analogues; the activation axis is therefore 2, not 3.
#   * alpha is an L2 penalty here; the paper's is L1. Both shrink the weights;
#     the qualitative sensitivity is what transfers, not the exact coefficient.
#
#   AN AXIS THE PAPER HAS THAT E17 DOES NOT: early stopping and its
#   validation-window choice. The paper treats the early-stopping window (6mo /
#   2yr / 10yr) as a hyperparameter and Figure 7 makes it central to the NN
#   story. E17 keeps early_stopping=FALSE because sklearn's holdout is a RANDOM
#   split across periods -- the exact leak purged_period_splits exists to
#   prevent -- so E17 CANNOT reproduce or test the paper's early-stopping axis,
#   and the report must say so rather than imply full Table 2 coverage. Training-
#   loss plateau (n_iter_no_change) still stops each fit, so nets do not run the
#   full 200 epochs (E07 measured 30-115); that is a different mechanism from the
#   paper's validation-based early stopping.
#
# THE METRIC
#   Paper: out-of-sample R^2. Here: this repo's OOS payout, 0.1*corr + mmc, the
#   mean over the 514 walk-forward periods -- one number per config, plus the
#   per-period series for a first-half/second-half robustness split. Payout is
#   what the tournament pays and what every other arm is scored on, so the 240
#   configs sit on the same axis as the whole board.
#
# WHY THIS IS WORTH 240 RUNS
#   Every MLP study so far pinned or searched WITHIN the repo's grid and asked
#   "does this feature block help". None asked whether the GRID ITSELF is right.
#   E07 pinned alpha at the 1e1 ceiling and I widened it to 1e2 -- which cost
#   ~0.007 payout across three studies before I reverted it, because the CV
#   optimum and the OOS optimum diverge for this hyperparameter. That is exactly
#   the paper's warning, learned the expensive way. This experiment maps the
#   whole surface once so the grid can be set from evidence, not from one arm's
#   pinned draw.
#
# COST  ~30-60 s per pinned config (4 fits, no search); ~3 h sequential, ~45 min
#   across 4 shards. SHARD=/OF= round-robins the 160 cells.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e17_hyperparam_sensitivity_crypto.sh   # ~5 min
#   SHARD=1 OF=4 bash experiments/e17_hyperparam_sensitivity_crypto.sh
#   ONLY_LR=0.1 / ONLY_ARCH=32,16 / DRY_RUN=1 / NO_MAIL=1 / HP_THREADS=3
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E17}"
HP_WINDOW="${HP_WINDOW:-130}"
HP_REFIT="${HP_REFIT:-130}"
HP_EMBARGO="${HP_EMBARGO:-20}"
HP_N_SPLITS="${HP_N_SPLITS:-5}"
HP_VAL_FRAC="${HP_VAL_FRAC:-0.3}"
HP_SEED="${HP_SEED:-42}"
HP_DATA="${HP_DATA:-crypto/v2.0}"
HP_BATCH="${HP_BATCH:-1024}"
HP_THREADS="${HP_THREADS:-3}"
HP_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${HP_WINDOW}" in ""|0|[0-9]) echo "HP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${HP_REFIT}"  in ""|0|[0-9]) echo "HP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e17"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e17scout"; HP_ERA_STEP=4; HP_N_SPLITS=2
    echo "SCOUT MODE: era-step 4 -- wiring only, arms compare to nothing."
fi

export OMP_NUM_THREADS="${HP_THREADS}"
export MKL_NUM_THREADS="${HP_THREADS}"
export OPENBLAS_NUM_THREADS="${HP_THREADS}"

# The factorial. Parallel arrays are unnecessary here -- these are scalar tokens
# with no '|' -- so plain space-lists are clearer.
ARCHS="32 | 32,16 | 32,16,8 | 32,16,8,4 | 32,16,8,4,2"   # '|' delimits archs
ACTS="identity relu"
LRS="0.0001 0.001 0.01 0.1"
ALPHAS="0.0001 0.001 0.01 0.1"

# Optional single-axis filters, for re-running one slice.
IFS='|' read -ra ARCH_ARR <<< "${ARCHS}"

# Preflight: dataset present with the meta-model file, alpha ceiling irrelevant
# (we pin, we do not search), and the pinning mechanism actually pins all five.
python3 - "${HP_DATA}" <<'PY' || exit 1
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
# the mechanism: pin all five and confirm set_params takes them
try:
    e = clone(build_wrapper("mlp", "regression", seed=42, n_jobs=1).ModelClass)
    e.set_params(mlp__hidden_layer_sizes=[32, 16], mlp__activation="tanh",
                 mlp__learning_rate_init=0.1, mlp__alpha=0.0001, mlp__batch_size=1024)
except Exception as error:
    bad.append("pinning the five axes failed: %s" % error)
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s has the raw 22 and its meta file; all five axes pin "
      "through the pipeline" % version)
PY

# Build the work list.
CFG_NAME=(); CFG_PARAMS=()
for arch in "${ARCH_ARR[@]}"; do
    arch="$(echo "$arch" | tr -d ' ')"
    [ -n "${ONLY_ARCH:-}" ] && [ "${ONLY_ARCH//[[:space:]]/}" != "$arch" ] && continue
    archtag="a$(echo "$arch" | tr ',' '_')"
    for act in ${ACTS}; do
        [ -n "${ONLY_ACT:-}" ] && [ "${ONLY_ACT}" != "$act" ] && continue
        for lr in ${LRS}; do
            [ -n "${ONLY_LR:-}" ] && [ "${ONLY_LR}" != "$lr" ] && continue
            for al in ${ALPHAS}; do
                [ -n "${ONLY_ALPHA:-}" ] && [ "${ONLY_ALPHA}" != "$al" ] && continue
                # Exponent tags, one per value, DISTINCT. The earlier sed
                # scheme collapsed 1e-4..1e-1 all to "1" (strip "0." then leading
                # zeros), so every config got the same name and would have
                # clobbered every other's output file. Do not reintroduce it.
                case "$lr" in 0.0001) lrtag=lre4;; 0.001) lrtag=lre3;; 0.01) lrtag=lre2;; 0.1) lrtag=lre1;; *) echo "unmapped lr $lr" >&2; exit 1;; esac
                case "$al" in 0.0001) altag=ale4;; 0.001) altag=ale3;; 0.01) altag=ale2;; 0.1) altag=ale1;; *) echo "unmapped alpha $al" >&2; exit 1;; esac
                name="${PREFIX}_${archtag}_${act}_${lrtag}_${altag}"
                params="{\"mlp__hidden_layer_sizes\":[${arch}],\"mlp__activation\":\"${act}\",\"mlp__learning_rate_init\":${lr},\"mlp__alpha\":${al},\"mlp__batch_size\":${HP_BATCH}}"
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

# Every run name must be unique: two configs sharing a name write to the same
# results file and one silently clobbers the other. This is exactly the bug the
# exponent tags above fix; the gate makes any recurrence fail loudly here rather
# than after hours of compute produce a grid of ~10 files instead of 160.
dupes="$(printf '%s\n' "${SEL_NAME[@]}" | sort | uniq -d)"
if [ -n "$dupes" ]; then
    echo "FATAL: run-name collision -- these names map >1 config:" >&2
    printf '  %s\n' $dupes >&2; exit 1
fi

index=0; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  configs   ${total}  (of 160; shard ${SHARD}/${OF})
  method    Kalfa-Timmermann-van der Zwan hyperparameter sensitivity: one pinned
            walk-forward per cell of a 5x3x4x4 NN factorial, then marginal
            distributions, importance, corners and priors
  data      ${HP_DATA} (raw 22)   cadence rolling ${HP_WINDOW}/${HP_REFIT}, embargo ${HP_EMBARGO}
  expect    ~30-60 s per config; ~45 min this shard

Where to look:
  runner    experiments/e17_hyperparam_sensitivity_crypto.sh
  analysis  src/analysis/hyperparam_sensitivity.py
  results   results/metrics/crypto/mlp/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: MLP hyperparameter sensitivity  |  ${total} configs  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    name="${SEL_NAME[$i]}"; params="${SEL_PARAMS[$i]}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${HP_WINDOW}" --refit-every "${HP_REFIT}"
         --embargo "${HP_EMBARGO}" --n-splits "${HP_N_SPLITS}" --val-frac "${HP_VAL_FRAC}"
         --era-step "${HP_ERA_STEP}" --seed "${HP_SEED}" --search-jobs 1
         --fixed-params "${params}" --data-version "${HP_DATA}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%3d/%3d] %s\n' "$index" "$total" "$name"; continue; fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    # A failed config must not cost the others: read PIPESTATUS[0] after the tee.
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
echo "${EXPERIMENT} -- MLP hyperparameter sensitivity (Kalfa-Timmermann-van der Zwan)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} configs in $((el_all/60))m$((el_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL. Run the analysis only once ALL shards are done."
echo
# The full framework analysis: marginals, importance (fANOVA), corners, priors.
# Skipped for a scout or a partial shard -- it needs the whole 240-cell grid.
if [ "${MLP_SCOUT:-0}" != "1" ] && [ "${OF}" -le 1 ]; then
    python3 -m src.analysis.hyperparam_sensitivity --prefix "${PREFIX}" || \
        echo "analysis failed -- run: python3 -m src.analysis.hyperparam_sensitivity --prefix ${PREFIX}"
else
    echo "partial/scout run: once every shard's configs are on disk, read the"
    echo "framework analysis with"
    echo "    python3 -m src.analysis.hyperparam_sensitivity --prefix e17"
fi
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
