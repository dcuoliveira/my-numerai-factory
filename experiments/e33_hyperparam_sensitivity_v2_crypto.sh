#!/usr/bin/env bash
#
# Experiment E33: MLP hyperparameter sensitivity, TAKE 2 -- with the two axes
# E17 held fixed now opened, on three datasets instead of one.
#
# E33 is the reference handle; every artefact it writes is e33_*.
#
# WHY A SECOND SWEEP
#   E17 ran the Kalfa-Timmermann-van der Zwan sensitivity framework on the raw 22
#   features and produced the mlp_ktz priors (identity, first layer <=32, alpha
#   <=0.1). IA_01 -- the best MLP slot in the book, an sklearn `mlp` on
#   fenccausal -- proves those priors do not transfer: its CV-selected config sits
#   OUTSIDE E17's swept space on TWO axes. E17 froze the first hidden layer at 32
#   and capped alpha at 0.1; IA_01 runs at width 64-128 and alpha 8-10. E17 could
#   not have found IA_01's optimum because that region was never in the grid.
#
#   So E33 re-runs the sweep with those two axes OPENED, seeded to bracket
#   IA_01's region with headroom, on three datasets spanning the book's
#   complexity -- so the new priors are calibrated to more than raw 22.
#
# THE FACTORIAL  (E17's, with width and alpha opened)
#   dataset        3   raw(22)  fenccausal(44)  tenc_onehot(763)
#   architecture  12   width {32,64,128,256} x depth {1,2,3}, halving:
#                        32 32,16 32,16,8 | 64 64,32 64,32,16 |
#                        128 128,64 128,64,32 | 256 256,128 256,128,64
#                      (E17 froze first layer at 32; depth capped at 3 -- E17
#                       found 4-5 layers collapse to a constant.)
#   activation     2   identity relu                      (E17's exact pair)
#   learning_rate  4   1e-4 1e-3 1e-2 1e-1                (E17's exact set)
#   alpha          6   1e-4 1e-2 1e-1 1 10 100            (E17 stopped at 0.1;
#                        1/10/100 bracket IA_01's measured 8-10 with headroom)
#                = 12 x 2 x 4 x 6 = 576 cells per dataset, 1728 total.
#
#   Same three departures from the paper as E17 (loss pinned squared_error;
#   leaky-relu dropped not substituted; alpha is L2 not L1), and the same axis it
#   cannot test (validation-window early stopping -- sklearn's holdout is a
#   period leak, so early_stopping stays FALSE). See E17 for the full note.
#
#   n_splits is 2, not E17's 5. The sweep pins every cell (--fixed-params), so
#   the inner CV only produces a recorded cv_score that the sensitivity analysis
#   does not use -- the OOS payout it DOES use comes from the full-window refit
#   and is identical at any n_splits. 2 is the floor (n_splits=1 yields 0 folds),
#   so this is a 3x compute saving with no effect on the analysis.
#
# THE METRIC  this repo's OOS payout (0.1*corr + mmc), mean over the walk-forward
#   periods -- one number per cell, plus the per-period series. Same axis as the
#   whole board. Analysis is run PER DATASET (the three have different scales; a
#   pooled marginal would be meaningless).
#
# COST  ~30-60 s per raw/fenccausal cell, minutes per wide tenc_onehot cell
#   (256-wide net on 763 inputs). ~30-60 CPU-hours total; ~3-5 h across ~12
#   shards, tenc_onehot's wide cells dominating. SHARD=/OF= round-robins.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e33_hyperparam_sensitivity_v2_crypto.sh
#   SHARD=1 OF=12 bash experiments/e33_hyperparam_sensitivity_v2_crypto.sh
#   ONLY_DS=fenc / ONLY_ACT=relu / ONLY_ALPHA=10 / DRY_RUN=1 / NO_MAIL=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E33}"
HP_WINDOW="${HP_WINDOW:-130}"
HP_REFIT="${HP_REFIT:-130}"
HP_EMBARGO="${HP_EMBARGO:-20}"
HP_N_SPLITS="${HP_N_SPLITS:-2}"
HP_VAL_FRAC="${HP_VAL_FRAC:-0.3}"
HP_SEED="${HP_SEED:-42}"
HP_BATCH="${HP_BATCH:-1024}"
HP_THREADS="${HP_THREADS:-3}"
HP_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${HP_WINDOW}" in ""|0|[0-9]) echo "HP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${HP_REFIT}"  in ""|0|[0-9]) echo "HP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${HP_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "HP_N_SPLITS must be >= 2 (1 yields no usable CV folds)" >&2; exit 1; }

PREFIX="e33"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e33scout"; HP_ERA_STEP=4
    echo "SCOUT MODE: era-step 4 -- wiring only, arms compare to nothing."
fi

export OMP_NUM_THREADS="${HP_THREADS}"
export MKL_NUM_THREADS="${HP_THREADS}"
export OPENBLAS_NUM_THREADS="${HP_THREADS}"

# Datasets: parallel arrays (tag, version, expected feature count).
DSTAG=(raw fenc oh)
DSVER=("crypto/v2.0" "crypto/v2.0-fenccausal" "crypto/v2.0-tenc-onehot")
DSFEAT=(22 44 763)

# The factorial. '|' delimits archs (a hidden_layer tuple has commas).
ARCHS="32 | 32,16 | 32,16,8 | 64 | 64,32 | 64,32,16 | 128 | 128,64 | 128,64,32 | 256 | 256,128 | 256,128,64"
ACTS="identity relu"
LRS="0.0001 0.001 0.01 0.1"
ALPHAS="0.0001 0.01 0.1 1 10 100"
IFS='|' read -ra ARCH_ARR <<< "${ARCHS}"

# Preflight: every dataset present with its meta file and expected feature count,
# and the five-axis pinning mechanism works (including the newly-opened values).
python3 - "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.base import clone
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
try:
    e = clone(build_wrapper("mlp", "regression", seed=42, n_jobs=1).ModelClass)
    # the opened axes: width 256 and alpha 100 must pin through
    e.set_params(mlp__hidden_layer_sizes=[256, 128, 64], mlp__activation="relu",
                 mlp__learning_rate_init=0.0001, mlp__alpha=100.0, mlp__batch_size=1024)
except Exception as error:
    bad.append("pinning the five axes (incl. width 256 / alpha 100) failed: %s" % error)
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d datasets present with meta files and expected feature "
      "counts; all five axes pin (incl. the opened width 256 / alpha 100)" % len(versions))
PY

# Build the work list across datasets x the factorial.
CFG_NAME=(); CFG_PARAMS=(); CFG_VER=()
for d in "${!DSTAG[@]}"; do
    dstag="${DSTAG[$d]}"; dsver="${DSVER[$d]}"
    [ -n "${ONLY_DS:-}" ] && [ "${ONLY_DS}" != "$dstag" ] && continue
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
                    # DISTINCT exponent tags per value -- see E17 on why a naive
                    # scheme collapsed 1e-4..1e-1 to one name and clobbered files.
                    case "$lr" in 0.0001) lrtag=lre4;; 0.001) lrtag=lre3;; 0.01) lrtag=lre2;; 0.1) lrtag=lre1;; *) echo "unmapped lr $lr" >&2; exit 1;; esac
                    case "$al" in 0.0001) altag=ale4;; 0.01) altag=ale2;; 0.1) altag=ale1;; 1) altag=al1;; 10) altag=al10;; 100) altag=al100;; *) echo "unmapped alpha $al" >&2; exit 1;; esac
                    name="${PREFIX}_${dstag}_${archtag}_${act}_${lrtag}_${altag}"
                    params="{\"mlp__hidden_layer_sizes\":[${arch}],\"mlp__activation\":\"${act}\",\"mlp__learning_rate_init\":${lr},\"mlp__alpha\":${al},\"mlp__batch_size\":${HP_BATCH}}"
                    CFG_NAME+=("$name"); CFG_PARAMS+=("$params"); CFG_VER+=("$dsver")
                done
            done
        done
    done
done

# Shard round-robin.
SEL_NAME=(); SEL_PARAMS=(); SEL_VER=(); picked=0
for i in "${!CFG_NAME[@]}"; do
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${CFG_NAME[$i]}"); SEL_PARAMS+=("${CFG_PARAMS[$i]}"); SEL_VER+=("${CFG_VER[$i]}")
done
total="${#SEL_NAME[@]}"
[ "$total" -eq 0 ] && { echo "no configs selected" >&2; exit 1; }

# Unique run-name gate: a collision means two configs write the same file and one
# clobbers the other. Fail loudly here, not after hours of compute.
dupes="$(printf '%s\n' "${SEL_NAME[@]}" | sort | uniq -d)"
if [ -n "$dupes" ]; then
    echo "FATAL: run-name collision -- these names map >1 config:" >&2
    printf '  %s\n' $dupes >&2; exit 1
fi

index=0; ran=0; failed=0; skipped=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  configs   ${total}  (of 1728; shard ${SHARD}/${OF})
  method    Kalfa-Timmermann-van der Zwan sensitivity, E17 re-run with the first
            hidden layer UNFROZEN (width 32/64/128/256) and alpha OPENED to 100
  data      raw(22) fenccausal(44) tenc_onehot(763), per-dataset analysis
  seed      IA_01's region: width 64-128, alpha 8-10 (relu) now inside the grid
  expect    ~3-5 h across the shards; wide tenc_onehot cells dominate

Where to look:
  runner    experiments/e33_hyperparam_sensitivity_v2_crypto.sh
  analysis  python3 -m src.analysis.hyperparam_sensitivity --prefix e33_raw|e33_fenc|e33_oh
  results   results/metrics/crypto/mlp/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: MLP sensitivity v2 (width + alpha opened)  |  ${total} configs  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    name="${SEL_NAME[$i]}"; params="${SEL_PARAMS[$i]}"; dsver="${SEL_VER[$i]}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${HP_WINDOW}" --refit-every "${HP_REFIT}"
         --embargo "${HP_EMBARGO}" --n-splits "${HP_N_SPLITS}" --val-frac "${HP_VAL_FRAC}"
         --era-step "${HP_ERA_STEP}" --seed "${HP_SEED}" --search-jobs 1
         --fixed-params "${params}" --data-version "${dsver}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%4d/%4d] %-44s %s\n' "$index" "$total" "$name" "$dsver"; continue; fi
    # Resume: a cell already on disk with a valid metrics json is skipped, so a
    # repartition or a restart -- including sharding the rest across deepsix over
    # the shared filesystem -- never recomputes finished work. Collapse cells
    # write no json (they error), so they are correctly retried, not skipped.
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/mlp/${name}.json" ]; then
        skipped=$((skipped+1)); continue
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  ${dsver}  $(date '+%H:%M:%S')"
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
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- MLP hyperparameter sensitivity v2 (width + alpha opened)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} configs in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed, ${skipped} already-done skipped"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL. Run the per-dataset analysis only once ALL shards are done."
echo
# Per-dataset framework analysis (marginals, fANOVA, corners, priors). The three
# datasets have different scales, so they are analysed separately. Only on a
# whole (unsharded, non-scout) run; otherwise print the three commands.
if [ "${MLP_SCOUT:-0}" != "1" ] && [ "${OF}" -le 1 ]; then
    for p in "${PREFIX}_raw" "${PREFIX}_fenc" "${PREFIX}_oh"; do
        echo "=== sensitivity analysis: ${p} ==="
        python3 -m src.analysis.hyperparam_sensitivity --prefix "${p}" || \
            echo "analysis failed -- run: python3 -m src.analysis.hyperparam_sensitivity --prefix ${p}"
    done
else
    echo "partial/scout run: once every shard's configs are on disk, read the"
    echo "three per-dataset analyses with"
    for p in "${PREFIX}_raw" "${PREFIX}_fenc" "${PREFIX}_oh"; do
        echo "    python3 -m src.analysis.hyperparam_sensitivity --prefix ${p}"
    done
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
