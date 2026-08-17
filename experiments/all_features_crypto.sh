#!/usr/bin/env bash
#
# Experiment: "everything at once"
#
# Stacks every feature block tested so far into one dataset and evaluates it
# against the same fixed model each arm has used:
#
#     ridge_glmnet, regression, rolling 6-month window, 6-month refit
#
#     22 raw
#    + 1 mean-target encoding   (tenc)
#    + 740 ticker one-hot       (onehot)
#    + 11 horizon spreads       (spread, 20d - 60d)
#    + 14 products / accel gaps (interact)
#   ---
#     788 features
#
# The builders compose through --source-version, so this is a chain rather than
# new code: tenc-onehot -> +spread -> +interact.
#
# What to expect, given the arms already run: every block tested in isolation
# came out flat or worse, and the prediction-correlation heatmap showed every
# variant at 0.886-1.000 against the others -- tenc was 1.000 against baseline,
# i.e. literally the same model. glmnet keeps choosing very heavy shrinkage
# (alpha up to 1.2M), which says the 22 raw features already span what is
# learnable from this input set.
#
# So this arm is a CEILING CHECK, not a candidate: if the combination also lands
# flat, that closes out feature engineering on these inputs and the remaining
# levers are a different model class (LightGBM measured rho=0.62 against this
# configuration, against 0.89-1.00 for every ridge variant) or a different
# target (the 60d label correlates only 0.52 with the 20d one).
#
# One caution on interpreting a gain here: the one-hot block alone COST 0.0075
# CORR, and onehot+rollsel posted the best CORR of any arm while doubling max
# drawdown (1.465 vs 0.805) and raising meta-model correlation to 0.586. Read
# the risk columns before the mean.
#
# Usage:
#   bash experiments/all_features_crypto.sh
#   DRY_RUN=1 bash experiments/all_features_crypto.sh
#   SKIP_BUILD=1 bash experiments/all_features_crypto.sh
#
# Under screen:
#   screen -L -Logfile all-features.txt bash experiments/all_features_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# Names are AF_-prefixed on purpose. GNU screen exports $WINDOW (the window
# number, 0) into every session it starts, so a bare WINDOW="${WINDOW:-6month}"
# silently becomes 0 under screen and every training block comes out empty.
AF_MODEL="${AF_MODEL:-ridge_glmnet}"
AF_TASK="${AF_TASK:-regression}"
AF_SCHEME="${AF_SCHEME:-rolling}"
AF_TRAIN_WINDOW="${AF_TRAIN_WINDOW:-6month}"
AF_REFIT="${AF_REFIT:-6month}"
AF_N_ITER="${AF_N_ITER:-50}"
AF_N_SPLITS="${AF_N_SPLITS:-5}"
AF_N_JOBS="${AF_N_JOBS:-8}"
AF_SEED="${AF_SEED:-42}"

case "${AF_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "AF_TRAIN_WINDOW=${AF_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

BASE="crypto/v2.0-tenc-onehot"
STEP1="${BASE}-spread"
FINAL="${STEP1}-interact"
RUN_NAME="w6r6_glmnet_reg_allfeat"

started_all=$(date +%s)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "build 1: python3 -m src.data_mani.build_target_encoding --smoothing 50 --keep-onehot --out-suffix=-tenc-onehot"
    echo "build 2: python3 -m src.data_mani.build_derived --kind spread   --source-version ${BASE}  --out-suffix=-spread"
    echo "build 3: python3 -m src.data_mani.build_derived --kind interact --source-version ${STEP1} --out-suffix=-interact"
    echo "run    : python3 -u -m src.run_forecast crypto ${AF_MODEL} --task ${AF_TASK}" \
         "--cv-scheme ${AF_SCHEME} --train-window ${AF_TRAIN_WINDOW} --refit-every ${AF_REFIT}" \
         "--data-version ${FINAL} --n-iter ${AF_N_ITER} --n-splits ${AF_N_SPLITS}" \
         "--n-jobs ${AF_N_JOBS} --seed ${AF_SEED} --run-name ${RUN_NAME}"
    exit 0
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "=============================================================="
    echo "building the feature chain   $(date '+%H:%M:%S')"
    echo "=============================================================="
    # Each step reads the previous dataset and appends its block. The reference
    # file is copied into every version directory, so BMC and MMC stay identical
    # across arms and the comparison remains like-for-like.
    python3 -m src.data_mani.build_target_encoding --smoothing 50 --keep-onehot \
        --out-suffix=-tenc-onehot 2>&1 | tee logs/build_allfeat_1.log || exit 1
    python3 -m src.data_mani.build_derived --kind spread \
        --source-version "${BASE}"  --out-suffix=-spread   2>&1 | tee logs/build_allfeat_2.log || exit 1
    python3 -m src.data_mani.build_derived --kind interact \
        --source-version "${STEP1}" --out-suffix=-interact 2>&1 | tee logs/build_allfeat_3.log || exit 1
fi

echo
echo "=============================================================="
echo "running ${RUN_NAME} on ${FINAL}   $(date '+%H:%M:%S')"
echo "=============================================================="

python3 -u -m src.run_forecast crypto "${AF_MODEL}" \
    --task "${AF_TASK}" \
    --cv-scheme "${AF_SCHEME}" --train-window "${AF_TRAIN_WINDOW}" --refit-every "${AF_REFIT}" \
    --data-version "${FINAL}" \
    --n-iter "${AF_N_ITER}" --n-splits "${AF_N_SPLITS}" --n-jobs "${AF_N_JOBS}" \
    --seed "${AF_SEED}" --run-name "${RUN_NAME}" 2>&1 | tee "logs/allfeat.log"
status="${PIPESTATUS[0]}"

echo
echo "=============================================================="
if [ "$status" -ne 0 ]; then
    echo "FAILED (exit ${status}) after $(( ($(date +%s) - started_all) / 60 )) min"
else
    echo "done in $(( ($(date +%s) - started_all) / 60 )) min"
    echo "compare against every other arm with:"
    echo "  python3 -c \"import json,glob;[print('%-38s corr %.6f  mmc %.6f  dd %.4f' % (p.split('/')[-1][:-5], json.load(open(p))['corr']['mean'], json.load(open(p))['mmc']['mean'], json.load(open(p))['corr']['max_drawdown'])) for p in sorted(glob.glob('results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg*.json'))]\""
fi
echo "=============================================================="
