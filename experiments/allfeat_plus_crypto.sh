#!/usr/bin/env bash
#
# Experiment: "allfeat++" -- every feature block, all at once
#
# The first "everything" arm was not everything. allfeat (788) traded the 24
# rolling means for 11 spreads + 14 interactions, so it was a superset of
# CURUPIRA_01 and _02 but NOT of _03. This one is the genuine union:
#
#      22 raw (shipped)
#   +   1 mean-target encoding      (tenc)          -> CURUPIRA_01
#   + 740 ticker one-hot            (onehot)   \
#   +  24 rolling means             (rollsel)   >------> CURUPIRA_03
#   +  11 horizon spreads           (20d - 60d) \
#   +  14 products / accel gaps     (interact)   >------> allfeat
#   ----
#     812 features
#
# Verified: contains CURUPIRA_01 (23), CURUPIRA_02 (22), CURUPIRA_03 (786) and
# allfeat (788) with zero columns missing from any of them.
#
# Same fixed model as every other arm, so the feature set is the only thing
# varying:  ridge_glmnet, regression, rolling 6-month window, 6-month refit.
#
# ---------------------------------------------------------------------------
# What to expect, and what would count as a result
# ---------------------------------------------------------------------------
#
# Backtested payout of the parts (510 periods, 0.1*CORR + 1*MMC):
#
#     tenc+1hot+roll (787, the CURUPIRA union)  0.087143   sharpe 1.0632
#     CURUPIRA_03    (786)                      0.086721   sharpe 1.0554
#     CURUPIRA_02    (22)                       0.085763   sharpe 1.0818
#     CURUPIRA_01    (23)                       0.084132   sharpe 1.2960
#     allfeat        (788)                      0.081098   sharpe 1.2546
#
# The 787-feature union already exists and beat allfeat by 0.006 payout, so the
# question this arm answers is narrow: do the 25 derived columns (spread +
# interact) add anything ON TOP of the union, or does adding them repeat what
# happened to allfeat -- where the same 25 columns coincided with the LOWEST
# payout of any deployable arm?
#
# The mechanism to expect: glmnet applies one alpha across all 812 columns. On
# the 787-feature union it already chooses very heavy shrinkage. Adding 25 more
# collinear columns pushes alpha higher still, which shrinks the blocks that
# were working. Every "more features" arm in this study has landed flat or
# worse for exactly that reason, and the prediction-correlation heatmap put
# every ridge variant at 0.886-1.000 against the others.
#
# So this is a CEILING CHECK. A flat result closes out feature engineering on
# these inputs for good. A gain would have to clear roughly +0.003 payout to
# beat the union, and even then a paired t of ~0.3 is what the sample supports
# -- read the risk columns and the correlation to the deployed book before
# treating any number here as a reason to redeploy.
#
# Usage:
#   bash experiments/allfeat_plus_crypto.sh
#   DRY_RUN=1 bash experiments/allfeat_plus_crypto.sh
#   SKIP_BUILD=1 bash experiments/allfeat_plus_crypto.sh   # dataset already built
#
# Under screen:
#   screen -L -Logfile allfeat-plus.txt bash experiments/allfeat_plus_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# AFP_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
AFP_MODEL="${AFP_MODEL:-ridge_glmnet}"
AFP_TASK="${AFP_TASK:-regression}"
AFP_SCHEME="${AFP_SCHEME:-rolling}"
AFP_TRAIN_WINDOW="${AFP_TRAIN_WINDOW:-6month}"
AFP_REFIT="${AFP_REFIT:-6month}"
AFP_N_ITER="${AFP_N_ITER:-50}"
AFP_N_SPLITS="${AFP_N_SPLITS:-5}"
AFP_N_JOBS="${AFP_N_JOBS:-8}"
AFP_SEED="${AFP_SEED:-42}"

case "${AFP_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "AFP_TRAIN_WINDOW=${AFP_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

# Chain: the 787-feature CURUPIRA union -> +spreads -> +interactions.
UNION="crypto/v2.0-tenc-onehot-rollsel"
STEP1="${UNION}-spread"
FINAL="${STEP1}-interact"
RUN_NAME="w6r6_glmnet_reg_allfeatpp"

started_all=$(date +%s)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "build 1: python3 -m src.data_mani.build_derived --kind spread   --source-version ${UNION} --out-suffix=-spread"
    echo "build 2: python3 -m src.data_mani.build_derived --kind interact --source-version ${STEP1} --out-suffix=-interact"
    echo "run    : python3 -u -m src.run_forecast crypto ${AFP_MODEL} --task ${AFP_TASK}" \
         "--cv-scheme ${AFP_SCHEME} --train-window ${AFP_TRAIN_WINDOW} --refit-every ${AFP_REFIT}" \
         "--data-version ${FINAL} --n-iter ${AFP_N_ITER} --n-splits ${AFP_N_SPLITS}" \
         "--n-jobs ${AFP_N_JOBS} --seed ${AFP_SEED} --run-name ${RUN_NAME}"
    exit 0
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "=============================================================="
    echo "building the feature chain   $(date '+%H:%M:%S')"
    echo "=============================================================="
    # Each step reads the previous dataset and appends its block. The reference
    # file is copied into every version directory, so BMC and MMC stay identical
    # across arms and the comparison remains like-for-like.
    python3 -m src.data_mani.build_derived --kind spread \
        --source-version "${UNION}" --out-suffix=-spread   2>&1 | tee logs/build_allfeatpp_1.log || exit 1
    python3 -m src.data_mani.build_derived --kind interact \
        --source-version "${STEP1}" --out-suffix=-interact 2>&1 | tee logs/build_allfeatpp_2.log || exit 1
fi

# Refuse to run on the wrong dataset. A silent feature-count drift is the one
# error that produces a plausible number attached to the wrong experiment.
n_feat=$(python3 - "$FINAL" <<'PY'
import sys, pyarrow.parquet as pq
path = "data/%s/train.parquet" % sys.argv[1]
print(sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names))
PY
)
if [ "${n_feat}" != "812" ]; then
    echo "expected 812 features in ${FINAL}, found ${n_feat} -- refusing to run" >&2
    exit 1
fi
echo "verified ${n_feat} features in ${FINAL}"

echo
echo "=============================================================="
echo "running ${RUN_NAME} on ${FINAL}   $(date '+%H:%M:%S')"
echo "=============================================================="

python3 -u -m src.run_forecast crypto "${AFP_MODEL}" \
    --task "${AFP_TASK}" \
    --cv-scheme "${AFP_SCHEME}" --train-window "${AFP_TRAIN_WINDOW}" --refit-every "${AFP_REFIT}" \
    --data-version "${FINAL}" \
    --n-iter "${AFP_N_ITER}" --n-splits "${AFP_N_SPLITS}" --n-jobs "${AFP_N_JOBS}" \
    --seed "${AFP_SEED}" --run-name "${RUN_NAME}" 2>&1 | tee "logs/allfeatpp.log"
status="${PIPESTATUS[0]}"

echo
echo "=============================================================="
if [ "$status" -ne 0 ]; then
    echo "FAILED (exit ${status}) after $(( ($(date +%s) - started_all) / 60 )) min"
else
    echo "done in $(( ($(date +%s) - started_all) / 60 )) min"
    echo "compare against every other arm with:"
    echo "  python3 -c \"import json,glob;[print('%-40s payout %.6f  corr %.6f  dd %.4f' % (p.split('/')[-1][:-5], 0.1*json.load(open(p))['corr']['mean']+json.load(open(p))['mmc']['mean'], json.load(open(p))['corr']['mean'], json.load(open(p))['corr']['max_drawdown'])) for p in sorted(glob.glob('results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg*.json'))]\""
fi
echo "=============================================================="
