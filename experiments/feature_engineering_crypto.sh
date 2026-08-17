#!/usr/bin/env bash
#
# Experiment: "feature engineering for crypto"
#
# Builds each derived dataset and evaluates it against the same fixed model, so
# the ONLY thing varying between arms is the feature set:
#
#     ridge_glmnet, regression, rolling 6-month window, 6-month refit
#
# That configuration is the best risk-adjusted model found so far (sharpe 1.81,
# max drawdown 0.805), and holding it fixed is what makes the arms comparable.
#
#   A  spread     11 columns: every indicator's 20d minus 60d.
#                 The raw features are LEVELS; a spread is a CHANGE OF STATE.
#                 momentum_20d - momentum_60d separates short-term reversal
#                 from longer-term momentum, which a single level conflates.
#
#   B  interact   14 columns: acceleration gaps (ewa - sma) and market-structure
#                 products (momentum x vol, mcap x momentum, volume x vol,
#                 rsi x vol). Ridge cannot form a product, so this is how a
#                 linear model gets some of what a tree gets for free.
#
# A and B are separate datasets on purpose. Combined, a gain could not be
# attributed, and glmnet applies one alpha across every column -- shrinkage
# strong enough for one block also crushes the other.
#
# Usage:
#   bash experiments/feature_engineering_crypto.sh            # build + run, in order
#   DRY_RUN=1 bash experiments/feature_engineering_crypto.sh  # print only
#   SKIP_BUILD=1 bash experiments/feature_engineering_crypto.sh
#   ONLY=spread bash experiments/feature_engineering_crypto.sh
#
# Run it under screen so it survives a disconnect:
#   screen -L -Logfile feature-engineering.txt bash experiments/feature_engineering_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# Model configuration, identical for every arm.
FE_MODEL="${FE_MODEL:-ridge_glmnet}"
FE_TASK="${FE_TASK:-regression}"
FE_SCHEME="${FE_SCHEME:-rolling}"
# NOT "WINDOW": GNU screen exports $WINDOW (the window number, 0) into
# every session it starts, so ${WINDOW:-6month} silently becomes 0 under
# screen and every training block comes out empty. Prefix the names.
FE_TRAIN_WINDOW="${FE_TRAIN_WINDOW:-6month}"
FE_REFIT="${FE_REFIT:-6month}"
FE_N_ITER="${FE_N_ITER:-50}"
FE_N_SPLITS="${FE_N_SPLITS:-5}"
FE_N_JOBS="${FE_N_JOBS:-8}"
FE_SEED="${FE_SEED:-42}"

# arm | build command | data-version | run name
ARMS=(
  "spread|python3 -m src.data_mani.build_derived --kind spread|crypto/v2.0-spread|w6r6_glmnet_reg_spread"
  "interact|python3 -m src.data_mani.build_derived --kind interact|crypto/v2.0-interact|w6r6_glmnet_reg_interact"
)

case "${FE_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "FE_TRAIN_WINDOW=${FE_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

total="${#ARMS[@]}"
index=0
failed=0
started_all=$(date +%s)

for entry in "${ARMS[@]}"; do
    IFS='|' read -r arm build version run_name <<< "$entry"
    index=$((index + 1))

    if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${arm}" ]; then
        continue
    fi

    cmd=(python3 -u -m src.run_forecast crypto "${FE_MODEL}"
         --task "${FE_TASK}"
         --cv-scheme "${FE_SCHEME}" --train-window "${FE_TRAIN_WINDOW}" --refit-every "${FE_REFIT}"
         --data-version "${version}"
         --n-iter "${FE_N_ITER}" --n-splits "${FE_N_SPLITS}" --n-jobs "${FE_N_JOBS}"
         --seed "${FE_SEED}" --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-9s build: %s\n' "$index" "$total" "$arm" "$build"
        printf '        %-9s run  : %s\n' "" "${cmd[*]}"
        continue
    fi

    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${arm}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    if [ "${SKIP_BUILD:-0}" != "1" ]; then
        echo "--- building ${version} ---"
        $build 2>&1 | tee "logs/build_${arm}.log"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            failed=$((failed + 1))
            echo ">>> ${arm} BUILD FAILED -- skipping its run"
            continue
        fi
    fi

    started=$(date +%s)
    # A failed arm must not cost you the others.
    "${cmd[@]}" 2>&1 | tee "logs/fe_${arm}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${arm} FAILED (exit ${status}) after ${elapsed}s -- continuing"
    else
        echo ">>> ${arm} done in ${elapsed}s"
    fi
done

echo
echo "=============================================================="
echo "finished ${total} arms in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo "compare with:"
echo "  python3 -c \"import json,glob;[print('%-34s corr %.6f  mmc %.6f' % (p.split('/')[-1][:-5], json.load(open(p))['corr']['mean'], json.load(open(p))['mmc']['mean'])) for p in sorted(glob.glob('results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg*.json'))]\""
echo "=============================================================="
