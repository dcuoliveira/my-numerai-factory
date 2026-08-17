#!/usr/bin/env bash
#
# Experiment: feature importance by ABLATION, off the allfeat++ dataset
#
# Every arm uses the same fixed model -- ridge_glmnet, regression, rolling
# 6-month window, 6-month refit -- and the same 812-column dataset on disk. The
# only thing that varies is which columns the model is allowed to see, set by
# --feature-include / --feature-exclude regexes. No arm builds its own parquet.
#
# The 812 columns partition exactly into six blocks (verified, they sum to 812):
#
#     raw        22   the shipped features
#     tenc        1   mean-target encoding per symbol, lagged 20 periods
#     onehot    740   ticker identity
#     rolling    24   4 indicators x 2 horizons x windows 5/20/60
#     spread     11   20d minus 60d per indicator
#     interact   14   products and ewa-minus-sma acceleration gaps
#
# ---------------------------------------------------------------------------
# Four families
# ---------------------------------------------------------------------------
#
#   lobo      LEAVE ONE BLOCK OUT of allfeat++.  812 -> 812 minus that block.
#             Reads as "what does this block contribute ON TOP OF everything
#             else?" This is the marginal-contribution question and it is the
#             one that matters for a model you would actually deploy.
#
#   alone     EACH BLOCK AS THE ENTIRE DATASET.  Not raw+block -- block only.
#             Reads as "how much signal is in this block by itself?" A block can
#             score well alone and contribute nothing marginally, which is the
#             signature of redundancy rather than uselessness.
#
#   raw_loo   DROP ONE RAW FEATURE, keeping only the other 21. Cheap (~4 min an
#             arm) and the cleanest importance signal in the study, because each
#             feature is 1/22 of the model rather than 1/812.
#
#   full_loo  DROP ONE RAW FEATURE from the full 812. This is the literal
#             "remove one feature at a time from allfeat++" reading, and it is
#             included because it was asked for -- but expect it to be near
#             silent. glmnet picks one alpha across 812 collinear columns and
#             shrinks hard; removing 1 of 812 moves almost nothing, and the
#             remaining 740 one-hot columns can absorb a dropped indicator
#             through symbol identity. Read raw_loo for importance and this for
#             confirmation that the full model is saturated.
#
# ---------------------------------------------------------------------------
# Reading the output -- deltas must be PAIRED
# ---------------------------------------------------------------------------
#
# Do not compare arm means side by side. Every arm here is a near-copy of its
# reference (prediction correlations across this family run 0.89-1.00), so the
# per-period DIFFERENCE series has a far smaller standard error than either
# series alone. In this study a paired test found t = 3.14 on an effect worth
# 0.0003 CORR, while unpaired comparison of two arms 0.008 apart gave t = 0.75.
#
# src/analysis/ablation_report.py does this: paired per-period delta against the
# right reference, Newey-West at lag 20 for the 20-day forward-target overlap.
#
# Expect most blocks to come out flat. Across eleven earlier arms nothing beat
# the 22-feature baseline by a meaningful margin, and the mean off-diagonal
# prediction correlation was 0.914. Ablation is the tool that says WHY: whether
# that is because the blocks are redundant (score alone, contribute nothing) or
# genuinely empty (score nothing either way).
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   FAMILY=lobo     bash experiments/ablation_crypto.sh
#   FAMILY=alone    bash experiments/ablation_crypto.sh
#   FAMILY=raw_loo  bash experiments/ablation_crypto.sh
#   FAMILY=full_loo bash experiments/ablation_crypto.sh
#
#   DRY_RUN=1 FAMILY=lobo bash experiments/ablation_crypto.sh   # print only
#   ONLY=onehot FAMILY=lobo bash ...                            # a single arm
#
# Split one family across machines (1-based shard of N):
#   SHARD=1 OF=3 FAMILY=full_loo bash experiments/ablation_crypto.sh
#
# Under screen:
#   screen -L -Logfile ablation-lobo.txt env FAMILY=lobo bash experiments/ablation_crypto.sh
#
# RUN THE REFERENCE FIRST -- lobo and full_loo are meaningless without it:
#   SKIP_BUILD=1 bash experiments/allfeat_plus_crypto.sh
# (raw_loo's reference is w6r6_glmnet_reg, which already exists.)
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# ABL_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
ABL_MODEL="${ABL_MODEL:-ridge_glmnet}"
ABL_TASK="${ABL_TASK:-regression}"
ABL_SCHEME="${ABL_SCHEME:-rolling}"
ABL_TRAIN_WINDOW="${ABL_TRAIN_WINDOW:-6month}"
ABL_REFIT="${ABL_REFIT:-6month}"
ABL_N_ITER="${ABL_N_ITER:-50}"
ABL_N_SPLITS="${ABL_N_SPLITS:-5}"
ABL_N_JOBS="${ABL_N_JOBS:-8}"
ABL_SEED="${ABL_SEED:-42}"

case "${ABL_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "ABL_TRAIN_WINDOW=${ABL_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact"
FAMILY="${FAMILY:-lobo}"

# Block regexes. NOTE the negative lookahead in the one-hot pattern: the target
# encoding is called feature_sym_tenc and shares the prefix, so a naive
# ^feature_sym_ would silently take it along and confound the two blocks.
RE_ONEHOT='^feature_sym_(?!tenc$)'
RE_TENC='^feature_sym_tenc$'
RE_ROLL='_rm[0-9]+$'
RE_SPREAD='_spread$'
RE_INTERACT='_x_|_accel_'
RE_DERIVED="${RE_ONEHOT}|${RE_TENC}|${RE_ROLL}|${RE_SPREAD}|${RE_INTERACT}"

# Arms are three PARALLEL ARRAYS, not packed "name|include|exclude" strings.
# Every one of these regexes contains "|" as alternation, so packing them into a
# pipe-delimited record and splitting with IFS='|' silently tears them apart:
# "alone_interact|_x_|_accel_" parses as include="_x_", exclude="_accel_" and
# quietly fits on 8 columns instead of 14. Several other arms survived that only
# because `read` puts the unsplit remainder in the last variable. Parallel arrays
# have no delimiter to collide with.
ARM_NAME=(); ARM_INC=(); ARM_EXC=()
add_arm() { ARM_NAME+=("$1"); ARM_INC+=("$2"); ARM_EXC+=("$3"); }

case "${FAMILY}" in
  lobo)
    add_arm lobo_raw      "-" "^(?!.*(_sym_|${RE_ROLL}|${RE_SPREAD}|_x_|_accel_))"
    add_arm lobo_tenc     "-" "${RE_TENC}"
    add_arm lobo_onehot   "-" "${RE_ONEHOT}"
    add_arm lobo_rolling  "-" "${RE_ROLL}"
    add_arm lobo_spread   "-" "${RE_SPREAD}"
    add_arm lobo_interact "-" "${RE_INTERACT}"
    ;;
  alone)
    add_arm alone_raw      "-"               "${RE_DERIVED}"
    add_arm alone_tenc     "${RE_TENC}"      "-"
    add_arm alone_onehot   "${RE_ONEHOT}"    "-"
    add_arm alone_rolling  "${RE_ROLL}"      "-"
    add_arm alone_spread   "${RE_SPREAD}"    "-"
    add_arm alone_interact "${RE_INTERACT}"  "-"
    ;;
  raw_loo|full_loo)
    # The 22 shipped features, in the order the parquet carries them.
    mapfile -t RAW < <(python3 - <<'PY'
import pyarrow.parquet as pq
cols = pq.ParquetFile(
    "data/crypto/v2.0/train.parquet").schema.names
print("\n".join(c for c in cols if c.startswith("feature_")))
PY
)
    if [ "${#RAW[@]}" -ne 22 ]; then
        echo "expected 22 raw features, found ${#RAW[@]} -- refusing to run" >&2; exit 1
    fi
    for f in "${RAW[@]}"; do
        short="${f#feature_}"
        if [ "${FAMILY}" = "raw_loo" ]; then
            # keep the other 21 raw only: drop every derived block AND this feature
            add_arm "rawloo_${short}" "-" "${RE_DERIVED}|^${f}\$"
        else
            add_arm "fullloo_${short}" "-" "^${f}\$"
        fi
    done ;;
  *)
    echo "unknown FAMILY='${FAMILY}' (lobo | alone | raw_loo | full_loo)" >&2; exit 1 ;;
esac

total="${#ARM_NAME[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "family=${FAMILY}  arms=${total}  shard ${SHARD}/${OF}  data=${DATA}"
echo "=============================================================="

for i in "${!ARM_NAME[@]}"; do
    arm="${ARM_NAME[$i]}"; inc="${ARM_INC[$i]}"; exc="${ARM_EXC[$i]}"
    index=$((index + 1))

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${arm}" ] && continue
    # Round-robin sharding so each machine gets a mix of cheap and dear arms.
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    cmd=(python3 -u -m src.run_forecast crypto "${ABL_MODEL}"
         --task "${ABL_TASK}"
         --cv-scheme "${ABL_SCHEME}" --train-window "${ABL_TRAIN_WINDOW}"
         --refit-every "${ABL_REFIT}" --data-version "${DATA}"
         --n-iter "${ABL_N_ITER}" --n-splits "${ABL_N_SPLITS}"
         --n-jobs "${ABL_N_JOBS}" --seed "${ABL_SEED}"
         --run-name "w6r6_${arm}")
    [ "${inc}" != "-" ] && cmd+=(--feature-include "${inc}")
    [ "${exc}" != "-" ] && cmd+=(--feature-exclude "${exc}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-34s %s\n' "$index" "$total" "$arm" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${arm}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # A failed arm must not cost you the others.
    "${cmd[@]}" 2>&1 | tee "logs/abl_${arm}.log"
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
echo "family=${FAMILY}: ran ${ran} arms in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo "report with:  python3 -m src.analysis.ablation_report --family ${FAMILY}"
echo "=============================================================="
