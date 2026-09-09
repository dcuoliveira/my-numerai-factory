#!/usr/bin/env bash
#
# E21 -- CLUSTER REGULARISATION on allfeat_pp: shrink some blocks harder than
#        others, and change what the model converges to.
#
# "Regularizing some but not all is super important ... I do like penalizing
#  e.g. interaction terms more than the normal terms. Like if I am throwing 10
#  variables in, that's 10 variables plus 10 squared terms plus 45 interactions,
#  and you can penalize based on which of the 3 groups they're in."
#
# "You need to think about the convergence of your model as lambda -> inf.
#  Sometimes you want that to be a constant but in some other cases ... we want
#  it to converge somewhere else."
#
# allfeat_pp is exactly that shape. Its 812 columns partition into six blocks by
# how they were BUILT:
#
#     raw       22   the base terms
#     spread    11   the same indicator differenced across horizons (20d - 60d)
#     interact  14   products and accelerations -- the interaction terms
#     rollsel   24   rolling-window versions
#     tenc       1   target encoding
#     onehot   740   ticker dummies
#
# Every model in this project shrinks all 812 with ONE alpha. That is the thing
# under test, and the repo already suspects it: cumlagged_features_crypto.sh
# states the mechanism as "glmnet applies ONE alpha across the whole design
# matrix, so each added block raises the shrinkage every existing column
# receives", and CURUPIRA_04 on these 812 columns scores BELOW the raw 22.
#
# ---------------------------------------------------------------------------
# Three different ways of cluster-regularising, not three doses of one
# ---------------------------------------------------------------------------
#
#   (a) LEVEL, up.   Penalise a derived block HARDER than raw. Delta's diagonal
#       carries a per-block multiplier. This is the quote, literally.
#
#   (b) LEVEL, down -- and this changes the LIMIT. Penalise raw LESS. As
#       alpha -> inf a flat ridge sends every coefficient to zero and the
#       prediction converges to a CONSTANT; with raw left nearly free the
#       penalised blocks vanish and the limit is the fit on raw alone.
#       Measured on a real 130-date block of this dataset:
#
#           alpha      flat: pred sd     raw:p1e-6: pred sd
#           1e+02        1.03e-02              3.32e-02
#           1e+04        1.63e-04              3.30e-02
#           1e+06        1.64e-06              2.99e-02
#
#       The flat model collapses; the other PLATEAUS across four decades. That
#       is "converge somewhere else", and it is why (b) is not just (a) with the
#       sign flipped -- it changes the shape of the whole alpha path.
#
#   (c) DISPERSION, not level. Fuse within a cluster instead of shrinking it:
#       Delta gains rho * S L S, which penalises how much a group's members
#       DISAGREE and leaves their common level alone. Separately identified from
#       (a): for a group of size k the block has eigenvalue m_g along the
#       group-mean direction and m_g + rho*s^2 along the other k-1.
#
# ---------------------------------------------------------------------------
# The control and what varies
# ---------------------------------------------------------------------------
#
# e21_flat is one alpha on all 812 columns -- the status quo, and what every
# deployed linear slot does. Every other arm shares its estimator, its 60-point
# alpha grid, its folds and its seed, and differs only in Delta's SHAPE.
#
# Expect a null on level. Fifteen feature blocks, five lag studies, a difference
# study and a recency-weighting study have all come back flat on this panel. The
# interesting outcome would be (b) beating (a), because that would say the value
# is in the alpha path's destination rather than its height.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/e21_cluster_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e21_cluster_reg_crypto.sh
#   SCOUT=1   bash experiments/e21_cluster_reg_crypto.sh
#   ONLY=e21_rawfree bash experiments/e21_cluster_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e21_cluster_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# CR_-prefixed. GNU screen exports $WINDOW=0 into every session it starts.
CR_MODEL="${CR_MODEL:-gen_ridge}"
CR_SCHEME="${CR_SCHEME:-rolling}"
CR_TRAIN_WINDOW="${CR_TRAIN_WINDOW:-130}"
CR_REFIT="${CR_REFIT:-130}"
CR_EMBARGO="${CR_EMBARGO:-20}"
CR_N_ITER="${CR_N_ITER:-50}"
CR_N_SPLITS="${CR_N_SPLITS:-5}"
CR_SEED="${CR_SEED:-42}"
CR_VAL_FRAC="${CR_VAL_FRAC:-0.3}"
CR_SEARCH_JOBS="${CR_SEARCH_JOBS:-4}"
CR_N_JOBS="${CR_N_JOBS:-1}"
DATA="${CR_DATA:-crypto/v2.0-tenc-onehot-rollsel-spread-interact}"

case "${CR_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "CR_TRAIN_WINDOW=${CR_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

# Parallel arrays, never pipe-delimited records: the specs contain no '|' today
# but a regex-bearing one would be torn apart by IFS='|' without a sound.
NAMES=(); OPTS=(); WHY=()
add() { NAMES+=("$1"); OPTS+=("$2"); WHY+=("$3"); }

#    arm                        --model-options                                                                     what it asks
add e21_flat        '{"structures":["identity"]}'                                                                   "control: one alpha on all 812"
# (a) penalise a derived block HARDER than raw
add e21_int_x10     '{"structures":["identity"],"group_specs":["interact:p10"]}'                                     "a: interactions 10x"
add e21_int_x100    '{"structures":["identity"],"group_specs":["interact:p100"]}'                                    "a: interactions 100x"
add e21_onehot_x10  '{"structures":["identity"],"group_specs":["onehot:p10"]}'                                       "a: the 740 dummies 10x"
add e21_onehot_x100 '{"structures":["identity"],"group_specs":["onehot:p100"]}'                                      "a: the 740 dummies 100x"
add e21_derived_x10 '{"structures":["identity"],"group_specs":["onehot:p10;interact:p10;spread:p10;rollsel:p10;tenc:p10"]}' "a: everything non-raw 10x"
# (b) penalise raw LESS -- this changes the lambda -> inf limit
add e21_rawfree     '{"structures":["identity"],"group_specs":["raw:p1e-3"]}'                                        "b: raw nearly free"
add e21_rawfree_hard '{"structures":["identity"],"group_specs":["raw:p1e-6"]}'                                       "b: raw free"
add e21_rawfree_oh10 '{"structures":["identity"],"group_specs":["raw:p1e-3;onehot:p10"]}'                            "b+a: raw free, dummies 10x"
# (c) fuse WITHIN clusters instead of shrinking them
add e21_fuse        '{"structures":["group"]}'                                                                       "c: within-stem fusion, rho searched"
# let the search choose among the shapes
add e21_searched    '{"structures":["identity"],"group_specs":["none","interact:p10","onehot:p10","onehot:p100","raw:p1e-3","raw:p1e-3;onehot:p10","onehot:p10;interact:p10;spread:p10;rollsel:p10;tenc:p10"]}' "the shape searched"

extra=(); suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="scout"
    echo "SCOUT MODE: --era-step 4. Wiring only; NOT comparable with a full baseline."
fi

REPORT="logs/e21_report${suffix:+_$suffix}.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E21 cluster regularisation -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on the BLOCK PARTITION, not just the feature count: an arm naming a block
# that does not exist would raise at the first refit, hours in.
python3 - "${DATA}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
from src.prediction.generalised_ridge import block_index
names = [c for c in pq.ParquetFile("data/%s/train.parquet" % sys.argv[1]).schema.names
         if c.startswith("feature_")]
blocks = block_index(names)
if len(names) != 812:
    sys.exit("expected 812 features, found %d" % len(names))
expected = {"raw": 22, "spread": 11, "interact": 14, "rollsel": 24,
            "tenc": 1, "onehot": 740}
got = {k: len(v) for k, v in blocks.items()}
if got != expected:
    sys.exit("block partition changed: %s != %s" % (got, expected))
print("block gate OK: %s" % ", ".join("%s=%d" % kv for kv in sorted(got.items())))
PY

total="${#NAMES[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)

say "=============================================================="
say "E21 cluster regularisation on allfeat_pp | ${total} arms"
say "812 columns: raw 22, spread 11, interact 14, rollsel 24, tenc 1, onehot 740"
say "control e21_flat = one alpha on all 812, which is what every slot does today"
say "model=${CR_MODEL} scheme=${CR_SCHEME} window=${CR_TRAIN_WINDOW} refit=${CR_REFIT}"
say "n-iter=${CR_N_ITER} n-splits=${CR_N_SPLITS} seed=${CR_SEED} search-jobs=${CR_SEARCH_JOBS}${suffix:+  SCOUT}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; opts="${OPTS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue

    if [ -n "${suffix}" ]; then run_name="e21scout_${name#e21_}"; else run_name="${name}"; fi
    cmd=(python3 -u -m src.run_forecast crypto "${CR_MODEL}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme "${CR_SCHEME}" --train-window "${CR_TRAIN_WINDOW}"
         --refit-every "${CR_REFIT}" --embargo "${CR_EMBARGO}"
         --n-splits "${CR_N_SPLITS}" --n-iter "${CR_N_ITER}"
         --seed "${CR_SEED}" --val-frac "${CR_VAL_FRAC}"
         --n-jobs "${CR_N_JOBS}" --search-jobs "${CR_SEARCH_JOBS}"
         --model-options "${opts}"
         --data-version "${DATA}" --run-name "${run_name}")
    cmd+=("${extra[@]+"${extra[@]}"}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-20s %-38s %s\n' "$index" "$total" "$name" "$why" "$opts"
        continue
    fi

    ran=$((ran + 1))
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log" >/dev/null
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%2d/%2d] %-20s FAILED (exit %d) after %ds  -- %s' \
             "$index" "$total" "$name" "$status" "$elapsed" "$why")"
    else
        say "$(printf '[%2d/%2d] %-20s done in %5ds  -- %s' "$index" "$total" "$name" "$elapsed" "$why")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

python3 -u -m src.analysis.e21_report ${suffix:+scout} >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"

trap - EXIT
if [ "${failed}" -eq 0 ]; then
    notify "E21 cluster regularisation -- OK (${ran}/${total})"
else
    notify "E21 cluster regularisation -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
