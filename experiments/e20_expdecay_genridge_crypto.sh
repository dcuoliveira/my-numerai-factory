#!/usr/bin/env bash
#
# E20 -- does EXPONENTIAL RECENCY WEIGHTING help, on every dataset that feeds a
#        submitted model?  Generalised Ridge, W varied, everything else held.
#
# E20 and not E18: experiments/e18_tabnet_crypto.sh already owns e18_ and a
# STUDIES entry. Adding a second "E18" key to that dict literal would silently
# delete the TabNet study -- Python keeps the last duplicate and raises nothing.
#
# ---------------------------------------------------------------------------
# The hypothesis
# ---------------------------------------------------------------------------
#
# Every fit in this project weights its training rows equally inside a rolling
# 130-date window -- a rectangular kernel with a hard edge. If the relationship
# between features and target drifts, recent dates should count for more, and
# the natural refinement is an exponential kernel:
#
#     w_t = 0.5 ^ (age_in_dates / half_life)
#
# GeneralisedRidge is the only estimator here that can express it, because the
# weighting matrix W is one of its three free objects.
#
# ---------------------------------------------------------------------------
# The design: two arms per dataset, and W is the ONLY thing that differs
# ---------------------------------------------------------------------------
#
#     e20_unif_<tag>   w_scheme = uniform      (W = I, the control)
#     e20_exp_<tag>    w_scheme = exp_decay    half_life searched in DATES
#
# Both arms pin structure="identity" and search the SAME 60-point alpha grid, so
# the paired delta measures the weighting and nothing else. Pinning the
# structure is deliberate: on ~26 effective independent windows every extra
# searched dimension is a chance to fit noise, and the group-fusion axis is E17's
# question, not this one.
#
# Nine datasets, because those are exactly the datasets behind the ten
# single-model slots on the account. The other ten slots are blends of these, so
# a weighting that helps here propagates to them for free -- and one that does
# not, does not.
#
# ---------------------------------------------------------------------------
# Why the half-life grid stops at 32 dates
# ---------------------------------------------------------------------------
#
# Exponential decay costs effective sample. Over a 130-date block:
#
#     half-life   10     21     42     65    130   uniform
#     eff dates   28.9   59.0   95.8  112.5  125.0   130
#     eff 20d      1.4    2.9    4.8    5.6    6.3    6.5
#      windows
#
# A 130-date block at a 20-day overlapping target is already only ~6.5 effective
# independent windows. A 21-date half-life takes that to 2.9. Below 32 the arm is
# fitting noise, so the searched grid is floored there; a shorter half-life
# belongs in a PINNED arm and is not run here.
#
# ---------------------------------------------------------------------------
# What would count as a result, and the trap in reading one
# ---------------------------------------------------------------------------
#
# A paired Newey-West delta at |t| >= 2 on payout, on the same periods. Expect a
# null: fifteen feature blocks and five lag studies have already come back flat
# on this panel.
#
# One asymmetry to read carefully. The walk-forward refits every 130 periods and
# predicts the NEXT 130 with a 20-period embargo, so the newest training row is
# 20 periods from the FIRST predicted date and 150 from the LAST. Weighting
# toward the end of the training block sharpens the fit for the START of a block
# that then goes stale for another 130 days. If recency genuinely pays, the more
# direct lever is --refit-every 65, which this study does NOT vary -- so a null
# here is evidence about WEIGHTING, not about staleness.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/e20_expdecay_genridge_crypto.sh
#   DRY_RUN=1 bash experiments/e20_expdecay_genridge_crypto.sh
#   SCOUT=1   bash experiments/e20_expdecay_genridge_crypto.sh
#   ONLY=e20_exp_raw bash experiments/e20_expdecay_genridge_crypto.sh
#   SHARD=1 OF=3 bash experiments/e20_expdecay_genridge_crypto.sh
#   NO_MAIL=1 bash experiments/e20_expdecay_genridge_crypto.sh
#
# Mails on success and on failure.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# ED_-prefixed. GNU screen exports $WINDOW=0 into every session it starts, so a
# bare WINDOW="${WINDOW:-...}" silently becomes 0 and every block comes out empty.
ED_MODEL="${ED_MODEL:-gen_ridge}"
ED_TASK="${ED_TASK:-regression}"
ED_METRIC="${ED_METRIC:-corr}"
ED_SCHEME="${ED_SCHEME:-rolling}"
ED_TRAIN_WINDOW="${ED_TRAIN_WINDOW:-130}"
ED_REFIT="${ED_REFIT:-130}"
ED_EMBARGO="${ED_EMBARGO:-20}"
ED_N_ITER="${ED_N_ITER:-50}"
ED_N_SPLITS="${ED_N_SPLITS:-5}"
ED_SEED="${ED_SEED:-42}"
ED_VAL_FRAC="${ED_VAL_FRAC:-0.3}"
ED_SEARCH_JOBS="${ED_SEARCH_JOBS:-8}"
ED_N_JOBS="${ED_N_JOBS:-1}"

case "${ED_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "ED_TRAIN_WINDOW=${ED_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

# Parallel arrays, never pipe-delimited records: a packed record split on IFS='|'
# tears apart any field containing regex alternation, and that failure is silent.
TAGS=();  VERSIONS=();  EXPECT=();  SLOTS=()
add() { TAGS+=("$1"); VERSIONS+=("$2"); EXPECT+=("$3"); SLOTS+=("$4"); }

#   tag          data version                                      feat  fed slots
add raw          "crypto/v2.0"                                       22  "CURUPIRA_02,SACI_04"
add tenc         "crypto/v2.0-tenc"                                  23  "CURUPIRA_01"
add interact     "crypto/v2.0-interact"                              36  "SACI_01"
add fenccausal   "crypto/v2.0-fenccausal"                            44  "IA_01"
add cumlag3      "crypto/v2.0-cumlag3"                               88  "SACI_05"
add tencohot     "crypto/v2.0-tenc-onehot"                          763  "SACI_03"
add onehotroll   "crypto/v2.0-onehot-rollsel"                       786  "CURUPIRA_03"
add allfeat      "crypto/v2.0-tenc-onehot-spread-interact"          788  "SACI_02"
add allfeatpp    "crypto/v2.0-tenc-onehot-rollsel-spread-interact"  812  "CURUPIRA_04"

# Both arms pin the structure so W is the only thing that varies. The control
# carries no w_scheme entry at all, which is the exact default estimator.
CONTROL_OPTS='{"structures":["identity"]}'
DECAY_OPTS='{"structures":["identity"],"w_schemes":["exp_decay"]}'

extra=(); suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="scout"
    echo "SCOUT MODE: --era-step 4. Wiring check only; NOT comparable with a"
    echo "full-resolution baseline, because the split boundary moves."
fi

REPORT="logs/e20_report${suffix:+_$suffix}.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }

notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E20 exponential decay -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on feature count before running anything. A silent count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${VERSIONS[@]}" --- "${EXPECT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
args = sys.argv[1:]
cut = args.index("---")
versions, expected = args[:cut], [int(x) for x in args[cut + 1:]]
for version, want in zip(versions, expected):
    path = "data/%s/train.parquet" % version
    try:
        names = pq.ParquetFile(path).schema.names
    except Exception as error:
        sys.exit("cannot read %s: %s" % (path, error))
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        sys.exit("%s has %d features, expected %d" % (version, got, want))
print("feature-count gate OK: %d datasets, %d..%d features"
      % (len(versions), min(expected), max(expected)))
PY

total=$(( ${#TAGS[@]} * 2 ))
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

say "=============================================================="
say "E20 exponential recency weighting | ${total} arms | shard ${SHARD}/${OF}"
say "gen_ridge, structure pinned to identity, W the only thing varying"
say "half_life searched in DATES over 32,45,65,90,130"
say "9 datasets = every dataset behind a single-model submitted slot"
say "scheme=${ED_SCHEME} window=${ED_TRAIN_WINDOW} refit=${ED_REFIT} embargo=${ED_EMBARGO}"
say "n-iter=${ED_N_ITER} n-splits=${ED_N_SPLITS} seed=${ED_SEED} search-jobs=${ED_SEARCH_JOBS}${suffix:+  SCOUT}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"; version="${VERSIONS[$i]}"
    for kind in unif exp; do
        index=$((index + 1))
        if [ "${kind}" = "unif" ]; then opts="${CONTROL_OPTS}"; else opts="${DECAY_OPTS}"; fi
        name="e20_${kind}_${tag}"
        [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
        [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

        if [ -n "${suffix}" ]; then run_name="e20scout_${kind}_${tag}"; else run_name="${name}"; fi
        cmd=(python3 -u -m src.run_forecast crypto "${ED_MODEL}"
             --task "${ED_TASK}" --metric "${ED_METRIC}" --bmc-reference benchmark
             --cv-scheme "${ED_SCHEME}" --train-window "${ED_TRAIN_WINDOW}"
             --refit-every "${ED_REFIT}" --embargo "${ED_EMBARGO}"
             --n-splits "${ED_N_SPLITS}" --n-iter "${ED_N_ITER}"
             --seed "${ED_SEED}" --val-frac "${ED_VAL_FRAC}"
             --n-jobs "${ED_N_JOBS}" --search-jobs "${ED_SEARCH_JOBS}"
             --model-options "${opts}"
             --data-version "${version}" --run-name "${run_name}")
        cmd+=("${extra[@]+"${extra[@]}"}")

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '[%2d/%2d] %-22s %s\n' "$index" "$total" "$name" "${cmd[*]}"
            continue
        fi

        ran=$((ran + 1))
        started=$(date +%s)
        "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log" >/dev/null
        status="${PIPESTATUS[0]}"
        elapsed=$(( $(date +%s) - started ))
        if [ "$status" -ne 0 ]; then
            failed=$((failed + 1))
            say "$(printf '[%2d/%2d] %-22s FAILED (exit %d) after %ds' \
                 "$index" "$total" "$name" "$status" "$elapsed")"
        else
            say "$(printf '[%2d/%2d] %-22s done in %5ds' "$index" "$total" "$name" "$elapsed")"
        fi
    done
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

python3 -u -m src.analysis.e20_report ${suffix:+scout} >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"

trap - EXIT
if [ "${failed}" -eq 0 ]; then
    notify "E20 exponential decay -- OK (${ran}/${total})"
else
    notify "E20 exponential decay -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
