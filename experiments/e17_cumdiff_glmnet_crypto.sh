#!/usr/bin/env bash
#
# E17 -- cumulative per-symbol DIFFERENCES, to a year, on ridge_glmnet
#
# E17, not E16: a concurrent session in this same checkout claimed E16,
# the e16_* prefix and the v2.0-e16d* datasets for the SAME question on
# lgb and mlp (experiments/e16_diffladder_lgb_reg_crypto.sh,
# experiments/e16_cumdiff_mlp_crypto.sh). The two studies are
# complementary -- different model trees, different ladders (that one
# uses 1,5,22,66,132,264; this one 1,2,3,5,10,21,63,130,261, measured
# from the panel at 261.0 dates/yr) -- so read them together. They do not
# share a run name, a dataset or a builder. The E13 handle is doubly
# booked for exactly the want of this paragraph.
#
#     control        crypto/v2.0                 22 raw features
#     diff  j        crypto/v2.0-e17diff<j>      the difference columns ALONE
#     rawdiff j      crypto/v2.0-e17rawdiff<j>   raw 22 beside them
#
# Difference orders are LOG-SPACED to a calendar year:
#
#     K = 1  2  3  5  10  21  63  130  261
#         d  2d 3d wk 2wk mo  qtr  6mo  yr
#
# and level j is CUMULATIVE over that ladder -- level 4 carries orders 1,2,3,5.
# 261 dates is one year on THIS panel, measured (1,714 dates over 2,399 calendar
# days = 261.0/yr), not the 252 a trading-day convention would assume.
#
# Consecutive daily orders to 261 were rejected: they would be 261
# near-identical copies of one fact and 5,742 columns of it. build_adl makes the
# same argument for its AR lags ("Consecutive daily lags would give 20 nearly
# identical copies of the same window").
#
# ---------------------------------------------------------------------------
# What this asks that the cumulative-LAG study did not
# ---------------------------------------------------------------------------
#
# Same span, different basis:
#
#     span{ f_t, f_t - f_{t-k} ... }  ==  span{ f_t, f_{t-k} ... }
#
# An unpenalised fit could not tell the two apart. Ridge can: ||b||^2 is not
# invariant under a change of basis, so shrinkage is applied along whichever
# axes the columns define. Same-feature autocorrelation at lag 1 here is 0.82
# (momentum), 0.94 (volatility), 0.99 (close_avg), so the LAGGED-LEVEL basis
# hands glmnet a matrix of near-copies while the DIFFERENCE basis replaces each
# near-copy with a residual nearly orthogonal to the level it came from. Under
# one alpha shared across the whole matrix -- glmnet's design -- the second
# should shrink far less destructively.
#
# That is the hypothesis, and the study tests it directly: at levels 3, 6 and 9
# the featdiff arm and the lag arm carry the SAME SPAN and differ only in basis.
#
# The prior on the LEVEL of the result is still bad. E05 swept cumulative lags
# on LightGBM and the best depth was ZERO; E06 narrowed to volatility and found
# them harmful (t = -2.18 .. -2.30); E13/E14 found ADL null on the MLP; the ridge
# ADL arms were 0 of 14 at |t| >= 2. Fifteen feature blocks have been tried on
# this panel and none has beaten the raw 22 by a margin ~26 effective
# independent windows can support. A flat result is the expected outcome and is
# itself the finding.
#
# ---------------------------------------------------------------------------
# Coverage -- read this before reading any long-order arm
# ---------------------------------------------------------------------------
#
# A difference of order k exists only for rows with k predecessors. On this
# panel:
#
#     k        1     21     63    130    261
#     rows  99.3%  94.8%  86.7%  76.0%  59.9%
#
# (measured by build_cumdiff on the 1,714-date snapshot, not estimated)
#
# So the year-order block is the 0.0 fill for 39% of rows -- tokens younger than
# a year, which is most of the universe. That is an honest "no view", but it
# means level 9 is measuring a column that is silent on two rows in five. Do not
# read a flat level 9 as "a year of change carries nothing"; read it as "a year
# of change, where it exists at all".
#
# ---------------------------------------------------------------------------
# The fill value
# ---------------------------------------------------------------------------
#
# build_cumdiff fills its own NaN with 0.0 and NOT with the features' neutral
# 0.5. A difference lives on [-1, +1] centred at zero; 0.5 would be half the
# full range, stamped with the same sign on every young token. build_lagged
# fills with 0.5 and is right to -- its columns are LEVELS. Mixing the two rules
# is silent and would bias exactly the rows with no history.
#
# ---------------------------------------------------------------------------
# The control is re-run, deliberately
# ---------------------------------------------------------------------------
#
# w6r6_glmnet_reg (corr 0.149169 mmc 0.068942 payout 0.083859) was fitted on an
# OLDER snapshot: 1,701 dates and 1,309 symbols against today's 1,714 and 1,317.
# Comparing a new arm to it would confound the feature set with the snapshot, so
# e17_raw refits the identical config on the current data and every delta in
# this study is measured against THAT. e17_raw against w6r6_glmnet_reg is itself
# worth reading: it prices the snapshot drift.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/e17_cumdiff_glmnet_crypto.sh
#   DRY_RUN=1 bash experiments/e17_cumdiff_glmnet_crypto.sh
#   SCOUT=1   bash experiments/e17_cumdiff_glmnet_crypto.sh   # --era-step 4
#   ONLY=e17_rawdiff3 bash experiments/e17_cumdiff_glmnet_crypto.sh
#   SHARD=1 OF=3 bash experiments/e17_cumdiff_glmnet_crypto.sh
#   NO_MAIL=1 bash experiments/e17_cumdiff_glmnet_crypto.sh
#
# Mails on success and on failure. SCOUT downsamples periods to shake out the
# wiring in seconds -- its numbers are NOT comparable with a full-resolution
# baseline, because the split boundary moves.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# CD_-prefixed. GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, so a bare WINDOW="${WINDOW:-...}" silently becomes 0 and
# every training block comes out empty.
CD_MODEL="${CD_MODEL:-ridge_glmnet}"
CD_TASK="${CD_TASK:-regression}"
CD_METRIC="${CD_METRIC:-corr}"
CD_SCHEME="${CD_SCHEME:-rolling}"
CD_TRAIN_WINDOW="${CD_TRAIN_WINDOW:-130}"
CD_REFIT="${CD_REFIT:-130}"
CD_EMBARGO="${CD_EMBARGO:-20}"
CD_N_ITER="${CD_N_ITER:-50}"
CD_N_SPLITS="${CD_N_SPLITS:-5}"
CD_SEED="${CD_SEED:-42}"
CD_VAL_FRAC="${CD_VAL_FRAC:-0.3}"
# Ridge fits single-threaded, so the parallelism belongs to the search.
# run_forecast refuses (search-jobs > 1, n-jobs in {-1,0}) as oversubscription.
CD_SEARCH_JOBS="${CD_SEARCH_JOBS:-8}"
CD_N_JOBS="${CD_N_JOBS:-1}"

case "${CD_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "CD_TRAIN_WINDOW=${CD_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

# Parallel arrays, never pipe-delimited records: a packed name|version record
# split on IFS='|' would tear apart any field containing regex alternation, and
# that failure is silent -- it fits on the wrong columns and still reports a
# plausible number.
NAMES=(); VERSIONS=(); EXPECT=()

add() { NAMES+=("$1"); VERSIONS+=("$2"); EXPECT+=("$3"); }

# The control, on the CURRENT snapshot.
add e17_raw            "crypto/v2.0"              22

# Ask 1: the difference columns standing alone.        23j features
add e17_diff1          "crypto/v2.0-e17diff1"      23
add e17_diff2          "crypto/v2.0-e17diff2"      46
add e17_diff3          "crypto/v2.0-e17diff3"      69
add e17_diff4          "crypto/v2.0-e17diff4"      92
add e17_diff5          "crypto/v2.0-e17diff5"     115
add e17_diff6          "crypto/v2.0-e17diff6"     138
add e17_diff7          "crypto/v2.0-e17diff7"     161
add e17_diff8          "crypto/v2.0-e17diff8"     184
add e17_diff9          "crypto/v2.0-e17diff9"     207

# Ask 2: the difference columns beside the raw set.    22 + 23j features
add e17_rawdiff1       "crypto/v2.0-e17rawdiff1"   45
add e17_rawdiff2       "crypto/v2.0-e17rawdiff2"   68
add e17_rawdiff3       "crypto/v2.0-e17rawdiff3"   91
add e17_rawdiff4       "crypto/v2.0-e17rawdiff4"  114
add e17_rawdiff5       "crypto/v2.0-e17rawdiff5"  137
add e17_rawdiff6       "crypto/v2.0-e17rawdiff6"  160
add e17_rawdiff7       "crypto/v2.0-e17rawdiff7"  183
add e17_rawdiff8       "crypto/v2.0-e17rawdiff8"  206
add e17_rawdiff9       "crypto/v2.0-e17rawdiff9"  229

# The basis test. featdiff and lag at the same level carry the SAME SPAN and
# differ only in which basis spans it, so their delta isolates the penalty
# geometry from the information content. 22(j+1) features each.
add e17_featdiff3      "crypto/v2.0-e17featdiff3"  88
add e17_lag3           "crypto/v2.0-e17lag3"       88
add e17_featdiff6      "crypto/v2.0-e17featdiff6" 154
add e17_lag6           "crypto/v2.0-e17lag6"      154
add e17_featdiff9      "crypto/v2.0-e17featdiff9" 220
add e17_lag9           "crypto/v2.0-e17lag9"      220

# Attribution: the target half alone, so a move in rawdiff6 says which half.
add e17_tgtdiff6       "crypto/v2.0-e17tgtdiff6"   28

extra=()
suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="_scout"
    echo "SCOUT MODE: --era-step 4. Wiring check only; these numbers are NOT"
    echo "comparable with the full-resolution baseline."
fi

REPORT="logs/e17_report${suffix}.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }

notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
# A crash four hours in must not be indistinguishable from a job that never
# started, so the trap reports too.
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E17 cumulative differences -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on feature count before running anything. A silent count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${VERSIONS[@]}" --- "${EXPECT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
args = sys.argv[1:]
cut = args.index("---")
versions, expected = args[:cut], [int(x) for x in args[cut + 1:]]
assert len(versions) == len(expected)
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

total="${#NAMES[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

say "=============================================================="
say "E17 cumulative differences to a year | ${total} arms | shard ${SHARD}/${OF}"
say "orders 1,2,3,5,10,21,63,130,261 (261 dates = 1 calendar year here)"
say "control e17_raw = w6r6_glmnet_reg's config on the CURRENT snapshot"
say "model=${CD_MODEL} scheme=${CD_SCHEME} window=${CD_TRAIN_WINDOW} refit=${CD_REFIT} embargo=${CD_EMBARGO}"
say "n-iter=${CD_N_ITER} n-splits=${CD_N_SPLITS} seed=${CD_SEED} search-jobs=${CD_SEARCH_JOBS}${suffix:+  SCOUT}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; version="${VERSIONS[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    [ -n "${SKIP:-}" ] && [ "${SKIP}" = "${name}" ] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    # Scouts take the e17scout_ PREFIX, not an _scout suffix: experiment_db
    # resolves a run to its study by PREFIX, so e17_diff1_scout would be counted
    # as a 27th full arm of E17 rather than a rehearsal of the 2nd.
    if [ -n "${suffix}" ]; then
        run_name="e17scout_${name#e17_}"
    else
        run_name="${name}"
    fi
    cmd=(python3 -u -m src.run_forecast crypto "${CD_MODEL}"
         --task "${CD_TASK}" --metric "${CD_METRIC}" --bmc-reference benchmark
         --cv-scheme "${CD_SCHEME}" --train-window "${CD_TRAIN_WINDOW}"
         --refit-every "${CD_REFIT}" --embargo "${CD_EMBARGO}"
         --n-splits "${CD_N_SPLITS}" --n-iter "${CD_N_ITER}"
         --seed "${CD_SEED}" --val-frac "${CD_VAL_FRAC}"
         --n-jobs "${CD_N_JOBS}" --search-jobs "${CD_SEARCH_JOBS}"
         --data-version "${version}" --run-name "${run_name}")
    cmd+=("${extra[@]+"${extra[@]}"}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-18s %s\n' "$index" "$total" "$name" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log" >/dev/null
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%2d/%2d] %-18s FAILED (exit %d) after %ds' \
             "$index" "$total" "$name" "$status" "$elapsed")"
    else
        say "$(printf '[%2d/%2d] %-18s done in %4ds' "$index" "$total" "$name" "$elapsed")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

python3 -u -m src.analysis.e17_report "${suffix}" >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"

trap - EXIT
if [ "${failed}" -eq 0 ]; then
    notify "E17 cumulative differences -- OK (${ran}/${total})"
else
    notify "E17 cumulative differences -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
