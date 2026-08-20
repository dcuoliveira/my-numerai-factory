#!/usr/bin/env bash
#
# Experiment: each GROUP of raw features, predicting the 60-DAY target
#
# The mirror of experiments/feature_groups_crypto.sh. Same seven groups, same
# protocol, same two ridge variants -- only the label changes:
#
#     group        n   features
#     ---------------------------------------------------------------
#     price        4   close_avg, close_ewa           x 20d/60d
#     size         4   market_cap_avg, market_cap_ewa x 20d/60d
#     liquidity    4   volume_avg, volume_ewa         x 20d/60d
#     oscillator   4   relative_strength_index, sharpe_ratio x 20d/60d
#     bollinger    2   bollinger                      x 20d/60d
#     momentum     2   momentum                       x 20d/60d
#     volatility   2   volatility                     x 20d/60d
#     ---------------------------------------------------------------
#                 22
#
# ---------------------------------------------------------------------------
# What the 20-day version found, and why the 60-day version is worth running
# ---------------------------------------------------------------------------
#
# On the 20-day target, `volatility` alone reached CORR 0.1436 -- 96% of the
# full 22-feature model -- and `rolling` turned out to be volatility wearing a
# different hat (rho 0.912). The whole 22-feature set collapses to roughly five
# independent quantities, and one of them dominates.
#
# The question here is whether that survives a horizon change. If volatility
# still dominates at 60 days, it is a property of the data; if the ordering
# moves, the 20-day ranking was partly fitted to its label.
#
# It also produces 14 more arms for the correlation matrix, and that is the
# practical payoff. CURUPIRA_05_60 was added to the book on the strength of a
# 0.39-0.43 correlation with the 20-day block -- which turned out to be the MEAN
# of a bimodal distribution whose per-date values sit above 0.90 on 45% of days.
# Narrow single-group models are the most likely place to find a decorrelator
# that is stable rather than intermittent.
#
# ---------------------------------------------------------------------------
# Automatic consequences of the 60-day label
# ---------------------------------------------------------------------------
#
#   embargo   derived from the target suffix: 60 periods, not 20. A 60-day
#             target overlaps 98% between consecutive daily dates, so the
#             20-day embargo would leak three times over.
#   rows      the 60-day label is NaN over the panel's last 40 dates; those
#             12,000 rows are dropped on the target being modelled.
#   periods   470 scored rather than 510, so these arms are NOT comparable as
#             levels with the 20-day ones. The correlation matrix is.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/feature_groups_target60_crypto.sh
#   DRY_RUN=1 bash experiments/feature_groups_target60_crypto.sh
#   ONLY=momentum bash experiments/feature_groups_target60_crypto.sh
#   SHARD=1 OF=3 bash experiments/feature_groups_target60_crypto.sh
#   NO_EMAIL=1 bash experiments/feature_groups_target60_crypto.sh
#
# It emails on completion AND on failure, and runs the correlation analysis
# against every CURUPIRA slot before mailing.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# G60_-prefixed: GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, and "window" is a real parameter here.
G60_TARGET="${G60_TARGET:-target_binned_return_60}"
G60_TASK="${G60_TASK:-regression}"
G60_SCHEME="${G60_SCHEME:-rolling}"
G60_TRAIN_WINDOW="${G60_TRAIN_WINDOW:-6month}"
G60_REFIT="${G60_REFIT:-6month}"
G60_N_ITER="${G60_N_ITER:-50}"
G60_N_SPLITS="${G60_N_SPLITS:-5}"
G60_SEED="${G60_SEED:-42}"
# The search is parallelised and the estimator is not. Ridge fits are
# single-threaded, so --n-jobs does nothing for it while --search-jobs spreads
# the 50 candidates x 5 folds across cores. run_forecast refuses the pair
# (search-jobs > 1, n-jobs in {-1, 0}) because that oversubscribes the box.
G60_SEARCH_JOBS="${G60_SEARCH_JOBS:-8}"
G60_N_JOBS="${G60_N_JOBS:-1}"

case "${G60_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "G60_TRAIN_WINDOW=${G60_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 6month" >&2; exit 1 ;;
esac

DATA="${G60_DATA:-crypto/v2.0}"
MODELS="${MODELS:-ridge_glmnet ridge}"

GROUP_NAME=(); GROUP_RE=()
add_group() { GROUP_NAME+=("$1"); GROUP_RE+=("$2"); }
add_group price      '^feature_close_(avg|ewa)_'
add_group size       '^feature_market_cap_(avg|ewa)_'
add_group liquidity  '^feature_volume_(avg|ewa)_'
add_group oscillator '^feature_(relative_strength_index|sharpe_ratio)_'
add_group bollinger  '^feature_bollinger_'
add_group momentum   '^feature_momentum_'
add_group volatility '^feature_volatility_'

# Refuse to start unless the groups exactly partition the 22 raw features. A
# silent gap or overlap makes every number below describe a different
# experiment than the one documented above.
python3 - "${GROUP_RE[@]}" <<'PY' || exit 1
import re, sys
import pyarrow.parquet as pq
cols = [c for c in pq.ParquetFile("data/crypto/v2.0/train.parquet").schema.names
        if c.startswith("feature_")]
seen, total = set(), 0
for pattern in sys.argv[1:]:
    hit = {c for c in cols if re.search(pattern, c)}
    if not hit:
        sys.exit("pattern %r matched nothing" % pattern)
    if hit & seen:
        sys.exit("patterns overlap on %s" % sorted(hit & seen))
    seen |= hit; total += len(hit)
missing = set(cols) - seen
if missing:
    sys.exit("groups do not cover %d feature(s): %s" % (len(missing), sorted(missing)))
print("group partition OK: %d groups cover all %d raw features" % (len(sys.argv) - 1, total))
PY

REPORT="logs/groups60_report.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }

notify() {
    [ "${NO_EMAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; the report is in ${REPORT})"
}
# Any unexpected exit still reports. Without this a crash three hours in is
# indistinguishable from a job that never started.
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "feature groups x target60 -- ABORTED (exit ${code})"
      fi' EXIT

total=$(( ${#GROUP_NAME[@]} * $(echo "${MODELS}" | wc -w) ))
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

say "=============================================================="
say "feature groups on ${G60_TARGET}"
say "${#GROUP_NAME[@]} groups x $(echo "${MODELS}" | wc -w) models = ${total} runs, shard ${SHARD}/${OF}"
say "window=${G60_TRAIN_WINDOW} refit=${G60_REFIT} embargo=auto (derives to 60)"
say "search-jobs=${G60_SEARCH_JOBS} n-jobs=${G60_N_JOBS}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!GROUP_NAME[@]}"; do
    group="${GROUP_NAME[$i]}"; pattern="${GROUP_RE[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${group}" ] && continue

    for model in ${MODELS}; do
        index=$((index + 1))
        [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

        tag="fixed"; [ "${model}" = "ridge_glmnet" ] && tag="glmnet"
        run_name="t60_grp_${group}_${tag}"

        cmd=(python3 -u -m src.run_forecast crypto "${model}"
             --task "${G60_TASK}" --target "${G60_TARGET}"
             --cv-scheme "${G60_SCHEME}" --train-window "${G60_TRAIN_WINDOW}"
             --refit-every "${G60_REFIT}" --embargo auto
             --data-version "${DATA}" --feature-include "${pattern}"
             --n-iter "${G60_N_ITER}" --n-splits "${G60_N_SPLITS}"
             --n-jobs "${G60_N_JOBS}" --search-jobs "${G60_SEARCH_JOBS}"
             --seed "${G60_SEED}" --run-name "${run_name}")

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '[%2d/%2d] %-26s %s\n' "$index" "$total" "$run_name" "${cmd[*]}"
            continue
        fi

        ran=$((ran + 1))
        started=$(date +%s)
        "${cmd[@]}" >"logs/${run_name}.log" 2>&1
        status=$?
        elapsed=$(( $(date +%s) - started ))
        if [ "$status" -ne 0 ]; then
            failed=$((failed + 1))
            say "$(printf '[%2d/%2d] %-26s FAILED (exit %d) after %ds' \
                 "$index" "$total" "$run_name" "$status" "$elapsed")"
        else
            say "$(printf '[%2d/%2d] %-26s done in %4ds' \
                 "$index" "$total" "$run_name" "$elapsed")"
        fi
    done
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

# --- correlation against every CURUPIRA slot ---------------------------------
say "computing correlations against the CURUPIRA slots..."
python3 -u -m src.analysis.groups60_corr >>"${REPORT}" 2>&1
analysis=$?
if [ "$analysis" -ne 0 ]; then
    say "(the correlation analysis exited ${analysis}; see the tail of this report)"
fi

trap - EXIT
if [ "${failed}" -eq 0 ] && [ "$analysis" -eq 0 ]; then
    notify "feature groups x target60 -- OK (${ran}/${total})"
else
    notify "feature groups x target60 -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
