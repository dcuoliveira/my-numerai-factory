#!/usr/bin/env bash
#
# Experiment: do lagged raw features add anything?
#
#     baseline   crypto/v2.0        22 raw features
#     lag p      crypto/v2.0-lagP   22 raw + 22 of the same features at t - p,
#                                   for p = 1 .. 10  (44 features per arm)
#
# One lag per arm, not cumulative. That isolates each horizon's marginal
# contribution; stacking lags 1..p would confound "does a lag help" with "does
# more of them help" and only the second question would get an answer.
#
# ---------------------------------------------------------------------------
# Why this axis, after nine flat feature blocks
# ---------------------------------------------------------------------------
#
# Ridge cannot form a difference unless both terms are columns. Two blocks in
# this project supplied one and are the only blocks that ever cleared |t| > 2:
#
#     spread     the same indicator across HORIZONS   (20d minus 60d)
#     interact   the same quantity across SMOOTHINGS  (ewa minus sma), t = 2.18
#
# A lag supplies the third axis, TIME: with b2 negative, b1*f_t + b2*f_{t-p} is a
# p-day change. It is the one difference the feature set does not already
# contain.
#
# What to expect, from the data rather than hope. Same-feature autocorrelation at
# lag 1 is 0.82 (momentum), 0.94 (volatility), 0.99 (close_avg) -- so lag 1 is
# very nearly a duplicate column and should do nothing except raise the shrinkage
# every other column receives. If any lag helps it should be a larger p, where
# f_t and f_{t-p} have actually separated. A flat result across all ten is the
# likely outcome and is itself the finding: eleven feature blocks have now been
# tried on this panel and none has beaten the raw 22 by a margin the sample
# supports.
#
# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------
#
# w6r6_glmnet_reg, already in results/: ridge_glmnet, regression, rolling
# 130-period window, 130-period refit, embargo 20, 5 folds, 50 draws, seed 42,
# val_frac 0.3, crypto/v2.0. corr 0.149169  mmc 0.068942  payout 0.083859.
# Every arm below copies that config field-for-field and changes only
# --data-version, so the feature set is the only thing that varies.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/lagged_features_crypto.sh
#   DRY_RUN=1 bash experiments/lagged_features_crypto.sh
#   SCOUT=1   bash experiments/lagged_features_crypto.sh   # --era-step 4, seconds
#   ONLY=3    bash experiments/lagged_features_crypto.sh
#   SHARD=1 OF=3 bash experiments/lagged_features_crypto.sh
#   NO_MAIL=1 bash experiments/lagged_features_crypto.sh
#
# Mails on success and on failure. SCOUT downsamples periods to shake out the
# wiring in seconds -- its numbers are NOT comparable with the full-resolution
# baseline, because the split boundary moves.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# LAG_-prefixed. GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, so a bare WINDOW="${WINDOW:-...}" silently becomes 0 and
# every training block comes out empty.
LAG_MODEL="${LAG_MODEL:-ridge_glmnet}"
LAG_TASK="${LAG_TASK:-regression}"
LAG_METRIC="${LAG_METRIC:-corr}"
LAG_SCHEME="${LAG_SCHEME:-rolling}"
LAG_TRAIN_WINDOW="${LAG_TRAIN_WINDOW:-130}"
LAG_REFIT="${LAG_REFIT:-130}"
LAG_EMBARGO="${LAG_EMBARGO:-20}"
LAG_N_ITER="${LAG_N_ITER:-50}"
LAG_N_SPLITS="${LAG_N_SPLITS:-5}"
LAG_SEED="${LAG_SEED:-42}"
LAG_VAL_FRAC="${LAG_VAL_FRAC:-0.3}"
# Ridge fits single-threaded, so the parallelism belongs to the search.
# run_forecast refuses (search-jobs > 1, n-jobs in {-1,0}) as oversubscription.
LAG_SEARCH_JOBS="${LAG_SEARCH_JOBS:-8}"
LAG_N_JOBS="${LAG_N_JOBS:-1}"

case "${LAG_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "LAG_TRAIN_WINDOW=${LAG_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

LAGS=(1 2 3 4 5 6 7 8 9 10)
EXPECTED_FEATURES=44

extra=()
suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="_scout"
    echo "SCOUT MODE: --era-step 4. Wiring check only; these numbers are NOT"
    echo "comparable with the full-resolution baseline."
fi

REPORT="logs/lagged_report${suffix}.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }

notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
# A crash three hours in must not be indistinguishable from a job that never
# started, so the trap reports too.
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "lagged features -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on feature count before running anything. A silent count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${EXPECTED_FEATURES}" "${LAGS[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
expected = int(sys.argv[1])
for lag in sys.argv[2:]:
    path = "data/crypto/v2.0-lag%s/train.parquet" % lag
    names = pq.ParquetFile(path).schema.names
    n = sum(c.startswith("feature_") for c in names)
    if n != expected:
        sys.exit("lag %s has %d features, expected %d" % (lag, n, expected))
    if not any(c.endswith("_lag%s" % lag) for c in names):
        sys.exit("lag %s dataset carries no _lag%s columns" % (lag, lag))
print("feature-count gate OK: %d datasets, %d features each"
      % (len(sys.argv) - 2, expected))
PY

total="${#LAGS[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

say "=============================================================="
say "lagged raw features | ${total} arms | shard ${SHARD}/${OF}"
say "baseline w6r6_glmnet_reg: corr 0.149169 mmc 0.068942 payout 0.083859"
say "model=${LAG_MODEL} scheme=${LAG_SCHEME} window=${LAG_TRAIN_WINDOW} refit=${LAG_REFIT} embargo=${LAG_EMBARGO}"
say "search-jobs=${LAG_SEARCH_JOBS} n-jobs=${LAG_N_JOBS}${suffix:+  SCOUT}"
say "started $(date -Is)"
say "=============================================================="

for p in "${LAGS[@]}"; do
    index=$((index + 1))
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${p}" ] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    run_name="w6r6_glmnet_reg_lag${p}${suffix}"
    cmd=(python3 -u -m src.run_forecast crypto "${LAG_MODEL}"
         --task "${LAG_TASK}" --metric "${LAG_METRIC}" --bmc-reference benchmark
         --cv-scheme "${LAG_SCHEME}" --train-window "${LAG_TRAIN_WINDOW}"
         --refit-every "${LAG_REFIT}" --embargo "${LAG_EMBARGO}"
         --n-splits "${LAG_N_SPLITS}" --n-iter "${LAG_N_ITER}"
         --seed "${LAG_SEED}" --val-frac "${LAG_VAL_FRAC}"
         --n-jobs "${LAG_N_JOBS}" --search-jobs "${LAG_SEARCH_JOBS}"
         --data-version "crypto/v2.0-lag${p}" --run-name "${run_name}")
    cmd+=("${extra[@]+"${extra[@]}"}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] lag %-2s %s\n' "$index" "$total" "$p" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log" >/dev/null
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%2d/%2d] lag %-2s FAILED (exit %d) after %ds' \
             "$index" "$total" "$p" "$status" "$elapsed")"
    else
        say "$(printf '[%2d/%2d] lag %-2s done in %4ds' "$index" "$total" "$p" "$elapsed")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

# Paired Newey-West deltas against the baseline, plus level and risk.
python3 -u - "${suffix}" >>"${REPORT}" 2>&1 <<'PY'
import json, os, sys
import numpy as np, pandas as pd
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
PP, MET = "results/per_period/crypto/ridge_glmnet", "results/metrics/crypto/ridge_glmnet"

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    se = np.sqrt(max(s, 0.0) / n)
    return mu, (mu / se if se > 0 else np.nan)

base_run = "w6r6_glmnet_reg" + suffix
bp = os.path.join(PP, base_run + ".csv")
if not os.path.exists(bp):
    print("baseline %s not found -- scout arms have no scout baseline" % base_run)
    raise SystemExit(0)
base = pd.read_csv(bp).set_index("date")
base_pay = 0.1 * base["corr"] + base["mmc"]

print("%-8s %5s %10s %10s %10s %8s %8s %11s %7s" % (
    "arm", "feat", "CORR", "MMC", "payout", "c.shp", "c.dd", "d payout", "t"))
bm = json.load(open(os.path.join(MET, base_run + ".json")))
print("%-8s %5d %10.6f %10.6f %10.6f %8.4f %8.4f %11s %7s" % (
    "baseline", bm["config"]["n_features"], bm["corr"]["mean"], bm["mmc"]["mean"],
    base_pay.mean(), bm["corr"]["sharpe"], bm["corr"]["max_drawdown"], "-", "-"))

rows = []
for p in range(1, 11):
    run = "w6r6_glmnet_reg_lag%d%s" % (p, suffix)
    mp = os.path.join(MET, run + ".json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp))
    t = pd.read_csv(os.path.join(PP, run + ".csv")).set_index("date")
    pay = 0.1 * t["corr"] + t["mmc"]
    j = pd.concat([base_pay, pay], axis=1, join="inner", keys=["b", "a"])
    d, tstat = nw(j["a"] - j["b"])
    rows.append((p, m, pay.mean(), d, tstat))
    print("%-8s %5d %10.6f %10.6f %10.6f %8.4f %8.4f %+11.6f %7.2f" % (
        "lag %d" % p, m["config"]["n_features"], m["corr"]["mean"],
        m["mmc"]["mean"], pay.mean(), m["corr"]["sharpe"],
        m["corr"]["max_drawdown"], d, tstat))

if rows:
    strong = [r for r in rows if abs(r[4]) > 2]
    print("\n%d of %d arms differ from the baseline at |t| > 2" % (len(strong), len(rows)))
    if not strong:
        lo = min(r[3] for r in rows); hi = max(r[3] for r in rows)
        print("None does. Payout deltas span %+.6f to %+.6f, all inside the noise "
              "of ~26 effective independent windows. Lagged raw features add "
              "nothing detectable on this panel." % (lo, hi))
PY

trap - EXIT
if [ "${failed}" -eq 0 ]; then
    notify "lagged features -- OK (${ran}/${total})"
else
    notify "lagged features -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
