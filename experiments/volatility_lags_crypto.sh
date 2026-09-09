#!/usr/bin/env bash
#
# Experiment: cumulative lags of VOLATILITY only
#
#     baseline   w6r6_grp_volatility_glmnet   2 features, CORR 0.143612
#     vol p      volatility + its own lags 1..p        2(p+1) features
#
# ---------------------------------------------------------------------------
# Why this is not a repeat of the lag study that just came back flat
# ---------------------------------------------------------------------------
#
# That study lagged all 22 raw features together, so arm p carried 22 lagged
# columns of which 20 belong to groups that score 0.015-0.060 standalone. glmnet
# applies ONE alpha across the whole design matrix, so those 20 near-duplicate
# columns raise the shrinkage that any useful volatility lag would also receive.
# A real signal in volatility's history could have been diluted out of sight --
# CORR fell monotonically with p in both sweeps, which is what dilution looks
# like.
#
# So this asks the same question where the signal actually is. Volatility is the
# dominant group by a wide margin: 2 columns reach CORR 0.143612, 96% of the full
# 22-feature model's 0.149169, and it stays first at the 60-day horizon (0.180).
# If lagged history helps anywhere, here is where it should show.
#
# No new datasets. The crypto/v2.0-cumlagP sets already carry every raw feature
# and every lag 1..p, so each arm is a --feature-include away.
#
# ---------------------------------------------------------------------------
# Two families
# ---------------------------------------------------------------------------
#
#   volonly   volatility + its lags, nothing else.   2(p+1) features.
#             Isolates whether volatility's own history carries information,
#             undiluted. This is the clean test.
#
#   rawvol    all 22 raw + volatility's lags only.   22 + 2p features.
#             The deployable question: does volatility history add to the model
#             you would actually run? Opt in with FAMILY=rawvol.
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Volatility's own autocorrelation at lag 1 is 0.94, so lag 1 is close to a
# duplicate column and should do little. If anything appears it should be at
# larger p, where volatility_t and volatility_{t-p} have separated -- a lagged
# volatility difference is a volatility-of-volatility term, which is a real
# market-structure quantity the feature set does not contain.
#
# Expect flat. Thirteen feature blocks have been tried on this panel and none has
# beaten its baseline by a margin ~26 effective independent windows can support.
# A null here is the finding, and the report states the interval.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/volatility_lags_crypto.sh
#   FAMILY=rawvol bash experiments/volatility_lags_crypto.sh
#   DRY_RUN=1 / SCOUT=1 / ONLY=3 / SHARD=1 OF=3 / NO_MAIL=1
#
# ~7.5 min per arm measured, 10 arms, so about 75 minutes. Mails on success and
# on failure.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# VOL_-prefixed. GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, so a bare WINDOW="${WINDOW:-...}" silently becomes 0 and
# every training block comes out empty.
VOL_MODEL="${VOL_MODEL:-ridge_glmnet}"
VOL_TASK="${VOL_TASK:-regression}"
VOL_METRIC="${VOL_METRIC:-corr}"
VOL_SCHEME="${VOL_SCHEME:-rolling}"
VOL_TRAIN_WINDOW="${VOL_TRAIN_WINDOW:-130}"
VOL_REFIT="${VOL_REFIT:-130}"
VOL_EMBARGO="${VOL_EMBARGO:-20}"
VOL_N_ITER="${VOL_N_ITER:-50}"
VOL_N_SPLITS="${VOL_N_SPLITS:-5}"
VOL_SEED="${VOL_SEED:-42}"
VOL_VAL_FRAC="${VOL_VAL_FRAC:-0.3}"
VOL_SEARCH_JOBS="${VOL_SEARCH_JOBS:-8}"
VOL_N_JOBS="${VOL_N_JOBS:-1}"

case "${VOL_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "VOL_TRAIN_WINDOW=${VOL_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

FAMILY="${FAMILY:-volonly}"
case "${FAMILY}" in
  volonly)
    # Matches feature_volatility_20d and feature_volatility_20d_lag3 alike.
    INCLUDE='^feature_volatility_'
    BASE_RUN="w6r6_grp_volatility_glmnet"; BASE_MODEL="ridge_glmnet"
    PREFIX="volcum" ;;
  rawvol)
    # Everything without a _lag suffix (the 22 raw), plus volatility's lags only.
    INCLUDE='^feature_(?!.*_lag[0-9]+$)|^feature_volatility_.*_lag[0-9]+$'
    BASE_RUN="w6r6_glmnet_reg"; BASE_MODEL="ridge_glmnet"
    PREFIX="rawvolcum" ;;
  *) echo "unknown FAMILY='${FAMILY}' (volonly | rawvol)" >&2; exit 1 ;;
esac

LAGS=(1 2 3 4 5 6 7 8 9 10)

extra=(); suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="_scout"
    echo "SCOUT MODE: --era-step 4. Wiring check only; NOT comparable with the"
    echo "full-resolution baseline, because the split boundary moves."
fi

REPORT="logs/${PREFIX}_report${suffix}.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }

notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "volatility lags (${FAMILY}) -- ABORTED (exit ${code})"
      fi' EXIT

# Gate on the exact width each arm should get. A silent count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${FAMILY}" "${INCLUDE}" "${LAGS[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
from src.prediction.functions import filter_features
family, include = sys.argv[1], sys.argv[2]
for arg in sys.argv[3:]:
    p = int(arg)
    names = pq.ParquetFile(
        "data/crypto/v2.0-cumlag%d/train.parquet" % p).schema.names
    features = [c for c in names if c.startswith("feature_")]
    kept = filter_features(features, include, None, verbose=False)
    expected = 2 * (p + 1) if family == "volonly" else 22 + 2 * p
    if len(kept) != expected:
        sys.exit("cumlag%d %s selects %d features, expected %d"
                 % (p, family, len(kept), expected))
    if family == "volonly" and not all("volatility" in c for c in kept):
        sys.exit("cumlag%d volonly selected a non-volatility column" % p)
print("feature gate OK: %s, %d arms, widths %s"
      % (family, len(sys.argv) - 3,
         "2(p+1)" if family == "volonly" else "22+2p"))
PY

total="${#LAGS[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

say "=============================================================="
say "cumulative volatility lags | family=${FAMILY} | ${total} arms | shard ${SHARD}/${OF}"
say "baseline ${BASE_RUN}"
say "model=${VOL_MODEL} scheme=${VOL_SCHEME} window=${VOL_TRAIN_WINDOW} refit=${VOL_REFIT} embargo=${VOL_EMBARGO}"
say "include=${INCLUDE}"
say "started $(date -Is)"
say "=============================================================="

for p in "${LAGS[@]}"; do
    index=$((index + 1))
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${p}" ] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    run_name="w6r6_${PREFIX}${p}${suffix}"
    cmd=(python3 -u -m src.run_forecast crypto "${VOL_MODEL}"
         --task "${VOL_TASK}" --metric "${VOL_METRIC}" --bmc-reference benchmark
         --cv-scheme "${VOL_SCHEME}" --train-window "${VOL_TRAIN_WINDOW}"
         --refit-every "${VOL_REFIT}" --embargo "${VOL_EMBARGO}"
         --n-splits "${VOL_N_SPLITS}" --n-iter "${VOL_N_ITER}"
         --seed "${VOL_SEED}" --val-frac "${VOL_VAL_FRAC}"
         --n-jobs "${VOL_N_JOBS}" --search-jobs "${VOL_SEARCH_JOBS}"
         --data-version "crypto/v2.0-cumlag${p}"
         --feature-include "${INCLUDE}"
         --run-name "${run_name}")
    cmd+=("${extra[@]+"${extra[@]}"}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] p=%-2s %s\n' "$index" "$total" "$p" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log" >/dev/null
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%2d/%2d] p=%-2s FAILED (exit %d) after %ds' "$index" "$total" "$p" "$status" "$elapsed")"
    else
        say "$(printf '[%2d/%2d] p=%-2s done in %4ds' "$index" "$total" "$p" "$elapsed")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""

python3 -u - "${PREFIX}" "${BASE_MODEL}" "${BASE_RUN}" "${suffix}" >>"${REPORT}" 2>&1 <<'PY'
import json, os, sys
import numpy as np, pandas as pd
prefix, base_model, base_run, suffix = sys.argv[1:5]
PP, MET = "results/per_period/crypto", "results/metrics/crypto"

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    se = np.sqrt(max(s, 0.0) / n)
    return mu, (mu / se if se > 0 else np.nan)

bp = os.path.join(PP, base_model, base_run + ".csv")
if not os.path.exists(bp) or suffix:
    print("no full-resolution baseline for this mode; absolute scores only")
    raise SystemExit(0)
base = pd.read_csv(bp).set_index("date")
base_pay = 0.1 * base["corr"] + base["mmc"]
bm = json.load(open(os.path.join(MET, base_model, base_run + ".json")))

print("%-9s %5s %10s %10s %10s %8s %8s %11s %7s" % (
    "arm", "feat", "CORR", "MMC", "payout", "c.shp", "c.dd", "d payout", "t"))
print("%-9s %5d %10.6f %10.6f %10.6f %8.4f %8.4f %11s %7s" % (
    "baseline", bm["config"]["n_features"], bm["corr"]["mean"], bm["mmc"]["mean"],
    base_pay.mean(), bm["corr"]["sharpe"], bm["corr"]["max_drawdown"], "-", "-"))

rows = []
for p in range(1, 11):
    run = "w6r6_%s%d" % (prefix, p)
    mp = os.path.join(MET, "ridge_glmnet", run + ".json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp))
    t = pd.read_csv(os.path.join(PP, "ridge_glmnet", run + ".csv")).set_index("date")
    pay = 0.1 * t["corr"] + t["mmc"]
    j = pd.concat([base_pay, pay], axis=1, join="inner", keys=["b", "a"])
    d, tstat = nw(j["a"] - j["b"])
    rows.append((p, d, tstat))
    print("%-9s %5d %10.6f %10.6f %10.6f %8.4f %8.4f %+11.6f %7.2f" % (
        "p=%d" % p, m["config"]["n_features"], m["corr"]["mean"], m["mmc"]["mean"],
        pay.mean(), m["corr"]["sharpe"], m["corr"]["max_drawdown"], d, tstat))

if rows:
    strong = [r for r in rows if abs(r[2]) > 2]
    print("\n%d of %d arms differ from the baseline at |t| > 2" % (len(strong), len(rows)))
    if not strong:
        lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
        print("None does. Payout deltas span %+.6f to %+.6f, inside the noise of "
              "~26 effective independent windows. Volatility's own lagged history "
              "adds nothing detectable." % (lo, hi))
    else:
        for p, d, t in strong:
            print("  p=%d: %+.6f at t=%.2f" % (p, d, t))
PY

trap - EXIT
if [ "${failed}" -eq 0 ]; then
    notify "volatility lags (${FAMILY}) -- OK (${ran}/${total})"
else
    notify "volatility lags (${FAMILY}) -- ${failed} arm(s) failed"
fi
echo
echo "report: ${REPORT}"
