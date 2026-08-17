#!/usr/bin/env bash
#
# Experiment: each GROUP of raw features on its own, full pseudo-OOS
#
# Seven groups partition the 22 shipped features. Each is run as the ENTIRE
# feature set -- not raw+group, just the group -- through the complete
# walk-forward: rolling 6-month window, 6-month refit, purged/embargoed inner CV,
# 50-draw randomised search, seed 42. Same protocol as every other arm, so these
# sit directly alongside the existing results.
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
# The grouping is not arbitrary: it is what hierarchical clustering on the
# within-date correlation matrix returns at a 0.30 cut, and it happens to agree
# with the semantic reading. The clusters are tight -- price 0.985, size 0.947,
# liquidity 0.935, rsi/sharpe 0.894-0.924 mean |r| -- which is the point. Twelve
# of the 22 raw features carry roughly three features' worth of information, and
# running the groups separately is how you measure that rather than assert it.
#
# BOTH ridge variants run for every group, because "the best ridge" is not the
# same model at every width:
#
#   ridge         fixed alpha grid, logspace(-4, 2, 100). Capped at 100.
#   ridge_glmnet  data-scaled path, alpha = lambda * N, rebuilt from each refit's
#                 own training block. On 39,000 rows its alphas run orders of
#                 magnitude above the fixed grid's ceiling.
#
# On the full 22 features the fixed grid won on payout (0.085763 vs 0.083859)
# while glmnet won on risk (sharpe 1.81 vs 1.54). With 2 features instead of 22
# there is far less collinearity to shrink, so the ranking may well invert --
# which is itself worth knowing.
#
# ---------------------------------------------------------------------------
# What to compare against
# ---------------------------------------------------------------------------
#
# The all-22 reference arms already exist and should NOT be re-run:
#
#     w6r6_glmnet_reg   22 raw, ridge_glmnet   payout 0.083859  sharpe 1.8093
#     w6r6_fixed_reg    22 raw, ridge          payout 0.085763  sharpe 1.5424
#
# A group scoring close to those on 2-4 features would say the other 18-20
# columns are decoration. A group scoring near zero says it carries nothing on
# its own -- which does not mean it contributes nothing in combination, and the
# lobo family answers that other half.
#
# Prediction to check this against: momentum should be the strongest small group
# (its 20d/60d correlation is 0.368, the most distinct in the set) and price the
# weakest per column (0.985 between its own horizons -- four columns of one
# thing, and a rank-binned price level is close to a constant within a date).
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/feature_groups_crypto.sh
#   DRY_RUN=1 bash experiments/feature_groups_crypto.sh
#   ONLY=momentum bash experiments/feature_groups_crypto.sh
#   MODELS="ridge_glmnet" bash experiments/feature_groups_crypto.sh   # one variant
#   SHARD=1 OF=2 bash experiments/feature_groups_crypto.sh            # split
#
# Under screen:
#   screen -L -Logfile feature-groups.txt bash experiments/feature_groups_crypto.sh
#
# 14 runs on 2-4 features each; expect ~3-5 min per run, well under an hour total.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# GRP_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
GRP_TASK="${GRP_TASK:-regression}"
GRP_SCHEME="${GRP_SCHEME:-rolling}"
GRP_TRAIN_WINDOW="${GRP_TRAIN_WINDOW:-6month}"
GRP_REFIT="${GRP_REFIT:-6month}"
GRP_N_ITER="${GRP_N_ITER:-50}"
GRP_N_SPLITS="${GRP_N_SPLITS:-5}"
GRP_N_JOBS="${GRP_N_JOBS:-8}"
GRP_SEED="${GRP_SEED:-42}"

case "${GRP_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "GRP_TRAIN_WINDOW=${GRP_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

# The 22-feature shipped dataset. Groups are carved out of it with
# --feature-include, so nothing new is built.
DATA="${GRP_DATA:-crypto/v2.0}"
MODELS="${MODELS:-ridge_glmnet ridge}"

# Parallel arrays, not packed "name|regex" strings: several of these regexes
# contain "|" as alternation, which would be torn apart by IFS='|' splitting.
GROUP_NAME=(); GROUP_RE=()
add_group() { GROUP_NAME+=("$1"); GROUP_RE+=("$2"); }

add_group price      '^feature_close_(avg|ewa)_'
add_group size       '^feature_market_cap_(avg|ewa)_'
add_group liquidity  '^feature_volume_(avg|ewa)_'
add_group oscillator '^feature_(relative_strength_index|sharpe_ratio)_'
add_group bollinger  '^feature_bollinger_'
add_group momentum   '^feature_momentum_'
add_group volatility '^feature_volatility_'

# Refuse to run if the groups do not exactly partition the 22 features. A silent
# gap or overlap would make every number here describe a different experiment
# than the one documented above.
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

total=$(( ${#GROUP_NAME[@]} * $(echo "${MODELS}" | wc -w) ))
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "feature groups  |  ${#GROUP_NAME[@]} groups x $(echo "${MODELS}" | wc -w) models = ${total} runs"
echo "data=${DATA}  window=${GRP_TRAIN_WINDOW}  refit=${GRP_REFIT}  shard ${SHARD}/${OF}"
echo "=============================================================="

for i in "${!GROUP_NAME[@]}"; do
    group="${GROUP_NAME[$i]}"; pattern="${GROUP_RE[$i]}"

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${group}" ] && continue

    for model in ${MODELS}; do
        index=$((index + 1))
        # Round-robin so a shard gets a mix of groups rather than one model's runs.
        [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

        # ridge_glmnet -> "glmnet", ridge -> "fixed", matching the naming the
        # earlier task x grid comparison used.
        tag="fixed"; [ "${model}" = "ridge_glmnet" ] && tag="glmnet"
        run_name="w6r6_grp_${group}_${tag}"

        cmd=(python3 -u -m src.run_forecast crypto "${model}"
             --task "${GRP_TASK}"
             --cv-scheme "${GRP_SCHEME}" --train-window "${GRP_TRAIN_WINDOW}"
             --refit-every "${GRP_REFIT}" --data-version "${DATA}"
             --feature-include "${pattern}"
             --n-iter "${GRP_N_ITER}" --n-splits "${GRP_N_SPLITS}"
             --n-jobs "${GRP_N_JOBS}" --seed "${GRP_SEED}"
             --run-name "${run_name}")

        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '[%2d/%2d] %-26s %s\n' "$index" "$total" "$run_name" "${cmd[*]}"
            continue
        fi

        ran=$((ran + 1))
        echo
        echo "=============================================================="
        echo "[${index}/${total}] ${run_name}   $(date '+%H:%M:%S')"
        echo "=============================================================="

        started=$(date +%s)
        # A failed run must not cost you the others.
        "${cmd[@]}" 2>&1 | tee "logs/grp_${run_name}.log"
        status="${PIPESTATUS[0]}"
        elapsed=$(( $(date +%s) - started ))

        if [ "$status" -ne 0 ]; then
            failed=$((failed + 1))
            echo ">>> ${run_name} FAILED (exit ${status}) after ${elapsed}s -- continuing"
        else
            echo ">>> ${run_name} done in ${elapsed}s"
        fi
    done
done

echo
echo "=============================================================="
echo "ran ${ran} of ${total} runs in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo
echo "compare, best ridge per group, against the all-22 references:"
cat <<'REPORT'
  python3 - <<'PY'
import json, glob, os
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/*/w6r6_grp_*.json")) + [
        "results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg.json",
        "results/metrics/crypto/ridge/w6r6_fixed_reg.json"]:
    if not os.path.exists(p): continue
    d=json.load(open(p)); n=os.path.basename(p)[:-5]
    rows.append((0.1*d["corr"]["mean"]+d["mmc"]["mean"], n, d["config"]["n_features"],
                 d["corr"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"]))
print("%-28s %5s %10s %9s %9s %10s" % ("run","feat","CORR","sharpe","maxdd","payout"))
for pay,n,f,c,s,dd in sorted(rows, reverse=True):
    print("%-28s %5d %10.6f %9.4f %9.4f %10.6f" % (n,f,c,s,dd,pay))
PY
REPORT
echo "=============================================================="
