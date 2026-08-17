#!/usr/bin/env bash
#
# Experiment: tune the four surviving weighting schemes
#
# The first sweep ran each scheme at ONE parameter setting -- my defaults, not
# tuned. Every scheme lost to 1/N, four of them significantly. These four lost
# by the least, so the question this answers is narrow and worth answering:
#
#     scheme               d payout vs 1/N      t
#     shrunk_tangency          -0.003300    -1.89
#     inverse_variance         -0.004266    -1.42
#     granger_ramanathan       -0.004439    -1.54
#     complete_subset          -0.006216    -2.21
#
# Does any of them beat 1/N at a BETTER setting, or is the loss structural?
#
# ---------------------------------------------------------------------------
# What is swept, and why each knob exists
# ---------------------------------------------------------------------------
#
#   update_every   22 (monthly), 65 (quarterly), 130 (matching the model refit).
#                  Weights are cheap to re-estimate, but re-estimating often on
#                  a ~25-observation sample means tracking noise. This is the
#                  bias-variance knob of the whole exercise.
#
#   window         0 (expanding) vs 260 (rolling ~1 year). Expanding uses every
#                  resolved period; rolling assumes the ranking of members
#                  drifts. The tangency weights flipped signs between thirds of
#                  this sample, which is evidence for drift OR for noise, and
#                  those two look identical at this sample size.
#
#   shrinkage      shrunk_tangency: 0.1 / 0.2 / 0.35 / 0.5. How much the
#                  covariance is trusted. 0 IS equal weighting, so this traces
#                  the whole path from 1/N to raw tangency.
#
#   subset_size    complete_subset: 2 / 3 / 4 members per subset.
#   keep_fraction  complete_subset: 0.3 / 0.5 / 1.0. At 1.0 every subset is kept
#                  and the scheme reduces exactly to 1/N -- a built-in control
#                  that should reproduce the baseline to six decimals.
#
#   constrained    granger_ramanathan: with and without sum-to-one plus
#                  non-negativity. The unconstrained variant is what produced
#                  the sign-flipping weights; it is included so that is a
#                  measurement rather than an assertion.
#
# inverse_variance has no parameters of its own -- it only sees cadence and
# window, which is precisely why it is the most robust of the four.
#
# ---------------------------------------------------------------------------
# The control that matters
# ---------------------------------------------------------------------------
#
# `equal` is re-run at every cadence and window. It should score identically at
# all of them, because 1/N does not depend on history. If it does not, the
# harness is not neutral and no comparison in this sweep means anything.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/weighting_sweep_crypto.sh
#   DRY_RUN=1 bash experiments/weighting_sweep_crypto.sh
#   ONLY=shrunk_tangency bash experiments/weighting_sweep_crypto.sh
#   SHARD=1 OF=3 bash experiments/weighting_sweep_crypto.sh
#
# Under screen:
#   screen -L -Logfile weighting-sweep.txt bash experiments/weighting_sweep_crypto.sh
#
# ~70s per configuration, 46 configurations, so about an hour single-threaded.
# Nothing is refitted; every run reads the members' existing forecast CSVs.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# SWP_-prefixed: GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, and "window" is a real parameter of this experiment.
SWP_LAG="${SWP_LAG:-20}"
SWP_MIN_HISTORY="${SWP_MIN_HISTORY:-60}"
SWP_OBJECTIVE="${SWP_OBJECTIVE:-sharpe}"
SWP_SEED="${SWP_SEED:-42}"

if [ "${SWP_LAG}" -lt 20 ]; then
    echo "SWP_LAG=${SWP_LAG} is below the 20-day target horizon; weights would" \
         "read scores that had not resolved. Worth ~0.006 payout of pure leakage." >&2
    [ "${ALLOW_LEAKY_LAG:-0}" = "1" ] || exit 1
fi

G="ridge_glmnet"
MEMBERS=(
  "${G}/w6r6_alone_onehot"
  "${G}/w6r6_alone_tenc"
  "${G}/w6r6_alone_interact"
  "ridge/w6r6_grp_size_fixed"
  "${G}/w6r6_grp_price_glmnet"
)

# Parallel arrays -- name, scheme, and the extra flags for that configuration.
NAME=(); SCHEME=(); EXTRA=()
add() { NAME+=("$1"); SCHEME+=("$2"); EXTRA+=("$3"); }

for cadence in 22 65 130; do
    for window in 0 260; do
        tag="c${cadence}w${window}"
        common="--update-every ${cadence} --window ${window}"

        # The neutrality control.
        add "equal_${tag}"            equal              "${common}"
        add "invvar_${tag}"           inverse_variance   "${common}"

        for lam in 0.1 0.2 0.35 0.5; do
            add "shrunk_${tag}_l${lam/./}" shrunk_tangency "${common} --shrinkage ${lam}"
        done
    done
done

# complete_subset and granger_ramanathan are swept at the monthly cadence on an
# expanding window only -- the grid above already shows how much cadence and
# window move things, and crossing everything would be 200+ runs for a scheme
# that lost by 0.006.
for size in 2 3 4; do
    for keep in 0.3 0.5 1.0; do
        add "subset_s${size}_k${keep/./}" complete_subset \
            "--update-every 22 --window 0 --subset-size ${size} --keep-fraction ${keep}"
    done
done
add "gr_constrained"   granger_ramanathan "--update-every 22 --window 0"
add "gr_unconstrained" granger_ramanathan "--update-every 22 --window 0 --gr-unconstrained"

total="${#NAME[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "weighting sweep | ${total} configurations | shard ${SHARD}/${OF}"
echo "lag=${SWP_LAG}  min_history=${SWP_MIN_HISTORY}  objective=${SWP_OBJECTIVE}"
echo "=============================================================="

for i in "${!NAME[@]}"; do
    name="${NAME[$i]}"; scheme="${SCHEME[$i]}"; extra="${EXTRA[$i]}"
    index=$((index + 1))

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${scheme}" ] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    run_name="w6r6_swp_${name}"
    cmd=(python3 -u -m src.run_weighted_ensemble crypto ridge_glmnet
         "${MEMBERS[@]}" --scheme "${scheme}"
         --lag "${SWP_LAG}" --min-history "${SWP_MIN_HISTORY}"
         --objective "${SWP_OBJECTIVE}" --seed "${SWP_SEED}"
         --run-name "${run_name}" -q)
    # shellcheck disable=SC2206
    cmd+=(${extra})

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-24s %s %s\n' "$index" "$total" "$name" "$scheme" "$extra"
        continue
    fi

    ran=$((ran + 1))
    started=$(date +%s)
    "${cmd[@]}" >"logs/swp_${name}.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        printf '[%2d/%2d] %-24s FAILED (exit %d)\n' "$index" "$total" "$name" "$status"
    else
        printf '[%2d/%2d] %-24s done in %3ds\n' "$index" "$total" "$name" \
               "$(( $(date +%s) - started ))"
    fi
done

echo
echo "=============================================================="
echo "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo "report with:  python3 -m src.analysis.weighting_report"
echo "=============================================================="
