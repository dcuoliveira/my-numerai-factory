#!/usr/bin/env bash
#
# E26 -- SIGN-CONSTRAINED Generalised Ridge on allfeat_pp: all betas >= 0, and
#        all betas <= 0, each searched from scratch against a free control.
#
# E26 and not E23: a concurrent session in this checkout took E23, E24 and E25
# while this was being built (cross-class blends, TabPFN tiers, weighted
# combiners). Adding a duplicate key to the STUDIES dict literal deletes the
# earlier one silently -- Python keeps the last and raises nothing -- so the
# number is claimed by re-reading the registry immediately before writing.
#
# A sign constraint is the one thing in the GRR that is not a change of penalty:
# it leaves the objective alone and shrinks the FEASIBLE SET, turning the closed
# form into a box-constrained quadratic program. Solved by coordinate descent
# warm-started from the unconstrained fit, with optimality judged on the KKT
# conditions and a raise -- never a returned iterate -- if they are not met.
#
# ---------------------------------------------------------------------------
# Both constraints BIND, which is what makes the experiment worth running
# ---------------------------------------------------------------------------
#
# Measured on the last 130 dates before launching, 394 live columns of 812:
#
#     alpha     free signs        all s+ binds      all s- binds
#     0.01      206 pos / 188 neg   139 / 812         87 / 812
#     1.0       176 pos / 218 neg   203 / 812        172 / 812
#     24.5      180 pos / 214 neg   217 / 812        179 / 812
#
# So neither arm is a no-op wearing a constraint's clothes -- the trap that
# sank an earlier design, where `feature_volatility:s-` turned out bit-exactly
# identical to its control because the free coefficients were already negative.
# Here the unconstrained solution is roughly half of each sign at every alpha,
# so each constraint removes real freedom, and n_binding is recorded per refit
# so a reader can check that claim rather than take it.
#
# Note the all-NEGATIVE arm is not the mirror of the all-positive one. The
# objective is not symmetric in b, and the intercept is unpenalised and free, so
# `all s-` keeps the 188-218 naturally negative loadings and zeroes the rest.
# In-sample it scored +0.3614 against `all s+`'s +0.3364 at alpha 0.01: a
# non-positive combination of these features predicts perfectly well.
#
# ---------------------------------------------------------------------------
# Why the alpha grid is floored at 1e-2 in EVERY arm
# ---------------------------------------------------------------------------
#
# The shipped grid is logspace(-4, 2). On this dataset E21 measured the search
# putting alpha on the 1e-4 floor at 16 of 44 refits -- an essentially
# unregularised fit on 812 columns. A sign-constrained solve there does not
# reach optimality in 2000 sweeps: it spends 17 seconds and raises.
#
# That raise is honest, but it would silently truncate the grid from below for
# the constrained arms and not for the control, so the comparison would confound
# the constraint with a different search space. Flooring at 1e-2 for all three
# arms keeps them searching the SAME 60 points, and removes a regime E21 already
# identified as an artefact. It is the one departure from the default and it is
# applied identically everywhere.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash experiments/e26_sign_constraint_crypto.sh
#   DRY_RUN=1 bash experiments/e26_sign_constraint_crypto.sh
#   ONLY=e26_pos bash experiments/e26_sign_constraint_crypto.sh
#   NO_MAIL=1 bash experiments/e26_sign_constraint_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

SC_MODEL="${SC_MODEL:-gen_ridge}"
SC_SCHEME="${SC_SCHEME:-rolling}"
SC_TRAIN_WINDOW="${SC_TRAIN_WINDOW:-130}"
SC_REFIT="${SC_REFIT:-130}"
SC_EMBARGO="${SC_EMBARGO:-20}"
SC_N_ITER="${SC_N_ITER:-50}"
SC_N_SPLITS="${SC_N_SPLITS:-5}"
SC_SEED="${SC_SEED:-42}"
SC_VAL_FRAC="${SC_VAL_FRAC:-0.3}"
SC_SEARCH_JOBS="${SC_SEARCH_JOBS:-4}"
SC_N_JOBS="${SC_N_JOBS:-1}"
DATA="${SC_DATA:-crypto/v2.0-tenc-onehot-rollsel-spread-interact}"

case "${SC_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "SC_TRAIN_WINDOW=${SC_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

# Every block of allfeat_pp named explicitly, so the constraint reaches ALL 812
# columns. Naming a block the data does not have raises, so this doubles as an
# assertion that the partition is what the study thinks it is.
BLOCKS="raw spread interact rollsel tenc onehot"
POS=""; NEG=""
for b in ${BLOCKS}; do POS="${POS}${b}:s+;"; NEG="${NEG}${b}:s-;"; done
POS="${POS%;}"; NEG="${NEG%;}"

FLOOR='"alpha_lo":-2,"alpha_hi":2'
NAMES=(); OPTS=(); WHY=()
add() { NAMES+=("$1"); OPTS+=("$2"); WHY+=("$3"); }
add e26_free "{\"structures\":[\"identity\"],${FLOOR}}"                              "control: betas unconstrained"
add e26_pos  "{\"structures\":[\"identity\"],${FLOOR},\"group_specs\":[\"${POS}\"]}"  "ALL betas >= 0"
add e26_neg  "{\"structures\":[\"identity\"],${FLOOR},\"group_specs\":[\"${NEG}\"]}"  "ALL betas <= 0"

REPORT="logs/e26_report.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
notify() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "E26 sign constraints -- ABORTED (exit ${code})"
      fi' EXIT

python3 - "${DATA}" "${POS}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
from src.prediction.generalised_ridge import block_index, parse_group_spec
names = [c for c in pq.ParquetFile("data/%s/train.parquet" % sys.argv[1]).schema.names
         if c.startswith("feature_")]
blocks = block_index(names)
spec = parse_group_spec(sys.argv[2])
missing = sorted(set(blocks) - set(spec))
if missing:
    sys.exit("the spec does not constrain every block; %s left free" % missing)
unknown = sorted(set(spec) - set(blocks))
if unknown:
    sys.exit("the spec names blocks the data does not have: %s" % unknown)
covered = sum(len(v) for v in blocks.values())
print("gate OK: %d features, %d blocks, ALL %d columns constrained  %s"
      % (len(names), len(blocks), covered, {k: len(v) for k, v in blocks.items()}))
PY

total="${#NAMES[@]}"; index=0; failed=0; ran=0
started_all=$(date +%s)
say "=============================================================="
say "E26 sign-constrained GRR on allfeat_pp | ${total} arms"
say "all 812 columns constrained; alpha floored at 1e-2 in EVERY arm"
say "model=${SC_MODEL} scheme=${SC_SCHEME} window=${SC_TRAIN_WINDOW} refit=${SC_REFIT}"
say "n-iter=${SC_N_ITER} n-splits=${SC_N_SPLITS} seed=${SC_SEED} search-jobs=${SC_SEARCH_JOBS}"
say "started $(date -Is)"
say "=============================================================="

for i in "${!NAMES[@]}"; do
    index=$((index + 1))
    name="${NAMES[$i]}"; opts="${OPTS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    cmd=(python3 -u -m src.run_forecast crypto "${SC_MODEL}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme "${SC_SCHEME}" --train-window "${SC_TRAIN_WINDOW}"
         --refit-every "${SC_REFIT}" --embargo "${SC_EMBARGO}"
         --n-splits "${SC_N_SPLITS}" --n-iter "${SC_N_ITER}"
         --seed "${SC_SEED}" --val-frac "${SC_VAL_FRAC}"
         --n-jobs "${SC_N_JOBS}" --search-jobs "${SC_SEARCH_JOBS}"
         --model-options "${opts}"
         --data-version "${DATA}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-10s %-28s %s\n' "$index" "$total" "$name" "$why" "$opts"
        continue
    fi
    ran=$((ran + 1)); started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log" >/dev/null
    status="${PIPESTATUS[0]}"; elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        say "$(printf '[%d/%d] %-10s FAILED (exit %d) after %5ds  -- %s' \
             "$index" "$total" "$name" "$status" "$elapsed" "$why")"
    else
        say "$(printf '[%d/%d] %-10s done in %5ds  -- %s' "$index" "$total" "$name" "$elapsed" "$why")"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }
say ""
say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
say ""
python3 -u -m src.analysis.e26_report >>"${REPORT}" 2>&1 || \
    say "(report generation failed; the runs are in results/)"
trap - EXIT
if [ "${failed}" -eq 0 ]; then notify "E26 sign constraints -- OK (${ran}/${total})"
else notify "E26 sign constraints -- ${failed} arm(s) failed"; fi
echo
echo "report: ${REPORT}"
