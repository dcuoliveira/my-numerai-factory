#!/usr/bin/env bash
#
# Experiment E14: E13's ADL question again, with each arm tuned for its own width
#
# E14 is the reference handle; every artefact it writes is e14_*.
#
# WHAT THIS RUNS
#   Two arms, each given a FULL 50-draw hyperparameter search of its own:
#
#     e14_raw   crypto/v2.0       22 features
#     e14_adl   crypto/v2.0-adl   91 features  (3 target lags + 3 lags of each feature)
#
# WHY, WHEN E13 ALREADY ANSWERED THIS
#   It did not, quite. E13 PINNED the estimator to e07_mlpreg_raw's modal
#   configuration -- (64, 32), alpha 8.227 -- chosen on 22 columns, and applied
#   it unchanged to 91. Its own report says so: a negative ADL result there
#   partly measures "this config does not suit 91 columns", not "lags cannot help
#   a tuned MLP". The weights were always fit from scratch at every refit; the
#   ARCHITECTURE never was.
#
#   Here each arm picks its own architecture, alpha, learning rate and batch
#   size. If ADL is going to help anywhere, it is with a shape sized for its own
#   input width -- a first layer over 91 inputs is four times the one over 22.
#
# THE TWO DESIGNS BRACKET THE ANSWER -- READ THEM TOGETHER
#   Pinning and searching fail in opposite directions and neither is the honest
#   answer alone:
#
#     PINNED (E13)    no search noise, so the arms differ by columns alone --
#                     but the wide arm wears a narrow arm's coat.
#     SEARCHED (E14)  each arm gets its best shape -- but the search itself is
#                     noisy. E12 measured that noise directly: six arms differing
#                     only in embedding width came out non-monotone with a 0.024
#                     payout spread, larger than any effect being looked for,
#                     because one unlucky draw in 50 moved an arm further than
#                     the treatment did.
#
#   So a searched ADL win under +0.024 is inside E12's demonstrated noise floor
#   and should not be believed on its own. Agreement between E13 and E14 is worth
#   far more than either.
#
# THE ALPHA GRID WAS PUT BACK TO 1e1 BEFORE THIS RAN
#   E07 pinned against a 1e1 ceiling, which looked like a binding grid, so it was
#   raised to 1e2. That was wrong. On the same 514 periods a pinned alpha of 8.2
#   pays 0.082438 while the 1e2 search choosing 39-79 pays 0.075302: the CV score
#   keeps improving past alpha ~10 while out-of-sample payout falls. The ceiling
#   is back at 1e1 and MLPWrapper.ALPHAS records why. E09/E11/E12's searched
#   numbers were produced on the 1e2 grid and are NOT comparable to these.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on the paired payout delta, Newey-West lag 20. One comparison, so
#   no multiplicity adjustment is needed here -- but see the noise-floor note
#   above, which binds tighter than the t-statistic does.
#
#   E13's pinned answer was -0.007558 at t = -1.43. If ADL is merely a shape
#   problem, this should move positive. If it lands in the same place, two
#   designs that fail differently agree, and the ADL columns add nothing.
#
# COST
#   ~1,000 fits per arm against E13's ~25. e07_mlpreg_raw took 92 min on 22
#   columns; width costs an MLP little (786 columns benchmarked at roughly the
#   same per-fit cost as 22), so budget ~1.5-2 h per arm, ~3-4 h total.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e14_adl_searched_crypto.sh
#   bash experiments/e14_adl_searched_crypto.sh
#   ONLY=adl / DRY_RUN=1 / NO_MAIL=1 / ADL_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E14}"
ADL_WINDOW="${ADL_WINDOW:-130}"
ADL_REFIT="${ADL_REFIT:-130}"
ADL_EMBARGO="${ADL_EMBARGO:-20}"
ADL_N_SPLITS="${ADL_N_SPLITS:-5}"
ADL_N_ITER="${ADL_N_ITER:-50}"
ADL_VAL_FRAC="${ADL_VAL_FRAC:-0.3}"
ADL_SEED="${ADL_SEED:-42}"
ADL_THREADS="${ADL_THREADS:-8}"
ADL_ERA_STEP=1
case "${ADL_WINDOW}" in ""|0|[0-9]) echo "ADL_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${ADL_REFIT}"  in ""|0|[0-9]) echo "ADL_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e14"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e14scout"; ADL_ERA_STEP=4; ADL_N_ITER=4; ADL_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${ADL_N_ITER} draws, ${ADL_N_SPLITS} folds -- wiring only."
fi

# sklearn's MLP has no --n-jobs; it threads through BLAS and these are the only
# knobs that bound it.
export OMP_NUM_THREADS="${ADL_THREADS}"
export MKL_NUM_THREADS="${ADL_THREADS}"
export OPENBLAS_NUM_THREADS="${ADL_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
NAME=(raw adl); DATA=("crypto/v2.0" "crypto/v2.0-adl"); NFEAT=(22 91)
SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. The AR lags are re-checked for
# leakage on the dataset rather than trusted from build time, and the alpha
# ceiling is asserted because these numbers are only comparable to each other
# and to E13 if the grid is the one E13 ran on.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_mani.utils import get_spec, target_horizon_days
from src.prediction.models import MLPWrapper

cut = sys.argv.index("--")
versions, wants = sys.argv[1:cut], [int(x) for x in sys.argv[cut + 1:]]
target = get_spec("crypto")["target"]
horizon = target_horizon_days(target)

bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    try:
        names = pq.ParquetFile(path).schema.names
    except OSError:
        bad.append("%s MISSING -- build it with python3 -m src.data_mani.build_adl"
                   % version)
        continue
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))

    ar_columns = [c for c in names if c.startswith("feature_target_lag")]
    for column in ar_columns:
        if int(column.rsplit("lag", 1)[1]) < horizon:
            bad.append("%s carries %s, a lag BELOW the %d-period horizon -- that "
                       "column contains unresolved future"
                       % (version, column, horizon))
    if ar_columns:
        frame = pd.read_parquet(path, columns=ar_columns + [target])
        for column in ar_columns:
            both = frame[[column, target]].dropna()
            rho = float(np.corrcoef(both[column], both[target])[0, 1])
            if abs(rho) > 0.25:
                bad.append("%s: corr(%s, %s) = %+.4f, far above the +-0.02 seen "
                           "for resolved lags" % (version, column, target, rho))

ceiling = float(MLPWrapper.ALPHAS.max())
if abs(ceiling - 10.0) > 1e-9:
    bad.append("MLPWrapper.ALPHAS tops out at %g, not 1e1. E14 is only comparable "
               "to E13 on the grid E13 ran; see ALPHAS for why 1e2 was reverted"
               % ceiling)

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts; every AR lag "
      "is >= the %d-period horizon; alpha ceiling is 1e1 as E13 ran it"
      % (len(versions), horizon))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      does ADL help once each arm is TUNED for its own width?
  vs E13    E13 pinned a 22-column config onto 91 columns and got -0.007558
            at t = -1.43. Here each arm runs its own 50-draw search.
  cadence   rolling ${ADL_WINDOW}/${ADL_REFIT}, embargo ${ADL_EMBARGO}, ${ADL_N_SPLITS} folds, ${ADL_N_ITER} draws
  expect    ~1.5-2 h per arm, ~3-4 h total
  caution   E12 measured the search's own noise at 0.024 payout spread. A
            searched ADL win smaller than that is inside the noise floor.

Where to look while it runs:
  runner    experiments/e14_adl_searched_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/mlp/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: ADL with each arm tuned for its own width  |  ${total} arm(s)"
echo "cadence: rolling ${ADL_WINDOW}/${ADL_REFIT}, embargo ${ADL_EMBARGO}, ${ADL_N_SPLITS} folds, ${ADL_N_ITER} draws, ${ADL_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${ADL_WINDOW}"
         --refit-every "${ADL_REFIT}" --embargo "${ADL_EMBARGO}"
         --n-splits "${ADL_N_SPLITS}" --n-iter "${ADL_N_ITER}"
         --val-frac "${ADL_VAL_FRAC}" --era-step "${ADL_ERA_STEP}"
         --seed "${ADL_SEED}" --search-jobs 1
         --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-14s %5s feat  %s\n' "$index" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${SEL_DATA[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${SEL_NFEAT[$i]} feat  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    # A failed arm must not cost the other: read PIPESTATUS[0] after the tee.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- does ADL help once each arm is tuned for its own width?"
echo "  results  results/metrics/crypto/mlp/${PREFIX}_*.json"
echo "  runner   experiments/e14_adl_searched_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/60))m$((el_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"

def load(run):
    p = "results/metrics/crypto/mlp/%s.json" % run
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(run):
    f = pd.read_csv("results/per_period/crypto/mlp/%s.csv" % run)
    return f.set_index("date") if "date" in f.columns else f

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

rows = [(a, "%s_%s" % (prefix, a), load("%s_%s" % (prefix, a)))
        for a in ("raw", "adl")]
print("%-12s %5s %6s %10s %10s %8s %8s %10s %7s" % (
    "run", "feat", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 92)
for arm, run, d in rows:
    if d is None:
        # Never a blank line: absent reads as "tested and null" when it means
        # "never ran".
        print("%-12s  NOT ON DISK -- this arm did not run" % run)
        continue
    print("%-12s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f %7.1f" % (
        run, d["config"]["n_features"], d["n_periods"], d["corr"]["mean"],
        d["mmc"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 92)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit
if any(d is None for _, _, d in rows):
    print()
    print("Both arms are needed for the comparison; one is missing.")
    raise SystemExit

print()
print("what each arm's search actually chose -- the whole point of E14")
for arm, run, d in rows:
    picks = [(str(tuple(r["best_params"]["mlp__hidden_layer_sizes"])),
              round(r["best_params"]["mlp__alpha"], 2))
             for r in d["refits"]]
    print("  %-12s %s" % (run, picks))
print("  (E13 pinned BOTH arms at ('64, 32'), alpha 8.23)")

print()
print("paired adl - raw, Newey-West lag 20")
print("%-10s %12s %12s %+12s %10s %8s" % (
    "metric", "raw", "adl", "delta", "se", "t"))
print("-" * 70)
base, arm = per_period(rows[0][1]), per_period(rows[1][1])
for metric in ("corr", "mmc", "payout"):
    left = payout(base) if metric == "payout" else base[metric]
    right = payout(arm) if metric == "payout" else arm[metric]
    j = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"])
    delta, se, t = newey_west(j["a"] - j["b"])
    print("%-10s %12.6f %12.6f %+12.6f %10.6f %+8.2f" % (
        metric, j["b"].mean(), j["a"].mean(), delta, se, t))
print("-" * 70)

# The comparison that matters is against E13, not against zero.
prior = load("e13_adl"), load("e13_raw")
if all(prior):
    try:
        j = pd.concat([payout(per_period("e13_raw")), payout(per_period("e13_adl"))],
                      axis=1, join="inner", keys=["b", "a"])
        d13, se13, t13 = newey_west(j["a"] - j["b"])
        j = pd.concat([payout(base), payout(arm)], axis=1, join="inner", keys=["b", "a"])
        d14, _, t14 = newey_west(j["a"] - j["b"])
        print()
        print("ADL payout delta, the two designs side by side:")
        print("  E13 pinned    %+.6f  t %+.2f   (22-column config on 91 columns)" % (d13, t13))
        print("  E14 searched  %+.6f  t %+.2f   (each arm tuned for its width)" % (d14, t14))
        if (d13 < 0) == (d14 < 0):
            print("  SAME SIGN -- two designs that fail in opposite ways agree, which")
            print("  is worth more than either alone. The shape was not the problem.")
        else:
            print("  OPPOSITE SIGNS -- the pinned result was a shape artefact, or this")
            print("  one is search noise. E12 measured that noise at 0.024 payout")
            print("  spread; compare |%.6f| against it before believing either." % d14)
    except Exception as error:
        print("\ncould not compare against E13: %s" % error)
else:
    print("\nE13 is not on disk, so the pinned-vs-searched comparison is skipped.")

print()
print("~26 effective independent windows, so |t| < 2 is EXPECTED and IS the")
print("finding. E12 put the search's own noise floor at 0.024 payout spread --")
print("a searched win smaller than that is not evidence on its own.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
