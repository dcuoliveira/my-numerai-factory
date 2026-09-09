#!/usr/bin/env bash
#
# Experiment E13: do autoregressive and distributed lags help the MLP?
#
# E13 is the reference handle; every artefact it writes is e13_*.
#
# WHAT THIS RUNS
#   Four arms, the SAME pinned MLP, differing only in which lag columns exist:
#
#     e13_raw   crypto/v2.0       22   contemporaneous features only (control)
#     e13_ar    crypto/v2.0-ar    25   + 3 lags of the TARGET          (AR)
#     e13_dl    crypto/v2.0-dl    88   + 3 lags of every FEATURE       (DL)
#     e13_adl   crypto/v2.0-adl   91   + both                          (ADL)
#
#   An ADL model regresses y_t on its own past AND on past drivers. Splitting it
#   into AR and DL arms is the point: if ADL moves, this says which half did it,
#   and the two halves have completely different priors (below).
#
# THE MODEL IS PINNED, AND THAT IS DELIBERATE
#   --fixed-params holds the estimator at the configuration e07_mlpreg_raw
#   actually selected in 3 of its 4 refits -- (64, 32), alpha 8.227, batch 2048,
#   lr 0.008886 -- the best MLP measured on raw data (payout 0.090428). Nothing
#   is searched, so the arms differ by their columns and by nothing else.
#
#   E12 is why. There, six arms differing only in embedding width came out
#   non-monotone with a 0.024 spread, because one unlucky draw inside a 50-draw
#   search moved an arm further than the thing being measured did. A pinned
#   estimator removes that entirely. It also makes each arm ~25 fits instead of
#   ~1,000, so this study costs minutes rather than half a day.
#
#   The cost of pinning: a configuration chosen for 22 features is not
#   necessarily right for 91, and alpha's per-parameter bite changes with the
#   input width. So a negative DL result partly measures "this config does not
#   suit 88 columns". Say so when reporting; do not upgrade it to "lags do not
#   help a tuned MLP".
#
# THE AR HALF IS THE ONE WORTH RUNNING
#   * A LAG OF THE TARGET SHORTER THAN 20 PERIODS IS LOOKAHEAD.
#     target_binned_return_20 at row t summarises [t, t+20] and is not observable
#     until t+20, so a lag k gives a window that has closed only when k >= 20.
#     build_adl.py REFUSES a shorter lag rather than warning, because such a run
#     completes, scores beautifully, and looks like a discovery. Lags are 20, 40
#     and 60 so the three AR terms are NON-OVERLAPPING past 20-day returns.
#     The builder prints corr(lag, target) as a tripwire: -0.0156, +0.0068,
#     +0.0197 here, i.e. no overlap.
#   * Nothing in this repo has tried an AR term before.
#
#   The DL half has been tried and did not help. E05 swept cumulative feature
#   lags 1..10 on LightGBM and the best depth by payout was ZERO. E06 narrowed it
#   to the volatility group and found lags harmful, t = -2.18 to -2.30. Expect
#   the DL and ADL arms to be flat or negative; that is the prior, not a
#   prediction.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta against e13_raw, Newey-West lag 20. Three
#   comparisons put the Bonferroni-style bar nearer |t| = 2.4; quote both. On
#   ~26 effective independent windows a null is expected and IS the finding.
#   Read maxdd beside payout.
#
# COST
#   Pinned, so roughly 25 fits per arm against ~1,000 for a searched one. Expect
#   single-digit minutes per narrow arm and perhaps 15-20 for the 88/91-column
#   ones. Well under an hour for all four.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e13_adl_mlp_crypto.sh
#   bash experiments/e13_adl_mlp_crypto.sh
#   ONLY="raw ar" / SKIP=dl / DRY_RUN=1 / NO_MAIL=1 / ADL_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E13}"
ADL_WINDOW="${ADL_WINDOW:-130}"
ADL_REFIT="${ADL_REFIT:-130}"
ADL_EMBARGO="${ADL_EMBARGO:-20}"
ADL_N_SPLITS="${ADL_N_SPLITS:-5}"
ADL_VAL_FRAC="${ADL_VAL_FRAC:-0.3}"
ADL_SEED="${ADL_SEED:-42}"
ADL_THREADS="${ADL_THREADS:-8}"
ADL_ERA_STEP=1
case "${ADL_WINDOW}" in ""|0|[0-9]) echo "ADL_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${ADL_REFIT}"  in ""|0|[0-9]) echo "ADL_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

# The best MLP measured on raw data: e07_mlpreg_raw's modal refit configuration.
ADL_PARAMS="${ADL_PARAMS:-{\"mlp__hidden_layer_sizes\":[64,32],\"mlp__alpha\":8.227241341700458,\"mlp__batch_size\":2048,\"mlp__learning_rate_init\":0.008886238162743407}}"

PREFIX="e13"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e13scout"; ADL_ERA_STEP=4; ADL_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${ADL_N_SPLITS} folds -- wiring only."
fi

# sklearn's MLP has no --n-jobs; it threads through BLAS and these are the only
# knobs that bound it.
export OMP_NUM_THREADS="${ADL_THREADS}"
export MKL_NUM_THREADS="${ADL_THREADS}"
export OPENBLAS_NUM_THREADS="${ADL_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
NAME=(); DATA=(); NFEAT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add raw "crypto/v2.0"     22
add ar  "crypto/v2.0-ar"  25
add dl  "crypto/v2.0-dl"  88
add adl "crypto/v2.0-adl" 91

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. The AR columns are re-checked for
# leakage here rather than trusted from build time: a dataset can be rebuilt with
# different flags and nothing downstream would notice.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_mani.utils import get_spec, target_horizon_days

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
        lag = int(column.rsplit("lag", 1)[1])
        if lag < horizon:
            bad.append("%s carries %s, a lag BELOW the %d-period horizon -- that "
                       "column contains unresolved future" % (version, column, horizon))
    if ar_columns:
        # A tripwire, not a proof: a lag short by even one period would overlap
        # the target it predicts and this correlation would jump.
        frame = pd.read_parquet(path, columns=ar_columns + [target])
        for column in ar_columns:
            both = frame[[column, target]].dropna()
            rho = float(np.corrcoef(both[column], both[target])[0, 1])
            if abs(rho) > 0.25:
                bad.append("%s: corr(%s, %s) = %+.4f, far above the +-0.02 seen "
                           "for resolved lags -- check the lag" % (version, column, target, rho))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts; every AR lag "
      "is >= the %d-period horizon and none overlaps the target" % (len(versions), horizon))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      do autoregressive and distributed lags help the MLP?
  model     PINNED to e07_mlpreg_raw's modal config -- nothing is searched, so
            the arms differ by their columns and by nothing else
  cadence   rolling ${ADL_WINDOW}/${ADL_REFIT}, embargo ${ADL_EMBARGO}, ${ADL_N_SPLITS} folds
  expect    under an hour for all four -- pinning makes each arm ~25 fits
  prior     DL has been tried and did not help (E05 best depth ZERO, E06
            t = -2.18..-2.30). The AR half is what is new.

Where to look while it runs:
  runner    experiments/e13_adl_mlp_crypto.sh
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
echo "${EXPERIMENT}: AR / DL / ADL on a pinned MLP  |  ${total} arm(s)"
echo "cadence: rolling ${ADL_WINDOW}/${ADL_REFIT}, embargo ${ADL_EMBARGO}, ${ADL_N_SPLITS} folds, ${ADL_THREADS} threads"
echo "estimator pinned: ${ADL_PARAMS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${ADL_WINDOW}"
         --refit-every "${ADL_REFIT}" --embargo "${ADL_EMBARGO}"
         --n-splits "${ADL_N_SPLITS}" --val-frac "${ADL_VAL_FRAC}"
         --era-step "${ADL_ERA_STEP}" --seed "${ADL_SEED}" --search-jobs 1
         --fixed-params "${ADL_PARAMS}"
         --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-16s %5s feat  %s\n' "$index" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${SEL_DATA[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${SEL_NFEAT[$i]} feat  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    # A failed arm must not cost the others: read PIPESTATUS[0] after the tee.
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
echo "${EXPERIMENT} -- do AR and DL lags help the MLP?"
echo "  results  results/metrics/crypto/mlp/${PREFIX}_*.json"
echo "  runner   experiments/e13_adl_mlp_crypto.sh"
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
ARMS = [("raw", "contemporaneous"), ("ar", "+ target lags"),
        ("dl", "+ feature lags"), ("adl", "+ both")]
CONTROL = "%s_raw" % prefix

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

rows = [(a, lab, "%s_%s" % (prefix, a), load("%s_%s" % (prefix, a)))
        for a, lab in ARMS]
print("%-14s %-18s %5s %6s %10s %10s %8s %8s %10s %7s" % (
    "run", "columns", "feat", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 106)
for arm, label, run, d in rows:
    if d is None:
        # Never a blank line: absent reads as "tested and null" when it means
        # "never ran".
        print("%-14s %-18s  NOT ON DISK -- this arm did not run" % (run, label))
        continue
    print("%-14s %-18s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f %7.1f" % (
        run, label, d["config"]["n_features"], d["n_periods"],
        d["corr"]["mean"], d["mmc"]["mean"], d["corr"]["sharpe"],
        d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 106)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit
if load(CONTROL) is None:
    print()
    print("The control %s is not on disk, so no delta is printed." % CONTROL)
    raise SystemExit

print()
print("paired vs %s, Newey-West lag 20" % CONTROL)
print("%-6s %11s %11s %11s %9s %7s" % ("arm", "corr d", "mmc d", "payout d", "se", "t"))
print("-" * 62)
base = per_period(CONTROL)
for arm, label, run, d in rows:
    if d is None or arm == "raw":
        continue
    a = per_period(run)
    line = {}
    for metric in ("corr", "mmc", "payout"):
        left = payout(base) if metric == "payout" else base[metric]
        right = payout(a) if metric == "payout" else a[metric]
        j = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"])
        line[metric] = newey_west(j["a"] - j["b"])
    print("%-6s %+11.6f %+11.6f %+11.6f %9.6f %+7.2f" % (
        arm, line["corr"][0], line["mmc"][0], line["payout"][0],
        line["payout"][1], line["payout"][2]))
print("-" * 62)
print()
print("Read AR and DL before ADL: if ADL moves, those two say which half did it.")
print("Three comparisons put the Bonferroni-style bar nearer |t| = 2.4 than 2.0.")
print("~26 effective independent windows, so a null is EXPECTED and IS the")
print("finding. The estimator is pinned to a config chosen on 22 features, so a")
print("negative DL/ADL result partly measures 'this config does not suit 88-91")
print("columns' -- it is not 'lags cannot help a tuned MLP'.")
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
