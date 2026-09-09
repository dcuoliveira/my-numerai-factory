#!/usr/bin/env bash
#
# Experiment E09: does a torch MLP reproduce the sklearn one?
#
# E09 is the reference handle. Everything this study writes is greppable as E09
# and as e09_* in results/{metrics,forecast,per_period,hyperparams}/crypto/.
#
# WHAT THIS RUNS
#   Two arms on crypto/v2.0's raw 22 features at the CURUPIRA cadence, differing
#   in ONE thing -- which framework trains the network:
#
#     e09_sklmlp_raw     --model mlp         sklearn MLPRegressor
#     e09_torchmlp_raw   --model torch_mlp   src/prediction/torch_mlp.py
#
#   Same architecture set, same alpha / learning-rate / batch grids, same folds,
#   same seed, same StandardScaler in front. TorchMLPWrapper REFERENCES
#   MLPWrapper's grid constants rather than copying them, so the two search
#   spaces cannot drift apart; a test pins that.
#
# WHY BOTH ARMS RUN, RATHER THAN COMPARING TO e07_mlpreg_raw
#   E07's alpha ceiling was 1e1 and is now 1e2, so e07_mlpreg_raw is on a
#   different search space and is NOT the control for this. Rerunning sklearn
#   here costs ~1.5 h and removes the confound entirely. Do not put an e07_*
#   number in the same row as an e09_* one.
#
# WHAT WOULD COUNT AS A RESULT -- note the direction
#   This is the rare study where the INTERESTING outcome is a null. The
#   implementations agree at unit level, five seeds on a real 130-period block at
#   matched hyperparameters:
#
#       impl      epochs   final loss   OOS spearman        pred std
#       sklearn     28.6     0.025217   0.11541 +- 0.00797   0.00789
#       torch       26.0     0.025262   0.11638 +- 0.00928   0.01022
#
#   Final losses agree to four decimals and the spearman gap is an eighth of the
#   seed spread. So |t| < 2 on the paired payout delta CONFIRMS the replication.
#   A significant delta would mean the walk-forward, not the estimator, is where
#   the two diverge -- and would need explaining before either arm is trusted.
#
#   Report level and risk both. Two implementations can reach the same mean by
#   different routes, and drawdown is where that shows.
#
# COST
#   torch is ~2x sklearn per fit on CPU (5.0s against 2.4s on a 39,000-row
#   block), and e07_mlpreg_raw took 92 min. Budget ~1.5 h for the sklearn arm and
#   ~3 h for the torch one; about 4.5 h in total, sequential.
#
#   MLP_DEVICE=cuda is accepted and UNMEASURED on this panel. The walk-forward
#   runs ~1,000 small fits and host-to-device transfer can cost more than the
#   kernels save at this size, so do not assume it is faster -- scout it first.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e09_torch_mlp_crypto.sh   # ~3 min, wiring
#   bash experiments/e09_torch_mlp_crypto.sh               # both arms
#   ONLY=torchmlp / SKIP=sklmlp / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E09}"
MLP_WINDOW="${MLP_WINDOW:-130}"
MLP_REFIT="${MLP_REFIT:-130}"
MLP_EMBARGO="${MLP_EMBARGO:-20}"
MLP_N_SPLITS="${MLP_N_SPLITS:-5}"
MLP_N_ITER="${MLP_N_ITER:-50}"
MLP_VAL_FRAC="${MLP_VAL_FRAC:-0.3}"
MLP_SEED="${MLP_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-8}"
MLP_DATA="${MLP_DATA:-crypto/v2.0}"
MLP_NFEAT="${MLP_NFEAT:-22}"
MLP_ERA_STEP=1
case "${MLP_WINDOW}" in ""|0|[0-9]) echo "MLP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${MLP_REFIT}"  in ""|0|[0-9]) echo "MLP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e09"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e09scout"; MLP_ERA_STEP=4; MLP_N_ITER=4; MLP_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${MLP_N_ITER} draws, ${MLP_N_SPLITS} folds -- wiring only."
    echo "            The split boundary moves, so scout arms compare to nothing."
fi

# Neither MLP has an --n-jobs. sklearn's threads through BLAS, torch's through
# its own pool; these cover both, and torch reads OMP_NUM_THREADS at import.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
NAME=(sklmlp torchmlp); MODEL=(mlp torch_mlp)
SEL_NAME=(); SEL_MODEL=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_MODEL+=("${MODEL[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. The grid identity is checked here too:
# if the two wrappers ever stop sharing a search space, this study measures the
# space rather than the framework, and nothing downstream would reveal it.
python3 - "${MLP_DATA}" "${MLP_NFEAT}" <<'PY' || exit 1
import sys
import numpy as np
import pyarrow.parquet as pq

version, want = sys.argv[1], int(sys.argv[2])
bad = []
path = "data/%s/train.parquet" % version
try:
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))
except OSError:
    bad.append("%s MISSING -- build it with setup.sh" % version)

try:
    import torch                                          # noqa: F401
except ImportError as error:
    bad.append("torch is not installed (%s) -- it is OPTIONAL in setup.sh, and "
               "the torch_mlp arm needs it" % error)

from src.prediction.models import build_wrapper, supported_tasks
if "regression" not in supported_tasks("torch_mlp"):
    bad.append("torch_mlp no longer declares regression support")
a = build_wrapper("mlp", "regression", seed=42, n_jobs=1).param_grid
b = build_wrapper("torch_mlp", "regression", seed=42, n_jobs=1).param_grid
if len(a) != len(b):
    bad.append("grids differ in sub-grid count: %d vs %d" % (len(a), len(b)))
else:
    for i, (ga, gb) in enumerate(zip(a, b)):
        if set(ga) != set(gb):
            bad.append("sub-grid %d has different keys" % i); continue
        for key in ga:
            if not np.array_equal(np.asarray(ga[key], dtype=object),
                                  np.asarray(gb[key], dtype=object)):
                bad.append("sub-grid %d differs on %s -- the arms would be "
                           "searching different spaces" % (i, key))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s carries %d features; torch present; the two wrappers "
      "search an identical grid (%d sub-grids)" % (version, want, len(a)))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

# Mail on START as well as at the end: a job that only mails on completion is
# indistinguishable from one that never started or silently died.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      does a torch MLP reproduce the sklearn one, same grid, same folds
  data      ${MLP_DATA}  (${MLP_NFEAT} features)
  cadence   rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws
  threads   ${MLP_THREADS}   device ${MLP_DEVICE:-cpu}
  expect    ~1.5 h sklearn + ~3 h torch = ~4.5 h
  NOTE      a NULL is the success condition here -- see the runner header

Where to look while it runs:
  runner    experiments/e09_torch_mlp_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/{mlp,torch_mlp}/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: sklearn vs torch MLP  |  ${total} arm(s)  |  ${MLP_DATA}"
echo "cadence: rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; model="${SEL_MODEL[$i]}"
    run_name="${PREFIX}_${arm}_raw"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${MLP_WINDOW}"
         --refit-every "${MLP_REFIT}" --embargo "${MLP_EMBARGO}"
         --n-splits "${MLP_N_SPLITS}" --n-iter "${MLP_N_ITER}"
         --val-frac "${MLP_VAL_FRAC}" --era-step "${MLP_ERA_STEP}"
         --seed "${MLP_SEED}" --search-jobs 1
         --data-version "${MLP_DATA}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-24s %-10s %s\n' "$index" "$total" "$run_name" "$model" "${MLP_DATA}"
        continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  (--model ${model})  $(date '+%H:%M:%S')"
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
echo "${EXPERIMENT} -- does a torch MLP reproduce the sklearn one?"
echo "  results  results/metrics/crypto/{mlp,torch_mlp}/${PREFIX}_*.json"
echo "  runner   experiments/e09_torch_mlp_crypto.sh"
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
ARMS = [("mlp", "%s_sklmlp_raw" % prefix, "sklearn"),
        ("torch_mlp", "%s_torchmlp_raw" % prefix, "torch")]

def load(model, run):
    p = "results/metrics/crypto/%s/%s.json" % (model, run)
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(model, run):
    f = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
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

found = [(m, r, lab, load(m, r)) for m, r, lab in ARMS]
print("%-24s %-9s %6s %10s %10s %8s %8s %10s %8s" % (
    "run", "impl", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 100)
for model, run, lab, d in found:
    if d is None:
        # Never a blank line: absent reads as "tested and null" when it means
        # "never ran".
        print("%-24s %-9s  NOT ON DISK -- this arm did not run" % (run, lab))
        continue
    print("%-24s %-9s %6d %10.6f %10.6f %8.4f %8.4f %10.6f %8.1f" % (
        run, lab, d["n_periods"], d["corr"]["mean"], d["mmc"]["mean"],
        d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 100)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit

if any(d is None for *_, d in found):
    print()
    print("Both arms are needed for the comparison; one is missing, so none is")
    print("printed. Rerun the missing arm with ONLY=.")
    raise SystemExit

a, b = found[0], found[1]
print()
print("paired torch - sklearn, inner join on periods, Newey-West lag 20")
print("%-10s %12s %12s %+12s %10s %8s" % (
    "metric", "sklearn", "torch", "delta", "se", "t"))
print("-" * 70)
pa, pb = per_period(a[0], a[1]), per_period(b[0], b[1])
for metric in ("corr", "mmc", "payout"):
    left = payout(pa) if metric == "payout" else pa[metric]
    right = payout(pb) if metric == "payout" else pb[metric]
    j = pd.concat([left, right], axis=1, join="inner", keys=["s", "t"])
    delta, se, t = newey_west(j["t"] - j["s"])
    print("%-10s %12.6f %12.6f %+12.6f %10.6f %+8.2f" % (
        metric, j["s"].mean(), j["t"].mean(), delta, se, t))
print("-" * 70)

# Prediction agreement is the sharper test: two implementations can land on the
# same mean by different routes, and a payout null would not catch that.
try:
    from src.analysis.corr_heatmap import within_period_rank_corr
    cols = {}
    for model, run, lab, _ in found:
        f = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (model, run),
                        parse_dates=["date"])
        cols[lab] = f.set_index(["symbol", "date"])["prediction"]
    panel = pd.DataFrame(cols).dropna()
    rho = within_period_rank_corr(panel).loc["sklearn", "torch"]
    print()
    print("within-period prediction rank correlation: %.4f  (%d dates)"
          % (rho, panel.index.get_level_values("date").nunique()))
    print("for scale, from the arm-board skill: ridge variants against each")
    print("other run 0.883-0.926 and are called effectively one model.")
except Exception as error:
    print("\nprediction correlation unavailable: %s" % error)

print()
print("A NULL IS THE SUCCESS CONDITION. The two implementations agree at unit")
print("level (final losses to four decimals over five seeds), so |t| < 2 here")
print("confirms the replication survives the walk-forward. A significant delta")
print("would locate the divergence in the walk-forward, not the estimator.")
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
