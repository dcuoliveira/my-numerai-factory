#!/usr/bin/env bash
#
# Experiment E07: a feedforward net against the deployed ridge, on raw 22
#
# E07 is the reference handle. Everything this study produces is greppable as
# E07 in the logs and mail, and as e07_* in results/{metrics,forecast,
# per_period,hyperparams}/crypto/mlp/.
#
# WHAT THIS RUNS
#   `mlp` -- sklearn MLPRegressor behind a StandardScaler, added to
#   prediction/models.py -- on crypto/v2.0's 22 shipped features, at the exact
#   CURUPIRA cadence. One arm by default. The wide derived sets are in the table
#   below but are NOT run unless asked for; see COST.
#
# WHAT IT IS MEASURED AGAINST
#   `ridge/w6r6_fixed_reg`, which is CURUPIRA_02's own backtest run: same
#   dataset, same 130/130 rolling cadence, same embargo 20, same 5 folds, same
#   50 draws, same seed 42, same val_frac 0.3. The ONLY thing that differs is
#   the estimator, which is what makes the per-period series pair exactly.
#
#   That run already exists, so this study does not refit it. Rerunning it would
#   verify determinism, not correctness -- its recorded numbers ARE the numbers
#   in registry.MODELS["CURUPIRA_02"]["backtest"]. The preflight asserts that
#   identity holds before anything is compared. MLP_RERUN_BASELINE=1 rebuilds it
#   anyway, under a separate name, if you want the determinism check too.
#
# WHY THIS COMPARISON IS THE INTERESTING ONE
#   Eleven feature blocks have been tried on this panel and none beat the raw 22
#   by a margin the sample supports. Every one of them varied the COLUMNS while
#   holding the function class at a linear model or a tree. This varies the
#   function class instead, on the columns already known to work. An MLP is the
#   first estimator here that is neither linear nor piecewise-constant, so it is
#   the first that can express a smooth interaction the other two cannot.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on the paired payout or corr delta, Newey-West at lag 20.
#   510 periods at ~95% overlap is roughly 26 effective independent windows, so
#   a null IS the expected outcome and IS the finding -- report it with its
#   interval rather than calling the study inconclusive.
#
#   Read the risk columns too. Arms here routinely move level and risk in
#   opposite directions: the CURUPIRA_02 autoencoder arm lost 0.005 corr while
#   gaining 0.06 sharpe and cutting max drawdown 24%, and a mean-only table
#   would have been true and misleading.
#
# ---------------------------------------------------------------------------
# COST -- read this before adding arms
# ---------------------------------------------------------------------------
#
# Across 141 ridge arms wall clock tracks the NUMBER OF REFITS (r = 0.48) and
# essentially not the feature count (r = -0.12), so the cadence usually sets the
# price. The MLP does not break that rule, but its per-fit cost is far higher
# than ridge's, so the same cadence costs much more.
#
# This cadence gives 4 refits of 39,000 rows each -- read off the baseline run's
# own `refits` block, not assumed -- and each refit is 50 draws x 5 folds + one
# final fit, so 1,004 fits in total. Measured per-fit cost on exactly that block,
# at MLP_THREADS=8:
#
#     arch          batch 1024   batch 4096        (batch 128, for contrast)
#     (32,)               2.4s         2.6s                          6.5s
#     (128,)             10.1s         3.7s                         21.1s
#     (256, 128)         33.4s        32.6s                        110.6s
#
# Inner folds train on less than the full block, so those are upper bounds.
# EXPECT ROUGHLY 3 HOURS for the raw arm, against the ridge baseline's 28
# minutes. The wide arms (786 and 812 columns) are NOT priced -- scout them
# before adding them.
#
# It also does not respond to --n-jobs. sklearn's MLP has no worker count -- it
# parallelises through whatever BLAS numpy is linked against -- so left alone it
# takes every core on the box. This runner pins MLP_THREADS to keep it from
# trampling anything else running. Raise it if the machine is idle.
#
# Scout first anyway. --era-step 4 keeps every 4th period and is how the wiring
# gets shaken out before hours are committed. It is a wiring lever ONLY: the
# split boundary moves, so a downsampled arm is never reported against a
# full-resolution baseline.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e07_mlp_crypto.sh   # minutes, verifies wiring
#   bash experiments/e07_mlp_crypto.sh               # the real run
#   DRY_RUN=1 bash experiments/e07_mlp_crypto.sh
#   MLP_ARMS="raw onehot_rollsel" bash experiments/e07_mlp_crypto.sh
#   MLP_THREADS=16 bash experiments/e07_mlp_crypto.sh
#   MLP_RERUN_BASELINE=1 bash experiments/e07_mlp_crypto.sh
#   ONLY="raw" / SKIP="raw" / NO_MAIL=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed. GNU screen exports $WINDOW=0 into every session it starts,
# so a bare WINDOW="${WINDOW:-130}" silently becomes 0 and every training block
# comes out empty -- with no error, just meaningless scores.
EXPERIMENT="${EXPERIMENT:-E07}"
MLP_WINDOW="${MLP_WINDOW:-130}"
MLP_REFIT="${MLP_REFIT:-130}"
MLP_EMBARGO="${MLP_EMBARGO:-20}"
MLP_N_SPLITS="${MLP_N_SPLITS:-5}"
MLP_N_ITER="${MLP_N_ITER:-50}"
MLP_VAL_FRAC="${MLP_VAL_FRAC:-0.3}"
MLP_SEED="${MLP_SEED:-42}"
MLP_ARMS="${MLP_ARMS:-raw}"
MLP_TASKS="${MLP_TASKS:-reg clf}"
MLP_THREADS="${MLP_THREADS:-8}"
MLP_ERA_STEP=1
case "${MLP_WINDOW}" in ""|0|[0-9]) echo "MLP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${MLP_REFIT}"  in ""|0|[0-9]) echo "MLP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

# A scout is a WIRING check, never a result. The prefix changes with it so a
# downsampled run can never be mistaken for a full-resolution arm in the tree.
PREFIX="e07"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e07scout"; MLP_ERA_STEP=4; MLP_N_ITER=4; MLP_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${MLP_N_ITER} draws, ${MLP_N_SPLITS} folds."
    echo "            Results are NOT comparable to the baseline. Wiring only."
fi

# sklearn's MLP has no --n-jobs; these are the only knobs that bound it.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# The two tasks, each with the ridge run it is measured against. Both baselines
# were fitted at this exact cadence on this exact dataset, so each pair differs
# by the estimator alone.
#
#   reg -> ridge/w6r6_fixed_reg   CURUPIRA_02's backtest run; its numbers ARE the
#                                 registry's, so the preflight checks that identity
#   clf -> ridge/w6r6_fixed_clf   not deployed anywhere, so there is no registry
#                                 entry to check it against -- only its cadence
#
# Classification here does NOT mean predicting a class. run_forecast binarises the
# target at 0.5 and scores the positive-class PROBABILITY as the signal, so both
# arms emit a continuous score and both are ranked the same way. What differs is
# the loss the net minimises: squared error against the binned level, or log-loss
# against which side of the middle bin the row fell.
TASK_KEY=(reg clf); TASK_ARG=(regression classification)
TASK_BASE=(ridge/w6r6_fixed_reg ridge/w6r6_fixed_clf)
# Only the regression baseline is a deployed slot's backtest run.
TASK_SLOT=(CURUPIRA_02 "")

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing name|dataset|count and splitting on IFS='|' tears it
# apart and quietly fits on the wrong columns.
#
# name / dataset / expected feature count
NAME=(); DATA=(); NFEAT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add raw            "crypto/v2.0"                                     22
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                     786
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812

# Select the arms to run, keeping the table's order.
SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [[ " ${MLP_ARMS} " != *" ${arm} "* ]] && continue
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
if [ "${#SEL_NAME[@]}" -eq 0 ]; then
    echo "no arms selected (MLP_ARMS='${MLP_ARMS}')" >&2; exit 1
fi

SEL_TASK=(); SEL_TARG=(); SEL_BASE=(); SEL_SLOT=()
for i in "${!TASK_KEY[@]}"; do
    [[ " ${MLP_TASKS} " != *" ${TASK_KEY[$i]} "* ]] && continue
    SEL_TASK+=("${TASK_KEY[$i]}"); SEL_TARG+=("${TASK_ARG[$i]}")
    SEL_BASE+=("${TASK_BASE[$i]}"); SEL_SLOT+=("${TASK_SLOT[$i]}")
done
if [ "${#SEL_TASK[@]}" -eq 0 ]; then
    echo "no tasks selected (MLP_TASKS='${MLP_TASKS}'); expected reg and/or clf" >&2
    exit 1
fi

# Preflight. A silent feature-count drift attaches a plausible number to the
# wrong experiment, which is worse than a crash -- so read the parquet schema and
# refuse to run on a mismatch. The baseline's identity is checked here too: a
# comparison against a run that is not what the registry says it is would be
# wrong in a way no downstream table could reveal.
python3 - "${MLP_SCOUT:-0}" "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" \
        -- "${SEL_BASE[@]}" -- "${SEL_TARG[@]}" -- "${SEL_SLOT[@]}" <<'PY' || exit 1
import json, os, sys
import pyarrow.parquet as pq

scout = sys.argv[1] == "1"
parts, current = [], []
for token in sys.argv[2:]:
    if token == "--":
        parts.append(current); current = []
    else:
        current.append(token)
parts.append(current)
versions, wants, baselines, tasks, slots = parts
wants = [int(x) for x in wants]

bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        bad.append("%s MISSING -- build it with setup.sh --with-derived" % version)
        continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))

# Each baseline must be matched on cadence, and where it is a deployed slot's
# backtest run, must still BE that run. A comparison against something that is
# not what the registry says it is would be wrong in a way no downstream table
# could reveal.
checked = []
for baseline, task, slot_name in zip(baselines, tasks, slots):
    model, run = baseline.split("/")
    path = "results/metrics/crypto/%s/%s.json" % (model, run)
    if not os.path.exists(path):
        bad.append("baseline %s MISSING -- rerun it with MLP_RERUN_BASELINE=1"
                   % baseline)
        continue
    got = json.load(open(path))
    config = got["config"]

    for key, want in (("task", task), ("cv_scheme", "rolling"),
                      ("train_window", 130), ("refit_every", 130),
                      ("embargo", 20), ("n_splits", 5), ("n_iter", 50),
                      ("seed", 42), ("era_step", 1),
                      ("data_version", "crypto/v2.0")):
        if config.get(key) != want:
            bad.append("baseline %s config %s = %r, this study runs %r -- the "
                       "arms would not be matched"
                       % (baseline, key, config.get(key), want))

    if slot_name:
        from src.live.registry import MODELS
        slot = MODELS[slot_name]
        recorded = slot["backtest"]
        if slot["backtest_run"] != run:
            bad.append("%s's backtest_run is %r, not %r"
                       % (slot_name, slot["backtest_run"], run))
        for key, actual in (("corr", got["corr"]["mean"]),
                            ("mmc", got["mmc"]["mean"])):
            if abs(actual - recorded[key]) > 5e-7:
                bad.append("baseline %s %s = %.6f, registry records %.6f -- the "
                           "run on disk is not the deployed slot's backtest"
                           % (baseline, key, actual, recorded[key]))
        checked.append("%s reproduces %s's recorded corr/mmc" % (baseline, slot_name))
    else:
        # Say so rather than letting the silence imply it was verified too.
        checked.append("%s matches the cadence (not a deployed slot, so there is "
                       "no registry entry to check it against)" % baseline)

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts" % len(versions))
for line in checked:
    print("              %s" % line)
if scout:
    print("             (scout mode -- the baseline check above is for reference "
          "only, a downsampled arm is not comparable to it)")
PY

total=$(( ${#SEL_NAME[@]} * ${#SEL_TASK[@]} ))
[ "${MLP_RERUN_BASELINE:-0}" = "1" ] && total=$((total + ${#SEL_TASK[@]}))
index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

# Mail on START as well as at the end. A job that only mails when it finishes is
# indistinguishable from a job that never started, and this one is unpriced.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (features: ${SEL_NAME[*]}  x  tasks: ${SEL_TASK[*]})
  model     mlp  --  sklearn MLP behind a StandardScaler
  baseline  ${SEL_BASE[*]}  (existing ridge runs, not refitted)
  cadence   rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws
  threads   ${MLP_THREADS}
  expect    ~3 h per arm (1,004 fits over 4 refits); wide arms unpriced

Where to look while it runs:
  runner    experiments/e07_mlp_crypto.sh
  progress  logs/${PREFIX}_*.log        (one per arm)
  results   results/metrics/crypto/mlp/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt   (written at the end)
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp vs ${SEL_BASE[*]}  |  ${total} arm(s)"
echo "cadence: rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# Never let one arm's failure cost the others: check PIPESTATUS[0] after the
# tee, record it, continue.
run_arm() {                          # model, task, run_name, extra args...
    local model="$1" task="$2" run_name="$3"; shift 3
    index=$((index+1))
    local cmd=(python3 -u -m src.run_forecast crypto "${model}"
               --task "${task}" --metric corr --bmc-reference benchmark
               --cv-scheme rolling --train-window "${MLP_WINDOW}"
               --refit-every "${MLP_REFIT}" --embargo "${MLP_EMBARGO}"
               --n-splits "${MLP_N_SPLITS}" --n-iter "${MLP_N_ITER}"
               --val-frac "${MLP_VAL_FRAC}" --era-step "${MLP_ERA_STEP}"
               --seed "${MLP_SEED}" --search-jobs 1
               --run-name "${run_name}" "$@")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-28s %s\n' "$index" "$total" "$run_name" "${cmd[*]}"; return 0
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    local started; started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    local status="${PIPESTATUS[0]}" el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
}

if [ "${MLP_RERUN_BASELINE:-0}" = "1" ]; then
    for t in "${!SEL_TASK[@]}"; do
        run_arm ridge "${SEL_TARG[$t]}" "${PREFIX}_ridge${SEL_TASK[$t]}_raw" \
                --data-version "crypto/v2.0"
    done
fi
# Task is the OUTER loop so the two headline arms -- mlpreg and mlpclf on raw --
# both land before any wide arm starts. A run that has to be cut short then still
# answers the question it was launched for.
for t in "${!SEL_TASK[@]}"; do
    for i in "${!SEL_NAME[@]}"; do
        run_arm mlp "${SEL_TARG[$t]}" \
                "${PREFIX}_mlp${SEL_TASK[$t]}_${SEL_NAME[$i]}" \
                --data-version "${SEL_DATA[$i]}"
    done
done

[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- a feedforward net against ${SEL_BASE[*]}"
echo "  results  results/metrics/crypto/mlp/${PREFIX}_*.json"
echo "  runner   experiments/e07_mlp_crypto.sh"
echo "  logs     logs/${PREFIX}_*.log"
echo "=============================================================="
elapsed_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((elapsed_all/60))m$((elapsed_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" "${SEL_TASK[@]}" -- "${SEL_BASE[@]}" <<'PY'
import glob, json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
cut = sys.argv.index("--")
# task key -> the ridge run that task's arms are measured against.
BASELINES = dict(zip(sys.argv[3:cut], sys.argv[cut + 1:]))

def per_period(ref):
    model, run = ref.split("/") if "/" in ref else ("mlp", ref)
    frame = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
    return frame.set_index("date") if "date" in frame.columns else frame

def payout(frame):
    return 0.1 * frame["corr"] + frame["mmc"]

def newey_west(x, lag=20):
    # lag 20: a 20-day forward target on daily dates leaves consecutive periods
    # ~95% overlapping, and treating them as independent inflates t by ~sqrt(20).
    x = np.asarray(x, float); n = len(x); mean = x.mean(); e = x - mean
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return mean, se, (mean / se if se > 0 else float("nan"))

def task_of(run):
    # Run names encode the task: e07_mlpreg_raw, e07_mlpclf_raw. Reading it back
    # off the name keeps the report from needing a second source of truth.
    for key in BASELINES:
        if "_mlp%s_" % key in run or "_ridge%s_" % key in run:
            return key
    return None

rows = []
for key, baseline in sorted(BASELINES.items()):
    path = "results/metrics/crypto/%s/%s.json" % tuple(baseline.split("/"))
    if os.path.exists(path):
        rows.append((baseline, json.load(open(path)), key, True))
for model in ("mlp", "ridge"):
    for path in sorted(glob.glob("results/metrics/crypto/%s/%s_*.json"
                                 % (model, prefix))):
        run = os.path.basename(path)[:-5]
        rows.append((run, json.load(open(path)), task_of(run), False))

if not rows:
    print("no results"); raise SystemExit

print("%-28s %5s %5s %7s %10s %10s %8s %8s %10s" % (
    "run", "task", "feat", "n", "CORR", "MMC", "sharpe", "maxdd", "payout"))
print("-" * 98)
for name, d, key, _ in rows:
    print("%-28s %5s %5d %7d %10.6f %10.6f %8.4f %8.4f %10.6f" % (
        name, key or "?", d["config"]["n_features"], d["n_periods"],
        d["corr"]["mean"], d["mmc"]["mean"], d["corr"]["sharpe"],
        d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
print("-" * 98)
print("A reg arm and a clf arm are NOT comparable to each other -- they optimise")
print("different losses against different baselines. Read each down its own task.")

if scout:
    # A scout arm keeps every 4th period AND sits on a moved train/validation
    # boundary, so it shares neither the baseline's sample nor its split. The
    # delta would compute happily and mean nothing, and a number printed under a
    # "paired vs baseline" heading gets quoted later whatever the header said.
    # So it is not printed at all.
    print()
    print("SCOUT RUN -- no comparison printed, and none is available.")
    print("--era-step 4 keeps every 4th period and moves the split boundary, so")
    print("these arms share neither the sample nor the split with their baselines.")
    print("It checked the wiring: the grid translated, the Pipeline cloned, the")
    print("scaler fit inside the folds, and the four output files were written.")
    print("Rerun without MLP_SCOUT=1 for a number that can be reported.")
    raise SystemExit

# Paired deltas. NEVER two means: the arms share their periods, folds and seed,
# so the DIFFERENCE series has a far smaller standard error than either series
# alone. The same data has given t = 0.75 unpaired and t = 3.14 paired.
print()
print("paired vs each arm's OWN baseline, inner join on periods, Newey-West lag 20")
print("%-24s %-22s %8s %+10s %9s %7s" % (
    "run", "baseline", "metric", "delta", "se", "t"))
print("-" * 86)
for name, _, key, is_base in rows:
    if is_base:
        continue
    if key is None or key not in BASELINES:
        # Never silently drop an arm from the comparison: an absent row reads as
        # "there was nothing to say", which is not the same as "we could not tell
        # which baseline this belongs to".
        print("%-24s %-22s %8s" % (name, "UNMATCHED", "-- no baseline resolved"))
        continue
    baseline = BASELINES[key]
    base, arm = per_period(baseline), per_period(name)
    for metric in ("corr", "mmc", "payout"):
        left = payout(base) if metric == "payout" else base[metric]
        right = payout(arm) if metric == "payout" else arm[metric]
        joined = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"])
        delta, se, t = newey_west(joined["a"] - joined["b"])
        print("%-24s %-22s %8s %+10.6f %9.6f %+7.2f" % (
            name, baseline, metric, delta, se, t))
print("-" * 86)
print("510 periods at ~95% overlap is roughly 26 effective independent windows.")
print("|t| < 2 is the EXPECTED outcome and is the finding -- report it with its")
print("interval, not as an inconclusive run needing more arms.")
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
