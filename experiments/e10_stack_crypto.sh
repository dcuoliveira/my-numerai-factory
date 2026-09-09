#!/usr/bin/env bash
#
# Experiment E10: walk-forward stacking -- a FITTED meta-learner over members
#
# E10 is the reference handle. Everything this study writes is greppable as E10
# in the logs and mail, and as e10_* in results/{metrics,forecast,per_period,
# hyperparams}/crypto/{ridge,mlp}/.
#
# WHAT THIS ASKS
#   Every blend on the board -- every CURUPIRA slot blend, every run_ensemble
#   output -- is a FIXED rank average with hand-set weights. run_weighted_ensemble
#   goes one step further and re-estimates NON-NEGATIVE weights walk-forward.
#   Neither is stacked generalization in Wolpert's sense, because neither fits an
#   arbitrary level-1 model: a convex combination cannot express a signed
#   coefficient, an intercept, or any interaction between two members.
#
#   So: does a meta-learner FITTED on the members' out-of-sample predictions beat
#   a fixed 1/N average of the same members, on the same dates?
#
# HOW IT IS BUILT
#   build_stack_dataset turns finished member forecasts into a dataset whose
#   feature_stk_* columns ARE the members' per-period-ranked predictions. The
#   meta-learner is then an ordinary src.run_forecast run against that dataset,
#   so it inherits the walk-forward, the inner CV, the search and the scoring
#   unmodified. There is no separate stacking code path to audit.
#
# TWO INNER CROSS-VALIDATIONS, BECAUSE WOLPERT USED K-FOLD
#   purged  the chronological, embargoed inner split the rest of the book uses.
#   kfold   shuffled random k-fold -- Wolpert's ORIGINAL stacked generalization.
#           Legitimate here precisely because the level-1 features are already
#           out-of-sample base predictions, so the time structure was spent
#           making them. It lets a forward target straddle a fold boundary, but
#           only in hyperparameter SELECTION: the OUTER walk-forward is identical
#           either way, so both arms are scored pseudo-OOS. kfold arms end `_kf`.
#           The paired _kf-vs-plain delta is the price the purge was paying.
#
# THREE MEMBER SETS, BECAUSE "STACKING" MEANS TWO DIFFERENT THINGS
#   stackR   two ridges          -- stacking within ONE model class
#   stackM   two MLPs            -- the same question for a non-linear class
#   stackX   both, all four      -- stacking ACROSS classes
#   If only stackX wins, the gain is function-class diversity and not stacking.
#
# TWO META-LEARNERS
#   ridge  a signed linear combiner with an intercept. This is the smallest step
#          beyond NNLS, and the honest test of whether the non-negativity
#          constraint was costing anything.
#   mlp    non-linear, so it can express "trust member A only where B agrees".
#
# TWO CONTROLS, ON THE IDENTICAL MEMBERS AND DATES
#   1/N    run_ensemble           -- the fixed rank average the book already uses
#   NNLS   run_weighted_ensemble --scheme stacking_nnls -- walk-forward weights,
#          constrained non-negative. Beating this is the real bar; beating 1/N
#          only says the members were unequal.
#
#   The meta refits every ${META_REFIT} periods and the NNLS control updates every
#   ${META_REFIT} too. That is deliberate: with different cadences the comparison
#   would confound "a fitted model helps" with "refitting more often helps".
#
# THE SCORED WINDOW IS SHORT, AND THAT IS THE DESIGN
#   Members validate on ~510 dates. Stacking inner-joins them, then splits that
#   panel AGAIN into meta-train and meta-validation, so a stack is scored on the
#   ~153-date tail. Three consequences, all of which bite:
#     * every arm in the comparison must be sliced to that tail -- the report
#       inner-joins per-period series, so it is handled, but a hand-written
#       comparison against a 510-period number is simply wrong;
#     * calmar is not comparable across different period counts (CLAUDE.md);
#     * 153 dates at ~95% overlap is ~8 effective independent windows. Nulls are
#       the expected outcome and ARE the finding.
#
# COST -- the meta arms are NOT cheap, which is counter-intuitive
#   MEASURED: a ridge meta on 4 stack features, n_iter 30, 5 folds, expanding,
#   refit_every 130 (so 2 blocks) over 153 scored periods took 30.1 min on one
#   core. Four features and a one-parameter estimator, and still half an hour.
#
#   The fits are not the cost -- the per-period SCORER is. Every candidate is
#   scored by computing numerai_corr date by date across the whole training
#   window, so cost scales with (draws x folds x refit blocks x window), and
#   with expanding CV the window grows all study. Narrow features buy nothing.
#
#   Hence META_N_ITER defaults to 20 rather than the base arms' 50: ridge has a
#   SINGLE hyperparameter over a 100-point logspace, so 50 draws is waste, not
#   rigour. Scaling the measurement above: ~70 min per ridge meta arm at the
#   defaults, and the three MLP metas are the long pole.
#
#   Budget: base arms ~4 h (the two MLPs), meta arms ~6-10 h, datasets and
#   controls seconds. STAGES lets you skip whatever is already on disk, and
#   DRY_RUN=1 lists the arms without running any of them.
#
# Usage:
#   E10_SCOUT=1 bash experiments/e10_stack_crypto.sh      # ~15 min, wiring only
#   bash experiments/e10_stack_crypto.sh                  # everything
#   STAGES="stack meta controls report" bash experiments/e10_stack_crypto.sh
#   ONLY="stackX" / SKIP="stackM" / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed. GNU screen exports $WINDOW=0 into every session it starts,
# so a bare WINDOW="${WINDOW:-130}" silently becomes 0 and every training block
# comes out empty -- no error, just meaningless scores.
EXPERIMENT="${EXPERIMENT:-E10}"
E10_EMBARGO="${E10_EMBARGO:-20}"
E10_VAL_FRAC="${E10_VAL_FRAC:-0.3}"
E10_SEED="${E10_SEED:-42}"
E10_N_SPLITS="${E10_N_SPLITS:-5}"
E10_N_ITER="${E10_N_ITER:-50}"
# Fewer draws for the meta than for the base arms, deliberately -- see COST.
META_N_ITER="${META_N_ITER:-20}"
# Which INNER hyperparameter CV each meta-learner is tuned under. "purged" is the
# chronological, embargoed split the rest of the book uses; "kfold" is Wolpert's
# original shuffled random k-fold, which is legitimate here because the level-1
# features are ALREADY out-of-sample base predictions. Running both is the point
# of the study: the two arms differ ONLY in the inner CV, so any gap between them
# is the price the time-series purge was paying. kfold arms carry a `_kf` suffix.
# The OUTER walk-forward is identical either way, so both are scored pseudo-OOS.
INNER_CVS="${INNER_CVS:-purged kfold}"
MLP_THREADS="${MLP_THREADS:-8}"
E10_ERA_STEP=1

# Base arms use the CURUPIRA cadence so they sit alongside the rest of the book.
BASE_WINDOW="${BASE_WINDOW:-130}"
BASE_REFIT="${BASE_REFIT:-130}"
# The meta refits monthly -- see the header. Must match the NNLS control's
# --update-every or the comparison confounds two things at once.
META_REFIT="${META_REFIT:-22}"

STAGES="${STAGES:-base stack meta controls report}"
case "${BASE_WINDOW}" in ""|0|[0-9]) echo "BASE_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${META_REFIT}"  in ""|0|[0-9]) echo "META_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e10"
if [ "${E10_SCOUT:-0}" = "1" ]; then
    PREFIX="e10scout"; E10_ERA_STEP=4; E10_N_ITER=4; E10_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${E10_N_ITER} draws, ${E10_N_SPLITS} folds."
    echo "            Wiring only -- the split boundary moves, so scout arms are"
    echo "            NOT comparable to any full-resolution baseline, and the"
    echo "            stack panel they build is a different window again."
fi

# sklearn's MLP has no --n-jobs; it parallelises through whatever BLAS numpy is
# linked against, so left alone it takes every core. These are the only knobs.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# --------------------------------------------------------------------------- #
# Level 0: the base arms. Parallel arrays, never pipe-delimited records.
# --------------------------------------------------------------------------- #
# All four share one cadence and one --val-frac ON PURPOSE: the stack panel is
# their INNER JOIN, so a member run at a different val-frac would silently shrink
# the window every later comparison is made on.
#
# arm / model / dataset
BNAME=(); BMODEL=(); BDATA=()
badd() { BNAME+=("$1"); BMODEL+=("$2"); BDATA+=("$3"); }
badd ridge_raw   ridge "crypto/v2.0"
badd ridge_fenc  ridge "crypto/v2.0-fenccausal"
badd mlp_raw     mlp   "crypto/v2.0"
badd mlp_fenc    mlp   "crypto/v2.0-fenccausal"

# --------------------------------------------------------------------------- #
# Level 1: the member sets. SMEMBERS holds space-separated "model/run" refs.
# --------------------------------------------------------------------------- #
SNAME=(); SMEMBERS=()
sadd() { SNAME+=("$1"); SMEMBERS+=("$2"); }
sadd stackR "ridge/${PREFIX}_ridge_raw ridge/${PREFIX}_ridge_fenc"
sadd stackM "mlp/${PREFIX}_mlp_raw mlp/${PREFIX}_mlp_fenc"
sadd stackX "ridge/${PREFIX}_ridge_raw ridge/${PREFIX}_ridge_fenc mlp/${PREFIX}_mlp_raw mlp/${PREFIX}_mlp_fenc"

SEL_SNAME=(); SEL_SMEMBERS=()
for i in "${!SNAME[@]}"; do
    s="${SNAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${s} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${s} "* ]] && continue
    SEL_SNAME+=("${s}"); SEL_SMEMBERS+=("${SMEMBERS[$i]}")
done
if [ "${#SEL_SNAME[@]}" -eq 0 ]; then
    echo "no member sets selected (ONLY='${ONLY:-}' SKIP='${SKIP:-}')" >&2; exit 1
fi

# --------------------------------------------------------------------------- #
# Preflight. A silent feature-count drift attaches a plausible number to the
# wrong experiment, which is worse than a crash.
# --------------------------------------------------------------------------- #
python3 - "${BDATA[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq

WANT = {"crypto/v2.0": 22, "crypto/v2.0-fenccausal": 44}
bad = []
for version in dict.fromkeys(sys.argv[1:]):
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        bad.append("%s MISSING -- build it with "
                   "python3 -m src.data_mani.build_feature_encoding --mode causal"
                   % version)
        continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    want = WANT.get(version)
    if want is not None and got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: base datasets at the expected feature counts")
PY

N_CV=$(echo ${INNER_CVS} | wc -w)
# per set: (2 metas x N_CV inner-CVs) + 2 controls (controls have no inner CV)
total_arms=$(( ${#BNAME[@]} + ${#SEL_SNAME[@]} * (2 * N_CV + 2) ))
ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

# Mail on START as well as at the end. A job that only mails on completion is
# indistinguishable from one that never started or silently died.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  question  does a FITTED meta-learner over member predictions beat a fixed
            1/N rank average, and beat constrained NNLS, on the same dates?
  base      ${#BNAME[@]} arms (${BNAME[*]}) at rolling ${BASE_WINDOW}/${BASE_REFIT}
  sets      ${SEL_SNAME[*]}
  meta      ridge (signed linear) and mlp (non-linear), refit every ${META_REFIT},
            ${META_N_ITER} draws (fewer than the base arms -- see COST in the runner)
  inner CV  ${INNER_CVS} (kfold = Wolpert's original; arms suffixed _kf)
  controls  run_ensemble 1/N, and run_weighted_ensemble --scheme stacking_nnls
            updating every ${META_REFIT} -- same cadence as the meta, on purpose
  stages    ${STAGES}
  threads   ${MLP_THREADS}
  expect    ~4-5 h end to end, nearly all of it the two MLP base arms

  CAVEAT    a stack is scored on the ~153-date TAIL of the members' window, not
            their 510 dates. Every comparison must be paired on that tail; ~8
            effective independent windows, so nulls are expected.

Where to look while it runs:
  runner    experiments/e10_stack_crypto.sh
  progress  logs/${PREFIX}_*.log        (one per arm)
  results   results/metrics/crypto/{ridge,mlp}/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt   (written at the end)
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: walk-forward stacking  |  ${#SEL_SNAME[@]} member set(s)"
echo "stages: ${STAGES}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

run_arm() {   # run_arm <run_name> <command...>
    local run_name="$1"; shift
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  would run  %s\n' "${run_name}"; return 0; fi
    ran=$((ran+1))
    echo; echo "--------------------------------------------------------------"
    echo "${run_name}  $(date '+%H:%M:%S')"
    echo "--------------------------------------------------------------"
    local started; started=$(date +%s)
    # A failed arm must not cost the others: check PIPESTATUS[0] after the tee,
    # record it, continue.
    "$@" 2>&1 | tee "logs/${run_name}.log"
    local status="${PIPESTATUS[0]}"; local el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
        return 1
    fi
    echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
}

# --------------------------------------------------------------------------- #
# Stage: base arms
# --------------------------------------------------------------------------- #
if has_stage base; then
  echo; echo "### stage: base arms (level 0)"
  for i in "${!BNAME[@]}"; do
    run_name="${PREFIX}_${BNAME[$i]}"
    run_arm "${run_name}" python3 -u -m src.run_forecast crypto "${BMODEL[$i]}" \
        --task regression --metric corr --bmc-reference benchmark \
        --cv-scheme rolling --train-window "${BASE_WINDOW}" \
        --refit-every "${BASE_REFIT}" --embargo "${E10_EMBARGO}" \
        --n-splits "${E10_N_SPLITS}" --n-iter "${E10_N_ITER}" \
        --val-frac "${E10_VAL_FRAC}" --era-step "${E10_ERA_STEP}" \
        --seed "${E10_SEED}" --search-jobs 1 \
        --data-version "${BDATA[$i]}" --run-name "${run_name}"
  done
fi

# --------------------------------------------------------------------------- #
# Stage: stack datasets
# --------------------------------------------------------------------------- #
if has_stage stack; then
  echo; echo "### stage: stack datasets (level 0 -> level 1 features)"
  for i in "${!SEL_SNAME[@]}"; do
    set_name="${SEL_SNAME[$i]}"
    # shellcheck disable=SC2086
    run_arm "${PREFIX}_build_${set_name}" python3 -u -m src.data_mani.build_stack_dataset \
        --members ${SEL_SMEMBERS[$i]} \
        --out-version "crypto/v2.0-${PREFIX}${set_name}"
  done
fi

# --------------------------------------------------------------------------- #
# Stage: meta arms
# --------------------------------------------------------------------------- #
if has_stage meta; then
  echo; echo "### stage: meta-learners (level 1)"
  for i in "${!SEL_SNAME[@]}"; do
    set_name="${SEL_SNAME[$i]}"
    for meta in ridge mlp; do
      for cv in ${INNER_CVS}; do
        suffix=""; [ "${cv}" = "kfold" ] && suffix="_kf"
        run_name="${PREFIX}_meta_${meta}_${set_name}${suffix}"
        # Expanding, not rolling: the meta panel is short and a 130-period rolling
        # window would leave the first blocks with almost nothing to fit on.
        #
        # --era-step is pinned to 1 and must stay pinned. The stack panel already
        # inherits whatever sampling the BASE arms were run at, so passing the
        # study's era-step here downsamples a second time. At era-step 4 that took
        # a 126-period panel down to 27 and the walk-forward produced nothing at
        # all -- which is how this was found. It is invisible in a full run,
        # because era-step is 1 there and the second application is a no-op.
        #
        # --inner-cv is the ONLY thing that differs between a meta arm and its
        # `_kf` twin. Everything else -- cadence, embargo on the OUTER loop,
        # draws, seed, dataset -- is held, so the paired delta isolates the
        # inner CV. --embargo still governs the outer walk-forward under kfold;
        # it is only the inner fold construction that stops using it.
        run_arm "${run_name}" python3 -u -m src.run_forecast crypto "${meta}" \
            --task regression --metric corr --bmc-reference benchmark \
            --cv-scheme expanding --refit-every "${META_REFIT}" \
            --embargo "${E10_EMBARGO}" --inner-cv "${cv}" \
            --n-splits "${E10_N_SPLITS}" --n-iter "${META_N_ITER}" \
            --val-frac "${E10_VAL_FRAC}" --era-step 1 \
            --seed "${E10_SEED}" --search-jobs 1 \
            --data-version "crypto/v2.0-${PREFIX}${set_name}" --run-name "${run_name}"
      done
    done
  done
fi

# --------------------------------------------------------------------------- #
# Stage: controls
# --------------------------------------------------------------------------- #
if has_stage controls; then
  echo; echo "### stage: controls (fixed 1/N, and constrained NNLS)"
  for i in "${!SEL_SNAME[@]}"; do
    set_name="${SEL_SNAME[$i]}"
    # shellcheck disable=SC2086
    run_arm "${PREFIX}_ctl_eq_${set_name}" python3 -u -m src.run_ensemble \
        crypto ridge ${SEL_SMEMBERS[$i]} --run-name "${PREFIX}_ctl_eq_${set_name}"
    # shellcheck disable=SC2086
    run_arm "${PREFIX}_ctl_nnls_${set_name}" python3 -u -m src.run_weighted_ensemble \
        crypto ridge ${SEL_SMEMBERS[$i]} --scheme stacking_nnls \
        --update-every "${META_REFIT}" --lag "${E10_EMBARGO}" --seed "${E10_SEED}" \
        --run-name "${PREFIX}_ctl_nnls_${set_name}"
  done
fi

[ "${DRY_RUN:-0}" = "1" ] && exit 0
has_stage report || exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- walk-forward stacking: a fitted meta-learner over members"
echo "  results  results/metrics/crypto/{ridge,mlp}/${PREFIX}_*.json"
echo "  runner   experiments/e10_stack_crypto.sh"
echo "  logs     logs/${PREFIX}_*.log"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} step(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${E10_SCOUT:-0}" "${SEL_SNAME[@]}" <<'PY'
import glob, json, os, sys
import numpy as np
import pandas as pd

# The SHARED report arithmetic, so this mail cannot diverge from report_pdf. Both
# charge a degenerate period (constant prediction -> corr NaN but mmc scored) the
# 0 it earned instead of dropping it, which is what stopped a collapsed arm from
# ranking first. See results_table.degenerate_periods.
from src.analysis.results_table import (charged_corr, degenerate_periods,
                                        payout_series)

prefix, scout, sets = sys.argv[1], sys.argv[2] == "1", sys.argv[3:]

def per_period(model, run):
    f = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
    return f.set_index("date") if "date" in f.columns else f

def payout(f):
    return payout_series(f)

def series(f, metric):
    return payout_series(f) if metric == "payout" else charged_corr(f) if metric == "corr" else f[metric]

def newey_west(x, lag=20):
    # lag 20: a 20-day forward target on daily dates leaves consecutive periods
    # ~95% overlapping; treating them as independent inflates t by ~sqrt(20).
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

rows = []
for model in ("ridge", "mlp"):
    for path in sorted(glob.glob("results/metrics/crypto/%s/%s_*.json" % (model, prefix))):
        run = os.path.basename(path)[:-5]
        rows.append((model, run, json.load(open(path))))
if not rows:
    print("no results"); raise SystemExit

def ranked_row(model, run):
    f = per_period(model, run)
    pay = payout_series(f).dropna()
    dead = int(degenerate_periods(f).sum()) if "corr" in f else 0
    return dead, len(pay), charged_corr(f).mean(), f["mmc"].mean(), pay.mean()

print("%-34s %-6s %5s %5s %10s %10s %10s" % (
    "run", "model", "n", "dead", "CORR", "MMC", "payout"))
print("-" * 96)
ranked = [(m, r, ranked_row(m, r)) for m, r, _ in rows]
for model, run, (dead, n, corr, mmc, pay) in sorted(ranked, key=lambda x: -x[2][4]):
    print("%-34s %-6s %5d %5d %10.6f %10.6f %10.6f%s" % (
        run, model, n, dead, corr, mmc, pay, "  <-- COLLAPSED" if dead else ""))
print("-" * 96)
print("n differs BY DESIGN: base arms score ~510 periods, stacks the ~153-date")
print("tail. Payout is rebuilt PER PERIOD and a `dead` (constant-prediction)")
print("period is charged the 0 it earned, not dropped. Read the paired tables")
print("below for the comparison -- they share dates.")

if scout:
    print()
    print("SCOUT RUN -- no comparison printed, and none is available.")
    print("--era-step 4 keeps every 4th period and moves the split boundary, so")
    print("these arms share neither the sample nor the split with any baseline,")
    print("and the stack panel they build is a different window again.")
    raise SystemExit

def find(run):
    for model, name, _ in rows:
        if name == run:
            return model, name
    return None

def table(title, pairs):
    print()
    print(title)
    print("%-34s %-34s %8s %11s %9s %7s" % (
        "arm", "control", "metric", "delta", "se", "t"))
    print("-" * 108)
    any_row = False
    for arm_run, ctl_run in pairs:
        arm, ctl = find(arm_run), find(ctl_run)
        if arm is None or ctl is None:
            missing = arm_run if arm is None else ctl_run
            print("%-34s %-34s  %s not on disk -- NOT COMPARED" % (
                arm_run, ctl_run, missing))
            continue
        try:
            a, b = per_period(*arm), per_period(*ctl)
        except FileNotFoundError:
            print("%-34s %-34s  per-period missing -- NOT COMPARED" % (arm_run, ctl_run))
            continue
        any_row = True
        for metric in ("corr", "mmc", "payout"):
            left, right = series(b, metric), series(a, metric)
            j = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"]).dropna()
            delta, se, t = newey_west(j["a"] - j["b"])
            print("%-34s %-34s %8s %+11.6f %9.6f %+7.2f  (n=%d)" % (
                arm_run, ctl_run, metric, delta, se, t, len(j)))
    if not any_row:
        print("  (nothing to compare)")
    print("-" * 108)

# Inner-CV variants present in this run. "" is the purged arm, "_kf" the kfold
# twin. Tables 1-3 repeat for each so a kfold arm is judged by the same bars as
# its purged sibling; a missing arm just prints NOT COMPARED.
VARIANTS = [("", "purged"), ("_kf", "kfold")]

# 1. the headline: a fitted meta against the fixed average the book already uses
table("1. DOES A FITTED META BEAT A FIXED 1/N?   paired, Newey-West lag 20",
      [("%s_meta_%s_%s%s" % (prefix, m, s, suf), "%s_ctl_eq_%s" % (prefix, s))
       for s in sets for m in ("ridge", "mlp") for suf, _ in VARIANTS])

# 2. the real bar: against walk-forward weights that are merely constrained
table("2. DOES IT BEAT CONSTRAINED NNLS?         paired, Newey-West lag 20",
      [("%s_meta_%s_%s%s" % (prefix, m, s, suf), "%s_ctl_nnls_%s" % (prefix, s))
       for s in sets for m in ("ridge", "mlp") for suf, _ in VARIANTS])

# 3. is the meta-learner's non-linearity worth anything over a signed linear one
table("3. DOES NON-LINEARITY BUY ANYTHING?       mlp meta vs ridge meta, lag 20",
      [("%s_meta_mlp_%s%s" % (prefix, s, suf), "%s_meta_ridge_%s%s" % (prefix, s, suf))
       for s in sets for suf, _ in VARIANTS])

# 4. Wolpert's k-fold inner CV against the purged one -- same meta, same members,
# same OUTER walk-forward. A null here says the time-series purge cost nothing on
# the meta level; a positive delta says the purge was leaving signal on the table.
table("4. DOES WOLPERT'S KFOLD INNER-CV CHANGE IT?   _kf vs purged, same arm, lag 20",
      [("%s_meta_%s_%s_kf" % (prefix, m, s), "%s_meta_%s_%s" % (prefix, m, s))
       for s in sets for m in ("ridge", "mlp")])

print()
print("READ IT THIS WAY")
print("  Table 2 is the finding, not table 1. Beating 1/N only says the members")
print("  were unequal; beating NNLS says the non-negativity constraint was")
print("  costing something, which is the actual claim stacking makes.")
print()
print("  If only stackX wins, the gain is function-class diversity rather than")
print("  stacking -- stackR and stackM are what separate those.")
print()
print("  ~153 periods at ~95% overlap is roughly 8 effective independent windows,")
print("  so |t| < 2 is the EXPECTED outcome and IS the finding. With 6 meta arms a")
print("  Bonferroni-style adjustment puts the honest threshold nearer |t| = 2.6.")
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
