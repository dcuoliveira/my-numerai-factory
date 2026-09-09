#!/usr/bin/env bash
#
# Experiment E16: cumulative DIFFERENCES of the features and target, on the MLP
#
# E16 is the reference handle; every artefact it writes is e16_*.
#
# WHAT THIS RUNS
#   Ten arms of sklearn `mlp` regression, each fully searched, sweeping the
#   difference depth in two variants:
#
#     e16_diff<p>     crypto/v2.0-cumdiff<p>      differences ALONE
#     e16_rawdiff<p>  crypto/v2.0-rawcumdiff<p>   raw 22 + the differences
#
#     p       1     2     3     5    10
#     diff   23    46    69   115   230   columns
#     rawdiff 45    68    91   137   252
#
#   "Cumulative" means every depth up to p is kept, as build_lagged's --through
#   does for lags: cumdiff 3 carries x_t - x_{t-1}, x_t - x_{t-2} and
#   x_t - x_{t-3}, so p columns per feature.
#
#   The standalone arms answer "is the difference the signal?"; the raw+ arms
#   answer "does the difference ADD to what the level already carries". Those are
#   different questions and the answers can disagree -- a block can be
#   informative alone and redundant beside the raw features.
#
# THE CONTROL IS e15_mlp_raw, NOT A NEW ARM
#   E15's raw arm is the same model, cadence, folds, seed, alpha grid and
#   dataset, so refitting it here would verify determinism rather than
#   correctness -- and cost 90 minutes. The preflight asserts its config matches
#   this study's field by field and refuses otherwise. Same pattern E07 used
#   against CURUPIRA_02's backtest run.
#
# THE TARGET DIFFERENCE IS ANCHORED, AND THAT IS THE WHOLE RISK
#   target_binned_return_20 at row t is a FORWARD return over [t, t+20]:
#   target_t - target_{t-k} IS THE LABEL. The builder anchors at the horizon,
#   feature_target_cumdiff<k> = target_{t-20} - target_{t-20-k}, so both ends
#   have closed by t. Measured, not asserted:
#
#       anchor  0 (unanchored)   corr with target = +0.2805
#       anchor 20 (what we use)  corr with target = -0.0114
#
#   The preflight recomputes that correlation on every dataset rather than
#   trusting build time -- a dataset can be rebuilt with different flags and
#   nothing downstream would notice. Feature differences need no anchor: a
#   feature is a level observed at t.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta against e15_mlp_raw, Newey-West lag 20.
#   TEN comparisons puts the Bonferroni-style bar near |t| = 2.8; quote both.
#
#   Read the SHAPE across depths, not any single arm. E12 is the cautionary
#   tale: six arms differing only in one hyperparameter came out non-monotone
#   with a spread larger than the effect, and the whole thing later traced to a
#   single bad draw. A depth sweep that wanders is measuring the search.
#
#   And compare the two variants at each depth. If diff<p> is strong but
#   d<p> is not, the differences carry the same information as the levels
#   rather than adding to it.
#
# THE PRIOR
#   Unkind, and worth stating before the numbers arrive. E05 swept cumulative
#   LAGS 1..10 on LightGBM and the best depth by payout was ZERO. E06 found them
#   harmful (t = -2.18 to -2.30). E13 found feature lags flat for this MLP
#   (t = +0.06) and E14 found the ADL combination negative at t = -2.59. A
#   difference is a different transform of the same past, so it earns its own
#   test -- but four studies across three model classes say the past does not
#   help on this panel.
#
# COST
#   ~90 min per arm (full 50-draw search), so ~15 h sequential. SHARD=/OF= is how
#   this is meant to run; 3 shards is ~5 h. NOTE E15 may still be using the box --
#   check before launching, and lower MLP_THREADS if so.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e16_cumdiff_mlp_crypto.sh    # ~12 min
#   SHARD=1 OF=3 bash experiments/e16_cumdiff_mlp_crypto.sh
#   ONLY="diff1 rawdiff1" / SKIP=diff10 / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=6
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E16}"
CD_WINDOW="${CD_WINDOW:-130}"
CD_REFIT="${CD_REFIT:-130}"
CD_EMBARGO="${CD_EMBARGO:-20}"
CD_N_SPLITS="${CD_N_SPLITS:-5}"
CD_N_ITER="${CD_N_ITER:-50}"
CD_VAL_FRAC="${CD_VAL_FRAC:-0.3}"
CD_SEED="${CD_SEED:-42}"
CD_MODEL="${CD_MODEL:-mlp}"
CD_CONTROL="${CD_CONTROL:-e15_mlp_raw}"
MLP_THREADS="${MLP_THREADS:-6}"
CD_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${CD_WINDOW}" in ""|0|[0-9]) echo "CD_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${CD_REFIT}"  in ""|0|[0-9]) echo "CD_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e16"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e16scout"; CD_ERA_STEP=4; CD_N_ITER=4; CD_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${CD_N_ITER} draws, ${CD_N_SPLITS} folds -- wiring only."
fi

# sklearn's MLP has no --n-jobs; it threads through BLAS and these are the only
# knobs that bound it.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
# The LADDER, not every integer. A month is 22 dates on Crypto so a year is 264,
# and every difference order from 1 to 264 would add 264 x 22 = 5,808 feature
# columns -- more columns than the panel has rows per date, burying the raw 22
# completely. One rung per horizon a trader would name instead:
#
#     1 day | 5 week | 22 month | 66 quarter | 132 half-year | 264 year
#
# Each rung is CUMULATIVE over the rungs below it, so e16d66 carries the 1, 5, 22
# and 66 differences -- 23 columns per rung.
#
# Datasets and columns are namespaced v2.0-e16d* / _e16d<k> by
# src/data_mani/build_e16_diffladder.py, which is NOT this study's file: a
# concurrent session in this checkout owns it and chose that namespace so two
# definitions of "cumulative difference" could not silently mix in one table.
# This runner consumes them and builds nothing.
DEPTHS="${CD_DEPTHS:-1 5 22 66 132 264}"
declare -A RUNG_FEAT=([1]=23 [5]=46 [22]=69 [66]=92 [132]=115 [264]=138)
NAME=(); DATA=(); NFEAT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
for p in ${DEPTHS}; do
    n="${RUNG_FEAT[$p]:-}"
    [ -z "$n" ] && { echo "depth ${p} is not a ladder rung (1 5 22 66 132 264)" >&2; exit 1; }
    add "d${p}only" "crypto/v2.0-e16d${p}only" "${n}"
    add "d${p}"     "crypto/v2.0-e16d${p}"     $(( 22 + n ))
done

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=()
picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1))
        [ $(( (picked - 1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. Three things, each of which would silently produce a different
# experiment than the one documented: a feature-count drift, an unanchored target
# difference, and a control that is not matched.
python3 - "${CD_CONTROL}" "${CD_MODEL}" "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import json, os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_mani.utils import get_spec, target_horizon_days
from src.prediction.models import MLPWrapper

control, model = sys.argv[1], sys.argv[2]
cut = sys.argv.index("--")
versions, wants = sys.argv[3:cut], [int(x) for x in sys.argv[cut + 1:]]
target = get_spec("crypto")["target"]
horizon = target_horizon_days(target)

bad, rebuild = [], []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    try:
        names = pq.ParquetFile(path).schema.names
    except OSError:
        # Not fatal any more: the arm loop rebuilds an absent rung with
        # build_e16_diffladder before running it. Still reported, because a
        # rung that has to be rebuilt costs minutes and the anchoring checks
        # below cannot run on it until it exists.
        rebuild.append(version)
        continue
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))

    # A dataset directory needs the meta-model file too, or load_tournament_data
    # falls through to numerapi and dies on "the requested file does not exist"
    # AFTER the run has started. Three arms failed that way in the scout because
    # this check did not exist -- train.parquet alone is not a complete dataset.
    meta = "data/%s/historical_meta_models.parquet" % version
    if not os.path.exists(meta):
        bad.append("%s has no historical_meta_models.parquet -- the loader would "
                   "try to download it and fail. Copy it from data/crypto/v2.0/"
                   % version)

    # Recomputed, not trusted from build time: a dataset can be rebuilt with
    # different flags and nothing downstream would notice. Unanchored target
    # differences correlate +0.28 with the label; resolved ones sit near -0.01.
    diffs = [c for c in names if c.startswith("feature_target_cumdiff")]
    if diffs:
        frame = pd.read_parquet(path, columns=diffs + [target])
        for column in diffs:
            both = frame[[column, target]].dropna()
            rho = float(np.corrcoef(both[column], both[target])[0, 1])
            if abs(rho) > 0.25:
                bad.append("%s: corr(%s, %s) = %+.4f -- that column is NOT "
                           "anchored at the %d-period horizon and contains the "
                           "label" % (version, column, target, rho, horizon))

path = "results/metrics/crypto/%s/%s.json" % (model, control)
if not os.path.exists(path):
    bad.append("control %s/%s is not on disk. Every comparison in this study is "
               "against it -- run E15's raw arm, or set CD_CONTROL" % (model, control))
else:
    config = json.load(open(path))["config"]
    for key, want in (("task", "regression"), ("cv_scheme", "rolling"),
                      ("train_window", 130), ("refit_every", 130),
                      ("embargo", 20), ("n_splits", 5), ("n_iter", 50),
                      ("seed", 42), ("era_step", 1),
                      ("data_version", "crypto/v2.0")):
        if config.get(key) != want:
            bad.append("control %s has %s=%r, this study runs %r -- not matched"
                       % (control, key, config.get(key), want))

ceiling = float(MLPWrapper.ALPHAS.max())
if abs(ceiling - 10.0) > 1e-9:
    bad.append("MLPWrapper.ALPHAS tops out at %g, not 1e1 -- the control was run "
               "on the 1e1 grid and these arms would not be comparable" % ceiling)

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
if rebuild:
    print("note: %d dataset(s) not on disk and will be REBUILT by the arm loop "
          "with build_e16_diffladder (the concurrent LightGBM ladder study "
          "removes each version after its arm): %s"
          % (len(rebuild), ", ".join(rebuild)))
    print("      their anchoring is re-checked by that builder's own preflight.")
print("preflight OK: %d dataset(s) at the expected feature counts; every target "
      "difference is anchored at the %d-period horizon; control %s is matched; "
      "alpha ceiling is 1e1" % (len(versions) - len(rebuild), horizon, control))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})  shard ${SHARD}/${OF}
  asks      do cumulative DIFFERENCES of the features and target help the MLP,
            alone and beside the raw levels?
  control   ${CD_MODEL}/${CD_CONTROL} -- reused, not refitted; preflight asserts it matches
  cadence   rolling ${CD_WINDOW}/${CD_REFIT}, embargo ${CD_EMBARGO}, ${CD_N_SPLITS} folds, ${CD_N_ITER} draws
  expect    ~90 min per arm; ~15 h sequential, ~5 h across 3 shards
  safety    target differences anchored at the 20-period horizon. Unanchored
            they correlate +0.2805 with the label; anchored, -0.0114.
  prior     against -- E05 best lag depth ZERO, E06 t=-2.18..-2.30, E13 lags
            flat (t=+0.06), E14 ADL negative (t=-2.59)

Where to look while it runs:
  runner    experiments/e16_cumdiff_mlp_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/${CD_MODEL}/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: cumulative differences  |  ${total} arm(s)  |  shard ${SHARD}/${OF}"
echo "model ${CD_MODEL}  control ${CD_CONTROL}  cadence rolling ${CD_WINDOW}/${CD_REFIT}, ${CD_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto "${CD_MODEL}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${CD_WINDOW}"
         --refit-every "${CD_REFIT}" --embargo "${CD_EMBARGO}"
         --n-splits "${CD_N_SPLITS}" --n-iter "${CD_N_ITER}"
         --val-frac "${CD_VAL_FRAC}" --era-step "${CD_ERA_STEP}"
         --seed "${CD_SEED}" --search-jobs 1
         --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-18s %4s feat  %s\n' "$index" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${SEL_DATA[$i]}"; continue
    fi

    # Build the dataset if it is absent. The concurrent LightGBM study on this
    # same ladder (e16_diffladder_lgb_reg_crypto.sh) removes each version after
    # its arm finishes, so a rung that existed at preflight can be gone by the
    # time this loop reaches it. Rebuilding here with THEIR builder is what keeps
    # the two studies on one definition of the block instead of two.
    if [ ! -f "data/${SEL_DATA[$i]}/train.parquet" ]; then
        rung="${SEL_NAME[$i]}"; rung="${rung#d}"; drop=""
        case "${rung}" in *only) rung="${rung%only}"; drop="--drop-originals";; esac
        echo ">>> ${SEL_DATA[$i]} absent -- rebuilding (rung ${rung} ${drop:-with raw})"
        if ! python3 -m src.data_mani.build_e16_diffladder --through "${rung}" ${drop} \
                >> "logs/${PREFIX}_build.log" 2>&1; then
            failed=$((failed+1)); FAILED_ARMS+=("${run_name} (build)")
            echo ">>> ${run_name} BUILD FAILED -- see logs/${PREFIX}_build.log"
            continue
        fi
        # The builder writes train.parquet but not always the meta-model file,
        # and the loader falls through to numerapi and dies on "the requested
        # file does not exist" AFTER the run starts. Three scout arms died that
        # way; copying it here is cheaper than rediscovering it each time.
        [ -f "data/${SEL_DATA[$i]}/historical_meta_models.parquet" ] || \
            cp data/crypto/v2.0/historical_meta_models.parquet \
               "data/${SEL_DATA[$i]}/" 2>/dev/null || true
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
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- cumulative differences of the features and target, on the MLP"
echo "  results  results/metrics/crypto/${CD_MODEL}/${PREFIX}_*.json"
echo "  runner   experiments/e16_cumdiff_mlp_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD} of ${OF} -- PARTIAL; other shards hold the rest"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" "${CD_MODEL}" "${CD_CONTROL}" "${DEPTHS}" <<'PY'
import json, os, sys
import numpy as np
import pandas as pd

prefix, scout, model, control = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4]
depths = [int(x) for x in sys.argv[5].split()]

def load(run, mdl=None):
    p = "results/metrics/crypto/%s/%s.json" % (mdl or model, run)
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(run, mdl=None):
    return pd.read_csv("results/per_period/crypto/%s/%s.csv"
                       % (mdl or model, run)).set_index("date")

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

ctl = load(control)
print("%-16s %5s %6s %10s %10s %8s %8s %10s %7s" % (
    "run", "feat", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 92)
if ctl:
    print("%-16s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f %7.1f  <- control" % (
        control, ctl["config"]["n_features"], ctl["n_periods"],
        ctl["corr"]["mean"], ctl["mmc"]["mean"], ctl["corr"]["sharpe"],
        ctl["corr"]["max_drawdown"],
        0.1 * ctl["corr"]["mean"] + ctl["mmc"]["mean"],
        ctl.get("runtime_minutes", 0)))
for kind in ("only", "raw"):
    for p in depths:
        run = "%s_d%d%s" % (prefix, p, "only" if kind == "only" else "")
        d = load(run)
        if d is None:
            # Never a blank row: absent reads as "tested and null" when it means
            # "never ran" -- a sharded run legitimately has gaps.
            print("%-16s  NOT ON DISK -- not run (or another shard holds it)" % run)
            continue
        print("%-16s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f %7.1f" % (
            run, d["config"]["n_features"], d["n_periods"], d["corr"]["mean"],
            d["mmc"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"],
            0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 92)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit
if ctl is None:
    print("\nThe control is not on disk, so no delta is printed.")
    raise SystemExit

print()
print("paired vs %s, Newey-West lag 20" % control)
print("%-6s | %11s %7s | %11s %7s" % (
    "depth", "diff-only dpay", "t", "raw+diff dpay", "t"))
print("-" * 52)
base = per_period(control)
curves = {"only": [], "raw": []}
for p in depths:
    cells = []
    for kind in ("d{p}only", "d{p}"):
        run = "%s_%s%d" % (prefix, kind, p)
        if load(run) is None:
            cells.append(("", "")); continue
        j = pd.concat([payout(base), payout(per_period(run))],
                      axis=1, join="inner", keys=["b", "a"])
        delta, _, t = newey_west(j["a"] - j["b"])
        curves[kind].append((p, delta))
        cells.append(("%+.6f" % delta, "%+.2f" % t))
    print("%-6d | %11s %7s | %11s %7s" % (
        p, cells[0][0] or "-", cells[0][1] or "-",
        cells[1][0] or "-", cells[1][1] or "-"))
print("-" * 52)

for kind, label in (("only", "differences alone"), ("raw", "raw + differences")):
    pts = curves[kind]
    if len(pts) < 3:
        continue
    vals = [v for _, v in pts]
    falling = all(b <= a for a, b in zip(vals, vals[1:]))
    rising = all(b >= a for a, b in zip(vals, vals[1:]))
    best = pts[int(np.argmax(vals))][0]
    print()
    if falling:
        print("SHAPE (%s): falls monotonically with depth -- more history is a"
              " pure cost." % label)
    elif rising:
        print("SHAPE (%s): rises monotonically -- depth %d is the grid edge, not"
              " an optimum. Extend before quoting a preferred depth."
              % (label, max(depths)))
    else:
        print("SHAPE (%s): wanders, peaking at depth %d. A depth sweep that is"
              % (label, best))
        print("  not monotone is usually measuring the SEARCH, not the columns --")
        print("  E12 looked exactly like this and traced to one bad draw.")

print()
print("Compare the two columns at each depth. If d<p>only is strong where")
print("d<p> is not, the differences carry what the levels already carry")
print("rather than adding to it.")
print()
print("~26 effective independent windows, so |t| < 2 is EXPECTED and IS the")
print("finding. TEN comparisons puts the Bonferroni-style bar near |t| = 2.8.")
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
