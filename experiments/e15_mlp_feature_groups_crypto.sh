#!/usr/bin/env bash
#
# Experiment E15: the MLP feature table -- 7 raw groups + 6 derived blocks, twice
#
# E15 is the reference handle; every artefact it writes is e15_*.
#
# WHAT THIS RUNS
#   14 feature sets, each fitted by BOTH MLP backends, every arm given its own
#   full 50-draw search. 28 runs.
#
#     raw          crypto/v2.0                22   all shipped features (control)
#     price        --feature-include on v2.0        close avg/ewa
#     size                                          market cap avg/ewa
#     liquidity                                     volume avg/ewa
#     oscillator                                    RSI + sharpe ratio
#     bollinger                                     bollinger
#     momentum                                      momentum
#     volatility                                    volatility
#     tenc         crypto/v2.0-tenc           23   + causal mean-target encoding
#     spread       crypto/v2.0-spread         33   + cross-horizon spreads
#     interact     crypto/v2.0-interact       36   + smoothing interactions
#     fenccausal   crypto/v2.0-fenccausal     44   + causal feature encoding
#     rollsel      crypto/v2.0-rollsel        46   + selected rolling means
#     cumlag3      crypto/v2.0-cumlag3        88   + cumulative lags 1..3
#
#   The seven groups PARTITION the raw 22 -- the preflight refuses to run unless
#   they cover it exactly with no overlap, because a silent gap would make every
#   number here describe a different experiment than the one documented.
#
# WHY BOTH BACKENDS
#   Two things for one study. Per feature set it doubles the evidence: E09
#   established that sklearn and torch agree on raw (payout delta t = +0.17,
#   prediction rank correlation 0.9364), so a block that moves ONE backend and
#   not the other is search noise, not a property of the columns. Across the
#   table it extends that replication from one feature set to fourteen.
#
#   This matters more than usual here. E12 and E14 both produced |t| > 2 results
#   that a per-refit breakdown later traced to a single bad hyperparameter draw.
#   Cross-backend agreement is a cheaper guard against that than any t-statistic.
#
# THE ALPHA GRID IS 1e1, AND THAT IS LOAD-BEARING
#   E07 pinned against a 1e1 ceiling so it was raised to 1e2. That was wrong: on
#   the same 514 periods a pinned alpha of 8.2 pays 0.082438 while the 1e2 search
#   choosing 39-79 pays 0.075302 -- the CV optimum keeps moving right while
#   out-of-sample payout falls. Worse, the entire negative verdict of E11 and E12
#   traced to the one refit where the search took alpha 79. The ceiling is back
#   at 1e1 and the preflight asserts it. E09/E11/E12 numbers were made on the 1e2
#   grid and are NOT comparable to these.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta against that backend's own raw arm,
#   Newey-West lag 20. THIRTEEN comparisons per backend, so the Bonferroni-style
#   bar is near |t| = 3.0 -- quote both, and treat a lone |t| just over 2 as
#   noise unless the other backend agrees.
#
#   The prior is unkind. Eleven feature blocks have been tried on this panel and
#   none beat the raw 22 by a margin the sample supports. The groups have been
#   run alone before on ridge; the derived blocks on ridge and LightGBM. Nothing
#   here is expected to clear the bar. The value is a complete MLP row of a table
#   the other two model classes already have -- see E08's motivation, which this
#   supersedes for the sets it covers.
#
# COST
#   Each arm is a full search: ~1,000 fits. An MLP's cost is dominated by its
#   hidden layers, not its input width (786 columns benchmarked at roughly the
#   same per-fit cost as 22), so a 2-feature group arm costs about what raw does.
#   sklearn took 92 min on raw, torch 67. Budget ~80 min per arm: roughly 37
#   hours sequential.
#
#   SHARD=/OF= is how this is meant to be run. Four shards is ~9-10 h wall clock
#   at 6 threads each on a 32-core box:
#       for s in 1 2 3 4; do SHARD=$s OF=4 MLP_THREADS=6 bash experiments/e15_... & done
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e15_mlp_feature_groups_crypto.sh   # ~15 min
#   SHARD=1 OF=4 bash experiments/e15_mlp_feature_groups_crypto.sh
#   ONLY="raw tenc" / SKIP=cumlag3 / BACKENDS=torch_mlp / DRY_RUN=1 / NO_MAIL=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E15}"
FG_WINDOW="${FG_WINDOW:-130}"
FG_REFIT="${FG_REFIT:-130}"
FG_EMBARGO="${FG_EMBARGO:-20}"
FG_N_SPLITS="${FG_N_SPLITS:-5}"
FG_N_ITER="${FG_N_ITER:-50}"
FG_VAL_FRAC="${FG_VAL_FRAC:-0.3}"
FG_SEED="${FG_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-6}"
BACKENDS="${BACKENDS:-mlp torch_mlp}"
FG_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${FG_WINDOW}" in ""|0|[0-9]) echo "FG_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${FG_REFIT}"  in ""|0|[0-9]) echo "FG_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e15"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e15scout"; FG_ERA_STEP=4; FG_N_ITER=4; FG_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${FG_N_ITER} draws, ${FG_N_SPLITS} folds -- wiring only."
fi

# Neither MLP has an --n-jobs: sklearn's threads through BLAS, torch's through
# its own pool. These are the only knobs that bound either.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: several of these regexes contain
# '|' as alternation and would be torn apart by IFS='|' splitting.
# arm / dataset / --feature-include regex ("" = every feature) / expected count
NAME=(); DATA=(); INCLUDE=(); NFEAT=()
add() { NAME+=("$1"); DATA+=("$2"); INCLUDE+=("$3"); NFEAT+=("$4"); }

add raw        "crypto/v2.0" ""                                         22
add price      "crypto/v2.0" '^feature_close_(avg|ewa)_'                 4
add size       "crypto/v2.0" '^feature_market_cap_(avg|ewa)_'            4
add liquidity  "crypto/v2.0" '^feature_volume_(avg|ewa)_'                4
add oscillator "crypto/v2.0" '^feature_(relative_strength_index|sharpe_ratio)_' 4
add bollinger  "crypto/v2.0" '^feature_bollinger_'                       2
add momentum   "crypto/v2.0" '^feature_momentum_'                        2
add volatility "crypto/v2.0" '^feature_volatility_'                      2
add tenc       "crypto/v2.0-tenc"       ""                              23
add spread     "crypto/v2.0-spread"     ""                              33
add interact   "crypto/v2.0-interact"   ""                              36
add fenccausal "crypto/v2.0-fenccausal" ""                              44
add rollsel    "crypto/v2.0-rollsel"    ""                              46
add cumlag3    "crypto/v2.0-cumlag3"    ""                              88

# Build the full (feature set x backend) work list, THEN shard it -- sharding the
# feature sets instead would put both backends of a set in the same shard and
# lose the point of running them concurrently.
W_ARM=(); W_MODEL=(); W_DATA=(); W_INC=(); W_NFEAT=()
picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    for backend in ${BACKENDS}; do
        if [ "${OF}" -gt 0 ]; then
            picked=$((picked+1))
            [ $(( (picked - 1) % OF + 1 )) -ne "${SHARD}" ] && continue
        fi
        W_ARM+=("${arm}"); W_MODEL+=("${backend}"); W_DATA+=("${DATA[$i]}")
        W_INC+=("${INCLUDE[$i]}"); W_NFEAT+=("${NFEAT[$i]}")
    done
done
[ "${#W_ARM[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. Two things that would each silently produce a different experiment
# than the one documented: a feature-count drift, and groups that do not exactly
# partition the raw 22.
python3 - "${MLP_THREADS}" "${W_DATA[@]}" -- "${W_INC[@]}" -- "${W_NFEAT[@]}" <<'PY' || exit 1
import re, sys
import pyarrow.parquet as pq

from src.prediction.models import MLPWrapper, supported_tasks

parts, cur = [], []
for tok in sys.argv[2:]:
    if tok == "--": parts.append(cur); cur = []
    else: cur.append(tok)
parts.append(cur)
versions, includes, wants = parts
wants = [int(x) for x in wants]

bad = []
for version, include, want in zip(versions, includes, wants):
    path = "data/%s/train.parquet" % version
    try:
        names = [c for c in pq.ParquetFile(path).schema.names
                 if c.startswith("feature_")]
    except OSError:
        bad.append("%s MISSING" % version); continue
    got = len([c for c in names if re.search(include, c)]) if include else len(names)
    if got != want:
        bad.append("%s with include %r selects %d features, expected %d"
                   % (version, include or "(all)", got, want))

# The seven groups must partition the raw 22 exactly. A gap or an overlap makes
# every group number describe something other than "this group alone".
raw = [c for c in pq.ParquetFile("data/crypto/v2.0/train.parquet").schema.names
       if c.startswith("feature_")]
group_res = [inc for inc, ver in zip(includes, versions) if inc and ver == "crypto/v2.0"]
if group_res:
    seen = set()
    for pattern in set(group_res):
        hit = {c for c in raw if re.search(pattern, c)}
        if hit & seen:
            bad.append("group patterns overlap on %s" % sorted(hit & seen))
        seen |= hit
    if len(set(group_res)) == 7:
        missing = set(raw) - seen
        if missing:
            bad.append("the 7 groups do not cover %d raw feature(s): %s"
                       % (len(missing), sorted(missing)))

ceiling = float(MLPWrapper.ALPHAS.max())
if abs(ceiling - 10.0) > 1e-9:
    bad.append("MLPWrapper.ALPHAS tops out at %g, not 1e1. E11/E12's negative "
               "verdict traced entirely to the one refit that took alpha 79 on "
               "the 1e2 grid; see ALPHAS" % ceiling)
if "regression" not in supported_tasks("torch_mlp"):
    bad.append("torch_mlp no longer declares regression support")
try:
    import torch                                          # noqa: F401
except ImportError as error:
    bad.append("torch is not installed (%s); the torch_mlp arms need it" % error)

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d run(s) at the expected feature counts; the raw groups "
      "partition cleanly; alpha ceiling is 1e1" % len(versions))
PY

total="${#W_ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  runs      ${total}  (shard ${SHARD}/${OF})
  asks      which feature sets suit an MLP -- 7 raw groups + 6 derived blocks,
            each fitted by BOTH sklearn and torch, every arm fully searched
  control   e15_<backend>_raw, per backend
  cadence   rolling ${FG_WINDOW}/${FG_REFIT}, embargo ${FG_EMBARGO}, ${FG_N_SPLITS} folds, ${FG_N_ITER} draws
  threads   ${MLP_THREADS}
  expect    ~80 min per arm; ~37 h sequential, ~9-10 h across 4 shards
  read      a block that moves ONE backend and not the other is search noise.
            13 comparisons per backend puts the honest bar near |t| = 3.

Where to look while it runs:
  runner    experiments/e15_mlp_feature_groups_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/{mlp,torch_mlp}/${PREFIX}_*.json
  report    logs/${PREFIX}_report*.txt
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: MLP feature table  |  ${total} run(s)  |  shard ${SHARD}/${OF}"
echo "backends: ${BACKENDS}   cadence: rolling ${FG_WINDOW}/${FG_REFIT}, ${FG_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!W_ARM[@]}"; do
    arm="${W_ARM[$i]}"; backend="${W_MODEL[$i]}"; index=$((index+1))
    run_name="${PREFIX}_${backend}_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${backend}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${FG_WINDOW}"
         --refit-every "${FG_REFIT}" --embargo "${FG_EMBARGO}"
         --n-splits "${FG_N_SPLITS}" --n-iter "${FG_N_ITER}"
         --val-frac "${FG_VAL_FRAC}" --era-step "${FG_ERA_STEP}"
         --seed "${FG_SEED}" --search-jobs 1
         --data-version "${W_DATA[$i]}" --run-name "${run_name}")
    [ -n "${W_INC[$i]}" ] && cmd+=(--feature-include "${W_INC[$i]}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-28s %-10s %3s feat  %s\n' "$index" "$total" \
               "$run_name" "$backend" "${W_NFEAT[$i]}" "${W_DATA[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${W_NFEAT[$i]} feat  $(date '+%H:%M:%S')"
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
echo "${EXPERIMENT} -- which feature sets suit an MLP, on both backends"
echo "  results  results/metrics/crypto/{mlp,torch_mlp}/${PREFIX}_*.json"
echo "  runner   experiments/e15_mlp_feature_groups_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} run(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD} of ${OF} -- PARTIAL; other shards hold the rest"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import glob, json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
ARMS = ["raw", "price", "size", "liquidity", "oscillator", "bollinger",
        "momentum", "volatility", "tenc", "spread", "interact", "fenccausal",
        "rollsel", "cumlag3"]
BACKENDS = [("mlp", "sklearn"), ("torch_mlp", "torch")]

def load(model, arm):
    p = "results/metrics/crypto/%s/%s_%s_%s.json" % (model, prefix, model, arm)
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(model, arm):
    return pd.read_csv("results/per_period/crypto/%s/%s_%s_%s.csv"
                       % (model, prefix, model, arm)).set_index("date")

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

print("%-12s %5s | %10s %8s %8s | %10s %8s %8s" % (
    "arm", "feat", "skl payout", "sharpe", "maxdd", "trc payout", "sharpe", "maxdd"))
print("-" * 84)
present = {}
for arm in ARMS:
    cells, feat = [], None
    for model, _ in BACKENDS:
        d = load(model, arm)
        present[(model, arm)] = d is not None
        if d is None:
            cells.append(("", "", ""))
        else:
            feat = d["config"]["n_features"]
            cells.append(("%.6f" % (0.1 * d["corr"]["mean"] + d["mmc"]["mean"]),
                          "%.3f" % d["corr"]["sharpe"],
                          "%.3f" % d["corr"]["max_drawdown"]))
    if feat is None:
        # Never a blank row: absent reads as "tested and null" when it means
        # "never ran" -- and a sharded run legitimately has gaps.
        print("%-12s %5s | %-30s %-30s" % (arm, "-", "not on disk", "not on disk"))
        continue
    print("%-12s %5d | %10s %8s %8s | %10s %8s %8s" % (
        arm, feat, cells[0][0] or "-", cells[0][1] or "-", cells[0][2] or "-",
        cells[1][0] or "-", cells[1][1] or "-", cells[1][2] or "-"))
print("-" * 84)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit

print()
print("paired vs each backend's OWN raw arm, Newey-West lag 20")
print("%-12s | %11s %7s | %11s %7s | %s" % (
    "arm", "skl dpay", "t", "trc dpay", "t", "agree?"))
print("-" * 76)
for arm in ARMS:
    if arm == "raw":
        continue
    cells, signs = [], []
    for model, _ in BACKENDS:
        if not (present.get((model, arm)) and present.get((model, "raw"))):
            cells.append(("", "")); signs.append(None); continue
        j = pd.concat([payout(per_period(model, "raw")),
                       payout(per_period(model, arm))],
                      axis=1, join="inner", keys=["b", "a"])
        delta, _, t = newey_west(j["a"] - j["b"])
        cells.append(("%+.6f" % delta, "%+.2f" % t)); signs.append(delta > 0)
    if signs[0] is None or signs[1] is None:
        verdict = "one backend missing"
    elif signs[0] == signs[1]:
        verdict = "yes"
    else:
        # The cheapest guard there is against the failure mode E12 and E14 both
        # hit: a |t| > 2 that a per-refit breakdown later traced to one draw.
        verdict = "NO -- one backend only, treat as noise"
    print("%-12s | %11s %7s | %11s %7s | %s" % (
        arm, cells[0][0] or "-", cells[0][1] or "-",
        cells[1][0] or "-", cells[1][1] or "-", verdict))
print("-" * 76)
print()
print("13 comparisons per backend puts the Bonferroni-style bar near |t| = 3.0,")
print("not 2.0. And a block that moves one backend and not the other is search")
print("noise: E09 showed the two agree on raw (t = +0.17, prediction rank corr")
print("0.9364), so disagreement here is about the draw, not the columns.")
print()
print("Eleven feature blocks have been tried on this panel and none beat the raw")
print("22 by a margin the sample supports. A null is the expected outcome and IS")
print("the finding -- report it with its interval.")
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
