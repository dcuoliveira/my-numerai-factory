#!/usr/bin/env bash
#
# Experiment E08: the MLP on the four ONE-HOT datasets the book actually runs on
#
# E08 is the reference handle; every artefact it writes is e08_*.
#
# WHAT THIS RUNS, AND WHY IT IS FOUR AND NOT TWELVE
#   E08 was planned as the MLP across the 12 datasets ridge was run on. E15 has
#   since covered six of them -- raw, tenc, spread, interact, fenccausal,
#   rollsel -- on BOTH backends, and cumlag3 too. What is left is the four the
#   deployed book uses that no MLP arm has ever touched, and they are the four
#   carrying the 740 one-hot ticker dummies:
#
#     e08_<backend>_onehot_rollsel   v2.0-onehot-rollsel                786  CURUPIRA_03
#     e08_<backend>_tenc_onehot      v2.0-tenc-onehot                   763  SACI_03
#     e08_<backend>_allfeat          v2.0-tenc-onehot-spread-interact   788  SACI_02
#     e08_<backend>_allfeatpp        v2.0-tenc-onehot-rollsel-spread-interact 812  CURUPIRA_04
#
#   Both backends, so 8 arms. That completes MLP coverage of every dataset the
#   submitted book trains on: 5 of 9 were already done, these are the other 4.
#
# THESE ARMS WERE BLOCKED ON A PREPROCESSING BUG, NOW FIXED
#   MLPWrapper used to standardise EVERY column. Standardising a binary column
#   divides it by sqrt(p(1-p)), so on a real 130-period block of v2.0-onehot:
#
#       the raw 22 features after scaling   max |z| =   2.29
#       the 740 ticker dummies              max |z| = 197.48
#
#   A ticker on 3 rows of 39,000 arrived 86x larger than anything real, and 346
#   of the 740 dummies do not occur at all in a given block. That is silent: the
#   run completes and scores worse, which reads as "the one-hot block does not
#   help" rather than "the preprocessing broke it". ScaleContinuousOnly now
#   passes 0/1 columns through untouched -- max |z| 197.48 -> 2.29 -- detecting
#   them from the DATA, not from the feature_sym_ prefix.
#
#   On a dataset with no dummies it is exactly StandardScaler, so every arm run
#   before the fix still reproduces. Only these four were ever affected, and no
#   MLP arm had been run on them. There is therefore nothing to re-run.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta against that backend's raw arm
#   (e15_mlp_raw / e15_torch_mlp_raw), Newey-West lag 20. Four comparisons per
#   backend puts the honest bar near |t| = 2.5.
#
#   Cross-backend agreement is the real guard: E09 showed sklearn and torch agree
#   on raw (t = +0.17, prediction rank corr 0.9364), so a block that moves ONE
#   and not the other is search noise. E12 and E14 both produced |t| > 2 that a
#   per-refit breakdown later traced to a single bad draw.
#
# THE PRIOR, AND WHY THE ARMS ARE STILL WORTH RUNNING
#   Against: build_symbol_embedding.py records both ticker encodings COSTING
#   ridge accuracy -- CORR 0.173 raw, 0.156 tenc, 0.142 onehot -- and E11/E12
#   found learned ticker embeddings null to harmful.
#
#   For: no model class has been able to USE these columns. Ridge cannot form a
#   per-ticker interaction from a dummy; LightGBM discards them outright, and
#   fe_lgb_reg_onehot came back BYTE-IDENTICAL to fe_lgb_reg_raw
#   (max|diff| = 0.00e+00) because at min_child_samples 313 no split can use one.
#   A dense first layer over a one-hot IS an embedding lookup, so the MLP is the
#   first arm here that can. That is the question, and it is unanswered.
#
# COST
#   ~90 min per arm (full 50-draw search; an MLP's cost is dominated by its
#   hidden layers, not its width -- 786 columns benchmarked at roughly the same
#   per-fit cost as 22). ~12 h sequential, ~3 h across 4 shards.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e08_mlp_onehot_crypto.sh
#   SHARD=1 OF=4 bash experiments/e08_mlp_onehot_crypto.sh
#   ONLY=tenc_onehot / BACKENDS=mlp / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=6
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E08}"
OH_WINDOW="${OH_WINDOW:-130}"
OH_REFIT="${OH_REFIT:-130}"
OH_EMBARGO="${OH_EMBARGO:-20}"
OH_N_SPLITS="${OH_N_SPLITS:-5}"
OH_N_ITER="${OH_N_ITER:-50}"
OH_VAL_FRAC="${OH_VAL_FRAC:-0.3}"
OH_SEED="${OH_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-6}"
BACKENDS="${BACKENDS:-mlp torch_mlp}"
OH_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${OH_WINDOW}" in ""|0|[0-9]) echo "OH_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${OH_REFIT}"  in ""|0|[0-9]) echo "OH_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e08"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e08scout"; OH_ERA_STEP=4; OH_N_ITER=4; OH_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${OH_N_ITER} draws, ${OH_N_SPLITS} folds -- wiring only."
fi

# Neither MLP has an --n-jobs: sklearn's threads through BLAS, torch's through
# its own pool. These are the only knobs that bound either.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
# arm / dataset / expected features / the slot that runs on it
NAME=(); DATA=(); NFEAT=(); SLOT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); SLOT+=("$4"); }
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                      786 CURUPIRA_03
add tenc_onehot    "crypto/v2.0-tenc-onehot"                         763 SACI_03
add allfeat        "crypto/v2.0-tenc-onehot-spread-interact"         788 SACI_02
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 CURUPIRA_04

W_ARM=(); W_MODEL=(); W_DATA=(); W_NFEAT=()
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
        W_ARM+=("${arm}"); W_MODEL+=("${backend}")
        W_DATA+=("${DATA[$i]}"); W_NFEAT+=("${NFEAT[$i]}")
    done
done
[ "${#W_ARM[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. The scaler is checked too: without
# ScaleContinuousOnly these four datasets are exactly the ones that break, and
# nothing downstream would reveal it.
python3 - "${W_DATA[@]}" -- "${W_NFEAT[@]}" <<'PY' || exit 1
import json, os, sys
import numpy as np
import pyarrow.parquet as pq

from src.prediction.models import MLPWrapper, ScaleContinuousOnly, build_wrapper

cut = sys.argv.index("--")
versions, wants = sys.argv[1:cut], [int(x) for x in sys.argv[cut + 1:]]

bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    try:
        names = pq.ParquetFile(path).schema.names
    except OSError:
        bad.append("%s MISSING -- build the chain with setup.sh --with-derived"
                   % version)
        continue
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % version):
        bad.append("%s has no historical_meta_models.parquet -- the loader would "
                   "try to download it and fail AFTER the run starts" % version)
    dummies = sum(c.startswith("feature_sym_") for c in names)
    if dummies < 100:
        bad.append("%s carries only %d one-hot columns; this study is about the "
                   "740-dummy datasets" % (version, dummies))

for backend in sorted(set(sys.argv[1:0]) | {"mlp", "torch_mlp"}):
    try:
        step = build_wrapper(backend, "regression", seed=42, n_jobs=1).ModelClass.steps[0][1]
    except Exception as error:
        bad.append("%s did not construct: %s" % (backend, error)); continue
    if not isinstance(step, ScaleContinuousOnly):
        bad.append("%s still scales with %s. On these datasets that sends a rare "
                   "ticker dummy to z = 197 beside features inside +-2.3, and the "
                   "run would complete looking merely unhelpful"
                   % (backend, type(step).__name__))

ceiling = float(MLPWrapper.ALPHAS.max())
if abs(ceiling - 10.0) > 1e-9:
    bad.append("MLPWrapper.ALPHAS tops out at %g, not 1e1 -- the raw controls "
               "were run on the 1e1 grid" % ceiling)

for control in ("mlp/e15_mlp_raw", "torch_mlp/e15_torch_mlp_raw"):
    if not os.path.exists("results/metrics/crypto/%s.json" % control):
        bad.append("control %s is not on disk; every delta is against it" % control)

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d run(s) at the expected feature counts, all carrying the "
      "740 dummies; both backends use ScaleContinuousOnly; alpha ceiling 1e1; "
      "both raw controls present" % len(versions))
PY

total="${#W_ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  runs      ${total}  (shard ${SHARD}/${OF})
  asks      can an MLP use the 740 one-hot ticker dummies that ridge cannot form
            an interaction from and LightGBM discards outright?
  covers    the last 4 of 9 datasets the submitted book trains on
  controls  e15_mlp_raw and e15_torch_mlp_raw, per backend
  cadence   rolling ${OH_WINDOW}/${OH_REFIT}, embargo ${OH_EMBARGO}, ${OH_N_SPLITS} folds, ${OH_N_ITER} draws
  expect    ~90 min per arm; ~12 h sequential, ~3 h across 4 shards
  note      these arms were blocked on a scaler bug (rare dummies reaching
            z = 197); ScaleContinuousOnly fixed it, 197.48 -> 2.29

Where to look while it runs:
  runner    experiments/e08_mlp_onehot_crypto.sh
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
echo "${EXPERIMENT}: the MLP on the one-hot book datasets  |  ${total} run(s)  |  shard ${SHARD}/${OF}"
echo "backends: ${BACKENDS}   cadence: rolling ${OH_WINDOW}/${OH_REFIT}, ${OH_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!W_ARM[@]}"; do
    arm="${W_ARM[$i]}"; backend="${W_MODEL[$i]}"; index=$((index+1))
    run_name="${PREFIX}_${backend}_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${backend}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${OH_WINDOW}"
         --refit-every "${OH_REFIT}" --embargo "${OH_EMBARGO}"
         --n-splits "${OH_N_SPLITS}" --n-iter "${OH_N_ITER}"
         --val-frac "${OH_VAL_FRAC}" --era-step "${OH_ERA_STEP}"
         --seed "${OH_SEED}" --search-jobs 1
         --data-version "${W_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-34s %-10s %3s feat  %s\n' "$index" "$total" \
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
echo "${EXPERIMENT} -- the MLP on the one-hot datasets the book runs on"
echo "  results  results/metrics/crypto/{mlp,torch_mlp}/${PREFIX}_*.json"
echo "  runner   experiments/e08_mlp_onehot_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} run(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD} of ${OF} -- PARTIAL; other shards hold the rest"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
ARMS = [("onehot_rollsel", "CURUPIRA_03"), ("tenc_onehot", "SACI_03"),
        ("allfeat", "SACI_02"), ("allfeatpp", "CURUPIRA_04")]
BACKENDS = [("mlp", "e15_mlp_raw"), ("torch_mlp", "e15_torch_mlp_raw")]

def load(model, run):
    p = "results/metrics/crypto/%s/%s.json" % (model, run)
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(model, run):
    return pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run)).set_index("date")

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

print("%-16s %-12s %5s | %10s %8s %8s | %10s %8s %8s" % (
    "arm", "slot", "feat", "skl payout", "sharpe", "maxdd",
    "trc payout", "sharpe", "maxdd"))
print("-" * 96)
for model, control in BACKENDS:
    d = load(model, control)
    if d:
        print("%-16s %-12s %5d | %10.6f %8.3f %8.3f |" % (
            control, "control", d["config"]["n_features"],
            0.1 * d["corr"]["mean"] + d["mmc"]["mean"],
            d["corr"]["sharpe"], d["corr"]["max_drawdown"]))
for arm, slot in ARMS:
    cells, feat = [], None
    for model, _ in BACKENDS:
        d = load(model, "%s_%s_%s" % (prefix, model, arm))
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
        print("%-16s %-12s   -   | %-30s %-30s" % (arm, slot, "not on disk", "not on disk"))
        continue
    print("%-16s %-12s %5d | %10s %8s %8s | %10s %8s %8s" % (
        arm, slot, feat, cells[0][0] or "-", cells[0][1] or "-", cells[0][2] or "-",
        cells[1][0] or "-", cells[1][1] or "-", cells[1][2] or "-"))
print("-" * 96)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit

print()
print("paired vs each backend's OWN raw arm, Newey-West lag 20")
print("%-16s | %11s %7s | %11s %7s | %s" % (
    "arm", "skl dpay", "t", "trc dpay", "t", "agree?"))
print("-" * 80)
for arm, _ in ARMS:
    cells, signs = [], []
    for model, control in BACKENDS:
        run = "%s_%s_%s" % (prefix, model, arm)
        if load(model, run) is None or load(model, control) is None:
            cells.append(("", "")); signs.append(None); continue
        j = pd.concat([payout(per_period(model, control)), payout(per_period(model, run))],
                      axis=1, join="inner", keys=["b", "a"])
        delta, _, t = newey_west(j["a"] - j["b"])
        cells.append(("%+.6f" % delta, "%+.2f" % t)); signs.append(delta > 0)
    if signs[0] is None or signs[1] is None:
        verdict = "one backend missing"
    elif signs[0] == signs[1]:
        verdict = "yes"
    else:
        verdict = "NO -- one backend only, treat as noise"
    print("%-16s | %11s %7s | %11s %7s | %s" % (
        arm, cells[0][0] or "-", cells[0][1] or "-",
        cells[1][0] or "-", cells[1][1] or "-", verdict))
print("-" * 80)
print()
print("Four comparisons per backend puts the honest bar near |t| = 2.5. A block")
print("that moves one backend and not the other is search noise: E09 showed the")
print("two agree on raw (t = +0.17, prediction rank corr 0.9364).")
print()
print("These are the columns ridge cannot form an interaction from and LightGBM")
print("discards outright -- fe_lgb_reg_onehot came back BYTE-IDENTICAL to raw.")
print("A dense first layer over a one-hot IS an embedding lookup, so this is the")
print("first arm that can use them. Prior is still against: both ticker encodings")
print("cost ridge accuracy (CORR 0.173 raw, 0.156 tenc, 0.142 onehot).")
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
