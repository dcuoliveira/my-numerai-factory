#!/usr/bin/env bash
#
# Experiment E18: TabNet at library defaults, on every submitted slot's dataset
#
# E18 is the reference handle. Everything this study writes is greppable as E18
# and as e18_* in results/{metrics,forecast,per_period,hyperparams}/crypto/tabnet/.
#
# WHAT THIS RUNS
#   `tabnet` -- dreamquark-ai's pytorch-tabnet behind a StandardScaler, added to
#   prediction/models.py (src/prediction/tabnet.py) -- at the CURUPIRA cadence,
#   ONE arm per DISTINCT training dataset that a submitted slot actually fits on.
#   The blend slots fit nothing (they rank-average members), so the datasets that
#   feed submitted models are the nine their leaf members train on:
#
#     e18_raw          crypto/v2.0                                       22
#     e18_tenc         crypto/v2.0-tenc                                  23
#     e18_interact     crypto/v2.0-interact                             36
#     e18_fenccausal   crypto/v2.0-fenccausal                           44
#     e18_cumlag3      crypto/v2.0-cumlag3                              88
#     e18_tenc_onehot  crypto/v2.0-tenc-onehot                         763
#     e18_allfeat      crypto/v2.0-tenc-onehot-spread-interact         788
#     e18_onehot_roll  crypto/v2.0-onehot-rollsel                      786
#     e18_allfeat_pp   crypto/v2.0-tenc-onehot-rollsel-spread-interact 812
#
#   Arms run narrow -> wide, so raw (the cheapest and the apples-to-apples
#   comparison against the MLP / IA_01) lands first and a run cut short still
#   answers the interesting question.
#
# THE ESTIMATOR IS PINNED TO THE LIBRARY DEFAULTS, AND THAT IS THE POINT
#   --fixed-params holds every knob at pytorch-tabnet's own defaults (n_d = n_a =
#   8, n_steps 3, gamma 1.3, n_independent 2, n_shared 2, lambda_sparse 1e-3,
#   momentum 0.02, sparsemax, Adam lr 0.02, 100 epochs, batch 1024, ghost batch
#   128). Nothing is searched. This is a VIABILITY PROBE, not a tuned model: the
#   panel is small and noisy for a net this expressive (~39,000 rows per refit,
#   effective sample ~130 periods), so the honest first question is whether the
#   default configuration is even in the game against the boosters and the MLP,
#   before any tuning budget is justified. See src/prediction/tabnet.py.
#
# WHY n_splits = 2, NOT 5
#   With fixed params the inner CV selects nothing -- it only computes a CV score
#   the walk-forward result does not use. The per-period pseudo-OOS series is
#   therefore byte-identical for n_splits 2 and 5 (verified: max|diff| = 0.0), so
#   2 is used purely to halve the wasted fits. The reported numbers are unchanged.
#
# NO eval_set, SO THE FIT IS LEAK-FREE
#   pytorch-tabnet's early stopping watches an eval_set; drawing one from inside a
#   fold would pull the holdout from the same periods it trains on, the leak
#   purged_period_splits exists to prevent. The wrapper passes no eval_set and
#   trains the full 100 epochs, the same posture MLPWrapper takes with
#   early_stopping=False.
#
# WHAT WOULD COUNT AS A RESULT
#   This study is not paired against a single baseline -- it spans nine datasets
#   on different feature counts. Read each arm against the deployed slot that
#   trains on the SAME dataset (raw -> CURUPIRA_02/SACI_04, tenc -> CURUPIRA_01,
#   ...), which the arm-board / crypto-experiment-reporting skills do properly
#   with paired Newey-West on the common dates. A default-param net BEATING a
#   tuned booster on any of these would be the headline; matching it would
#   justify a tuning study; losing everywhere would retire the model class.
#   Read risk beside level -- arms here move level and drawdown independently.
#
# ---------------------------------------------------------------------------
# COST -- read this before launching
# ---------------------------------------------------------------------------
#
# Measured on this box (CPU only, no GPU) at TABNET_THREADS=8:
#   raw  (22 feat)  ~3.1 s/epoch  -> ~5 min/fit  -> ~1 h/arm
#   wide (812 feat) ~10 s/epoch   -> ~17 min/fit -> ~3.5 h/arm
# A pinned arm is n_splits(2) folds + 1 final = 3 fits per refit x 4 refits = 12
# fits (the CV folds train on less, so this is an upper bound). Nine arms total
# roughly 19 h SEQUENTIAL on an idle box; longer when the box is loaded.
#
# It has no --n-jobs: torch threads through OMP_NUM_THREADS, pinned here so the
# run does not trample the daily 9am live crypto crons or any other job. The
# launcher below nices itself. Raise TABNET_THREADS on an idle machine.
#
# SCOUT FIRST. TABNET_SCOUT=1 keeps every 4th period and cuts epochs/folds; it is
# a WIRING check only and its prefix changes so it can never be read as a result.
#
# Usage:
#   TABNET_SCOUT=1 bash experiments/e18_tabnet_crypto.sh   # ~minutes, wiring
#   bash experiments/e18_tabnet_crypto.sh                  # all nine arms
#   DRY_RUN=1 bash experiments/e18_tabnet_crypto.sh
#   TABNET_ARMS="raw tenc" bash experiments/e18_tabnet_crypto.sh
#   ONLY="raw" / SKIP="allfeat_pp" / NO_MAIL=1 / TABNET_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed. GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E18}"
TABNET_WINDOW="${TABNET_WINDOW:-130}"
TABNET_REFIT="${TABNET_REFIT:-130}"
TABNET_EMBARGO="${TABNET_EMBARGO:-20}"
TABNET_N_SPLITS="${TABNET_N_SPLITS:-2}"      # 2 by design; see header
TABNET_SEED="${TABNET_SEED:-42}"
TABNET_THREADS="${TABNET_THREADS:-8}"
TABNET_MAX_EPOCHS="${TABNET_MAX_EPOCHS:-100}"  # library default
TABNET_ARMS="${TABNET_ARMS:-raw tenc interact fenccausal cumlag3 tenc_onehot allfeat onehot_roll allfeat_pp}"
TABNET_ERA_STEP=1
case "${TABNET_WINDOW}" in ""|0|[0-9]) echo "TABNET_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${TABNET_REFIT}"  in ""|0|[0-9]) echo "TABNET_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

# A scout is a WIRING check, never a result. The prefix changes with it so a
# downsampled run can never be mistaken for a full-resolution arm in the tree.
PREFIX="e18"
if [ "${TABNET_SCOUT:-0}" = "1" ]; then
    PREFIX="e18scout"; TABNET_ERA_STEP=4; TABNET_N_SPLITS=2; TABNET_MAX_EPOCHS=5
    echo "SCOUT MODE: era-step 4, ${TABNET_MAX_EPOCHS} epochs, ${TABNET_N_SPLITS} folds."
    echo "            Results are NOT comparable to anything. Wiring only."
fi

# torch reads these at import; they are the only knobs that bound its threads.
export OMP_NUM_THREADS="${TABNET_THREADS}"
export MKL_NUM_THREADS="${TABNET_THREADS}"
export OPENBLAS_NUM_THREADS="${TABNET_THREADS}"

# The pinned configuration = pytorch-tabnet's library defaults, written out in
# full so the hyperparams table records exactly what ran. max_epochs is a var so
# the scout can shorten it. n_a is tied to n_d, as the paper ties them.
FIXED_PARAMS=$(cat <<JSON
{"net__n_d":8,"net__n_a":8,"net__n_steps":3,"net__gamma":1.3,"net__n_independent":2,"net__n_shared":2,"net__lambda_sparse":0.001,"net__momentum":0.02,"net__mask_type":"sparsemax","net__learning_rate":0.02,"net__max_epochs":${TABNET_MAX_EPOCHS},"net__patience":10,"net__batch_size":1024,"net__virtual_batch_size":128}
JSON
)

# Parallel arrays, never pipe-delimited records: nothing here contains '|', but
# the convention keeps this runner identical in shape to e07/e09.
# name / dataset / expected feature count / the submitted slot(s) on that dataset
NAME=(); DATA=(); NFEAT=(); SLOTS=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); SLOTS+=("$4"); }
add raw          "crypto/v2.0"                                     22 "CURUPIRA_02, SACI_04"
add tenc         "crypto/v2.0-tenc"                                23 "CURUPIRA_01"
add interact     "crypto/v2.0-interact"                            36 "SACI_01"
add fenccausal   "crypto/v2.0-fenccausal"                          44 "IA_01 -> CU_SA_IA_BO_01"
add cumlag3      "crypto/v2.0-cumlag3"                             88 "SACI_05"
add tenc_onehot  "crypto/v2.0-tenc-onehot"                        763 "SACI_03"
add allfeat      "crypto/v2.0-tenc-onehot-spread-interact"        788 "SACI_02"
add onehot_roll  "crypto/v2.0-onehot-rollsel"                     786 "CURUPIRA_03"
add allfeat_pp   "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 "CURUPIRA_04"

# Select the arms to run, keeping the narrow -> wide order.
SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=(); SEL_SLOTS=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [[ " ${TABNET_ARMS} " != *" ${arm} "* ]] && continue
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}")
    SEL_NFEAT+=("${NFEAT[$i]}"); SEL_SLOTS+=("${SLOTS[$i]}")
done
if [ "${#SEL_NAME[@]}" -eq 0 ]; then
    echo "no arms selected (TABNET_ARMS='${TABNET_ARMS}')" >&2; exit 1
fi

# Preflight. A silent feature-count drift attaches a plausible number to the
# wrong experiment, which is worse than a crash -- so read the parquet schema and
# refuse on a mismatch. torch must be present (it is OPTIONAL in setup.sh) and
# tabnet must be registered, or every arm would fail identically hours apart.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
import os

parts, cur = [], []
for tok in sys.argv[1:]:
    if tok == "--":
        parts.append(cur); cur = []
    else:
        cur.append(tok)
parts.append(cur)
versions, wants = parts
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

try:
    import torch  # noqa: F401
except ImportError as error:
    bad.append("torch is not installed (%s) -- OPTIONAL in setup.sh, needed by "
               "tabnet" % error)
try:
    import pytorch_tabnet  # noqa: F401
except ImportError as error:
    bad.append("pytorch-tabnet is not installed (%s)" % error)

from src.prediction.models import MODELS, supported_tasks
if "tabnet" not in MODELS:
    bad.append("tabnet is not registered in src.prediction.models.MODELS")
elif "regression" not in supported_tasks("tabnet"):
    bad.append("tabnet no longer declares regression support")

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts; torch and "
      "pytorch-tabnet present; tabnet registered" % len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

# Mail on START as well as at the end: a long unattended job that only mails on
# completion is indistinguishable from one that never started.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      is default-parameter TabNet in the game on each submitted slot's dataset
  model     tabnet  --  pytorch-tabnet behind a StandardScaler, PINNED to library defaults
  cadence   rolling ${TABNET_WINDOW}/${TABNET_REFIT}, embargo ${TABNET_EMBARGO}, ${TABNET_N_SPLITS} folds, no search, ${TABNET_MAX_EPOCHS} epochs
  threads   ${TABNET_THREADS}  (nice'd; device cpu)
  expect    ~1 h/narrow arm, ~3.5 h/wide arm; ~19 h total on an idle box, longer when loaded

Where to look while it runs:
  runner    experiments/e18_tabnet_crypto.sh
  progress  logs/${PREFIX}_*.log        (one per arm)
  results   results/metrics/crypto/tabnet/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt   (written at the end)
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: tabnet (library defaults) on ${total} dataset(s)"
echo "cadence: rolling ${TABNET_WINDOW}/${TABNET_REFIT}, embargo ${TABNET_EMBARGO}, ${TABNET_N_SPLITS} folds, ${TABNET_MAX_EPOCHS} epochs, ${TABNET_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

run_arm() {                          # run_name, data_version, extra args...
    local run_name="$1" data_version="$2"; shift 2
    index=$((index+1))
    local cmd=(python3 -u -m src.run_forecast crypto tabnet
               --task regression --metric corr --bmc-reference benchmark
               --cv-scheme rolling --train-window "${TABNET_WINDOW}"
               --refit-every "${TABNET_REFIT}" --embargo "${TABNET_EMBARGO}"
               --n-splits "${TABNET_N_SPLITS}" --era-step "${TABNET_ERA_STEP}"
               --seed "${TABNET_SEED}" --search-jobs 1
               --data-version "${data_version}" --run-name "${run_name}"
               --fixed-params "${FIXED_PARAMS}" "$@")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-24s %s\n' "$index" "$total" "$run_name" "$data_version"; return 0
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  (${data_version})  $(date '+%H:%M:%S')"
    echo "=============================================================="
    local started; started=$(date +%s)
    # A failed arm must not cost the others: read PIPESTATUS[0] after the tee.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    local status="${PIPESTATUS[0]}" el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
}

for i in "${!SEL_NAME[@]}"; do
    run_arm "${PREFIX}_${SEL_NAME[$i]}" "${SEL_DATA[$i]}"
done

[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- TabNet at library defaults, one arm per submitted slot's dataset"
echo "  results  results/metrics/crypto/tabnet/${PREFIX}_*.json"
echo "  runner   experiments/e18_tabnet_crypto.sh"
echo "  logs     logs/${PREFIX}_*.log"
echo "=============================================================="
elapsed_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((elapsed_all/60))m$((elapsed_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${TABNET_SCOUT:-0}" <<'PY'
import glob, json, os, sys

prefix, scout = sys.argv[1], sys.argv[2] == "1"
rows = []
for path in sorted(glob.glob("results/metrics/crypto/tabnet/%s_*.json" % prefix)):
    run = os.path.basename(path)[:-5]
    rows.append((run, json.load(open(path))))

if not rows:
    print("no results on disk"); raise SystemExit

print("%-22s %6s %7s %10s %10s %8s %8s %10s" % (
    "run", "feat", "n", "CORR", "MMC", "c.shrp", "dd(corr)", "payout"))
print("-" * 88)
for run, d in rows:
    print("%-22s %6d %7d %10.6f %10.6f %8.4f %8.4f %10.6f" % (
        run, d["config"]["n_features"], d["n_periods"],
        d["corr"]["mean"], d["mmc"]["mean"], d["corr"]["sharpe"],
        d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
print("-" * 88)
print("payout = 0.1*CORR + MMC, rebuilt from the two means (a RANKING aid only;")
print("for a claim, rebuild it per-period -- see the arm-board skill).")
print("Datasets carry different feature counts and different targets are NOT")
print("mixed here (all fit target_binned_return_20). Compare each arm to the")
print("deployed slot on the SAME dataset with a paired Newey-West test; use the")
print("crypto-experiment-reporting / arm-board-reporting skill to do it properly.")

if scout:
    print()
    print("SCOUT RUN -- era-step 4 moved the split boundary and cut epochs, so")
    print("these arms are a wiring check and share no sample with any baseline.")
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
