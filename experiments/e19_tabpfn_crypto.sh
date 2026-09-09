#!/usr/bin/env bash
#
# Experiment E19: TabPFN (tabular foundation model) at defaults, per submitted dataset
#
# E19 is the reference handle; every artefact it writes is e19_* in
# results/{metrics,forecast,per_period,hyperparams}/crypto/tabpfn/.
#
# WHAT THIS RUNS
#   `tabpfn` -- Prior-Labs TabPFN v2 regressor (a pre-trained transformer that
#   solves a dataset by IN-CONTEXT learning: no per-dataset training, no search)
#   -- at the CURUPIRA cadence, one arm per submitted slot's dataset. Same nine
#   as E18. TabPFN needs none of the tuning TabNet did; the only knobs are cost
#   knobs (context size, ensemble members), set below.
#
#   The 5 NARROW datasets are the default set. The 4 WIDE ones (763-812 features)
#   exceed TabPFN v2's ~500-feature design range and are listed but NOT in the
#   default TABPFN_ARMS -- run them only with feature reduction, and read the
#   result knowing it is outside the model's trained regime.
#
# TWO THINGS THAT ARE NOT LIKE THE OTHER ARMS
#   1. IT WANTS A GPU. TabPFN's forward pass attends every query row to the whole
#      context, so cost is O(context x query x n_estimators). On CPU a single
#      1,000-context / 300-query predict at n_estimators=1 is ~11 s, and a full
#      refit predicts tens of thousands of rows -- hours per arm on CPU, minutes
#      on GPU. Set TABPFN_DEVICE=cuda on a GPU box; the defaults below are the
#      CHEAPEST CPU settings and produce a context-limited, single-estimator
#      TabPFN that UNDERSELLS the model -- say so when reporting a CPU run.
#   2. THE WEIGHTS ARE VENDORED. The pip package downloads its checkpoint from a
#      gated HuggingFace repo, which a restricted-egress box cannot do. This uses
#      the ungated v2 regressor checkpoint at models/tabpfn/tabpfn-v2-regressor.ckpt
#      (44 MB), so nothing here touches the network.
#
# CONTEXT SUBSAMPLING IS A REAL CHOICE, NOT A DETAIL
#   Each 39,000-row training block is subsampled to TABPFN_MAX_CONTEXT rows before
#   TabPFN sees it (it is both outside the prior's range and too slow otherwise).
#   The model therefore sees a SAMPLE of each block. Larger context = higher
#   fidelity and linearly higher cost. It is written into the run's config.
#
# WHAT WOULD COUNT AS A RESULT
#   Same as E18: compare each arm to the deployed slot on the SAME dataset with a
#   paired Newey-West test (crypto-experiment-reporting / arm-board skill). A
#   default-parameter foundation model beating a tuned booster would be the
#   headline; on a small, noisy, pre-binned panel it is a genuine open question,
#   which is why it is worth one honest arm even though TabNet lost.
#
# Usage:
#   TABPFN_SCOUT=1 bash experiments/e19_tabpfn_crypto.sh          # wiring, minutes
#   TABPFN_DEVICE=cuda bash experiments/e19_tabpfn_crypto.sh      # the real run, on GPU
#   TABPFN_MAX_CONTEXT=1000 TABPFN_N_EST=1 bash experiments/e19_tabpfn_crypto.sh  # cheap CPU
#   TABPFN_ARMS="raw tenc" / ONLY= / SKIP= / DRY_RUN=1 / NO_MAIL=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E19}"
TABPFN_WINDOW="${TABPFN_WINDOW:-130}"
TABPFN_REFIT="${TABPFN_REFIT:-130}"
TABPFN_EMBARGO="${TABPFN_EMBARGO:-20}"
TABPFN_N_SPLITS="${TABPFN_N_SPLITS:-2}"      # nothing is searched; 2 is the floor
TABPFN_SEED="${TABPFN_SEED:-42}"
TABPFN_THREADS="${TABPFN_THREADS:-8}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cpu}"        # set cuda on a GPU box
TABPFN_MAX_CONTEXT="${TABPFN_MAX_CONTEXT:-4096}"
TABPFN_N_EST="${TABPFN_N_EST:-4}"
TABPFN_CKPT="${TABPFN_CKPT:-$REPO/models/tabpfn/tabpfn-v2-regressor.ckpt}"
TABPFN_ARMS="${TABPFN_ARMS:-raw tenc interact fenccausal cumlag3}"
TABPFN_ERA_STEP=1
case "${TABPFN_WINDOW}" in ""|0|[0-9]) echo "TABPFN_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${TABPFN_REFIT}"  in ""|0|[0-9]) echo "TABPFN_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e19"
if [ "${TABPFN_SCOUT:-0}" = "1" ]; then
    PREFIX="e19scout"; TABPFN_ERA_STEP=16; TABPFN_WINDOW=80; TABPFN_REFIT=80
    TABPFN_MAX_CONTEXT=400; TABPFN_N_EST=1
    echo "SCOUT MODE: era-step 16, window 80, context 400, 1 estimator. Wiring only."
fi

export OMP_NUM_THREADS="${TABPFN_THREADS}"
export MKL_NUM_THREADS="${TABPFN_THREADS}"
export OPENBLAS_NUM_THREADS="${TABPFN_THREADS}"

MODEL_OPTIONS="{\"n_estimators\":${TABPFN_N_EST},\"max_context\":${TABPFN_MAX_CONTEXT},\"device\":\"${TABPFN_DEVICE}\",\"model_path\":\"${TABPFN_CKPT}\"}"
# Pin the one grid key so hyper_params_search collapses to a single candidate.
FIXED_PARAMS="{\"n_estimators\":${TABPFN_N_EST}}"

# name / dataset / expected feature count / >500 feat (outside TabPFN range) / slot
NAME=(); DATA=(); NFEAT=(); WIDE=(); SLOTS=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); WIDE+=("$4"); SLOTS+=("$5"); }
add raw          "crypto/v2.0"                                     22 0 "CURUPIRA_02, SACI_04"
add tenc         "crypto/v2.0-tenc"                                23 0 "CURUPIRA_01"
add interact     "crypto/v2.0-interact"                            36 0 "SACI_01"
add fenccausal   "crypto/v2.0-fenccausal"                          44 0 "IA_01 -> CU_SA_IA_BO_01"
add cumlag3      "crypto/v2.0-cumlag3"                             88 0 "SACI_05"
add tenc_onehot  "crypto/v2.0-tenc-onehot"                        763 1 "SACI_03"
add allfeat      "crypto/v2.0-tenc-onehot-spread-interact"        788 1 "SACI_02"
add onehot_roll  "crypto/v2.0-onehot-rollsel"                     786 1 "CURUPIRA_03"
add allfeat_pp   "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 1 "CURUPIRA_04"

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [[ " ${TABPFN_ARMS} " != *" ${arm} "* ]] && continue
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${WIDE[$i]}" = "1" ] && [ "${TABPFN_ALLOW_WIDE:-0}" != "1" ]; then
        echo "note: skipping ${arm} (${NFEAT[$i]} feat > TabPFN's ~500 range); set TABPFN_ALLOW_WIDE=1 to force"
        continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected (TABPFN_ARMS='${TABPFN_ARMS}')" >&2; exit 1; }

# Preflight: datasets at expected counts, torch+tabpfn present, checkpoint on disk.
python3 - "${TABPFN_CKPT}" "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import sys, os
import pyarrow.parquet as pq
ckpt = sys.argv[1]
parts, cur = [], []
for tok in sys.argv[2:]:
    if tok == "--": parts.append(cur); cur = []
    else: cur.append(tok)
parts.append(cur)
versions, wants = parts; wants = [int(x) for x in wants]
bad = []
for v, want in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    if not os.path.exists(p): bad.append("%s MISSING" % v); continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if got != want: bad.append("%s has %d features, expected %d" % (v, got, want))
if not os.path.exists(ckpt):
    bad.append("checkpoint MISSING: %s -- download the ungated Prior-Labs/"
               "TabPFN-v2-reg regressor there" % ckpt)
try:
    import torch, tabpfn  # noqa: F401
except ImportError as e:
    bad.append("torch/tabpfn missing (%s) -- OPTIONAL in setup.sh" % e)
from src.prediction.models import MODELS, supported_tasks
if "tabpfn" not in MODELS: bad.append("tabpfn not registered in MODELS")
elif "regression" not in supported_tasks("tabpfn"): bad.append("tabpfn dropped regression")
if bad: sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) OK; checkpoint present; torch+tabpfn present; "
      "tabpfn registered" % len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      is a default tabular FOUNDATION model (TabPFN v2) competitive per dataset
  model     tabpfn  --  in-context, no training; context ${TABPFN_MAX_CONTEXT}, ${TABPFN_N_EST} estimator(s)
  device    ${TABPFN_DEVICE}  (CUDA strongly preferred; CPU is hours/arm and context-limited)
  cadence   rolling ${TABPFN_WINDOW}/${TABPFN_REFIT}, embargo ${TABPFN_EMBARGO}, ${TABPFN_N_SPLITS} folds, no search

  runner    experiments/e19_tabpfn_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/tabpfn/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt"
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed start')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: tabpfn (foundation model, defaults) on ${total} dataset(s)"
echo "device ${TABPFN_DEVICE}, context ${TABPFN_MAX_CONTEXT}, ${TABPFN_N_EST} estimator(s)"
echo "cadence rolling ${TABPFN_WINDOW}/${TABPFN_REFIT}, embargo ${TABPFN_EMBARGO}, ${TABPFN_N_SPLITS} folds"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

run_arm() {                          # run_name, data_version
    local run_name="$1" data_version="$2"; index=$((index+1))
    local cmd=(python3 -u -m src.run_forecast crypto tabpfn
               --task regression --metric corr --bmc-reference benchmark
               --cv-scheme rolling --train-window "${TABPFN_WINDOW}"
               --refit-every "${TABPFN_REFIT}" --embargo "${TABPFN_EMBARGO}"
               --n-splits "${TABPFN_N_SPLITS}" --era-step "${TABPFN_ERA_STEP}"
               --seed "${TABPFN_SEED}" --search-jobs 1
               --data-version "${data_version}" --run-name "${run_name}"
               --model-options "${MODEL_OPTIONS}" --fixed-params "${FIXED_PARAMS}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-24s %s\n' "$index" "$total" "$run_name" "$data_version"; return 0
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${run_name}  (${data_version})  $(date '+%H:%M:%S')"
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

for i in "${!SEL_NAME[@]}"; do run_arm "${PREFIX}_${SEL_NAME[$i]}" "${SEL_DATA[$i]}"; done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- TabPFN (defaults) on submitted datasets  (context ${TABPFN_MAX_CONTEXT}, ${TABPFN_N_EST} est, ${TABPFN_DEVICE})"
echo "  results  results/metrics/crypto/tabpfn/${PREFIX}_*.json"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} in $((el_all/60))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo
python3 - "${PREFIX}" <<'PY'
import glob, json, os, sys
prefix = sys.argv[1]
rows = [(os.path.basename(p)[:-5], json.load(open(p)))
        for p in sorted(glob.glob("results/metrics/crypto/tabpfn/%s_*.json" % prefix))]
if not rows: print("no results"); raise SystemExit
print("%-22s %6s %6s %10s %10s %8s %10s" % ("run","feat","n","CORR","MMC","c.shrp","payout"))
print("-"*74)
for run, d in rows:
    print("%-22s %6d %6d %10.6f %10.6f %8.3f %10.6f" % (
        run, d["config"]["n_features"], d["n_periods"], d["corr"]["mean"],
        d["mmc"]["mean"], d["corr"]["sharpe"], 0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
print("-"*74)
print("CPU/context-limited settings UNDERSELL TabPFN. Compare each arm to the")
print("deployed slot on the same dataset with a paired Newey-West test.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
