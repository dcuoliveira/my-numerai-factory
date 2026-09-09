#!/usr/bin/env bash
#
# Experiment E24: can TabPFN be pushed past its E19 default? Three levers.
#
# E24 asks the "make TabPFN the best model" Tier-1 questions, each an arm that
# differs from the E19 baseline (regressor, context 4096, n_estimators 4,
# softmax_temperature 0.9) in ONE thing:
#
#   1. FULL STRENGTH  (prefix e22_reg_) -- regressor, context 8192, n_estimators
#      "auto". E19 deliberately clipped context to ~10% of each block and used a
#      quarter of the default ensemble to fit an 8 GB card; this is TabPFN closer
#      to how it is meant to be run. memory_saving_mode="auto" keeps context 8192
#      under 5 GB (the E19 cumlag3 OOM was without it).
#   2. CLASSIFIER HEAD  (prefix e22_clf_) -- TabPFNClassifier on the 5 target
#      bins, returning E[y|x]. Same context 4096 / n_estimators 4 as the E19
#      regressor, so the ONLY difference is the head. TabPFN was born a
#      classifier; this asks whether its stronger head reads this target better.
#   3. TEMPERATURE  (prefix e22_temp) -- the full-strength regressor on the best
#      dataset (fenccausal), softmax_temperature swept {0.5,0.75,1.25}. 0.9 is
#      already covered by the option-1 fenccausal arm; this brackets it.
#
# ALL ON GPU. TabPFN's forward pass is O(context x query); on CPU this is
# untenable. Requires TABPFN_DEVICE=cuda and a card. Runs must SERIALISE on one
# 8 GB GPU -- do not fan these out.
#
# WHAT IS MEASURED AGAINST WHAT
#   opt1 vs the E19 regressor arm on the same dataset  -> "does full strength help"
#   opt2 vs the E19 regressor arm on the same dataset  -> "does the classifier head help"
#   opt3 vs e22_reg_full_fenccausal                    -> "does temperature help"
#   and every arm against the deployed slot on its dataset, paired NW lag 20,
#   via the crypto-experiment-reporting skill.
#
# Usage (on the GPU box, base env active):
#   TABPFN_DEVICE=cuda bash experiments/e22_tabpfn_tuning_crypto.sh
#   TABPFN_ARMS="opt1 opt2" ... / ONLY= / SKIP= / DRY_RUN=1 / NO_MAIL=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E24}"
TABPFN_WINDOW="${TABPFN_WINDOW:-130}"
TABPFN_REFIT="${TABPFN_REFIT:-130}"
TABPFN_EMBARGO="${TABPFN_EMBARGO:-20}"
TABPFN_N_SPLITS="${TABPFN_N_SPLITS:-2}"
TABPFN_SEED="${TABPFN_SEED:-42}"
TABPFN_THREADS="${TABPFN_THREADS:-4}"
TABPFN_DEVICE="${TABPFN_DEVICE:-cpu}"
TABPFN_REG_CKPT="${TABPFN_REG_CKPT:-$REPO/models/tabpfn/tabpfn-v2-regressor.ckpt}"
TABPFN_CLF_CKPT="${TABPFN_CLF_CKPT:-$REPO/models/tabpfn/tabpfn-v2-classifier.ckpt}"
TABPFN_GROUPS="${TABPFN_GROUPS:-opt1 opt2 opt3}"
PREFIX="e24"

export OMP_NUM_THREADS="${TABPFN_THREADS}" MKL_NUM_THREADS="${TABPFN_THREADS}" \
       OPENBLAS_NUM_THREADS="${TABPFN_THREADS}" \
       PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Arm table: name | model | data_version | n_estimators | max_context | temp | group
NAME=(); MODEL=(); DATA=(); NEST=(); CTX=(); TEMP=(); GRP=()
add(){ NAME+=("$1"); MODEL+=("$2"); DATA+=("$3"); NEST+=("$4"); CTX+=("$5"); TEMP+=("$6"); GRP+=("$7"); }
# opt1: full-strength regressor on the 4 in-range datasets
add reg_full_raw        tabpfn     crypto/v2.0            auto 8192 0.9  opt1
add reg_full_tenc       tabpfn     crypto/v2.0-tenc       auto 8192 0.9  opt1
add reg_full_interact   tabpfn     crypto/v2.0-interact   auto 8192 0.9  opt1
add reg_full_fenccausal tabpfn     crypto/v2.0-fenccausal auto 8192 0.9  opt1
# opt2: classifier head at the E19 baseline settings (only the head differs)
add clf_raw             tabpfn_clf crypto/v2.0            4    4096 0.9  opt2
add clf_tenc            tabpfn_clf crypto/v2.0-tenc       4    4096 0.9  opt2
add clf_interact        tabpfn_clf crypto/v2.0-interact   4    4096 0.9  opt2
add clf_fenccausal      tabpfn_clf crypto/v2.0-fenccausal 4    4096 0.9  opt2
# opt3: temperature around 0.9, full-strength regressor, best dataset
add temp050_fenccausal  tabpfn     crypto/v2.0-fenccausal auto 8192 0.5  opt3
add temp075_fenccausal  tabpfn     crypto/v2.0-fenccausal auto 8192 0.75 opt3
add temp125_fenccausal  tabpfn     crypto/v2.0-fenccausal auto 8192 1.25 opt3

SEL=()
for i in "${!NAME[@]}"; do
    [[ " ${TABPFN_GROUPS} " != *" ${GRP[$i]} "* ]] && continue
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${NAME[$i]} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${NAME[$i]} "* ]] && continue
    SEL+=("$i")
done
[ "${#SEL[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight: datasets, both checkpoints, torch+tabpfn, both models registered.
python3 - "$TABPFN_REG_CKPT" "$TABPFN_CLF_CKPT" <<'PY' || exit 1
import os, sys
bad=[]
for ck in sys.argv[1:3]:
    if not os.path.exists(ck): bad.append("checkpoint MISSING: %s"%ck)
try:
    import torch, tabpfn  # noqa
except ImportError as e: bad.append("torch/tabpfn missing (%s)"%e)
from src.prediction.models import MODELS, supported_tasks
for m in ("tabpfn","tabpfn_clf"):
    if m not in MODELS: bad.append("%s not registered"%m)
    elif "regression" not in supported_tasks(m): bad.append("%s dropped regression"%m)
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: both checkpoints present, torch+tabpfn present, tabpfn+tabpfn_clf registered")
PY

total="${#SEL[@]}"; index=0; ran=0; failed=0; declare -a FAILED=(); started_all=$(date +%s)
if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 -c "import sys; from src.live.notify import send_text; send_text('[crypto] E24 started', sys.argv[1]); print('  emailed start')" \
    "E24 started $(date '+%F %T') on $(hostname): ${total} arms (${TABPFN_GROUPS}), device ${TABPFN_DEVICE}" \
    || echo "note: start email failed"
fi
echo "=============================================================="
echo "E24: TabPFN tuning -- ${total} arm(s), device ${TABPFN_DEVICE}, groups ${TABPFN_GROUPS}"
echo "started $(date '+%F %T')"; echo "=============================================================="

for i in "${SEL[@]}"; do
    name="${PREFIX}_${NAME[$i]}"; model="${MODEL[$i]}"; data="${DATA[$i]}"
    ck="$TABPFN_REG_CKPT"; [ "$model" = "tabpfn_clf" ] && ck="$TABPFN_CLF_CKPT"
    mopts="{\"n_estimators\":\"${NEST[$i]}\",\"max_context\":${CTX[$i]},\"softmax_temperature\":${TEMP[$i]},\"device\":\"${TABPFN_DEVICE}\",\"model_path\":\"${ck}\"}"
    # n_estimators may be the literal auto (unquoted in JSON it is invalid); the
    # wrapper accepts the string, so quote it. Numeric stays numeric-as-string,
    # which int() still parses.
    mopts="${mopts/\"auto\"/\"auto\"}"
    fixed="{\"max_context\":${CTX[$i]}}"
    index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto "$model"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "$TABPFN_WINDOW" --refit-every "$TABPFN_REFIT"
         --embargo "$TABPFN_EMBARGO" --n-splits "$TABPFN_N_SPLITS" --era-step 1
         --seed "$TABPFN_SEED" --search-jobs 1 --data-version "$data"
         --run-name "$name" --model-options "$mopts" --fixed-params "$fixed")
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '[%d/%d] %-24s %s | %s\n' "$index" "$total" "$name" "$data" "$mopts"; continue; fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  (${model} on ${data})  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${name}.log"
    st="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$name"); echo ">>> ${name} FAILED (exit $st) after $((el/60))m -- continuing"
    else echo ">>> ${name} done in $((el/60))m$((el%60))s"; fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "E24 -- TabPFN tuning (full-strength regressor / classifier head / temperature)"
el=$(( $(date +%s)-started_all )); echo "ran ${ran}/${total} in $((el/60))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"; echo
python3 - "${PREFIX}" <<'PY'
import glob, json, os, sys
prefix=sys.argv[1]
rows=[]
for m in ("tabpfn","tabpfn_clf"):
    for p in sorted(glob.glob("results/metrics/crypto/%s/%s_*.json"%(m,prefix))):
        d=json.load(open(p)); rows.append((os.path.basename(p)[:-5],m,d))
if not rows: print("no results"); raise SystemExit
print("%-24s %-11s %6s %6s %10s %10s %8s %10s"%("run","model","feat","n","CORR","MMC","c.shrp","payout"))
print("-"*92)
for run,m,d in sorted(rows,key=lambda r:-(0.1*r[2]["corr"]["mean"]+r[2]["mmc"]["mean"])):
    print("%-24s %-11s %6d %6d %10.6f %10.6f %8.3f %10.6f"%(run,m,d["config"]["n_features"],
        d["n_periods"],d["corr"]["mean"],d["mmc"]["mean"],d["corr"]["sharpe"],
        0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
print("-"*92)
print("Compare each to its E19 baseline (regressor ctx4096/nest4/temp0.9) and to")
print("the deployed slot on the same dataset, paired NW lag 20.")
PY
} 2>&1 | tee "${REPORT}"
if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" <<'PY' || echo "note: email failed"
import sys; from src.live.notify import send_text
send_text("[crypto] E24 finished -- %s"%sys.argv[1], open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
