#!/usr/bin/env bash
#
# Experiment E183: TabNet with epochs TIED to model size (Timmermann-Vulicevic), raw + fenccausal
#
# Follow-up to E153/E158, which searched max_epochs in {30,60,100,150}, far below
# the TV schedule for every architecture in the grid (smallest TabNet: 136 epochs).
# Model `tabnet_ktz_tv` (src/prediction/tabnet_tv.py) has no max_epochs axis: each
# candidate trains for E = floor(0.2 * P**0.8) + 10 epochs, where P is the trainable
# parameter count of the network ITS sampled settings build on THIS dataset's column
# count. That guarantees the pairing of epochs and model size by construction.
# The space is capped where TV is affordable inside a search: width 8-16, steps 2-5,
# GLU depth 1-2 (TV 136-837 epochs on raw, 174-906 on fenccausal). Every other
# axis keeps tabnet_ktz's range.
#
# Same walk-forward and the same per-candidate capture as E158 (--save-cv-results,
# --reseed-per-refit), so src.analysis.hparam_importance reads it directly, and
# E183 vs E158 compares TV-length training against short training.
#
# Arms: e183_{raw,fenccausal}_s{42,43}; n_iter 40, n_splits 3; 2 torch threads per
# search worker (measured 1.1-2.3 s/epoch on 39k rows).
#
# Launched 2026-11-17 with a one-off `src.cluster --extra-cores 10` allowance.
#
# Usage:
#   E183_ARM=raw E183_SEED=42 E183_SEARCH_JOBS=6 bash experiments/e183_tabnet_tv_crypto.sh
#   E183_SCOUT=1 E183_ARM=raw bash ...     # wiring: era-step 8, n_iter 2
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PATH="/home/danielco/miniconda3/bin:${PATH}"

EXPERIMENT="E183"
ARM="${E183_ARM:?set E183_ARM=raw or fenccausal}"
N_ITER="${E183_N_ITER:-40}"
N_SPLITS="${E183_N_SPLITS:-3}"
SEARCH_JOBS="${E183_SEARCH_JOBS:-4}"
THREADS="${E183_THREADS:-2}"
SEED="${E183_SEED:-42}"
ERA_STEP=1
PREFIX="e183"

case "${ARM}" in
    raw)        DATA="crypto/v2.0";            NFEAT=22 ;;
    fenccausal) DATA="crypto/v2.0-fenccausal"; NFEAT=44 ;;
    *) echo "unknown arm '${ARM}'" >&2; exit 1 ;;
esac

if [ "${E183_SCOUT:-0}" = "1" ]; then
    PREFIX="e183scout"; ERA_STEP=8; N_ITER=2; N_SPLITS=2
    echo "SCOUT MODE: wiring only, results not comparable to anything."
fi
RUN="${PREFIX}_${ARM}_s${SEED}"

# Each search worker is a process. Bound torch's threads PER WORKER so that
# search-jobs x threads fills the slot and does not exceed it.
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
       OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"

python3 - "${DATA}" "${NFEAT}" <<'PY' || exit 1
import sys, os
import pyarrow.parquet as pq
version, want = sys.argv[1], int(sys.argv[2])
path = "data/%s/train.parquet" % version
if not os.path.exists(path):
    sys.exit("preflight FAILED: %s missing" % path)
got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
if got != want:
    sys.exit("preflight FAILED: %s has %d features, expected %d" % (version, got, want))
import torch, pytorch_tabnet  # noqa: F401
from src.prediction.models import MODELS
assert "tabnet_ktz_tv" in MODELS, "tabnet_ktz_tv not registered"
print("preflight OK: %s %d features, torch + pytorch-tabnet present" % (version, got))
PY

cmd=(python3 -u -m src.run_forecast crypto tabnet_ktz_tv
     --task regression --metric corr --bmc-reference benchmark
     --cv-scheme rolling --train-window 130 --refit-every 130 --embargo 20
     --n-splits "${N_SPLITS}" --n-iter "${N_ITER}" --era-step "${ERA_STEP}"
     --seed "${SEED}" --search-jobs "${SEARCH_JOBS}" --n-jobs "${THREADS}"
     --data-version "${DATA}" --run-name "${RUN}"
     --save-cv-results --reseed-per-refit)

if [ "${DRY_RUN:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi

mail() {  # subject body
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2])" \
        "$1" "$2" || echo "note: email failed, run continues"
}

mail "[crypto] ${EXPERIMENT} ${RUN} started" "${EXPERIMENT} ${RUN} started $(date '+%F %T') on $(hostname)
  model    tabnet_ktz_tv (TV epochs tied to size), random search n_iter ${N_ITER}, ${N_SPLITS} folds
  data     ${DATA} (${NFEAT} features)
  slot     search-jobs ${SEARCH_JOBS} x ${THREADS} threads, CPU
  log      logs/${RUN}.log
  results  results/metrics/crypto/tabnet_ktz_tv/${RUN}.json"

started=$(date +%s)
"${cmd[@]}" 2>&1 | tee "logs/${RUN}.log"
status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))

summary=$(python3 - "${RUN}" <<'PY'
import json, os, sys
p = "results/metrics/crypto/tabnet_ktz_tv/%s.json" % sys.argv[1]
if not os.path.exists(p):
    print("no metrics json written"); raise SystemExit
d = json.load(open(p))
print("n %d  CORR %.6f  MMC %.6f  c.sharpe %.4f  payout(ranking aid) %.6f" % (
    d["n_periods"], d["corr"]["mean"], d["mmc"]["mean"], d["corr"]["sharpe"],
    0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
PY
)
verdict=$([ "${status}" -eq 0 ] && echo SUCCESS || echo "FAILED (exit ${status})")
echo "${RUN}: ${verdict} in $((el/3600))h$(((el%3600)/60))m -- ${summary}"
mail "[crypto] ${EXPERIMENT} ${RUN} ${verdict}" "${RUN} ${verdict} after $((el/3600))h$(((el%3600)/60))m on $(hostname)
${summary}
Per-candidate CV scores: results/tuning/crypto/tabnet_ktz_tv/${RUN}_cv.csv
Compare paired (Newey-West, per-period payout) against e18_${ARM}, e158_${ARM}_s${SEED} and the deployed slot
on the same dataset via the arm-board-reporting skill before reading anything into it."
exit "${status}"
