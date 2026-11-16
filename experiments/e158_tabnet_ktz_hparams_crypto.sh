#!/usr/bin/env bash
#
# Experiment E158: tabnet_ktz hyperparameter IMPORTANCE map (raw + fenccausal, 2 seeds)
#
# Same model, grid and walk-forward as E153. The difference is what it keeps:
# E153 kept only each refit's winning configuration, which says nothing about
# the rest of the space. E158 keeps every candidate:
#   --save-cv-results   every candidate's per-fold inner-CV CORR and MMC (real meta
#                       model), written to results/tuning/crypto/tabnet_ktz/<run>_cv.csv
#   --reseed-per-refit  a fresh candidate draw at every refit (seed + refit), so
#                       K refits x n_iter distinct points rather than the same n_iter
#                       re-scored K times (E153's fenccausal refits 1-2 picked the
#                       identical config for exactly that reason)
# Two seeds per dataset double the sample and measure how reproducible the
# picked winner is. src.analysis.hparam_importance turns the _cv.csv files into
# main-effect importance (eta^2 vs a permutation null), stability across refits,
# interactions, sensitivity curves and a suggested walk-forward grid. The goal is
# to specify a narrower search space for later walk-forward studies.
#
# Arms: e158_{raw,fenccausal}_s{42,43}; n_iter 120 per refit, n_splits 3 (2 purged folds).
#
# Usage:
#   E158_ARM=raw E158_SEED=42 bash experiments/e158_tabnet_ktz_hparams_crypto.sh
#   E158_SCOUT=1 E158_ARM=raw bash ...     # wiring: era-step 8, n_iter 3
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PATH="/home/danielco/miniconda3/bin:${PATH}"

EXPERIMENT="E158"
ARM="${E158_ARM:?set E158_ARM=raw or fenccausal}"
N_ITER="${E158_N_ITER:-120}"
N_SPLITS="${E158_N_SPLITS:-3}"
SEARCH_JOBS="${E158_SEARCH_JOBS:-4}"
THREADS="${E158_THREADS:-4}"
SEED="${E158_SEED:-42}"
ERA_STEP=1
PREFIX="e158"

case "${ARM}" in
    raw)        DATA="crypto/v2.0";            NFEAT=22 ;;
    fenccausal) DATA="crypto/v2.0-fenccausal"; NFEAT=44 ;;
    *) echo "unknown arm '${ARM}'" >&2; exit 1 ;;
esac

if [ "${E158_SCOUT:-0}" = "1" ]; then
    PREFIX="e158scout"; ERA_STEP=8; N_ITER=3; N_SPLITS=2
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
assert "tabnet_ktz" in MODELS, "tabnet_ktz not registered"
print("preflight OK: %s %d features, torch + pytorch-tabnet present" % (version, got))
PY

cmd=(python3 -u -m src.run_forecast crypto tabnet_ktz
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
  model    tabnet_ktz (wide KTZ grid), random search n_iter ${N_ITER}, ${N_SPLITS} folds
  data     ${DATA} (${NFEAT} features)
  slot     search-jobs ${SEARCH_JOBS} x ${THREADS} threads, CPU
  log      logs/${RUN}.log
  results  results/metrics/crypto/tabnet_ktz/${RUN}.json"

started=$(date +%s)
"${cmd[@]}" 2>&1 | tee "logs/${RUN}.log"
status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))

summary=$(python3 - "${RUN}" <<'PY'
import json, os, sys
p = "results/metrics/crypto/tabnet_ktz/%s.json" % sys.argv[1]
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
Per-candidate CV scores: results/tuning/crypto/tabnet_ktz/${RUN}_cv.csv
Compare paired (Newey-West, per-period payout) against e18_${ARM} and the deployed slot
on the same dataset via the arm-board-reporting skill before reading anything into it."
exit "${status}"
