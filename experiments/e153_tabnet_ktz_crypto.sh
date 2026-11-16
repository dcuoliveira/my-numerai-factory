#!/usr/bin/env bash
#
# Experiment E153: TabNet with a wide KTZ-style search (tabnet_ktz), raw + fenccausal
#
# Every artefact is e153_* in results/{metrics,forecast,per_period,hyperparams}/crypto/tabnet_ktz/.
#
# WHY
#   E18 pinned TabNet at library defaults, and it lost to the deployed slot on every
#   dataset (t -2.0..-5.3). Its write-up said tuning was the only lever left.
#   E153 pulls that lever as wide as is affordable: `tabnet_ktz` opens every
#   axis pytorch-tabnet exposes (width 8-64 with n_a tied to n_d, n_steps 2-7, gamma,
#   GLU depth, lambda_sparse 1e-6..1e-1, BN momentum, sparsemax/entmax, lr 3e-4..5e-2,
#   batch/ghost batch, epochs 30-150). See TabNetKtzWrapper in src/prediction/models.py.
#
# ARMS (same datasets E18 lost on, so the pairing is direct)
#     e153_raw          crypto/v2.0               22   vs CURUPIRA_02 / SACI_04 / e18_raw
#     e153_fenccausal   crypto/v2.0-fenccausal    44   vs IA_01 (MLP)  / e18_fenccausal
#
# SEARCH
#   Random search, --n-iter 60, --n-splits 3 (2 usable purged folds), seed 42,
#   rolling 130/130, embargo 20. That is 4 refits x (60 x 2 + 1) ~ 484 fits per arm.
#   No eval_set anywhere, so the fit is leak-free; epochs is a searched axis instead.
#
# COST: CPU on purpose. At 22-44 columns TabNet is small, and E18 measured
#   ~3 s/epoch at 8 threads. Candidates run 4 at a time (--search-jobs 4) with
#   4 torch threads each = a 16-cpu slot. Expect roughly 8-14 h per arm.
#
# Usage (one arm per invocation, so the cluster manager can place each separately):
#   E153_ARM=raw        bash experiments/e153_tabnet_ktz_crypto.sh
#   E153_ARM=fenccausal bash experiments/e153_tabnet_ktz_crypto.sh
#   E153_SCOUT=1 E153_ARM=raw bash ...   # wiring: era-step 8, n_iter 2, prefix e153scout
#   NO_MAIL=1 / DRY_RUN=1
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PATH="/home/danielco/miniconda3/bin:${PATH}"

EXPERIMENT="E153"
ARM="${E153_ARM:?set E153_ARM=raw or fenccausal}"
N_ITER="${E153_N_ITER:-60}"
N_SPLITS="${E153_N_SPLITS:-3}"
SEARCH_JOBS="${E153_SEARCH_JOBS:-4}"
THREADS="${E153_THREADS:-4}"
SEED="${E153_SEED:-42}"
ERA_STEP=1
PREFIX="e153"

case "${ARM}" in
    raw)        DATA="crypto/v2.0";            NFEAT=22 ;;
    fenccausal) DATA="crypto/v2.0-fenccausal"; NFEAT=44 ;;
    *) echo "unknown arm '${ARM}'" >&2; exit 1 ;;
esac

if [ "${E153_SCOUT:-0}" = "1" ]; then
    PREFIX="e153scout"; ERA_STEP=8; N_ITER=2; N_SPLITS=2
    echo "SCOUT MODE: wiring only, results not comparable to anything."
fi
RUN="${PREFIX}_${ARM}"

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
     --data-version "${DATA}" --run-name "${RUN}")

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
Compare paired (Newey-West, per-period payout) against e18_${ARM} and the deployed slot
on the same dataset via the arm-board-reporting skill before reading anything into it."
exit "${status}"
