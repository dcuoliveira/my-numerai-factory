#!/usr/bin/env bash
#
# Experiment E187: E183's tabnet_ktz_tv (TV epochs tied to model size) on every
# dataset behind a deployed slot.
#
# Same model, space, walk-forward and per-candidate capture as E183 (n_iter 40,
# n_splits 3, --save-cv-results, --reseed-per-refit), seed 42, one arm per
# deployed feature set. raw and fenccausal are NOT re-run: E183's
# e183_{raw,fenccausal}_s42 are the same recipe and are reused in the report.
#
#   arm               data version                                         cols  slot(s) on it
#   tenc              crypto/v2.0-tenc                                       23  CURUPIRA_01, IA_03
#   interact          crypto/v2.0-interact                                   36  IA_04, SACI_01
#   cumlag3           crypto/v2.0-cumlag3                                    88  SACI_05
#   onehot_roll       crypto/v2.0-onehot-rollsel                            786  CURUPIRA_03
#   tenc_onehot       crypto/v2.0-tenc-onehot                               763  SACI_03, SACI_06
#   allfeat           crypto/v2.0-tenc-onehot-spread-interact               788  SACI_02, IAUARETE_02, MA_ALLFEAT
#   allfeat_pp        crypto/v2.0-tenc-onehot-rollsel-spread-interact       812  CURUPIRA_04/_05/_06/_07/_08, CURUPIRA_SACI_01-03
#   allfeat_pp_fenc   crypto/v2.0-tenc-onehot-rollsel-spread-interact-fenccausal  834  CU_SA_IA_BO_* and the other 834-col blends
#
# Not run: allfeat_pp_lag3 (878, CURUPIRA_SACI_04) -- no prebuilt research
# dataset exists; it is assembled only in src/live/features.py.
# TV epochs on the wide sets are 1,059-2,944 (vs 136-906 on raw), so those arms
# are many times slower than E183's.
#
# Usage: E187_ARM=tenc E187_SEARCH_JOBS=4 bash experiments/e187_tabnet_tv_deployed_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PATH="/home/danielco/miniconda3/bin:${PATH}"

EXPERIMENT="E187"
ARM="${E187_ARM:?set E187_ARM}"
N_ITER="${E187_N_ITER:-40}"
N_SPLITS="${E187_N_SPLITS:-3}"
SEARCH_JOBS="${E187_SEARCH_JOBS:-4}"
THREADS="${E187_THREADS:-2}"
SEED="${E187_SEED:-42}"
ERA_STEP=1
PREFIX="e187"

case "${ARM}" in
    tenc)            DATA="crypto/v2.0-tenc";            NFEAT=23 ;;
    interact)        DATA="crypto/v2.0-interact";        NFEAT=36 ;;
    cumlag3)         DATA="crypto/v2.0-cumlag3";         NFEAT=88 ;;
    onehot_roll)     DATA="crypto/v2.0-onehot-rollsel";  NFEAT=786 ;;
    tenc_onehot)     DATA="crypto/v2.0-tenc-onehot";     NFEAT=763 ;;
    allfeat)         DATA="crypto/v2.0-tenc-onehot-spread-interact"; NFEAT=788 ;;
    allfeat_pp)      DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact"; NFEAT=812 ;;
    allfeat_pp_fenc) DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact-fenccausal"; NFEAT=834 ;;
    *) echo "unknown arm '${ARM}'" >&2; exit 1 ;;
esac

if [ "${E187_SCOUT:-0}" = "1" ]; then
    PREFIX="e187scout"; ERA_STEP=8; N_ITER=2; N_SPLITS=2
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
Compare paired (Newey-West, per-period payout) against e183_raw_s42 and the deployed slots
on the same dataset via the arm-board-reporting skill before reading anything into it."
exit "${status}"
