#!/usr/bin/env bash
#
# Experiment E42: torch MLP training-LOSS study, config PINNED (learning dynamics).
#
# E42 is the reference handle; every artefact is e42_*.
#
# WHY
#   E09/E15 showed sklearn and torch MLP are equivalent on MSE. This asks whether
#   a different training OBJECTIVE -- optimising the thing we are scored on (corr,
#   MMC/BMC) or a risk-adjusted ratio (sharpe, calmar) -- changes what the net
#   learns and how stably. Because the read is about the LEARNING PROCESS, every
#   arm shares ONE pinned config (--fixed-params) so the only thing varying is the
#   loss; E43 is the companion that searches per loss for the best-payout read.
#
# THE ARMS (6 losses x 2 datasets = 12), model class torch_mlp_loss:
#   mse       control -- byte-identical to the plain torch_mlp half-MSE
#   corr      per-period CorrLoss (Pearson surrogate of numerai_corr)
#   bmcbench  BMCLoss vs the REAL meta model (sparse ~33% coverage = "MMC where
#             it exists"): --bmc-reference benchmark
#   bmcproxy  BMCLoss vs the full-coverage Ridge meta_proxy ("estimate MMC via
#             BMC"): --bmc-reference meta_proxy
#   sharpe    full-window -(mean/std) of the per-period corr series
#   calmar    full-window -(mean / soft-max-drawdown) of the per-period corr series
#
#   bmcbench vs bmcproxy IS the "optimise on MMC-only vs estimate with BMC"
#   contrast: on crypto the benchmark and meta model are the same sparse series,
#   so the only full-coverage reference is the proxy.
#
# EVALUATION
#   (a) the board (report_html: corr, c.sharpe, dd, payout, calmar, mmc, +bmc)
#       with paired Newey-West vs the mse control; and
#   (b) learning dynamics -- per-epoch loss curve, epochs-to-converge (n_iter),
#       final-vs-best gap, pred_std (collapse), diverged flag -- persisted in the
#       metrics json (src/analysis/loss_dynamics.py reads them).
#
# COST  a pinned torch MLP fit x ~4 refits; sharpe/calmar are full-batch (one step
#   per epoch) so cheaper per epoch but more epochs. ~minutes/arm; ~1 h across 4
#   shards.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e42_torch_mlp_loss_dynamics_crypto.sh
#   SHARD=1 OF=4 bash experiments/e42_torch_mlp_loss_dynamics_crypto.sh
#   ONLY="corr sharpe" / ONLY_DS=raw / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E42}"
L_WINDOW="${L_WINDOW:-130}"
L_REFIT="${L_REFIT:-130}"
L_EMBARGO="${L_EMBARGO:-20}"
L_N_SPLITS="${L_N_SPLITS:-2}"
L_SEED="${L_SEED:-42}"
L_THREADS="${MLP_THREADS:-4}"
L_ERA_STEP=1
# One config, held fixed across every loss arm -- the whole point of E42.
PIN='{"mlp__hidden_layer_sizes":[64,32],"mlp__alpha":0.001,"mlp__learning_rate_init":0.001,"mlp__batch_size":1024}'
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${L_WINDOW}" in ""|0|[0-9]) echo "L_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
[ "${L_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "L_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e42"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e42scout"; L_ERA_STEP=4
    echo "SCOUT MODE: era-step 4 -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${L_THREADS}"
export MKL_NUM_THREADS="${L_THREADS}"
export OPENBLAS_NUM_THREADS="${L_THREADS}"

# Loss arms: (tag, loss, bmc-reference). Parallel arrays, never pipe-packed.
LTAG=(); LOSS=(); LREF=()
ladd() { LTAG+=("$1"); LOSS+=("$2"); LREF+=("$3"); }
ladd mse      mse    meta_proxy
ladd corr     corr   meta_proxy
ladd bmcbench bmc    benchmark
ladd bmcproxy bmc    meta_proxy
ladd sharpe   sharpe meta_proxy
ladd calmar   calmar meta_proxy

# Datasets.
DSTAG=(raw fenccausal); DSVER=("crypto/v2.0" "crypto/v2.0-fenccausal"); DSFEAT=(22 44)

# Build the (arm x dataset) work list.
NAME=(); ARM_LOSS=(); ARM_REF=(); ARM_VER=(); ARM_FEAT=()
for li in "${!LTAG[@]}"; do
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${LTAG[$li]} "* ]] && continue
    for di in "${!DSTAG[@]}"; do
        [ -n "${ONLY_DS:-}" ] && [ "${ONLY_DS}" != "${DSTAG[$di]}" ] && continue
        NAME+=("${PREFIX}_${LTAG[$li]}_${DSTAG[$di]}")
        ARM_LOSS+=("${LOSS[$li]}"); ARM_REF+=("${LREF[$li]}")
        ARM_VER+=("${DSVER[$di]}"); ARM_FEAT+=("${DSFEAT[$di]}")
    done
done

# Shard round-robin.
SEL=(); picked=0
for i in "${!NAME[@]}"; do
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL+=("$i")
done
[ "${#SEL[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight: torch present, datasets present at expected feature counts, every
# loss constructs and declares its passthrough columns.
python3 - "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import sys, importlib.util
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper
from src.prediction.torch_mlp_losses import LOSSES
if importlib.util.find_spec("torch") is None:
    sys.exit("preflight FAILED: torch is not installed (torch_mlp_loss needs it)")
cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    try: got=sum(c.startswith("feature_") for c in pq.ParquetFile("data/%s/train.parquet"%v).schema.names)
    except OSError: bad.append("%s MISSING"%v); continue
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
for loss in ("mse","corr","bmc","sharpe","calmar"):
    try: build_wrapper("torch_mlp_loss","regression",loss=loss)
    except Exception as e: bad.append("loss=%s failed to construct: %s"%(loss,e))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: torch present; %d datasets; all losses (%s) construct"%(len(versions),",".join(LOSSES)))
PY

total="${#SEL[@]}"; index=0; ran=0; failed=0; declare -a FAILED=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)
  arms      ${total} (of 12; shard ${SHARD}/${OF})  --  6 losses x raw/fenccausal
  model     torch_mlp_loss, config PINNED (only the loss varies)
  losses    mse(control) corr bmc(bench=sparse-MMC / proxy=full) sharpe calmar
  report    board (report_html) + learning dynamics (loss_dynamics.py)
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started'%sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${MSG}" 2>/dev/null || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: torch_mlp loss study (PINNED)  |  ${total} arms  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${SEL[@]}"; do
    name="${NAME[$i]}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto torch_mlp_loss
         --task regression --metric corr --bmc-reference "${ARM_REF[$i]}"
         --cv-scheme rolling --train-window "${L_WINDOW}" --refit-every "${L_REFIT}"
         --embargo "${L_EMBARGO}" --n-splits "${L_N_SPLITS}" --era-step "${L_ERA_STEP}"
         --seed "${L_SEED}" --search-jobs 1
         --fixed-params "${PIN}" --model-options "{\"loss\": \"${ARM_LOSS[$i]}\"}"
         --data-version "${ARM_VER[$i]}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-26s loss=%-7s ref=%-10s %s\n' "$index" "$total" "$name" \
               "${ARM_LOSS[$i]}" "${ARM_REF[$i]}" "${ARM_VER[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  loss=${ARM_LOSS[$i]} ref=${ARM_REF[$i]}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED+=("${name}")
        echo ">>> ${name} FAILED (exit ${status}) after $((el/60))m -- continuing (logs/${name}.log)"
    else
        echo ">>> ${name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- torch_mlp loss study (PINNED); board + learning dynamics"
echo "  results  results/metrics/crypto/torch_mlp_loss/${PREFIX}_*.json"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran}/${total} arms in $((el_all/60))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; run the reports once all shards finish."
echo
if [ "${MLP_SCOUT:-0}" != "1" ] && [ "${OF}" -le 1 ]; then
    for ds in raw fenccausal; do
        runs=$(ls results/metrics/crypto/torch_mlp_loss/${PREFIX}_*_${ds}.json 2>/dev/null | sed 's#.*/##;s#.json##' | tr '\n' ' ')
        [ -z "$runs" ] && continue
        echo "=== learning dynamics: ${ds} ==="
        python3 -m src.analysis.loss_dynamics --model torch_mlp_loss --runs $runs 2>&1 || echo "(loss_dynamics.py pending)"
    done
    echo "Board + paired-vs-mse: python3 -m src.analysis.report_html --model torch_mlp_loss --runs <e42_*>"
else
    echo "Once every shard's arms are on disk, read:"
    echo "  python3 -m src.analysis.loss_dynamics --model torch_mlp_loss --runs ${PREFIX}_mse_raw ${PREFIX}_corr_raw ..."
    echo "  python3 -m src.analysis.report_html --model torch_mlp_loss --runs ${PREFIX}_mse_raw ..."
fi
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' 2>/dev/null || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
