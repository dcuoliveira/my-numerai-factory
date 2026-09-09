#!/usr/bin/env bash
#
# Experiment E43: torch MLP training-LOSS study, config SEARCHED (best payout).
#
# E43 is the reference handle; every artefact is e43_*. Companion to E42 (pinned):
# same 6 loss arms x raw/fenccausal, but each loss runs a full search over the
# torch_mlp grid and finds its OWN config -- the "best achievable per loss" read,
# where E42 holds the config fixed to isolate the loss's effect on learning.
#
# THE ARMS (6 losses x 2 datasets = 12), model class torch_mlp_loss:
#   mse corr bmcbench(sparse real meta) bmcproxy(full proxy) sharpe calmar
#   -- identical set and references to E42; see that runner's header for the map.
#
# COST  a full --n-iter search per arm x ~4 refits; ~E12's torch footprint,
#   ~2-3 h across 4 shards (sharpe/calmar are full-batch, fewer steps/epoch).
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e43_torch_mlp_loss_search_crypto.sh
#   SHARD=1 OF=4 bash experiments/e43_torch_mlp_loss_search_crypto.sh
#   ONLY="corr" / ONLY_DS=raw / L_N_ITER=40 / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=4
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E43}"
L_WINDOW="${L_WINDOW:-130}"
L_REFIT="${L_REFIT:-130}"
L_EMBARGO="${L_EMBARGO:-20}"
L_N_SPLITS="${L_N_SPLITS:-5}"
L_N_ITER="${L_N_ITER:-40}"
L_VAL_FRAC="${L_VAL_FRAC:-0.3}"
L_SEED="${L_SEED:-42}"
L_THREADS="${MLP_THREADS:-4}"
L_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${L_WINDOW}" in ""|0|[0-9]) echo "L_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
[ "${L_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "L_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e43"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e43scout"; L_ERA_STEP=4; L_N_ITER=3; L_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${L_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${L_THREADS}"
export MKL_NUM_THREADS="${L_THREADS}"
export OPENBLAS_NUM_THREADS="${L_THREADS}"

LTAG=(); LOSS=(); LREF=()
ladd() { LTAG+=("$1"); LOSS+=("$2"); LREF+=("$3"); }
ladd mse      mse    meta_proxy
ladd corr     corr   meta_proxy
ladd bmcbench bmc    benchmark
ladd bmcproxy bmc    meta_proxy
ladd sharpe   sharpe meta_proxy
ladd calmar   calmar meta_proxy

DSTAG=(raw fenccausal); DSVER=("crypto/v2.0" "crypto/v2.0-fenccausal"); DSFEAT=(22 44)

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

SEL=(); picked=0
for i in "${!NAME[@]}"; do
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL+=("$i")
done
[ "${#SEL[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

python3 - "${DSVER[@]}" -- "${DSFEAT[@]}" <<'PY' || exit 1
import sys, importlib.util
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper
if importlib.util.find_spec("torch") is None:
    sys.exit("preflight FAILED: torch is not installed")
cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    try: got=sum(c.startswith("feature_") for c in pq.ParquetFile("data/%s/train.parquet"%v).schema.names)
    except OSError: bad.append("%s MISSING"%v); continue
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
for loss in ("mse","corr","bmc","sharpe","calmar"):
    try: build_wrapper("torch_mlp_loss","regression",loss=loss)
    except Exception as e: bad.append("loss=%s failed: %s"%(loss,e))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: torch present; %d datasets; all losses construct"%len(versions))
PY

total="${#SEL[@]}"; index=0; ran=0; failed=0; declare -a FAILED=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)
  arms   ${total} (of 12; shard ${SHARD}/${OF}); 6 losses x raw/fenccausal, SEARCHED
  model  torch_mlp_loss, --n-iter ${L_N_ITER} per loss
  handle ${EXPERIMENT}; every artefact is ${PREFIX}_*"
    python3 -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started'%sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${MSG}" 2>/dev/null || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: torch_mlp loss study (SEARCHED, ${L_N_ITER} draws)  |  ${total} arms  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${SEL[@]}"; do
    name="${NAME[$i]}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto torch_mlp_loss
         --task regression --metric corr --bmc-reference "${ARM_REF[$i]}"
         --cv-scheme rolling --train-window "${L_WINDOW}" --refit-every "${L_REFIT}"
         --embargo "${L_EMBARGO}" --n-splits "${L_N_SPLITS}" --n-iter "${L_N_ITER}"
         --val-frac "${L_VAL_FRAC}" --era-step "${L_ERA_STEP}" --seed "${L_SEED}"
         --search-jobs 1 --model-options "{\"loss\": \"${ARM_LOSS[$i]}\"}"
         --data-version "${ARM_VER[$i]}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-26s loss=%-7s ref=%-10s %s\n' "$index" "$total" "$name" \
               "${ARM_LOSS[$i]}" "${ARM_REF[$i]}" "${ARM_VER[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  loss=${ARM_LOSS[$i]}  $(date '+%H:%M:%S')"
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
echo "${EXPERIMENT} -- torch_mlp loss study (SEARCHED); best achievable per loss"
echo "  results  results/metrics/crypto/torch_mlp_loss/${PREFIX}_*.json"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran}/${total} arms in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; run the reports once all shards finish."
echo
echo "Board + paired NW vs the mse control (per dataset), and learning dynamics:"
echo "  python3 -m src.analysis.report_html   --model torch_mlp_loss --runs ${PREFIX}_mse_raw ${PREFIX}_corr_raw ..."
echo "  python3 -m src.analysis.loss_dynamics --model torch_mlp_loss --runs ${PREFIX}_mse_raw ${PREFIX}_corr_raw ..."
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' 2>/dev/null || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
