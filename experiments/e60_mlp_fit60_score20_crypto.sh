#!/usr/bin/env bash
#
# Experiment E60: tune the best-on-MMC MLP setups FROM SCRATCH on the 60-day
# label, then score them on the payable 20-day one. E60 is the reference handle;
# every artefact is e60_*.
#
# THE IDEA
#   The five MLP setups that scored best on MMC (on the 20-day label) span five
#   feature sets. E60 re-tunes each FROM SCRATCH -- a fresh search, not the pinned
#   20-day config -- but FITS on target_binned_return_60 and SCORES on the payable
#   target_binned_return_20. This is the INVERSE of SACI_03 (which fits 20, scores
#   60): train on the smoother, less noisy 60-day signal and ask whether that
#   signal pays on the 20-day label it is actually graded on -- especially on MMC.
#
# THE ARMS  (5 diverse setups, one per the feature sets that led MMC per dataset)
#   e60_mlp_fenccausal    mlp        crypto/v2.0-fenccausal   (44)   MMC leader overall
#   e60_mlp_onehot        mlp        crypto/v2.0-tenc-onehot  (763)
#   e60_torch_tenc        torch_mlp  crypto/v2.0-tenc         (23)   best tenc MLP was torch
#   e60_mlp_raw           mlp        crypto/v2.0              (22)
#   e60_ktz2_fenccausal   mlp_ktz2   crypto/v2.0-fenccausal   (44)   the E33-reprior WIDER grid
#
# TWO THINGS THE 60-DAY LABEL CHANGES (see E01):
#   * EMBARGO is 60, not 20 -- the target is a 60-day forward return, so train and
#     validation must sit 60 periods apart (as CURUPIRA_05_60 does).
#   * The PANEL SHRINKS -- the 60-day target is NaN over its last ~40 dates, and
#     470 periods at ~95% overlap are roughly EIGHT effective independent windows,
#     so the paired test uses Newey-West lag 60, not 20.
#
# HOW IT IS SCORED
#   Each arm fits+searches on 60d (native 60d metrics land in its json). The
#   deliverable is the 20-day RE-SCORE via src/analysis/cross_target.py
#   (score_on_payable_target): the SAME meta model and reference every deployed
#   slot is graded against. Baseline = each dataset's fit-20 MLP already on disk
#   (paired NW lag 60 on MMC / payout): does fitting on 60d beat fitting on 20d
#   for 20-day MMC?
#
# COST  5 full searches at the CURUPIRA cadence; the 60d panel is smaller but
#   onehot(763) mlp dominates. ~4-6 h serial, ~2 h across 4 shards.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e60_mlp_fit60_score20_crypto.sh   # wiring only
#   SHARD=1 OF=4 bash experiments/e60_mlp_fit60_score20_crypto.sh
#   ONLY="mlp_fenccausal" / DRY_RUN=1 / NO_MAIL=1 / E60_N_ITER=50 / MLP_THREADS=4
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PYTHONPATH="${PYTHONPATH:-.}"

EXPERIMENT="${EXPERIMENT:-E60}"
E60_WINDOW="${E60_WINDOW:-130}"
E60_REFIT="${E60_REFIT:-130}"
E60_EMBARGO="${E60_EMBARGO:-60}"          # 60-day label -> 60, not 20
E60_N_SPLITS="${E60_N_SPLITS:-5}"
E60_N_ITER="${E60_N_ITER:-50}"
E60_VAL_FRAC="${E60_VAL_FRAC:-0.3}"
E60_SEED="${E60_SEED:-42}"
E60_TARGET="${E60_TARGET:-target_binned_return_60}"
L_THREADS="${MLP_THREADS:-4}"
L_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${E60_WINDOW}" in ""|0|[0-9]) echo "E60_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
[ "${E60_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E60_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e60"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e60scout"; L_ERA_STEP=4; E60_N_ITER=3; E60_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${E60_N_ITER} draws -- wiring only, not comparable."
fi

export OMP_NUM_THREADS="${L_THREADS}" MKL_NUM_THREADS="${L_THREADS}" OPENBLAS_NUM_THREADS="${L_THREADS}"

# Parallel arrays: tag, model tree, data-version, expected feature count.
TAG=();   MODEL=(); DV=();                         FEAT=()
add(){ TAG+=("$1"); MODEL+=("$2"); DV+=("$3"); FEAT+=("$4"); }
add mlp_fenccausal   mlp        crypto/v2.0-fenccausal   44
add mlp_onehot       mlp        crypto/v2.0-tenc-onehot  763
add torch_tenc       torch_mlp  crypto/v2.0-tenc         23
add mlp_raw          mlp        crypto/v2.0              22
add ktz2_fenccausal  mlp_ktz2   crypto/v2.0-fenccausal   44

# Selection (ONLY) + shard round-robin.
SEL=(); picked=0
for i in "${!TAG[@]}"; do
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${TAG[$i]} "* ]] && continue
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL+=("$i")
done
[ "${#SEL[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight: datasets present at the expected feature count WITH the 60-day
# target, torch importable (torch_mlp arm), and every model constructs.
python3 - "${DV[@]}" -- "${FEAT[@]}" <<'PY' || exit 1
import sys, importlib.util
import pyarrow.parquet as pq
from src.prediction.models import build_wrapper
cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    try:
        names=pq.ParquetFile("data/%s/train.parquet"%v).schema.names
        got=sum(c.startswith("feature_") for c in names)
        if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
        if "target_binned_return_60" not in names: bad.append("%s lacks target_binned_return_60"%v)
    except OSError: bad.append("%s MISSING"%v)
if importlib.util.find_spec("torch") is None: bad.append("torch not installed (torch_mlp arm)")
for m in ("mlp","torch_mlp","mlp_ktz2"):
    try: build_wrapper(m,"regression",seed=42,n_jobs=1)
    except Exception as e: bad.append("model %s failed: %s"%(m,e))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: datasets present with 60d target and expected feature counts; all models construct")
PY

total="${#SEL[@]}"; index=0; ran=0; failed=0; declare -a FAILED=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)
  arms   ${total} of 5 (shard ${SHARD}/${OF}); FIT on ${E60_TARGET} (embargo ${E60_EMBARGO}), SCORE on 20d
  models mlp / torch_mlp / mlp_ktz2, --n-iter ${E60_N_ITER} full search
  handle ${EXPERIMENT}; every artefact is ${PREFIX}_*
  report python3 -m src.analysis.results_table --model mlp --runs ${PREFIX}_mlp_fenccausal ...
         then the 20d re-score (cross_target) -- see the runner tail"
    python3 -c "import sys; from src.live.notify import send_text; send_text('[crypto] %s started'%sys.argv[1], sys.argv[2])" \
        "${EXPERIMENT}" "${MSG}" 2>/dev/null || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: MLP fit-60 / score-20 (SEARCHED)  |  ${total} arms  |  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  target=${E60_TARGET} embargo=${E60_EMBARGO}"
echo "=============================================================="

for i in "${SEL[@]}"; do
    name="${PREFIX}_${TAG[$i]}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto "${MODEL[$i]}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E60_WINDOW}" --refit-every "${E60_REFIT}"
         --embargo "${E60_EMBARGO}" --n-splits "${E60_N_SPLITS}" --n-iter "${E60_N_ITER}"
         --val-frac "${E60_VAL_FRAC}" --era-step "${L_ERA_STEP}" --seed "${E60_SEED}"
         --search-jobs 1 --target "${E60_TARGET}"
         --data-version "${DV[$i]}" --run-name "${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-22s %-10s %s\n' "$index" "$total" "$name" "${MODEL[$i]}" "${DV[$i]}"; continue; fi
    ran=$((ran+1))
    echo; echo "[${index}/${total}] ${name}  ${MODEL[$i]}  ${DV[$i]}  $(date '+%H:%M:%S')"
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

# 20-day RE-SCORE of every finished e60 arm, beside its native 60d numbers.
echo; echo "=== 20-day re-score (cross_target) -- the deliverable ==="
python3 - "${PREFIX}" "${TAG[@]}" -- "${MODEL[@]}" <<'PY' 2>&1 | tee "logs/${PREFIX}_rescore.txt"
import sys, os, json
import numpy as np, pandas as pd
from src.analysis.cross_target import load_scoring_frame, score_on_payable_target
cut=sys.argv.index("--"); prefix=sys.argv[1]; tags=sys.argv[2:cut]; models=sys.argv[cut+1:]
extras=load_scoring_frame()
print("%-22s %-10s | %8s %8s (60d native) | %8s %8s %8s (20d re-score) %6s"%(
    "arm","model","corr60","mmc60","corr20","mmc20","pay20","n20"))
print("-"*104)
for tag,model in zip(tags,models):
    run="%s_%s"%(prefix,tag)
    mpath="results/metrics/crypto/%s/%s.json"%(model,run)
    fpath="results/forecast/crypto/%s/%s.csv"%(model,run)
    if not os.path.exists(fpath): print("%-22s %-10s | not finished"%(run,model)); continue
    m=json.load(open(mpath)) if os.path.exists(mpath) else {}
    c60=(m.get("corr") or {}).get("mean",float("nan")); mmc60=(m.get("mmc") or {}).get("mean",float("nan"))
    fc=pd.read_csv(fpath,usecols=["symbol","date","prediction"],parse_dates=["date"]).set_index(["symbol","date"])
    pred=fc.groupby(level=1)["prediction"].rank(pct=True)
    _,s=score_on_payable_target(pred, extras)
    c20=s.get("corr",{}).get("mean",float("nan")) if isinstance(s.get("corr"),dict) else float("nan")
    mmc20=s.get("mmc",{}).get("mean",float("nan")) if isinstance(s.get("mmc"),dict) else float("nan")
    pay20=0.1*c20+mmc20
    print("%-22s %-10s | %8.5f %8.5f | %8.5f %8.5f %8.5f %6d"%(
        run,model,c60,mmc60,c20,mmc20,pay20,s.get("n_periods",0)))
print("-"*104)
print("Baseline (fit-20/score-20) MMC per dataset, already on disk:")
print("  fenccausal 0.0825 (e15_mlp_fenccausal) | onehot 0.0781 (e33_oh_a128_relu_lre1_al10)")
print("  tenc 0.0775 (e15_torch_mlp_tenc) | raw 0.0746 (e07_mlpreg_raw) | ktz2 best 0.0647 (e34_cumlag3)")
print("Headline: does fit-60 lift 20d MMC over fit-20? Paired NW lag 60 -> report step.")
PY

REPORT="logs/${PREFIX}_report.txt"; [ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{ echo "${EXPERIMENT} -- MLP fit-60/score-20"; el_all=$(( $(date +%s)-started_all ))
  echo "ran ${ran}/${total} in $((el_all/3600))h$(((el_all%3600)/60))m, ${failed} failed"
  [ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]}"
  [ "${OF}" -gt 0 ] && echo "SHARD ${SHARD}/${OF} PARTIAL -- run the re-score once all shards finish."
  echo; cat "logs/${PREFIX}_rescore.txt" 2>/dev/null; } | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' 2>/dev/null || echo "note: email failed"
import sys; from src.live.notify import send_text
send_text("[crypto] %s finished -- %s"%(sys.argv[3],sys.argv[1]), open(sys.argv[2]).read())
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
