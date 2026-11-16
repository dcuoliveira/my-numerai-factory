#!/usr/bin/env bash
#
# Experiment E74: does a proper realised-volatility block beat the raw 22?
# crypto/v2.0-vol VS crypto/v2.0 (the raw 22 as the base).
#
# E74 is the reference handle; every artefact is e74_*.
#
# THE QUESTION
#   The base 22 already carries two crude volatility features
#   (feature_volatility_{20,60}d). E74 asks whether a proper volatility BLOCK -- 22
#   realised-volatility estimators built from spot OHLCV over a hyperparameter grid
#   -- adds over them. The block (feature_vol_*) is four estimators x their lookback:
#     squared-returns realised variance   feature_vol_sqret_{5,10,20,40,60,120}d
#     EWMA / RiskMetrics variance         feature_vol_ewma_lam{90,94,97,99}
#     Parkinson (high/low range)          feature_vol_park_{5,10,20,40,60,120}d
#     Yang-Zhang (full OHLC + prev close) feature_vol_yz_{5,10,20,40,60,120}d
#   All per-symbol, raw from spot, attached as-of <= D-1, rank-binned per date
#   (build_vol.py -> volatility_features.py). Paired against the raw 22:
#     base   crypto/v2.0        22f  (raw, the original)
#     vol    crypto/v2.0-vol    44f  (raw 22 + 22 volatility estimators)
#
#   This is an ADD-block ablation: vol == base + the 22 feature_vol_ columns, so the
#   paired delta reads cleanly as "does the volatility block help". To race the block
#   ALONE (22f, tuned from scratch, no base), add E74_ISOLATE=1 (uses
#   --feature-include '^feature_vol_' on the same crypto/v2.0-vol dataset).
#
# WHAT THIS RUNS  10 arms = 5 models x {base, vol}, tuned from scratch, uniform
#   n_iter=50, same cadence (rolling 130/130, embargo auto=20, 5 folds, val_frac
#   0.3, seed 42, target_binned_return_20, metric corr). Paired within model.
#
# WHAT WOULD COUNT AS A RESULT  vol beating raw at paired Newey-West |t|>=2 (lag
#   20). Prior: no engineered block has beaten the raw 22 on this panel at |t|>=2,
#   and the base already has feature_volatility_{20,60}d, so the block must add
#   OVER those two, not merely re-supply volatility.
#
# RUNTIME  small (44f), ~2-4h for ten, serial. Background it; mails start + done.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e74_vol_vs_raw_crypto.sh
#   DRY_RUN=1 / E74_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_vol lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
#   E74_ISOLATE=1  -> the vol arms use only the 22 feature_vol_ columns (block alone)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E74}"
E74_WINDOW="${E74_WINDOW:-130}"
E74_REFIT="${E74_REFIT:-130}"
E74_EMBARGO="${E74_EMBARGO:-auto}"
E74_N_ITER="${E74_N_ITER:-50}"
E74_N_SPLITS="${E74_N_SPLITS:-5}"
E74_VAL_FRAC="${E74_VAL_FRAC:-0.3}"
E74_SEED="${E74_SEED:-42}"
E74_TARGET="${E74_TARGET:-target_binned_return_20}"
E74_ERA_STEP="${E74_ERA_STEP:-1}"
E74_BASE_DATA="${E74_BASE_DATA:-crypto/v2.0}"
E74_VOL_DATA="${E74_VOL_DATA:-crypto/v2.0-vol}"
E74_ISOLATE="${E74_ISOLATE:-0}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E74_WINDOW}" in ""|0|[0-9]) echo "E74_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E74_REFIT}"  in ""|0|[0-9]) echo "E74_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E74_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E74_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e74"; SCOUT=0
if [ "${E74_SCOUT:-0}" = "1" ]; then
    PREFIX="e74scout"; SCOUT=1; E74_ERA_STEP=6; E74_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E74_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

# The vol arms optionally isolate the block (22 feature_vol_ only) instead of
# adding it to the base. --feature-include is safe here: '^feature_vol_' matches the
# 22 estimators and NOT the base feature_volatility_* / feature_volume_* columns.
VOL_INCLUDE=()
if [ "${E74_ISOLATE}" = "1" ]; then VOL_INCLUDE=(--feature-include '^feature_vol_'); fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base" "${mn}" "${E74_BASE_DATA}" "${sj}" "${nj}"
    aadd "${mn}_vol"  "${mn}" "${E74_VOL_DATA}"  "${sj}" "${nj}"
done
NITER_USE="${E74_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=22, vol=44 with the 22 feature_vol_ columns present + label + meta,
# and the 5 wrappers construct with the expected grid sizes.
"$PYTHON" - "${E74_BASE_DATA}" "${E74_VOL_DATA}" "${E74_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, vol_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 22), (vol_v, 44)]:
    p = "data/%s/train.parquet" % v
    try:
        names = pq.ParquetFile(p).schema.names
    except OSError:
        bad.append("%s MISSING" % v); continue
    got = sum(cc.startswith("feature_") for cc in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (v, got, want))
    if target not in names:
        bad.append("%s lacks %s" % (v, target))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)
vv = pq.ParquetFile("data/%s/train.parquet" % vol_v).schema.names
n_vol = sum(cc.startswith("feature_vol_") for cc in vv)
if n_vol != 22:
    bad.append("%s has %d feature_vol_ columns, expected 22" % (vol_v, n_vol))
for name, want_cells in zip(models, cells):
    try:
        w = build_wrapper(name, "regression", seed=42, n_jobs=1)
        got_cells = len(list(ParameterGrid(w.param_grid)))
    except Exception as e:
        bad.append("%s construct/grid failed: %s" % (name, e)); continue
    if got_cells != want_cells:
        bad.append("%s grid has %d cells, expected %d" % (name, got_cells, want_cells))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: base=22 (raw), vol=44 (incl. 22 feature_vol_ estimators) "
      "with label+meta; 5 wrappers construct.")
PY

SEL_ARM=(); SEL_MODEL=(); SEL_DATA=(); SEL_SJOBS=(); SEL_NJOBS=(); picked=0
for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
    SEL_ARM+=("$arm"); SEL_MODEL+=("${MODEL[$i]}"); SEL_DATA+=("${DATA[$i]}")
    SEL_SJOBS+=("${SJOBS[$i]}"); SEL_NJOBS+=("${NJOBS[$i]}")
done
total="${#SEL_ARM[@]}"
[ "$total" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }
ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: 5 models x {base v2.0 raw (22f), vol v2.0-vol (44f)}, paired
  block     22 realised-volatility estimators (feature_vol_*): sqret/ewma/parkinson/yang-zhang x lookback grid
  question  does the volatility block beat the raw 22? (add-block ablation; isolate=${E74_ISOLATE})
  cadence   rolling ${E74_WINDOW}/${E74_REFIT}, embargo ${E74_EMBARGO}, ${E74_N_SPLITS} folds, seed ${E74_SEED}, n_iter=${E74_N_ITER}

  runner    experiments/e74_vol_vs_raw_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: vol(44) vs raw(22)  |  ${total} arm(s)  shard ${SHARD}/${OF}  isolate=${E74_ISOLATE}"
echo "cadence rolling ${E74_WINDOW}/${E74_REFIT}, embargo ${E74_EMBARGO}, ${E74_N_SPLITS} folds, seed ${E74_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E74_WINDOW}" --refit-every "${E74_REFIT}"
         --embargo "${E74_EMBARGO}" --n-splits "${E74_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E74_VAL_FRAC}" --era-step "${E74_ERA_STEP}" --seed "${E74_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E74_TARGET}" --run-name "${run_name}")
    # isolate the block on the vol arms only (base arms are the raw 22, unfiltered)
    if [ "${E74_ISOLATE}" = "1" ] && [[ "${arm}" == *_vol ]]; then
        cmd+=("${VOL_INCLUDE[@]}")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-34s\n' "$((i+1))" "$total" "$run_name" "$model" "$dsver"
        printf '       %s\n' "${cmd[*]}"; continue
    fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run_name}.json" ]; then
        echo "[$((i+1))/$total] ${run_name} already on disk -- skipping (resume-safe)"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${run_name}  ${model}  ${dsver}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${run_name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m$((el%60))s -- continuing (logs/${run_name}.log)"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
PAIRED="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- vol(44) vs raw(22)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; vv="${PREFIX}_${mn}_vol"
    bj="results/metrics/crypto/${mn}/${base}.json"; cj="results/metrics/crypto/${mn}/${vv}.json"
    echo "=== ${mn}: raw vs vol ==="
    if [ ! -s "$bj" ] || [ ! -s "$cj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$cj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
def row(t,d):
    cc=d.get("corr") or {}; m=d.get("mmc") or {}; cm=cc.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-6s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,cc.get("sharpe") or 0.0,d.get("n_periods")))
row("raw",b); row("vol",c)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${vv}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_vol"
done
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
    "$PYTHON" - "$([ "${failed}" -gt 0 ] && echo "WITH ${failed} FAILED" || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
