#!/usr/bin/env bash
#
# Experiment E72: does the fully-engineered carver set beat the raw 22?
# crypto/v2.0-momswap-macro-carver VS crypto/v2.0 (the raw 22 as the base).
#
# E72 is the reference handle; every artefact is e72_*.
#
# THE QUESTION
#   The carver dataset (64f) is the whole engineering stack: the momentum swap (14
#   non-momentum base + 17 literature momentum), the 18 macro / cross-asset features
#   (incl. DXY momentum), and 15 Robert Carver-style TREND-FOLLOWING features
#   (feature_tf_*: EWMAC crossovers 2/8..32/128, breakouts 20/40/80/160, Bollinger-z
#   20/40/80, Clenow 60/90, new-high-20-in-5). E72 asks the bottom-line question:
#   does all of that, tuned from scratch, beat the plain raw 22 the book started on?
#     base    crypto/v2.0                       22f  (raw, the original)
#     carver  crypto/v2.0-momswap-macro-carver  64f  (swap + macro + 15 tf)
#
#   NOTE this is a whole-stack comparison, not a single-block ablation: base and
#   carver differ in many blocks at once (the raw momentum proxies are GONE in
#   carver, replaced by literature momentum + macro + trend-following). Read the
#   paired delta as "is the engineered 64 better than the raw 22", the deployment
#   question, not "which block did it".
#
# WHAT THIS RUNS  10 arms = 5 models x {base, carver}, tuned from scratch, uniform
#   n_iter=50, same cadence (rolling 130/130, embargo auto=20, 5 folds, val_frac
#   0.3, seed 42, target_binned_return_20, metric corr). Paired within model.
#
# WHAT WOULD COUNT AS A RESULT  carver beating raw at paired Newey-West |t|>=2
#   (lag 20). Prior: no engineered block has beaten the raw 22 on this panel at
#   |t|>=2 across ~11 tries; the carver trend features are the freshest input, so
#   this is their bottom-line test.
#
# RUNTIME  small (64f), ~2-4h for ten, serial. Background it; mails start + done.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e72_carver_vs_raw_crypto.sh
#   DRY_RUN=1 / E72_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_carver lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E72}"
E72_WINDOW="${E72_WINDOW:-130}"
E72_REFIT="${E72_REFIT:-130}"
E72_EMBARGO="${E72_EMBARGO:-auto}"
E72_N_ITER="${E72_N_ITER:-50}"
E72_N_SPLITS="${E72_N_SPLITS:-5}"
E72_VAL_FRAC="${E72_VAL_FRAC:-0.3}"
E72_SEED="${E72_SEED:-42}"
E72_TARGET="${E72_TARGET:-target_binned_return_20}"
E72_ERA_STEP="${E72_ERA_STEP:-1}"
E72_BASE_DATA="${E72_BASE_DATA:-crypto/v2.0}"
E72_CARVER_DATA="${E72_CARVER_DATA:-crypto/v2.0-momswap-macro-carver}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E72_WINDOW}" in ""|0|[0-9]) echo "E72_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E72_REFIT}"  in ""|0|[0-9]) echo "E72_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E72_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E72_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e72"; SCOUT=0
if [ "${E72_SCOUT:-0}" = "1" ]; then
    PREFIX="e72scout"; SCOUT=1; E72_ERA_STEP=6; E72_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E72_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base"   "${mn}" "${E72_BASE_DATA}"   "${sj}" "${nj}"
    aadd "${mn}_carver" "${mn}" "${E72_CARVER_DATA}" "${sj}" "${nj}"
done
NITER_USE="${E72_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=22, carver=64 with the 15 tf_ (carver) features present + label +
# meta, and the 5 wrappers construct.
"$PYTHON" - "${E72_BASE_DATA}" "${E72_CARVER_DATA}" "${E72_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, carver_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 22), (carver_v, 64)]:
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
cv = pq.ParquetFile("data/%s/train.parquet" % carver_v).schema.names
n_tf = sum(cc.startswith("feature_tf_") for cc in cv)
if n_tf != 15:
    bad.append("%s has %d feature_tf_ (carver) columns, expected 15" % (carver_v, n_tf))
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
print("preflight OK: base=22 (raw), carver=64 (incl. 15 feature_tf_ trend-following) "
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

  arms      ${total}: 5 models x {base v2.0 raw (22f), carver v2.0-momswap-macro-carver (64f)}, paired
  block     whole stack: momentum swap + 18 macro (incl DXY) + 15 Carver trend-following (tf_)
  question  does the engineered 64 beat the raw 22? (whole-stack, not a single-block ablation)
  cadence   rolling ${E72_WINDOW}/${E72_REFIT}, embargo ${E72_EMBARGO}, ${E72_N_SPLITS} folds, seed ${E72_SEED}, n_iter=${E72_N_ITER}

  runner    experiments/e72_carver_vs_raw_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: carver(64) vs raw(22)  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E72_WINDOW}/${E72_REFIT}, embargo ${E72_EMBARGO}, ${E72_N_SPLITS} folds, seed ${E72_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E72_WINDOW}" --refit-every "${E72_REFIT}"
         --embargo "${E72_EMBARGO}" --n-splits "${E72_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E72_VAL_FRAC}" --era-step "${E72_ERA_STEP}" --seed "${E72_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E72_TARGET}" --run-name "${run_name}")
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
echo "${EXPERIMENT} -- carver(64) vs raw(22)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; cv="${PREFIX}_${mn}_carver"
    bj="results/metrics/crypto/${mn}/${base}.json"; cj="results/metrics/crypto/${mn}/${cv}.json"
    echo "=== ${mn}: raw vs carver ==="
    if [ ! -s "$bj" ] || [ ! -s "$cj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$cj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
def row(t,d):
    cc=d.get("corr") or {}; m=d.get("mmc") or {}; cm=cc.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-6s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,cc.get("sharpe") or 0.0,d.get("n_periods")))
row("raw",b); row("carver",c)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${cv}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_carver"
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
