#!/usr/bin/env bash
#
# Experiment E68: does a cross-sectional momentum REGIME block help on top of the
# momswap feature set? Standalone models tuned from scratch on v2.0-momswap VS the
# same plus 3 cross-sectional momentum-moment features.
#
# E68 is the reference handle; every artefact is e68_*.
#
# THE QUESTION
#   v2.0-momswap = 14 non-momentum base + 17 literature momentum (31f). E68 adds a
#   MARKET-REGIME descriptor of momentum: for each day, the cross-sectional MEAN,
#   VARIANCE and SKEW of the raw (un-binned) momentum distribution across coins,
#   collapsed from the 17 raw signals (per-signal moment -> causal percentile ->
#   average -> causal percentile), broadcast POINT-IN-TIME (D-1) as 3 day-level
#   features (src/data_mani/build_xsmom.py -> crypto/v2.0-momswap-xsmom, 34f).
#     base    crypto/v2.0-momswap          31f  (the swap set)
#     xsmom   crypto/v2.0-momswap-xsmom     34f  (+ xsmom_mean/var/skew)
#
# WHAT TO EXPECT (state it up front, it is structural not a surprise)
#   The 3 features are IDENTICAL for every symbol on a date, so they carry ~zero
#   standalone within-era numerai_corr, and for a LINEAR model (ridge) they are
#   INERT -- a day-constant shifts every same-date prediction equally and cannot
#   change a within-date rank. So expect e68_ridge_xsmom == e68_ridge_base. Any
#   effect must come from the trees / MLP using the regime via INTERACTIONS /
#   gating. Ridge is the negative control that proves the pipeline is honest.
#
# WHAT THIS RUNS  10 arms = 5 models x {base, xsmom}, tuned from scratch, uniform
#   n_iter=50, same cadence as E64-E67 (rolling 130/130, embargo auto=20, 5 folds,
#   val_frac 0.3, seed 42, target_binned_return_20, metric corr). catboost_ktz's
#   12-cell grid clamps 50 -> 12 (a full sweep). Paired within model: xsmom vs base.
#
# WHAT WOULD COUNT AS A RESULT  xsmom beating base on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20) on a tree or the MLP (ridge cannot move). Prior:
#   regime features are hard to make pay on ~26 effective windows, so a null is
#   likely and is itself the finding.
#
# RUNTIME  ~4-6h for ten, serial. Background it; mails start + completion.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 = 3.8
#   no-pyarrow sandbox; preflight refuses it).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e68_xsmom_momswap_crypto.sh
#   DRY_RUN=1 / E68_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_xsmom lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E68}"
E68_WINDOW="${E68_WINDOW:-130}"
E68_REFIT="${E68_REFIT:-130}"
E68_EMBARGO="${E68_EMBARGO:-auto}"
E68_N_ITER="${E68_N_ITER:-50}"
E68_N_SPLITS="${E68_N_SPLITS:-5}"
E68_VAL_FRAC="${E68_VAL_FRAC:-0.3}"
E68_SEED="${E68_SEED:-42}"
E68_TARGET="${E68_TARGET:-target_binned_return_20}"
E68_ERA_STEP="${E68_ERA_STEP:-1}"
E68_BASE_DATA="${E68_BASE_DATA:-crypto/v2.0-momswap}"
E68_XSMOM_DATA="${E68_XSMOM_DATA:-crypto/v2.0-momswap-xsmom}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E68_WINDOW}" in ""|0|[0-9]) echo "E68_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E68_REFIT}"  in ""|0|[0-9]) echo "E68_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E68_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E68_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e68"; SCOUT=0
if [ "${E68_SCOUT:-0}" = "1" ]; then
    PREFIX="e68scout"; SCOUT=1; E68_ERA_STEP=6; E68_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E68_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base"  "${mn}" "${E68_BASE_DATA}"  "${sj}" "${nj}"
    aadd "${mn}_xsmom" "${mn}" "${E68_XSMOM_DATA}" "${sj}" "${nj}"
done
NITER_USE="${E68_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=31, xsmom=34 with the 3 xsmom features present + label + meta,
# and the 5 wrappers construct with their KTZ grids.
"$PYTHON" - "${E68_BASE_DATA}" "${E68_XSMOM_DATA}" "${E68_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, xs_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 31), (xs_v, 34)]:
    p = "data/%s/train.parquet" % v
    try:
        names = pq.ParquetFile(p).schema.names
    except OSError:
        bad.append("%s MISSING" % v); continue
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (v, got, want))
    if target not in names:
        bad.append("%s lacks %s" % (v, target))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)
xs_names = pq.ParquetFile("data/%s/train.parquet" % xs_v).schema.names
for f in ("feature_xsmom_mean", "feature_xsmom_var", "feature_xsmom_skew"):
    if f not in xs_names:
        bad.append("%s lacks %s" % (xs_v, f))
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
print("preflight OK: base=31, xsmom=34 (incl. feature_xsmom_{mean,var,skew}) with "
      "label+meta; 5 wrappers construct.")
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

  arms      ${total}: 5 models x {base v2.0-momswap (31f), xsmom v2.0-momswap-xsmom (34f)}, paired
  block     3 cross-sectional momentum moments (mean/var/skew), day-level, causal-time percentile
  note      xsmom is within-date constant -> ridge is INERT (negative control); only trees/MLP can use it
  cadence   rolling ${E68_WINDOW}/${E68_REFIT}, embargo ${E68_EMBARGO}, ${E68_N_SPLITS} folds, seed ${E68_SEED}, n_iter=${E68_N_ITER}

  runner    experiments/e68_xsmom_momswap_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: cross-sectional momentum regime, base vs +xsmom  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E68_WINDOW}/${E68_REFIT}, embargo ${E68_EMBARGO}, ${E68_N_SPLITS} folds, seed ${E68_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E68_WINDOW}" --refit-every "${E68_REFIT}"
         --embargo "${E68_EMBARGO}" --n-splits "${E68_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E68_VAL_FRAC}" --era-step "${E68_ERA_STEP}" --seed "${E68_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E68_TARGET}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-26s\n' "$((i+1))" "$total" "$run_name" "$model" "$dsver"
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
echo "${EXPERIMENT} -- cross-sectional momentum regime: base(31) vs +xsmom(34)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; xs="${PREFIX}_${mn}_xsmom"
    bj="results/metrics/crypto/${mn}/${base}.json"; xj="results/metrics/crypto/${mn}/${xs}.json"
    echo "=== ${mn}: base vs +xsmom ==="
    if [ ! -s "$bj" ] || [ ! -s "$xj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$xj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); x=json.load(open(sys.argv[2]))
def row(t,d):
    c=d.get("corr") or {}; m=d.get("mmc") or {}; cm=c.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-6s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.0,d.get("n_periods")))
row("base",b); row("xsmom",x)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${xs}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_xsmom"
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
