#!/usr/bin/env bash
#
# Experiment E69: does a MACRO / cross-asset information set help on top of the
# momswap features? Standalone models tuned from scratch on v2.0-momswap VS the
# same plus the 15 shared macro features. The E68 sibling with a different
# information set: E68 added a cross-sectional momentum REGIME block, E69 adds
# EXTERNAL macro state.
#
# E69 is the reference handle; every artefact is e69_*.
#
# THE QUESTION
#   v2.0-momswap = 14 non-momentum base + 17 literature momentum (31f). E69 adds
#   the 18 shared macro / cross-asset features (feature_macro_*: gold & silver
#   returns 20d/60d, S&P returns 5d/20d/60d, DXY dollar-index momentum 5d/20d/60d,
#   US Treasury yields y1/y2/y5/y10/y30 and curve slopes 1s10s/2s10s/5s30s),
#   causal-percentile normalised across TIME and broadcast POINT-IN-TIME (D-1) as
#   day-level features (src/data_mani/build_macro.py -> crypto/v2.0-momswap-macro,
#   49f).
#     base    crypto/v2.0-momswap          31f  (the swap set)
#     macro   crypto/v2.0-momswap-macro     49f  (+ 18 feature_macro_, incl. 3 DXY momentum)
#
# WHAT TO EXPECT (structural, not a surprise)
#   Like E68's xsmom block, macro features are IDENTICAL for every symbol on a
#   date, so they carry ~zero standalone within-era numerai_corr and are INERT for
#   a LINEAR model (ridge): a day-constant shifts every same-date prediction
#   equally and cannot change a within-date rank. So expect e69_ridge_macro ==
#   e69_ridge_base -- ridge is the negative control that proves the pipeline is
#   honest. Any effect must come from the trees / MLP using the macro state via
#   regime gating / interactions with the per-symbol features.
#
# WHAT THIS RUNS  10 arms = 5 models x {base, macro}, tuned from scratch, uniform
#   n_iter=50, same cadence as E64-E68 (rolling 130/130, embargo auto=20, 5 folds,
#   val_frac 0.3, seed 42, target_binned_return_20, metric corr). catboost_ktz's
#   12-cell grid clamps 50 -> 12 (a full sweep). Paired within model: macro vs base.
#
# WHAT WOULD COUNT AS A RESULT  macro beating base on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20) on a tree or the MLP (ridge cannot move). Prior:
#   the macro block's own note (macro-features memory) reports ~zero standalone
#   within-era IC and help only via interactions, so a null is likely and is itself
#   the finding.
#
# RUNTIME  ~4-6h for ten, serial. Runs concurrently with E68 (64-core box).
#   Background it; mails start + completion.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 = 3.8
#   no-pyarrow sandbox; preflight refuses it).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e69_macro_momswap_crypto.sh
#   DRY_RUN=1 / E69_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_macro lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E69}"
E69_WINDOW="${E69_WINDOW:-130}"
E69_REFIT="${E69_REFIT:-130}"
E69_EMBARGO="${E69_EMBARGO:-auto}"
E69_N_ITER="${E69_N_ITER:-50}"
E69_N_SPLITS="${E69_N_SPLITS:-5}"
E69_VAL_FRAC="${E69_VAL_FRAC:-0.3}"
E69_SEED="${E69_SEED:-42}"
E69_TARGET="${E69_TARGET:-target_binned_return_20}"
E69_ERA_STEP="${E69_ERA_STEP:-1}"
E69_BASE_DATA="${E69_BASE_DATA:-crypto/v2.0-momswap}"
E69_MACRO_DATA="${E69_MACRO_DATA:-crypto/v2.0-momswap-macro}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E69_WINDOW}" in ""|0|[0-9]) echo "E69_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E69_REFIT}"  in ""|0|[0-9]) echo "E69_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E69_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E69_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e69"; SCOUT=0
if [ "${E69_SCOUT:-0}" = "1" ]; then
    PREFIX="e69scout"; SCOUT=1; E69_ERA_STEP=6; E69_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E69_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base"  "${mn}" "${E69_BASE_DATA}"  "${sj}" "${nj}"
    aadd "${mn}_macro" "${mn}" "${E69_MACRO_DATA}" "${sj}" "${nj}"
done
NITER_USE="${E69_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=31, macro=46 with 15 feature_macro_ present + label + meta,
# and the 5 wrappers construct with their KTZ grids.
"$PYTHON" - "${E69_BASE_DATA}" "${E69_MACRO_DATA}" "${E69_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, macro_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 31), (macro_v, 49)]:
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
macro_names = pq.ParquetFile("data/%s/train.parquet" % macro_v).schema.names
n_macro = sum(cc.startswith("feature_macro_") for cc in macro_names)
if n_macro != 18:
    bad.append("%s has %d feature_macro_ columns, expected 18" % (macro_v, n_macro))
n_dxy = sum("dxy" in cc for cc in macro_names)
if n_dxy != 3:
    bad.append("%s has %d dxy columns, expected 3 (feature_macro_dxy_ret_*)" % (macro_v, n_dxy))
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
print("preflight OK: base=31, macro=49 (incl. 18 feature_macro_, +3 DXY momentum) "
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

  arms      ${total}: 5 models x {base v2.0-momswap (31f), macro v2.0-momswap-macro (49f)}, paired
  block     18 shared macro/cross-asset features (gold/silver/S&P returns, DXY momentum, US yields, curve slopes)
  note      macro is within-date constant -> ridge is INERT (negative control); only trees/MLP can use it
  cadence   rolling ${E69_WINDOW}/${E69_REFIT}, embargo ${E69_EMBARGO}, ${E69_N_SPLITS} folds, seed ${E69_SEED}, n_iter=${E69_N_ITER}

  runner    experiments/e69_macro_momswap_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: macro information set, base vs +macro  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E69_WINDOW}/${E69_REFIT}, embargo ${E69_EMBARGO}, ${E69_N_SPLITS} folds, seed ${E69_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E69_WINDOW}" --refit-every "${E69_REFIT}"
         --embargo "${E69_EMBARGO}" --n-splits "${E69_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E69_VAL_FRAC}" --era-step "${E69_ERA_STEP}" --seed "${E69_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E69_TARGET}" --run-name "${run_name}")
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
echo "${EXPERIMENT} -- macro information set (incl. DXY momentum): base(31) vs +macro(49)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; mac="${PREFIX}_${mn}_macro"
    bj="results/metrics/crypto/${mn}/${base}.json"; xj="results/metrics/crypto/${mn}/${mac}.json"
    echo "=== ${mn}: base vs +macro ==="
    if [ ! -s "$bj" ] || [ ! -s "$xj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$xj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); x=json.load(open(sys.argv[2]))
def row(t,d):
    c=d.get("corr") or {}; m=d.get("mmc") or {}; cm=c.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-6s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.0,d.get("n_periods")))
row("base",b); row("macro",x)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${mac}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_macro"
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
