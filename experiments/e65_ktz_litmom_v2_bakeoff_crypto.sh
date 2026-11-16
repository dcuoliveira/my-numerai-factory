#!/usr/bin/env bash
#
# Experiment E65: E64 re-run on the UPDATED litmom, at a uniform n_iter=50.
#
# E65 is the reference handle; every artefact is e65_*.
#
# WHY A RE-RUN (what changed since E64)
#   1. litmom grew. E64 ran on crypto/v2.0-litmom = 22 base + 10 momentum (32
#      feature columns). Since then the dataset was rebuilt to 22 + 17 momentum
#      (39 columns) -- the original 10 plus a 7-feature trend block
#      (feature_mom_trend_7d/14d/30d/60d/120d/240d and feature_mom_trend_robust,
#      the Tzotchev robust trend). E64's litmom numbers do NOT reflect those.
#   2. Uniform search budget. E64 used a per-model n_iter (ridge 50 / lgb_ktz 50 /
#      catboost_ktz 12 / rf_ktz 40 / mlp_ktz 80). E65 fixes n_iter=50 for every
#      model so the search budget is one number across the board.
#
#   E64 is left intact -- its 22+10 results stand. Reading E65 against E64 shows
#   whether the added trend block moved the paired base-vs-litmom delta.
#
# WHAT THIS RUNS  10 arms = 5 models x {base, litmom}, matched cadence
#   (rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed 42, metric
#   corr, MMC over the real meta model), n_iter=50 EVERY model. base pairs against
#   litmom on the SAME model, same seed, same periods -- the paired Newey-West
#   delta is the finding. base arms are re-run at 50 too so the WITHIN-model
#   pairing is exact (ridge/lgb base at 50 reproduce their E64 twins; the point is
#   a self-contained, uniformly-budgeted study).
#
#     model         grid   n_iter  note
#     ridge          100      50   linear control (no ktz variant)
#     lgb_ktz        432      50
#     catboost_ktz    12      50   n_iter>grid: the sampler CLAMPS to 12 (a full
#                                  sweep, with a harmless "smaller than n_iter"
#                                  warning) -- catboost is exhaustive either way.
#     rf_ktz          60      50
#     mlp_ktz        120      50
#
# WHAT WOULD COUNT AS A RESULT  per model, litmom beating its matched base on
#   numerai_corr at paired Newey-West |t|>=2 (lag 20). Clean finding = CONSISTENCY
#   across function classes. A null across all five, with intervals, is equally a
#   finding. Also read E65 vs E64: did the trend block change any model's verdict?
#
# RUNTIME  ~4-6h for all ten, serial, at this cadence. Background it; it mails a
#   start notice and a completion report.
#
# INTERPRETER  bare `python3` is a 3.8 sandbox WITHOUT pyarrow/lgbm/catboost and
#   preflight refuses it. Pass the full stack:
#     PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e65_...sh
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e65_ktz_litmom_v2_bakeoff_crypto.sh
#   DRY_RUN=1 / E65_SCOUT=1 / NO_MAIL=1
#   ONLY="ridge_litmom lgb_ktz_base"   SKIP=...   SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E65}"
E65_WINDOW="${E65_WINDOW:-130}"
E65_REFIT="${E65_REFIT:-130}"
E65_EMBARGO="${E65_EMBARGO:-auto}"
E65_N_ITER="${E65_N_ITER:-50}"
E65_N_SPLITS="${E65_N_SPLITS:-5}"
E65_VAL_FRAC="${E65_VAL_FRAC:-0.3}"
E65_SEED="${E65_SEED:-42}"
E65_TARGET="${E65_TARGET:-target_binned_return_20}"
E65_ERA_STEP="${E65_ERA_STEP:-1}"
E65_LITMOM_FEAT="${E65_LITMOM_FEAT:-39}"   # 22 base + 17 momentum, post-rebuild
SHARD="${SHARD:-1}"; OF="${OF:-1}"

# GNU screen exports $WINDOW=0 into every session it starts -- refuse the silent
# empty training block that a bare WINDOW="${WINDOW:-...}" would produce.
case "${E65_WINDOW}" in ""|0|[0-9]) echo "E65_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E65_REFIT}"  in ""|0|[0-9]) echo "E65_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E65_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E65_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e65"
SCOUT=0
if [ "${E65_SCOUT:-0}" = "1" ]; then
    PREFIX="e65scout"; SCOUT=1; E65_ERA_STEP=6; E65_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E65_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

# --------------------------------------------------------------------------- #
# The arm table. Parallel arrays. Built from a per-model spec x {base, litmom}
# so the two views are identical in everything but the dataset. n_iter is the
# SAME (E65_N_ITER) for every model; only search/thread jobs differ per model.
#   model  search_jobs  n_jobs  expected_grid_cells
# --------------------------------------------------------------------------- #
MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); NITER=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); NITER+=("$4"); SJOBS+=("$5"); NJOBS+=("$6"); }
NITER_USE="${E65_N_ITER}"
[ "${SCOUT}" = "1" ] && NITER_USE=4
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base"   "${mn}" "crypto/v2.0"        "${NITER_USE}" "${sj}" "${nj}"
    aadd "${mn}_litmom" "${mn}" "crypto/v2.0-litmom" "${NITER_USE}" "${sj}" "${nj}"
done

# --------------------------------------------------------------------------- #
# Preflight: both datasets present with meta files and the expected feature
# counts (v2.0 = 22, v2.0-litmom = E65_LITMOM_FEAT), the 20-day label present,
# and every wrapper constructs with its KTZ grid. A silent feature-count drift
# attaches a plausible number to the wrong experiment -- so gate on it.
# --------------------------------------------------------------------------- #
"$PYTHON" - "${E65_LITMOM_FEAT}" -- "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" -- "${E65_TARGET}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

a = sys.argv[1:]
litfeat = int(a[0])
c1 = a.index("--"); c2 = a.index("--", c1 + 1); c3 = a.index("--", c2 + 1)
models = a[c1 + 1:c2]; cells = [int(x) for x in a[c2 + 1:c3]]; target = a[c3 + 1]

DS = [("crypto/v2.0", 22), ("crypto/v2.0-litmom", litfeat)]
bad = []
for v, want in DS:
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

for name, want_cells in zip(models, cells):
    try:
        w = build_wrapper(name, "regression", seed=42, n_jobs=1)
    except Exception as e:
        bad.append("%s failed to construct: %s" % (name, e)); continue
    try:
        got_cells = len(list(ParameterGrid(w.param_grid)))
    except Exception:
        got_cells = None
    if got_cells is not None and got_cells != want_cells:
        bad.append("%s grid has %d cells, expected %d" % (name, got_cells, want_cells))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: v2.0 (22f) and v2.0-litmom (%df) present with meta files and "
      "the 20-day label; all 5 wrappers construct with their KTZ grids "
      "(ridge 100 / lgb_ktz 432 / catboost_ktz 12 / rf_ktz 60 / mlp_ktz 120)." % litfeat)
PY

# --------------------------------------------------------------------------- #
# Select the work list (ONLY / SKIP / SHARD).
# --------------------------------------------------------------------------- #
SEL_ARM=(); SEL_MODEL=(); SEL_DATA=(); SEL_NITER=(); SEL_SJOBS=(); SEL_NJOBS=(); picked=0
for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_ARM+=("$arm"); SEL_MODEL+=("${MODEL[$i]}"); SEL_DATA+=("${DATA[$i]}")
    SEL_NITER+=("${NITER[$i]}"); SEL_SJOBS+=("${SJOBS[$i]}"); SEL_NJOBS+=("${NJOBS[$i]}")
done
total="${#SEL_ARM[@]}"
[ "$total" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text(sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: 5 models x {base v2.0 (22f), litmom v2.0-litmom (${E65_LITMOM_FEAT}f)}, paired
  vs E64    re-run at UNIFORM n_iter=${E65_N_ITER}, on the rebuilt litmom (22+17, +7 trend incl. robust)
  cadence   rolling ${E65_WINDOW}/${E65_REFIT}, embargo ${E65_EMBARGO}, ${E65_N_SPLITS} folds, seed ${E65_SEED}
  target    ${E65_TARGET}  (numerai_corr; MMC over the real meta model)
  note      catboost_ktz has 12 cells, so n_iter=${E65_N_ITER} clamps to a full 12-cell sweep

  runner    experiments/e65_ktz_litmom_v2_bakeoff_crypto.sh
  results   results/metrics/crypto/{ridge,lgb_ktz,catboost_ktz,rf_ktz,mlp_ktz}/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: KTZ bake-off re-run, litmom(${E65_LITMOM_FEAT}f) vs base, n_iter=${E65_N_ITER}  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E65_WINDOW}/${E65_REFIT}, embargo ${E65_EMBARGO}, ${E65_N_SPLITS} folds, seed ${E65_SEED}, target ${E65_TARGET}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    niter="${SEL_NITER[$i]}"; sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E65_WINDOW}" --refit-every "${E65_REFIT}"
         --embargo "${E65_EMBARGO}" --n-splits "${E65_N_SPLITS}" --n-iter "${niter}"
         --val-frac "${E65_VAL_FRAC}" --era-step "${E65_ERA_STEP}" --seed "${E65_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E65_TARGET}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-22s %-14s %-20s n_iter=%-3s sj=%s nj=%s\n' \
               "$((i+1))" "$total" "$run_name" "$model" "$dsver" "$niter" "$sjobs" "$njobs"
        printf '       %s\n' "${cmd[*]}"; continue
    fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run_name}.json" ]; then
        echo "[$((i+1))/$total] ${run_name} already on disk -- skipping (resume-safe)"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${run_name}  ${model}  ${dsver}  n_iter=${niter}  $(date '+%H:%M:%S')"
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

# --------------------------------------------------------------------------- #
# Report: per model, base vs litmom summary + the paired Newey-West delta.
# --------------------------------------------------------------------------- #
REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
PAIRED="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- KTZ bake-off re-run: litmom(${E65_LITMOM_FEAT}f) vs base, uniform n_iter=${E65_N_ITER}"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "  results  results/metrics/crypto/<model>/${PREFIX}_*.json"
echo "  runner   experiments/e65_ktz_litmom_v2_bakeoff_crypto.sh"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; lit="${PREFIX}_${mn}_litmom"
    bj="results/metrics/crypto/${mn}/${base}.json"; lj="results/metrics/crypto/${mn}/${lit}.json"
    echo "=== ${mn}: base vs litmom ==="
    if [ ! -s "$bj" ] || [ ! -s "$lj" ]; then
        echo "  (one or both arms not on disk -- skipped or another shard)"; echo; continue
    fi
    "$PYTHON" - "$bj" "$lj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b = json.load(open(sys.argv[1])); l = json.load(open(sys.argv[2]))
def row(tag, d):
    c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    print("  %-7s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  c.max_dd %.4f  n=%s"
          % (tag, cm, mm, 0.1*cm + mm, c.get("sharpe") or 0.0,
             c.get("max_drawdown") or 0.0, d.get("n_periods")))
row("base", b); row("litmom", l)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${lit}" 2>/dev/null \
        || echo "  (paired_delta unavailable -- run it by hand once both arms exist)"
    echo
done
echo "=============================================================="
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_litmom"
done
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
    "$PYTHON" - "$([ "${failed}" -gt 0 ] && echo "WITH ${failed} FAILED" || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: completion email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
