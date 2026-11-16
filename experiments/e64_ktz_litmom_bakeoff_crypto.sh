#!/usr/bin/env bash
#
# Experiment E64: do the litmom momentum features help ACROSS MODEL CLASSES,
# when every model is TUNED FROM SCRATCH under its KTZ priors?
#
# E64 is the reference handle; every artefact is e64_*.
#
# THE QUESTION
#   E63 asked "does litmom help?" for two models (lgb, ridge_glmnet). E64 widens
#   that to five function classes -- a linear model, two boosters, a bagged forest
#   and a feedforward net -- so the answer is not read off one model family. Each
#   is fit FROM SCRATCH (a real RandomizedSearchCV that SELECTS a config per
#   refit, no pinned/deployed fit reused) on target_binned_return_20, and each is
#   run once on the raw 22 (crypto/v2.0) and once on 22 + 17 literature momentum
#   (crypto/v2.0-litmom). base pairs cleanly against litmom on the SAME model, same
#   seed, same periods -- the paired Newey-West delta is the finding.
#
# "KTZ PRIORS"  = the Kalfa-Timmermann-van der Zwan hyperparameter-sensitivity
#   findings, compiled per model into a compact search grid that only ever draws
#   from the study's GOOD REGION (so the budget fine-tunes, it does not explore):
#     lgb_ktz       E27  (learning_rate x colsample x num_leaves x n_est, 432 cells)
#     catboost_ktz  E29/E32 (lr x depth x rsm, 12 cells -- swept exhaustively)
#     rf_ktz        E48  (depth x max_features x min_samples_leaf, 60 cells)
#     mlp_ktz       E17  (dense learning-rate axis x 3 archs x 2 alpha, 120 cells)
#   ridge has NO ktz variant and does not need one: its only knob is the L2
#   penalty alpha, so the plain RidgeWrapper's fixed log-alpha grid IS the
#   evidenced search. It is the linear control the four tuned models are read
#   against, and it is called out here so "ridge under ktz priors" is not read as
#   a missing wrapper.
#
# WHAT THIS RUNS  10 arms = 5 models x {base, litmom}, matched cadence
#   (rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3, seed 42, metric
#   corr, MMC over the real meta model). n_iter is set PER MODEL at each grid's
#   evidenced budget (ridge 50 / lgb_ktz 50 / catboost_ktz 12 / rf_ktz 40 /
#   mlp_ktz 80); it is identical between a model's two arms, which is all the
#   WITHIN-model pairing needs. It is deliberately NOT equal across models -- a
#   12-cell grid swept and a 432-cell grid sampled are different budgets by
#   design, and cross-model levels are read with that in mind (and numerai_corr
#   is comparable across these arms only because every arm scores the SAME
#   20-day label).
#
# WHAT WOULD COUNT AS A RESULT
#   Per model, litmom beating its matched base on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20). The clean finding is CONSISTENCY: if the momentum
#   block carries ranking information the binned 22 lack, it should help more than
#   one function class; if it helps a booster but not a linear model, that says
#   the signal is an interaction the binned features already span linearly. A null
#   across all five, with intervals, is equally a finding -- E63's pre-run EDA put
#   the incremental IC at |IC|~0.015 (medium-horizon returns), which is small.
#
# RUNTIME  measured before launch, not guessed. Price with a scout first:
#     E64_SCOUT=1 PYTHON=... bash experiments/e64_ktz_litmom_bakeoff_crypto.sh
#   Rough full-resolution budget at this cadence: ridge ~25m/arm, catboost_ktz
#   ~15m, rf_ktz ~35m, mlp_ktz ~90m, lgb_ktz ~75m -> ~8h for all ten, serial.
#   Run it in the background; it mails a start notice and a completion report.
#
# INTERPRETER  bare `python3` on this box is a 3.8 sandbox WITHOUT pyarrow/lgbm/
#   catboost and preflight will refuse it. Pass the full stack explicitly:
#     PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e64_...sh
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e64_ktz_litmom_bakeoff_crypto.sh
#   DRY_RUN=1 / E64_SCOUT=1 / NO_MAIL=1
#   ONLY="ridge_litmom lgb_ktz_base"   SKIP=...   SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E64}"
E64_WINDOW="${E64_WINDOW:-130}"
E64_REFIT="${E64_REFIT:-130}"
E64_EMBARGO="${E64_EMBARGO:-auto}"
E64_N_SPLITS="${E64_N_SPLITS:-5}"
E64_VAL_FRAC="${E64_VAL_FRAC:-0.3}"
E64_SEED="${E64_SEED:-42}"
E64_TARGET="${E64_TARGET:-target_binned_return_20}"
E64_ERA_STEP="${E64_ERA_STEP:-1}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

# GNU screen exports $WINDOW=0 into every session it starts -- refuse the silent
# empty training block that a bare WINDOW="${WINDOW:-...}" would produce.
case "${E64_WINDOW}" in ""|0|[0-9]) echo "E64_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E64_REFIT}"  in ""|0|[0-9]) echo "E64_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E64_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E64_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e64"
SCOUT=0
if [ "${E64_SCOUT:-0}" = "1" ]; then
    PREFIX="e64scout"; SCOUT=1; E64_ERA_STEP=6; E64_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E64_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

# --------------------------------------------------------------------------- #
# The arm table. Parallel arrays (never pipe-delimited: a feature regex uses
# '|'). Built from a per-model spec x {base, litmom} so the two views are
# guaranteed identical in everything but the dataset.
#   MODELS_SPEC: model  n_iter  search_jobs  n_jobs   expected_grid_cells
# --------------------------------------------------------------------------- #
MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_NITER=(50           50       12            40      80)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); NITER=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); NITER+=("$4"); SJOBS+=("$5"); NJOBS+=("$6"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; ni="${MODEL_NITER[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    if [ "${SCOUT}" = "1" ]; then ni=4; [ "${mn}" = "catboost_ktz" ] && ni=3; fi
    aadd "${mn}_base"   "${mn}" "crypto/v2.0"        "${ni}" "${sj}" "${nj}"
    aadd "${mn}_litmom" "${mn}" "crypto/v2.0-litmom" "${ni}" "${sj}" "${nj}"
done

# --------------------------------------------------------------------------- #
# Preflight: both datasets present with meta files and the expected feature
# counts (v2.0 = 22, v2.0-litmom = 32), the 20-day label present, and every one
# of the five wrappers constructs on regression with the KTZ grid it should --
# a silent wrapper/grid drift would attach a plausible number to the wrong
# experiment, which is worse than a crash.
# --------------------------------------------------------------------------- #
"$PYTHON" - "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" -- "${E64_TARGET}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

a = sys.argv[1:]; c1 = a.index("--"); c2 = a.index("--", c1 + 1)
models = a[:c1]; cells = [int(x) for x in a[c1 + 1:c2]]; target = a[c2 + 1]

DS = [("crypto/v2.0", 22), ("crypto/v2.0-litmom", 32)]
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
    grid = w.param_grid
    try:
        got_cells = len(list(ParameterGrid(grid)))
    except Exception:
        got_cells = None            # a list-of-subgrids or a distribution grid
    if got_cells is not None and got_cells != want_cells:
        bad.append("%s grid has %d cells, expected %d" % (name, got_cells, want_cells))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: v2.0 (22f) and v2.0-litmom (32f) present with meta files and "
      "the 20-day label; all 5 wrappers construct with their KTZ grids "
      "(ridge 100 / lgb_ktz 432 / catboost_ktz 12 / rf_ktz 60 / mlp_ktz 120).")
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

  arms      ${total}: 5 models x {base v2.0, litmom v2.0-litmom}, paired within model
  models    ridge (linear control) + lgb_ktz catboost_ktz rf_ktz mlp_ktz (KTZ priors)
  budget    n_iter per model {ridge 50, lgb_ktz 50, catboost_ktz 12, rf_ktz 40, mlp_ktz 80}
  cadence   rolling ${E64_WINDOW}/${E64_REFIT}, embargo ${E64_EMBARGO}, ${E64_N_SPLITS} folds, seed ${E64_SEED}
  target    ${E64_TARGET}  (numerai_corr; MMC over the real meta model)

  runner    experiments/e64_ktz_litmom_bakeoff_crypto.sh
  results   results/metrics/crypto/{ridge,lgb_ktz,catboost_ktz,rf_ktz,mlp_ktz}/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    this study is ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: KTZ-prior model bake-off, litmom vs base  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E64_WINDOW}/${E64_REFIT}, embargo ${E64_EMBARGO}, ${E64_N_SPLITS} folds, seed ${E64_SEED}, target ${E64_TARGET}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    niter="${SEL_NITER[$i]}"; sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E64_WINDOW}" --refit-every "${E64_REFIT}"
         --embargo "${E64_EMBARGO}" --n-splits "${E64_N_SPLITS}" --n-iter "${niter}"
         --val-frac "${E64_VAL_FRAC}" --era-step "${E64_ERA_STEP}" --seed "${E64_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E64_TARGET}" --run-name "${run_name}")
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
# Report: per model, base vs litmom summary + the paired Newey-West delta (the
# finding) + the report_html/report_pdf command. Everything read from the same
# metrics json and per_period csv the renderers read.
# --------------------------------------------------------------------------- #
REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
PAIRED="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- KTZ-prior model bake-off: litmom vs base, 5 model classes"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "  results  results/metrics/crypto/<model>/${PREFIX}_*.json"
echo "  runner   experiments/e64_ktz_litmom_bakeoff_crypto.sh"
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
    # The finding: paired Newey-West delta (litmom - base), lag 20, inner join of periods.
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${lit}" 2>/dev/null \
        || echo "  (paired_delta unavailable -- run it by hand once both arms exist)"
    echo
done
echo "=============================================================="
echo "FULL REPORT (ranked table, corr matrices, screened ensembles, backtest curve):"
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
