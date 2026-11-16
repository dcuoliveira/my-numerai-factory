#!/usr/bin/env bash
#
# Experiment E67: OLD momentum proxies vs NEW (literature) momentum, MOMENTUM-ONLY,
# each tuned from scratch. No other features -- neither block is diluted by the
# rest of the panel, so this isolates the momentum REPRESENTATION itself.
#
# E67 is the reference handle; every artefact is e67_*.
#
# THE QUESTION
#   E64/E65 (add litmom to the full 22) and E66 (swap the base proxies for litmom
#   inside a 31-feature set) both mix momentum with the rest of the panel. E67
#   strips everything else away and races the two momentum blocks head to head:
#     old   the 8 base momentum proxies -- feature_momentum, RSI, Sharpe, Bollinger
#           (each 20d/60d)
#     new   the 17 literature momentum features (feature_mom_ret/max/volscaled/trend)
#   Both are selected from crypto/v2.0-litmom with --feature-include (a regex), so
#   the two arms share rows, dates, cadence and seed and differ ONLY in which
#   momentum representation the model sees. Does the literature block beat the
#   base's own proxies when momentum is all there is?
#
# WHAT THIS RUNS  10 arms = 5 models x {old (8f), new (17f)}, each TUNED FROM
#   SCRATCH (RandomizedSearchCV that selects per step), uniform n_iter=50, same
#   cadence as E64-E66 (rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3,
#   seed 42, target_binned_return_20, metric corr). catboost_ktz's 12-cell grid
#   clamps 50 -> 12 (a full sweep). Paired within model: new vs old.
#
# WHAT WOULD COUNT AS A RESULT  new beating old on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20). A momentum-only arm scores far below a full-panel
#   model (these are 8-17 feature toy sets), so read the DELTA, not the level --
#   the level is not comparable to the deployed book.
#
# RUNTIME  small feature sets, so cheap: ~2-4h for ten, serial. Background it;
#   mails start + completion.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 = 3.8
#   no-pyarrow sandbox; preflight refuses it).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e67_old_vs_new_momentum_crypto.sh
#   DRY_RUN=1 / E67_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_new lgb_ktz_old"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E67}"
E67_WINDOW="${E67_WINDOW:-130}"
E67_REFIT="${E67_REFIT:-130}"
E67_EMBARGO="${E67_EMBARGO:-auto}"
E67_N_ITER="${E67_N_ITER:-50}"
E67_N_SPLITS="${E67_N_SPLITS:-5}"
E67_VAL_FRAC="${E67_VAL_FRAC:-0.3}"
E67_SEED="${E67_SEED:-42}"
E67_TARGET="${E67_TARGET:-target_binned_return_20}"
E67_ERA_STEP="${E67_ERA_STEP:-1}"
E67_DATA="${E67_DATA:-crypto/v2.0-litmom}"
# The two momentum-only feature selections. Single regexes (alternation with '|'),
# each kept in ONE variable and passed as ONE quoted --feature-include arg.
OLD_RX='^feature_(momentum|relative_strength_index|sharpe_ratio|bollinger)_[0-9]+d$'
NEW_RX='^feature_mom_'
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E67_WINDOW}" in ""|0|[0-9]) echo "E67_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E67_REFIT}"  in ""|0|[0-9]) echo "E67_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E67_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E67_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e67"
SCOUT=0
if [ "${E67_SCOUT:-0}" = "1" ]; then
    PREFIX="e67scout"; SCOUT=1; E67_ERA_STEP=6; E67_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E67_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

# Arm table. Per-arm INCL carries the feature-include regex (old vs new).
MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); INCL=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); INCL+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_old" "${mn}" "${OLD_RX}" "${sj}" "${nj}"
    aadd "${mn}_new" "${mn}" "${NEW_RX}" "${sj}" "${nj}"
done
NITER_USE="${E67_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: dataset + meta present; the OLD regex selects EXACTLY the 8 base
# momentum proxies, the NEW regex EXACTLY the 17 literature momentum, the two are
# disjoint, and neither pulls in a non-momentum feature. Gate on it.
"$PYTHON" - "${E67_DATA}" "${OLD_RX}" "${NEW_RX}" "${E67_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, re, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

data, old_rx, new_rx, target = sys.argv[1:5]
rest = sys.argv[5:]; c1 = rest.index("--")
models = rest[:c1]; cells = [int(x) for x in rest[c1 + 1:]]
old = re.compile(old_rx); new = re.compile(new_rx)
bad = []

p = "data/%s/train.parquet" % data
try:
    names = pq.ParquetFile(p).schema.names
except OSError:
    sys.exit("preflight FAILED: %s MISSING" % data)
if target not in names:
    bad.append("%s lacks %s" % (data, target))
if not os.path.exists("data/%s/historical_meta_models.parquet" % data):
    bad.append("%s lacks historical_meta_models.parquet" % data)

feats = [c for c in names if c.startswith("feature_")]
old_sel = sorted(c for c in feats if old.search(c))
new_sel = sorted(c for c in feats if new.search(c))
want_old = sorted([
    "feature_momentum_20d", "feature_momentum_60d",
    "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
    "feature_sharpe_ratio_20d", "feature_sharpe_ratio_60d",
    "feature_bollinger_20d", "feature_bollinger_60d"])
if old_sel != want_old:
    bad.append("OLD include selects %s, expected the 8 base momentum proxies" % old_sel)
if len(new_sel) != 17 or not all(c.startswith("feature_mom_") for c in new_sel):
    bad.append("NEW include selects %d cols (%s...), expected the 17 feature_mom_" % (len(new_sel), new_sel[:3]))
overlap = set(old_sel) & set(new_sel)
if overlap:
    bad.append("OLD and NEW overlap: %s" % sorted(overlap))

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
print("preflight OK: %s has the label+meta; OLD include = 8 base momentum proxies, "
      "NEW include = 17 literature momentum, disjoint, momentum-only; 5 wrappers construct." % data)
PY

# Select work list.
SEL_ARM=(); SEL_MODEL=(); SEL_INCL=(); SEL_SJOBS=(); SEL_NJOBS=(); picked=0
for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
    SEL_ARM+=("$arm"); SEL_MODEL+=("${MODEL[$i]}"); SEL_INCL+=("${INCL[$i]}")
    SEL_SJOBS+=("${SJOBS[$i]}"); SEL_NJOBS+=("${NJOBS[$i]}")
done
total="${#SEL_ARM[@]}"
[ "$total" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text(sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: 5 models x {old (8 base momentum proxies), new (17 literature momentum)}
  momentum-only  NO other features -- both selected from ${E67_DATA} via --feature-include
  question  does the literature momentum block beat the base's own proxies, head to head?
  cadence   rolling ${E67_WINDOW}/${E67_REFIT}, embargo ${E67_EMBARGO}, ${E67_N_SPLITS} folds, seed ${E67_SEED}, n_iter=${E67_N_ITER}
  note      levels are toy (8-17 feat) -- read the paired delta, not the level

  runner    experiments/e67_old_vs_new_momentum_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: OLD vs NEW momentum (momentum-only, from scratch)  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "data ${E67_DATA} | cadence rolling ${E67_WINDOW}/${E67_REFIT}, embargo ${E67_EMBARGO}, ${E67_N_SPLITS} folds, seed ${E67_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; incl="${SEL_INCL[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E67_WINDOW}" --refit-every "${E67_REFIT}"
         --embargo "${E67_EMBARGO}" --n-splits "${E67_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E67_VAL_FRAC}" --era-step "${E67_ERA_STEP}" --seed "${E67_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${E67_DATA}" --target "${E67_TARGET}"
         --feature-include "${incl}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s incl=%s\n' "$((i+1))" "$total" "$run_name" "$model" "$incl"
        printf '       %s\n' "${cmd[*]}"; continue
    fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run_name}.json" ]; then
        echo "[$((i+1))/$total] ${run_name} already on disk -- skipping (resume-safe)"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${run_name}  ${model}  incl=${incl}  $(date '+%H:%M:%S')"
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
echo "${EXPERIMENT} -- OLD (8 base momentum proxies) vs NEW (17 literature momentum), momentum-only"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    old="${PREFIX}_${mn}_old"; new="${PREFIX}_${mn}_new"
    oj="results/metrics/crypto/${mn}/${old}.json"; nj="results/metrics/crypto/${mn}/${new}.json"
    echo "=== ${mn}: old(8) vs new(17) ==="
    if [ ! -s "$oj" ] || [ ! -s "$nj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$oj" "$nj" <<'PY' || echo "  (metrics not readable)"
import json, sys
o = json.load(open(sys.argv[1])); n = json.load(open(sys.argv[2]))
def row(tag, d):
    c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    print("  %-5s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"
          % (tag, cm, mm, 0.1*cm+mm, c.get("sharpe") or 0.0, d.get("n_periods")))
row("old", o); row("new", n)
PY
    # paired delta: new - old (baseline old, arm new)
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${old}" --arms "${new}" 2>/dev/null \
        || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_old ${PREFIX}_${mn}_new"
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
