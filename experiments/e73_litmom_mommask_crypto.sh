#!/usr/bin/env bash
#
# Experiment E73: litmom with the BASE features masked to neutral on
# momentum-missing rows. crypto/v2.0-litmom-mommask, compared THREE-WAY against
# crypto/v2.0 (raw 22) and crypto/v2.0-litmom (22+17) -- the latter two reused
# from E65 (same protocol), so E73 only runs the 5 new mommask arms.
#
# E73 is the reference handle; every artefact is e73_*.
#
# WHAT mommask IS (verified on the data, not the name)
#   Same 39 columns as litmom; the 17 momentum columns are IDENTICAL. The change
#   is that the 22 BASE features are set to the neutral 0.5 in ~17% of cells,
#   CONCENTRATED on rows where the momentum block is missing:
#     P(base masked | momentum missing on row) = 0.996
#     P(base masked | momentum present on row) = 0.263
#   ~11 of the 22 base features masked per masked row. Net effect: on the ~22% of
#   symbols/rows that have no spot momentum (and therefore already lean entirely on
#   the base features), roughly half the base features are ALSO hidden. Masking only
#   removes information (0.5 = neutral), so it is leakage-safe -- but it degrades the
#   exact rows that depend most on the base block. Read the result with that in mind.
#
# WHAT THIS RUNS  5 arms = 5 models x {mommask}, tuned from scratch, uniform
#   n_iter=50, EXACTLY E65's protocol (rolling 130/130, embargo auto=20, 5 folds,
#   val_frac 0.3, seed 42, target_binned_return_20, metric corr) so each mommask
#   arm pairs cleanly against E65's e65_<model>_base (v2.0) and e65_<model>_litmom.
#
# WHAT WOULD COUNT AS A RESULT  mommask beating litmom (or raw) at paired
#   Newey-West |t|>=2 (lag 20). Prior: masking base information should not HELP on
#   numerai_corr; a null-or-worse (esp. vs litmom) is the expected finding, and a
#   significant DROP would quantify how much the base block carries the no-spot names.
#
# RUNTIME  ~1.5-3h for five, serial. Background it; mails start + done.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e73_litmom_mommask_crypto.sh
#   DRY_RUN=1 / E73_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge lgb_ktz"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E73}"
E73_WINDOW="${E73_WINDOW:-130}"
E73_REFIT="${E73_REFIT:-130}"
E73_EMBARGO="${E73_EMBARGO:-auto}"
E73_N_ITER="${E73_N_ITER:-50}"
E73_N_SPLITS="${E73_N_SPLITS:-5}"
E73_VAL_FRAC="${E73_VAL_FRAC:-0.3}"
E73_SEED="${E73_SEED:-42}"
E73_TARGET="${E73_TARGET:-target_binned_return_20}"
E73_ERA_STEP="${E73_ERA_STEP:-1}"
E73_DATA="${E73_DATA:-crypto/v2.0-litmom-mommask}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E73_WINDOW}" in ""|0|[0-9]) echo "E73_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E73_REFIT}"  in ""|0|[0-9]) echo "E73_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E73_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E73_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e73"; SCOUT=0
if [ "${E73_SCOUT:-0}" = "1" ]; then
    PREFIX="e73scout"; SCOUT=1; E73_ERA_STEP=6; E73_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E73_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

# Preflight: mommask present with 39 features + label + meta; base v2.0 and litmom
# results exist to compare against; the 5 wrappers construct.
"$PYTHON" - "${E73_DATA}" "${E73_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
data, target = sys.argv[1:3]
rest = sys.argv[3:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
p = "data/%s/train.parquet" % data
try:
    names = pq.ParquetFile(p).schema.names
except OSError:
    sys.exit("preflight FAILED: %s MISSING" % data)
got = sum(cc.startswith("feature_") for cc in names)
if got != 39:
    bad.append("%s has %d features, expected 39" % (data, got))
if target not in names:
    bad.append("%s lacks %s" % (data, target))
if not os.path.exists("data/%s/historical_meta_models.parquet" % data):
    bad.append("%s lacks historical_meta_models.parquet" % data)
for name, want_cells in zip(models, cells):
    try:
        w = build_wrapper(name, "regression", seed=42, n_jobs=1)
        got_cells = len(list(ParameterGrid(w.param_grid)))
    except Exception as e:
        bad.append("%s construct/grid failed: %s" % (name, e)); continue
    if got_cells != want_cells:
        bad.append("%s grid has %d cells, expected %d" % (name, got_cells, want_cells))
# comparison arms from E65 (reused)
for mdl in models:
    for ref in ("base", "litmom"):
        if not os.path.exists("results/metrics/crypto/%s/e65_%s_%s.json" % (mdl, mdl, ref)):
            bad.append("missing E65 comparison arm e65_%s_%s" % (mdl, ref))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %s (39f) with label+meta; E65 v2.0+litmom arms present to reuse; "
      "5 wrappers construct." % data)
PY

SEL=(); SEL_SJOBS=(); SEL_NJOBS=(); picked=0
for i in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${mn} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${mn} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
    SEL+=("$mn"); SEL_SJOBS+=("${MODEL_SJOBS[$i]}"); SEL_NJOBS+=("${MODEL_NJOBS[$i]}")
done
total="${#SEL[@]}"
[ "$total" -eq 0 ] && { echo "no models selected" >&2; exit 1; }
NITER_USE="${E73_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4
ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; send_text(sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: 5 models on ${E73_DATA} (litmom w/ base masked to 0.5 on momentum-missing rows)
  compare   3-way vs v2.0 (e65_*_base) and litmom (e65_*_litmom), reused from E65 (same protocol)
  cadence   rolling ${E73_WINDOW}/${E73_REFIT}, embargo ${E73_EMBARGO}, ${E73_N_SPLITS} folds, seed ${E73_SEED}, n_iter=${E73_N_ITER}

  runner    experiments/e73_litmom_mommask_crypto.sh
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; artefacts ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: litmom-mommask (${total} arms) | 3-way vs v2.0 + litmom (E65)"
echo "cadence rolling ${E73_WINDOW}/${E73_REFIT}, embargo ${E73_EMBARGO}, ${E73_N_SPLITS} folds, seed ${E73_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL[@]}"; do
    model="${SEL[$i]}"; sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${model}_mommask"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E73_WINDOW}" --refit-every "${E73_REFIT}"
         --embargo "${E73_EMBARGO}" --n-splits "${E73_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E73_VAL_FRAC}" --era-step "${E73_ERA_STEP}" --seed "${E73_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${E73_DATA}" --target "${E73_TARGET}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-22s %-14s %s\n' "$((i+1))" "$total" "$run_name" "$model" "$E73_DATA"
        printf '       %s\n' "${cmd[*]}"; continue
    fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run_name}.json" ]; then
        echo "[$((i+1))/$total] ${run_name} already on disk -- skipping"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${run_name}  ${model}  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${run_name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m$((el%60))s -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
PAIRED="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
{
echo "${EXPERIMENT} -- litmom-mommask, 3-way vs v2.0 (raw) and litmom"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for mn in "${SEL[@]}"; do
    b="e65_${mn}_base"; l="e65_${mn}_litmom"; k="${PREFIX}_${mn}_mommask"
    echo "=== ${mn}: v2.0 vs litmom vs mommask ==="
    "$PYTHON" - "results/metrics/crypto/${mn}/${b}.json" "results/metrics/crypto/${mn}/${l}.json" "results/metrics/crypto/${mn}/${k}.json" <<'PY' || echo "  (metrics not readable)"
import json, sys
def row(t,p):
    try: d=json.load(open(p))
    except Exception: print("  %-8s (missing)"%t); return
    c=d.get("corr") or {}; m=d.get("mmc") or {}; cm=c.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-8s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.0,d.get("n_periods")))
row("v2.0",sys.argv[1]); row("litmom",sys.argv[2]); row("mommask",sys.argv[3])
PY
    echo "  -- paired: mommask vs litmom --"
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${l}" --arms "${k}" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (unavailable)"
    echo "  -- paired: mommask vs v2.0 --"
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${b}" --arms "${k}" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (unavailable)"
    echo
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
