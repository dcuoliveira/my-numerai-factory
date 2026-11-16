#!/usr/bin/env bash
#
# Experiment E71: isolate the INTERACTIONS block (no onehot/tenc/rollsel/spread) on
# top of momswap-macro. The E70 follow-up: E70 dumped the whole 925f allfeat-style
# block and it was null-to-negative (the 752 onehot dummies crushed ridge/MLP on
# CORR; the boosters showed only a slight, non-significant +MMC). E71 strips it to
# just the 24 interaction features -- so the boosters' hint, and especially the
# momentum x macro-momentum regime gating, is tested WITHOUT the dummy noise.
#
# E71 is the reference handle; every artefact is e71_*.
#
# THE DATASET  crypto/v2.0-momswap-macro-xinteractions (73f) =
#   v2.0-momswap-macro (49f) + the 24-column interact block (build_momswap_eng.py
#   --blocks interact): 6 accel (close/volume/market_cap EWA-SMA x 20d/60d), 6
#   adapted products (feature_mom_ret_28d stands in for the removed feature_momentum;
#   volume x volatility; market_cap x mom_ret_28d), and 12 momentum x macro-momentum
#   products (crypto momentum GATED by the macro regime -- a per-symbol cross term).
#     base   crypto/v2.0-momswap-macro                49f
#     xint   crypto/v2.0-momswap-macro-xinteractions  73f  (+24)
#
# WHAT THIS RUNS  10 arms = 5 models x {base, xint}, tuned from scratch, uniform
#   n_iter=50, same cadence (rolling 130/130, embargo auto=20, 5 folds, val_frac
#   0.3, seed 42, target_binned_return_20, metric corr). Paired within model.
#
# WHAT WOULD COUNT AS A RESULT  xint beating base on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20). The momentum x macro products vary per symbol (a
#   monotone gate on momentum flipped/scaled by the macro state), so unlike the
#   day-level macro block they CAN move a within-date rank -- ridge is no longer a
#   pure negative control here. Prior: E70's boosters hinted +MMC ~0.006 at t<1, so
#   a small effect at most; a null with a tighter interval is the likely finding.
#
# RUNTIME  small (73f), ~2-4h for ten, serial. Background it; mails start + done.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 refuses).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e71_xinteractions_momswap_macro_crypto.sh
#   DRY_RUN=1 / E71_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_xint lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E71}"
E71_WINDOW="${E71_WINDOW:-130}"
E71_REFIT="${E71_REFIT:-130}"
E71_EMBARGO="${E71_EMBARGO:-auto}"
E71_N_ITER="${E71_N_ITER:-50}"
E71_N_SPLITS="${E71_N_SPLITS:-5}"
E71_VAL_FRAC="${E71_VAL_FRAC:-0.3}"
E71_SEED="${E71_SEED:-42}"
E71_TARGET="${E71_TARGET:-target_binned_return_20}"
E71_ERA_STEP="${E71_ERA_STEP:-1}"
E71_BASE_DATA="${E71_BASE_DATA:-crypto/v2.0-momswap-macro}"
E71_XINT_DATA="${E71_XINT_DATA:-crypto/v2.0-momswap-macro-xinteractions}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E71_WINDOW}" in ""|0|[0-9]) echo "E71_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E71_REFIT}"  in ""|0|[0-9]) echo "E71_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E71_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E71_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e71"; SCOUT=0
if [ "${E71_SCOUT:-0}" = "1" ]; then
    PREFIX="e71scout"; SCOUT=1; E71_ERA_STEP=6; E71_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E71_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base" "${mn}" "${E71_BASE_DATA}" "${sj}" "${nj}"
    aadd "${mn}_xint" "${mn}" "${E71_XINT_DATA}" "${sj}" "${nj}"
done
NITER_USE="${E71_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=49, xint=73 with the interact block present (>=12 momentum x
# macro products, >=6 accel) + label + meta, and the 5 wrappers construct.
"$PYTHON" - "${E71_BASE_DATA}" "${E71_XINT_DATA}" "${E71_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, re, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, xint_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 49), (xint_v, 73)]:
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
xi = pq.ParquetFile("data/%s/train.parquet" % xint_v).schema.names
n_momxmacro = sum(bool(re.search(r"^feature_mom.*_x_(dxy|gold|snp|silver)", cc)) for cc in xi)
n_accel = sum("_accel_" in cc for cc in xi)
if n_momxmacro != 12:
    bad.append("xint has %d momentumXmacro products, expected 12" % n_momxmacro)
if n_accel != 6:
    bad.append("xint has %d accel features, expected 6" % n_accel)
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
print("preflight OK: base=49, xint=73 (24 interact incl. 12 momentumXmacro + 6 accel) "
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

  arms      ${total}: 5 models x {base v2.0-momswap-macro (49f), xint (73f)}, paired
  block     24 interactions ONLY (6 accel + 6 products + 12 momentumXmacro) -- the E70 follow-up, no onehot/tenc noise
  cadence   rolling ${E71_WINDOW}/${E71_REFIT}, embargo ${E71_EMBARGO}, ${E71_N_SPLITS} folds, seed ${E71_SEED}, n_iter=${E71_N_ITER}

  runner    experiments/e71_xinteractions_momswap_macro_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: interactions-only, momswap-macro base(49) vs xint(73)  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E71_WINDOW}/${E71_REFIT}, embargo ${E71_EMBARGO}, ${E71_N_SPLITS} folds, seed ${E71_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E71_WINDOW}" --refit-every "${E71_REFIT}"
         --embargo "${E71_EMBARGO}" --n-splits "${E71_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E71_VAL_FRAC}" --era-step "${E71_ERA_STEP}" --seed "${E71_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E71_TARGET}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-40s\n' "$((i+1))" "$total" "$run_name" "$model" "$dsver"
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
echo "${EXPERIMENT} -- interactions-only: momswap-macro base(49) vs xint(73)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; xint="${PREFIX}_${mn}_xint"
    bj="results/metrics/crypto/${mn}/${base}.json"; xj="results/metrics/crypto/${mn}/${xint}.json"
    echo "=== ${mn}: base vs xint ==="
    if [ ! -s "$bj" ] || [ ! -s "$xj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$xj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); x=json.load(open(sys.argv[2]))
def row(t,d):
    c=d.get("corr") or {}; m=d.get("mmc") or {}; cm=c.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-5s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.0,d.get("n_periods")))
row("base",b); row("xint",x)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${xint}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_xint"
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
