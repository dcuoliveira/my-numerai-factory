#!/usr/bin/env bash
#
# Experiment E70: does the engineered allfeat-style block help on top of
# momswap-macro? Standalone models tuned from scratch on v2.0-momswap-macro (49f)
# VS the same plus tenc/onehot/rollsel/spread/interact (925f).
#
# E70 is the reference handle; every artefact is e70_*.
#
# THE QUESTION
#   v2.0-momswap-macro = 14 base + 17 momentum + 18 macro (49f). E70 adds the
#   allfeat_pp recipe retargeted to this set (src/data_mani/build_momswap_eng.py ->
#   crypto/v2.0-momswap-macro-eng, 925f):
#     tenc     1    causal lag-20 mean-target encoding of the symbol
#     onehot   752  one identity dummy per symbol (>=130 dates)
#     rollsel  87   rolling means {5,20,60} of 17 momentum + 10 macro-return + 2 vol
#                   (RSI is ABSENT in momswap, dropped from the ask)
#     spread   12   term-structure differences within each momentum / macro family
#     interact 24   accel + adapted products (feature_momentum gone -> mom_ret_28d
#                   stands in) + 12 momentum x macro-momentum products (crypto
#                   trend GATED by the macro regime, a per-symbol cross term)
#     base     crypto/v2.0-momswap-macro       49f
#     eng      crypto/v2.0-momswap-macro-eng   925f  (+876)
#
# WHAT THIS RUNS  10 arms = 5 models x {base, eng}, tuned from scratch, uniform
#   n_iter=50, same cadence as E64-E69 (rolling 130/130, embargo auto=20, 5 folds,
#   val_frac 0.3, seed 42, target_binned_return_20, metric corr). catboost_ktz's
#   12-cell grid clamps 50 -> 12. Paired within model: eng vs base.
#
# WHAT WOULD COUNT AS A RESULT  eng beating base on numerai_corr at paired
#   Newey-West |t|>=2 (lag 20). The onehot/tenc give symbol identity, the
#   momentum x macro products give macro-regime gating -- both work through
#   INTERACTIONS a tree/MLP can split on, less so a linear model. Prior: the
#   allfeat_pp block historically did not beat the raw set on this panel (11 blocks
#   tried, none clears |t|>=2), so a null is likely and is itself the finding.
#
# RUNTIME  wider (925f, 752 onehot) so heavier per fit, ~5-8h for ten, serial.
#   64-core box, runs alongside anything else. Background it; mails start + done.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 = 3.8
#   no-pyarrow sandbox; preflight refuses it).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e70_eng_momswap_macro_crypto.sh
#   DRY_RUN=1 / E70_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_eng lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E70}"
E70_WINDOW="${E70_WINDOW:-130}"
E70_REFIT="${E70_REFIT:-130}"
E70_EMBARGO="${E70_EMBARGO:-auto}"
E70_N_ITER="${E70_N_ITER:-50}"
E70_N_SPLITS="${E70_N_SPLITS:-5}"
E70_VAL_FRAC="${E70_VAL_FRAC:-0.3}"
E70_SEED="${E70_SEED:-42}"
E70_TARGET="${E70_TARGET:-target_binned_return_20}"
E70_ERA_STEP="${E70_ERA_STEP:-1}"
E70_BASE_DATA="${E70_BASE_DATA:-crypto/v2.0-momswap-macro}"
E70_ENG_DATA="${E70_ENG_DATA:-crypto/v2.0-momswap-macro-eng}"
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E70_WINDOW}" in ""|0|[0-9]) echo "E70_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E70_REFIT}"  in ""|0|[0-9]) echo "E70_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E70_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E70_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e70"; SCOUT=0
if [ "${E70_SCOUT:-0}" = "1" ]; then
    PREFIX="e70scout"; SCOUT=1; E70_ERA_STEP=6; E70_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E70_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); SJOBS+=("$4"); NJOBS+=("$5"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base" "${mn}" "${E70_BASE_DATA}" "${sj}" "${nj}"
    aadd "${mn}_eng"  "${mn}" "${E70_ENG_DATA}"  "${sj}" "${nj}"
done
NITER_USE="${E70_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: base=49, eng=925 with the engineered blocks present + label + meta,
# and the 5 wrappers construct.
"$PYTHON" - "${E70_BASE_DATA}" "${E70_ENG_DATA}" "${E70_TARGET}" "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" <<'PY' || exit 1
import os, re, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

base_v, eng_v, target = sys.argv[1:4]
rest = sys.argv[4:]; c = rest.index("--")
models = rest[:c]; cells = [int(x) for x in rest[c + 1:]]
bad = []
for v, want in [(base_v, 49), (eng_v, 925)]:
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
eng = pq.ParquetFile("data/%s/train.parquet" % eng_v).schema.names
checks = {
    "tenc": sum(cc == "feature_sym_tenc" for cc in eng) == 1,
    "onehot>=700": sum(cc.startswith("feature_sym_") and cc != "feature_sym_tenc" for cc in eng) >= 700,
    "rollsel==87": sum(bool(re.search(r"_rm(5|20|60)$", cc)) for cc in eng) == 87,
    "spread==12": sum(cc.startswith("feature_spread_") for cc in eng) == 12,
    "mom_x_macro==12": sum(bool(re.search(r"^feature_mom.*_x_(dxy|gold|snp|silver)", cc)) for cc in eng) == 12,
}
for name, ok in checks.items():
    if not ok:
        bad.append("eng block check failed: %s" % name)
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
print("preflight OK: base=49, eng=925 (tenc+onehot+rollsel87+spread12+momXmacro12) "
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

  arms      ${total}: 5 models x {base v2.0-momswap-macro (49f), eng v2.0-momswap-macro-eng (925f)}, paired
  block     tenc(1) + onehot(752) + rollsel(87) + spread(12) + interact(24, incl. 12 momentumXmacro)
  note      onehot/tenc = symbol identity; momXmacro = macro-regime gating -- both help via INTERACTIONS
  cadence   rolling ${E70_WINDOW}/${E70_REFIT}, embargo ${E70_EMBARGO}, ${E70_N_SPLITS} folds, seed ${E70_SEED}, n_iter=${E70_N_ITER}

  runner    experiments/e70_eng_momswap_macro_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: engineered block, momswap-macro base(49) vs eng(925)  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E70_WINDOW}/${E70_REFIT}, embargo ${E70_EMBARGO}, ${E70_N_SPLITS} folds, seed ${E70_SEED}, n_iter=${NITER_USE}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E70_WINDOW}" --refit-every "${E70_REFIT}"
         --embargo "${E70_EMBARGO}" --n-splits "${E70_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E70_VAL_FRAC}" --era-step "${E70_ERA_STEP}" --seed "${E70_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E70_TARGET}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-30s\n' "$((i+1))" "$total" "$run_name" "$model" "$dsver"
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
echo "${EXPERIMENT} -- engineered block: momswap-macro base(49) vs eng(925)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; eng="${PREFIX}_${mn}_eng"
    bj="results/metrics/crypto/${mn}/${base}.json"; ej="results/metrics/crypto/${mn}/${eng}.json"
    echo "=== ${mn}: base vs eng ==="
    if [ ! -s "$bj" ] || [ ! -s "$ej" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$ej" <<'PY' || echo "  (metrics not readable)"
import json, sys
b=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2]))
def row(t,d):
    c=d.get("corr") or {}; m=d.get("mmc") or {}; cm=c.get("mean") or 0.0; mm=m.get("mean") or 0.0
    print("  %-5s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"%(t,cm,mm,0.1*cm+mm,c.get("sharpe") or 0.0,d.get("n_periods")))
row("base",b); row("eng",e)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${eng}" 2>/dev/null || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_eng"
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
