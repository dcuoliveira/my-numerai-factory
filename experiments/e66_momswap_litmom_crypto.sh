#!/usr/bin/env bash
#
# Experiment E66: SWAP the base momentum proxies for the literature momentum block.
#
# E66 is the reference handle; every artefact is e66_*.
#
# THE QUESTION
#   E64/E65 found that ADDING litmom to the full 22 is null-to-negative on
#   numerai_corr, and the reason is redundancy: the new momentum features load
#   0.5-0.81 on the base's OWN momentum proxies -- feature_momentum, RSI, Sharpe
#   and Bollinger (each 20d/60d, 8 columns). So E66 asks the substitution question
#   instead of the addition one: REMOVE those 8 base momentum proxies and REPLACE
#   them with the 17 literature momentum features. Does the literature block do the
#   momentum job BETTER than the base's crude proxies?
#
# THE TWO ARMS (paired within model)
#   base   crypto/v2.0                     the raw 22 (the standing baseline)
#   swap   crypto/v2.0-litmom, minus the   14 non-momentum base + 17 literature
#          8 base momentum proxies         momentum = 31 features
#
#   The 8 dropped (feature-exclude regex):
#     feature_momentum_{20,60}d
#     feature_relative_strength_index_{20,60}d
#     feature_sharpe_ratio_{20,60}d
#     feature_bollinger_{20,60}d
#   Kept on the swap arm: close/market_cap/volatility/volume (14) + all feature_mom_ (17).
#   NOTE this confounds "removed the native momentum" with "added the literature
#   momentum" -- that is inherent to a swap. It answers the deployment question
#   "is the literature-momentum feature set a better 31 than the raw 22", not "do
#   the new features add on top of everything" (that was E64/E65).
#
# WHAT THIS RUNS  10 arms = 5 models x {base, swap}, uniform n_iter=50, same
#   cadence as E64/E65 (rolling 130/130, embargo auto=20, 5 folds, val_frac 0.3,
#   seed 42, target_binned_return_20, metric corr). catboost_ktz's 12-cell grid
#   clamps 50 -> 12 (a full sweep). base pairs against swap on the SAME periods
#   (v2.0 and v2.0-litmom share rows/dates).
#
# WHAT WOULD COUNT AS A RESULT  swap beating the raw-22 base on numerai_corr at
#   paired Newey-West |t|>=2 (lag 20). Prior: the new features are mostly
#   redundant and reversal-signed (E63 EDA; E65 was null-to-negative), so a null
#   or a loss is likely -- but this isolates whether the literature block is at
#   least a viable substitute for the native proxies.
#
# RUNTIME  ~4-6h for ten, serial. Background it; mails start + completion.
# INTERPRETER  PYTHON=/home/danielco/miniconda3/bin/python3 (bare python3 is the
#   3.8 no-pyarrow sandbox; preflight refuses it).
#
# Usage:
#   PYTHON=/home/danielco/miniconda3/bin/python3 bash experiments/e66_momswap_litmom_crypto.sh
#   DRY_RUN=1 / E66_SCOUT=1 / NO_MAIL=1  |  ONLY="ridge_swap lgb_ktz_base"  SKIP=...  SHARD=1 OF=2
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-python3}"
EXPERIMENT="${EXPERIMENT:-E66}"
E66_WINDOW="${E66_WINDOW:-130}"
E66_REFIT="${E66_REFIT:-130}"
E66_EMBARGO="${E66_EMBARGO:-auto}"
E66_N_ITER="${E66_N_ITER:-50}"
E66_N_SPLITS="${E66_N_SPLITS:-5}"
E66_VAL_FRAC="${E66_VAL_FRAC:-0.3}"
E66_SEED="${E66_SEED:-42}"
E66_TARGET="${E66_TARGET:-target_binned_return_20}"
E66_ERA_STEP="${E66_ERA_STEP:-1}"
# The base momentum proxies to drop from the swap arm. Single regex, alternation
# with '|' -- kept in ONE variable and passed as ONE quoted arg (never split).
MOMPROX_RX='^feature_(momentum|relative_strength_index|sharpe_ratio|bollinger)_[0-9]+d$'
SHARD="${SHARD:-1}"; OF="${OF:-1}"

case "${E66_WINDOW}" in ""|0|[0-9]) echo "E66_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E66_REFIT}"  in ""|0|[0-9]) echo "E66_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac
[ "${E66_N_SPLITS}" -ge 2 ] 2>/dev/null || { echo "E66_N_SPLITS must be >= 2" >&2; exit 1; }

PREFIX="e66"
SCOUT=0
if [ "${E66_SCOUT:-0}" = "1" ]; then
    PREFIX="e66scout"; SCOUT=1; E66_ERA_STEP=6; E66_N_SPLITS=2
    echo "SCOUT MODE: era-step 6, tiny search, ${E66_N_SPLITS}-fold -- wiring/timing only, NOT comparable."
fi

# Arm table. Parallel arrays; a per-arm EXCL carries the feature-exclude regex
# (empty on the base arm, MOMPROX_RX on the swap arm).
MODEL_NAMES=(ridge        lgb_ktz  catboost_ktz  rf_ktz  mlp_ktz)
MODEL_SJOBS=(8            4        1             8       8)
MODEL_NJOBS=(1            2        6             2       1)
MODEL_CELLS=(100          432      12            60      120)

ARM=(); MODEL=(); DATA=(); EXCL=(); SJOBS=(); NJOBS=()
aadd() { ARM+=("$1"); MODEL+=("$2"); DATA+=("$3"); EXCL+=("$4"); SJOBS+=("$5"); NJOBS+=("$6"); }
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"; sj="${MODEL_SJOBS[$m]}"; nj="${MODEL_NJOBS[$m]}"
    aadd "${mn}_base" "${mn}" "crypto/v2.0"        ""             "${sj}" "${nj}"
    aadd "${mn}_swap" "${mn}" "crypto/v2.0-litmom" "${MOMPROX_RX}" "${sj}" "${nj}"
done
NITER_USE="${E66_N_ITER}"; [ "${SCOUT}" = "1" ] && NITER_USE=4

# Preflight: datasets + meta present; base=22; and the swap regex drops EXACTLY
# the 8 named proxies from litmom, leaving 31 (incl. all 17 feature_mom_). Gate
# on that -- a silent regex drift would score the wrong feature set.
"$PYTHON" - "${MOMPROX_RX}" -- "${MODEL_NAMES[@]}" -- "${MODEL_CELLS[@]}" -- "${E66_TARGET}" <<'PY' || exit 1
import os, re, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

a = sys.argv[1:]
rx = re.compile(a[0])
c1 = a.index("--"); c2 = a.index("--", c1 + 1); c3 = a.index("--", c2 + 1)
models = a[c1 + 1:c2]; cells = [int(x) for x in a[c2 + 1:c3]]; target = a[c3 + 1]
bad = []

for v, want in [("crypto/v2.0", 22), ("crypto/v2.0-litmom", None)]:
    p = "data/%s/train.parquet" % v
    try:
        names = pq.ParquetFile(p).schema.names
    except OSError:
        bad.append("%s MISSING" % v); continue
    got = sum(c.startswith("feature_") for c in names)
    if want is not None and got != want:
        bad.append("%s has %d features, expected %d" % (v, got, want))
    if target not in names:
        bad.append("%s lacks %s" % (v, target))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % v):
        bad.append("%s lacks historical_meta_models.parquet" % v)

# The swap arm's feature set, reproduced here.
lit = [c for c in pq.ParquetFile("data/crypto/v2.0-litmom/train.parquet").schema.names
       if c.startswith("feature_")]
dropped = sorted(c for c in lit if rx.search(c))
kept = [c for c in lit if not rx.search(c)]
mom = [c for c in kept if c.startswith("feature_mom_")]
want_drop = sorted([
    "feature_momentum_20d", "feature_momentum_60d",
    "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
    "feature_sharpe_ratio_20d", "feature_sharpe_ratio_60d",
    "feature_bollinger_20d", "feature_bollinger_60d"])
if dropped != want_drop:
    bad.append("swap regex drops %s, expected exactly the 8 base momentum proxies" % dropped)
if len(kept) != 31:
    bad.append("swap arm keeps %d features, expected 31 (14 base + 17 mom)" % len(kept))
if len(mom) != 17:
    bad.append("swap arm keeps %d feature_mom_, expected all 17" % len(mom))

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
print("preflight OK: base=22; swap drops the 8 base momentum proxies and keeps 31 "
      "(14 non-momentum base + all 17 literature momentum); 5 wrappers construct.")
PY

# Select work list.
SEL_ARM=(); SEL_MODEL=(); SEL_DATA=(); SEL_EXCL=(); SEL_SJOBS=(); SEL_NJOBS=(); picked=0
for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue; fi
    SEL_ARM+=("$arm"); SEL_MODEL+=("${MODEL[$i]}"); SEL_DATA+=("${DATA[$i]}")
    SEL_EXCL+=("${EXCL[$i]}"); SEL_SJOBS+=("${SJOBS[$i]}"); SEL_NJOBS+=("${NJOBS[$i]}")
done
total="${#SEL_ARM[@]}"
[ "$total" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    "$PYTHON" -c "import sys; from src.live.notify import send_text; \
send_text(sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}: 5 models x {base v2.0 (22f), swap (31f)}, paired
  swap      drop 8 base momentum proxies (momentum/rsi/sharpe/bollinger 20d+60d)
            from v2.0-litmom, keep 14 base + 17 literature momentum
  question  does the literature momentum block beat the base's own momentum proxies?
  cadence   rolling ${E66_WINDOW}/${E66_REFIT}, embargo ${E66_EMBARGO}, ${E66_N_SPLITS} folds, seed ${E66_SEED}, n_iter=${E66_N_ITER}
  note      catboost_ktz (12 cells) clamps n_iter to a full 12-cell sweep

  runner    experiments/e66_momswap_litmom_crypto.sh
  results   results/metrics/crypto/<model>/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: momentum-proxy SWAP, base(22) vs swap(31)  |  ${total} arm(s)  shard ${SHARD}/${OF}"
echo "cadence rolling ${E66_WINDOW}/${E66_REFIT}, embargo ${E66_EMBARGO}, ${E66_N_SPLITS} folds, seed ${E66_SEED}, n_iter=${NITER_USE}, target ${E66_TARGET}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  python=${PYTHON}"
echo "=============================================================="

for i in "${!SEL_ARM[@]}"; do
    arm="${SEL_ARM[$i]}"; model="${SEL_MODEL[$i]}"; dsver="${SEL_DATA[$i]}"
    excl="${SEL_EXCL[$i]}"; sjobs="${SEL_SJOBS[$i]}"; njobs="${SEL_NJOBS[$i]}"
    run_name="${PREFIX}_${arm}"
    cmd=("$PYTHON" -u -m src.run_forecast crypto "${model}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E66_WINDOW}" --refit-every "${E66_REFIT}"
         --embargo "${E66_EMBARGO}" --n-splits "${E66_N_SPLITS}" --n-iter "${NITER_USE}"
         --val-frac "${E66_VAL_FRAC}" --era-step "${E66_ERA_STEP}" --seed "${E66_SEED}"
         --search-jobs "${sjobs}" --n-jobs "${njobs}"
         --data-version "${dsver}" --target "${E66_TARGET}" --run-name "${run_name}")
    [ -n "${excl}" ] && cmd+=(--feature-exclude "${excl}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s %-20s excl=%s\n' "$((i+1))" "$total" "$run_name" "$model" "$dsver" "${excl:-(none)}"
        printf '       %s\n' "${cmd[*]}"; continue
    fi
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${model}/${run_name}.json" ]; then
        echo "[$((i+1))/$total] ${run_name} already on disk -- skipping (resume-safe)"; continue
    fi
    ran=$((ran+1))
    echo; echo "[$((i+1))/${total}] ${run_name}  ${model}  ${dsver}  excl=${excl:-(none)}  $(date '+%H:%M:%S')"
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
echo "${EXPERIMENT} -- momentum-proxy SWAP: base(22) vs swap(14 base + 17 literature mom = 31)"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    base="${PREFIX}_${mn}_base"; sw="${PREFIX}_${mn}_swap"
    bj="results/metrics/crypto/${mn}/${base}.json"; sj="results/metrics/crypto/${mn}/${sw}.json"
    echo "=== ${mn}: base(22) vs swap(31) ==="
    if [ ! -s "$bj" ] || [ ! -s "$sj" ]; then echo "  (one/both arms not on disk)"; echo; continue; fi
    "$PYTHON" - "$bj" "$sj" <<'PY' || echo "  (metrics not readable)"
import json, sys
b = json.load(open(sys.argv[1])); l = json.load(open(sys.argv[2]))
def row(tag, d):
    c = d.get("corr") or {}; m = d.get("mmc") or {}
    cm = c.get("mean") or 0.0; mm = m.get("mean") or 0.0
    print("  %-6s CORR %.5f  MMC %.5f  payout %.5f  c.sharpe %.3f  n=%s"
          % (tag, cm, mm, 0.1*cm+mm, c.get("sharpe") or 0.0, d.get("n_periods")))
row("base", b); row("swap", l)
PY
    "$PYTHON" "$PAIRED" --model "${mn}" --baseline "${base}" --arms "${sw}" 2>/dev/null \
        || echo "  (paired_delta unavailable)"
    echo
done
echo "FULL REPORT per model:"
for m in "${!MODEL_NAMES[@]}"; do
    mn="${MODEL_NAMES[$m]}"
    echo "  $PYTHON -m src.analysis.report_html --model ${mn} --runs ${PREFIX}_${mn}_base ${PREFIX}_${mn}_swap"
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
