#!/usr/bin/env bash
#
# Experiment E27 (ktz): LightGBM hyperparameter sensitivity, Kalfa-Timmermann-
# van der Zwan (2025) style
#
# E27 is the reference handle; every artefact is ktz_*.
#
# METHOD, from the paper. Run a FULL FACTORIAL grid, then for each single
# hyperparameter value marginalise over all the others to get the DISTRIBUTION
# of out-of-sample payout. That distribution -- not a single run's mean -- is
# what reveals corner cases, the safe range (a prior), and which hyperparameters
# matter. Analysis is src.analysis.hyper_sensitivity.
#
# THE GRID (315 = 7 x 5 x 3 x 3 cells, each a pinned single-config walk-forward):
#
#     num_leaves        {2, 4, 8, 16, 31, 63, 255}     complexity (max_depth=-1)
#     learning_rate     {0.001, 0.005, 0.02, 0.1, 0.3}
#     n_estimators      {2000, 8000, 30000}
#     colsample_bytree  {0.05, 0.2, 1.0}               per-tree feature diversity
#
#     FIXED: max_depth=-1  min_child_samples=100  subsample=0.8  subsample_freq=1
#            reg_lambda=10  deterministic=true  force_row_wise=true  seed=42
#
# Two grid choices came out of an adversarial design review and are not obvious:
#
#   * num_leaves=2 (a stump) IS the grid, because the paper's headline result is
#     that the shallowest tree "virtually guarantees" a good model. In leaf-wise
#     LightGBM a depth-1 tree is 2 leaves, so omitting 2 would omit exactly where
#     that result lives. It is also the cheapest cell.
#
#   * num_leaves=1023 is NOT in the grid. Verified against the data: a 6-month
#     rolling refit is ~33-39k rows, so at min_child_samples=100 with
#     subsample=0.8 the realizable-leaf ceiling is ~264-312. num_leaves=1023
#     cannot grow past that and ALIASES onto 255 -- a near-duplicate that would
#     have cost ~half the deep-end wall-clock and muddied the complexity
#     marginal. 255 (fully realizable, at the ceiling) is the deep/overfit
#     corner. Dropping 1023 leaves every retained num_leaves value realizable,
#     so the marginal is alias-free.
#
# max_depth=-1 IN EVERY CELL is load-bearing: with any max_depth>0 LightGBM
# silently caps num_leaves at 2^max_depth, which would corrupt the complexity
# axis with no error. The preflight asserts it.
#
# ---------------------------------------------------------------------------
# What differs from the paper, and is handled in ANALYSIS not hidden
# ---------------------------------------------------------------------------
#
#   * NO EARLY STOPPING. n_estimators is a real capacity dial, not a truncated
#     ceiling, so a raw learning_rate or n_estimators marginal is confounded by
#     total learning L = lr * n_estimators (spans 2 to 9000). The analyzer never
#     presents the raw lr or n_est marginal as "the effect of" that parameter;
#     it reports L, the L-marginal, and the equal-L collision pairs
#     [(0.005,8000)=(0.02,2000)=40; (0.02,30000)=(0.3,2000)=600] that test
#     whether the product is the sufficient statistic.
#   * LightGBM is leaf-wise; the paper's XGBoost is depth-wise. num_leaves (with
#     max_depth=-1) is the analogue of the paper's "depth". colsample_bytree is
#     per-TREE, not the paper's per-SPLIT F -- 0.05 (~1 feature/tree) is the F~=1
#     analogue but not the literal per-split result.
#   * Payout is compressed near 0 (not R2 running to -1,-2), so the paper's
#     dramatic bad-corner tails appear here as "tighter vs wider and lower rank",
#     not as a magnitude blow-up.
#
# ---------------------------------------------------------------------------
# Harness neutrality: a bridge control, REPORTED not gated
# ---------------------------------------------------------------------------
#
# BRIDGE=1 also runs the canonical booster config (the 4327-tree pin every prior
# lgb arm used) so its number can be compared to e13_raw, the same config on a
# recent panel. It is NOT hard-gated: the panel has drifted (510 -> 514 -> 515
# periods as data was appended), so demanding a bit-match against the old
# 0.149754 would be a false alarm. The report classifies it: reproduces e13_raw
# to ~1e-3 => harness neutral; a large gap => investigate before trusting the
# sweep. Pipeline determinism itself is covered by the test suite.
#
# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
#
# A cell is ~10s (num_leaves=2, few trees) to ~8min (255 x 30000). Expensive
# cells run LAST so the cheap marginals land first. At KTZ_POOL concurrent cells
# the whole grid is roughly 45-90 min on an idle 32-core box.
#
# Usage:
#   bash experiments/e27_hyperparam_sensitivity_crypto.sh
#   DRY_RUN=1 bash experiments/e27_hyperparam_sensitivity_crypto.sh
#   BRIDGE=1  bash experiments/e27_hyperparam_sensitivity_crypto.sh   # + control
#   KTZ_POOL=6 KTZ_JOBS=5 bash experiments/e27_hyperparam_sensitivity_crypto.sh
#   ONLY_SHALLOW=1 bash experiments/e27_hyperparam_sensitivity_crypto.sh  # nl<=16
#   NO_MAIL=1 bash experiments/e27_hyperparam_sensitivity_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E27}"
KTZ_DATA="${KTZ_DATA:-crypto/v2.0}"
KTZ_WINDOW="${KTZ_WINDOW:-6month}"
KTZ_REFIT="${KTZ_REFIT:-6month}"
KTZ_JOBS="${KTZ_JOBS:-6}"       # threads per cell
KTZ_POOL="${KTZ_POOL:-5}"       # concurrent cells
KTZ_SEED="${KTZ_SEED:-42}"
case "${KTZ_WINDOW}" in ""|0|[0-9]) echo "KTZ_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

# The swept axes and the fixed remainder. Kept as bash arrays so the cell list
# is generated here and every cell is greppable and recoverable from its name.
LEAVES=(2 4 8 16 31 63 255)
LRS=(0.001 0.005 0.02 0.1 0.3)
NESTS=(2000 8000 30000)
COLS=(0.05 0.2 1.0)
FIXED='"max_depth":-1,"min_child_samples":100,"subsample":0.8,"subsample_freq":1,"reg_lambda":10,"deterministic":true,"force_row_wise":true'

# lr / colsample -> filename-safe tokens (0.02 -> 0p020, 1.0 -> 1p00); lexical
# sort then equals numeric sort, so the results tree browses in grid order.
lrtok() { printf "0p%03d" "$(python3 -c "print(int(round($1*1000)))")"; }
cstok() { python3 -c "print(('%.2f'%$1).replace('.','p'))"; }

# --------------------------------------------------------------------------- #
# Preflight (assert before spending the sweep's wall-clock)
# --------------------------------------------------------------------------- #
python3 - "${KTZ_DATA}" <<'PY' || exit 1
import sys, pyarrow.parquet as pq
names = pq.ParquetFile("data/%s/train.parquet" % sys.argv[1]).schema.names
feats = [c for c in names if c.startswith("feature_")]
if len(feats) != 22:
    sys.exit("expected exactly 22 raw features in %s, found %d -- refusing to run"
             % (sys.argv[1], len(feats)))
if "target_binned_return_20" not in names:
    sys.exit("target_binned_return_20 missing from %s" % sys.argv[1])
import lightgbm, numpy
print("preflight OK: 22 raw features | lightgbm %s | numpy %s"
      % (lightgbm.__version__, numpy.__version__))
PY

# Build the cell list, cheap-first (ascending num_leaves, then n_estimators),
# so the shallow marginals finish before the expensive deep corners.
CELL_NAME=(); CELL_PARAMS=()
for nl in "${LEAVES[@]}"; do
  [ "${ONLY_SHALLOW:-0}" = "1" ] && [ "$nl" -gt 16 ] && continue
  for ne in "${NESTS[@]}"; do
    for lr in "${LRS[@]}"; do
      for cs in "${COLS[@]}"; do
        name=$(printf "ktz_nl%04d_lr%s_ne%05d_cs%s" "$nl" "$(lrtok "$lr")" "$ne" "$(cstok "$cs")")
        params="{\"num_leaves\":${nl},\"learning_rate\":${lr},\"n_estimators\":${ne},\"colsample_bytree\":${cs},${FIXED}}"
        CELL_NAME+=("$name"); CELL_PARAMS+=("$params")
      done
    done
  done
done
total="${#CELL_NAME[@]}"

if [ "${ONLY_SHALLOW:-0}" != "1" ] && [ "$total" -ne 315 ]; then
    echo "expected 315 cells, generated ${total} -- grid is wrong, refusing to run" >&2; exit 1
fi

run_cell() {  # $1 name  $2 params-json
    # Idempotent resume: a cell already on disk is skipped, so relaunching after
    # an interruption completes the grid rather than redoing it. KTZ_SKIP is a
    # space-separated set of cells to leave alone (e.g. ones still running as
    # orphans from a killed pool) so two processes never write the same file.
    if [ -f "results/metrics/crypto/lgb/$1.json" ]; then return 0; fi
    case " ${KTZ_SKIP:-} " in *" $1 "*) return 0;; esac
    python3 -u -m src.run_forecast crypto lgb --task regression --metric corr \
        --cv-scheme rolling --train-window "${KTZ_WINDOW}" --refit-every "${KTZ_REFIT}" \
        --embargo 20 --data-version "${KTZ_DATA}" \
        --fixed-params "$2" --n-splits 2 --n-jobs "${KTZ_JOBS}" --era-step 1 \
        --seed "${KTZ_SEED}" --run-name "$1" -q > "logs/$1.log" 2>&1
}

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "${EXPERIMENT}: ${total} cells, pool ${KTZ_POOL} x ${KTZ_JOBS} threads"
    for i in 0 1 2 $((total-2)) $((total-1)); do
        printf '  %-34s %s\n' "${CELL_NAME[$i]}" "${CELL_PARAMS[$i]}"
    done
    echo "  ... (${total} total)"
    exit 0
fi

started_all=$(date +%s)
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} (ktz hyperparameter sensitivity) started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  cells     ${total}  (num_leaves x learning_rate x n_estimators x colsample_bytree)
  method    Kalfa-Timmermann-van der Zwan factorial sensitivity: marginalise each
            hyperparameter over the rest; find corner cases, the safe range, and
            which hyperparameters actually move payout
  model     LightGBM regression, pinned per cell, raw 22, rolling 6m/6m, seed 42
  pool      ${KTZ_POOL} concurrent x ${KTZ_JOBS} threads
  expect    ~45-90 min

Where to look while it runs:
  runner    experiments/e27_hyperparam_sensitivity_crypto.sh
  progress  logs/ktz_*.log            (one per cell)
  results   results/metrics/crypto/lgb/ktz_*.json
  analysis  python3 -m src.analysis.hyper_sensitivity --prefix ktz_
  handle    this study is ${EXPERIMENT}; every artefact is ktz_*

A second mail follows on completion, with the sensitivity report and SUCCESS/FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
        "${EXPERIMENT}" "${START_MSG}" || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: ktz hyperparameter sensitivity | ${total} cells"
echo "pool ${KTZ_POOL} x ${KTZ_JOBS} threads | started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# Concurrency pool: keep KTZ_POOL cells in flight, reap with wait -n.
declare -A PID_NAME=(); done=0; failed=0
declare -a FAILED_CELLS=()
reap() {  # block for one job, record its status
    local pid rc
    if wait -n 2>/dev/null; then rc=0; else rc=$?; fi
    # wait -n does not tell us which pid; poll for a finished one.
    for pid in "${!PID_NAME[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$?
            local name="${PID_NAME[$pid]}"; unset "PID_NAME[$pid]"
            done=$((done+1))
            if [ "$rc" -ne 0 ] || [ ! -f "results/metrics/crypto/lgb/${name}.json" ]; then
                failed=$((failed+1)); FAILED_CELLS+=("$name")
                printf '[%3d/%3d] %-34s FAILED (rc %s)\n' "$done" "$total" "$name" "$rc"
            else
                printf '[%3d/%3d] %-34s ok\n' "$done" "$total" "$name"
            fi
            return
        fi
    done
}

for i in "${!CELL_NAME[@]}"; do
    while [ "${#PID_NAME[@]}" -ge "${KTZ_POOL}" ]; do reap; done
    name="${CELL_NAME[$i]}"
    if [ -f "results/metrics/crypto/lgb/${name}.json" ] || { case " ${KTZ_SKIP:-} " in *" ${name} "*) true;; *) false;; esac; }; then
        done=$((done+1)); continue
    fi
    run_cell "${name}" "${CELL_PARAMS[$i]}" &
    PID_NAME[$!]="${name}"
done
while [ "${#PID_NAME[@]}" -gt 0 ]; do reap; done

# Optional bridge control -- canonical config, reported not gated.
if [ "${BRIDGE:-0}" = "1" ]; then
    echo "running bridge control (canonical 4327-tree config) ..."
    run_cell "ktz_bridge_canonical" \
      '{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}' \
      && echo "  bridge control done" || echo "  bridge control FAILED"
fi

REPORT="logs/ktz_report.txt"
{
echo "${EXPERIMENT} -- ktz hyperparameter sensitivity (LightGBM, raw 22)"
echo "  results  results/metrics/crypto/lgb/ktz_*.json"
echo "  runner   experiments/e27_hyperparam_sensitivity_crypto.sh"
echo "=============================================================="
echo "ran ${done} of ${total} cells in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed cells: %s\n' "${FAILED_CELLS[*]}"
echo "=============================================================="
echo
python3 -m src.analysis.hyper_sensitivity --prefix ktz_ --metric payout 2>&1
echo
# bridge classification
if [ -f results/metrics/crypto/lgb/ktz_bridge_canonical.json ]; then
python3 - <<'PY'
import json, pandas as pd, os
b=json.load(open("results/metrics/crypto/lgb/ktz_bridge_canonical.json"))
bp=len(pd.read_csv("results/per_period/crypto/lgb/ktz_bridge_canonical.csv"))
ref=json.load(open("results/metrics/crypto/lgb/e13_raw.json"))
rp=len(pd.read_csv("results/per_period/crypto/lgb/e13_raw.csv"))
gap=abs(b["corr"]["mean"]-ref["corr"]["mean"])
verdict=("HARNESS NEUTRAL (gap %.2e, within panel-drift tolerance)"%gap if gap<2e-3
         else "INVESTIGATE: gap %.4f exceeds panel drift"%gap)
print("bridge control: canonical config corr.mean %.6f over %d periods"%(b["corr"]["mean"],bp))
print("  vs e13_raw (same config, recent panel) %.6f over %d periods -> %s"
      %(ref["corr"]["mean"],rp,verdict))
print("  (old cad_lgb_reg_w6m_r6m reference was 0.149754 over 510 -- panel has since grown)")
PY
fi
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
