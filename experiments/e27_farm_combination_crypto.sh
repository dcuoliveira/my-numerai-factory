#!/usr/bin/env bash
# E27 -- FARM forecast combination vs the simple average.
#
# Replicates Lee & Lee (2025), "Solving the Forecast Combination Puzzle", on the
# crypto book. The paper treats the simple average f_bar as a common factor and
# regresses its own error on the members' idiosyncratic deviations d_i = f_i -
# f_bar, adding back what is predictable (FARM1); FARM2/FARM3 first strip r <= 2
# extra PCA / supervised-PCA factors. See src/prediction/farm.py.
#
# THE QUESTION, and why it has a known null. E10, E23 and E25 all found that a
# fitted combiner loses to 1/N on this book, and the ensemble screener records the
# same across 3,276 combinations. FARM is the one combiner family that does not
# just reweight -- it can move a rank relative to the average -- so it is worth a
# clean, paired test. The prior is still that it fails on numerai_corr, because
# FARM minimises SQUARED error and the book is paid on rank correlation.
#
# TWO POOLS OF THE FIVE NAMED FORECASTERS
#   decorr5  the most-decorrelated glmnet/lgb/mlp run (CORR-floored so
#            'decorrelated' cannot pick noise) + the two E26 sign-constrained GRRs
#   strong5  each family's best-CORR run + the same two GRRs -- robustness against
#            the floor choice
#   The grr_pos / grr_neg members (E26, beta >= 0 and beta <= 0) are the
#   decorrelating pair: 0.25 with each other, ~0.4-0.5 with the rest.
#
# TWELVE ARMS PER POOL. equal is the 1/N baseline every arm is paired against;
# farm1/2/3 x {ridge, l2boost, postalasso} are the paper's FARM estimators; gr and
# ds are its Granger-Ramanathan and Egalitarian-Lasso reference combiners. Every
# arm is written through the SAME scoring path as run_ensemble, so it lands in the
# experiment database and reports with the standard tooling.
#
# COST: minutes per arm. run_farm_ensemble reads finished forecasts and refits
# nothing -- the FARM fit is a K x K Gram solve per update.
#
# Usage:  bash experiments/e27_farm_combination_crypto.sh
#         POOLS="decorr5 strong5 highN"   ARMS="equal farm1:l2boost"   DRY_RUN=1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E27}"
PREFIX="e27"
TREE="ridge_glmnet"                 # output tree; cosmetic for a combination
UPDATE_EVERY="${UPDATE_EVERY:-22}"
LAG="${LAG:-20}"                    # the 20-day target's resolution lag

GRR_POS="gen_ridge/e26_pos"; GRR_NEG="gen_ridge/e26_neg"
DECORR5="ridge_glmnet/e17_featdiff9 lgb/e05_cumlag7 mlp/e08_mlp_onehot_rollsel ${GRR_POS} ${GRR_NEG}"
STRONG5="ridge_glmnet/w6r6_lobo_tenc lgb/e02_concat_trio mlp/a2_mlp_fenccausal ${GRR_POS} ${GRR_NEG}"
# highN is the paper's high-dimensional regime; opt in with POOLS, it is slow.
HIGHN="__built_at_runtime__"

POOLS="${POOLS:-decorr5 strong5}"
ARMS="${ARMS:-equal farm1:ridge farm1:l2boost farm1:postalasso farm2:ridge farm2:l2boost farm2:postalasso farm3:ridge farm3:l2boost farm3:postalasso gr ds}"

build_highn() {   # every dataset's best glmnet/lgb/mlp run + the two GRRs
  python3 - "${GRR_POS}" "${GRR_NEG}" <<'PY'
import glob, json, os, sys
import pandas as pd
from src.analysis.results_table import target_of, payout_series
best = {}
for path in sorted(glob.glob("results/metrics/crypto/*/*.json")):
    model = path.split(os.sep)[-2]; run = os.path.basename(path)[:-5]
    if model not in ("ridge_glmnet", "lgb", "mlp"): continue
    if "clf" in run: continue
    cfg = json.load(open(path)).get("config", {})
    if cfg.get("members"): continue                      # skip blends
    if target_of("crypto", model, run) != "target_binned_return_20": continue
    pp = "results/per_period/crypto/%s/%s.csv" % (model, run)
    if not os.path.exists(pp): continue
    f = pd.read_csv(pp)
    if "date" in f.columns: f = f.set_index("date")
    if "corr" not in f.columns or len(f) < 470: continue
    c = float(f["corr"].mean()); ds = str(cfg.get("feature_set") or cfg.get("data_version"))
    key = (model, ds)
    if any(k in run.lower() or k in ds.lower() for k in ("leak", "fencglobal")): continue
    if key not in best or c > best[key][1]: best[key] = ("%s/%s" % (model, run), c)
print(" ".join(v[0] for v in best.values()) + " " + sys.argv[1] + " " + sys.argv[2])
PY
}

ran=0; failed=0; FAILED_ARMS=()
echo "=============================================================="
echo "${EXPERIMENT}: FARM combination vs the simple average  |  pools: ${POOLS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for pool in ${POOLS}; do
  case "${pool}" in
    decorr5) members="${DECORR5}" ;;
    strong5) members="${STRONG5}" ;;
    highN)   members="$(build_highn)" ;;
    *) echo "unknown pool ${pool}" >&2; exit 1 ;;
  esac
  for arm in ${ARMS}; do
    method="${arm%%:*}"; beta="${arm#*:}"; [ "${beta}" = "${arm}" ] && beta="ridge"
    run_name="${PREFIX}_${pool}_${arm/:/_}"
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  would run  %-32s %s\n' "${run_name}" "${method}/${beta}"; continue; fi
    ran=$((ran+1))
    echo; echo "--- ${run_name}"
    python3 -u -m src.run_farm_ensemble crypto "${TREE}" ${members} \
        --method "${method}" --beta-method "${beta}" \
        --update-every "${UPDATE_EVERY}" --lag "${LAG}" \
        --run-name "${run_name}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) -- continuing"
    fi
  done
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- FARM forecast combination vs the simple average"
echo "ran ${ran} arm(s), ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo
python3 - "${PREFIX}" "${TREE}" "${POOLS}" <<'PY'
import glob, os, sys
import numpy as np, pandas as pd
from src.analysis.results_table import payout_series

prefix, tree, pools = sys.argv[1], sys.argv[2], sys.argv[3].split()

def pp(run):
    p = "results/per_period/crypto/%s/%s.csv" % (tree, run)
    return pd.read_csv(p).set_index("date") if os.path.exists(p) else None

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, (m / se if se > 0 else float("nan"))

for pool in pools:
    base = pp("%s_%s_equal" % (prefix, pool))
    if base is None:
        print("%s: no equal baseline, skipping" % pool); continue
    print("=== %s === paired vs equal (1/N), Newey-West lag 20" % pool)
    print("%-22s %8s %8s %10s %8s" % ("arm", "CORR", "payout", "dPay(t)", "dCORR(t)"))
    rows = []
    for path in sorted(glob.glob("results/metrics/crypto/%s/%s_%s_*.json" % (tree, prefix, pool))):
        run = os.path.basename(path)[:-5]; arm = run[len("%s_%s_" % (prefix, pool)):]
        f = pp(run)
        if f is None: continue
        common = f.index.intersection(base.index)
        dpay = payout_series(f).reindex(common) - payout_series(base).reindex(common)
        dcor = f["corr"].reindex(common) - base["corr"].reindex(common)
        mp, tp = nw(dpay.dropna()); mc, tc = nw(dcor.dropna())
        rows.append((arm, float(payout_series(f).mean()) * 0 + f["corr"].mean(),
                     float(payout_series(f).mean()), mp, tp, mc, tc,
                     arm == "equal"))
    for arm, corr, pay, mp, tp, mc, tc, is_eq in sorted(rows, key=lambda r: -r[2]):
        tag = "  <- baseline" if is_eq else ""
        print("%-22s %8.4f %8.4f %+8.4f(%+.2f) %+.4f(%+.2f)%s"
              % (arm, corr, pay, mp, tp, mc, tc, tag))
    print()
PY
} | tee "${REPORT}"
echo "wrote ${REPORT}"
