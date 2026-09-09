#!/usr/bin/env bash
#
# Experiment E25: weighted and selection combiners over a diverse pool
#
# WHAT THIS ASKS, AND WHY IT IS THE NEXT QUESTION
#   E10  a FITTED meta-learner (ridge / MLP on member predictions) beat nothing.
#        Diagnosis: its members were 0.86 correlated -- no decorrelated error to
#        combine, so every meta collapsed onto 1/N.
#   E23  a FIXED 1/N over a genuinely diverse pool ALSO beat nothing. Diagnosis:
#        the only orthogonal member (torch LSTM, 0.35-0.41 against everything)
#        is also the WEAKEST (payout 0.061 against 0.083), and 1/N cannot
#        downweight anything BY CONSTRUCTION, so its level drag cancelled its
#        decorrelation benefit.
#
#   Between those two sits the untested cell: a combiner that CAN set unequal
#   weights, over members that genuinely differ. That is this study.
#
# WHAT THE WEIGHTS ALREADY SHOW (from a single pilot arm, stacking_nnls)
#   The scheme does not do what the E23 write-up predicted. It does NOT
#   downweight the weak LSTM -- it holds it near 0.20, i.e. its 1/N share -- and
#   instead zeroes the MLP outright:
#
#       ridge 0.20-0.36   mlp 0.00   lgb 0.28-0.35   glmnet 0.08-0.24   lstm 0.18-0.23
#
#   So the lever these schemes actually pull is REDUNDANCY (the MLP is 0.922
#   correlated with glmnet and adds nothing), not WEAKNESS. Worth stating in
#   advance, because it changes what a null here would mean.
#
# TWO POOLS, AND THE SECOND IS THE CONTROL
#   pool5  ridge + mlp + lgb + glmnet + lstm     contains the orthogonal member
#   tab4   ridge + mlp + lgb + glmnet            same minus the LSTM
#
#   Every scheme runs on both. If schemes beat 1/N only on pool5, the gain is
#   about handling the orthogonal member. If they beat it on both, the gain is
#   about weighting/redundancy and has nothing to do with diversity. Without the
#   second pool those two stories are indistinguishable.
#
# THE BAR
#   Each arm is paired against the 1/N blend of the IDENTICAL members from E23
#   (e23_all5 for pool5, e23_tab4 for tab4) on the dates they share. 16 arms
#   means 16 tests, so a Bonferroni-style threshold is |t| ~ 2.9, not 2.
#   ~500 dates at ~95% overlap is ~26 effective independent windows.
#
# NOT DEPLOYABLE: every pool5 arm contains the LSTM, which has no live path.
# COST: ~75s per arm, ~20 min total. Nothing is refitted; forecasts are read.
#
# Usage:  bash experiments/e25_weighted_combiners_crypto.sh   [NO_MAIL=1] [DRY_RUN=1]
#         ONLY="stacking_nnls caruana"   SKIP="bagged_caruana"   POOLS="pool5"
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E25}"
PREFIX="e25"
TREE="ridge_glmnet"
UPDATE_EVERY="${UPDATE_EVERY:-22}"   # matches E10's meta cadence and the NNLS control
LAG="${LAG:-20}"                     # the 20-day target's resolution lag, NOT a CV embargo
SEED="${SEED:-42}"

RIDGE="ridge/e10_ridge_raw"; MLP="mlp/e10_mlp_raw"; LGB="lgb/e05_cumlag0"
GLM="ridge_glmnet/w6r6_glmnet_reg"; LSTM="lstm/a2_fenccausal"

POOL5="${RIDGE} ${MLP} ${LGB} ${GLM} ${LSTM}"
TAB4="${RIDGE} ${MLP} ${LGB} ${GLM}"
POOLS="${POOLS:-pool5 tab4}"
SCHEMES="${SCHEMES:-equal inverse_variance complete_subset caruana bagged_caruana shrunk_tangency granger_ramanathan stacking_nnls}"

ran=0; failed=0; FAILED_ARMS=()
echo "=============================================================="
echo "${EXPERIMENT}: weighted combiners  |  pools: ${POOLS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for pool in ${POOLS}; do
  case "${pool}" in
    pool5) members="${POOL5}" ;;
    tab4)  members="${TAB4}"  ;;
    *) echo "unknown pool ${pool}" >&2; exit 1 ;;
  esac
  for scheme in ${SCHEMES}; do
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${scheme} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${scheme} "* ]] && continue
    run_name="${PREFIX}_${pool}_${scheme}"
    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  would run  %s\n' "${run_name}"; continue; fi
    ran=$((ran+1))
    echo; echo "--- ${run_name}"
    python3 -u -m src.run_weighted_ensemble crypto "${TREE}" ${members} \
        --scheme "${scheme}" --update-every "${UPDATE_EVERY}" --lag "${LAG}" \
        --seed "${SEED}" --run-name "${run_name}" 2>&1 | tee "logs/${run_name}.log"
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
echo "${EXPERIMENT} -- weighted and selection combiners over a diverse pool"
echo "ran ${ran} arm(s), ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo
python3 - "${PREFIX}" "${TREE}" <<'PY'
import glob, json, os, sys
import numpy as np, pandas as pd
# Shared arithmetic so this cannot diverge from report_pdf.
from src.analysis.results_table import payout_series, charged_corr, degenerate_periods

prefix, tree = sys.argv[1], sys.argv[2]
CTRL = {"pool5": "e23_all5", "tab4": "e23_tab4"}   # 1/N on the IDENTICAL members
MEMBERS = {"ridge": ("ridge","e10_ridge_raw"), "mlp": ("mlp","e10_mlp_raw"),
           "lgb": ("lgb","e05_cumlag0"), "glmnet": ("ridge_glmnet","w6r6_glmnet_reg"),
           "lstm": ("lstm","a2_fenccausal")}

def pp(m, r):
    return pd.read_csv("results/per_period/crypto/%s/%s.csv" % (m, r)).set_index("date")

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n-1)+1):
        v += 2*(1-k/(lag+1))*((e[k:] @ e[:-k])/n)
    se = np.sqrt(max(v, 0)/n)
    return m, se, (m/se if se > 0 else float("nan"))

arms = {}
for path in sorted(glob.glob("results/metrics/crypto/%s/%s_*.json" % (tree, prefix))):
    run = os.path.basename(path)[:-5]
    rest = run[len(prefix)+1:]
    pool = "pool5" if rest.startswith("pool5_") else "tab4"
    arms[run] = (pool, rest[len(pool)+1:], json.load(open(path)))

frames = {}
for run in arms: frames[run] = pp(tree, run)
for k,(m,r) in list(MEMBERS.items()) : frames[k] = pp(m, r)
for c in set(CTRL.values()): frames[c] = pp(tree, c)

idx = None
for f in frames.values():
    idx = set(f.index) if idx is None else idx & set(f.index)
idx = sorted(idx)
print("Every figure on the %d dates ALL rows share. Payout rebuilt per period." % len(idx))
print()

def pay(run): return payout_series(frames[run]).loc[idx].mean()

print("%-30s %-6s %5s %10s %10s %10s" % ("run","pool","dead","CORR","MMC","payout"))
print("-"*76)
rowset = ([(r, arms[r][0]) for r in arms]
          + [(c, "ctl-1/N") for c in sorted(set(CTRL.values()))]
          + [(k, "member") for k in MEMBERS])
for run, pool in sorted(rowset, key=lambda x: -pay(x[0])):
    f = frames[run]
    print("%-30s %-6s %5d %10.6f %10.6f %10.6f" % (
        run, pool, int(degenerate_periods(f).sum()),
        charged_corr(f).loc[idx].mean(), f["mmc"].loc[idx].mean(), pay(run)))
print("-"*76)

def table(title, pairs, note=None):
    print(); print(title)
    if note: print("  " + note)
    print("%-30s %-14s %8s %11s %9s %7s" % ("arm","control","metric","delta","se","t"))
    print("-"*84)
    for a, b in pairs:
        if a not in frames or b not in frames:
            print("%-30s %-14s  missing -- NOT COMPARED" % (a, b)); continue
        for metric in ("corr","payout"):
            la = (payout_series(frames[a]) if metric=="payout" else charged_corr(frames[a])).loc[idx]
            lb = (payout_series(frames[b]) if metric=="payout" else charged_corr(frames[b])).loc[idx]
            j = pd.concat([lb, la], axis=1, keys=["b","a"]).dropna()
            d, se, t = nw(j["a"] - j["b"])
            flag = "  <-- clears 2.9" if abs(t) >= 2.9 else ""
            print("%-30s %-14s %8s %+11.6f %9.6f %+7.2f%s" % (a, b, metric, d, se, t, flag))
    print("-"*84)

for pool in ("pool5","tab4"):
    runs = sorted(r for r in arms if arms[r][0] == pool)
    if runs:
        table("DOES A WEIGHTED/SELECTION SCHEME BEAT 1/N?  pool=%s, vs %s, lag 20"
              % (pool, CTRL[pool]), [(r, CTRL[pool]) for r in runs],
              "identical members and dates; the ONLY difference is how weights are set")

# What weight did each scheme actually give the orthogonal member?
print()
print("WEIGHT ON THE ORTHOGONAL MEMBER (lstm), pool5 -- the mechanism evidence")
print("%-30s %10s %10s %10s" % ("run","mean w(lstm)","min","max"))
print("-"*64)
for run in sorted(r for r in arms if arms[r][0] == "pool5"):
    ups = arms[run][2].get("weight_updates") or []
    ws = []
    for u in ups:
        w = u.get("weights")
        if isinstance(w, dict):
            for k, v in w.items():
                if "fenccausal" in k or "lstm" in k: ws.append(float(v))
        elif isinstance(w, (list, tuple)) and len(w) == 5:
            ws.append(float(w[4]))
    if ws: print("%-30s %10.3f %10.3f %10.3f" % (run, np.mean(ws), min(ws), max(ws)))
    else:  print("%-30s %10s" % (run, "n/a"))
print("-"*64)
print("1/N over 5 members would give it 0.200.")

print()
print("READ IT THIS WAY")
print("  16 arms means 16 tests: the honest threshold is |t| ~ 2.9, not 2.")
print("  A scheme beating 1/N on pool5 but NOT on tab4 is handling the orthogonal")
print("  member. Beating it on BOTH is about redundancy/weighting and says nothing")
print("  about diversity. Beating it on NEITHER closes the combiner question.")
print("  DEPLOYABILITY: every pool5 arm contains the LSTM and cannot run live.")
PY
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
