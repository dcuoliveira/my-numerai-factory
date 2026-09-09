#!/usr/bin/env bash
#
# Experiment E23: cross-class 1/N blends -- does DIVERSITY buy what FITTING did not?
#
# WHAT THIS ASKS
#   E10 fitted ridge and MLP meta-learners over member predictions and beat
#   nothing: every meta landed on top of a plain 1/N average, and none beat the
#   better of its own two members. The diagnosis was not the combiner but the
#   MEMBERS -- they were 0.86 correlated, so there were no decorrelated errors
#   left to combine.
#
#   The within-period rank correlation of one representative arm per class,
#   measured over 503 common dates:
#
#              ridge     mlp     lgb  glmnet    lstm
#     ridge    1.000   0.863   0.747   0.881   0.354
#     mlp      0.863   1.000   0.726   0.922   0.373
#     lgb      0.747   0.726   1.000   0.716   0.356
#     glmnet   0.881   0.922   0.716   1.000   0.406
#     lstm     0.354   0.373   0.356   0.406   1.000
#
#   The four tabular models are one block. The torch LSTM is the only genuinely
#   orthogonal thing on the board -- and it is also the WEAKEST alone (0.061
#   payout against 0.084 for glmnet). So this asks the cheap question before any
#   expensive one: with NO fitting at all, how much does a fixed 1/N rank average
#   gain from adding a decorrelated but weak member?
#
# FOUR ARMS, AND WHY THE CONTROLS ARE THE POINT
#   div3   glmnet + lgb + lstm                  the diverse trio
#   tab2   glmnet + lgb                         div3 MINUS the lstm
#   all5   ridge + mlp + lgb + glmnet + lstm    everything
#   tab4   ridge + mlp + lgb + glmnet           all5 MINUS the lstm
#
#   div3-vs-tab2 and all5-vs-tab4 are the mechanism test. Without them a gain is
#   indistinguishable from "averaging more things helps", which is true of any
#   noisy series and says nothing about diversity.
#
# NOT DEPLOYABLE
#   Every arm containing the lstm is a MEASUREMENT ONLY. src/prediction/nn/ is
#   deliberately absent from prediction.models.MODELS, so there is no live
#   fit/predict path for it (CLAUDE.md, "The LSTM cannot run live"). tab2/tab4
#   are deployable; div3/all5 are not, today.
#
# COST: seconds. run_ensemble reads finished forecasts and refits nothing.
#
# Usage:  bash experiments/e23_diverse_blend_crypto.sh        [NO_MAIL=1] [DRY_RUN=1]
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E23}"
PREFIX="e23"
TREE="ridge_glmnet"          # output tree; cosmetic for a blend, members are in the config
TARGET="target_binned_return_20"

RIDGE="ridge/e10_ridge_raw"
MLP="mlp/e10_mlp_raw"
LGB="lgb/e05_cumlag0"
GLM="ridge_glmnet/w6r6_glmnet_reg"
LSTM="lstm/a2_fenccausal"

ANAME=(); AMEMB=()
aadd() { ANAME+=("$1"); AMEMB+=("$2"); }
aadd div3 "${GLM} ${LGB} ${LSTM}"
aadd tab2 "${GLM} ${LGB}"
aadd all5 "${RIDGE} ${MLP} ${LGB} ${GLM} ${LSTM}"
aadd tab4 "${RIDGE} ${MLP} ${LGB} ${GLM}"

ran=0; failed=0; FAILED_ARMS=()
echo "=============================================================="
echo "${EXPERIMENT}: cross-class 1/N blends  |  ${#ANAME[@]} arms"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ANAME[@]}"; do
    run_name="${PREFIX}_${ANAME[$i]}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '  would run  %-12s <- %s\n' "${run_name}" "${AMEMB[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "--- ${run_name}  <- ${AMEMB[$i]}"
    python3 -u -m src.run_ensemble crypto "${TREE}" ${AMEMB[$i]} \
        --target "${TARGET}" --run-name "${run_name}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) -- continuing"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
{
echo "${EXPERIMENT} -- cross-class 1/N blends: does diversity buy what fitting did not?"
echo "ran ${ran} arm(s), ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo
python3 - "${PREFIX}" "${TREE}" <<'PY'
import sys
import numpy as np, pandas as pd
# Shared arithmetic, so this cannot diverge from report_pdf: payout is rebuilt
# per period and a degenerate period is charged the 0 it earned, not dropped.
from src.analysis.results_table import payout_series, charged_corr, degenerate_periods

prefix, tree = sys.argv[1], sys.argv[2]
MEMBERS = {"ridge": ("ridge","e10_ridge_raw"), "mlp": ("mlp","e10_mlp_raw"),
           "lgb": ("lgb","e05_cumlag0"), "glmnet": ("ridge_glmnet","w6r6_glmnet_reg"),
           "lstm": ("lstm","a2_fenccausal")}
ARMS = {"div3": (tree, prefix+"_div3"), "tab2": (tree, prefix+"_tab2"),
        "all5": (tree, prefix+"_all5"), "tab4": (tree, prefix+"_tab4")}

def pp(m, r):
    return pd.read_csv("results/per_period/crypto/%s/%s.csv" % (m, r)).set_index("date")

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n-1)+1):
        v += 2*(1-k/(lag+1))*((e[k:] @ e[:-k])/n)
    se = np.sqrt(max(v, 0)/n)
    return m, se, (m/se if se > 0 else float("nan"))

everything = dict(MEMBERS); everything.update(ARMS)
frames, missing = {}, []
for k, (m, r) in everything.items():
    try: frames[k] = pp(m, r)
    except FileNotFoundError: missing.append(k)
if missing: print("NOT ON DISK (excluded, and named):", ", ".join(missing)); print()

idx = None
for f in frames.values():
    idx = set(f.index) if idx is None else idx & set(f.index)
idx = sorted(idx)

print("Every figure on the %d dates ALL rows share. Payout rebuilt per period." % len(idx))
print()
print("%-10s %-34s %5s %10s %10s %10s" % ("row","members / arm","dead","CORR","MMC","payout"))
print("-"*86)
rows = []
for k, (m, r) in everything.items():
    f = frames.get(k)
    if f is None: continue
    rows.append((k, "%s/%s" % (m, r), int(degenerate_periods(f).sum()),
                 charged_corr(f).loc[idx].mean(), f["mmc"].loc[idx].mean(),
                 payout_series(f).loc[idx].mean()))
for k, lab, dead, c, mm, p in sorted(rows, key=lambda x: -x[5]):
    tag = "  <- BLEND" if k in ARMS else ""
    print("%-10s %-34s %5d %10.6f %10.6f %10.6f%s" % (k, lab, dead, c, mm, p, tag))
print("-"*86)

best_member = max((r for r in rows if r[0] in MEMBERS), key=lambda x: x[5])
print("best single member on this window: %s (%.6f)" % (best_member[0], best_member[5]))

def table(title, pairs, note=None):
    print(); print(title)
    if note: print("  " + note)
    print("%-14s %-14s %8s %11s %9s %7s" % ("arm","control","metric","delta","se","t"))
    print("-"*70)
    for a, b in pairs:
        if a not in frames or b not in frames:
            print("%-14s %-14s  missing -- NOT COMPARED" % (a, b)); continue
        for metric in ("corr", "payout"):
            fa, fb = frames[a], frames[b]
            la = (payout_series(fa) if metric == "payout" else charged_corr(fa)).loc[idx]
            lb = (payout_series(fb) if metric == "payout" else charged_corr(fb)).loc[idx]
            j = pd.concat([lb, la], axis=1, keys=["b","a"]).dropna()
            d, se, t = nw(j["a"] - j["b"])
            print("%-14s %-14s %8s %+11.6f %9.6f %+7.2f" % (a, b, metric, d, se, t))
    print("-"*70)

table("1. DOES THE BLEND BEAT ITS BEST SINGLE MEMBER?   paired, Newey-West lag 20",
      [(a, best_member[0]) for a in ("div3","tab2","all5","tab4")],
      "the bar a combiner must clear before anything else matters")

table("2. WHAT DOES THE ORTHOGONAL MEMBER ADD?   with-lstm vs the same blend without",
      [("div3","tab2"), ("all5","tab4")],
      "THE mechanism test: isolates diversity from 'averaging more things helps'")

print()
print("READ IT THIS WAY")
print("  Table 2 is the finding. Table 1 can be positive merely because averaging")
print("  reduces variance; only table 2 attributes a gain to DECORRELATION.")
print("  ~500 periods at ~95% overlap is ~26 effective independent windows, so the")
print("  bar here is stricter than E10's 151-date tail -- but still read |t|>=2.")
print("  DEPLOYABILITY: div3 and all5 contain the LSTM and CANNOT run live today.")
print("  tab2 and tab4 can. A gain that exists only with the LSTM is a research")
print("  result, not a deployable one, until a torch live path exists.")
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
