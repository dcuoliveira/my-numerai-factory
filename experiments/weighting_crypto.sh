#!/usr/bin/env bash
#
# Experiment: how to weight CURUPIRA_05's five members
#
# CURUPIRA_05 rank-averages five models at a fixed 1/N. This runs eight
# weighting schemes over those SAME five members, with the weights re-estimated
# walk-forward, and asks whether any of them beats 1/N once the estimation is
# done honestly.
#
#     equal               1/N -- the benchmark, and what is deployed
#     inverse_variance    Bates-Granger (1969), diagonal only
#     complete_subset     Elliott, Gargano & Timmermann (2013)
#     shrunk_tangency     Diebold-Pauly (1990), covariance shrunk toward 1/N
#     granger_ramanathan  Granger-Ramanathan (1984), constrained NNLS
#     caruana             Caruana et al. (2004) greedy selection with replacement
#     bagged_caruana      the bagged variant from the same paper
#     stacking_nnls       ridge-penalised NNLS on centred ranks
#
# ---------------------------------------------------------------------------
# Why this is a real experiment and not a formality
# ---------------------------------------------------------------------------
#
# The weights CURUPIRA_05 runs were chosen by searching subsets and a weight
# simplex over the SAME 510 periods it is scored on. That is not evidence about
# live behaviour. Every scheme here estimates its weights using only periods
# whose targets had resolved at the time -- weights applied at t may read member
# scores only through t - 20, because a 20-day forward target opened at p is not
# known until p + 20.
#
# That lag is not cosmetic. Running the greedy scheme with --lag 0 instead of 20
# lifts payout from 0.070391 to 0.076740. The entire +0.0063 is leakage.
#
# What to expect: the forecast combination puzzle (Timmermann 2006; Smith &
# Wallis 2009) says estimated weights usually lose to 1/N when the sample is
# short, and this sample is very short -- ~25 independent windows for 4 free
# weights. Equal weighting already sits at the 97th percentile of 20,000 random
# long-only weightings on this data. A scheme has to beat that, out of sample,
# to justify the machinery.
#
# The schemes are ordered above by how much they read from the covariance, which
# is the same as ordering them by how much damage they can do.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/weighting_crypto.sh
#   DRY_RUN=1 bash experiments/weighting_crypto.sh
#   ONLY=caruana bash experiments/weighting_crypto.sh
#   WGT_UPDATE_EVERY=65 bash experiments/weighting_crypto.sh   # quarterly
#   WGT_WINDOW=260 bash experiments/weighting_crypto.sh        # rolling, not expanding
#
# Under screen:
#   screen -L -Logfile weighting.txt bash experiments/weighting_crypto.sh
#
# Nothing is refitted -- every scheme reads the members' existing forecast CSVs,
# so the whole sweep is a couple of minutes.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# WGT_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW= would silently become 0.
WGT_UPDATE_EVERY="${WGT_UPDATE_EVERY:-22}"   # ~monthly
WGT_LAG="${WGT_LAG:-20}"                     # = the target horizon
WGT_MIN_HISTORY="${WGT_MIN_HISTORY:-60}"
WGT_WINDOW="${WGT_WINDOW:-0}"                # 0 = expanding
WGT_OBJECTIVE="${WGT_OBJECTIVE:-sharpe}"
WGT_SHRINKAGE="${WGT_SHRINKAGE:-0.2}"
WGT_SUBSET_SIZE="${WGT_SUBSET_SIZE:-3}"
WGT_SEED="${WGT_SEED:-42}"

if [ "${WGT_LAG}" -lt 20 ]; then
    echo "WGT_LAG=${WGT_LAG} is below the 20-day target horizon. Weights would" \
         "read scores that had not resolved yet; that is leakage, and it is" \
         "worth ~0.006 payout. Set it deliberately or leave it at 20." >&2
    [ "${ALLOW_LEAKY_LAG:-0}" = "1" ] || exit 1
fi

# CURUPIRA_05's members, exactly as registry.py declares them. Note size uses
# the fixed-alpha ridge, so the member list crosses wrapper directories.
G="ridge_glmnet"
MEMBERS=(
  "${G}/w6r6_alone_onehot"
  "${G}/w6r6_alone_tenc"
  "${G}/w6r6_alone_interact"
  "ridge/w6r6_grp_size_fixed"
  "${G}/w6r6_grp_price_glmnet"
)

SCHEMES=(equal inverse_variance complete_subset shrunk_tangency
         granger_ramanathan caruana bagged_caruana stacking_nnls)

total="${#SCHEMES[@]}"
index=0; failed=0
started_all=$(date +%s)

echo "=============================================================="
echo "weighting sweep | ${total} schemes over ${#MEMBERS[@]} members"
echo "update_every=${WGT_UPDATE_EVERY}  lag=${WGT_LAG}  min_history=${WGT_MIN_HISTORY}" \
     " window=${WGT_WINDOW}  objective=${WGT_OBJECTIVE}"
echo "=============================================================="

for scheme in "${SCHEMES[@]}"; do
    index=$((index + 1))
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${scheme}" ] && continue

    run_name="w6r6_wgt_${scheme}"
    cmd=(python3 -u -m src.run_weighted_ensemble crypto ridge_glmnet
         "${MEMBERS[@]}"
         --scheme "${scheme}"
         --update-every "${WGT_UPDATE_EVERY}" --lag "${WGT_LAG}"
         --min-history "${WGT_MIN_HISTORY}" --window "${WGT_WINDOW}"
         --objective "${WGT_OBJECTIVE}" --shrinkage "${WGT_SHRINKAGE}"
         --subset-size "${WGT_SUBSET_SIZE}" --seed "${WGT_SEED}"
         --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %s\n' "$index" "$total" "$scheme" "${cmd[*]}"
        continue
    fi

    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${scheme}   $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/wgt_${scheme}.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        failed=$((failed + 1)); echo ">>> ${scheme} FAILED -- continuing"
    else
        echo ">>> ${scheme} done in $(( $(date +%s) - started ))s"
    fi
done

echo
echo "=============================================================="
echo "finished ${total} schemes in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo
cat <<'REPORT'
compare against 1/N and against CURUPIRA_05 as deployed:

  python3 - <<'PY'
import json, glob, os
import numpy as np, pandas as pd
def nw(x, lag=20):
    x=np.asarray(x,float); n=len(x); mu=x.mean(); e=x-mu; s=(e@e)/n
    for L in range(1,lag+1): s+=2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    return mu, mu/np.sqrt(s/n)
S={}
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/*/w6r6_wgt_*.json")) + [
        "results/metrics/crypto/ridge_glmnet/w6r6_blend_quintet2.json"]:
    if not os.path.exists(p): continue
    n=os.path.basename(p)[:-5]; d=json.load(open(p))
    s=pd.read_csv("results/per_period/crypto/%s/%s.csv" % (p.split("/")[3], n))
    pay=0.1*s["corr"]+s["mmc"]; cs=pay.cumsum(); S[n]=pay
    rows.append((pay.mean()/pay.std(ddof=0), n, d["corr"]["mean"], d["mmc"]["mean"],
                 pay.mean(), (cs.expanding(1).max()-cs).max()))
print("%-30s %10s %10s %10s %8s %8s" % ("scheme","CORR","MMC","payout","p.shp","p.dd"))
for sh,n,c,m,pay,dd in sorted(rows, reverse=True):
    print("%-30s %10.6f %10.6f %10.6f %8.4f %8.4f" % (n,c,m,pay,sh,dd))
ref="w6r6_wgt_equal"
if ref in S:
    print("\npaired vs 1/N (Newey-West lag 20):")
    for n in sorted(S):
        if n == ref: continue
        j=pd.concat([S[ref],S[n]],axis=1,join="inner",keys=["r","a"])
        m,t=nw(j["a"]-j["r"])
        print("  %-30s d payout %+.6f  t %+.2f" % (n,m,t))
PY
REPORT
echo "=============================================================="
