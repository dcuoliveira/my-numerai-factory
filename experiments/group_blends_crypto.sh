#!/usr/bin/env bash
#
# Experiment: the best TRIO and the best QUINTET of feature groups
#
# The group study fitted every block on its own. This one asks whether combining
# a few of them beats using all 22 raw features in one model -- and does it two
# structurally different ways, because they are not the same experiment and the
# study has never separated them:
#
#   BLEND       fit each group as its own model, rank its predictions within
#               each period, and average the ranks with fixed weights. Three or
#               five estimators, each with its own alpha chosen for its own
#               block. This is also exactly what splitting a stake across
#               several Numerai model slots does.
#
#   CONCAT      one model on the union of those groups' columns. One estimator,
#               ONE alpha across every column.
#
# The distinction matters here more than usual. glmnet picks a single shrinkage
# for the whole design matrix, so a 740-column identity block and a 4-column
# price block cannot be regularised differently inside one fit -- the blend can
# do that and the concat cannot. Every "more features" arm in this study came
# out flat, and this is the cleanest test of whether that is the features'
# fault or the single-alpha constraint's.
#
# ---------------------------------------------------------------------------
# The combinations, and where they come from
# ---------------------------------------------------------------------------
#
# Chosen by searching every 3- and 5-subset of the twelve fitted blocks over a
# long-only weight simplex in steps of 0.2, maximising Sharpe on the payable
# series (0.1*CORR + 1*MMC):
#
#   trio      onehot 0.4 + interact 0.4 + price 0.2     sharpe 1.2653  payout 0.0524
#   quintet   onehot + tenc + interact + price + bollinger, equal   1.3454  0.0428
#
# A second quintet is included because the top one buys its Sharpe by holding
# bollinger, which scores 0.0038 payout alone -- almost certainly a weight fitted
# to noise. The runner-up swaps bollinger for size, costs 0.0023 Sharpe and gains
# 0.0115 payout, and is the more defensible object:
#
#   quintet2  onehot + tenc + interact + size + price, equal        1.3431  0.0543
#
# NOTE what is absent: volatility and rolling, the two highest-scoring groups
# individually (CORR 0.1436 and 0.1402). They correlate 0.912 with each other
# and carry payout drawdowns above 4.3, so Sharpe-optimal weighting drops both.
#
# ---------------------------------------------------------------------------
# The bar to beat
# ---------------------------------------------------------------------------
#
#     raw 22, ridge_glmnet    sharpe 1.2825   payout 0.083859
#     deployed 3-slot book    sharpe 1.2413   payout 0.085391
#
# The blends' in-sample Sharpes (1.27-1.35) are in that range while their payouts
# are 35-50% lower, and those weights were fitted on the same 510 periods they
# are scored on -- roughly 26 effective independent windows. Treat any blend
# Sharpe above the raw model's as unproven until it survives a paired test.
# src/analysis/ablation_report.py's newey_west is the tool for that.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/group_blends_crypto.sh
#   DRY_RUN=1 bash experiments/group_blends_crypto.sh
#   ONLY=trio bash experiments/group_blends_crypto.sh
#   SKIP_CONCAT=1 bash experiments/group_blends_crypto.sh   # blends only (seconds)
#
# Under screen:
#   screen -L -Logfile group-blends.txt bash experiments/group_blends_crypto.sh
#
# The blends are free -- they read existing forecast CSVs and refit nothing. The
# concat arms are full pseudo-OOS runs at roughly 50 min each, so the whole
# script is about 2.5 hours, essentially all of it the three concats.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# BLD_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
BLD_MODEL="${BLD_MODEL:-ridge_glmnet}"
BLD_TASK="${BLD_TASK:-regression}"
BLD_SCHEME="${BLD_SCHEME:-rolling}"
BLD_TRAIN_WINDOW="${BLD_TRAIN_WINDOW:-6month}"
BLD_REFIT="${BLD_REFIT:-6month}"
BLD_N_ITER="${BLD_N_ITER:-50}"
BLD_N_SPLITS="${BLD_N_SPLITS:-5}"
BLD_N_JOBS="${BLD_N_JOBS:-8}"
BLD_SEED="${BLD_SEED:-42}"

case "${BLD_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "BLD_TRAIN_WINDOW=${BLD_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact"

# Member runs, as model_dir/run_name. run_ensemble resolves that syntax so a
# blend can cross wrapper directories -- size's best variant is fixed-alpha
# ridge while the rest are glmnet, and forcing them into one tree would mean
# comparing the wrong fits.
M_ONEHOT="ridge_glmnet/w6r6_alone_onehot"
M_INTERACT="ridge_glmnet/w6r6_alone_interact"
M_TENC="ridge_glmnet/w6r6_alone_tenc"
M_PRICE="ridge_glmnet/w6r6_grp_price_glmnet"
M_SIZE="ridge/w6r6_grp_size_fixed"
M_BOLL="ridge_glmnet/w6r6_grp_bollinger_glmnet"

# Regexes selecting the same groups out of the 812-column dataset, for CONCAT.
RE_ONEHOT='^feature_sym_(?!tenc$)'
RE_INTERACT='_x_|_accel_'
RE_TENC='^feature_sym_tenc$'
RE_PRICE='^feature_close_(avg|ewa)_'
RE_SIZE='^feature_market_cap_(avg|ewa)_'
RE_BOLL='^feature_bollinger_'

# Parallel arrays: several regexes contain "|" and would be torn apart by any
# pipe-delimited packing.
NAME=(); MEMBERS=(); WEIGHTS=(); CONCAT_RE=()
add() { NAME+=("$1"); MEMBERS+=("$2"); WEIGHTS+=("$3"); CONCAT_RE+=("$4"); }

add trio     "${M_ONEHOT} ${M_INTERACT} ${M_PRICE}" \
             "0.4,0.4,0.2" \
             "${RE_ONEHOT}|${RE_INTERACT}|${RE_PRICE}"
add quintet  "${M_ONEHOT} ${M_TENC} ${M_INTERACT} ${M_PRICE} ${M_BOLL}" \
             "0.2,0.2,0.2,0.2,0.2" \
             "${RE_ONEHOT}|${RE_TENC}|${RE_INTERACT}|${RE_PRICE}|${RE_BOLL}"
add quintet2 "${M_ONEHOT} ${M_TENC} ${M_INTERACT} ${M_SIZE} ${M_PRICE}" \
             "0.2,0.2,0.2,0.2,0.2" \
             "${RE_ONEHOT}|${RE_TENC}|${RE_INTERACT}|${RE_SIZE}|${RE_PRICE}"

started_all=$(date +%s)
failed=0

for i in "${!NAME[@]}"; do
    name="${NAME[$i]}"; members="${MEMBERS[$i]}"
    weights="${WEIGHTS[$i]}"; pattern="${CONCAT_RE[$i]}"

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue

    echo
    echo "=============================================================="
    echo "${name}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    # --- BLEND: rank-average the existing member runs -----------------------
    blend_cmd=(python3 -u -m src.run_ensemble crypto "${BLD_MODEL}"
               ${members} --weights "${weights}" --run-name "w6r6_blend_${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '  blend : %s\n' "${blend_cmd[*]}"
    else
        "${blend_cmd[@]}" 2>&1 | tee "logs/blend_${name}.log"
        [ "${PIPESTATUS[0]}" -ne 0 ] && { failed=$((failed+1)); echo ">>> ${name} BLEND FAILED"; }
    fi

    [ "${SKIP_CONCAT:-0}" = "1" ] && continue

    # --- CONCAT: one model on the union of the same columns -----------------
    concat_cmd=(python3 -u -m src.run_forecast crypto "${BLD_MODEL}"
                --task "${BLD_TASK}"
                --cv-scheme "${BLD_SCHEME}" --train-window "${BLD_TRAIN_WINDOW}"
                --refit-every "${BLD_REFIT}" --data-version "${DATA}"
                --feature-include "${pattern}"
                --n-iter "${BLD_N_ITER}" --n-splits "${BLD_N_SPLITS}"
                --n-jobs "${BLD_N_JOBS}" --seed "${BLD_SEED}"
                --run-name "w6r6_concat_${name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '  concat: %s\n' "${concat_cmd[*]}"
        continue
    fi

    started=$(date +%s)
    "${concat_cmd[@]}" 2>&1 | tee "logs/concat_${name}.log"
    status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); echo ">>> ${name} CONCAT FAILED (exit ${status})"
    else
        echo ">>> ${name} concat done in $(( $(date +%s) - started ))s"
    fi
done

echo
echo "=============================================================="
echo "finished in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo
echo "compare blend vs concat vs the references:"
cat <<'REPORT'
  python3 - <<'PY'
import json, glob, os
import pandas as pd, numpy as np
rows=[]
paths = sorted(glob.glob("results/metrics/crypto/*/w6r6_blend_*.json")) \
      + sorted(glob.glob("results/metrics/crypto/*/w6r6_concat_*.json")) \
      + ["results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg.json",
         "results/metrics/crypto/ridge/w6r6_fixed_reg.json"]
for p in paths:
    if not os.path.exists(p): continue
    n=os.path.basename(p)[:-5]; d=json.load(open(p))
    s=pd.read_csv("results/per_period/crypto/%s/%s.csv" % (p.split("/")[3], n))
    pay=0.1*s["corr"]+s["mmc"]; cs=pay.cumsum()
    rows.append((pay.mean()/pay.std(ddof=0), n, d["corr"]["mean"], pay.mean(),
                 (cs.expanding(1).max()-cs).max()))
print("%-26s %10s %10s %9s %9s" % ("run","CORR","payout","p.shp","p.dd"))
for sh,n,c,m,dd in sorted(rows, reverse=True):
    print("%-26s %10.6f %10.6f %9.4f %9.4f" % (n,c,m,sh,dd))
PY
REPORT
echo "=============================================================="
