#!/usr/bin/env bash
#
# Experiment: every deployed model, refit from scratch on the 60-day target
#
# Same datasets, same protocol, same seeds. One thing changes: the label.
#
#     target_binned_return_20   what all six CURUPIRA slots predict today
#     target_binned_return_60   the other label crypto/v2.0 ships, never tested
#
# Why this and not another feature set: the 60-day target correlates 0.52 with
# the 20-day one. Nothing else in this project comes close to that little
# overlap -- LightGBM measured 0.62 against the deployed configuration, and the
# whole six-slot book spans 0.61-0.998 internally. After nine flat feature arms
# and seven weighting schemes that all lost to 1/N, the target is the largest
# untried lever left.
#
# ---------------------------------------------------------------------------
# The nine models
# ---------------------------------------------------------------------------
#
# Four deployed slots, plus the five members CURUPIRA_05 blends. Running the
# members separately means the blend can be reconstructed at 1/N afterwards
# without refitting anything, so the tenth arm is free:
#
#     c01_tenc        ridge_glmnet   23   CURUPIRA_01
#     c02_raw         ridge          22   CURUPIRA_02
#     c03_onehotroll  ridge_glmnet  786   CURUPIRA_03
#     c04_allfeatpp   ridge_glmnet  812   CURUPIRA_04
#     m_onehot        ridge_glmnet  740   CURUPIRA_05 member
#     m_tenc          ridge_glmnet    1   CURUPIRA_05 member
#     m_interact      ridge_glmnet   14   CURUPIRA_05 member
#     m_size          ridge           4   CURUPIRA_05 member
#     m_price         ridge_glmnet    4   CURUPIRA_05 member
#
# ---------------------------------------------------------------------------
# What changes automatically, and what does not
# ---------------------------------------------------------------------------
#
# CHANGES BY ITSELF. The embargo is derived from the target's horizon suffix, so
# --embargo auto gives 60 periods here rather than 20. That is not cosmetic: a
# 60-day target on daily dates overlaps 98% between consecutive periods, and
# keeping the 20-period embargo would leak three times over.
#
# ALSO CHANGES. The 60-day target is NaN over the last 40 dates -- 12,000 rows
# where the window has not resolved. The loader now drops on the target BEING
# MODELLED rather than the tournament default, so those rows go. Before that fix
# they would have been handed to the estimator as NaN.
#
# DOES NOT CHANGE, deliberately. The datasets are used exactly as they are, per
# the brief. That means feature_sym_tenc still encodes the 20-day target with a
# 20-period lag, in c01_tenc, c04_allfeatpp and m_tenc. This is causally sound --
# it reads only resolved 20-day outcomes from at least 20 periods back, so
# nothing about the 60-day future leaks -- but it is encoding a related rather
# than the same quantity. A stricter version would rebuild the encoding from the
# 60-day target at a 60-period lag; that is a different experiment and it is not
# this one.
#
# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------
#
# The scores are NOT comparable with the 20-day arms as levels. A different
# label has a different achievable correlation, and MMC is computed against a
# meta model built for the 20-day target, so the reference is not aligned
# either. Two things are comparable and worth looking at:
#
#   * the RANKING of the nine models within this experiment, against the same
#     ranking on the 20-day target. If the ordering holds, the model choices
#     generalise across horizons; if it inverts, they were fitted to one label.
#   * the CORRELATION between a 60-day arm and its 20-day twin, which is the
#     real question. A pair correlating ~0.5 would be the most diversifying
#     thing available -- the whole book currently sits at 0.61-0.998.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/target60_crypto.sh
#   DRY_RUN=1 bash experiments/target60_crypto.sh
#   ONLY=c01_tenc bash experiments/target60_crypto.sh
#   SHARD=1 OF=3 bash experiments/target60_crypto.sh
#
# Under screen:
#   screen -L -Logfile target60.txt bash experiments/target60_crypto.sh
#
# ~35-50 min per arm on this box, nine arms, so about 6 hours single-threaded
# or 2 hours across three machines.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# T60_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
T60_TARGET="${T60_TARGET:-target_binned_return_60}"
T60_TASK="${T60_TASK:-regression}"
T60_SCHEME="${T60_SCHEME:-rolling}"
T60_TRAIN_WINDOW="${T60_TRAIN_WINDOW:-6month}"
T60_REFIT="${T60_REFIT:-6month}"
T60_N_ITER="${T60_N_ITER:-50}"
T60_N_SPLITS="${T60_N_SPLITS:-5}"
T60_N_JOBS="${T60_N_JOBS:-8}"
T60_SEED="${T60_SEED:-42}"

case "${T60_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "T60_TRAIN_WINDOW=${T60_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 6month" >&2; exit 1 ;;
esac

ALL="crypto/v2.0-tenc-onehot-rollsel-spread-interact"

# The one-hot pattern needs its negative lookahead: feature_sym_tenc shares the
# prefix, and a naive ^feature_sym_ would drag the target encoding into the
# identity block. The two raw groups are anchored on (20d|60d)$ because the
# unanchored pattern also matches their _spread columns in the 812-set.
RE_ONEHOT='^feature_sym_(?!tenc$)'
RE_TENC='^feature_sym_tenc$'
RE_INTERACT='_x_|_accel_'
RE_SIZE='^feature_market_cap_(avg|ewa)_(20d|60d)$'
RE_PRICE='^feature_close_(avg|ewa)_(20d|60d)$'

# Parallel arrays: several regexes contain "|", which any pipe-delimited
# packing would tear apart.
NAME=(); MODEL=(); DATA=(); INCLUDE=()
add() { NAME+=("$1"); MODEL+=("$2"); DATA+=("$3"); INCLUDE+=("$4"); }

# --- the four deployed single-model slots -------------------------------------
add c01_tenc       ridge_glmnet "crypto/v2.0-tenc"            "-"
add c02_raw        ridge        "crypto/v2.0"                 "-"
add c03_onehotroll ridge_glmnet "crypto/v2.0-onehot-rollsel"  "-"
add c04_allfeatpp  ridge_glmnet "${ALL}"                      "-"

# --- the five members CURUPIRA_05 blends ---------------------------------------
add m_onehot       ridge_glmnet "${ALL}"          "${RE_ONEHOT}"
add m_tenc         ridge_glmnet "${ALL}"          "${RE_TENC}"
add m_interact     ridge_glmnet "${ALL}"          "${RE_INTERACT}"
add m_size         ridge        "${ALL}"          "${RE_SIZE}"
add m_price        ridge_glmnet "${ALL}"          "${RE_PRICE}"

total="${#NAME[@]}"
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "target sweep | ${total} models on ${T60_TARGET}"
echo "window=${T60_TRAIN_WINDOW} refit=${T60_REFIT} embargo=auto (derived: 60)"
echo "shard ${SHARD}/${OF}"
echo "=============================================================="

for i in "${!NAME[@]}"; do
    name="${NAME[$i]}"; model="${MODEL[$i]}"
    data="${DATA[$i]}"; include="${INCLUDE[$i]}"
    index=$((index + 1))

    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue

    run_name="t60_${name}"
    cmd=(python3 -u -m src.run_forecast crypto "${model}"
         --task "${T60_TASK}" --target "${T60_TARGET}"
         --cv-scheme "${T60_SCHEME}" --train-window "${T60_TRAIN_WINDOW}"
         --refit-every "${T60_REFIT}" --embargo auto
         --data-version "${data}"
         --n-iter "${T60_N_ITER}" --n-splits "${T60_N_SPLITS}"
         --n-jobs "${T60_N_JOBS}" --seed "${T60_SEED}"
         --run-name "${run_name}")
    [ "${include}" != "-" ] && cmd+=(--feature-include "${include}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-16s %s\n' "$index" "$total" "$name" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${run_name}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # A failed arm must not cost you the others.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${run_name} FAILED (exit ${status}) after $(( $(date +%s) - started ))s -- continuing"
    else
        echo ">>> ${run_name} done in $(( $(date +%s) - started ))s"
    fi
done

# The 1/N blend of the five members, reconstructed for free from the arms above.
if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${ONLY:-}" ] && [ "${OF}" -eq 1 ]; then
    if ls results/forecast/crypto/*/t60_m_*.csv >/dev/null 2>&1; then
        echo
        echo "=============================================================="
        echo "reconstructing the CURUPIRA_05 blend at 1/N on ${T60_TARGET}"
        echo "=============================================================="
        python3 -u -m src.run_ensemble crypto ridge_glmnet \
            ridge_glmnet/t60_m_onehot ridge_glmnet/t60_m_tenc \
            ridge_glmnet/t60_m_interact ridge/t60_m_size \
            ridge_glmnet/t60_m_price \
            --run-name t60_c05_blend 2>&1 | tee logs/t60_c05_blend.log
    fi
fi

echo
echo "=============================================================="
echo "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"
echo
cat <<'REPORT'
compare within the 60-day experiment, and against each arm's 20-day twin:

  python3 - <<'PY'
import json, glob, os
import numpy as np, pandas as pd
TWIN = {"t60_c01_tenc":("ridge_glmnet","w6r6_glmnet_reg_tenc"),
        "t60_c02_raw":("ridge","w6r6_fixed_reg"),
        "t60_c03_onehotroll":("ridge_glmnet","w6r6_glmnet_reg_onehot_rollsel"),
        "t60_c04_allfeatpp":("ridge_glmnet","w6r6_glmnet_reg_allfeatpp"),
        "t60_m_onehot":("ridge_glmnet","w6r6_alone_onehot"),
        "t60_m_tenc":("ridge_glmnet","w6r6_alone_tenc"),
        "t60_m_interact":("ridge_glmnet","w6r6_alone_interact"),
        "t60_m_size":("ridge","w6r6_grp_size_fixed"),
        "t60_m_price":("ridge_glmnet","w6r6_grp_price_glmnet"),
        "t60_c05_blend":("ridge_glmnet","w6r6_blend_quintet2")}
print("%-20s %10s %10s %10s %9s   %s" % ("run","CORR","MMC","payout","p.shp","rho vs its 20d twin"))
for p in sorted(glob.glob("results/metrics/crypto/*/t60_*.json")):
    n=os.path.basename(p)[:-5]; d=json.load(open(p)); m=p.split("/")[3]
    s=pd.read_csv("results/per_period/crypto/%s/%s.csv" % (m,n))
    pay=0.1*s["corr"]+s["mmc"]
    rho=float("nan")
    if n in TWIN:
        tm,tr = TWIN[n]
        f60="results/forecast/crypto/%s/%s.csv" % (m,n)
        f20="results/forecast/crypto/%s/%s.csv" % (tm,tr)
        if os.path.exists(f20):
            a=pd.read_csv(f60,usecols=["symbol","date","prediction"])
            b=pd.read_csv(f20,usecols=["symbol","date","prediction"])
            j=a.merge(b,on=["symbol","date"],suffixes=("_60","_20"))
            if len(j):
                r=j.groupby("date")[["prediction_60","prediction_20"]].rank(pct=True)
                rho=r["prediction_60"].corr(r["prediction_20"], method="spearman")
    print("%-20s %10.6f %10.6f %10.6f %9.4f   %s" % (
        n, d["corr"]["mean"], d["mmc"]["mean"], pay.mean(),
        pay.mean()/pay.std(ddof=0),
        "%.3f" % rho if rho==rho else "-"))
PY
REPORT
echo "=============================================================="
