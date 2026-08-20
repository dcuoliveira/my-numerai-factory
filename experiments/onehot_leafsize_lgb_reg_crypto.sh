#!/usr/bin/env bash
#
# Side experiment: can LightGBM use the ticker one-hot block at all?
#
# WHY THIS EXISTS. In the feature-block study every one-hot arm came back
# BYTE-IDENTICAL to its non-one-hot twin:
#
#     fe_lgb_reg_onehot      == fe_lgb_reg_raw      (max|diff| = 0.00e+00)
#     fe_lgb_reg_tenc_onehot == fe_lgb_reg_tenc     (max|diff| = 0.00e+00)
#
# Not a null result -- a dead block. In a 6-month training window each ticker
# holds at most 130 rows (one per date; median 103 over 468 tickers), while the
# pinned config carries min_child_samples=313. No split on a ticker dummy can
# ever leave 313 rows on both sides, so LightGBM's feature_pre_filter discards
# all 740 columns at Dataset construction and the trees are literally the same.
#
# So the earlier arms measured nothing about one-hot. This one lowers the leaf
# floor below the per-ticker row count and asks the question properly.
#
# ---------------------------------------------------------------------------
# Design: a 4 x 5 cross, because the control has to move too
# ---------------------------------------------------------------------------
#
#     min_child_samples   20   50   80   99      (all < 100, so the median
#                                                 ticker at 103 rows is usable)
#     dataset             raw 22              <- CONTROL, same leaf size
#                         onehot 762
#                         tenc_onehot 763
#                         onehot_rollsel 786
#                         tenc_onehot_rollsel 787
#
# The raw control is run AT EVERY LEAF SIZE on purpose. Comparing a one-hot arm
# at min_child_samples=20 against the raw baseline at 313 would confound two
# changes and credit the columns with whatever the leaf size did. Every paired
# test below is onehot-vs-raw AT MATCHED LEAF SIZE.
#
# Everything else is the pinned 6m/6m configuration, unchanged:
#     n_estimators 4327  learning_rate 0.002572  max_depth 8  num_leaves 255
#     colsample_bytree 0.070213  subsample 0.816153 (freq 1)  reg_lambda 1.757511
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# FIRST, a wiring check that is not optional: at these leaf sizes a one-hot arm
# MUST differ from its raw control. If any pair is still identical the block is
# still inert and its row is not a measurement. The report asserts this.
#
# Then the actual question, in two parts:
#
#   * does lowering the leaf floor help AT ALL (raw at 20/50/80/99 vs the
#     pinned raw at 313, payout 0.083272)? A tree on 39,000 rows with 255
#     leaves and a floor of 20 is far less regularised, and this panel has
#     punished under-regularisation before.
#   * conditional on that, do the 740 ticker dummies add anything over raw at
#     the SAME leaf size?
#
# The prior is not encouraging on either count. Ridge's one-hot arm cost 0.0075
# CORR alone, and onehot+rollsel doubled max drawdown (1.465 vs 0.805) while
# raising meta-model correlation to 0.586. On ~26 effective independent windows
# |t| < 2 remains the expected outcome and a null IS the finding.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/onehot_leafsize_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/onehot_leafsize_lgb_reg_crypto.sh
#   LEAVES="20 99" bash experiments/onehot_leafsize_lgb_reg_crypto.sh
#   ONLY="raw_m20 onehot_m20" bash experiments/onehot_leafsize_lgb_reg_crypto.sh
#
# Under screen:
#   screen -L -Logfile ohl-lgb-reg.txt bash experiments/onehot_leafsize_lgb_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# OHL_-prefixed on purpose. GNU screen exports $WINDOW=0 into every session it
# starts, so a bare WINDOW="${WINDOW:-6month}" silently becomes 0.
OHL_MODEL="${OHL_MODEL:-lgb}"
OHL_TASK="${OHL_TASK:-regression}"
OHL_SCHEME="${OHL_SCHEME:-rolling}"
OHL_WINDOW="${OHL_WINDOW:-6month}"
OHL_REFIT="${OHL_REFIT:-6month}"
OHL_METRIC="${OHL_METRIC:-corr}"
OHL_N_SPLITS="${OHL_N_SPLITS:-2}"
OHL_N_JOBS="${OHL_N_JOBS:-8}"
OHL_ERA_STEP="${OHL_ERA_STEP:-1}"
OHL_SEED="${OHL_SEED:-42}"

case "${OHL_WINDOW}" in
    ""|0|[0-9]) echo "OHL_WINDOW=${OHL_WINDOW} looks wrong (screen exports \$WINDOW=0)" >&2; exit 1 ;;
esac

LEAVES="${LEAVES:-20 50 80 99}"

# Everything except min_child_samples, which is substituted per arm.
BASE_PARAMS='"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"reg_lambda":1.757511'

DS_NAME=(); DS_PATH=(); DS_FEAT=()
adddata() { DS_NAME+=("$1"); DS_PATH+=("$2"); DS_FEAT+=("$3"); }
adddata raw                 "crypto/v2.0"                          22
adddata onehot              "crypto/v2.0-onehot"                  762
adddata tenc_onehot         "crypto/v2.0-tenc-onehot"             763
adddata onehot_rollsel      "crypto/v2.0-onehot-rollsel"          786
adddata tenc_onehot_rollsel "crypto/v2.0-tenc-onehot-rollsel"     787

# Gate on feature counts, and on the premise itself: report the per-ticker row
# count so the chosen leaf sizes are visibly below it rather than assumed to be.
python3 - "${OHL_WINDOW}" "${LEAVES}" "${DS_PATH[@]}" -- "${DS_FEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq, pandas as pd
window, leaves = sys.argv[1], [int(x) for x in sys.argv[2].split()]
cut = sys.argv.index("--")
versions, wants = sys.argv[3:cut], [int(x) for x in sys.argv[cut+1:]]
bad = []
for v, want in zip(versions, wants):
    p = "data/%s/train.parquet" % v
    if not os.path.exists(p):
        bad.append("%s MISSING (bash setup.sh --with-derived)" % v); continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (v, got, want))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
df = pq.read_table("data/crypto/v2.0/train.parquet", columns=["date","symbol"]).to_pandas()
dates = sorted(df["date"].unique())[-130:]
per = df[df["date"].isin(dates)].groupby("symbol").size()
print("preflight OK: %d datasets, feature counts match" % len(versions))
print("  per-ticker rows in a %s block: median %d, max %d over %d tickers"
      % (window, per.median(), per.max(), per.size))
print("  leaf sizes to test: %s -- usable tickers at each:" % leaves)
for L in leaves:
    print("     min_child_samples=%-4d %d of %d tickers have >= %d rows"
          % (L, int((per >= L).sum()), per.size, L))
if min(leaves) >= per.median():
    sys.exit("every leaf size is above the median ticker row count -- the block "
             "would stay inert, which is the bug this study exists to avoid")
PY

ARM=(); ADATA=(); ALEAF=()
for L in ${LEAVES}; do
    for i in "${!DS_NAME[@]}"; do
        ARM+=("${DS_NAME[$i]}_m${L}"); ADATA+=("${DS_PATH[$i]}"); ALEAF+=("${L}")
    done
done

total="${#ARM[@]}"
index=0; ran=0; failed=0; skipped=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

echo "=============================================================="
echo "one-hot x leaf-size cross  |  ${total} arms  (${OHL_WINDOW}/${OHL_REFIT})"
echo "leaf sizes: ${LEAVES}   (pinned study used 313, which made one-hot inert)"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; data="${ADATA[$i]}"; leaf="${ALEAF[$i]}"
    index=$((index + 1))
    if [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi
    if [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi

    run_name="ohl_lgb_reg_${arm}"
    params="{${BASE_PARAMS},\"min_child_samples\":${leaf}}"
    cmd=(python3 -u -m src.run_forecast crypto "${OHL_MODEL}"
         --task "${OHL_TASK}" --metric "${OHL_METRIC}"
         --cv-scheme "${OHL_SCHEME}" --train-window "${OHL_WINDOW}"
         --refit-every "${OHL_REFIT}" --data-version "${data}"
         --fixed-params "${params}"
         --n-splits "${OHL_N_SPLITS}" --n-jobs "${OHL_N_JOBS}"
         --era-step "${OHL_ERA_STEP}" --seed "${OHL_SEED}"
         --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-34s %-46s leaf=%s\n' "$index" "$total" "$run_name" "$data" "$leaf"
        continue
    fi

    ran=$((ran + 1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  leaf=${leaf}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((elapsed/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((elapsed/60))m$((elapsed%60))s"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && exit 0

echo
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s) - started_all) / 60 )) min"
echo "(${skipped} skipped, ${failed} failed)"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo

python3 - <<'PY'
import json, glob, os, re
import pandas as pd

rows = []
for p in sorted(glob.glob("results/metrics/crypto/lgb/ohl_lgb_reg_*.json")):
    d = json.load(open(p)); c = d["config"]; name = os.path.basename(p)[:-5]
    stem = name.replace("ohl_lgb_reg_", "")
    m = re.match(r"(.+)_m(\d+)$", stem)
    ds, leaf = (m.group(1), int(m.group(2))) if m else (stem, -1)
    rows.append(dict(run=name, ds=ds, leaf=leaf, feat=c["n_features"],
                     corr=d["corr"]["mean"], mmc=d["mmc"]["mean"],
                     sh=d["corr"]["sharpe"], dd=d["corr"]["max_drawdown"],
                     pay=0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
if not rows:
    print("no results"); raise SystemExit
t = pd.DataFrame(rows)

print("%-22s %5s %5s %10s %8s %8s %10s" % ("dataset","leaf","feat","CORR","sharpe","maxdd","payout"))
print("-"*74)
for _, r in t.sort_values("pay", ascending=False).iterrows():
    print("%-22s %5d %5d %10.6f %8.4f %8.4f %10.6f"
          % (r["ds"], r["leaf"], r["feat"], r["corr"], r["sh"], r["dd"], r["pay"]))
print("-"*74)

# THE WIRING CHECK: at these leaf sizes a one-hot arm must differ from its raw control.
print("\ninertness check -- a one-hot arm identical to raw is NOT a measurement:")
F = "results/forecast/crypto/lgb/"
alive = dead = 0
for leaf in sorted(t["leaf"].unique()):
    rawp = F + "ohl_lgb_reg_raw_m%d.csv" % leaf
    if not os.path.exists(rawp): continue
    base = pd.read_csv(rawp)["prediction"]
    for ds in ("onehot", "tenc_onehot", "onehot_rollsel", "tenc_onehot_rollsel"):
        p = F + "ohl_lgb_reg_%s_m%d.csv" % (ds, leaf)
        if not os.path.exists(p): continue
        diff = (pd.read_csv(p)["prediction"] - base).abs().max()
        state = "INERT (identical to raw)" if diff == 0 else "uses the block"
        alive += diff > 0; dead += diff == 0
        print("   leaf %-4d %-22s max|diff| vs raw = %.3e   %s" % (leaf, ds, diff, state))
print("   -> %d arms use the one-hot block, %d still inert" % (alive, dead))

# Does lowering the floor help at all, against the 313 baseline?
ref = "results/metrics/crypto/lgb/fe_lgb_reg_raw.json"
if os.path.exists(ref):
    d = json.load(open(ref)); base_pay = 0.1*d["corr"]["mean"]+d["mmc"]["mean"]
    print("\nraw at each leaf size vs raw at min_child_samples=313 (payout %.6f):" % base_pay)
    for _, r in t[t["ds"] == "raw"].sort_values("leaf").iterrows():
        print("   leaf %-4d payout %.6f  (%+.6f)" % (r["leaf"], r["pay"], r["pay"] - base_pay))
PY

echo
echo "paired: each one-hot arm vs the RAW ARM AT THE SAME LEAF SIZE (NW lag 20)"
SCRIPT_PD=.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py
if [ -f "${SCRIPT_PD}" ]; then
    for L in ${LEAVES}; do
        A=$(ls results/metrics/crypto/lgb/ohl_lgb_reg_*_m${L}.json 2>/dev/null \
            | sed 's|.*/||;s|\.json||' | grep -v "^ohl_lgb_reg_raw_m" | tr '\n' ' ')
        [ -z "${A}" ] && continue
        echo; echo "--- leaf size ${L} ---"
        python3 "${SCRIPT_PD}" --model lgb --baseline "ohl_lgb_reg_raw_m${L}" \
            --arms ${A} 2>&1 | grep -E "vs |payout|verdict"
    done
else
    echo "  (paired_delta.py not found)"
fi
echo "=============================================================="

exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
