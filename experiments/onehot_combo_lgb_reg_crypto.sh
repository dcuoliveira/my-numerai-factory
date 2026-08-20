#!/usr/bin/env bash
#
# Experiment: the one-hot block combined with every other block, at leaf 50
#
# Follow-up to the leaf-size study, which established two things this one builds
# on: at min_child_samples=313 the 740 ticker dummies are INERT (no split can
# leave 313 rows on both sides when a ticker holds at most 130 rows in a 6m
# block), and at leaf 50 they are live and score best of the sizes tried
# (onehot +0.008093 payout vs raw at the same leaf, t = +1.58).
#
# So: does one-hot combine with anything?
#
#     arm                 dataset                                            feat
#     ---------------------------------------------------------------------------
#     raw                 crypto/v2.0                                          22   control
#     interact            crypto/v2.0-interact                                 36   control
#     onehot              crypto/v2.0-onehot                                  762   +raw
#     onehot_interact     crypto/v2.0-onehot-interact                         776   NEW
#     allfeat             crypto/v2.0-tenc-onehot-spread-interact             788
#     allfeatpp           crypto/v2.0-tenc-onehot-rollsel-spread-interact     812
#     allfeatppp          ...-spread-interact-fenccausal                      834
#
# NOTE the last three ALREADY CONTAIN all 740 one-hot columns -- verified by
# column comparison, not by name. "onehot+allfeat" is allfeat. What is new for
# them here is the leaf size: they have only ever run at 313, where the one-hot
# columns inside them were silently discarded. At leaf 50 those blocks are live
# for the first time, which is the whole point of re-running them.
#
# Everything else is the pinned 6m/6m configuration with min_child_samples=50:
#     n_estimators 4327  learning_rate 0.002572  max_depth 8  num_leaves 255
#     colsample_bytree 0.070213  subsample 0.816153 (freq 1)  reg_lambda 1.757511
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Baseline is raw AT LEAF 50, not the pinned 313 baseline -- comparing across
# leaf sizes would confound the columns with the regularisation.
#
# The prior, from every arm run so far, is dilution: colsample_bytree=0.0702
# draws ~54 of 776 columns, so the 14 interact columns that won at 36 features
# (t = +2.99) appear in ~1.8% of trees here. interact lost its edge inside
# allfeat (788) for exactly this reason, and each block added to allfeat has
# cost ~0.009 payout monotonically. If onehot_interact tracks onehot rather
# than interact, that is the mechanism confirmed a third time.
#
# The interesting alternative: one-hot identity may let a tree condition the
# interaction terms per token, which is a genuine interaction a linear model
# cannot express. If onehot_interact beats BOTH its parents at |t| > 2, that is
# a real finding and the first combination result on this panel.
#
# On ~26 effective independent windows |t| < 2 remains the expected outcome.
#
# Usage:
#   bash experiments/onehot_combo_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/onehot_combo_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/onehot_combo_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

OHC_LEAF="${OHC_LEAF:-50}"
OHC_WINDOW="${OHC_WINDOW:-6month}"
OHC_REFIT="${OHC_REFIT:-6month}"
OHC_N_SPLITS="${OHC_N_SPLITS:-2}"
OHC_N_JOBS="${OHC_N_JOBS:-8}"
OHC_SEED="${OHC_SEED:-42}"
case "${OHC_WINDOW}" in ""|0|[0-9]) echo "OHC_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

PARAMS="{\"n_estimators\":4327,\"learning_rate\":0.002572,\"max_depth\":8,\"num_leaves\":255,\"colsample_bytree\":0.070213,\"subsample\":0.816153,\"subsample_freq\":1,\"reg_lambda\":1.757511,\"min_child_samples\":${OHC_LEAF}}"

ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }
add raw             "crypto/v2.0"                                                    22
add interact        "crypto/v2.0-interact"                                           36
add onehot          "crypto/v2.0-onehot"                                            762
add onehot_interact "crypto/v2.0-onehot-interact"                                   776
add allfeat         "crypto/v2.0-tenc-onehot-spread-interact"                       788
add allfeatpp       "crypto/v2.0-tenc-onehot-rollsel-spread-interact"               812
add allfeatppp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact-fenccausal"    834

python3 - "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,w in zip(versions,wants):
    p="data/%s/train.parquet"%v
    if not os.path.exists(p): bad.append("%s MISSING"%v); continue
    g=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if g!=w: bad.append("%s has %d features, expected %d"%(v,g,w))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d datasets, feature counts match"%len(versions))
PY

total="${#ARM[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
echo "=============================================================="
echo "one-hot combinations at min_child_samples=${OHC_LEAF}  |  ${total} arms"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; data="${DATA[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    run_name="ohc_lgb_reg_${arm}_m${OHC_LEAF}"
    cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
         --cv-scheme rolling --train-window "${OHC_WINDOW}" --refit-every "${OHC_REFIT}"
         --data-version "${data}" --fixed-params "${PARAMS}"
         --n-splits "${OHC_N_SPLITS}" --n-jobs "${OHC_N_JOBS}" --era-step 1
         --seed "${OHC_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-34s %-52s %s feat\n' "$index" "$total" "$run_name" "$data" "${NFEAT[$i]}"; continue; fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${data}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; elapsed=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((elapsed/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((elapsed/60))m$((elapsed%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/ohc_report_m${OHC_LEAF}.txt"
{
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${OHC_LEAF}" <<'PY'
import json, glob, os, sys
leaf=sys.argv[1]; rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb/ohc_lgb_reg_*_m%s.json"%leaf)):
    d=json.load(open(p)); c=d["config"]
    rows.append((0.1*d["corr"]["mean"]+d["mmc"]["mean"],
                 os.path.basename(p)[:-5].replace("ohc_lgb_reg_","").replace("_m%s"%leaf,""),
                 c["n_features"], d["corr"]["mean"], d["mmc"]["mean"],
                 d["corr"]["sharpe"], d["corr"]["max_drawdown"]))
print("%-18s %5s %10s %10s %8s %8s %10s"%("arm","feat","CORR","MMC","sharpe","maxdd","payout"))
print("-"*74)
for pay,n,f,corr,mmc,sh,dd in sorted(rows,reverse=True):
    print("%-18s %5d %10.6f %10.6f %8.4f %8.4f %10.6f"%(n,f,corr,mmc,sh,dd,pay))
print("-"*74)
PY
echo
echo "paired vs raw at the SAME leaf size (Newey-West lag 20):"
S=.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py
A=$(ls results/metrics/crypto/lgb/ohc_lgb_reg_*_m${OHC_LEAF}.json 2>/dev/null \
    | sed 's|.*/||;s|\.json||' | grep -v "^ohc_lgb_reg_raw_m")
[ -f "$S" ] && python3 "$S" --model lgb --baseline "ohc_lgb_reg_raw_m${OHC_LEAF}" --arms $A 2>&1 \
    | grep -E "vs |payout|verdict"
echo
echo "and onehot_interact against BOTH its parents:"
[ -f "$S" ] && python3 "$S" --model lgb --baseline "ohc_lgb_reg_onehot_m${OHC_LEAF}" \
    --arms "ohc_lgb_reg_onehot_interact_m${OHC_LEAF}" 2>&1 | grep -E "vs |payout|verdict"
[ -f "$S" ] && python3 "$S" --model lgb --baseline "ohc_lgb_reg_interact_m${OHC_LEAF}" \
    --arms "ohc_lgb_reg_onehot_interact_m${OHC_LEAF}" 2>&1 | grep -E "vs |payout|verdict"
echo "=============================================================="
} 2>&1 | tee "${REPORT}"

# Mail on success AND failure -- a job that only mails when it breaks is
# indistinguishable from a job that silently stopped running. Never let SMTP
# break the run: report the mail failure and exit on the experiment's status.
if [ "${NO_MAIL:-0}" != "1" ]; then
    STATUS=$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)
    python3 - "${STATUS}" "${REPORT}" "${OHC_LEAF}" <<'PY' || echo "note: email failed, experiment status unaffected"
import sys
from src.live.notify import send_text
status, path, leaf = sys.argv[1], sys.argv[2], sys.argv[3]
send_text("[crypto] one-hot combinations leaf=%s -- %s" % (leaf, status), open(path).read())
print("emailed the report")
PY
fi

exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
