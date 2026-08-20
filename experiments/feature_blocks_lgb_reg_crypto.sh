#!/usr/bin/env bash
#
# Experiment: every feature block, on LightGBM regression
#
# The ridge feature-engineering study, re-run with a different model class. Same
# twelve datasets, same protocol, same baseline logic -- the estimator is the
# only thing that changed:
#
#     block                 dataset                                          feat
#     ---------------------------------------------------------------------------
#     raw                   crypto/v2.0                                        22
#     tenc                  crypto/v2.0-tenc                                   23
#     spread                crypto/v2.0-spread                                 33
#     interact              crypto/v2.0-interact                               36
#     fenccausal            crypto/v2.0-fenccausal                             44
#     fencglobal            crypto/v2.0-fencglobal                             44
#     onehot                crypto/v2.0-onehot                                762
#     tenc_onehot           crypto/v2.0-tenc-onehot                           763
#     onehot_rollsel        crypto/v2.0-onehot-rollsel                        786
#     tenc_onehot_rollsel   crypto/v2.0-tenc-onehot-rollsel                   787
#     allfeat               crypto/v2.0-tenc-onehot-spread-interact           788
#     allfeatpp             crypto/v2.0-tenc-onehot-rollsel-spread-interact   812
#
# THE MODEL IS PINNED (--fixed-params) to the configuration w6r6_lgb_reg's search
# converged on, at the 6m window / 6m refit cadence the cadence cross confirmed:
#
#     n_estimators 4327  learning_rate 0.002572  max_depth 8  num_leaves 255
#     colsample_bytree 0.070213  subsample 0.816153 (freq 1)
#     min_child_samples 313  reg_lambda 1.757511
#
# Pinning is justified by the cadence study: at 6m/6m the pinned config scored
# payout 0.083272 against the searched arm's 0.082061, in 1.7 min against 159.
# Fifty draws bought nothing detectable, and pinning is what makes a 12-arm
# study cost under an hour instead of ~32 of them.
#
# ---------------------------------------------------------------------------
# The caveat the wide arms carry, stated rather than buried
# ---------------------------------------------------------------------------
#
# That config was selected ON THE RAW 22 FEATURES. colsample_bytree=0.0702 is
# ~1.5 columns per tree on raw but ~57 on allfeat++, so the pinned model is not
# the same model once the width changes by 37x. The wide-arm results therefore
# answer "what do these columns do to THIS model" and not "what is the best
# model on these columns".
#
# The control is free and already on disk: w6r6_lgb_reg_allfeatpp is the same
# 812-column dataset with a full 50-draw search (payout 0.066020). Compare it
# against this study's fe_lgb_reg_allfeatpp to see whether the pin, rather than
# the columns, drove a wide-arm result.
#
# ---------------------------------------------------------------------------
# What this is measured against, and what would count as a result
# ---------------------------------------------------------------------------
#
# Baseline is fe_lgb_reg_raw, run inside this study so the comparison is
# self-contained. It is the same configuration as cad_lgb_reg_w6m_r6m and MUST
# reproduce it (payout 0.083272); the runner checks that and says so.
#
# The ridge prior is unambiguous and is the thing being retested: every block
# came out flat or worse, and the prediction-correlation heatmap put every ridge
# variant at 0.886-1.000 against the others. Eleven blocks have now been tried
# on this panel without one beating raw by a margin the sample supports.
#
# On ~26 effective independent windows, |t| < 2 is the EXPECTED outcome and a
# null is the finding, not a failure. What would change the conclusion is a
# block clearing |t| > 2 on payout in the paired test -- which would be the
# first in this study to do so, and would say the ridge nulls were a property
# of the linear model rather than of the features.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/feature_blocks_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/feature_blocks_lgb_reg_crypto.sh
#   ONLY="raw tenc" bash experiments/feature_blocks_lgb_reg_crypto.sh
#   SKIP=onehot bash experiments/feature_blocks_lgb_reg_crypto.sh
#   SHARD=1 OF=2 bash experiments/feature_blocks_lgb_reg_crypto.sh
#
# Under screen:
#   screen -L -Logfile fe-lgb-reg.txt bash experiments/feature_blocks_lgb_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# FE_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 and every training block comes out empty.
FE_MODEL="${FE_MODEL:-lgb}"
FE_TASK="${FE_TASK:-regression}"
FE_SCHEME="${FE_SCHEME:-rolling}"
FE_WINDOW="${FE_WINDOW:-6month}"
FE_REFIT="${FE_REFIT:-6month}"
FE_METRIC="${FE_METRIC:-corr}"
FE_N_SPLITS="${FE_N_SPLITS:-2}"       # vestigial: nothing is being selected
FE_N_JOBS="${FE_N_JOBS:-8}"
FE_ERA_STEP="${FE_ERA_STEP:-1}"
FE_SEED="${FE_SEED:-42}"

case "${FE_WINDOW}" in
    ""|0|[0-9]) echo "FE_WINDOW=${FE_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

FE_PARAMS="${FE_PARAMS:-{\"n_estimators\":4327,\"learning_rate\":0.002572,\"max_depth\":8,\"num_leaves\":255,\"colsample_bytree\":0.070213,\"subsample\":0.816153,\"subsample_freq\":1,\"min_child_samples\":313,\"reg_lambda\":1.757511}}"

# Parallel arrays: name / dataset / expected feature count. Never pipe-delimited
# records -- feature regexes elsewhere in this repo contain "|" as alternation.
ARM=(); DATA=(); NFEAT=()
add() { ARM+=("$1"); DATA+=("$2"); NFEAT+=("$3"); }

add raw                 "crypto/v2.0"                                        22
add tenc                "crypto/v2.0-tenc"                                   23
add spread              "crypto/v2.0-spread"                                 33
add interact            "crypto/v2.0-interact"                               36
add fenccausal          "crypto/v2.0-fenccausal"                             44
add fencglobal          "crypto/v2.0-fencglobal"                             44
add onehot              "crypto/v2.0-onehot"                                762
add tenc_onehot         "crypto/v2.0-tenc-onehot"                           763
add onehot_rollsel      "crypto/v2.0-onehot-rollsel"                        786
add tenc_onehot_rollsel "crypto/v2.0-tenc-onehot-rollsel"                   787
add allfeat             "crypto/v2.0-tenc-onehot-spread-interact"           788
add allfeatpp           "crypto/v2.0-tenc-onehot-rollsel-spread-interact"   812

# Gate on feature count BEFORE running anything. A silent count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
cut = sys.argv.index("--")
versions, wants = sys.argv[1:cut], [int(x) for x in sys.argv[cut+1:]]
bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        bad.append("%s is MISSING (build it: bash setup.sh --with-derived)" % version); continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))
if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d datasets, feature counts all match" % len(versions))
PY

total="${#ARM[@]}"
index=0; ran=0; failed=0; skipped=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "lgb ${FE_TASK} feature blocks  |  ${total} arms, model PINNED"
echo "cadence ${FE_WINDOW}/${FE_REFIT} ${FE_SCHEME}  seed=${FE_SEED}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  shard ${SHARD}/${OF}"
echo "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; data="${DATA[$i]}"
    index=$((index + 1))

    if [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi
    if [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi
    if [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ]; then
        skipped=$((skipped + 1)); continue; fi

    run_name="fe_lgb_reg_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${FE_MODEL}"
         --task "${FE_TASK}" --metric "${FE_METRIC}"
         --cv-scheme "${FE_SCHEME}" --train-window "${FE_WINDOW}"
         --refit-every "${FE_REFIT}" --data-version "${data}"
         --fixed-params "${FE_PARAMS}"
         --n-splits "${FE_N_SPLITS}" --n-jobs "${FE_N_JOBS}"
         --era-step "${FE_ERA_STEP}" --seed "${FE_SEED}"
         --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-30s %s\n' "$index" "$total" "$run_name" "$data"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${data}  $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # A failed arm must not cost you the other eleven.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((elapsed / 60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((elapsed / 60))m$((elapsed % 60))s"
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

# ---------------------------------------------------------------------------
# Reporting. Level AND risk, then the paired test -- a mean-only ranking on
# near-identical arms finds differences that a paired t cannot support.
# ---------------------------------------------------------------------------
python3 - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("results/metrics/crypto/lgb/fe_lgb_reg_*.json")):
    d = json.load(open(p)); c = d["config"]; name = os.path.basename(p)[:-5]
    rows.append((0.1 * d["corr"]["mean"] + d["mmc"]["mean"],
                 name.replace("fe_lgb_reg_", ""), c["n_features"],
                 d["corr"]["mean"], d["mmc"]["mean"], d["corr"]["sharpe"],
                 d["corr"]["max_drawdown"], d.get("runtime_minutes", 0)))
if not rows:
    print("no results yet"); raise SystemExit
print("%-22s %5s %10s %10s %8s %8s %10s %6s" % (
    "block", "feat", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 86)
for pay, n, f, corr, mmc, sh, dd, rt in sorted(rows, reverse=True):
    print("%-22s %5d %10.6f %10.6f %8.4f %8.4f %10.6f %6.1f" % (
        n, f, corr, mmc, sh, dd, pay, rt))
print("-" * 86)

# The baseline must reproduce the cadence study's identical configuration.
base = [r for r in rows if r[1] == "raw"]
ref = "results/metrics/crypto/lgb/cad_lgb_reg_w6m_r6m.json"
if base and os.path.exists(ref):
    d = json.load(open(ref)); expect = 0.1 * d["corr"]["mean"] + d["mmc"]["mean"]
    got = base[0][0]
    ok = abs(got - expect) < 1e-9
    print("baseline check: fe_lgb_reg_raw payout %.6f vs cad_lgb_reg_w6m_r6m %.6f -> %s"
          % (got, expect, "REPRODUCES" if ok else "*** MISMATCH: config differs ***"))
# The free searched control on the widest arm.
pin = [r for r in rows if r[1] == "allfeatpp"]
sp = "results/metrics/crypto/lgb/w6r6_lgb_reg_allfeatpp.json"
if pin and os.path.exists(sp):
    d = json.load(open(sp)); s = 0.1 * d["corr"]["mean"] + d["mmc"]["mean"]
    print("pin control  : allfeatpp pinned %.6f vs searched %.6f (delta %+.6f)"
          % (pin[0][0], s, pin[0][0] - s))
PY

echo
echo "paired vs fe_lgb_reg_raw (Newey-West lag 20):"
ARMS=$(ls results/metrics/crypto/lgb/fe_lgb_reg_*.json 2>/dev/null \
       | sed 's|.*/||;s|\.json||' | grep -v "_raw$" | tr '\n' ' ')
if [ -n "${ARMS}" ] && [ -f .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py ]; then
    python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
        --model lgb --baseline fe_lgb_reg_raw --arms ${ARMS} 2>&1 | tail -80
else
    echo "  (paired_delta.py not found -- run it manually)"
fi
echo "=============================================================="

exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
