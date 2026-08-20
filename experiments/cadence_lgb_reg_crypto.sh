#!/usr/bin/env bash
#
# Experiment: train window x refit cadence for LightGBM regression
#
# Full 4x4 cross of rolling train window against refit cadence:
#
#     window   3m  6m  9m  12m
#     refit    3m  6m  9m  12m      -> 16 arms
#
# THE MODEL IS HELD FIXED. Every arm uses the configuration the raw-feature
# search converged on (w6r6_lgb_reg chose it in 3 of its 4 refits):
#
#     n_estimators 4327   learning_rate 0.002572   max_depth 8   num_leaves 255
#     colsample_bytree 0.070213   subsample 0.816153 (freq 1)
#     min_child_samples 313   reg_lambda 1.757511
#
# so the only things varying are the two cadence dials. That is also what makes
# the study affordable: a searched arm costs ~158 min, a pinned one costs a
# couple of minutes, turning 16 arms from ~42 hours into ~1-2 hours.
#
# ---------------------------------------------------------------------------
# What this is measured against, and what would count as a result
# ---------------------------------------------------------------------------
#
# Baseline is the 6m/6m arm IN THIS STUDY (cad_lgb_reg_w6m_r6m), not the
# searched w6r6_lgb_reg -- that one differs by two things at once. Comparing the
# two afterwards is a free second result: what the 50-draw search was worth.
#
# The ridge cadence study is the prior. It varied the two axes ONE AT A TIME
# (windows at a 3m refit, refits at a 6m window) and found:
#
#     6m window / 6m refit   payout 0.085763  sharpe 1.5424   <- the CURUPIRA cadence
#     6m / 9m                       0.082083         1.2898
#     24m / 3m                      0.078512         1.2993
#     6m / 1m                       0.059835         1.1267
#     3m / 3m                       0.053558         1.0249
#
# Nobody has run the full cross, so whether 6m/6m is a genuine joint peak or an
# artifact of those two slices is untested. That is the question here.
#
# Two things worth watching, both specific to a booster:
#
#   * INTERACTION. If the best refit differs by window, the one-at-a-time ridge
#     design was hiding something.
#   * SHORT WINDOWS. Ridge collapsed at 3m (0.0536) because it had nothing left
#     to shrink. A tree on 19,500 rows with 255 leaves fails differently -- it
#     overfits rather than over-shrinks -- so it may degrade less, or worse.
#
# With ~26 effective independent windows on 510 periods, |t| < 2 is the expected
# outcome and a null IS the finding. Report the interval, not "inconclusive".
#
# ---------------------------------------------------------------------------
# A confound this runner does NOT remove, stated so it is not read as absent
# ---------------------------------------------------------------------------
#
# min_child_samples is pinned at 313 ABSOLUTE. As a fraction of the training
# block that is 1.6% at a 3m window but 0.4% at 12m, so "the same model" across
# windows also means progressively weaker leaf regularisation. Part of any
# window effect is that. The two SENS arms at the extremes hold it at a constant
# 0.80% of rows instead (156 at 3m, 626 at 12m); if they move the result
# materially, the window axis has to be read as confounded.
# Run them with:  SENS=1 bash experiments/cadence_lgb_reg_crypto.sh
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/cadence_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/cadence_lgb_reg_crypto.sh
#   ONLY="w6m_r6m w3m_r3m" bash experiments/cadence_lgb_reg_crypto.sh
#   SHARD=1 OF=2 bash experiments/cadence_lgb_reg_crypto.sh
#   SENS=1 bash experiments/cadence_lgb_reg_crypto.sh     # the 2 sensitivity arms
#
# Under screen:
#   screen -L -Logfile cadence-lgb-reg.txt bash experiments/cadence_lgb_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# CAD_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 and every training block comes out empty.
CAD_MODEL="${CAD_MODEL:-lgb}"
CAD_TASK="${CAD_TASK:-regression}"
CAD_SCHEME="${CAD_SCHEME:-rolling}"
CAD_METRIC="${CAD_METRIC:-corr}"
CAD_N_SPLITS="${CAD_N_SPLITS:-2}"     # vestigial: nothing is being selected
CAD_N_JOBS="${CAD_N_JOBS:-8}"
CAD_ERA_STEP="${CAD_ERA_STEP:-1}"
CAD_SEED="${CAD_SEED:-42}"
CAD_DATA="${CAD_DATA:-crypto/v2.0}"

# The configuration w6r6_lgb_reg's search converged on. Kept as one string so
# every arm provably gets the identical model.
CAD_PARAMS="${CAD_PARAMS:-{\"n_estimators\":4327,\"learning_rate\":0.002572,\"max_depth\":8,\"num_leaves\":255,\"colsample_bytree\":0.070213,\"subsample\":0.816153,\"subsample_freq\":1,\"min_child_samples\":313,\"reg_lambda\":1.757511}}"

WINDOWS="${WINDOWS_LIST:-3month 6month 9month 12month}"
REFITS="${REFITS_LIST:-3month 6month 9month 12month}"

# Refuse to run on the wrong dataset. A silent feature-count drift attaches a
# plausible number to the wrong experiment, which is worse than a crash.
python3 - "${CAD_DATA}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
path = "data/%s/train.parquet" % sys.argv[1]
if not os.path.exists(path):
    sys.exit("%s is missing -- run: bash setup.sh" % path)
n = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
if n != 22:
    sys.exit("expected 22 raw features in %s, found %d -- refusing to run"
             % (sys.argv[1], n))
print("preflight OK: %d raw features" % n)
PY

ARM_NAME=(); ARM_WIN=(); ARM_REFIT=(); ARM_PARAMS=()
add_arm() { ARM_NAME+=("$1"); ARM_WIN+=("$2"); ARM_REFIT+=("$3"); ARM_PARAMS+=("$4"); }

if [ "${SENS:-0}" = "1" ]; then
    # min_child_samples held at 0.80% of the block instead of 313 absolute.
    # 3m = 19,500 rows -> 156 ; 12m = 78,000 rows -> 626
    p3="${CAD_PARAMS/\"min_child_samples\":313/\"min_child_samples\":156}"
    p12="${CAD_PARAMS/\"min_child_samples\":313/\"min_child_samples\":626}"
    add_arm "w3m_r3m_sens"   3month  3month  "$p3"
    add_arm "w12m_r12m_sens" 12month 12month "$p12"
else
    for w in ${WINDOWS}; do
        for r in ${REFITS}; do
            add_arm "w${w%onth}_r${r%onth}" "$w" "$r" "${CAD_PARAMS}"
        done
    done
fi

total="${#ARM_NAME[@]}"
index=0; ran=0; failed=0; skipped=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "lgb ${CAD_TASK} cadence cross  |  ${total} arms, model PINNED"
echo "windows: ${WINDOWS}"
echo "refits : ${REFITS}"
echo "data=${CAD_DATA} scheme=${CAD_SCHEME} n_splits=${CAD_N_SPLITS} seed=${CAD_SEED}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  shard ${SHARD}/${OF}"
echo "=============================================================="

for i in "${!ARM_NAME[@]}"; do
    arm="${ARM_NAME[$i]}"; win="${ARM_WIN[$i]}"; ref="${ARM_REFIT[$i]}"
    params="${ARM_PARAMS[$i]}"
    index=$((index + 1))

    if [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi
    if [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue; fi
    if [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ]; then
        skipped=$((skipped + 1)); continue; fi

    run_name="cad_lgb_reg_${arm}"
    cmd=(python3 -u -m src.run_forecast crypto "${CAD_MODEL}"
         --task "${CAD_TASK}" --metric "${CAD_METRIC}"
         --cv-scheme "${CAD_SCHEME}" --train-window "${win}" --refit-every "${ref}"
         --data-version "${CAD_DATA}" --fixed-params "${params}"
         --n-splits "${CAD_N_SPLITS}" --n-jobs "${CAD_N_JOBS}"
         --era-step "${CAD_ERA_STEP}" --seed "${CAD_SEED}"
         --run-name "${run_name}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-26s window=%-8s refit=%-8s\n' \
               "$index" "$total" "$run_name" "$win" "$ref"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  window=${win} refit=${ref}  $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # A failed arm must not cost you the other fifteen.
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

python3 - <<'PY'
import json, glob, os, re
rows = []
for p in sorted(glob.glob("results/metrics/crypto/lgb/cad_lgb_reg_*.json")):
    d = json.load(open(p)); c = d["config"]; name = os.path.basename(p)[:-5]
    m = re.search(r"_w(\d+m)_r(\d+m)", name)
    rows.append((0.1 * d["corr"]["mean"] + d["mmc"]["mean"],
                 m.group(1) if m else "?", m.group(2) if m else "?",
                 c["train_window"], c["refit_every"], d["corr"]["mean"],
                 d["corr"]["sharpe"], d["corr"]["max_drawdown"],
                 d.get("runtime_minutes", 0), "sens" in name))
if not rows:
    print("no cadence results yet"); raise SystemExit
print("%-7s %-7s %7s %7s %10s %8s %8s %10s %6s" % (
    "window", "refit", "w_per", "r_per", "CORR", "sharpe", "maxdd", "payout", "min"))
print("-" * 78)
for pay, w, r, wp, rp, corr, sh, dd, rt, sens in sorted(rows, reverse=True):
    print("%-7s %-7s %7s %7s %10.6f %8.4f %8.4f %10.6f %6.1f%s" % (
        w, r, wp, rp, corr, sh, dd, pay, rt, "  (sens)" if sens else ""))
print("-" * 78)
print("baseline in-study: cad_lgb_reg_w6m_r6m")
print("searched arm for comparison: w6r6_lgb_reg payout 0.082061 sharpe 1.7951")
print()
print("paired test against the 6m/6m arm (Newey-West lag 20):")
print("  python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \\")
print("      --model lgb --baseline cad_lgb_reg_w6m_r6m --arms " +
      " ".join(sorted(os.path.basename(p)[:-5]
                      for p in glob.glob("results/metrics/crypto/lgb/cad_lgb_reg_*.json")
                      if "w6m_r6m" not in p)))
PY
echo "=============================================================="

exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
