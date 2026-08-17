#!/usr/bin/env bash
#
# Experiment: LightGBM, trained from scratch pseudo-OOS, 6m window / 6m refit
#
# This is the shared implementation. Do not invoke it directly -- use the two
# task runners, which exist so regression and classification can be launched,
# logged and shipped independently:
#
#     bash experiments/lgb_6m6m_reg_crypto.sh
#     bash experiments/lgb_6m6m_clf_crypto.sh
#
# ---------------------------------------------------------------------------
# What this runs
# ---------------------------------------------------------------------------
#
# Nine arms per task, in this order, every one a complete walk-forward with a
# fresh model per refit -- no warm starts, no carried-over hyperparameters:
#
#     1  raw          22 shipped features            crypto/v2.0
#     2  allfeatpp    812 features (the union)       crypto/v2.0-tenc-onehot-rollsel-spread-interact
#     3  price         4   close_avg, close_ewa                   x 20d/60d
#     4  size          4   market_cap_avg, market_cap_ewa         x 20d/60d
#     5  liquidity     4   volume_avg, volume_ewa                 x 20d/60d
#     6  oscillator    4   relative_strength_index, sharpe_ratio  x 20d/60d
#     7  bollinger     2   bollinger                              x 20d/60d
#     8  momentum      2   momentum                               x 20d/60d
#     9  volatility    2   volatility                             x 20d/60d
#
# Arms 3-9 are the same seven groups the ridge study used, carved out of the
# shipped dataset with --feature-include, each fitted as the ENTIRE feature set
# rather than raw+group. The partition is checked before anything runs.
#
# The protocol is byte-for-byte the ridge protocol -- rolling 6-month window,
# 6-month refit, purged/embargoed inner CV, 50-draw randomised search, 5 folds,
# seed 42 -- so these numbers drop straight into the existing comparison table.
# The model class is the only thing that changed.
#
# ---------------------------------------------------------------------------
# Why LightGBM, and what would count as a result
# ---------------------------------------------------------------------------
#
# Every ridge arm in this study landed flat: the prediction-correlation heatmap
# put all of them at 0.886-1.000 against each other, glmnet chose shrinkage up
# to alpha=1.2M, and both feature-engineering ceiling checks (allfeat 788,
# allfeat++ 812) failed to beat the 22 raw columns. That is the signature of a
# model class that has already extracted what it can from these inputs.
#
# LightGBM measured rho=0.62 against the deployed ridge configuration, against
# 0.89-1.00 for every ridge variant. So the interesting number here is not only
# payout but CORRELATION TO THE DEPLOYED BOOK: an arm that scores slightly worse
# but decorrelates is worth more to a multi-slot stake than one that scores
# slightly better and moves together.
#
# The bar, from the all-22 ridge references (510 periods, payout = 0.1*CORR + 1*MMC):
#
#     w6r6_fixed_reg     22 raw, ridge          payout 0.085763  sharpe 1.5424
#     w6r6_glmnet_reg    22 raw, ridge_glmnet   payout 0.083859  sharpe 1.8093
#
# Two things to expect from a booster that did not apply to ridge. First, the
# groups will behave differently: 2-4 features gave ridge almost nothing to
# shrink, but they give a tree almost nothing to split on either, and depth 10
# on 2 columns is pure interaction-of-noise. Second, allfeatpp is the arm where
# a booster can actually beat a linear model -- colsample_bytree 0.05-0.30 over
# 812 columns is the Numerai idiom of starving each tree for diversity, which
# has no ridge analogue.
#
# ---------------------------------------------------------------------------
# Runtime -- read this before launching
# ---------------------------------------------------------------------------
#
# The ridge arms took ~27 min each. These do not. Measured on this machine
# (32 cores), one LGBM fit at the top of the grid (30,000 trees, depth 10,
# 1023 leaves) on a 39,000-row training block:
#
#     22 features    56s          812 features    79s
#     2,000 trees / depth 5:  2.3s and 2.7s
#
# A run is 50 draws x 5 folds x 4 refits = ~1,000 fits. At LGB_SEARCH_JOBS=4 x
# LGB_N_JOBS=8 (= 32 threads, the whole machine) budget roughly:
#
#     raw          ~1.0-1.5 h        each group   ~0.5-0.7 h
#     allfeatpp    ~1.5-2.0 h        nine arms    ~8 h per task
#
# So run ONE task at a time on a full machine. To run both concurrently, halve
# the threading on each:
#
#     LGB_N_JOBS=4 LGB_SEARCH_JOBS=4 bash experiments/lgb_6m6m_reg_crypto.sh
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/lgb_6m6m_reg_crypto.sh
#   DRY_RUN=1 bash experiments/lgb_6m6m_reg_crypto.sh      # print the commands
#   SMOKE=1   bash experiments/lgb_6m6m_reg_crypto.sh      # 2 draws, every 8th period
#   ONLY=raw  bash experiments/lgb_6m6m_reg_crypto.sh      # one arm
#   ONLY="raw allfeatpp" bash experiments/lgb_6m6m_reg_crypto.sh
#   SKIP=allfeatpp bash experiments/lgb_6m6m_clf_crypto.sh
#   SHARD=1 OF=2 bash experiments/lgb_6m6m_reg_crypto.sh   # split across machines
#
# Under screen:
#   screen -L -Logfile lgb-w6r6-reg.txt bash experiments/lgb_6m6m_reg_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# LGB_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0)
# into every session it starts, so a bare WINDOW="${WINDOW:-6month}" silently
# becomes 0 under screen and every training block comes out empty.
LGB_TASK="${LGB_TASK:-regression}"
LGB_MODEL="${LGB_MODEL:-lgb}"
LGB_SCHEME="${LGB_SCHEME:-rolling}"
LGB_TRAIN_WINDOW="${LGB_TRAIN_WINDOW:-6month}"
LGB_REFIT="${LGB_REFIT:-6month}"
LGB_METRIC="${LGB_METRIC:-corr}"
LGB_N_ITER="${LGB_N_ITER:-50}"
LGB_N_SPLITS="${LGB_N_SPLITS:-5}"
# random keeps these arms like-for-like with every ridge arm. bayes is
# available, but read the warning it prints: LightGBM's grid is six sub-grids
# (one per max_depth) and skopt fits a surrogate per sub-grid, so --n-iter 50
# gives each one ~8 draws and the search never leaves its random warm-up.
# LGB_SEARCH_TYPE=bayes is only meaningful alongside LGB_N_ITER=150 or more.
LGB_SEARCH_TYPE="${LGB_SEARCH_TYPE:-random}"
LGB_N_JOBS="${LGB_N_JOBS:-8}"
LGB_SEARCH_JOBS="${LGB_SEARCH_JOBS:-4}"
LGB_ERA_STEP="${LGB_ERA_STEP:-1}"
LGB_SEED="${LGB_SEED:-42}"

case "${LGB_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "LGB_TRAIN_WINDOW=${LGB_TRAIN_WINDOW} looks wrong (screen exports \$WINDOW=0);" \
                     "expected e.g. 6month" >&2; exit 1 ;;
esac

case "${LGB_TASK}" in
    regression)     TAG="reg" ;;
    classification) TAG="clf" ;;
    *) echo "LGB_TASK must be regression or classification, got '${LGB_TASK}'" >&2; exit 1 ;;
esac

# SMOKE is for checking the wiring, not for producing numbers. It writes to
# _smoke-suffixed run names so it can never overwrite a real result.
SUFFIX=""

# A bayes run must never land on a random run's filename. The two search a
# DIFFERENT space -- bayes turns every explicit grid into a continuous interval
# -- so silently overwriting one with the other would leave a results tree whose
# rows are not comparable and whose provenance is only recoverable from the
# config block inside each json.
[ "${LGB_SEARCH_TYPE}" = "bayes" ] && SUFFIX="_bayes"

if [ "${SMOKE:-0}" = "1" ]; then
    LGB_N_ITER=2; LGB_N_SPLITS=2; LGB_ERA_STEP=8; SUFFIX="${SUFFIX}_smoke"
    echo ">>> SMOKE MODE: n_iter=2 n_splits=2 era_step=8, results suffixed ${SUFFIX}"
fi

RAW_DATA="crypto/v2.0"
# The 812-feature union, already built by experiments/allfeat_plus_crypto.sh.
AFP_DATA="crypto/v2.0-tenc-onehot-rollsel-spread-interact"

# Parallel arrays, not packed "name|regex" strings: several of these regexes
# contain "|" as alternation, which would be torn apart by IFS='|' splitting.
# An empty regex means "use every feature in the dataset".
ARM_NAME=(); ARM_DATA=(); ARM_RE=()
add_arm() { ARM_NAME+=("$1"); ARM_DATA+=("$2"); ARM_RE+=("$3"); }

add_arm raw        "${RAW_DATA}" ''
add_arm allfeatpp  "${AFP_DATA}" ''
add_arm price      "${RAW_DATA}" '^feature_close_(avg|ewa)_'
add_arm size       "${RAW_DATA}" '^feature_market_cap_(avg|ewa)_'
add_arm liquidity  "${RAW_DATA}" '^feature_volume_(avg|ewa)_'
add_arm oscillator "${RAW_DATA}" '^feature_(relative_strength_index|sharpe_ratio)_'
add_arm bollinger  "${RAW_DATA}" '^feature_bollinger_'
add_arm momentum   "${RAW_DATA}" '^feature_momentum_'
add_arm volatility "${RAW_DATA}" '^feature_volatility_'

# ---------------------------------------------------------------------------
# Preflight. A silent feature-count drift is the one error that produces a
# plausible number attached to the wrong experiment, so refuse to run on it.
# ---------------------------------------------------------------------------
TAG="${TAG}" python3 - "${AFP_DATA}" "${ARM_RE[@]:2}" <<'PY' || exit 1
import os, re, sys
import pyarrow.parquet as pq

afp = sys.argv[1]
patterns = sys.argv[2:]

raw = [c for c in pq.ParquetFile("data/crypto/v2.0/train.parquet").schema.names
       if c.startswith("feature_")]
if len(raw) != 22:
    sys.exit("expected 22 features in crypto/v2.0, found %d" % len(raw))

afp_path = "data/%s/train.parquet" % afp
if not os.path.exists(afp_path):
    # setup.sh, NOT allfeat_plus_crypto.sh: that script starts from the
    # 787-feature union and only appends spread + interact, so on a machine
    # where the union has never been built it fails too. setup.sh builds the
    # whole chain from the shipped 22 features.
    sys.exit("%s is missing -- build the feature chain first:\n"
             "  bash setup.sh --with-derived        (~3 min)\n"
             "then re-run this study. To run the other eight arms meanwhile:\n"
             "  SKIP=allfeatpp bash experiments/lgb_6m6m_%s_crypto.sh"
             % (afp_path, os.environ.get("TAG", "reg")))
n_afp = sum(c.startswith("feature_")
            for c in pq.ParquetFile(afp_path).schema.names)
if n_afp != 812:
    sys.exit("expected 812 features in %s, found %d" % (afp, n_afp))

# The seven groups must exactly partition the 22 raw features -- no gap, no
# overlap -- or the group arms describe a different experiment than documented.
seen, total = set(), 0
for pattern in patterns:
    hit = {c for c in raw if re.search(pattern, c)}
    if not hit:
        sys.exit("pattern %r matched nothing" % pattern)
    if hit & seen:
        sys.exit("patterns overlap on %s" % sorted(hit & seen))
    seen |= hit
    total += len(hit)
missing = set(raw) - seen
if missing:
    sys.exit("groups do not cover %d feature(s): %s" % (len(missing), sorted(missing)))

print("preflight OK: 22 raw features, %d in allfeat++, "
      "%d groups partitioning all %d raw" % (n_afp, len(patterns), total))
PY

total="${#ARM_NAME[@]}"
index=0; failed=0; ran=0; skipped=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

echo "=============================================================="
echo "lgb ${LGB_TASK} (${TAG})  |  ${total} arms"
echo "window=${LGB_TRAIN_WINDOW} refit=${LGB_REFIT} scheme=${LGB_SCHEME} metric=${LGB_METRIC}"
echo "n_iter=${LGB_N_ITER} n_splits=${LGB_N_SPLITS} search=${LGB_SEARCH_TYPE}"
echo "era_step=${LGB_ERA_STEP} seed=${LGB_SEED}"
echo "threads=${LGB_N_JOBS} x search_jobs=${LGB_SEARCH_JOBS}  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!ARM_NAME[@]}"; do
    arm="${ARM_NAME[$i]}"; data="${ARM_DATA[$i]}"; pattern="${ARM_RE[$i]}"
    index=$((index + 1))

    # ONLY / SKIP take space-separated lists, so " ${LIST} " word-matching is
    # what keeps "raw" from matching inside another arm's name.
    if [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue
    fi
    if [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]]; then
        skipped=$((skipped + 1)); continue
    fi
    if [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ]; then
        skipped=$((skipped + 1)); continue
    fi

    # raw/allfeatpp keep the flat naming the ridge arms use; the groups get the
    # w6r6_grp_ prefix so they sort together with the ridge group results.
    case "${arm}" in
        raw)       run_name="w6r6_lgb_${TAG}${SUFFIX}" ;;
        allfeatpp) run_name="w6r6_lgb_${TAG}_allfeatpp${SUFFIX}" ;;
        *)         run_name="w6r6_lgb_grp_${arm}_${TAG}${SUFFIX}" ;;
    esac

    cmd=(python3 -u -m src.run_forecast crypto "${LGB_MODEL}"
         --task "${LGB_TASK}"
         --metric "${LGB_METRIC}"
         --cv-scheme "${LGB_SCHEME}" --train-window "${LGB_TRAIN_WINDOW}"
         --refit-every "${LGB_REFIT}" --data-version "${data}"
         --n-iter "${LGB_N_ITER}" --n-splits "${LGB_N_SPLITS}"
         --search-type "${LGB_SEARCH_TYPE}"
         --n-jobs "${LGB_N_JOBS}" --search-jobs "${LGB_SEARCH_JOBS}"
         --era-step "${LGB_ERA_STEP}" --seed "${LGB_SEED}"
         --run-name "${run_name}")
    [ -n "${pattern}" ] && cmd+=(--feature-include "${pattern}")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-30s %s\n' "$index" "$total" "$run_name" "${cmd[*]}"
        continue
    fi

    ran=$((ran + 1))
    echo
    echo "=============================================================="
    echo "[${index}/${total}] ${run_name}   $(date '+%H:%M:%S')"
    echo "=============================================================="

    started=$(date +%s)
    # A failed arm must not cost you the other eight.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - started ))

    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> ${run_name} FAILED (exit ${status}) after $((elapsed / 60)) min -- continuing"
    else
        echo ">>> ${run_name} done in $((elapsed / 60)) min"
    fi
done

[ "${DRY_RUN:-0}" = "1" ] && exit 0

echo
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s) - started_all) / 60 )) min"
echo "(${skipped} skipped, ${failed} failed)"
echo
echo "rank this task's arms against the all-22 ridge references:"
cat <<REPORT
  python3 - <<'PY'
import json, glob, os
# Glob everything and filter on the task tag, rather than pattern-matching the
# suffix: run names carry an optional _bayes, so an exact-ending glob would
# quietly drop half the table.
rows = []
paths = sorted(glob.glob("results/metrics/crypto/lgb/w6r6_lgb_*.json")) + [
        "results/metrics/crypto/ridge_glmnet/w6r6_glmnet_reg.json",
        "results/metrics/crypto/ridge/w6r6_fixed_reg.json"]
for p in dict.fromkeys(paths):
    if not os.path.exists(p) or "_smoke" in p: continue
    name = os.path.basename(p)[:-5]
    if name.startswith("w6r6_lgb_") and "_${TAG}" not in name + "_": continue
    d = json.load(open(p))
    rows.append((0.1 * d["corr"]["mean"] + d["mmc"]["mean"], name,
                 d["config"]["n_features"], d["config"].get("search_type", "random"),
                 d["corr"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"]))
print("%-36s %5s %7s %10s %9s %9s %10s"
      % ("run", "feat", "search", "CORR", "sharpe", "maxdd", "payout"))
for pay, n, f, st, c, s, dd in sorted(rows, reverse=True):
    print("%-36s %5d %7s %10.6f %9.4f %9.4f %10.6f" % (n, f, st, c, s, dd, pay))
PY
REPORT
echo
echo "then check DECORRELATION against the deployed book, which is the point:"
echo "  python3 -m src.analysis.corr_heatmap --help"
echo "=============================================================="

[ "${failed}" -gt 0 ] && exit 1
exit 0
