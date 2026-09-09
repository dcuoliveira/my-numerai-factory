#!/usr/bin/env bash
#
# Experiment E08: the MLP feature-block table, on the datasets ridge was run on
#
# E08 is the reference handle. Everything this study writes is greppable as E08
# in the logs and mail, and as e08_* in results/{metrics,forecast,per_period,
# hyperparams}/crypto/mlp/.
#
# WHAT THIS RUNS
#   `mlp` regression on each of the 12 datasets the canonical ridge feature-block
#   table covers -- the w6r6_glmnet_reg_* family, which is one arm per dataset
#   with no ablations, no leave-one-block-out, and no lag sweeps. Same CURUPIRA
#   cadence as those arms, so each MLP arm has a ridge twin measured on the same
#   periods, folds and seed.
#
#   The 12 were derived from results/metrics/crypto/ridge_glmnet/, not from a
#   list someone typed: every w6r6_glmnet_reg_* run at rolling 130/130, embargo
#   20, 5 folds, 50 draws, seed 42, era_step 1, on the 20-day label.
#
# TWO QUESTIONS, AND THEY NEED DIFFERENT CONTROLS
#
#   1. Does a feature block help the MLP?      control: e08_mlp_raw
#      Same model, same cadence, one block of columns different. This is the
#      within-study comparison and the one the report prints.
#
#   2. Does the MLP beat ridge on this block?  control: the ridge twin
#      Same dataset and cadence, different function class. Printed second.
#
#   Reporting only (1) would answer "which columns suit an MLP" while sounding
#   like "should we deploy an MLP". They are different questions and the answers
#   have already diverged once: fenccausal HELPED the MLP (+1.55 vs mlp-raw) and
#   is recorded in EXPERIMENTS.md as consistently harmful to ridge and lgb
#   (-2.15, -2.24). One of those is wrong or the interaction is real; this table
#   is how you find out.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta, Newey-West lag 20. With 12 arms a
#   Bonferroni-style adjustment puts the honest threshold nearer |t| = 2.9 --
#   quote both. 510 periods at ~95% overlap is ~26 effective independent windows,
#   so nulls are the expected outcome and ARE the finding.
#
#   Read risk as well as level. On raw, the MLP's headline payout gain over ridge
#   was null (+1.15) while its CORR drawdown HALVED (1.22 -> 0.59). A mean-only
#   table would have missed the only large effect in E07.
#
# THE ALPHA GRID CHANGED AFTER E07 -- these numbers are not E07's
#   Every E07 regression refit pinned alpha at 98-100% of a grid that topped out
#   at 1e1, so those runs were grid-limited. MLPWrapper.ALPHAS now reaches 1e2.
#   e08_mlp_raw therefore RERUNS the E07 raw arm on the corrected grid: it is
#   this study's control, and comparing it to e07_mlpreg_raw says what the wider
#   ceiling bought. Do not mix an e07_* and an e08_* number in one row.
#
# COST
#   Measured on a real 130-period block (39,000 rows), MLP_THREADS=8. Width costs
#   far less than it looks like it should, because the hidden layers dominate and
#   the input layer is one matmul:
#
#       arch          22 feat   786 feat
#       (32,)            2.4s      19.7s
#       (256, 128)      33.4s      35.0s
#
#   E07's 22-feature arm took 92 min and its 44-feature arm 108 min. Budget
#   ~1.5-2 h for a narrow arm and ~2.5-4 h for a wide one: roughly 25-35 h for
#   all 12 sequentially. SHARD=/OF= splits that across concurrent runners --
#   SHARD=1 OF=3 ... SHARD=3 OF=3 is ~10 h wall clock on a 32-core box at 8
#   threads each.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e08_mlp_blocks_crypto.sh    # ~10 min, wiring
#   bash experiments/e08_mlp_blocks_crypto.sh                # all 12, sequential
#   SHARD=1 OF=3 bash experiments/e08_mlp_blocks_crypto.sh   # one third of them
#   ONLY="raw onehot" / SKIP="fencglobal" / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed. GNU screen exports $WINDOW=0 into every session it starts,
# so a bare WINDOW="${WINDOW:-130}" silently becomes 0 and every training block
# comes out empty -- no error, just meaningless scores.
EXPERIMENT="${EXPERIMENT:-E08}"
MLP_WINDOW="${MLP_WINDOW:-130}"
MLP_REFIT="${MLP_REFIT:-130}"
MLP_EMBARGO="${MLP_EMBARGO:-20}"
MLP_N_SPLITS="${MLP_N_SPLITS:-5}"
MLP_N_ITER="${MLP_N_ITER:-50}"
MLP_VAL_FRAC="${MLP_VAL_FRAC:-0.3}"
MLP_SEED="${MLP_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-8}"
MLP_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${MLP_WINDOW}" in ""|0|[0-9]) echo "MLP_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${MLP_REFIT}"  in ""|0|[0-9]) echo "MLP_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e08"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e08scout"; MLP_ERA_STEP=4; MLP_N_ITER=4; MLP_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${MLP_N_ITER} draws, ${MLP_N_SPLITS} folds."
    echo "            Wiring only -- the split boundary moves, so scout arms are"
    echo "            NOT comparable to any full-resolution baseline."
fi

# sklearn's MLP has no --n-jobs; it parallelises through whatever BLAS numpy is
# linked against, so left alone it takes every core. These are the only knobs.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing name|dataset|count and splitting on IFS='|' tears it
# apart and quietly fits on the wrong columns.
#
# arm / dataset / expected features / the ridge twin on the same dataset
NAME=(); DATA=(); NFEAT=(); TWIN=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); TWIN+=("$4"); }
add raw                 "crypto/v2.0"                                     22 ridge_glmnet/w6r6_glmnet_reg
add tenc                "crypto/v2.0-tenc"                                23 ridge_glmnet/w6r6_glmnet_reg_tenc
add spread              "crypto/v2.0-spread"                              33 ridge_glmnet/w6r6_glmnet_reg_spread
add interact            "crypto/v2.0-interact"                            36 ridge_glmnet/w6r6_glmnet_reg_interact
add fenccausal          "crypto/v2.0-fenccausal"                          44 ridge_glmnet/w6r6_glmnet_reg_fenccausal
add fencglobal          "crypto/v2.0-fencglobal"                          44 ridge_glmnet/w6r6_glmnet_reg_fencglobal
add onehot              "crypto/v2.0-onehot"                             762 ridge_glmnet/w6r6_glmnet_reg_onehot
add tenc_onehot         "crypto/v2.0-tenc-onehot"                        763 ridge_glmnet/w6r6_glmnet_reg_tenc_onehot
add onehot_rollsel      "crypto/v2.0-onehot-rollsel"                     786 ridge_glmnet/w6r6_glmnet_reg_onehot_rollsel
add tenc_onehot_rollsel "crypto/v2.0-tenc-onehot-rollsel"                787 ridge_glmnet/w6r6_glmnet_reg_tenc_onehot_rollsel
add allfeat             "crypto/v2.0-tenc-onehot-spread-interact"        788 ridge_glmnet/w6r6_glmnet_reg_allfeat
add allfeatpp           "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 ridge_glmnet/w6r6_glmnet_reg_allfeatpp

# fencglobal is included and is NOT a deployment candidate: a whole-sample feature
# mean encodes values you would not have live, and EXPERIMENTS.md records its
# lookahead strengthening with horizon (+2.83 on 20-day, +3.65 on 60-day). It is
# kept so the next reader does not rediscover and deploy it; the report footnotes
# it rather than dropping it, because a silently missing arm reads as "tested and
# null" when it means "never run".

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=(); SEL_TWIN=()
picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1))
        [ $(( (picked - 1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}")
    SEL_NFEAT+=("${NFEAT[$i]}"); SEL_TWIN+=("${TWIN[$i]}")
done
if [ "${#SEL_NAME[@]}" -eq 0 ]; then
    echo "no arms selected (ONLY='${ONLY:-}' SKIP='${SKIP:-}' SHARD=${SHARD}/${OF})" >&2
    exit 1
fi

# Preflight. A silent feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash -- read the parquet schema and refuse on
# a mismatch. The ridge twins are checked for cadence too: a cross-model claim
# against an arm fitted differently would be wrong in a way no table could reveal.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" -- "${SEL_TWIN[@]}" <<'PY' || exit 1
import json, os, sys
import pyarrow.parquet as pq

parts, cur = [], []
for tok in sys.argv[1:]:
    if tok == "--": parts.append(cur); cur = []
    else: cur.append(tok)
parts.append(cur)
versions, wants, twins = parts
wants = [int(x) for x in wants]

bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    if not os.path.exists(path):
        bad.append("%s MISSING -- build it with setup.sh --with-derived" % version)
        continue
    got = sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))

CADENCE = (("task", "regression"), ("cv_scheme", "rolling"), ("train_window", 130),
           ("refit_every", 130), ("embargo", 20), ("n_splits", 5), ("n_iter", 50),
           ("seed", 42), ("era_step", 1))
missing_twins = []
for twin, version in zip(twins, versions):
    model, run = twin.split("/")
    path = "results/metrics/crypto/%s/%s.json" % (model, run)
    if not os.path.exists(path):
        missing_twins.append(twin); continue
    config = json.load(open(path))["config"]
    for key, want in CADENCE:
        if config.get(key) != want:
            bad.append("ridge twin %s has %s=%r, this study runs %r -- not matched"
                       % (twin, key, config.get(key), want))
    if config.get("data_version") != version:
        bad.append("ridge twin %s is on %r, not %r"
                   % (twin, config.get("data_version"), version))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts" % len(versions))
print("              %d ridge twin(s) matched on cadence and dataset"
      % (len(twins) - len(missing_twins)))
if missing_twins:
    # Name it rather than letting a blank cell imply the comparison was made.
    print("              NO ridge twin on disk for: %s -- those arms get the "
          "within-study comparison only" % ", ".join(missing_twins))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

# Mail on START as well as at the end. A job that only mails on completion is
# indistinguishable from one that never started or silently died -- which is
# exactly the state you are in at hour three of a 30-hour study.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  model     mlp regression, sklearn MLP behind a StandardScaler
  cadence   rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws
  threads   ${MLP_THREADS}   shard ${SHARD}/${OF}
  controls  within-study e08_mlp_raw, and each arm's ridge twin
  expect    ~1.5-2 h narrow / ~2.5-4 h wide; ~25-35 h for all 12 sequentially
  NOTE      alpha ceiling raised 1e1 -> 1e2 after E07 pinned against it, so
            these numbers are NOT comparable to e07_mlpreg_raw's

Where to look while it runs:
  runner    experiments/e08_mlp_blocks_crypto.sh
  progress  logs/${PREFIX}_*.log        (one per arm)
  results   results/metrics/crypto/mlp/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt   (written at the end)
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: mlp feature blocks  |  ${total} arm(s)  |  shard ${SHARD}/${OF}"
echo "cadence: rolling ${MLP_WINDOW}/${MLP_REFIT}, embargo ${MLP_EMBARGO}, ${MLP_N_SPLITS} folds, ${MLP_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_mlp_${arm}"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto mlp
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${MLP_WINDOW}"
         --refit-every "${MLP_REFIT}" --embargo "${MLP_EMBARGO}"
         --n-splits "${MLP_N_SPLITS}" --n-iter "${MLP_N_ITER}"
         --val-frac "${MLP_VAL_FRAC}" --era-step "${MLP_ERA_STEP}"
         --seed "${MLP_SEED}" --search-jobs 1
         --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%2d/%2d] %-30s %5s feat  %s\n' "$index" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${SEL_DATA[$i]}"; continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  ${SEL_NFEAT[$i]} feat  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    # A failed arm must not cost the others: check PIPESTATUS[0] after the tee,
    # record it, continue.
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
        echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 0 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- mlp feature blocks on the datasets ridge was run on"
echo "  results  results/metrics/crypto/mlp/${PREFIX}_*.json"
echo "  runner   experiments/e08_mlp_blocks_crypto.sh"
echo "  logs     logs/${PREFIX}_*.log"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD} of ${OF} -- this is a PARTIAL table; the other shards hold the rest"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" "${SEL_NAME[@]}" -- "${SEL_TWIN[@]}" <<'PY'
import glob, json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
cut = sys.argv.index("--")
TWIN = dict(zip(sys.argv[3:cut], sys.argv[cut + 1:]))
CONTROL = "%s_mlp_raw" % prefix

def per_period(ref):
    model, run = ref.split("/") if "/" in ref else ("mlp", ref)
    f = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
    return f.set_index("date") if "date" in f.columns else f

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    # lag 20: a 20-day forward target on daily dates leaves consecutive periods
    # ~95% overlapping; treating them as independent inflates t by ~sqrt(20).
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

rows = []
for path in sorted(glob.glob("results/metrics/crypto/mlp/%s_mlp_*.json" % prefix)):
    run = os.path.basename(path)[:-5]
    rows.append((run, run[len("%s_mlp_" % prefix):], json.load(open(path))))
if not rows:
    print("no results"); raise SystemExit

print("%-30s %5s %6s %10s %10s %8s %8s %10s" % (
    "run", "feat", "n", "CORR", "MMC", "sharpe", "maxdd", "payout"))
print("-" * 94)
for run, arm, d in sorted(rows, key=lambda r: -(0.1*r[2]["corr"]["mean"] + r[2]["mmc"]["mean"])):
    print("%-30s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f" % (
        run, d["config"]["n_features"], d["n_periods"], d["corr"]["mean"],
        d["mmc"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"]))
print("-" * 94)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed, and none is available.")
    print("--era-step 4 keeps every 4th period and moves the split boundary, so")
    print("these arms share neither the sample nor the split with any baseline.")
    print("Rerun without MLP_SCOUT=1 for numbers that can be reported.")
    raise SystemExit

def table(title, control_of):
    print()
    print(title)
    print("%-30s %-34s %8s %11s %9s %7s" % (
        "arm", "control", "metric", "delta", "se", "t"))
    print("-" * 104)
    any_row = False
    for run, arm, _ in sorted(rows):
        ctrl = control_of(run, arm)
        if ctrl is None:
            continue
        try:
            base, a = per_period(ctrl), per_period(run)
        except FileNotFoundError:
            print("%-30s %-34s  control not on disk -- NOT COMPARED" % (run, ctrl))
            continue
        any_row = True
        for metric in ("corr", "mmc", "payout"):
            left = payout(base) if metric == "payout" else base[metric]
            right = payout(a) if metric == "payout" else a[metric]
            j = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"])
            delta, se, t = newey_west(j["a"] - j["b"])
            print("%-30s %-34s %8s %+11.6f %9.6f %+7.2f" % (
                run, ctrl, metric, delta, se, t))
    if not any_row:
        print("  (nothing to compare)")
    print("-" * 104)

table("1. DOES THE BLOCK HELP THE MLP?   paired vs %s, Newey-West lag 20" % CONTROL,
      lambda run, arm: None if arm == "raw" else CONTROL)
table("2. DOES THE MLP BEAT RIDGE HERE?  paired vs each arm's ridge twin, lag 20",
      lambda run, arm: TWIN.get(arm))

print()
print("510 periods at ~95% overlap is roughly 26 effective independent windows,")
print("so |t| < 2 is the EXPECTED outcome and IS the finding. With 12 arms a")
print("Bonferroni-style adjustment puts the honest threshold nearer |t| = 2.9.")
print()
print("fencglobal is NOT a deployment candidate whatever it scores: a whole-sample")
print("feature mean encodes values you would not have live, and EXPERIMENTS.md")
print("records its lookahead strengthening with horizon (+2.83 20d, +3.65 60d).")
print("It is here so the next reader does not rediscover and deploy it.")
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
