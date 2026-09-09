#!/usr/bin/env bash
#
# Experiment E11: does a learned ticker embedding earn its place in the MLP?
#
# E11 is the reference handle; every artefact it writes is e11_*.
#
# WHAT THIS RUNS
#   Two torch MLP arms on the raw 22 features at the CURUPIRA cadence, differing
#   in ONE thing -- whether the network can see which symbol a row belongs to:
#
#     e11_noemb_raw   --model torch_mlp      crypto/v2.0        22 features
#     e11_emb_raw     --model torch_mlp_emb  crypto/v2.0-symid  22 + one id code
#
#   TorchMLPEmbWrapper SUBCLASSES TorchMLPWrapper and overrides nothing but the
#   pipeline's first step and the addition of embed_dim, so the architecture set,
#   alpha range, learning rates, batch sizes, optimiser, stopping rule and seed
#   are inherited rather than restated. A test asserts the two grids differ in
#   exactly one key.
#
#   The scalers differ in name only: with no id column present
#   IdPassthroughScaler IS StandardScaler, and a test pins that too. It exists
#   because an integer ticker code must reach the network UNSCALED -- standardising
#   it produces a meaningless number and destroys the integer the lookup needs.
#
# THE EMBEDDING IS THE LSTM'S, DELIBERATELY
#   src/prediction/nn/model.py already answered "what is a symbol embedding here"
#   for the sequence model, and the two must not disagree. Copied from it:
#   nn.Embedding(n_train_symbols + 1, embed_dim, padding_idx=n_train_symbols); the
#   extra row absorbs codes the training block never saw, and padding_idx keeps it
#   at zero so an unseen ticker contributes NO identity signal rather than a
#   random one. The index is built from training codes only. The vector is
#   concatenated to the features, so the first hidden layer can interact the two.
#   What differs from the LSTM is only what follows -- a dense stack instead of a
#   recurrent layer.
#
# WHAT WOULD COUNT AS A RESULT, AND THE PRIOR AGAINST IT
#   |t| >= 2 on the paired payout delta, Newey-West lag 20, e11_emb vs
#   e11_noemb. Three measured facts argue against:
#
#     * A 130-period block holds 460 distinct symbols with a median of 111 rows
#       each -- but 36% have fewer than 60 rows and 8% have fewer than 5. Those
#       vectors are estimated from almost nothing.
#     * At embed_dim 32 the table is 14,752 parameters against roughly 3,488 in a
#       (64, 32) stack over 22 features. Most of the model becomes identity.
#     * build_symbol_embedding.py records that both existing encodings have COST
#       ridge accuracy on this panel: best CORR 0.173 raw, 0.156 with -tenc,
#       0.142 with -onehot. The per-symbol effect is real (5.85x the dispersion
#       noise would give, persisting at 0.309 between halves); capturing it
#       without paying more than it is worth is what has never worked.
#
#   So a null is the expected outcome and IS the finding. What would make this
#   study worth more than its cost is the RISK column: an embedding that leaves
#   payout alone and cuts drawdown is still a result. Read both.
#
# COST
#   E09's torch arm took 66.7 min on the same data and cadence. The embedding
#   adds a table and widens the first layer; budget ~1.5 h for the no-embedding
#   arm and ~2 h for the embedding one, so about 3.5 h in total.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e11_symbol_embedding_crypto.sh   # ~3 min
#   bash experiments/e11_symbol_embedding_crypto.sh
#   ONLY=emb / SKIP=noemb / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E11}"
EMB_WINDOW="${EMB_WINDOW:-130}"
EMB_REFIT="${EMB_REFIT:-130}"
EMB_EMBARGO="${EMB_EMBARGO:-20}"
EMB_N_SPLITS="${EMB_N_SPLITS:-5}"
EMB_N_ITER="${EMB_N_ITER:-50}"
EMB_VAL_FRAC="${EMB_VAL_FRAC:-0.3}"
EMB_SEED="${EMB_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-8}"
EMB_ERA_STEP=1
case "${EMB_WINDOW}" in ""|0|[0-9]) echo "EMB_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${EMB_REFIT}"  in ""|0|[0-9]) echo "EMB_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e11"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e11scout"; EMB_ERA_STEP=4; EMB_N_ITER=4; EMB_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${EMB_N_ITER} draws, ${EMB_N_SPLITS} folds -- wiring only."
fi

# Neither MLP has an --n-jobs; these are the only knobs that bound either.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
NAME=(noemb emb)
MODEL=(torch_mlp torch_mlp_emb)
DATA=("crypto/v2.0" "crypto/v2.0-symid")
NFEAT=(22 23)
SEL_NAME=(); SEL_MODEL=(); SEL_DATA=(); SEL_NFEAT=()
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    SEL_NAME+=("${arm}"); SEL_MODEL+=("${MODEL[$i]}")
    SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. The grid relationship is checked here
# too: if the two wrappers ever differ by more than embed_dim, this study stops
# measuring the embedding and nothing downstream would reveal it.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq

cut = sys.argv.index("--")
versions, wants = sys.argv[1:cut], [int(x) for x in sys.argv[cut + 1:]]
bad = []
for version, want in zip(versions, wants):
    path = "data/%s/train.parquet" % version
    try:
        names = pq.ParquetFile(path).schema.names
    except OSError:
        bad.append("%s MISSING -- build it with "
                   "python3 -m src.data_mani.build_symbol_id" % version)
        continue
    got = sum(c.startswith("feature_") for c in names)
    if got != want:
        bad.append("%s has %d features, expected %d" % (version, got, want))
    if version.endswith("-symid") and "feature_symbol_id" not in names:
        bad.append("%s carries no feature_symbol_id column" % version)

try:
    import torch                                          # noqa: F401
except ImportError as error:
    bad.append("torch is not installed (%s); both arms need it" % error)

from src.prediction.models import build_wrapper
a = build_wrapper("torch_mlp", "regression", seed=42, n_jobs=1)
b = build_wrapper("torch_mlp_emb", "regression", seed=42, n_jobs=1)
if len(a.param_grid) != len(b.param_grid):
    bad.append("the two grids have different sub-grid counts")
else:
    for i, (ga, gb) in enumerate(zip(a.param_grid, b.param_grid)):
        extra, missing = set(gb) - set(ga), set(ga) - set(gb)
        if extra != {"mlp__embed_dim"} or missing:
            bad.append("sub-grid %d differs by %s (missing %s); the arms would "
                       "differ in more than the embedding" % (i, extra, missing))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d dataset(s) at the expected feature counts; torch present; "
      "the two wrappers differ by mlp__embed_dim and nothing else"
      % len(versions))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      does a learned ticker embedding earn its place in the MLP?
  cadence   rolling ${EMB_WINDOW}/${EMB_REFIT}, embargo ${EMB_EMBARGO}, ${EMB_N_SPLITS} folds, ${EMB_N_ITER} draws
  threads   ${MLP_THREADS}
  expect    ~1.5 h no-embedding + ~2 h embedding = ~3.5 h
  prior     against: 36% of a block's 460 symbols have <60 rows, and at
            embed_dim 32 the table is 4x the dense stack. Read the RISK
            column -- a null on payout with a lower drawdown is still a result.

Where to look while it runs:
  runner    experiments/e11_symbol_embedding_crypto.sh
  progress  logs/${PREFIX}_*.log
  results   results/metrics/crypto/{torch_mlp,torch_mlp_emb}/${PREFIX}_*.json
  report    logs/${PREFIX}_report.txt
  handle    this study is ${EXPERIMENT}; every artefact it writes is ${PREFIX}_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: ticker embedding on the torch MLP  |  ${total} arm(s)"
echo "cadence: rolling ${EMB_WINDOW}/${EMB_REFIT}, embargo ${EMB_EMBARGO}, ${EMB_N_SPLITS} folds, ${EMB_N_ITER} draws, ${MLP_THREADS} threads"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}_raw"; index=$((index+1))
    cmd=(python3 -u -m src.run_forecast crypto "${SEL_MODEL[$i]}"
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${EMB_WINDOW}"
         --refit-every "${EMB_REFIT}" --embargo "${EMB_EMBARGO}"
         --n-splits "${EMB_N_SPLITS}" --n-iter "${EMB_N_ITER}"
         --val-frac "${EMB_VAL_FRAC}" --era-step "${EMB_ERA_STEP}"
         --seed "${EMB_SEED}" --search-jobs 1
         --data-version "${SEL_DATA[$i]}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-22s %-14s %-20s %s feat\n' "$index" "$total" \
               "$run_name" "${SEL_MODEL[$i]}" "${SEL_DATA[$i]}" "${SEL_NFEAT[$i]}"
        continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  (--model ${SEL_MODEL[$i]})  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    # A failed arm must not cost the other: read PIPESTATUS[0] after the tee.
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
{
echo "${EXPERIMENT} -- does a learned ticker embedding earn its place?"
echo "  results  results/metrics/crypto/{torch_mlp,torch_mlp_emb}/${PREFIX}_*.json"
echo "  runner   experiments/e11_symbol_embedding_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/60))m$((el_all%60))s, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
ARMS = [("torch_mlp", "%s_noemb_raw" % prefix, "no embedding"),
        ("torch_mlp_emb", "%s_emb_raw" % prefix, "embedding")]

def load(model, run):
    p = "results/metrics/crypto/%s/%s.json" % (model, run)
    return json.load(open(p)) if os.path.exists(p) else None

def per_period(model, run):
    f = pd.read_csv("results/per_period/crypto/%s/%s.csv" % (model, run))
    return f.set_index("date") if "date" in f.columns else f

def payout(f):
    return 0.1 * f["corr"] + f["mmc"]

def newey_west(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n)
    return m, se, (m / se if se > 0 else float("nan"))

found = [(m, r, lab, load(m, r)) for m, r, lab in ARMS]
print("%-22s %-13s %6s %10s %10s %8s %8s %10s %8s" % (
    "run", "arm", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 104)
for model, run, lab, d in found:
    if d is None:
        # Never a blank line: absent reads as "tested and null" when it means
        # "never ran".
        print("%-22s %-13s  NOT ON DISK -- this arm did not run" % (run, lab))
        continue
    print("%-22s %-13s %6d %10.6f %10.6f %8.4f %8.4f %10.6f %8.1f" % (
        run, lab, d["n_periods"], d["corr"]["mean"], d["mmc"]["mean"],
        d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 104)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit
if any(d is None for *_, d in found):
    print()
    print("Both arms are needed for the comparison; one is missing, so none is")
    print("printed. Rerun the missing arm with ONLY=.")
    raise SystemExit

print()
print("paired embedding - no embedding, inner join on periods, Newey-West lag 20")
print("%-10s %12s %12s %+12s %10s %8s" % (
    "metric", "no emb", "emb", "delta", "se", "t"))
print("-" * 70)
pa, pb = per_period(*found[0][:2]), per_period(*found[1][:2])
for metric in ("corr", "mmc", "payout"):
    left = payout(pa) if metric == "payout" else pa[metric]
    right = payout(pb) if metric == "payout" else pb[metric]
    j = pd.concat([left, right], axis=1, join="inner", keys=["a", "b"])
    delta, se, t = newey_west(j["b"] - j["a"])
    print("%-10s %12.6f %12.6f %+12.6f %10.6f %+8.2f" % (
        metric, j["a"].mean(), j["b"].mean(), delta, se, t))
print("-" * 70)

# What embed_dim the search actually chose is the second half of the answer: a
# search that keeps picking the SMALLEST width is saying the table is a cost.
try:
    picked = []
    for refit in load(*found[1][:2])["refits"]:
        picked.append(refit["best_params"].get("mlp__embed_dim"))
    print()
    print("embed_dim chosen per refit: %s   (grid %s)"
          % (picked, [4, 8, 16, 32]))
    print("a search that keeps taking the smallest width is saying the table")
    print("costs more than the identity it captures is worth.")
except Exception as error:
    print("\ncould not read the chosen embed_dim: %s" % error)

print()
print("510+ periods at ~95%% overlap is roughly 26 effective independent")
print("windows, so |t| < 2 is the EXPECTED outcome and IS the finding. Report")
print("it with its interval. Read maxdd as well as payout: an embedding that")
print("leaves the mean alone and cuts drawdown is still a result.")
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
