#!/usr/bin/env bash
#
# Experiment E12: how wide should the ticker embedding be?
#
# E12 is the reference handle; every artefact it writes is e12_*.
#
# WHAT THIS RUNS
#   Six torch MLP arms on the raw 22 features at the CURUPIRA cadence. The only
#   thing varying is the width of the ticker embedding table:
#
#     e12_emb0_raw    --model torch_mlp      no embedding at all (the control)
#     e12_emb4_raw    --model torch_mlp_emb  --model-options '{"embed_dim": 4}'
#     e12_emb8_raw                                              8
#     e12_emb16_raw                                            16
#     e12_emb32_raw                                            32
#     e12_emb64_raw                                            64
#
#   Each width is PINNED and everything else -- architecture, alpha, learning
#   rate, batch size -- is still searched, so each arm gets its own best
#   configuration at that width. That is what --model-options buys and what
#   --fixed-params could not do: pinning there collapses the grid to one
#   candidate and disables the search entirely.
#
#   E11 asked the same question by putting embed_dim IN the grid and letting one
#   search choose. This asks it as six separate models, which is stronger: a
#   search reports one winner and no interval, while six arms give a curve, six
#   paired deltas, and an answer to "does it degrade gracefully or fall off a
#   cliff". They are complementary; E11's arm is not a substitute for e12_emb0.
#
# THE CONTROL IS A DIFFERENT NETWORK, NOT A ZERO-WIDTH ONE
#   e12_emb0_raw uses --model torch_mlp: no embedding module, no id column, a
#   first layer of exactly 22 inputs. embed_dim=0 is rejected by the wrapper on
#   purpose, because a zero-width table is not the same object as no table --
#   it would still carry the id column, the padding row and the concatenation.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired payout delta against e12_emb0_raw, Newey-West lag 20.
#   With five comparisons a Bonferroni-style adjustment puts the honest
#   threshold nearer |t| = 2.6; quote both. On ~26 effective independent windows
#   nulls are expected and ARE the finding.
#
#   The SHAPE across widths is worth more than any single arm. Five nulls that
#   fall monotonically with width say the table is a pure cost; five nulls with a
#   hump at 8 say something is there that this sample cannot resolve. Read the
#   curve, and read maxdd alongside payout -- an embedding that leaves the mean
#   alone and cuts drawdown is still a result.
#
# THE PRIOR, MEASURED
#   * A 130-period block holds 460 distinct symbols, median 111 rows each -- but
#     36% have under 60 rows and 8% under 5. Those vectors are estimated from
#     almost nothing, and widening the table does not give them more data.
#   * Table parameters against roughly 3,488 in a (64, 32) stack over 22 features:
#         embed_dim   4      8     16     32     64
#         params  1,844  3,688  7,376 14,752 29,504
#         ratio    0.5x   1.1x   2.1x   4.2x   8.5x
#     At 64 the model is ~89% ticker identity by parameter count.
#   * build_symbol_embedding.py records both existing encodings COSTING ridge
#     accuracy here: CORR 0.173 raw, 0.156 -tenc, 0.142 -onehot. The per-symbol
#     effect is real; capturing it for less than it costs is what has not worked.
#
# COST
#   E09's torch arm took 66.7 min on this data and cadence; the embedding adds a
#   table and widens the first layer. Budget ~1.5 h for emb0 and ~2 h for each
#   embedding arm -- roughly 11 h for all six sequentially. SHARD=/OF= splits it:
#   SHARD=1 OF=3 ... SHARD=3 OF=3 is about 4 h wall clock at 8 threads each.
#
# Usage:
#   MLP_SCOUT=1 bash experiments/e12_embed_dim_crypto.sh    # ~8 min, wiring
#   bash experiments/e12_embed_dim_crypto.sh                # all six
#   SHARD=1 OF=3 bash experiments/e12_embed_dim_crypto.sh   # one third
#   ONLY="emb0 emb8" / SKIP=emb64 / DRY_RUN=1 / NO_MAIL=1 / MLP_THREADS=16
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
EXPERIMENT="${EXPERIMENT:-E12}"
EMB_WINDOW="${EMB_WINDOW:-130}"
EMB_REFIT="${EMB_REFIT:-130}"
EMB_EMBARGO="${EMB_EMBARGO:-20}"
EMB_N_SPLITS="${EMB_N_SPLITS:-5}"
EMB_N_ITER="${EMB_N_ITER:-50}"
EMB_VAL_FRAC="${EMB_VAL_FRAC:-0.3}"
EMB_SEED="${EMB_SEED:-42}"
MLP_THREADS="${MLP_THREADS:-8}"
EMB_ERA_STEP=1
SHARD="${SHARD:-0}"; OF="${OF:-0}"
case "${EMB_WINDOW}" in ""|0|[0-9]) echo "EMB_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${EMB_REFIT}"  in ""|0|[0-9]) echo "EMB_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e12"
if [ "${MLP_SCOUT:-0}" = "1" ]; then
    PREFIX="e12scout"; EMB_ERA_STEP=4; EMB_N_ITER=4; EMB_N_SPLITS=2
    echo "SCOUT MODE: era-step 4, ${EMB_N_ITER} draws, ${EMB_N_SPLITS} folds -- wiring only."
fi

# Neither MLP has an --n-jobs; these are the only knobs that bound either.
export OMP_NUM_THREADS="${MLP_THREADS}"
export MKL_NUM_THREADS="${MLP_THREADS}"
export OPENBLAS_NUM_THREADS="${MLP_THREADS}"

# Parallel arrays, never pipe-delimited records: a feature regex contains '|' as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
# arm / model / dataset / expected features / embed_dim ("" = no embedding)
NAME=();  MODEL=(); DATA=(); NFEAT=(); DIM=()
add() { NAME+=("$1"); MODEL+=("$2"); DATA+=("$3"); NFEAT+=("$4"); DIM+=("$5"); }
add emb0  torch_mlp     "crypto/v2.0"       22 ""
add emb4  torch_mlp_emb "crypto/v2.0-symid" 23 4
add emb8  torch_mlp_emb "crypto/v2.0-symid" 23 8
add emb16 torch_mlp_emb "crypto/v2.0-symid" 23 16
add emb32 torch_mlp_emb "crypto/v2.0-symid" 23 32
add emb64 torch_mlp_emb "crypto/v2.0-symid" 23 64

SEL_NAME=(); SEL_MODEL=(); SEL_DATA=(); SEL_NFEAT=(); SEL_DIM=()
picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 0 ]; then
        picked=$((picked+1))
        [ $(( (picked - 1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_MODEL+=("${MODEL[$i]}"); SEL_DATA+=("${DATA[$i]}")
    SEL_NFEAT+=("${NFEAT[$i]}"); SEL_DIM+=("${DIM[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no arms selected" >&2; exit 1; }

# Preflight. A feature-count drift attaches a plausible number to the wrong
# experiment, which is worse than a crash. Every pinned width is constructed here
# too, so a rejected option fails now rather than hours in.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" -- "${SEL_DIM[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq

parts, cur = [], []
for tok in sys.argv[1:]:
    if tok == "--": parts.append(cur); cur = []
    else: cur.append(tok)
parts.append(cur)
versions, wants, dims = parts
wants = [int(x) for x in wants]

bad = []
for version, want in zip(sorted(set(versions)),
                         [wants[versions.index(v)] for v in sorted(set(versions))]):
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
    bad.append("torch is not installed (%s); every arm needs it" % error)

from src.prediction.models import build_wrapper
for dim in dims:
    if not dim:
        continue
    try:
        w = build_wrapper("torch_mlp_emb", "regression", seed=42, n_jobs=1,
                          embed_dim=int(dim))
        widths = w.param_grid[0]["mlp__embed_dim"]
        if widths != [int(dim)]:
            bad.append("embed_dim %s did not pin: grid holds %s" % (dim, widths))
    except Exception as error:
        bad.append("embed_dim %s rejected: %s" % (dim, error))

if bad:
    sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: datasets at the expected feature counts; torch present; "
      "%d pinned width(s) construct and pin cleanly" % len([d for d in dims if d]))
PY

total="${#SEL_NAME[@]}"; index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (${SEL_NAME[*]})
  asks      how wide should the ticker embedding be? Each width its own model.
  control   e12_emb0_raw -- no embedding module at all, not a zero-width one
  cadence   rolling ${EMB_WINDOW}/${EMB_REFIT}, embargo ${EMB_EMBARGO}, ${EMB_N_SPLITS} folds, ${EMB_N_ITER} draws
  threads   ${MLP_THREADS}   shard ${SHARD}/${OF}
  expect    ~1.5 h emb0, ~2 h each embedding arm; ~11 h for all six
  read      the CURVE across widths, not any single arm, and maxdd beside payout

Where to look while it runs:
  runner    experiments/e12_embed_dim_crypto.sh
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
echo "${EXPERIMENT}: embedding width  |  ${total} arm(s)  |  shard ${SHARD}/${OF}"
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
    [ -n "${SEL_DIM[$i]}" ] && cmd+=(--model-options "{\"embed_dim\": ${SEL_DIM[$i]}}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-20s %-14s dim %-4s %s\n' "$index" "$total" "$run_name" \
               "${SEL_MODEL[$i]}" "${SEL_DIM[$i]:-none}" "${SEL_DATA[$i]}"
        continue
    fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  dim ${SEL_DIM[$i]:-none}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    # A failed arm must not cost the others: read PIPESTATUS[0] after the tee.
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
echo "${EXPERIMENT} -- how wide should the ticker embedding be?"
echo "  results  results/metrics/crypto/{torch_mlp,torch_mlp_emb}/${PREFIX}_*.json"
echo "  runner   experiments/e12_embed_dim_crypto.sh"
echo "=============================================================="
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} arm(s) in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 0 ] && echo "SHARD ${SHARD} of ${OF} -- PARTIAL; the other shards hold the rest"
echo "=============================================================="
echo
python3 - "${PREFIX}" "${MLP_SCOUT:-0}" <<'PY'
import json, os, sys
import numpy as np
import pandas as pd

prefix, scout = sys.argv[1], sys.argv[2] == "1"
WIDTHS = [("torch_mlp", "emb0", 0), ("torch_mlp_emb", "emb4", 4),
          ("torch_mlp_emb", "emb8", 8), ("torch_mlp_emb", "emb16", 16),
          ("torch_mlp_emb", "emb32", 32), ("torch_mlp_emb", "emb64", 64)]
CONTROL = "%s_emb0_raw" % prefix

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

rows = [(m, "%s_%s_raw" % (prefix, a), d, load(m, "%s_%s_raw" % (prefix, a)))
        for m, a, d in WIDTHS]

print("%-20s %5s %6s %10s %10s %8s %8s %10s %8s" % (
    "run", "dim", "n", "CORR", "MMC", "sharpe", "maxdd", "payout", "min"))
print("-" * 96)
for model, run, dim, d in rows:
    if d is None:
        # Never a blank line: absent reads as "tested and null" when it means
        # "never ran". A sharded run legitimately has gaps.
        print("%-20s %5d  NOT ON DISK -- not run (or another shard holds it)"
              % (run, dim))
        continue
    print("%-20s %5d %6d %10.6f %10.6f %8.4f %8.4f %10.6f %8.1f" % (
        run, dim, d["n_periods"], d["corr"]["mean"], d["mmc"]["mean"],
        d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        0.1 * d["corr"]["mean"] + d["mmc"]["mean"], d.get("runtime_minutes", 0)))
print("-" * 96)

if scout:
    print()
    print("SCOUT RUN -- no comparison printed. --era-step 4 moves the split")
    print("boundary, so these arms are a wiring check and nothing more.")
    raise SystemExit

control = load("torch_mlp", CONTROL)
if control is None:
    print()
    print("The control %s is not on disk, so no delta is printed. Run it with"
          % CONTROL)
    print("ONLY=emb0 -- every comparison in this study is against it.")
    raise SystemExit

print()
print("paired vs %s (no embedding), Newey-West lag 20" % CONTROL)
print("%-6s %11s %11s %11s %9s %7s" % (
    "dim", "corr d", "mmc d", "payout d", "se", "t"))
print("-" * 62)
base = per_period("torch_mlp", CONTROL)
curve = []
for model, run, dim, d in rows:
    if d is None or dim == 0:
        continue
    arm = per_period(model, run)
    line = {}
    for metric in ("corr", "mmc", "payout"):
        left = payout(base) if metric == "payout" else base[metric]
        right = payout(arm) if metric == "payout" else arm[metric]
        j = pd.concat([left, right], axis=1, join="inner", keys=["b", "a"])
        line[metric] = newey_west(j["a"] - j["b"])
    curve.append((dim, line["payout"]))
    print("%-6d %+11.6f %+11.6f %+11.6f %9.6f %+7.2f" % (
        dim, line["corr"][0], line["mmc"][0], line["payout"][0],
        line["payout"][1], line["payout"][2]))
print("-" * 62)

if len(curve) >= 3:
    # The shape is worth more than any single arm, so say it out loud rather
    # than leaving it to be eyeballed off six rows.
    deltas = [d for _, (d, _, _) in curve]
    dims = [w for w, _ in curve]
    falling = all(b <= a for a, b in zip(deltas, deltas[1:]))
    rising = all(b >= a for a, b in zip(deltas, deltas[1:]))
    best = dims[int(np.argmax(deltas))]
    print()
    if falling:
        print("SHAPE: payout delta falls monotonically with width -- the table")
        print("reads as a pure cost, and the narrowest arm is the least bad.")
    elif rising:
        print("SHAPE: payout delta rises monotonically with width -- the grid is")
        print("BINDING at %d and the widest arm is not the optimum. Extend it "
              "before quoting a preferred width." % max(dims))
    else:
        print("SHAPE: payout delta peaks at embed_dim %d, so it is neither a pure"
              % best)
        print("cost nor unbounded. Whether the hump is real is what the t column")
        print("has to answer -- with 5 comparisons the honest bar is |t| ~ 2.6.")

print()
print("510+ periods at ~95%% overlap is ~26 effective independent windows, so")
print("|t| < 2 is EXPECTED and IS the finding. Five comparisons put the")
print("Bonferroni-style bar nearer |t| = 2.6 -- quote both. Read maxdd beside")
print("payout: an embedding that leaves the mean alone and cuts drawdown is")
print("still a result.")
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
