#!/usr/bin/env bash
#
# Fit LSTM symbol embeddings on the autoencoder's train block, then attach them to
# a dataset ridge and lgb can read.
#
# WHAT THIS RUNS
#   1. src.data_mani.fit_symbol_embedding -- SEM_SEEDS networks, SEM_EPOCHS epochs
#      each, NO early stopping, on the pipeline's train block (the same rows
#      build_autoencoder.py fits its encoder on). Every seed's embedding matrix is
#      exported separately.
#   2. src.data_mani.build_symbol_embedding -- folds the best-ranked seed's matrix
#      into crypto/v2.0, producing crypto/v2.0${SEM_SUFFIX}.
#
# WHAT IT IS MEASURED AGAINST
#   Nothing yet -- this BUILDS a feature. The comparison is a later ridge/lgb arm
#   on crypto/v2.0${SEM_SUFFIX} against the same arm on crypto/v2.0, paired and
#   Newey-West at lag 20. The raw 22 features are currently the stronger baseline
#   (0.173 CORR), and eleven feature blocks have failed to beat them, so the prior
#   is that this does not help either.
#
# THE COVERAGE CEILING -- read this before spending the GPU time
#   A single-boundary export can only serve rows AFTER that boundary, because the
#   embedding is learned from the target and build_symbol_embedding enforces
#   train_end <= date - lag. Measured on a smoke build: 31.5% of rows clear the
#   causal rule, and only 60.7% of those belong to a symbol that existed in the fit
#   block -- 19.1% of rows end up with a vector, the rest are zeros.
#
#   That is NOT the autoencoder's situation: its codes are an unsupervised
#   transform applied to every row, so they cover 100%. The practical consequence
#   is that a rolling 130/130 ridge arm sees an all-zero feature while training its
#   FIRST block (that window is entirely pre-boundary) and a live one afterwards.
#   Later blocks are fine. Judge the arm on that basis, or raise coverage with the
#   per-block export from run_nn_forecast --export-embeddings instead.
#
# COST
#   ~SEM_EPOCHS x SEM_SEEDS epochs over ~250k windows, about 95 s/epoch on this
#   box: 100 epochs x 5 seeds is roughly 13 h. Halve SEM_SEEDS to halve it. Step 2
#   is seconds.
#
#   DRY_RUN=1  print commands, run nothing        NO_MAIL=1  skip the report mail
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefixed: GNU screen exports $WINDOW=0 into every session it starts.
SEM_EPOCHS="${SEM_EPOCHS:-100}"
SEM_SEEDS="${SEM_SEEDS:-5}"
SEM_DIM="${SEM_DIM:-16}"
SEM_SEQ="${SEM_SEQ:-21}"
SEM_BUFFER="${SEM_BUFFER:-0}"
SEM_LOSS="${SEM_LOSS:-mixture}"
SEM_BMC_W="${SEM_BMC_W:-1.0}"
SEM_CORR_W="${SEM_CORR_W:-0.1}"
SEM_RANKFRAC="${SEM_RANKFRAC:-0.15}"
SEM_SOURCE="${SEM_SOURCE:-crypto/v2.0}"
SEM_SUFFIX="${SEM_SUFFIX:--symemb}"
SEM_NAME="${SEM_NAME:-symemb_e${SEM_EPOCHS}_d${SEM_DIM}}"
SEM_NFEAT="${SEM_NFEAT:-22}"
SEM_ENCODER="${SEM_ENCODER:-data/crypto/v2.0-aetuned/encoder.json}"
DRY_RUN="${DRY_RUN:-0}"
NO_MAIL="${NO_MAIL:-0}"

mkdir -p logs models/embeddings
EXPORT="models/embeddings/${SEM_NAME}.parquet"
REPORT="logs/${SEM_NAME}_report.txt"
STATUS=0

# Gate on feature count: a silent drift attaches a plausible number to the wrong
# experiment, which is worse than a crash.
ACTUAL=$(python3 - <<PY
import pyarrow.parquet as pq
v = "${SEM_SOURCE}".split("/")[-1]
s = pq.ParquetFile("data/crypto/%s/train.parquet" % v).schema_arrow
print(sum(1 for n in s.names if n.startswith("feature_")))
PY
)
[ "$ACTUAL" = "$SEM_NFEAT" ] || { echo "FAIL  ${SEM_SOURCE} has ${ACTUAL} features, expected ${SEM_NFEAT}" >&2; exit 1; }
echo "ok    ${SEM_SOURCE}: ${ACTUAL} features"

# The leakage boundary is VERIFIED against the encoder, not asserted: --val-frac is
# a run-time flag, so an embedding built at one value and an encoder at another sit
# on different train samples while both still say "the train block".
[ -f "$SEM_ENCODER" ] || { echo "FAIL  no encoder.json at ${SEM_ENCODER} to verify the boundary against" >&2; exit 1; }

FIT_CMD="python3 -u -m src.data_mani.fit_symbol_embedding \
--source-version ${SEM_SOURCE} --verify-boundary ${SEM_ENCODER} \
--epochs ${SEM_EPOCHS} --seeds ${SEM_SEEDS} --embed-dim ${SEM_DIM} \
--seq-length ${SEM_SEQ} --overlapping --train-buffer ${SEM_BUFFER} \
--rank-frac ${SEM_RANKFRAC} --loss ${SEM_LOSS} \
--bmc-weight ${SEM_BMC_W} --corr-weight ${SEM_CORR_W} --name ${SEM_NAME}"

ATTACH_CMD="python3 -u -m src.data_mani.build_symbol_embedding \
--export ${EXPORT} --source-version ${SEM_SOURCE} --out-suffix=${SEM_SUFFIX}"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY  $FIT_CMD"; echo "DRY  $ATTACH_CMD"; exit 0
fi

echo "==> fitting embeddings ($(date +%H:%M))"
$FIT_CMD 2>&1 | tee "logs/${SEM_NAME}_fit.log"
# PIPESTATUS, not $?: $? is tee's status and is always 0.
if [ "${PIPESTATUS[0]}" != "0" ]; then
    echo "FAIL  the fit step failed" >&2; STATUS=1
else
    echo "==> attaching to ${SEM_SOURCE}${SEM_SUFFIX} ($(date +%H:%M))"
    $ATTACH_CMD 2>&1 | tee "logs/${SEM_NAME}_attach.log"
    [ "${PIPESTATUS[0]}" = "0" ] || { echo "FAIL  the attach step failed" >&2; STATUS=1; }
fi

{
    if [ "$STATUS" = "0" ]; then echo "symbol embedding build SUCCESS"; else echo "symbol embedding build FAILED"; fi
    echo
    echo "config: ${SEM_EPOCHS} epochs (no early stopping), ${SEM_SEEDS} seeds, "
    echo "        embed_dim ${SEM_DIM}, seq ${SEM_SEQ} overlapping, loss ${SEM_LOSS}"
    echo
    [ -f "models/embeddings/${SEM_NAME}.json" ] && python3 - <<PY
import json
m = json.load(open("models/embeddings/${SEM_NAME}.json"))
print("fit boundary : %s (%d periods, %d trained on, %d ranking)" % (
    m["fit_boundary_period"], m["fit_periods"], m["trained_periods"], m["rank_periods"]))
print("symbols      : %d get a vector" % m["n_symbols"])
print("early stop   : %s" % m["early_stopping"])
print()
print("seeds, by ranking loss (ranking only -- it never chose an epoch):")
for r in sorted(m["seeds"], key=lambda r: r["rank_loss"]):
    print("  seed %-6d rank %+.4f  final train %+.4f" % (
        r["seed"], r["rank_loss"], r["final_train_loss"]))
PY
    echo
    sed -n '/rows with a usable embedding/,$p' "logs/${SEM_NAME}_attach.log" 2>/dev/null | head -6
    echo
    echo "NEXT: score it against the raw baseline, paired and NW lag 20:"
    echo "  python3 -m src.run_forecast crypto ridge_glmnet --cv-scheme rolling \\"
    echo "      --train-window 130 --refit-every 130 --data-version ${SEM_SOURCE}${SEM_SUFFIX} \\"
    echo "      --run-name w6r6_symemb"
    echo "  python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \\"
    echo "      --model ridge_glmnet --baseline w6r6_glmnet_reg --arms w6r6_symemb"
} > "$REPORT" 2>&1

cat "$REPORT"

if [ "$NO_MAIL" != "1" ]; then
    # Mail on success AND failure; never let SMTP decide the exit status.
    python3 - <<PY || echo "WARN  mail failed (the build itself is unaffected)"
import os
from src.live.env import load_env
load_env()
from src.live import notify
notify.ENV = os.environ
notify.send_text("symbol embedding build %s" % ("SUCCESS" if ${STATUS} == 0 else "FAILED"),
                 open("${REPORT}").read())
PY
fi

exit $STATUS
