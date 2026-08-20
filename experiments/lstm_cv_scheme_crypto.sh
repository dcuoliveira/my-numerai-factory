#!/usr/bin/env bash
#
# LSTM: rolling vs expanding walk-forward, matched.
#
# WHAT THIS RUNS
#   Two arms differing in EXACTLY ONE THING -- the walk-forward scheme:
#     expanding   train from the start of history, refit every 130 periods
#     rolling     train on a 130-period window,   refit every 130 periods
#   Everything else is held: mixture loss (BMC 1.0 / CORR 0.1), seq_length 21,
#   overlapping windows, train_buffer 0, 20 seeds keeping 5, epochs 8, patience 3,
#   seed 42, crypto/v2.0. refit_every is 130 for BOTH so the two arms predict the
#   same blocks and their per-period series pair exactly.
#
# WHAT IT IS MEASURED AGAINST
#   Each other, paired, Newey-West lag 20. Not against the earlier LSTM runs:
#   lstm_w6r6_L21 and lstm_mixture_1y_L21_seeds100 differ in seed count AND
#   train_buffer AND scheme, so neither can settle the scheme question.
#
# WHY train_buffer 0
#   lstm_w6r6_L21 silently lost early stopping: a 130-period rolling block splits
#   into a 71-period fit slice and a 39-period stop slice, and a window needed
#   seq_length + train_buffer = 41 > 39, so no stop window could be built and every
#   seed ran the full 8 epochs unsupervised. At train_buffer 0 a window needs 21,
#   comfortably inside 39. The runner asserts early stopping actually engaged.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on the paired corr or payout delta. On ~26 effective independent
#   windows a null is the expected outcome and IS the finding -- report it with its
#   interval rather than calling it inconclusive.
#
# COST
#   Phase 1 (seed selection) does not depend on the scheme -- select_seeds reads
#   neither cv_scheme nor train_window -- so it is run ONCE for the expanding arm
#   and its checkpoint is copied to the rolling arm, which then resumes instantly.
#   That also makes both arms use the identical seed set. ~3-4 h total.
#
#   DRY_RUN=1  print commands, run nothing      ONLY=rolling   run one arm
#   NO_MAIL=1  skip the report mail             SKIP=expanding skip one arm
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Every var prefixed: GNU screen exports $WINDOW=0 into sessions it starts, so a
# bare WINDOW= silently becomes 0 and every training block comes out empty.
LCV_SEQ="${LCV_SEQ:-21}"
LCV_TRIALS="${LCV_TRIALS:-20}"
LCV_TOPFRAC="${LCV_TOPFRAC:-0.25}"
LCV_EPOCHS="${LCV_EPOCHS:-8}"
LCV_PATIENCE="${LCV_PATIENCE:-3}"
LCV_REFIT="${LCV_REFIT:-130}"
LCV_WINDOW="${LCV_WINDOW:-130}"
LCV_BUFFER="${LCV_BUFFER:-0}"
LCV_DATA="${LCV_DATA:-crypto/v2.0}"
LCV_NFEAT="${LCV_NFEAT:-22}"
LCV_SEED="${LCV_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
NO_MAIL="${NO_MAIL:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

# Parallel arrays, never pipe-delimited records: feature regexes use | as
# alternation, so packing fields and splitting on IFS='|' tears them apart.
ARM_NAMES=(expanding rolling)
ARM_RUNS=(lstm_cv_expanding lstm_cv_rolling)
ARM_FLAGS=("--cv-scheme expanding" "--cv-scheme rolling --train-window ${LCV_WINDOW}")

mkdir -p logs
REPORT="logs/lstm_cv_scheme_report.txt"
STATUS=0
FAILED=()

# --- gate on feature count before running anything -------------------------- #
# A silent count drift attaches a plausible number to the wrong experiment, which
# is worse than a crash.
ACTUAL=$(python3 - <<PY
import pyarrow.parquet as pq
v = "${LCV_DATA}".split("/")[-1]
s = pq.ParquetFile("data/crypto/%s/train.parquet" % v).schema_arrow
print(sum(1 for n in s.names if n.startswith("feature_")))
PY
)
if [ "$ACTUAL" != "$LCV_NFEAT" ]; then
    echo "FAIL  ${LCV_DATA} has ${ACTUAL} features, expected ${LCV_NFEAT}" >&2
    exit 1
fi
echo "ok    ${LCV_DATA}: ${ACTUAL} features"

run_arm() {
    local name="$1" run="$2" flags="$3"
    local log="logs/${run}.log"

    # shellcheck disable=SC2086
    local cmd="python3 -u -m src.run_nn_forecast crypto \
--loss mixture --bmc-weight 1.0 --corr-weight 0.1 \
--seq-length ${LCV_SEQ} --overlapping --train-buffer ${LCV_BUFFER} \
--data-version ${LCV_DATA} --refit-every ${LCV_REFIT} ${flags} \
--max-trials ${LCV_TRIALS} --top-frac ${LCV_TOPFRAC} \
--epochs ${LCV_EPOCHS} --patience ${LCV_PATIENCE} \
--seed ${LCV_SEED} --run-name ${run}"

    if [ "$DRY_RUN" = "1" ]; then echo "DRY  $cmd"; return 0; fi

    echo "==> arm ${name} -> ${run}  ($(date +%H:%M))"
    $cmd 2>&1 | tee "$log"
    # PIPESTATUS, not $?: $? is tee's status and is always 0.
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
        echo "FAIL  arm ${name} exited ${rc}" >&2
        FAILED+=("$name")
        STATUS=1
        return 1        # a failed arm must not cost the others
    fi
    echo "ok    arm ${name} done ($(date +%H:%M))"
    return 0
}

wanted() {
    local n="$1"
    [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && return 1
    [ -n "$SKIP" ] && [ "$SKIP" = "$n" ] && return 1
    return 0
}

for i in "${!ARM_NAMES[@]}"; do
    name="${ARM_NAMES[$i]}"; run="${ARM_RUNS[$i]}"
    wanted "$name" || { echo "skip  arm ${name}"; continue; }

    # Share Phase 1: select_seeds reads neither cv_scheme nor train_window, so the
    # seed search is identical across arms. Copying the checkpoint makes the second
    # arm resume instantly AND guarantees both arms use the same seed set.
    ckpt="results/metrics/crypto/lstm/${run}.seeds.json"
    if [ ! -f "$ckpt" ]; then
        for j in "${!ARM_RUNS[@]}"; do
            donor="results/metrics/crypto/lstm/${ARM_RUNS[$j]}.seeds.json"
            if [ "$j" != "$i" ] && [ -f "$donor" ]; then
                mkdir -p "$(dirname "$ckpt")"
                cp "$donor" "$ckpt"
                echo "ok    reused seed checkpoint from ${ARM_RUNS[$j]}"
                break
            fi
        done
    fi

    run_arm "$name" "$run" "${ARM_FLAGS[$i]}"
done

# --- report ----------------------------------------------------------------- #
if [ "$DRY_RUN" = "1" ]; then exit 0; fi

python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
    --model lstm --baseline "${ARM_RUNS[0]}" --arms "${ARM_RUNS[1]}" \
    > "$REPORT" 2>&1 || echo "(paired_delta failed -- see $REPORT)" >> "$REPORT"

python3 - >> "$REPORT" 2>&1 <<'PY'
import json, os
print()
print("LEVEL AND RISK")
print("%-24s %-9s %-7s %-8s %-9s %-9s %-6s %s" % (
    "run","corr","sharpe","max_dd","mmc","payout","min","early-stop/block"))
for r in ("lstm_cv_expanding","lstm_cv_rolling"):
    p = "results/metrics/crypto/lstm/%s.json" % r
    if not os.path.exists(p):
        print("%-24s MISSING" % r); continue
    d = json.load(open(p))
    payout = 0.1*d["corr"]["mean"] + (d["mmc"]["mean"] if d.get("mmc") else 0.0)
    es = [b.get("early_stopping") for b in d["refits"]]
    print("%-24s %-9.5f %-7.2f %-8.2f %-9.5f %-9.5f %-6.0f %s" % (
        r, d["corr"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        d["mmc"]["mean"], payout, d["runtime_minutes"], es))
    if not all(es):
        print("   WARNING: early stopping did not engage in every block -- "
              "this arm is not a clean test")
PY

echo; cat "$REPORT"

if [ "$NO_MAIL" != "1" ]; then
    # Mail on success AND failure: a job that only mails when it breaks is
    # indistinguishable from one that silently stopped running. And never let SMTP
    # decide the exit status.
    python3 - <<PY || echo "WARN  mail failed (the run itself is unaffected)"
import os
from src.live.env import load_env
load_env()
from src.live import notify
notify.ENV = os.environ
verdict = "SUCCESS" if ${STATUS} == 0 else "FAILED: ${FAILED[*]:-}"
notify.send_text("LSTM cv-scheme study %s" % verdict, open("${REPORT}").read())
PY
fi

exit $STATUS
