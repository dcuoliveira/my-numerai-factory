#!/usr/bin/env bash
#
# LSTM: which objective should the network be trained on?
#
# WHAT THIS RUNS
#   Three arms differing in EXACTLY ONE THING -- the training loss:
#     corr      -numerai-style correlation with the target
#     mixture   -(1.0 * BMC + 0.1 * CORR)          <- the current default
#     bmc       -contribution over the reference
#   Everything else is held: seq 21 overlapping, train_buffer 0, rolling 130/130,
#   LOSS_TRIALS seeds keeping LOSS_TOPFRAC, epochs 8, patience 3, seed 42,
#   crypto/v2.0, embed_dim 50, hidden 60, lr 0.001, batch 32.
#
#   All three are SCORED identically -- numerai_corr and correlation_contribution
#   from numerai_tools, the same functions for every arm. Only what they were
#   TRAINED to maximise differs. That is the whole comparison.
#
# WHY EACH ARM REFITS ITS OWN SEEDS
#   The cv-scheme study shared one seed checkpoint across arms, because
#   select_seeds reads neither cv_scheme nor train_window. It DOES read the loss:
#   seeds are trained under the objective and ranked by it. Sharing a checkpoint
#   here would hand every arm the seeds that happened to suit whichever loss ran
#   first. So Phase 1 runs three times, and that is most of the cost.
#
# WHY ROLLING RATHER THAN EXPANDING
#   Settled by lstm_cv_scheme_crypto.sh: on 510 paired periods the two are
#   indistinguishable on payout (delta +0.0001, t = +0.01) and rolling's Phase 2 is
#   26x cheaper -- 8 min against 211. For a three-arm comparison that is the right
#   harness. Note rolling carried worse risk there (max_dd 2.54 vs 0.80), so read
#   these arms against each other, not against the expanding runs.
#
# THE SIGN GATE -- "be sure to maximize those quantities"
#   Every loss returns the NEGATED statistic so the trainer, which minimises, is
#   maximising corr / bmc. That is easy to assert and easy to get wrong, so the
#   runner PROVES it before spending any GPU time: it checks the extremes, the
#   mixture arithmetic, and that a few gradient steps actually raise the real
#   numerai_corr. A sign error would otherwise show up as three arms that all
#   quietly minimise their objective and score near zero.
#
# WHAT WOULD COUNT AS A RESULT
#   |t| >= 2 on a paired corr or payout delta. On ~26 effective independent windows
#   a null is the expected outcome and IS the finding. Report it with its interval.
#
# A CAVEAT ON THE BMC ARM
#   BMC is taken against meta_proxy -- a Ridge stand-in fitted on the 37 training
#   periods where the real meta model overlaps the train block. The bmc arm is
#   therefore optimising contribution over a thin proxy, not over the real meta
#   model. MMC (against the real thing) is reported for every arm as the check.
#
# COST  ~3 x (Phase 1 + Phase 2). At LOSS_TRIALS=10 that is roughly 5-6 h.
#   DRY_RUN=1 print commands   NO_MAIL=1 skip the mail   ONLY=/SKIP= select arms
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefixed: GNU screen exports $WINDOW=0 into every session it starts.
LOSS_TRIALS="${LOSS_TRIALS:-10}"
LOSS_TOPFRAC="${LOSS_TOPFRAC:-0.3}"
LOSS_EPOCHS="${LOSS_EPOCHS:-8}"
LOSS_PATIENCE="${LOSS_PATIENCE:-3}"
LOSS_SEQ="${LOSS_SEQ:-21}"
LOSS_BUFFER="${LOSS_BUFFER:-0}"
LOSS_REFIT="${LOSS_REFIT:-130}"
LOSS_WINDOW="${LOSS_WINDOW:-130}"
LOSS_BMC_W="${LOSS_BMC_W:-1.0}"
LOSS_CORR_W="${LOSS_CORR_W:-0.1}"
LOSS_REFERENCE="${LOSS_REFERENCE:-meta_proxy}"
LOSS_DATA="${LOSS_DATA:-crypto/v2.0}"
LOSS_NFEAT="${LOSS_NFEAT:-22}"
LOSS_SEED="${LOSS_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
NO_MAIL="${NO_MAIL:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

# Parallel arrays, never pipe-delimited records.
ARM_NAMES=(corr mixture bmc)
ARM_RUNS=(lstm_loss_corr lstm_loss_mixture lstm_loss_bmc)
ARM_LOSS=(corr mixture bmc)

mkdir -p logs
REPORT="logs/lstm_loss_report.txt"
STATUS=0
FAILED=()

# --- feature-count gate ----------------------------------------------------- #
ACTUAL=$(python3 - <<PY
import pyarrow.parquet as pq
v = "${LOSS_DATA}".split("/")[-1]
s = pq.ParquetFile("data/crypto/%s/train.parquet" % v).schema_arrow
print(sum(1 for n in s.names if n.startswith("feature_")))
PY
)
[ "$ACTUAL" = "$LOSS_NFEAT" ] || { echo "FAIL  ${LOSS_DATA} has ${ACTUAL} features, expected ${LOSS_NFEAT}" >&2; exit 1; }
echo "ok    ${LOSS_DATA}: ${ACTUAL} features"

# --- sign gate: prove every loss MAXIMISES its quantity ---------------------- #
python3 - <<PY || { echo "FAIL  sign gate: a loss is not maximising its quantity" >&2; exit 1; }
import sys, numpy as np, pandas as pd, torch
from numerai_tools.scoring import numerai_corr
from src.prediction.nn.losses import CorrLoss, BMCLoss, MixtureLoss

torch.manual_seed(0); np.random.seed(0)
n = 600
target = torch.tensor(np.random.choice([0., .25, .5, .75, 1.], n), dtype=torch.float32).unsqueeze(0)
bad = []

# 1. Extremes. A negated statistic means perfect agreement is the MOST NEGATIVE.
perfect = CorrLoss()(target.clone(), target).item()
worst   = CorrLoss()(-target.clone(), target).item()
print("  sign gate  CorrLoss(pred==target) %+.4f | CorrLoss(pred==-target) %+.4f" % (perfect, worst))
if not (perfect < -0.99 and worst > 0.99):
    bad.append("CorrLoss extremes are not -1 / +1")

# 2. Contributing nothing over the reference must score zero.
ref = torch.rand(1, n)
zero = BMCLoss()(ref.clone(), target, reference=ref).item()
print("  sign gate  BMCLoss(pred==reference) %+.4f  (must be ~0)" % zero)
if abs(zero) > 1e-3:
    bad.append("BMCLoss is not zero when the prediction IS the reference")

# 3. Mixture arithmetic: 1.0 * bmc + 0.1 * corr, independent multipliers.
sig = torch.rand(1, n)
b = BMCLoss()(sig, target, reference=ref).item()
c = CorrLoss()(sig, target).item()
m = MixtureLoss(bmc_weight=${LOSS_BMC_W}, corr_weight=${LOSS_CORR_W})(sig, target, reference=ref).item()
expect = ${LOSS_BMC_W} * b + ${LOSS_CORR_W} * c
print("  sign gate  mixture %+.5f vs %.1f*bmc + %.1f*corr = %+.5f" % (m, ${LOSS_BMC_W}, ${LOSS_CORR_W}, expect))
if abs(m - expect) > 1e-6:
    bad.append("MixtureLoss is not the weighted sum of its parts")

# 4. The one that matters: does DESCENDING the loss RAISE the real numerai_corr?
w = torch.nn.Parameter(torch.randn(1, n) * 0.01)
opt = torch.optim.Adam([w], lr=0.05)
def real_corr():
    with torch.no_grad():
        # numerai_corr does targets.to_frame()[targets.name], so the Series MUST
        # be named -- an unnamed one dies with KeyError: None.
        return float(numerai_corr(pd.DataFrame({"p": w[0].numpy()}),
                                  pd.Series(target[0].numpy(), name="t")).iloc[0])
first_loss, first_corr = CorrLoss()(w, target).item(), real_corr()
for _ in range(120):
    opt.zero_grad(); CorrLoss()(w, target).backward(); opt.step()
last_loss, last_corr = CorrLoss()(w, target).item(), real_corr()
print("  sign gate  loss %+.4f -> %+.4f   numerai_corr %+.4f -> %+.4f"
      % (first_loss, last_loss, first_corr, last_corr))
if not (last_loss < first_loss and last_corr > first_corr):
    bad.append("descending CorrLoss did not raise numerai_corr")

if bad:
    for b_ in bad: print("  FAIL  " + b_)
    sys.exit(1)
print("  sign gate  PASSED -- every loss maximises its quantity")
PY
echo "ok    sign gate passed"

run_arm() {
    local name="$1" run="$2" loss="$3"
    local extra=""
    [ "$loss" = "mixture" ] && extra="--bmc-weight ${LOSS_BMC_W} --corr-weight ${LOSS_CORR_W}"

    local cmd="python3 -u -m src.run_nn_forecast crypto \
--loss ${loss} ${extra} --bmc-reference ${LOSS_REFERENCE} \
--seq-length ${LOSS_SEQ} --overlapping --train-buffer ${LOSS_BUFFER} \
--data-version ${LOSS_DATA} \
--cv-scheme rolling --train-window ${LOSS_WINDOW} --refit-every ${LOSS_REFIT} \
--max-trials ${LOSS_TRIALS} --top-frac ${LOSS_TOPFRAC} \
--epochs ${LOSS_EPOCHS} --patience ${LOSS_PATIENCE} \
--seed ${LOSS_SEED} --run-name ${run}"

    if [ "$DRY_RUN" = "1" ]; then echo "DRY  $cmd"; return 0; fi

    echo "==> arm ${name} -> ${run}  ($(date +%H:%M))"
    $cmd 2>&1 | tee "logs/${run}.log"
    # PIPESTATUS, not $?: $? is tee's and is always 0.
    local rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
        echo "FAIL  arm ${name} exited ${rc}" >&2
        FAILED+=("$name"); STATUS=1; return 1     # a failed arm must not cost the others
    fi
    echo "ok    arm ${name} done ($(date +%H:%M))"
}

for i in "${!ARM_NAMES[@]}"; do
    name="${ARM_NAMES[$i]}"
    { [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; } && { echo "skip  ${name}"; continue; }
    { [ -n "$SKIP" ] && [ "$SKIP" = "$name" ]; } && { echo "skip  ${name}"; continue; }
    run_arm "$name" "${ARM_RUNS[$i]}" "${ARM_LOSS[$i]}"
done

[ "$DRY_RUN" = "1" ] && exit 0

# --- report ----------------------------------------------------------------- #
python3 .claude/skills/crypto-experiment-reporting/scripts/paired_delta.py \
    --model lstm --baseline "${ARM_RUNS[0]}" --arms "${ARM_RUNS[1]}" "${ARM_RUNS[2]}" \
    > "$REPORT" 2>&1 || echo "(paired_delta failed -- see above)" >> "$REPORT"

python3 - >> "$REPORT" 2>&1 <<'PY'
import json, os
print()
print("LEVEL AND RISK   (all arms scored identically; only the training loss differs)")
print("%-20s %-9s %-7s %-8s %-9s %-9s %-9s %-6s %s" % (
    "arm","corr","sharpe","max_dd","bmc","mmc","payout","min","early-stop/block"))
for run in ("lstm_loss_corr","lstm_loss_mixture","lstm_loss_bmc"):
    p = "results/metrics/crypto/lstm/%s.json" % run
    if not os.path.exists(p):
        print("%-20s MISSING (arm did not finish)" % run); continue
    d = json.load(open(p))
    bmc = (d.get("bmc") or {}).get("mean")
    mmc = (d.get("mmc") or {}).get("mean")
    payout = 0.1*d["corr"]["mean"] + (mmc or 0.0)
    es = [b.get("early_stopping") for b in d["refits"]]
    print("%-20s %-9.5f %-7.2f %-8.2f %-9s %-9s %-9.5f %-6.0f %s" % (
        run, d["corr"]["mean"], d["corr"]["sharpe"], d["corr"]["max_drawdown"],
        ("%.5f" % bmc) if bmc is not None else "-",
        ("%.5f" % mmc) if mmc is not None else "-",
        payout, d["runtime_minutes"], es))
    if not all(es):
        print("   WARNING: early stopping did not engage in every block -- not a clean arm")
print()
print("Read the diagonal: each arm should lead on the quantity it was trained to")
print("maximise. If the corr arm does not lead on corr, or the bmc arm on bmc, the")
print("objective is not reaching the score -- which is a finding in itself.")
PY

echo; cat "$REPORT"

if [ "$NO_MAIL" != "1" ]; then
    # Mail on success AND failure; never let SMTP decide the exit status.
    python3 - <<PY || echo "WARN  mail failed (the run itself is unaffected)"
import os
from src.live.env import load_env
load_env()
from src.live import notify
notify.ENV = os.environ
verdict = "SUCCESS" if ${STATUS} == 0 else "FAILED: ${FAILED[*]:-}"
notify.send_text("LSTM loss-function study %s" % verdict, open("${REPORT}").read())
PY
fi

exit $STATUS
