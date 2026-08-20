#!/usr/bin/env bash
#
# Experiment: does CURUPIRA_02 improve when handed autoencoder codes?
#
# One question, two arms, one variable. The model is CURUPIRA_02's own fitted
# configuration, copied field for field out of src/live/registry.py -- fixed-grid
# ridge, regression, rolling 130-period window, refit every 130, embargo 20,
# 5 splits, 50 search iterations, seed 42, scored on corr against the benchmark
# reference. Nothing about the model moves between the arms.
#
#   c02_raw      crypto/v2.0          22 features   the deployed slot, reproduced
#   c02_raw_ae   crypto/v2.0-aetuned  30 features   the same 22 plus 8 AE codes
#
# CURUPIRA_02 is the only slot in the book on the raw 22 -- every other one needs
# a built dataset -- and it is the cheapest to run, which makes it the right
# place to ask whether a learned feature block earns its keep.
#
# ---------------------------------------------------------------------------
# Why the evaluation cannot be reading its own training data
# ---------------------------------------------------------------------------
#
# Three separate boundaries have to hold, and the script CHECKS all three before
# it runs anything rather than asserting them in a comment:
#
# 1. THE FORECASTER. run_forecast walks forward over the validation block. Each
#    refit trains on the 130 periods ending 20 periods before the block it then
#    predicts, so no fitted model has seen the rows it is scored on, and the
#    20-period gap covers the 20-day forward target's resolution.
#
# 2. THE ENCODER. The autoencoder that produced feature_ae_00..07 was fitted only
#    on periods at or before its recorded fit_boundary_period. If that boundary
#    fell inside the validation block, the codes on the scored rows would carry
#    information from the future of those rows -- silently, with no NaN and no
#    error. The check below refuses to run unless the validation block starts
#    strictly after the boundary.
#
# 3. THE TWO AGREEING. The encoder's boundary is a function of the --val-frac it
#    was BUILT with; the validation block is a function of the --val-frac the run
#    uses. Build at 0.3 and run at 0.5 and the block moves earlier, back across
#    the boundary. Both are pinned to AE_VAL_FRAC here and compared.
#
# As built: the encoder saw nothing after 2024-06-26 and the scored block starts
# 2024-07-25, so every scored row is 29+ days beyond anything either the encoder
# or the estimator was fitted on.
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Read the PAIRED per-period delta, not the two means. The arms share their
# periods, their folds and their seed, so almost all of their variance is common
# -- ablation_report.py measured t=0.75 unpaired against t=3.14 paired on the
# same data. Newey-West at lag 20, because a 20-day target on daily dates makes
# consecutive periods ~95% overlapping.
#
# The prior is FLAT. Eleven feature blocks have been tried on this panel and none
# has beaten the raw 22 by a margin the sample supports. What is different here
# is only that the codes are learned rather than specified -- and the tuning
# search that produced them maximised RECONSTRUCTION, which is not what this
# measures. A tuned autoencoder that reconstructs far better than PCA and still
# adds nothing here is a clean answer, and the likeliest one.
#
# Usage:
#   bash experiments/curupira02_ae_crypto.sh
#   DRY_RUN=1 bash experiments/curupira02_ae_crypto.sh
#   NO_MAIL=1 bash experiments/curupira02_ae_crypto.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# AE_-prefixed on purpose. GNU screen exports $WINDOW (the window number, 0) into
# every session it starts, so a bare WINDOW= would silently become 0.
AE_VAL_FRAC="${AE_VAL_FRAC:-0.3}"
AE_TUNED="${AE_TUNED:-crypto/v2.0-aetuned}"
AE_RAW="${AE_RAW:-crypto/v2.0}"

# CURUPIRA_02's fit block, verbatim from src/live/registry.py.
C02=(--task regression --metric corr --bmc-reference benchmark
     --cv-scheme rolling --train-window 130 --refit-every 130 --embargo 20
     --n-splits 5 --n-iter 50 --seed 42 --val-frac "${AE_VAL_FRAC}")

started_all=$(date +%s)
STATUS="unknown"
SUMMARY=""

# Mails on failure AND on success, following src/live/notify.py's rule: a job
# that only mails when it breaks is indistinguishable from a job that silently
# stopped running. SMTP is also the least reliable and least important thing
# here, so a mail failure is reported and never propagated.
send_mail() {
    local subject="$1" body_file="$2"
    [ "${NO_MAIL:-0}" = "1" ] && { echo "(NO_MAIL=1, not mailing)"; return 0; }
    python3 - "$subject" "$body_file" <<'PY'
import sys
try:
    from src.live.notify import send_text
    ok = send_text(sys.argv[1], open(sys.argv[2]).read())
    print("mail sent" if ok else "mail NOT sent (mailer unconfigured or refused)")
except Exception as error:                      # never let SMTP break the run
    print("mail failed: %s" % error)
PY
}

REPORT="logs/c02_ae_report.txt"

finish() {
    local code="$1"
    {
        echo "CURUPIRA_02 raw vs raw+autoencoder"
        echo "$(date '+%Y-%m-%d %H:%M:%S')   elapsed $(( ($(date +%s) - started_all) / 60 )) min"
        echo
        echo "$SUMMARY"
    } > "$REPORT"

    if [ "$code" -eq 0 ]; then
        send_mail "c02-ae SUCCESS" "$REPORT"
    else
        send_mail "c02-ae FAILED (exit ${code})" "$REPORT"
    fi
    exit "$code"
}

fail() {
    STATUS="failed"
    SUMMARY="FAILED: $*"
    echo "$*" >&2
    finish 1
}

# --------------------------------------------------------------------------- #
# Refuse to run on the wrong data, or on data that would leak
# --------------------------------------------------------------------------- #
GUARD=$(python3 - "$AE_RAW" "$AE_TUNED" "$AE_VAL_FRAC" <<'PY'
import json, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq

raw_version, ae_version, val_frac = sys.argv[1], sys.argv[2], float(sys.argv[3])

def n_features(version):
    path = "data/%s/train.parquet" % version
    return sum(c.startswith("feature_") for c in pq.ParquetFile(path).schema.names)

try:
    n_raw, n_ae = n_features(raw_version), n_features(ae_version)
except OSError as error:
    sys.exit("missing dataset: %s" % error)

if n_raw != 22:
    sys.exit("expected 22 features in %s, found %d" % (raw_version, n_raw))
if n_ae <= n_raw:
    sys.exit("%s has %d features, not more than the %d raw ones -- wrong dataset"
             % (ae_version, n_ae, n_raw))

meta = json.load(open("data/%s/encoder.json" % ae_version))
if meta["fit_scope"] != "train":
    sys.exit("the encoder for %s was built with --fit-scope %s, which is "
             "LOOKAHEAD by construction and must not be scored as an arm"
             % (ae_version, meta["fit_scope"]))
if meta["val_frac"] < val_frac - 1e-9:
    sys.exit("the encoder was built for val_frac=%.2f but this study runs %.2f, "
             "so its fit window reaches into the scored block. Rebuild it with "
             "--val-frac %.2f." % (meta["val_frac"], val_frac, val_frac))

# The check that actually matters: where does scoring start, and did the encoder
# see any of it?
dates = pd.read_parquet("data/%s/train.parquet" % ae_version, columns=["date"])
periods = np.sort(pd.to_datetime(dates["date"]).unique())
val_start = pd.Timestamp(periods[-int(round(len(periods) * val_frac))])
boundary = pd.Timestamp(meta["fit_boundary_period"])

if val_start <= boundary:
    sys.exit("the scored block starts %s but the encoder saw data through %s -- "
             "refusing to run a leaking arm" % (val_start.date(), boundary.date()))

print("%d|%d|%s|%s|%d|%s" % (
    n_raw, n_ae, boundary.date(), val_start.date(), (val_start - boundary).days,
    ",".join(meta["code_columns"])))
PY
) || fail "pre-flight check failed: ${GUARD}"

IFS='|' read -r N_RAW N_AE BOUNDARY VAL_START GAP_DAYS CODE_COLS <<< "$GUARD"

echo "=============================================================="
echo "CURUPIRA_02 raw vs raw+autoencoder"
echo "=============================================================="
echo "  baseline    ${AE_RAW}      ${N_RAW} features"
echo "  treatment   ${AE_TUNED}  ${N_AE} features (+$((N_AE - N_RAW)) codes)"
echo "  encoder saw nothing after ${BOUNDARY}"
echo "  scoring starts            ${VAL_START}  (${GAP_DAYS} days later)"
echo "  model: CURUPIRA_02's own fit block, identical across both arms"
echo

ARM_NAME=(c02_raw c02_raw_ae)
ARM_DATA=("${AE_RAW}" "${AE_TUNED}")

if [ "${DRY_RUN:-0}" = "1" ]; then
    for i in 0 1; do
        echo "run: python3 -u -m src.run_forecast crypto ridge ${C02[*]}" \
             "--data-version ${ARM_DATA[$i]} --run-name ${ARM_NAME[$i]}"
    done
    exit 0
fi

for i in 0 1; do
    arm="${ARM_NAME[$i]}"
    echo "--------------------------------------------------------------"
    echo "[$((i + 1))/2] ${arm} on ${ARM_DATA[$i]}   $(date '+%H:%M:%S')"
    echo "--------------------------------------------------------------"
    python3 -u -m src.run_forecast crypto ridge "${C02[@]}" \
        --data-version "${ARM_DATA[$i]}" --run-name "${arm}" \
        2>&1 | tee "logs/${arm}.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && fail "arm ${arm} exited non-zero; see logs/${arm}.log"
done

# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #
SUMMARY=$(python3 - "${VAL_START}" "${BOUNDARY}" "${CODE_COLS}" <<'PY'
import sys
import pandas as pd
from src.analysis.ablation_report import newey_west

val_start, boundary, code_cols = sys.argv[1], sys.argv[2], sys.argv[3]

# An INNER join on the period index, so the delta is only ever taken between two
# scores of the same date. Anything an arm failed to score drops out of both.
base = pd.read_csv("results/per_period/crypto/ridge/c02_raw.csv", index_col=0)
arm = pd.read_csv("results/per_period/crypto/ridge/c02_raw_ae.csv", index_col=0)
joined = base.join(arm, lsuffix="_base", rsuffix="_arm", how="inner")

if len(joined) != len(base) or len(joined) != len(arm):
    print("WARNING: %d base periods and %d arm periods share only %d dates"
          % (len(base), len(arm), len(joined)))

joined["payout_base"] = 0.1 * joined["corr_base"] + joined["mmc_base"]
joined["payout_arm"] = 0.1 * joined["corr_arm"] + joined["mmc_arm"]

lines = ["scored on %d periods, %s onward -- the encoder saw nothing after %s"
         % (len(joined), val_start, boundary),
         "added features: %s" % code_cols,
         "",
         "%-10s %12s %12s %14s %10s %8s" % (
             "metric", "raw 22", "raw+ae 30", "paired delta", "NW se", "t")]

results = {}
for label in ("corr", "mmc", "payout"):
    delta = (joined[label + "_arm"] - joined[label + "_base"]).dropna()
    mean, se, t = newey_west(delta)
    results[label] = t
    lines += ["%-10s %12.6f %12.6f %+14.6f %10.6f %+8.2f" % (
        label, joined[label + "_base"].mean(), joined[label + "_arm"].mean(),
        mean, se, t)]

t_payout = results["payout"]
verdict = ("the autoencoder codes HELP at |t| > 2" if t_payout > 2 else
           "the autoencoder codes HURT at |t| > 2" if t_payout < -2 else
           "no distinguishable difference")

lines += ["",
          "verdict: %s (paired payout t = %+.2f)" % (verdict, t_payout),
          "",
          "Paired and Newey-West at lag 20: the arms share every period, fold and",
          "seed, so the difference series has a far smaller standard error than",
          "either arm alone, and a 20-day target on daily dates leaves consecutive",
          "periods ~95% overlapping. On ~26 effective independent windows |t| < 2",
          "is the expected result for a saturated feature set, and is itself the",
          "finding rather than an inconclusive one."]

print("\n".join(lines))
PY
) || fail "the paired comparison failed after both arms ran"

echo
echo "=============================================================="
echo "$SUMMARY"
echo "=============================================================="
finish 0
