#!/usr/bin/env bash
#
# Experiment: every CURUPIRA slot, with and without the autoencoder codes
#
# The c02-only study said the eight codes did nothing for the one slot on the raw
# 22. This asks the same question of the whole book: does a learned feature block
# help ANY of the deployed models, or none of them?
#
# Eight slots come out of fourteen model configurations, because four of the
# slots are blends of the other twelve:
#
#   CURUPIRA_01   ridge_glmnet on tenc                     23 features
#   CURUPIRA_02   ridge        on raw                      22
#   CURUPIRA_03   ridge_glmnet on onehot+rolling          786
#   CURUPIRA_04   ridge_glmnet on allfeat++               812
#   CURUPIRA_05   rank-average of five feature-block members
#   CURUPIRA_05_60  the same five on the 60-day target
#   CURUPIRA_06   0.50 * C01 + 0.50 * C05
#   CURUPIRA_07   0.25 each of C01, C04, C05, C05_60
#
# 06 and 07 combine per-period ranks of runs that already exist, so they cost
# seconds once their inputs are there and add no fitting of their own.
#
# ---------------------------------------------------------------------------
# One encoder, applied everywhere
# ---------------------------------------------------------------------------
#
# The eight codes come from ONE autoencoder -- the tuned 256,128 network fitted
# on the raw 22 -- attached to each slot's own dataset with
# `build_autoencoder --encoder-from`. Fitting a separate encoder per dataset was
# the obvious alternative and is wrong: an autoencoder's bottleneck is only
# identified up to rotation, permutation and sign, so feature_ae_03 would mean
# something different in every arm and the table would not be a table of one
# treatment. It would also be re-deriving the same thing from the same inputs,
# since the codes are a function of the raw 22 that every derived dataset still
# carries, and then reading the differences as signal.
#
# ---------------------------------------------------------------------------
# Where the codes go in the two blends
# ---------------------------------------------------------------------------
#
# CURUPIRA_05's five members are feature-block subsets picked by regex, and none
# of those regexes matches feature_ae_*. So the codes have to be put somewhere
# deliberately. Here EVERY member gets them: each member's include becomes
# `<its own block>|^feature_ae_`. The five weights stay at 0.2.
#
# The alternative -- a sixth 'ae' member at 1/6 -- would have kept the
# one-member-per-block reading, but it also changes the blend's structure rather
# than its features, and would have diluted every existing member's weight from
# 0.20 to 0.167. This way the only thing that moves between the arms is what each
# member can see, which is the same treatment the four single-model slots get.
#
# ---------------------------------------------------------------------------
# Why the evaluation is not reading its own training data
# ---------------------------------------------------------------------------
#
# Checked before anything runs, not asserted in a comment:
#
#   the forecaster  walk-forward; each refit trains on the 130 periods ending
#                   `embargo` periods before the block it predicts
#   the encoder     fitted only on periods at or before its fit_boundary_period,
#                   which must fall strictly before the scored block starts
#   the two         the encoder's boundary comes from the --val-frac it was BUILT
#                   with, the scored block from the --val-frac the runs USE; both
#                   are pinned to XAE_VAL_FRAC and compared
#
# Every arm is re-run here, baselines included. The existing runs have identical
# configs and c02_raw reproduced CURUPIRA_02's registry numbers to six decimals,
# so inheriting them would have been defensible -- but a table where one column
# was produced today and the other months ago invites exactly the question this
# is meant to settle.
#
# Usage:
#   bash experiments/curupira_ae_table_crypto.sh
#   DRY_RUN=1 bash experiments/curupira_ae_table_crypto.sh
#   ONLY=c01 bash experiments/curupira_ae_table_crypto.sh
#   NO_MAIL=1 bash experiments/curupira_ae_table_crypto.sh
#   XAE_JOBS=2 bash experiments/curupira_ae_table_crypto.sh   # less parallelism
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# XAE_-prefixed on purpose: GNU screen exports $WINDOW=0 into every session it
# starts, so a bare WINDOW= would silently zero the training window.
XAE_VAL_FRAC="${XAE_VAL_FRAC:-0.3}"
XAE_JOBS="${XAE_JOBS:-4}"          # concurrent run_forecast processes
XAE_NJOBS="${XAE_NJOBS:-8}"        # threads inside each one; 4 x 8 = 32 cores
XAE_ENCODER="${XAE_ENCODER:-crypto/v2.0-aetuned}"

RAW="crypto/v2.0"
TENC="crypto/v2.0-tenc"
ONEHOT="crypto/v2.0-onehot-rollsel"
ALLFEAT="crypto/v2.0-tenc-onehot-rollsel-spread-interact"
RE_AE='^feature_ae_'

# Shared across every arm, both columns. Only --data-version, --feature-include
# and (for the 60-day members) --target/--embargo ever differ.
COMMON=(--task regression --metric corr --bmc-reference benchmark
        --cv-scheme rolling --train-window 130 --refit-every 130
        --n-splits 5 --n-iter 50 --seed 42 --val-frac "${XAE_VAL_FRAC}"
        --n-jobs "${XAE_NJOBS}")

# Six parallel arrays, never a packed pipe-delimited string: every include regex
# below contains "|" as alternation, and splitting on IFS='|' would tear
# "^feature_sym_(?!tenc$)|^feature_ae_" into two useless halves. This is the trap
# ablation_crypto.sh documents having fallen into.
CFG_NAME=(); CFG_MODEL=(); CFG_BASE=(); CFG_AE=(); CFG_INC=(); CFG_EXTRA=()
add_cfg() { CFG_NAME+=("$1"); CFG_MODEL+=("$2"); CFG_BASE+=("$3")
            CFG_AE+=("$4"); CFG_INC+=("$5"); CFG_EXTRA+=("$6"); }

T60=(--target target_binned_return_60 --embargo 60)
T20=(--embargo 20)

#       name        model          base dataset  +ae dataset          include ("-" = all)                extra
add_cfg c01         ridge_glmnet   "$TENC"       "${TENC}-ae"         "-"                                ""
add_cfg c02         ridge          "$RAW"        "$XAE_ENCODER"       "-"                                ""
add_cfg c03         ridge_glmnet   "$ONEHOT"     "${ONEHOT}-ae"       "-"                                ""
add_cfg c04         ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      "-"                                ""
add_cfg m_onehot    ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      '^feature_sym_(?!tenc$)'           ""
add_cfg m_tenc      ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      '^feature_sym_tenc$'               ""
add_cfg m_interact  ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      '_x_|_accel_'                      ""
add_cfg m_size      ridge          "$RAW"        "$XAE_ENCODER"       '^feature_market_cap_(avg|ewa)_'   ""
add_cfg m_price     ridge_glmnet   "$RAW"        "$XAE_ENCODER"       '^feature_close_(avg|ewa)_'        ""
add_cfg t60_onehot  ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      '^feature_sym_(?!tenc$)'           "t60"
add_cfg t60_tenc    ridge_glmnet   "$ALLFEAT"    "${ALLFEAT}-ae"      '^feature_sym_tenc$'               "t60"
add_cfg t60_interact ridge_glmnet  "$ALLFEAT"    "${ALLFEAT}-ae"      '_x_|_accel_'                      "t60"
add_cfg t60_size    ridge          "$RAW"        "$XAE_ENCODER"       '^feature_market_cap_(avg|ewa)_'   "t60"
add_cfg t60_price   ridge_glmnet   "$RAW"        "$XAE_ENCODER"       '^feature_close_(avg|ewa)_'        "t60"

started_all=$(date +%s)
SUMMARY="(the run did not reach the reporting stage)"
REPORT="logs/curupira_ae_table.txt"

send_mail() {
    local subject="$1" body_file="$2"
    [ "${NO_MAIL:-0}" = "1" ] && { echo "(NO_MAIL=1, not mailing)"; return 0; }
    python3 - "$subject" "$body_file" <<'PY'
import sys
try:
    from src.live.notify import send_text
    print("mail sent" if send_text(sys.argv[1], open(sys.argv[2]).read())
          else "mail NOT sent (mailer unconfigured or refused)")
except Exception as error:            # SMTP must never decide the exit status
    print("mail failed: %s" % error)
PY
}

finish() {
    local code="$1"
    { echo "CURUPIRA slots, with and without the autoencoder codes"
      echo "$(date '+%Y-%m-%d %H:%M:%S')   elapsed $(( ($(date +%s) - started_all) / 60 )) min"
      echo; echo "$SUMMARY"; } > "$REPORT"
    cat "$REPORT"
    if [ "$code" -eq 0 ]; then send_mail "curupira-ae-table SUCCESS" "$REPORT"
    else send_mail "curupira-ae-table FAILED (exit ${code})" "$REPORT"; fi
    exit "$code"
}
fail() { SUMMARY="FAILED: $*"; echo "$*" >&2; finish 1; }

# --------------------------------------------------------------------------- #
# Pre-flight: the datasets exist, carry what they should, and do not leak
# --------------------------------------------------------------------------- #
GUARD=$(python3 - "$XAE_VAL_FRAC" "$RAW" "$TENC" "$ONEHOT" "$ALLFEAT" "$XAE_ENCODER" <<'PY'
import json, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq

val_frac = float(sys.argv[1])
raw, tenc, onehot, allfeat, encoder = sys.argv[2:7]
pairs = [(raw, encoder), (tenc, tenc + "-ae"),
         (onehot, onehot + "-ae"), (allfeat, allfeat + "-ae")]

def n_features(version):
    return sum(c.startswith("feature_")
               for c in pq.ParquetFile("data/%s/train.parquet" % version).schema.names)

boundary = None
for base, augmented in pairs:
    try:
        n_base, n_aug = n_features(base), n_features(augmented)
    except OSError as error:
        sys.exit("missing dataset: %s" % error)

    meta = json.load(open("data/%s/encoder.json" % augmented))
    if n_aug - n_base != meta["code_dim"]:
        sys.exit("%s has %d features and %s has %d -- a difference of %d, not the "
                 "%d codes the encoder emits"
                 % (base, n_base, augmented, n_aug, n_aug - n_base, meta["code_dim"]))
    if meta["fit_scope"] != "train":
        sys.exit("%s was built with --fit-scope %s, which is LOOKAHEAD"
                 % (augmented, meta["fit_scope"]))
    if meta["val_frac"] < val_frac - 1e-9:
        sys.exit("%s was built for val_frac=%.2f but this study runs %.2f"
                 % (augmented, meta["val_frac"], val_frac))
    if boundary and meta["fit_boundary_period"] != boundary:
        sys.exit("the augmented datasets disagree about the encoder boundary: "
                 "%s vs %s" % (boundary, meta["fit_boundary_period"]))
    boundary = meta["fit_boundary_period"]

    # Every arm must be a strict superset: the base rows must survive untouched,
    # or the two columns of the table are not measuring the same panel.
    if len(pd.read_parquet("data/%s/train.parquet" % base, columns=["date"])) != \
       len(pd.read_parquet("data/%s/train.parquet" % augmented, columns=["date"])):
        sys.exit("%s and %s have different row counts" % (base, augmented))

# The check that matters: does scoring start after the encoder stopped looking?
dates = pd.read_parquet("data/%s/train.parquet" % raw, columns=["date"])
periods = np.sort(pd.to_datetime(dates["date"]).unique())
val_start = pd.Timestamp(periods[-int(round(len(periods) * val_frac))])
if val_start <= pd.Timestamp(boundary):
    sys.exit("scoring starts %s but the encoder saw through %s -- refusing"
             % (val_start.date(), boundary))

print("%s|%s|%d" % (boundary, val_start.date(),
                    (val_start - pd.Timestamp(boundary)).days))
PY
) || fail "pre-flight failed: ${GUARD}"

IFS='|' read -r BOUNDARY VAL_START GAP <<< "$GUARD"

echo "=============================================================="
echo "CURUPIRA slots, with and without the autoencoder codes"
echo "=============================================================="
echo "  encoder            ${XAE_ENCODER}, saw nothing after ${BOUNDARY}"
echo "  scoring starts     ${VAL_START}  (${GAP} days later)"
echo "  configs            ${#CFG_NAME[@]} x 2 arms, ${XAE_JOBS} at a time"
echo

# --------------------------------------------------------------------------- #
# The 28 model runs
# --------------------------------------------------------------------------- #
launch() {
    local name="$1" model="$2" data="$3" include="$4" extra="$5" run="$6"
    local cmd=(python3 -u -m src.run_forecast crypto "$model" "${COMMON[@]}"
               --data-version "$data" --run-name "$run")
    if [ "$extra" = "t60" ]; then cmd+=("${T60[@]}"); else cmd+=("${T20[@]}"); fi
    [ "$include" != "-" ] && cmd+=(--feature-include "$include")

    if [ "${DRY_RUN:-0}" = "1" ]; then printf '  %s\n' "${cmd[*]}"; return 0; fi
    "${cmd[@]}" > "logs/${run}.log" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then echo "  >>> ${run} FAILED (exit ${status})"
    else echo "  >>> ${run} done"; fi
    return $status
}

RUNS=()
for i in "${!CFG_NAME[@]}"; do
    name="${CFG_NAME[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue

    for arm in base ae; do
        run="xae_${name}_${arm}"
        RUNS+=("$run")
        if [ "$arm" = "base" ]; then
            data="${CFG_BASE[$i]}"; inc="${CFG_INC[$i]}"
        else
            data="${CFG_AE[$i]}"
            if [ "${CFG_INC[$i]}" = "-" ]; then inc="-"
            else inc="${CFG_INC[$i]}|${RE_AE}"; fi
        fi

        # Skip work already done, so an interrupted run resumes cheaply.
        if [ "${DRY_RUN:-0}" != "1" ] && \
           [ -f "results/per_period/crypto/${CFG_MODEL[$i]}/${run}.csv" ]; then
            echo "  ${run} already present, skipping"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] launching ${run} (${CFG_MODEL[$i]}, ${data})"
        launch "$name" "${CFG_MODEL[$i]}" "$data" "$inc" "${CFG_EXTRA[$i]}" "$run" &

        # Cap concurrency: 4 processes x 8 threads fills 32 cores without
        # oversubscribing, which would slow every one of them down.
        while [ "$(jobs -rp | wc -l)" -ge "${XAE_JOBS}" ]; do wait -n; done
    done
done
wait

[ "${DRY_RUN:-0}" = "1" ] && { echo; echo "(dry run: blends and report skipped)"; exit 0; }

echo
echo "[$(date '+%H:%M:%S')] all model runs finished after $(( ($(date +%s) - started_all) / 60 )) min"

# A comparison is only worth as much as its baseline. Before reading any delta,
# check that the four single-model baselines reproduce the numbers the registry
# records for the slots they claim to be. A mismatch means the config is wrong
# and every delta below it is measuring the wrong thing.
echo
echo "[$(date '+%H:%M:%S')] checking the baselines against src/live/registry.py"
python3 - <<'PY'
import json, glob
from src.live import registry

pairs = [("CURUPIRA_01", "xae_c01_base"), ("CURUPIRA_02", "xae_c02_base"),
         ("CURUPIRA_03", "xae_c03_base"), ("CURUPIRA_04", "xae_c04_base")]
for slot, run in pairs:
    hits = glob.glob("results/metrics/crypto/*/%s.json" % run)
    if not hits:
        print("  %-14s %-16s NOT RUN" % (slot, run)); continue
    got = json.load(open(hits[0]))
    want = registry.get(slot)["backtest"]
    marks = []
    for key in ("corr", "mmc"):
        delta = abs(got[key]["mean"] - want[key])
        marks.append("%s %.6f vs %.6f%s" % (
            key, want[key], got[key]["mean"],
            "" if delta < 1e-6 else "  <-- DIFFERS by %.2e" % delta))
    print("  %-14s %s" % (slot, "   ".join(marks)))
PY

# --------------------------------------------------------------------------- #
# The blends: per-period rank averages of runs that now exist
# --------------------------------------------------------------------------- #
blend() {
    local run="$1" weights="$2"; shift 2
    python3 -u -m src.run_ensemble crypto ridge_glmnet "$@" \
        --weights "$weights" --run-name "$run" > "logs/${run}.log" 2>&1 \
        || { echo "  >>> blend ${run} FAILED"; return 1; }
    echo "  >>> blend ${run} done"
}

blend60() {
    local run="$1" weights="$2"; shift 2
    python3 -u -m src.run_ensemble crypto ridge_glmnet "$@" \
        --weights "$weights" --target target_binned_return_60 --run-name "$run" \
        > "logs/${run}.log" 2>&1 || { echo "  >>> blend ${run} FAILED"; return 1; }
    echo "  >>> blend ${run} done"
}

echo "[$(date '+%H:%M:%S')] blending"
for arm in base ae; do
    blend "xae_c05_${arm}" "0.2,0.2,0.2,0.2,0.2" \
        "ridge_glmnet/xae_m_onehot_${arm}" "ridge_glmnet/xae_m_tenc_${arm}" \
        "ridge_glmnet/xae_m_interact_${arm}" "ridge/xae_m_size_${arm}" \
        "ridge_glmnet/xae_m_price_${arm}" || fail "the C05 ${arm} blend failed"

    blend60 "xae_c05_60_${arm}" "0.2,0.2,0.2,0.2,0.2" \
        "ridge_glmnet/xae_t60_onehot_${arm}" "ridge_glmnet/xae_t60_tenc_${arm}" \
        "ridge_glmnet/xae_t60_interact_${arm}" "ridge/xae_t60_size_${arm}" \
        "ridge_glmnet/xae_t60_price_${arm}" || fail "the C05_60 ${arm} blend failed"

    blend "xae_c06_${arm}" "0.5,0.5" \
        "ridge_glmnet/xae_c01_${arm}" "ridge_glmnet/xae_c05_${arm}" \
        || fail "the C06 ${arm} blend failed"

    # C07 mixes a 60-day member into a 20-day book. run_ensemble ranks each
    # member within period before averaging, so the two labels never meet on the
    # same scale -- but the scored target must be the 20-day one, since that is
    # what the slot is paid on.
    blend "xae_c07_${arm}" "0.25,0.25,0.25,0.25" \
        "ridge_glmnet/xae_c01_${arm}" "ridge_glmnet/xae_c04_${arm}" \
        "ridge_glmnet/xae_c05_${arm}" "ridge_glmnet/xae_c05_60_${arm}" \
        || fail "the C07 ${arm} blend failed"
done

# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
SUMMARY=$(python3 - "$BOUNDARY" "$VAL_START" <<'PY'
import glob, json, sys
import pandas as pd
from src.analysis.ablation_report import newey_west

boundary, val_start = sys.argv[1], sys.argv[2]

SLOTS = [("CURUPIRA_01", "c01"), ("CURUPIRA_02", "c02"), ("CURUPIRA_03", "c03"),
         ("CURUPIRA_04", "c04"), ("CURUPIRA_05", "c05"),
         ("CURUPIRA_05_60", "c05_60"), ("CURUPIRA_06", "c06"),
         ("CURUPIRA_07", "c07")]

def per_period(run):
    hits = glob.glob("results/per_period/crypto/*/%s.csv" % run)
    if not hits:
        return None
    return pd.read_csv(hits[0], index_col=0)

def metrics_json(run):
    hits = glob.glob("results/metrics/crypto/*/%s.json" % run)
    return json.load(open(hits[0])) if hits else None

lines = ["scored from %s onward; the encoder saw nothing after %s"
         % (val_start, boundary),
         "paired per-period deltas, Newey-West at lag 20",
         "",
         "%-15s %5s %10s %10s %11s %7s %10s %10s %11s %7s" % (
             "slot", "n", "corr base", "corr +ae", "delta", "t",
             "payout base", "payout +ae", "delta", "t")]

rows, missing = [], []
for label, key in SLOTS:
    base, arm = per_period("xae_%s_base" % key), per_period("xae_%s_ae" % key)
    if base is None or arm is None:
        missing.append(label)
        continue
    j = base.join(arm, lsuffix="_b", rsuffix="_a", how="inner")
    j["payout_b"] = 0.1 * j["corr_b"] + j["mmc_b"]
    j["payout_a"] = 0.1 * j["corr_a"] + j["mmc_a"]

    d_corr, _, t_corr = newey_west((j["corr_a"] - j["corr_b"]).dropna())
    d_pay, _, t_pay = newey_west((j["payout_a"] - j["payout_b"]).dropna())
    rows.append((label, t_pay, d_pay))
    lines.append("%-15s %5d %10.6f %10.6f %+11.6f %+7.2f %10.6f %10.6f %+11.6f %+7.2f"
                 % (label, len(j), j["corr_b"].mean(), j["corr_a"].mean(),
                    d_corr, t_corr, j["payout_b"].mean(), j["payout_a"].mean(),
                    d_pay, t_pay))

# Level and risk move in opposite directions often enough that a mean-only table
# is true and misleading -- the c02 study lost 0.005 corr while gaining 0.06
# sharpe and cutting max drawdown by a quarter.
lines += ["",
          "%-15s %11s %11s %11s %11s" % (
              "slot", "sharpe base", "sharpe +ae", "max_dd base", "max_dd +ae")]
for label, key in SLOTS:
    b, a = metrics_json("xae_%s_base" % key), metrics_json("xae_%s_ae" % key)
    if not b or not a:
        continue
    lines.append("%-15s %11.4f %11.4f %11.4f %11.4f" % (
        label, b["corr"]["sharpe"], a["corr"]["sharpe"],
        b["corr"]["max_drawdown"], a["corr"]["max_drawdown"]))

if missing:
    lines += ["", "MISSING (their runs did not complete): %s" % ", ".join(missing)]

helped = [r for r in rows if r[1] > 2]
hurt = [r for r in rows if r[1] < -2]
lines += ["",
          "slots where the codes help at |t| > 2 on payout: %s"
          % (", ".join(r[0] for r in helped) if helped else "none"),
          "slots where they hurt at |t| > 2:                %s"
          % (", ".join(r[0] for r in hurt) if hurt else "none"),
          "mean payout delta across slots: %+.6f"
          % (sum(r[2] for r in rows) / len(rows) if rows else float("nan")),
          "",
          "Read the paired columns, not the two means. The arms share every",
          "period, fold and seed, so the difference series has a far smaller",
          "standard error than either arm alone -- ablation_report measured",
          "t=0.75 unpaired against t=3.14 paired on this panel. With eight",
          "slots tested at once, a single |t| just over 2 is roughly what",
          "multiplicity produces on its own; treat one lone winner as a",
          "hypothesis to re-run, not a result."]
print("\n".join(lines))
PY
) || fail "the table failed to build after every run completed"

finish 0
