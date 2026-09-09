#!/usr/bin/env bash
#
# E13 -- Autoregressive distributed lags (ADL) against the best raw ridge
#
# An ADL specification regresses y_t on its own past AND on past x:
#
#     y_t = a + SUM_q rho_q y_{t-q} + SUM_p B_p' x_{t-p} + B_0' x_t + e
#
# The DL half has been tested (20 single-lag + 10 cumulative arms, all null).
# The AR half -- lags of the TARGET -- has not, and it is the half that can
# leak.
#
# ---------------------------------------------------------------------------
# Why an AR term here is dangerous, and what the floor is
# ---------------------------------------------------------------------------
#
# target_binned_return_20 at date t is the return over [t, t+20]. It resolves at
# t+20, so a row at t may only condition on a target observed at t-q for q >= 20.
# y_{t-1} overlaps the label by 19 of its 20 days.
#
# Measured, not asserted -- correlation of each AR column with the label:
#
#     arlag20 (legitimate)   -0.0147
#     arlag60 (legitimate)   +0.0190
#     arlag1  (LEAKY)        +0.8393
#     feature_volatility_20d -0.1249   <- the strongest raw feature, for scale
#
# build_adl refuses a target lag below the horizon unless --allow-leaky-ar is
# passed. The e13_leak arm below uses it deliberately, so the size of the leak
# is a measurement in this table rather than a trap for the next reader. It is
# NOT a candidate.
#
# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
#
#   control  raw 22            w6r6_fixed_reg / w6r6_glmnet_reg  (already run)
#   ar1      + y_{t-20}                                 23 feat
#   ar3      + y at 20,25,30                            25
#   ar5      + y at 20,25,30,40,60                      27
#   dl3      + x lags 1,2,3                             88
#   adl1     + x lags 1-3 + y_{t-20}                    89
#   adl3     + x lags 1-3 + y at 20,25,30               91
#   adl5     + x lags 1-3 + y at 20,25,30,40,60         93
#   leak     + y_{t-1}   LOOKAHEAD CONTROL              23
#
# Two estimators, because "the best ridge on raw data" is not one model:
#   ridge         (fixed alpha grid, capped at 100)  payout 0.085763  calmar 0.0518
#   ridge_glmnet  (data-scaled path to ~1.2M)        payout 0.083859  calmar 0.0594
# The fixed grid wins on level, glmnet on risk. ADL widens the design matrix
# from 22 to 93 columns, where a ceiling of 100 may bind -- so both run.
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# Expect null. Fifteen feature blocks have been tried on this panel and none has
# beaten its control by a margin ~26 effective independent windows can support;
# the DL half of this very specification was already flat across 30 arms.
#
# The AR half is a genuinely new object though. If y's own past carries anything
# at a 20-period remove it would show as ar1/ar3/ar5 beating raw while dl3 stays
# flat -- and that would be the first evidence of exploitable dynamics here.
# A null is the finding, reported with its interval.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   bash experiments/e13_adl_crypto.sh
#   DRY_RUN=1 / SCOUT=1 / ONLY=ar3 / MODELS="ridge" / SHARD=1 OF=2 / NO_MAIL=1
#
# 16 arms at ~7.5 min measured -> about 2 hours. Mails at START and at the end,
# on success and on failure.
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# E13_-prefixed. GNU screen exports $WINDOW (the window number, 0) into every
# session it starts, so a bare WINDOW="${WINDOW:-130}" silently becomes 0.
E13_TASK="${E13_TASK:-regression}"
E13_METRIC="${E13_METRIC:-corr}"
E13_SCHEME="${E13_SCHEME:-rolling}"
E13_TRAIN_WINDOW="${E13_TRAIN_WINDOW:-130}"
E13_REFIT="${E13_REFIT:-130}"
E13_EMBARGO="${E13_EMBARGO:-20}"
E13_N_ITER="${E13_N_ITER:-50}"
E13_N_SPLITS="${E13_N_SPLITS:-5}"
E13_SEED="${E13_SEED:-42}"
E13_VAL_FRAC="${E13_VAL_FRAC:-0.3}"
E13_SEARCH_JOBS="${E13_SEARCH_JOBS:-8}"
E13_N_JOBS="${E13_N_JOBS:-1}"
MODELS="${MODELS:-ridge ridge_glmnet}"

case "${E13_TRAIN_WINDOW}" in
    ""|0|[0-9]) echo "E13_TRAIN_WINDOW=${E13_TRAIN_WINDOW} looks wrong (screen exports" \
                     "\$WINDOW=0); expected e.g. 130" >&2; exit 1 ;;
esac

ARM=(); DATA=(); FEAT=()
add() { ARM+=("$1"); DATA+=("$2"); FEAT+=("$3"); }
add ar1   crypto/v2.0-e13ar1   23
add ar3   crypto/v2.0-e13ar3   25
add ar5   crypto/v2.0-e13ar5   27
add dl3   crypto/v2.0-e13dl3   88
add adl1  crypto/v2.0-e13adl1  89
add adl3  crypto/v2.0-e13adl3  91
add adl5  crypto/v2.0-e13adl5  93
add leak  crypto/v2.0-e13leak  23

extra=(); suffix=""
if [ "${SCOUT:-0}" = "1" ]; then
    extra=(--era-step 4); suffix="scout"
    echo "SCOUT: --era-step 4, wiring only. Artefacts are e13scout_*, never e13_*,"
    echo "because a downsampled arm moves the split boundary."
fi

HANDLE="e13${suffix}"
REPORT="logs/${HANDLE}_report.txt"
: > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
mail() {
    [ "${NO_MAIL:-0}" = "1" ] && return 0
    python3 -m src.live.notify --subject "$1" --body-file "$2" >/dev/null 2>&1 \
        || echo "(email failed; see ${REPORT})"
}
trap 'code=$?; if [ "$code" -ne 0 ]; then
        say ""; say "ABORTED exit ${code} at $(date -Is)"
        mail "E13 ADL -- ABORTED (exit ${code})" "${REPORT}"
      fi' EXIT

# Refuse to run on a drifted feature count: a silent drift attaches a plausible
# number to the wrong experiment, which is worse than a crash.
python3 - "${ARM[@]}" --sep "${DATA[@]}" --sep "${FEAT[@]}" <<'PY' || exit 1
import sys
import pyarrow.parquet as pq
parts, cur = [], []
for a in sys.argv[1:]:
    if a == "--sep": parts.append(cur); cur = []
    else: cur.append(a)
parts.append(cur)
arms, datas, feats = parts
for arm, data, want in zip(arms, datas, feats):
    names = pq.ParquetFile("data/%s/train.parquet" % data).schema.names
    n = sum(c.startswith("feature_") for c in names)
    if n != int(want):
        sys.exit("%s (%s) has %d features, expected %s" % (arm, data, n, want))
    ar = sorted(int(c.split("arlag")[-1]) for c in names if "arlag" in c)
    bad = [q for q in ar if q < 20]
    if bad and arm != "leak":
        sys.exit("%s carries sub-horizon AR lag(s) %s -- that is lookahead" % (arm, bad))
    if arm == "leak" and not bad:
        sys.exit("the leak control carries no sub-horizon AR lag; it is not a control")
print("feature + AR-horizon gate OK: %d arms" % len(arms))
PY

total=$(( ${#ARM[@]} * $(echo "${MODELS}" | wc -w) ))
index=0; failed=0; ran=0
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"

START="logs/${HANDLE}_start.txt"
cat > "${START}" <<EOF
E13 started $(date -Is) on $(hostname)

  arms      ${total}  (8 specifications x $(echo "${MODELS}" | wc -w) estimators)
  question  does y's own past, at a legitimate >=20-period remove, add anything
            the 22 raw features do not already carry?
  controls  ridge/w6r6_fixed_reg and ridge_glmnet/w6r6_glmnet_reg (already run)
  expect    ~2 hours

Where to look while it runs:
  runner    experiments/e13_adl_crypto.sh
  progress  logs/${HANDLE}_*.log      (one per arm)
  results   results/metrics/crypto/*/${HANDLE}_*.json
  report    logs/${HANDLE}_report.txt (written at the end)
  handle    this study is E13; every artefact it writes is ${HANDLE}_*

The e13_*_leak arm is a LOOKAHEAD CONTROL (y_{t-1}, corr 0.839 with the label).
It is expected to score best and is not a candidate.
EOF
mail "E13 ADL started -- ${total} arms" "${START}"

say "=============================================================="
say "E13 autoregressive distributed lags | ${total} arms | shard ${SHARD}/${OF}"
say "estimators: ${MODELS}"
say "cadence: ${E13_SCHEME} window=${E13_TRAIN_WINDOW} refit=${E13_REFIT} embargo=${E13_EMBARGO}"
say "controls: ridge/w6r6_fixed_reg payout 0.085763 | ridge_glmnet/w6r6_glmnet_reg payout 0.083859"
say "started $(date -Is)"
say "=============================================================="

for i in "${!ARM[@]}"; do
    arm="${ARM[$i]}"; data="${DATA[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${arm}" ] && continue
    for model in ${MODELS}; do
        index=$((index + 1))
        [ "${OF}" -gt 1 ] && [ $(( (index - 1) % OF + 1 )) -ne "${SHARD}" ] && continue
        tag="fixed"; [ "${model}" = "ridge_glmnet" ] && tag="glmnet"
        run_name="${HANDLE}_${arm}_${tag}"
        cmd=(python3 -u -m src.run_forecast crypto "${model}"
             --task "${E13_TASK}" --metric "${E13_METRIC}" --bmc-reference benchmark
             --cv-scheme "${E13_SCHEME}" --train-window "${E13_TRAIN_WINDOW}"
             --refit-every "${E13_REFIT}" --embargo "${E13_EMBARGO}"
             --n-splits "${E13_N_SPLITS}" --n-iter "${E13_N_ITER}"
             --seed "${E13_SEED}" --val-frac "${E13_VAL_FRAC}"
             --n-jobs "${E13_N_JOBS}" --search-jobs "${E13_SEARCH_JOBS}"
             --data-version "${data}" --run-name "${run_name}")
        cmd+=("${extra[@]+"${extra[@]}"}")
        if [ "${DRY_RUN:-0}" = "1" ]; then
            printf '[%2d/%2d] %-18s %s\n' "$index" "$total" "$run_name" "${data}"
            continue
        fi
        ran=$((ran + 1)); started=$(date +%s)
        "${cmd[@]}" >"logs/${run_name}.log" 2>&1
        status=$?; elapsed=$(( $(date +%s) - started ))
        if [ "$status" -ne 0 ]; then
            failed=$((failed + 1))
            say "$(printf '[%2d/%2d] %-18s FAILED (exit %d) after %ds' "$index" "$total" "$run_name" "$status" "$elapsed")"
        else
            say "$(printf '[%2d/%2d] %-18s done in %4ds' "$index" "$total" "$run_name" "$elapsed")"
        fi
    done
done

[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }
say ""; say "ran ${ran} of ${total} in $(( ($(date +%s) - started_all) / 60 )) min, ${failed} failed"; say ""

python3 -u - "${HANDLE}" >>"${REPORT}" 2>&1 <<'PY'
import json, os, sys
import numpy as np, pandas as pd
h = sys.argv[1]
PP, MET = "results/per_period/crypto", "results/metrics/crypto"
def load(model, run):
    p = os.path.join(PP, model, run + ".csv")
    if not os.path.exists(p): return None, None
    t = pd.read_csv(p).set_index("date")
    return 0.1*t["corr"] + t["mmc"], json.load(open(os.path.join(MET, model, run + ".json")))
def nw(x, lag=20):
    x=np.asarray(x,float); n=len(x); mu=x.mean(); e=x-mu; v=(e@e)/n
    for L in range(1,min(lag,n-1)+1): v+=2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    se=np.sqrt(max(v,0.0)/n); return mu, (mu/se if se>0 else np.nan)
def st(x):
    x=np.asarray(x,float); mu=x.mean(); sd=x.std(ddof=0); cs=x.cumsum()
    dd=(np.maximum.accumulate(cs)-cs).max(); return mu, mu/sd, dd, mu/dd

for model, tag, ctrl in (("ridge","fixed","w6r6_fixed_reg"),
                         ("ridge_glmnet","glmnet","w6r6_glmnet_reg")):
    base, bm = load(model, ctrl)
    if base is None: continue
    print("\n=== %s -- control %s ===" % (model, ctrl))
    print("%-10s %5s %10s %9s %9s %8s %8s | %11s %6s" % (
        "arm","feat","CORR","MMC","payout","p.shp","calmar","d vs ctrl","t"))
    mu,sh,dd,cal = st(base)
    print("%-10s %5d %10.6f %9.6f %9.6f %8.4f %8.4f | %11s %6s" % (
        "control", bm["config"]["n_features"], bm["corr"]["mean"], bm["mmc"]["mean"],
        mu, sh, cal, "-", "-"))
    rows=[]
    for arm in ("ar1","ar3","ar5","dl3","adl1","adl3","adl5","leak"):
        s, m = load(model, "%s_%s_%s" % (h, arm, tag))
        if s is None: continue
        mu,sh,dd,cal = st(s)
        j = pd.concat([base,s],axis=1,join="inner",keys=["b","a"])
        d,t = nw(j["a"]-j["b"])
        flag = "  <- LOOKAHEAD CONTROL, not a candidate" if arm=="leak" else ""
        print("%-10s %5d %10.6f %9.6f %9.6f %8.4f %8.4f | %+11.6f %6.2f%s" % (
            arm, m["config"]["n_features"], m["corr"]["mean"], m["mmc"]["mean"],
            mu, sh, cal, d, t, flag))
        if arm != "leak": rows.append((arm,d,t))
    strong=[r for r in rows if abs(r[2])>2]
    print("  %d of %d candidate arms differ from the control at |t| > 2"
          % (len(strong), len(rows)))
    if rows and not strong:
        print("  None does. Payout deltas span %+.6f to %+.6f -- inside the noise of"
              % (min(r[1] for r in rows), max(r[1] for r in rows)))
        print("  ~26 effective independent windows. ADL terms add nothing detectable.")
PY

trap - EXIT
if [ "${failed}" -eq 0 ]; then mail "E13 ADL -- OK (${ran}/${total})" "${REPORT}"
else mail "E13 ADL -- ${failed} arm(s) failed" "${REPORT}"; fi
echo; echo "report: ${REPORT}"
