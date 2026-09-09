#!/usr/bin/env bash
#
# Experiment E04: raw features with and without autoencoder codes, on LightGBM
#
# Three arms, one model, one thing varying -- the 8 extra columns:
#
#     raw     crypto/v2.0          22   the baseline
#     ae      crypto/v2.0-aetuned  30   raw + 8 bottleneck autoencoder codes
#     pca     crypto/v2.0-aepca    30   raw + 8 PCA components
#
# THE PCA ARM IS NOT DECORATION. Without it, an autoencoder result answers
# "do 8 extra compressed columns help", not "does the AUTOENCODER help". The
# encoder's own diagnostics put its holdout reconstruction R2 at 0.954 against
# PCA's 0.932 -- close enough that any advantage should be small, and a linear
# control is the only way to attribute one.
#
# ---------------------------------------------------------------------------
# The leakage gate, which is why this runner exists rather than three commands
# ---------------------------------------------------------------------------
#
# The codes come from a FITTED builder, so unlike a spread or a product they
# have a train/validation boundary that can be got wrong. The encoder is fitted
# once, on rows before the boundary; the boundary is derived from --val-frac at
# BUILD time, while the scored block follows the --val-frac the RUN uses. Build
# at 0.3, run at 0.5, and the validation block moves back across the boundary:
# silent lookahead, no error, a better-looking number.
#
# So this refuses to run unless the encoder's recorded boundary sits at or
# before the first scored period, and prints both dates. Do not remove that gate.
#
#     encoder fit_boundary_period  2024-06-26   (val_frac 0.3, seed 42)
#
# ---------------------------------------------------------------------------
# What would count as a result
# ---------------------------------------------------------------------------
#
# The model is PINNED, which is what makes this a feature comparison rather than
# a model comparison. Eleven feature blocks have been tried on this panel and
# none beat raw by a margin the sample supports; the autoencoder is the twelfth.
# On ~26 effective independent windows |t| < 2 is the expected outcome, and a
# null is the finding.
#
# Read the ridge result as the prior: on CURUPIRA_02 the codes lost 0.005 corr
# while GAINING 0.06 sharpe and cutting max drawdown 24%. If that pattern holds
# here, report it as a risk result and say so -- the mean alone would call it a
# loss.
#
# Usage:
#   bash experiments/e04_autoencoder_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e04_autoencoder_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e04_autoencoder_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
EXPERIMENT="${EXPERIMENT:-E04}"
E4_WINDOW="${E4_WINDOW:-6month}"; E4_REFIT="${E4_REFIT:-6month}"
E4_VAL_FRAC="${E4_VAL_FRAC:-0.3}"; E4_N_SPLITS="${E4_N_SPLITS:-2}"
E4_N_JOBS="${E4_N_JOBS:-8}"; E4_SEED="${E4_SEED:-42}"
case "${E4_WINDOW}" in ""|0|[0-9]) echo "E4_WINDOW looks wrong" >&2; exit 1;; esac
PARAMS='{"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"min_child_samples":313,"reg_lambda":1.757511}'

ARM=(raw ae pca); DATA=(crypto/v2.0 crypto/v2.0-aetuned crypto/v2.0-aepca); NF=(22 30 30)

# Gate on feature counts AND on the leakage boundary. Computing where the scored
# block starts, rather than trusting that the build and the run agree.
python3 - "${E4_VAL_FRAC}" "${DATA[@]}" -- "${NF[@]}" <<'PY' || exit 1
import sys, os, json
import pyarrow.parquet as pq, pandas as pd
val_frac=float(sys.argv[1]); cut=sys.argv.index("--")
versions=sys.argv[2:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,w in zip(versions,wants):
    p="data/%s/train.parquet"%v
    if not os.path.exists(p): bad.append("%s MISSING"%v); continue
    g=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    if g!=w: bad.append("%s has %d features, expected %d"%(v,g,w))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d datasets, feature counts match"%len(versions))

# where this run will start scoring
dates=sorted(pd.unique(pq.read_table("data/crypto/v2.0/train.parquet",
                                     columns=["date"]).to_pandas()["date"]))
first_scored=pd.Timestamp(dates[int(len(dates)*(1-val_frac))])
for v in versions:
    j="data/%s/encoder.json"%v
    if not os.path.exists(j): continue
    m=json.load(open(j)); b=pd.Timestamp(m["fit_boundary_period"])
    ok = b <= first_scored and m["val_frac"] >= val_frac - 1e-9
    print("  %-22s encoder fit through %s | first scored period %s | %s"
          %(v, b.date(), first_scored.date(),
            "OK, no lookahead" if ok else "*** LOOKAHEAD ***"))
    if not ok:
        sys.exit("  encoder built for val_frac=%.2f but this run uses %.2f -- its "
                 "fit window covers periods this run scores"%(m["val_frac"],val_frac))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  python3 -c "import sys;from src.live.notify import send_text;send_text('[crypto] %s started'%sys.argv[1],sys.argv[2])" \
    "${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  raw vs raw+autoencoder vs raw+PCA, LightGBM regression, estimator pinned
  3 arms, ~6 min
  encoder fit boundary 2024-06-26, verified against the first scored period

Where to look:
  runner    experiments/e04_autoencoder_lgb_reg_crypto.sh
  results   results/metrics/crypto/lgb/e04_*.json
  report    logs/e04_report.txt
  handle    E04; every artefact is e04_*" || echo "note: start email failed"
fi

started=$(date +%s); failed=0
echo "=============================================================="
echo "${EXPERIMENT}: raw vs +autoencoder vs +PCA, model pinned"
echo "=============================================================="
for i in "${!ARM[@]}"; do
  n="e04_${ARM[$i]}"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "  would run $n on ${DATA[$i]} (${NF[$i]} feat)"; continue; fi
  echo; echo "--- $n  ${DATA[$i]}  $(date '+%H:%M:%S')"
  python3 -u -m src.run_forecast crypto lgb --task regression --metric corr \
    --cv-scheme rolling --train-window "${E4_WINDOW}" --refit-every "${E4_REFIT}" \
    --data-version "${DATA[$i]}" --fixed-params "${PARAMS}" \
    --val-frac "${E4_VAL_FRAC}" --n-splits "${E4_N_SPLITS}" --n-jobs "${E4_N_JOBS}" \
    --era-step 1 --seed "${E4_SEED}" --run-name "$n" 2>&1 | tee "logs/${n}.log" >/dev/null
  [ "${PIPESTATUS[0]}" -ne 0 ] && { failed=$((failed+1)); echo "  FAILED $n"; } || echo "  done $n"
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e04_report.txt"
{
echo "${EXPERIMENT} -- raw vs autoencoder vs PCA, LightGBM regression"
echo "  results results/metrics/crypto/lgb/e04_*.json"
echo "  ran in $(( ($(date +%s)-started)/60 )) min, ${failed} failed"
echo
python3 - <<'PY'
import json, os
import numpy as np, pandas as pd
def pp(r):
    f=pd.read_csv("results/per_period/crypto/lgb/%s.csv"%r)
    return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
base=pp("e04_raw")
print("%-10s %5s %10s %10s %8s %8s %10s %11s %7s"%(
  "arm","feat","CORR","MMC","sharpe","maxdd","payout","d vs raw","t"))
print("-"*94)
for a in ("raw","ae","pca"):
    p="results/metrics/crypto/lgb/e04_%s.json"%a
    if not os.path.exists(p): continue
    d=json.load(open(p)); pay=0.1*d["corr"]["mean"]+d["mmc"]["mean"]
    if a=="raw": dd=t=None
    else:
        j=pd.concat([payout(base),payout(pp("e04_"+a))],axis=1,join="inner",keys=["b","a"])
        dd,t=nw(j["a"]-j["b"])
    print("%-10s %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s"%(
      a,d["config"]["n_features"],d["corr"]["mean"],d["mmc"]["mean"],
      d["corr"]["sharpe"],d["corr"]["max_drawdown"],pay,
      "baseline" if dd is None else "%+.6f"%dd,"-" if t is None else "%+.2f"%t))
print("-"*94)
if os.path.exists("results/per_period/crypto/lgb/e04_pca.csv"):
    j=pd.concat([payout(pp("e04_pca")),payout(pp("e04_ae"))],axis=1,join="inner",keys=["p","a"])
    d,t=nw(j["a"]-j["p"])
    print("autoencoder vs its PCA control: %+.6f  t %+.2f%s"
          %(d,t,"" if abs(t)>2 else "   (not distinguishable -- any gain is compression, not the AE)"))
print("\nRead risk as well as level: the ridge autoencoder arm lost 0.005 corr")
print("while gaining 0.06 sharpe and cutting max drawdown 24%.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo FAILED || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s"%(sys.argv[3],sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
