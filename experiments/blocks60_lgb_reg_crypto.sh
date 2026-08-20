#!/usr/bin/env bash
#
# Experiment BOITATA: the LightGBM feature-block table, on the 60-day target
#
# Named for the reference handle. Everything this study produces is greppable as
# BOITATA in the logs and mail, and as boitata_* in results/{metrics,forecast,
# per_period,hyperparams}/crypto/lgb/.
#
# Reruns the arm board's main table against target_binned_return_60 instead of
# _20. Same 23 arms, same pinned estimator, same two leaf sizes -- the label is
# the only thing that changes, so the two tables can be read side by side.
#
#     leaf 313  raw tenc spread interact fenccausal fencglobal onehot
#               tenc_onehot onehot_rollsel tenc_onehot_rollsel
#               allfeat allfeatpp allfeatppp                      (13 arms)
#     leaf  50  raw interact onehot tenc_onehot onehot_rollsel
#               tenc_onehot_rollsel onehot_interact
#               allfeat allfeatpp allfeatppp                      (10 arms)
#
# ---------------------------------------------------------------------------
# Three things the longer horizon changes, none of them cosmetic
# ---------------------------------------------------------------------------
#
# 1. THE EMBARGO TRIPLES. --embargo auto derives it from the target, so a 60-day
#    label gets 60 periods between train and validation, not 20. Passing the
#    20-day default here would leak three times over, silently, with a
#    better-looking score as the only symptom.
#
# 2. THE PANEL SHRINKS. The 60-day target is NaN over its last 40 dates, where
#    the window has not resolved -- about 12,000 rows. Runs land on ~470
#    periods, so NOTHING here is directly comparable to a 510-period 20-day arm.
#    Compare within this table, or against the t60_* ridge arms.
#
# 3. THE EFFECTIVE SAMPLE COLLAPSES. Consecutive daily periods overlap ~98% on a
#    60-day target against ~95% on a 20-day one. 470 periods are roughly EIGHT
#    effective independent windows, not the ~26 the 20-day table gets. Newey-West
#    at lag 20 is the wrong correction here -- use lag 60. Almost nothing will
#    clear |t| > 2, and that is the honest answer rather than a weak study.
#
# The estimator stays pinned to the configuration chosen on the 20-DAY raw
# features. That is deliberate: holding the model fixed is what makes this a
# clean label comparison. It is not a claim that the config is optimal for the
# 60-day problem -- a cadence and hyperparameter study on this label has not
# been run, and the first table's cadence cross does not transfer.
#
# Usage:
#   bash experiments/blocks60_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/blocks60_lgb_reg_crypto.sh
#   LEAVES="313" bash experiments/blocks60_lgb_reg_crypto.sh
#   ONLY="raw interact" bash experiments/blocks60_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/blocks60_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-BOITATA}"
T60_TARGET="${T60_TARGET:-target_binned_return_60}"
T60_WINDOW="${T60_WINDOW:-6month}"
T60_REFIT="${T60_REFIT:-6month}"
T60_N_SPLITS="${T60_N_SPLITS:-2}"
T60_N_JOBS="${T60_N_JOBS:-8}"
T60_SEED="${T60_SEED:-42}"
LEAVES="${LEAVES:-313 50}"
case "${T60_WINDOW}" in ""|0|[0-9]) echo "T60_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

BASE_PARAMS='"n_estimators":4327,"learning_rate":0.002572,"max_depth":8,"num_leaves":255,"colsample_bytree":0.070213,"subsample":0.816153,"subsample_freq":1,"reg_lambda":1.757511'

# name / dataset / expected features / which leaf sizes it runs at
NAME=(); DATA=(); NFEAT=(); LEAFSET=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); LEAFSET+=("$4"); }
add raw                 "crypto/v2.0"                                                    22 "313 50"
add tenc                "crypto/v2.0-tenc"                                               23 "313"
add spread              "crypto/v2.0-spread"                                             33 "313"
add interact            "crypto/v2.0-interact"                                           36 "313 50"
add fenccausal          "crypto/v2.0-fenccausal"                                         44 "313"
add fencglobal          "crypto/v2.0-fencglobal"                                         44 "313"
add onehot              "crypto/v2.0-onehot"                                            762 "313 50"
add tenc_onehot         "crypto/v2.0-tenc-onehot"                                       763 "313 50"
add onehot_interact     "crypto/v2.0-onehot-interact"                                   776 "50"
add onehot_rollsel      "crypto/v2.0-onehot-rollsel"                                    786 "313 50"
add tenc_onehot_rollsel "crypto/v2.0-tenc-onehot-rollsel"                               787 "313 50"
add allfeat             "crypto/v2.0-tenc-onehot-spread-interact"                       788 "313 50"
add allfeatpp           "crypto/v2.0-tenc-onehot-rollsel-spread-interact"               812 "313 50"
add allfeatppp          "crypto/v2.0-tenc-onehot-rollsel-spread-interact-fenccausal"    834 "313 50"

python3 - "${T60_TARGET}" "${DATA[@]}" -- "${NFEAT[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
target=sys.argv[1]; cut=sys.argv.index("--")
versions=sys.argv[2:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,w in zip(versions,wants):
    p="data/%s/train.parquet"%v
    if not os.path.exists(p): bad.append("%s MISSING"%v); continue
    names=pq.ParquetFile(p).schema.names
    g=sum(c.startswith("feature_") for c in names)
    if g!=w: bad.append("%s has %d features, expected %d"%(v,g,w))
    if target not in names: bad.append("%s has no column %r"%(v,target))
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d datasets carry %s with the expected feature counts"%(len(versions),target))
PY

total=0
for i in "${!NAME[@]}"; do for L in ${LEAVES}; do
  [[ " ${LEAFSET[$i]} " == *" ${L} "* ]] && total=$((total+1)); done; done
index=0; ran=0; failed=0
declare -a FAILED_ARMS=()
started_all=$(date +%s)
# Mail on START as well as at the end. A job that only mails when it finishes is
# indistinguishable from a job that never started, and these run for hours -- the
# start note is what makes "is it running?" answerable without an ssh session.
# It carries REFERENCES, not just a heartbeat: where to watch it, and the handle
# to grep for afterwards.
if [ "${NO_MAIL:-0}" != "1" ]; then
    START_MSG="${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  arms      ${total}  (leaf sizes ${LEAVES})
  target    ${T60_TARGET}  --  embargo auto, 60 periods
  cadence   ${T60_WINDOW} window / ${T60_REFIT} refit, estimator pinned
  expect    ~60-90 min

Where to look while it runs:
  runner    experiments/blocks60_lgb_reg_crypto.sh
  progress  logs/boitata_*.log        (one per arm)
  results   results/metrics/crypto/lgb/boitata_*.json
  report    logs/boitata_report.txt   (written at the end)
  handle    this study is BOITATA; every artefact it writes is boitata_*

A second mail follows on completion, with the table and SUCCESS or FAILED."
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); \
print('   emailed the start notice')" "${EXPERIMENT}" "${START_MSG}" \
        || echo "note: start email failed, the run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: lgb feature blocks on ${T60_TARGET}  |  ${total} arms"
echo "leaf sizes: ${LEAVES}   embargo: auto (60 periods for this target)"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!NAME[@]}"; do
  for L in ${LEAVES}; do
    [[ " ${LEAFSET[$i]} " != *" ${L} "* ]] && continue
    arm="${NAME[$i]}"; index=$((index+1))
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    run_name="boitata_${arm}_m${L}"
    params="{${BASE_PARAMS},\"min_child_samples\":${L}}"
    cmd=(python3 -u -m src.run_forecast crypto lgb --task regression --metric corr
         --target "${T60_TARGET}" --embargo auto
         --cv-scheme rolling --train-window "${T60_WINDOW}" --refit-every "${T60_REFIT}"
         --data-version "${DATA[$i]}" --fixed-params "${params}"
         --n-splits "${T60_N_SPLITS}" --n-jobs "${T60_N_JOBS}" --era-step 1
         --seed "${T60_SEED}" --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
      printf '[%2d/%2d] %-36s leaf=%-4s %s\n' "$index" "$total" "$run_name" "$L" "${DATA[$i]}"; continue; fi
    ran=$((ran+1))
    echo; echo "=============================================================="
    echo "[${index}/${total}] ${run_name}  $(date '+%H:%M:%S')"
    echo "=============================================================="
    started=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "logs/${run_name}.log"
    status="${PIPESTATUS[0]}"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
      failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
      echo ">>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing"
    else
      echo ">>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
  done
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/boitata_report.txt"
{
echo "${EXPERIMENT} -- lgb feature blocks on ${T60_TARGET}"
echo "  results  results/metrics/crypto/lgb/boitata_*.json"
echo "  runner   experiments/blocks60_lgb_reg_crypto.sh"
echo "  logs     logs/boitata_*.log"
echo "=============================================================="
echo "ran ${ran} of ${total} arms in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed arms: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
echo
python3 - <<'PY'
import json, glob, os, re
import numpy as np, pandas as pd
def pp(r):
    p="results/per_period/crypto/lgb/%s.csv"%r
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=60):                      # lag 60, not 20: a 60-day target
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
rows=[]
for p in sorted(glob.glob("results/metrics/crypto/lgb/boitata_*.json")):
    d=json.load(open(p)); c=d["config"]; n=os.path.basename(p)[:-5]
    m=re.match(r"boitata_(.+)_m(\d+)$",n)
    rows.append(dict(run=n,arm=m.group(1),leaf=int(m.group(2)),feat=c["n_features"],
        corr=d["corr"]["mean"],mmc=d["mmc"]["mean"],sharpe=d["corr"]["sharpe"],
        maxdd=d["corr"]["max_drawdown"],payout=0.1*d["corr"]["mean"]+d["mmc"]["mean"]))
if not rows: print("no results"); raise SystemExit
t=pd.DataFrame(rows)
print("%-22s %5s %5s %10s %10s %8s %8s %10s %11s %7s"%(
  "arm","leaf","feat","CORR","MMC","sharpe","maxdd","payout","d vs raw","t"))
print("-"*112)
for _,r in t.sort_values("payout",ascending=False).iterrows():
    base=t[(t.arm=="raw")&(t.leaf==r["leaf"])]
    d=tt=None
    if len(base) and base["run"].iloc[0]!=r["run"]:
        j=pd.concat([payout(pp(base["run"].iloc[0])),payout(pp(r["run"]))],
                    axis=1,join="inner",keys=["b","a"])
        d,tt=nw(j["a"]-j["b"])
    print("%-22s %5d %5d %10.6f %10.6f %8.4f %8.4f %10.6f %11s %7s"%(
      r["arm"],r["leaf"],r["feat"],r["corr"],r["mmc"],r["sharpe"],r["maxdd"],r["payout"],
      "baseline" if d is None else "%+.6f"%d, "-" if tt is None else "%+.2f"%tt))
print("-"*112)
print("Newey-West lag 60 (a 60-day target overlaps ~98%% between daily periods).")
print("470 periods is roughly EIGHT effective independent windows -- expect nulls.")
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
