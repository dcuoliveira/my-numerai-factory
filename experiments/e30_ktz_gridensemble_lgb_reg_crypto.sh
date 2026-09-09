#!/usr/bin/env bash
#
# Experiment E30: ensemble the whole hyperparameter grid instead of searching it
#
# E30 is the reference handle; every artefact is e30_*.
#
# THE IDEA. E29 ran one lgb_ktz SEARCH per dataset and kept the single best
# config. E30 asks the opposite: fit EVERY config in a grid as its own pinned
# walk-forward model, then RANK-AVERAGE their out-of-sample predictions into one
# ensemble per dataset -- a "bag over hyperparameters" / poor-man's Bayesian
# model average -- and see whether averaging beats picking.
#
# THE GRID (54 cells per dataset), drawn entirely from E27's good corner so we
# are averaging plausible models, not wasting compute on known-bad corners:
#
#     num_leaves        {2, 4, 8}
#     learning_rate     {0.001, 0.003, 0.01}
#     colsample_bytree  {0.05, 0.10, 0.20}
#     n_estimators      {2000, 4000}          L = lr*n_est <= 0.01*4000 = 40
#     FIXED: max_depth=-1, min_child_samples=100, subsample=0.8, reg_lambda=10
#
# Every cell is a --fixed-params pinned lgb walk-forward (no search); the models
# differ ONLY in the four grid axes. Per dataset the cells are then combined with
# run_ensemble (equal-weight per-period rank average). That average IS the E30
# model for that dataset.
#
# DATASETS: the 9 the submitted book trains on, same set as E28/E29:
#   raw(22) tenc(23) interact(36) fenccausal(44) cumlag3(88)
#   tenc_onehot(763) onehot_rollsel(786) allfeat(788) allfeatpp(812)
#
#   9 datasets x 54 cells = 486 pinned runs + 9 ensembles.
#
# WHAT WOULD COUNT AS A RESULT
#   Per dataset, three numbers: the grid-ENSEMBLE, the single BEST cell (an
#   oracle upper bound -- not achievable live, it needs the test scores to pick),
#   and the E29 SEARCH-and-pick arm (the achievable comparator). The question is
#   whether the ensemble beats E29's picked model, paired Newey-West lag 20. The
#   prior from the whole programme is that averaging buys risk (lower drawdown,
#   steadier sharpe) more than level -- report it that way.
#
# RUNTIME. A cell is ~1 min (small trees, n_splits=2). 486 cells is ~8 h on an
# idle box and more under contention. Expensive datasets (wide feature sets) run
# their cells no slower here because num_leaves<=8 keeps trees tiny. Use SHARD/OF
# to split across machines and ONLY to run a subset.
#
# Usage:
#   bash experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
#   DRY_RUN=1 bash experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
#   ONLY="raw interact" bash experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
#   SHARD=1 OF=3 bash experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
#   NO_MAIL=1 bash experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E30}"
E30_WINDOW="${E30_WINDOW:-6month}"
E30_REFIT="${E30_REFIT:-6month}"
E30_N_SPLITS="${E30_N_SPLITS:-2}"
E30_N_JOBS="${E30_N_JOBS:-6}"
E30_SEED="${E30_SEED:-42}"
case "${E30_WINDOW}" in ""|0|[0-9]) echo "E30_WINDOW looks wrong (screen \$WINDOW=0)" >&2; exit 1;; esac

# dataset name -> version
DS_NAME=(raw tenc interact fenccausal cumlag3 tenc_onehot onehot_rollsel allfeat allfeatpp)
DS_VER=(crypto/v2.0 crypto/v2.0-tenc crypto/v2.0-interact crypto/v2.0-fenccausal crypto/v2.0-cumlag3 \
        crypto/v2.0-tenc-onehot crypto/v2.0-onehot-rollsel crypto/v2.0-tenc-onehot-spread-interact \
        crypto/v2.0-tenc-onehot-rollsel-spread-interact)

NLS=(2 4 8); LRS=(0.001 0.003 0.01); CSS=(0.05 0.10 0.20); NES=(2000 4000)

python3 - "${DS_VER[@]}" <<'PY' || exit 1
import sys, os, pyarrow.parquet as pq
bad=[v for v in sys.argv[1:] if not os.path.exists("data/%s/train.parquet"%v)]
if bad: sys.exit("preflight FAILED, missing datasets: %s"%bad)
# verify every grid cell satisfies L<=40
for lr in (0.001,0.003,0.01):
    for ne in (2000,4000):
        if lr*ne>40+1e-6: sys.exit("cell lr=%s ne=%s has L=%.0f>40"%(lr,ne,lr*ne))
print("preflight OK: %d datasets present; all 54 grid cells have L<=40"%len(sys.argv[1:]))
PY

lrtag(){ printf "%s" "${1/0./}"; }   # 0.003 -> 003
nek(){ echo "$(( $1/1000 ))k"; }     # 2000 -> 2k
started_all=$(date +%s)
SHARD="${SHARD:-1}"; OF="${OF:-1}"
declare -a FAILED=(); ran=0; ens_done=0; dsi=0

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; send_text(
'[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  idea      ensemble the whole lgb_ktz grid per dataset instead of searching it
  grid      54 cells: nl{2,4,8} x lr{.001,.003,.01} x cs{.05,.10,.20} x ne{2000,4000}, L<=40
  datasets  9 book datasets  ->  486 pinned cells + 9 rank-average ensembles
  compares  ensemble vs E29 search-and-pick vs the oracle best single cell
  runtime   ~8 h idle, more under contention; SHARD/OF splits it

  runner    experiments/e30_ktz_gridensemble_lgb_reg_crypto.sh
  results   results/metrics/crypto/lgb/e30_*   (cells) and e30_*_ens (ensembles)
  report    logs/e30_report.txt
  handle    ${EXPERIMENT}; every artefact is e30_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: grid-ensemble lgb_ktz  |  9 datasets x 54 cells  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for di in "${!DS_NAME[@]}"; do
    ds="${DS_NAME[$di]}"; ver="${DS_VER[$di]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${ds} "* ]] && continue
    dsi=$((dsi+1))
    [ "${OF}" -gt 1 ] && [ $(( (dsi-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    members=()
    for nl in "${NLS[@]}"; do for lr in "${LRS[@]}"; do for cs in "${CSS[@]}"; do for ne in "${NES[@]}"; do
        cell="e30_${ds}_nl${nl}_lr$(lrtag $lr)_cs$(lrtag $cs)_ne$(nek $ne)"
        members+=("$cell")
        [ -f "results/metrics/crypto/lgb/${cell}.json" ] && continue   # resume-safe
        params="{\"num_leaves\":${nl},\"learning_rate\":${lr},\"colsample_bytree\":${cs},\"n_estimators\":${ne},\"max_depth\":-1,\"min_child_samples\":100,\"subsample\":0.8,\"subsample_freq\":1,\"reg_lambda\":10}"
        if [ "${DRY_RUN:-0}" = "1" ]; then echo "  cell ${cell}"; continue; fi
        python3 -u -m src.run_forecast crypto lgb --task regression --metric corr \
            --cv-scheme rolling --train-window "${E30_WINDOW}" --refit-every "${E30_REFIT}" \
            --data-version "${ver}" --fixed-params "${params}" \
            --n-splits "${E30_N_SPLITS}" --n-jobs "${E30_N_JOBS}" --era-step 1 \
            --seed "${E30_SEED}" --run-name "${cell}" -q >> "logs/e30_${ds}.log" 2>&1 \
            && ran=$((ran+1)) || { FAILED+=("${cell}"); echo "  FAILED ${cell}"; }
    done; done; done; done
    [ "${DRY_RUN:-0}" = "1" ] && { echo "[${ds}] 54 cells -> ensemble e30_${ds}_ens"; continue; }
    # rank-average every cell that produced a forecast
    have=(); for m in "${members[@]}"; do [ -f "results/forecast/crypto/lgb/${m}.csv" ] && have+=("$m"); done
    if [ "${#have[@]}" -ge 2 ]; then
        if python3 -u -m src.run_ensemble crypto lgb "${have[@]}" --run-name "e30_${ds}_ens" -q \
               >> "logs/e30_${ds}.log" 2>&1; then
            ens_done=$((ens_done+1))
            echo ">>> ${ds}: ensembled ${#have[@]}/54 cells -> e30_${ds}_ens"
            # Reclaim the per-row cell FORECASTS now that the ensemble is built.
            # They are large (~240 MB/dataset) intermediates on a SHARED QUOTA;
            # the per-period series and metrics json are kept, so the report and
            # the resume guard (which checks metrics json) are unaffected.
            rm -f results/forecast/crypto/lgb/e30_${ds}_nl*.csv
        else
            echo ">>> ${ds}: ENSEMBLE FAILED"
        fi
    else
        echo ">>> ${ds}: too few cells (${#have[@]}) to ensemble"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/e30_report.txt"
{
echo "${EXPERIMENT} -- grid-ensemble lgb_ktz vs search-and-pick (E29)"
echo "  cells ran this pass: ${ran} | ensembles built: ${ens_done} | failed cells: ${#FAILED[@]}"
[ "${#FAILED[@]}" -gt 0 ] && printf 'failed: %s\n' "${FAILED[*]:0:8}..."
echo "  time: $(( ($(date +%s)-started_all)/60 )) min"
echo "=============================================================="
python3 - <<'PY'
import json, glob, os, re
import numpy as np, pandas as pd
def ser(model,run):
    p="results/per_period/crypto/%s/%s.csv"%(model,run)
    if not os.path.exists(p): return None
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
def payout(f): return 0.1*f["corr"]+f["mmc"]
def nw(x,lag=20):
    x=np.asarray(x,float);n=len(x);m=x.mean();e=x-m;v=(e@e)/n
    for k in range(1,min(lag,n-1)+1): v+=2*(1-k/(lag+1))*((e[k:]@e[:-k])/n)
    se=np.sqrt(max(v,0)/n); return m,(m/se if se>0 else float("nan"))
def st(p):
    c=p.cumsum(); return p.mean(),p.mean()/p.std(ddof=0),(np.maximum.accumulate(c)-c).max()
DS=["raw","tenc","interact","fenccausal","cumlag3","tenc_onehot","onehot_rollsel","allfeat","allfeatpp"]
print("%-15s %6s %10s %8s %8s | %10s %8s | %10s  %s"%(
  "dataset","cells","ENS pay","sharpe","maxdd","best cell","(oracle)","E29 pay","ens vs E29 (t)"))
print("-"*108)
for ds in DS:
    ens=ser("lgb","e30_%s_ens"%ds)
    if ens is None: continue
    cells=[os.path.basename(p)[:-5] for p in glob.glob("results/metrics/crypto/lgb/e30_%s_nl*.json"%ds)]
    cellpays=[]
    for c in cells:
        f=ser("lgb",c)
        if f is not None: cellpays.append(payout(f).mean())
    ep=payout(ens); em,esh,edd=st(ep); best=max(cellpays) if cellpays else float("nan")
    e29=ser("lgb_ktz","e29_%s"%ds)
    d=t=None; e29p=float("nan")
    if e29 is not None:
        j=pd.concat([ep,payout(e29)],axis=1,join="inner",keys=["ens","e29"])
        e29p=payout(e29).mean(); d,t=nw(j["ens"]-j["e29"])
    print("%-15s %6d %10.6f %8.4f %8.4f | %10.6f %8s | %10.6f  %s"%(
      ds,len(cells),em,esh,edd,best,"",e29p,
      "-" if t is None else "%+.6f (t %+.2f)"%(d,t)))
print("-"*108)
print("ENS = equal-weight per-period rank average of the dataset's grid cells.")
print("best cell = highest single-cell payout (ORACLE, needs test scores to pick -- not achievable live).")
print("E29 = the search-and-pick arm. Newey-West lag 20. +t => the grid-ensemble beats search-and-pick.")
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${#FAILED[@]}" -gt 0 ] && echo "WITH ${#FAILED[@]} FAILED CELLS" || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit 0
