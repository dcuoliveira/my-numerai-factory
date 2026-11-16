#!/usr/bin/env bash
#
# E121 -- how much HISTORY should each refit use? The training window of the
#         best single CATBOOST, from one week to six months.
#
# The CatBoost twin of E117 (which asked this of the best single ridge). The
# model is the payout- and MMC-best single catboost of E32: catboost_ktz on
# fenccausal (crypto/v2.0-fenccausal, 44 features = raw 22 + 22 causal per-symbol
# means). That model is deployed as the component IAUARETE_01, and its 130-period
# fit is run catboost_ktz/e32_sel_fenccausal. Everything is held fixed -- data,
# features, 130-period refit cadence, embargo 20, seed, the 12-cell KTZ grid --
# and ONLY the rolling training window moves.
#
# ---------------------------------------------------------------------------
# The windows, in PERIODS. The panel is weekday-only: ~5.01 periods per week,
# ~21.8 per month (measured on the last 260 dates, as E117).
# ---------------------------------------------------------------------------
#
#     1 week  5    2 weeks 10    3 weeks 15    4 weeks 20
#     1 month 22   3 months 65   6 months 130 (the deployed window = IAUARETE_01)
#
# 4 weeks (20) and 1 month (22) are two periods apart and both run because both
# were asked for in E117; expect them indistinguishable.
#
# ---------------------------------------------------------------------------
# Hyperparameters are SEARCHED FROM SCRATCH at every refit, for every window
# ---------------------------------------------------------------------------
#
# The same obstacle E117 hit: a purged inner fold needs training periods + the
# 20-period embargo + validation periods INSIDE the block it searches, so
# purged_period_splits RAISES "no usable CV folds" for a window of 22 periods or
# fewer. The fix is the same --tune-window 130: at each refit the 12-cell KTZ
# search runs on the 130 periods ending at the SAME date the window ends, with
# rolling inner folds that each TRAIN ON THE WINDOW'S LENGTH, and the winning
# config is then refitted on the training window alone. Every arm therefore
# selects its config on the SAME four validation slices; only how much history
# each fold (and the final fit) sees differs.
#
# Unlike ridge_glmnet, catboost_ktz's grid is STATIC (the 12 KTZ cells do not
# depend on N), so there is no lambda-path-scaling subtlety here -- grid_from is
# simply ignored for a wrapper with no fit_grid. The 130 arm needs no tuning
# block (window == tune_window) and reproduces catboost_ktz/e32_sel_fenccausal
# (IAUARETE_01) byte-for-byte.
#
# ---------------------------------------------------------------------------
# What this does NOT vary, and why it matters for reading it
# ---------------------------------------------------------------------------
#
# The REFIT CADENCE stays at 130. A 1-week window is therefore fitted once every
# six months and predicts the next 130 periods from one week of data -- the
# how-much-history question at the cadence the account runs, NOT the freshness
# question (cadence at fixed window, E116's axis). Expect the short windows to
# lose badly: 5 periods at ~95% target overlap is about ONE effective time
# observation, and a 2000-tree booster on ~1,500 rows will overfit hard.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash experiments/e121_catboost_train_window_crypto.sh
#   DRY_RUN=1 bash experiments/e121_catboost_train_window_crypto.sh
#   ONLY=e121_w5 bash experiments/e121_catboost_train_window_crypto.sh
#   E121_SCOUT=1 / NO_MAIL=1 bash experiments/e121_catboost_train_window_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PY="${PY:-/home/danielco/miniconda3/bin/python3}"
TW_N_JOBS="${TW_N_JOBS:-6}"
TW_SEARCH_JOBS="${TW_SEARCH_JOBS:-1}"
TW_ITERS="${TW_ITERS:-2000}"
TW_TUNE_WINDOW="${TW_TUNE_WINDOW:-130}"
DATA="crypto/v2.0-fenccausal"

# TW_TARGET picks the FITTED label. 20 is E121 (the deployed config, fit and
# scored on the 20-day label). 60 is E127 -- the SAME ladder fitted AND tuned on
# target_binned_return_60 with the embargo that label forces (60, not 20), then
# RE-SCORED on the payable 20-day label by src.analysis.e127_report. Only the
# label and its embargo change; windows, 12-cell grid, cadence, seed stay put.
# --tune-window stays 130, so at embargo 60 each search keeps the LAST TWO of
# its inner folds (the production 60-day search, CURUPIRA_05_60, does the same).
# fit-60 arms are named e127_w<N> so they never clobber E121's e121_w<N>.
TW_TARGET="${TW_TARGET:-20}"
case "${TW_TARGET}" in
    20) TARGET_COL="target_binned_return_20"; EMBARGO=20; BASE="e121"; STUDY="E121"; TARGET_ARGS=() ;;
    60) TARGET_COL="target_binned_return_60"; EMBARGO=60; BASE="e127"; STUDY="E127"
        TARGET_ARGS=(--target "${TARGET_COL}") ;;
    *)  echo "TW_TARGET must be 20 or 60, got '${TW_TARGET}'" >&2; exit 1 ;;
esac
EXPERIMENT="${EXPERIMENT:-${STUDY}}"

export OMP_NUM_THREADS="${TW_N_JOBS}" MKL_NUM_THREADS="${TW_N_JOBS}" OPENBLAS_NUM_THREADS="${TW_N_JOBS}"

PREFIX="${BASE}"
if [ "${E121_SCOUT:-0}" = "1" ]; then PREFIX="${BASE}scout"; TW_ITERS=300
    echo "SCOUT MODE: iterations ${TW_ITERS} -- wiring only, NOT comparable."; fi

NAMES=(); WINDOWS=(); WHY=()
add() { NAMES+=("${PREFIX}_$1"); WINDOWS+=("$2"); WHY+=("$3"); }
add w5     5 "1 week"
add w10   10 "2 weeks"
add w15   15 "3 weeks"
add w20   20 "4 weeks"
add w22   22 "1 month"
add w65   65 "3 months"
add w130 130 "6 months (control = e32_sel_fenccausal / IAUARETE_01)"
[ "${TW_TARGET}" = "60" ] && WHY[6]="6 months (longest window; cf. catboost_ktz/e59_fenccausal, catboost on 60d)"

REPORT="logs/${PREFIX}_report.txt"; : > "${REPORT}"
say() { echo "$*" | tee -a "${REPORT}"; }
notify() { [ "${NO_MAIL:-0}" = "1" ] && return 0
    "${PY}" -m src.live.notify --subject "$1" --body-file "${REPORT}" >/dev/null 2>&1 \
        || echo "(email failed; report is in ${REPORT})"; }
trap 'code=$?; if [ "$code" -ne 0 ] && [ "${DRY_RUN:-0}" != "1" ]; then
        say ""; say "ABORTED with exit ${code} at $(date -Is)"
        notify "${EXPERIMENT} catboost train window -- ABORTED (exit ${code})"; fi' EXIT

# Gate: fenccausal present with its 44 features, the FITTED label AND the 20-day
# label (the re-score needs it), the meta file, and the 12-cell KTZ grid.
"${PY}" - "${DATA}" "${TARGET_COL}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
data=sys.argv[1]; target=sys.argv[2]; p="data/%s/train.parquet"%data
names=pq.ParquetFile(p).schema.names
feats=[c for c in names if c.startswith("feature_")]
bad=[]
if len(feats)!=44: bad.append("expected 44 features, found %d"%len(feats))
if target not in names: bad.append("no %s label (the fitted target)"%target)
if "target_binned_return_20" not in names: bad.append("no 20-day label (needed for the re-score)")
if not os.path.exists("data/%s/historical_meta_models.parquet"%data): bad.append("no meta file")
w=build_wrapper("catboost_ktz","regression",seed=42,n_jobs=1)
if len(list(ParameterGrid(w.param_grid)))!=12: bad.append("catboost_ktz grid is not 12 cells")
if bad: sys.exit("gate FAILED:\n  "+"\n  ".join(bad))
print("gate OK: fenccausal 44 features + %s + 20-day label + meta file; catboost_ktz = 12 KTZ cells"%target)
PY

total="${#NAMES[@]}"; index=0; failed=0; ran=0; started_all=$(date +%s)
say "=============================================================="
say "${EXPERIMENT} catboost training window, 1 week to 6 months | ${total} arms"
say "catboost_ktz on fenccausal (44 feat), fit ${TARGET_COL}, refit 130, embargo ${EMBARGO}, iters ${TW_ITERS}"
say "12-cell KTZ grid SEARCHED at every refit; windows < ${TW_TUNE_WINDOW} tuned on ${TW_TUNE_WINDOW} (folds train on the window)"
say "started $(date -Is)"
say "=============================================================="

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    notify "${EXPERIMENT} catboost train window -- started (${total} arms)"
fi

for i in "${!NAMES[@]}"; do
    index=$((index+1)); name="${NAMES[$i]}"; window="${WINDOWS[$i]}"; why="${WHY[$i]}"
    [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name}" ] && continue
    if [ -f "results/metrics/crypto/catboost_ktz/${name}.json" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        say "$(printf '[%d/%d] %-18s already on disk -- skipping' "$index" "$total" "$name")"; continue
    fi
    cmd=("${PY}" -u -m src.run_forecast crypto catboost_ktz
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${window}"
         --refit-every 130 --embargo "${EMBARGO}" "${TARGET_ARGS[@]}"
         --n-splits 5 --n-iter 12 --seed 42 --val-frac 0.3
         --n-jobs "${TW_N_JOBS}" --search-jobs "${TW_SEARCH_JOBS}"
         --data-version "${DATA}" --model-options "{\"iterations\":${TW_ITERS}}"
         --run-name "${name}")
    if [ "${window}" -lt "${TW_TUNE_WINDOW}" ]; then
        cmd+=(--tune-window "${TW_TUNE_WINDOW}"); why="${why}, tuned on ${TW_TUNE_WINDOW}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-12s win %-3s  %s\n' "$index" "$total" "$name" "$window" "$why"; continue
    fi
    ran=$((ran+1)); started=$(date +%s)
    "${cmd[@]}" > "logs/${name}.log" 2>&1
    status="$?"; elapsed=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1))
        say "$(printf '[%d/%d] %-18s FAILED (exit %d) after %5ds  -- %s' "$index" "$total" "$name" "$status" "$elapsed" "$why")"
    else
        say "$(printf '[%d/%d] %-18s done in %5ds  -- %s' "$index" "$total" "$name" "$elapsed" "$why")"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && { trap - EXIT; exit 0; }

say ""; say "ran ${ran} of ${total} in $(( ($(date +%s)-started_all)/60 )) min, ${failed} failed"; say ""
if [ "${TW_TARGET}" != "20" ]; then
    say "fit-${TW_TARGET} arms (${PREFIX}_w*) carry their NATIVE ${TARGET_COL} metrics; Numerai pays on"
    say "the 20-day label, so the deliverable is the consolidation that RE-SCORES them there:"
    say "    ${PY} -m src.analysis.e127_report"
    say "which puts all 14 arms (E121 fit-20 + E127 fit-60) on one common window, paired NW lag 20 and 60."
    trap - EXIT
    [ "${failed}" -eq 0 ] && notify "${EXPERIMENT} catboost train window (fit ${TW_TARGET}d) -- OK (${ran}/${total})" \
                           || notify "${EXPERIMENT} catboost train window (fit ${TW_TARGET}d) -- ${failed} failed"
    echo; echo "report: ${REPORT}"
    exit $([ "${failed}" -eq 0 ] && echo 0 || echo 1)
fi
"${PY}" - "${PREFIX}" <<'PY' 2>&1 | tee -a "${REPORT}"
import os, sys
import numpy as np, pandas as pd
from src.analysis.results_table import payout_series, row_from_frame
from src.analysis.ablation_report import newey_west
prefix=sys.argv[1]
ARMS=[("w5","1wk",5),("w10","2wk",10),("w15","3wk",15),("w20","4wk",20),
      ("w22","1mo",22),("w65","3mo",65),("w130","6mo",130)]
def pp(run):
    p="results/per_period/crypto/catboost_ktz/%s_%s.csv"%(prefix,run)
    if not os.path.exists(p): return None
    f=pd.read_csv(p); return f.set_index("date") if "date" in f.columns else f
frames={a:pp(a) for a,_,_ in ARMS if pp(a) is not None}
if not frames:
    print("no arms on disk yet"); sys.exit()
common=sorted(set.intersection(*[set(f.index) for f in frames.values()]))
base=payout_series(frames["w130"].loc[common]).dropna() if "w130" in frames else None
print("common window: %d periods\n"%len(common))
print("%-6s %-4s %5s %8s %8s %9s %7s %7s | %s"%(
    "arm","win","nper","CORR","MMC","payout","p.shrp","dd(pay)","d vs 6mo (NW t)"))
print("-"*96)
for a,lab,win in ARMS:
    f=frames.get(a)
    if f is None: print("%-6s %-4s  not on disk"%(a,lab)); continue
    r=row_from_frame(a, f.loc[common], target="target_binned_return_20")
    line="%-6s %-4s %5d %8.5f %8.5f %9.5f %7.2f %7.2f"%(
        a,lab,r["n"],r["CORR"],r["MMC"],r["payout"],r["p.sharpe"],r["dd(pay)"])
    if base is not None and a!="w130":
        pay=payout_series(f.loc[common]).dropna()
        j=pd.concat([pay,base],axis=1,join="inner").dropna(); j.columns=["a","b"]
        _,_,t=newey_west((j["a"]-j["b"]).values,20); line+=" | %+.5f (t %+.2f)"%((j["a"]-j["b"]).mean(),t)
    elif a=="w130": line+=" | baseline (= IAUARETE_01)"
    print(line)
print("-"*96)
print("All on the 20-day label. w130 reproduces e32_sel_fenccausal (IAUARETE_01).")
print("Paired Newey-West lag 20 vs the 6-month window. Expect short windows to lose badly.")
PY
trap - EXIT
[ "${failed}" -eq 0 ] && notify "${EXPERIMENT} catboost train window -- OK (${ran}/${total})" \
                       || notify "${EXPERIMENT} catboost train window -- ${failed} failed"
echo; echo "report: ${REPORT}"
exit $([ "${failed}" -eq 0 ] && echo 0 || echo 1)
