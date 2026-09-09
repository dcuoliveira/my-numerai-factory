#!/usr/bin/env bash
#
# Experiment E32: ensemble the CatBoost KTZ grid instead of searching it
#
# E32 is the reference handle; every artefact is e32_*.
#
# THE IDEA. The CatBoost member of the E30 (lgb) / E31 (mlp) family: "ensemble
# the ktz grid instead of searching it". For each dataset the book trains on, two
# arms over the SAME 12-cell CatBoost KTZ grid --
#
#   e32_ens_<ds>   catboost_ktz_ens : fit ALL 12 cells, rank-average them. One
#                  deployable estimator (CatBoostGridEnsemble); no cell is
#                  discarded and no selection variance is paid on ~26 effective
#                  windows. --n-splits 2 so the unused inner CV costs one fold.
#   e32_sel_<ds>   catboost_ktz     : the same 12 cells SEARCHED (RandomizedSearch,
#                  --n-iter 12 sweeps them exhaustively), keeping the CV-best. The
#                  honest "hyperparameter searching picks one" baseline.
#
# THE GRID (12 cells), the E27/KTZ booster priors on CatBoost's names, compact ON
# PURPOSE so the ensemble can fit ALL of it every refit:
#
#     depth          {2, 4}              shallow symmetric trees (4, 16 leaves)
#     learning_rate  {0.005, 0.01, 0.02} the axis the ensemble AVERAGES over
#     rsm            {0.1, 0.2}          low column sampling, the Numerai idiom
#     FIXED: iterations=2000 (L = lr*iters in [10,40]), l2_leaf_reg=10,
#            min_data_in_leaf=100 (the E27 leaf floor)
#
#   The grid lives in prediction/catboost_ensemble.CatBoostGridEnsemble and is
#   READ by both wrappers, so the searched space and the ensembled space are
#   provably identical -- ensemble-of-12 vs CV-pick-1-of-12, nothing else varying.
#
# DATASETS: the 9 the submitted book trains on, same set as E28/E29/E30:
#   raw(22) tenc(23) interact(36) fenccausal(44) cumlag3(88)
#   tenc_onehot(763) onehot_rollsel(786) allfeat(788) allfeatpp(812)
#
#   9 datasets x 2 arms = 18 runs.
#
# WHAT WOULD COUNT AS A RESULT
#   Per dataset the headline is the paired ENS - SEL delta (Newey-West lag 20):
#   does averaging the grid beat picking one out of it? The prior from the whole
#   programme (E10/E23/E25/E27) is that averaging buys RISK -- steadier sharpe,
#   shallower drawdown -- more than LEVEL, so read dd(pay) and p.sharpe, not just
#   payout. Each e32_ens_<ds> is ALSO an arm to rank against the deployed slot on
#   that dataset; build that board with the arm-board-reporting tools
#   (report_html / results_table), which pair against the book automatically.
#
# RUNTIME. A CatBoost member fit is ~5-7s even at 812 features (rsm~=0.1 starves
# each tree of columns). ENS ~= 24 member-fits/refit (2 est-fits x 12), SEL ~= 49
# fits/refit (12 cells x 4 usable folds + 1 refit), ~4 refits/dataset. Budget
# ~10 min/ENS and ~20 min/SEL, so ~4-5 h for all 18 sequentially, less across
# shards. Measured before launch, not guessed.
#
# Usage:
#   bash experiments/e32_catboost_ktz_ensemble_crypto.sh
#   DRY_RUN=1 bash experiments/e32_catboost_ktz_ensemble_crypto.sh
#   ONLY="raw interact" bash experiments/e32_catboost_ktz_ensemble_crypto.sh
#   ARMS="ens" bash experiments/e32_catboost_ktz_ensemble_crypto.sh   # ens|sel|both
#   SHARD=1 OF=3 bash experiments/e32_catboost_ktz_ensemble_crypto.sh
#   E32_SCOUT=1 bash experiments/e32_catboost_ktz_ensemble_crypto.sh  # era-step 4, tiny
#   NO_MAIL=1 bash experiments/e32_catboost_ktz_ensemble_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E32}"
E32_WINDOW="${E32_WINDOW:-130}"
E32_REFIT="${E32_REFIT:-130}"
E32_EMBARGO="${E32_EMBARGO:-20}"
E32_SEED="${E32_SEED:-42}"
E32_N_JOBS="${E32_N_JOBS:-6}"
E32_ERA_STEP=1
E32_ITERS="${E32_ITERS:-2000}"
ARMS="${ARMS:-both}"      # ens | sel | both
SHARD="${SHARD:-1}"; OF="${OF:-1}"
# GNU screen exports $WINDOW=0 into every session -- refuse the silent empty block.
case "${E32_WINDOW}" in ""|0|[0-9]) echo "E32_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E32_REFIT}"  in ""|0|[0-9]) echo "E32_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e32"
if [ "${E32_SCOUT:-0}" = "1" ]; then
    PREFIX="e32scout"; E32_ERA_STEP=4; E32_ITERS=100
    echo "SCOUT MODE: era-step 4, iterations ${E32_ITERS} -- wiring only, NOT comparable."
fi

export OMP_NUM_THREADS="${E32_N_JOBS}"
export MKL_NUM_THREADS="${E32_N_JOBS}"
export OPENBLAS_NUM_THREADS="${E32_N_JOBS}"

# Parallel arrays, never pipe-delimited records (a feature regex uses '|').
NAME=(); DATA=(); NFEAT=(); SLOT=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); SLOT+=("$4"); }
add raw            "crypto/v2.0"                                     22 CURUPIRA_02
add tenc           "crypto/v2.0-tenc"                                23 CURUPIRA_01
add interact       "crypto/v2.0-interact"                            36 SACI_01
add fenccausal     "crypto/v2.0-fenccausal"                          44 IA_01
add cumlag3        "crypto/v2.0-cumlag3"                             88 SACI_05
add tenc_onehot    "crypto/v2.0-tenc-onehot"                        763 SACI_03
add onehot_rollsel "crypto/v2.0-onehot-rollsel"                     786 CURUPIRA_03
add allfeat        "crypto/v2.0-tenc-onehot-spread-interact"        788 SACI_02
add allfeatpp      "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 CURUPIRA_04

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=(); SEL_SLOT=(); picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}"); SEL_SLOT+=("${SLOT[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

# Preflight: datasets + meta files present with the right feature counts, both
# wrappers construct, and the search grid is exactly the ensemble's cells.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper
from src.prediction.catboost_ensemble import CatBoostGridEnsemble

cut=sys.argv.index("--"); versions=sys.argv[1:cut]; wants=[int(x) for x in sys.argv[cut+1:]]
bad=[]
for v,want in zip(versions,wants):
    p="data/%s/train.parquet"%v
    try: got=sum(c.startswith("feature_") for c in pq.ParquetFile(p).schema.names)
    except OSError: bad.append("%s MISSING"%v); continue
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
    if not os.path.exists("data/%s/historical_meta_models.parquet"%v):
        bad.append("%s lacks historical_meta_models.parquet"%v)

sw=build_wrapper("catboost_ktz","regression",seed=42,n_jobs=1)
ew=build_wrapper("catboost_ktz_ens","regression",seed=42,n_jobs=1)
cells={(c["learning_rate"],c["depth"],c["rsm"]) for c in ParameterGrid(sw.param_grid)}
members=set(ew.ModelClass._grid())
if cells!=members:
    bad.append("catboost_ktz search grid != catboost_ktz_ens members (%d vs %d)"%(len(cells),len(members)))
if len(cells)!=12:
    bad.append("expected 12 grid cells, got %d"%len(cells))
if any(d>4 for _,d,_ in cells):
    bad.append("a depth > 4 escaped the shallow KTZ gate")
if ew.param_grid!={"combine":["rank"]}:
    bad.append("catboost_ktz_ens is not a single rank-combine candidate: %r"%ew.param_grid)
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files; catboost_ktz search "
      "grid == catboost_ktz_ens's 12 members; ensemble rank-combines"%len(versions))
PY

# One walk-forward run. $1 name(ens|sel) $2 dataset-arm $3 version
run_arm() {
    local kind="$1" arm="$2" ver="$3" run_name model n_iter n_splits opts
    run_name="${PREFIX}_${kind}_${arm}"
    if [ -f "results/metrics/crypto/$([ "$kind" = ens ] && echo catboost_ktz_ens || echo catboost_ktz)/${run_name}.json" ]; then
        echo "    ${run_name} already on disk -- skipping (resume-safe)"; return 0
    fi
    if [ "$kind" = "ens" ]; then
        model="catboost_ktz_ens"; n_iter=1; n_splits=2
        opts="{\"iterations\":${E32_ITERS}}"
    else
        model="catboost_ktz"; n_iter=12; n_splits=5
        opts="{\"iterations\":${E32_ITERS}}"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '    %-18s model=%-16s n_iter=%-2s n_splits=%s\n' "$run_name" "$model" "$n_iter" "$n_splits"; return 0
    fi
    python3 -u -m src.run_forecast crypto "${model}" \
        --task regression --metric corr --bmc-reference benchmark \
        --cv-scheme rolling --train-window "${E32_WINDOW}" --refit-every "${E32_REFIT}" \
        --embargo "${E32_EMBARGO}" --n-splits "${n_splits}" --n-iter "${n_iter}" \
        --val-frac 0.3 --era-step "${E32_ERA_STEP}" --seed "${E32_SEED}" \
        --search-jobs 1 --n-jobs "${E32_N_JOBS}" --data-version "${ver}" \
        --model-options "${opts}" --run-name "${run_name}" \
        > "logs/${run_name}.log" 2>&1
}

total="${#SEL_NAME[@]}"; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)
want_ens=1; want_sel=1
[ "${ARMS}" = "ens" ] && want_sel=0
[ "${ARMS}" = "sel" ] && want_ens=0

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  idea      ensemble the CatBoost ktz grid vs searching it (the E30/E31 family)
  grid      12 cells: depth{2,4} x lr{.005,.01,.02} x rsm{.1,.2}; iters ${E32_ITERS} pinned
  arms      per dataset: catboost_ktz_ens (rank-average all 12) vs catboost_ktz (pick 1)
  datasets  ${total} of the 9 the book trains on (shard ${SHARD}/${OF})
  cadence   rolling ${E32_WINDOW}/${E32_REFIT}, embargo ${E32_EMBARGO}, seed ${E32_SEED}
  runtime   ~4-5 h for 18 runs sequential; less across shards

  runner    experiments/e32_catboost_ktz_ensemble_crypto.sh
  results   results/metrics/crypto/catboost_ktz_ens/${PREFIX}_ens_* and .../catboost_ktz/${PREFIX}_sel_*
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: catboost_ktz grid ensemble vs search  |  ${total} datasets  arms=${ARMS}  shard ${SHARD}/${OF}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')  |  iters ${E32_ITERS}  threads ${E32_N_JOBS}"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; ver="${SEL_DATA[$i]}"
    echo; echo "[$((i+1))/${total}] ${arm}  ${SEL_NFEAT[$i]} feat  (slot ${SEL_SLOT[$i]})  $(date '+%H:%M:%S')"
    for kind in ens sel; do
        [ "$kind" = ens ] && [ "$want_ens" = 0 ] && continue
        [ "$kind" = sel ] && [ "$want_sel" = 0 ] && continue
        started=$(date +%s)
        if run_arm "$kind" "$arm" "$ver"; then
            [ "${DRY_RUN:-0}" = "1" ] || { el=$(( $(date +%s)-started )); ran=$((ran+1)); \
                echo "    >>> ${PREFIX}_${kind}_${arm} done in $((el/60))m$((el%60))s"; }
        else
            failed=$((failed+1)); FAILED_ARMS+=("${PREFIX}_${kind}_${arm}")
            echo "    >>> ${PREFIX}_${kind}_${arm} FAILED -- continuing (logs/${PREFIX}_${kind}_${arm}.log)"
        fi
    done
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- catboost_ktz grid ensemble (ens) vs search-and-pick (sel)"
echo "  results  results/metrics/crypto/catboost_ktz_ens/${PREFIX}_ens_* | .../catboost_ktz/${PREFIX}_sel_*"
echo "  runner   experiments/e32_catboost_ktz_ensemble_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} runs in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
[ "${OF}" -gt 1 ] && echo "SHARD ${SHARD}/${OF} -- PARTIAL; other shards hold the rest"
echo "=============================================================="
python3 - "${PREFIX}" <<'PY'
import os, sys
import numpy as np, pandas as pd
prefix = sys.argv[1]
DS = [("raw","CURUPIRA_02"),("tenc","CURUPIRA_01"),("interact","SACI_01"),
      ("fenccausal","IA_01"),("cumlag3","SACI_05"),("tenc_onehot","SACI_03"),
      ("onehot_rollsel","CURUPIRA_03"),("allfeat","SACI_02"),("allfeatpp","CURUPIRA_04")]

def ser(model, run):
    p = "results/per_period/crypto/%s/%s.csv" % (model, run)
    if not os.path.exists(p): return None
    f = pd.read_csv(p)
    return f.set_index("date") if "date" in f.columns else f

def payout(f): return 0.1 * f["corr"] + f["mmc"]

def nw(x, lag=20):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(v, 0) / n); return m, (m / se if se > 0 else float("nan"))

def stats(p):
    c = p.cumsum(); dd = (np.maximum.accumulate(c) - c).max()
    return p.mean(), (p.mean() / p.std(ddof=0) if p.std(ddof=0) > 0 else float("nan")), dd

print("%-15s %6s %10s %8s %8s | %10s | %s" % (
    "dataset", "n", "ENS pay", "p.shrp", "dd(pay)", "SEL pay", "ens - sel  (NW t, lag20)"))
print("-" * 100)
for ds, slot in DS:
    ens = ser("catboost_ktz_ens", "%s_ens_%s" % (prefix, ds))
    sel = ser("catboost_ktz",     "%s_sel_%s" % (prefix, ds))
    if ens is None and sel is None:
        continue
    row = "%-15s " % ds
    if ens is not None:
        ep = payout(ens); em, esh, edd = stats(ep)
        row += "%6d %10.6f %8.4f %8.4f | " % (len(ep), em, esh, edd)
    else:
        row += "%6s %10s %8s %8s | " % ("-", "not-run", "-", "-")
    if sel is not None:
        sp = payout(sel); row += "%10.6f | " % sp.mean()
    else:
        row += "%10s | " % "not-run"
    if ens is not None and sel is not None:
        j = pd.concat([payout(ens), payout(sel)], axis=1, join="inner", keys=["e", "s"])
        d, t = nw(j["e"] - j["s"]); row += "%+.6f (t %+.2f)" % (d, t)
    else:
        row += "-"
    print(row)
print("-" * 100)
print("ENS = catboost_ktz_ens (rank-average of all 12 grid cells).")
print("SEL = catboost_ktz (RandomizedSearch over the same 12, CV-best kept).")
print("+t => averaging the grid beat picking one out of it. Newey-West lag 20, paired on common dates.")
print()
print("Each ENS arm is also a candidate against its deployed slot; rank it on the book with:")
print("  python3 -m src.analysis.report_html --model catboost_ktz_ens --runs %s_ens_raw [<run> ...]" % prefix)
PY
} 2>&1 | tee "${REPORT}"

if [ "${NO_MAIL:-0}" != "1" ]; then
  python3 - "$([ "${failed}" -gt 0 ] && echo "WITH ${failed} FAILED" || echo SUCCESS)" "${REPORT}" "${EXPERIMENT}" <<'PY' || echo "note: email failed"
import sys
from src.live.notify import send_text
send_text("[crypto] %s finished -- %s" % (sys.argv[3], sys.argv[1]), open(sys.argv[2]).read())
print("emailed the report")
PY
fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
