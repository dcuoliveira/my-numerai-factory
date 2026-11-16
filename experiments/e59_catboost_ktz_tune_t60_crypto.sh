#!/usr/bin/env bash
#
# Experiment E59: from-scratch catboost_ktz tune on the 60-DAY label, for the
# five feature sets where catboost scored best on MMC at 20 days (E32).
#
# E59 is the reference handle; every artefact is e59_*.
#
# THE IDEA. E32 searched catboost_ktz per dataset on the 20-day label and read
# off its MMC. The five best-MMC feature sets were:
#
#     fenccausal    MMC 0.07738   (44f)
#     allfeatpp     MMC 0.07536   (812f)
#     tenc          MMC 0.07461   (23f)
#     allfeat       MMC 0.07450   (788f)
#     tenc_onehot   MMC 0.07345   (763f)
#
# This re-tunes each FROM SCRATCH on target_binned_return_60 -- the 60-day
# version of the same catboost picks. Only the label changes (and the embargo it
# forces): same 12-cell KTZ grid (depth{2,4} x lr{.005,.01,.02} x rsm{.1,.2};
# iterations 2000, l2_leaf_reg 10, min_data_in_leaf 100 pinned), same exhaustive
# --n-iter 12 search, same seed. It asks whether the feature sets that gave
# catboost the most contribution at 20 days still do at 60.
#
# WHAT THE 60-DAY LABEL CHANGES, and why it is not a free knob:
#   * THE EMBARGO TRIPLES. --embargo auto derives it from the target horizon, so
#     a 60-day forward return needs 60 periods of gap (98% overlap between
#     consecutive daily periods) where the 20-day one needs 20. This is FORCED by
#     the label, not chosen -- a smaller embargo would leak the label across the
#     train/predict boundary. It also squeezes the inner CV: embargo 60 inside a
#     130-period training window leaves ~2 usable purged folds, so the SEARCH is
#     selecting on a thinner signal than at 20 days. The OUTER walk-forward stays
#     honest OOS regardless; T60_WINDOW widens the window if the search needs more.
#   * CORR IS NOT COMPARABLE ACROSS TARGETS. numerai_corr on the 60-day label is
#     structurally higher, so these numbers must NOT be put in one table with the
#     20-day E32 arms. Read them against the 60-day book instead: SACI_03 and
#     CURUPIRA_05_60 are the only 60-day-scored slots. report_html splits by
#     target and ranks them there automatically.
#
# WHAT WOULD COUNT AS A RESULT
#   Per feature set, one from-scratch tune scored on the 60-day label. Read each
#   against the two deployed 60-day slots (paired Newey-West, lag 20) and against
#   each other. A feature set whose 20-day MMC edge SURVIVES at 60 days is the
#   clean finding; one that does not says the contribution was horizon-specific.
#   Note SACI_03's live slot FITS the 20-day label -- its 60-day figure is a
#   re-score/backtest, so it is a shape reference, not a fitted-on-60 competitor;
#   this study is the first catboost actually fit on the 60-day target.
#
# RUNTIME. catboost member fit ~5-7s; --n-iter 12 x ~2 usable folds + refit,
# ~3-4 refits per dataset. Budget ~10 min/dataset, ~1 h for all five. Measured
# before launch, not guessed.
#
# Usage:
#   bash experiments/e59_catboost_ktz_tune_t60_crypto.sh
#   DRY_RUN=1 bash experiments/e59_catboost_ktz_tune_t60_crypto.sh
#   ONLY="fenccausal tenc" bash experiments/e59_catboost_ktz_tune_t60_crypto.sh
#   T60_WINDOW=260 bash experiments/e59_catboost_ktz_tune_t60_crypto.sh   # widen inner CV
#   SHARD=1 OF=2 / NO_MAIL=1 / E59_SCOUT=1 bash experiments/e59_catboost_ktz_tune_t60_crypto.sh
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

EXPERIMENT="${EXPERIMENT:-E59}"
E59_TARGET="${E59_TARGET:-target_binned_return_60}"
E59_WINDOW="${T60_WINDOW:-${E59_WINDOW:-130}}"
E59_REFIT="${E59_REFIT:-130}"
E59_N_SPLITS="${E59_N_SPLITS:-5}"
E59_N_ITER="${E59_N_ITER:-12}"
E59_SEED="${E59_SEED:-42}"
E59_N_JOBS="${E59_N_JOBS:-6}"
E59_ITERS="${E59_ITERS:-2000}"
E59_ERA_STEP=1
SHARD="${SHARD:-1}"; OF="${OF:-1}"
# GNU screen exports $WINDOW=0 into every session -- refuse the silent empty block.
case "${E59_WINDOW}" in ""|0|[0-9]) echo "E59_WINDOW looks wrong (screen \$WINDOW=0?)" >&2; exit 1;; esac
case "${E59_REFIT}"  in ""|0|[0-9]) echo "E59_REFIT looks wrong (screen \$WINDOW=0?)"  >&2; exit 1;; esac

PREFIX="e59"
if [ "${E59_SCOUT:-0}" = "1" ]; then
    PREFIX="e59scout"; E59_ITERS=300
    echo "SCOUT MODE: iterations ${E59_ITERS} -- wiring only, NOT comparable."
fi

export OMP_NUM_THREADS="${E59_N_JOBS}"
export MKL_NUM_THREADS="${E59_N_JOBS}"
export OPENBLAS_NUM_THREADS="${E59_N_JOBS}"

# Parallel arrays, never pipe-delimited records (a feature regex uses '|').
# The five top-MMC feature sets from E32, with their 20-day MMC for the header.
NAME=(); DATA=(); NFEAT=(); MMC20=()
add() { NAME+=("$1"); DATA+=("$2"); NFEAT+=("$3"); MMC20+=("$4"); }
add fenccausal  "crypto/v2.0-fenccausal"                          44 0.07738
add allfeatpp   "crypto/v2.0-tenc-onehot-rollsel-spread-interact" 812 0.07536
add tenc        "crypto/v2.0-tenc"                                23 0.07461
add allfeat     "crypto/v2.0-tenc-onehot-spread-interact"        788 0.07450
add tenc_onehot "crypto/v2.0-tenc-onehot"                        763 0.07345

SEL_NAME=(); SEL_DATA=(); SEL_NFEAT=(); SEL_MMC=(); picked=0
for i in "${!NAME[@]}"; do
    arm="${NAME[$i]}"
    [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${arm} "* ]] && continue
    [ -n "${SKIP:-}" ] && [[ " ${SKIP} " == *" ${arm} "* ]] && continue
    if [ "${OF}" -gt 1 ]; then
        picked=$((picked+1)); [ $(( (picked-1) % OF + 1 )) -ne "${SHARD}" ] && continue
    fi
    SEL_NAME+=("${arm}"); SEL_DATA+=("${DATA[$i]}"); SEL_NFEAT+=("${NFEAT[$i]}"); SEL_MMC+=("${MMC20[$i]}")
done
[ "${#SEL_NAME[@]}" -eq 0 ] && { echo "no datasets selected" >&2; exit 1; }

# Preflight: datasets + meta files present, the 60-day label present, and
# catboost_ktz constructs with the expected 12-cell grid.
python3 - "${SEL_DATA[@]}" -- "${SEL_NFEAT[@]}" -- "${E59_TARGET}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from sklearn.model_selection import ParameterGrid
from src.prediction.models import build_wrapper

a=sys.argv[1:]; c1=a.index("--"); c2=a.index("--", c1+1)
versions=a[:c1]; wants=[int(x) for x in a[c1+1:c2]]; target=a[c2+1]
bad=[]
for v,want in zip(versions,wants):
    p="data/%s/train.parquet"%v
    try: names=pq.ParquetFile(p).schema.names
    except OSError: bad.append("%s MISSING"%v); continue
    got=sum(c.startswith("feature_") for c in names)
    if got!=want: bad.append("%s has %d features, expected %d"%(v,got,want))
    if target not in names: bad.append("%s lacks %s"%(v,target))
    if not os.path.exists("data/%s/historical_meta_models.parquet"%v):
        bad.append("%s lacks historical_meta_models.parquet"%v)
w=build_wrapper("catboost_ktz","regression",seed=42,n_jobs=1)
ncells=len(list(ParameterGrid(w.param_grid)))
if ncells!=12: bad.append("catboost_ktz grid has %d cells, expected 12"%ncells)
if any(d>4 for d in w.param_grid["depth"]): bad.append("a depth>4 escaped the shallow KTZ gate")
if bad: sys.exit("preflight FAILED:\n  "+"\n  ".join(bad))
print("preflight OK: %d dataset(s) present with meta files and the 60-day label; "
      "catboost_ktz grid is the 12 KTZ cells"%len(versions))
PY

total="${#SEL_NAME[@]}"; ran=0; failed=0; declare -a FAILED_ARMS=(); started_all=$(date +%s)

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    python3 -c "import sys; from src.live.notify import send_text; \
send_text('[crypto] %s started' % sys.argv[1], sys.argv[2]); print('   emailed the start notice')" \
"${EXPERIMENT}" "${EXPERIMENT} started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)

  idea      from-scratch catboost_ktz tune on the 60-day label, top-5 MMC sets (E32)
  sets      fenccausal allfeatpp tenc allfeat tenc_onehot  (of 9; shard ${SHARD}/${OF})
  grid      the 12 KTZ cells, exhaustive --n-iter ${E59_N_ITER}
  target    ${E59_TARGET}  --  embargo auto (60 periods, forced by the label)
  cadence   rolling ${E59_WINDOW}/${E59_REFIT}, ${E59_N_SPLITS} folds, seed ${E59_SEED}
  runtime   ~10 min/set; ~1 h for five

  runner    experiments/e59_catboost_ktz_tune_t60_crypto.sh
  results   results/metrics/crypto/catboost_ktz/${PREFIX}_*
  report    logs/${PREFIX}_report.txt
  handle    ${EXPERIMENT}; every artefact is ${PREFIX}_*

Read against the 60-day book (SACI_03, CURUPIRA_05_60); CORR is NOT comparable
to the 20-day E32 arms. A second mail follows on completion." || echo "note: start email failed, run continues"
fi

echo "=============================================================="
echo "${EXPERIMENT}: catboost_ktz from-scratch tune on ${E59_TARGET}  |  ${total} sets  shard ${SHARD}/${OF}"
echo "grid: 12 KTZ cells, --n-iter ${E59_N_ITER} | rolling ${E59_WINDOW}/${E59_REFIT}, embargo auto | iters ${E59_ITERS}"
echo "started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

for i in "${!SEL_NAME[@]}"; do
    arm="${SEL_NAME[$i]}"; run_name="${PREFIX}_${arm}"; ver="${SEL_DATA[$i]}"
    if [ -f "results/metrics/crypto/catboost_ktz/${run_name}.json" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        echo "[$((i+1))/${total}] ${run_name} already on disk -- skipping (resume-safe)"; continue
    fi
    cmd=(python3 -u -m src.run_forecast crypto catboost_ktz
         --task regression --metric corr --bmc-reference benchmark
         --cv-scheme rolling --train-window "${E59_WINDOW}" --refit-every "${E59_REFIT}"
         --embargo auto --n-splits "${E59_N_SPLITS}" --n-iter "${E59_N_ITER}"
         --val-frac 0.3 --era-step "${E59_ERA_STEP}" --seed "${E59_SEED}"
         --search-jobs 1 --n-jobs "${E59_N_JOBS}" --data-version "${ver}"
         --target "${E59_TARGET}" --model-options "{\"iterations\":${E59_ITERS}}"
         --run-name "${run_name}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '[%d/%d] %-16s %4sf  %-46s  20d MMC %s\n' "$((i+1))" "$total" "$run_name" \
               "${SEL_NFEAT[$i]}" "${ver}" "${SEL_MMC[$i]}"; continue
    fi
    echo; echo "[$((i+1))/${total}] ${run_name}  ${SEL_NFEAT[$i]}f  (20d MMC ${SEL_MMC[$i]})  $(date '+%H:%M:%S')"
    started=$(date +%s)
    "${cmd[@]}" > "logs/${run_name}.log" 2>&1
    status="$?"; el=$(( $(date +%s)-started ))
    if [ "$status" -ne 0 ]; then
        failed=$((failed+1)); FAILED_ARMS+=("${run_name}")
        echo "    >>> ${run_name} FAILED (exit ${status}) after $((el/60))m -- continuing (logs/${run_name}.log)"
    else
        ran=$((ran+1)); echo "    >>> ${run_name} done in $((el/60))m$((el%60))s"
    fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0

REPORT="logs/${PREFIX}_report.txt"
[ "${OF}" -gt 1 ] && REPORT="logs/${PREFIX}_report_shard${SHARD}.txt"
{
echo "${EXPERIMENT} -- catboost_ktz from-scratch tune on ${E59_TARGET} (top-5 MMC sets from E32)"
echo "  results  results/metrics/crypto/catboost_ktz/${PREFIX}_*"
echo "  runner   experiments/e59_catboost_ktz_tune_t60_crypto.sh"
el_all=$(( $(date +%s)-started_all ))
echo "ran ${ran} of ${total} tunes in $((el_all/3600))h$(( (el_all%3600)/60 ))m, ${failed} failed"
[ "${failed}" -gt 0 ] && printf 'failed: %s\n' "${FAILED_ARMS[*]}"
echo "=============================================================="
python3 - "${PREFIX}" <<'PY'
import glob, json, os, sys
prefix=sys.argv[1]
SETS=[("fenccausal",0.07738),("allfeatpp",0.07536),("tenc",0.07461),
      ("allfeat",0.07450),("tenc_onehot",0.07345)]
def load(run):
    p="results/metrics/crypto/catboost_ktz/%s.json"%run
    return json.load(open(p)) if os.path.exists(p) else None
print("%-14s %5s %6s %9s %8s %8s %8s   %s"%(
    "set","n","20dMMC","60dCORR","60dMMC","60dpay","p.shrp","lr/depth/rsm chosen per refit"))
print("-"*104)
for ds,mmc20 in SETS:
    d=load("%s_%s"%(prefix,ds))
    if d is None:
        print("%-14s  NOT ON DISK -- not run (or another shard)"%("%s_%s"%(prefix,ds))); continue
    ch=[(r["best_params"].get("learning_rate"),r["best_params"].get("depth"),r["best_params"].get("rsm"))
        for r in d.get("refits",[])]
    chs=" ".join("%.3f/%d/%.2f"%(a,b,c) for a,b,c in ch)
    print("%-14s %5d %6.5f %9.5f %8.5f %8.5f %8.3f   %s"%(
        ds, d["n_periods"], mmc20, d["corr"]["mean"], d["mmc"]["mean"],
        0.1*d["corr"]["mean"]+d["mmc"]["mean"], d["corr"]["sharpe"], chs))
print("-"*104)
print("60d CORR is NOT comparable to the 20d E32 arms. Rank each against the 60-day book:")
print("  python3 -m src.analysis.report_html --model catboost_ktz --runs %s_fenccausal [<run> ...]"%prefix)
print("  (report_html puts them in the t60 block with SACI_03 and CURUPIRA_05_60.)")
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
