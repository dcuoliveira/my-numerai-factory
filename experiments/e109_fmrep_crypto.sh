#!/usr/bin/env bash
#
# E109 -- foundation-model representations as features, with vs without, on three bases.
#
# The 9 representations of experiments/features-foundation.md, built causally
# (src/data_mani/foundation_repr.py on the GPU box for TabPFN rows 1-7,
# foundation_repr_ts.py in the TimesFM / Chronos-2 venvs for rows 8-9) and merged
# as one 41-column feature_fm_ block by src/data_mani/build_fmrep.py:
#
#   raw         crypto/v2.0-fmrep                                       22 + 41
#   fenc        crypto/v2.0-fenccausal-fmrep                            44 + 41
#   afpp        crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmrep  812 + 41   (allfeat_pp)
#
# 30 arms = 3 bases x 5 models x {without, with}:
#   e109_<model>_<base>      without the block
#   e109_<model>_<base>_fm   with it
# "Without" is the base and nothing else on the SAME 1,701-date spine. For fenc and
# afpp that is the base dataset itself (build_fmrep asserts the variant minus the
# block is identical to it, row for row), so those 10 arms need no block and can
# run first. For raw it is crypto/v2.0-fmrep under --feature-exclude '^feature_fm_',
# because v2.0 itself now runs to 2026-09-17 and the block stops at 2026-07-08.
#
# Measured against: each arm's own "without" -- paired Newey-West at lag 20 on the
# 510 shared periods, CORR / MMC / payout, plus c.sharpe, calmar, burn and n.
# A result: the block moves payout (or CORR) at |t| >= 2 on a base. Prior from the
# pre-run EDA: TabPFN means add partial IC +0.04 (t +2.6..3.6) over the fenccausal
# ridge's own prediction; everything TS-on-label was null; wide add-ons have
# historically hurt the ridge (E82, E95).
#
# Protocol (= every shared bake-off): rolling 130/130, embargo auto=20, 5-fold,
# val_frac 0.3, seed 42, n_iter 50, target_binned_return_20, era-step 1.
#
# Knobs:  DRY_RUN=1  ONLY=<regex on run name>  SHARD=k OF=n  SKIP_DONE=0
#         SCOUT=1 (cheap, not comparable)  NO_MAIL=1  REPORT_ONLY=1
# e.g. the block-free arms now, three shards:
#   ONLY='_(fenc|afpp)$' SHARD=1 OF=3 bash experiments/e109_fmrep_crypto.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E109; PREFIX=e109
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; TGT=target_binned_return_20; ERA=1; NITER=50
SHARD="${SHARD:-1}"; OF="${OF:-1}"
if [ "${SCOUT:-0}" = "1" ]; then PREFIX="${PREFIX}scout"; ERA=6; NSPL=2; NITER=4; echo "SCOUT MODE (not comparable)"; fi
BLOCK='^feature_fm_'

MODELS=(ridge lgb_ktz catboost_ktz rf_ktz mlp_ktz)
SJOBS=(8 4 1 8 8); NJOBS=(1 2 6 2 1)
BASES=(raw fenc afpp)
declare -A BASE_DATA=([raw]=crypto/v2.0 [fenc]=crypto/v2.0-fenccausal [afpp]=crypto/v2.0-tenc-onehot-rollsel-spread-interact)
declare -A FM_DATA=([raw]=crypto/v2.0-fmrep [fenc]=crypto/v2.0-fenccausal-fmrep [afpp]=crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmrep)
declare -A N_BASE=([raw]=22 [fenc]=44 [afpp]=812)
N_BLOCK=41

# Arm table: run | model index | data version | exclude regex ("-" = none) | expected features
ARMS=()
for b in "${BASES[@]}"; do
  for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"
    if [ "$b" = raw ]; then ARMS+=("${PREFIX}_${m}_${b}|$i|${FM_DATA[$b]}|${BLOCK}|${N_BASE[$b]}")
    else ARMS+=("${PREFIX}_${m}_${b}|$i|${BASE_DATA[$b]}|-|${N_BASE[$b]}"); fi
    ARMS+=("${PREFIX}_${m}_${b}_fm|$i|${FM_DATA[$b]}|-|$(( ${N_BASE[$b]} + N_BLOCK ))")
  done
done
SELECTED=()
for k in "${!ARMS[@]}"; do
  run="${ARMS[$k]%%|*}"
  if [ -n "${ONLY:-}" ] && ! [[ "$run" =~ ${ONLY} ]]; then continue; fi
  SELECTED+=("${ARMS[$k]}")
done
MINE=()
for k in "${!SELECTED[@]}"; do [ $(( k % OF + 1 )) -eq "$SHARD" ] && MINE+=("${SELECTED[$k]}"); done

report() {
  local REPORT="logs/${PREFIX}_report.txt" PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
  {
  echo "${EXPERIMENT} -- foundation-model representation block (41 feature_fm_), with vs without, 3 bases"
  echo "=============================================================="
  for b in "${BASES[@]}"; do
    echo "######## base ${b}: without = ${PREFIX}_*_${b}, with = ${PREFIX}_*_${b}_fm (${FM_DATA[$b]})"
    for m in "${MODELS[@]}"; do
      echo "=== ${m} / ${b} ==="
      "$PYTHON" - "results/per_period/crypto/${m}/${PREFIX}_${m}_${b}.csv" "results/per_period/crypto/${m}/${PREFIX}_${m}_${b}_fm.csv" <<'PY' || echo "  (unreadable)"
import sys
import pandas as pd
from src.prediction.metrics import summarize
from src.analysis.results_table import burn_days
for tag, path in (("without", sys.argv[1]), ("with", sys.argv[2])):
  try: f = pd.read_csv(path)
  except Exception: print("  %-8s (missing)" % tag); continue
  pay = 0.1 * f["corr"] + f["mmc"]
  s, cs = summarize(pay), summarize(f["corr"])
  dd = s.get("max_drawdown") or 0.0
  print("  %-8s CORR %.5f MMC %.5f payout %.5f c.sharpe %.3f calmar %.3f burn %d n=%d"
        % (tag, f["corr"].mean(), f["mmc"].mean(), s["mean"], cs.get("sharpe") or 0.0,
           s["mean"] / dd if dd else float("nan"), burn_days(f) or 0, len(f)))
PY
      "$PYTHON" "$PD" --model "$m" --baseline "${PREFIX}_${m}_${b}" --arms "${PREFIX}_${m}_${b}_fm" 2>/dev/null \
        | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (paired n/a)"
    done
    runs=(); for m in "${MODELS[@]}"; do runs+=("${m}/${PREFIX}_${m}_${b}" "${m}/${PREFIX}_${m}_${b}_fm"); done
    "$PYTHON" -m src.analysis.report_html --model ridge --runs "${runs[@]}" > "logs/${PREFIX}_html_${b}.txt" 2>&1 \
      && echo "  html: $(grep -o 'results/reports/crypto/[^ ]*\.html' "logs/${PREFIX}_html_${b}.txt" | tail -1)" \
      || echo "  html report failed -- logs/${PREFIX}_html_${b}.txt"
    echo
  done
  } 2>&1 | tee "$REPORT"
  if [ "${NO_MAIL:-0}" != "1" ]; then
    "$PYTHON" - "$REPORT" <<'PY' || echo "note: email failed"
import sys; from src.live.notify import send_text
send_text("[crypto] E109 report", open(sys.argv[1]).read())
PY
  fi
}

if [ "${REPORT_ONLY:-0}" = "1" ]; then report; exit 0; fi

# Preflight: every dataset this shard needs, with its feature count, label and meta.
"$PYTHON" - "$TGT" "${MINE[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
tgt, arms = sys.argv[1], sys.argv[2:]
bad = []
for arm in arms:
    run, _, data, excl, want = arm.split("|")
    path = "data/%s/train.parquet" % data
    if not os.path.exists(path): bad.append("%s: %s MISSING" % (run, data)); continue
    names = pq.ParquetFile(path).schema.names
    feats = [c for c in names if c.startswith("feature_")]
    if excl != "-": feats = [c for c in feats if not c.startswith("feature_fm_")]
    if len(feats) != int(want): bad.append("%s: %d features, expected %s" % (run, len(feats), want))
    if tgt not in names: bad.append("%s: %s lacks %s" % (run, data, tgt))
    if not os.path.exists("data/%s/historical_meta_models.parquet" % data): bad.append("%s: %s lacks meta" % (run, data))
if bad: sys.exit("preflight FAILED:\n  " + "\n  ".join(sorted(set(bad))))
print("preflight OK: %d arms, datasets present with the expected widths" % len(arms))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ "$SHARD" = 1 ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname): ${#SELECTED[@]} arms (ONLY='${ONLY:-}') in ${OF} shard(s).
  foundation-model block (41 feature_fm_) on raw / fenccausal / allfeat_pp, with vs without; 5 models, n_iter=${NITER}, rolling ${WIN}/${REFIT}." 2>/dev/null || echo "note: start email failed"
fi

echo "=== ${EXPERIMENT} shard ${SHARD}/${OF}: ${#MINE[@]} of ${#SELECTED[@]} selected arms | n_iter=${NITER} ==="
ran=0; failed=0; FAILED=()
for arm in "${MINE[@]}"; do
  IFS='|' read -r run i data excl want <<< "$arm"; m="${MODELS[$i]}"
  extra=(); [ "$excl" != "-" ] && extra=(--feature-exclude "$excl")
  cmd=("$PYTHON" -u -m src.run_forecast crypto "$m" --task regression --metric corr --bmc-reference benchmark
       --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL"
       --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED"
       --search-jobs "${SJOBS[$i]}" --n-jobs "${NJOBS[$i]}" --data-version "$data" --target "$TGT"
       "${extra[@]}" --run-name "$run")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%-28s %-13s %-55s %s\n' "$run" "$m" "$data" "${extra[*]:-}"; continue; fi
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${m}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
  ran=$((ran+1)); echo; echo "[${run}] ${m} on ${data} ${extra[*]:-} $(date '+%H:%M:%S')"; s=$(date +%s)
  "${cmd[@]}" > "logs/${run}.log" 2>&1; st=$?; el=$(( $(date +%s)-s ))
  if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$run"); echo ">>> ${run} FAILED ($st) after $((el/60))m -- continue"
  else echo ">>> ${run} done $((el/60))m$((el%60))s"; fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo "shard ${SHARD}/${OF}: ran ${ran}, failed ${failed} ${FAILED[*]:-}"

# The shard that finds all 30 on disk writes the report.
done_n=0
for arm in "${ARMS[@]}"; do IFS='|' read -r run i _ _ _ <<< "$arm"; [ -s "results/metrics/crypto/${MODELS[$i]}/${run}.json" ] && done_n=$((done_n+1)); done
if [ "$done_n" -eq "${#ARMS[@]}" ] && [ "${SCOUT:-0}" != "1" ]; then report; fi
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
