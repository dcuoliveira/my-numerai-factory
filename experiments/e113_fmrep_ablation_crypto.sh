#!/usr/bin/env bash
#
# E113 -- row ablation of E109's feature_fm_ block on the fenccausal ridge.
#
# E109's whole 41-column foundation-model block cost the fenccausal ridge payout
# -0.0170 (t -2.35) and CORR -0.028 (t -4.05). This finds which part does it: ridge
# on crypto/v2.0-fenccausal-fmrep with the 44 base features ALWAYS kept and ONE
# sub-block of the block added per arm (rows of experiments/features-foundation.md):
#
#   e113_ridge_tp1    row 1     TabPFN mean / median / mode                       3
#   e113_ridge_tp14   rows 1-4  + std/IQR/skew, 5 bin probabilities, disagreement 12
#   e113_ridge_tp24   rows 2-4  the distributional block alone (the note's
#                               proposed first study)                            9
#   e113_ridge_emb    row 5     TabPFN test-token embedding, 8 PCs                 8
#   e113_ridge_head   rows 6-7  TabPFN per-block heads + 30/390-date contexts     7
#   e113_ridge_ts     rows 8-9  TimesFM / Chronos-2 on spot + Chronos-2 PCs      14
#
# Each arm drops every feature_fm_ column EXCEPT its sub-block, by a negative
# lookahead: --feature-exclude '^feature_fm_(?!<sub-block>)'.
#
# Measured against, both REUSED from E109 on the same dataset and 510 periods:
#   e109_ridge_fenc     no block      (payout 0.08622)
#   e109_ridge_fenc_fm  whole block   (payout 0.06923)
# paired Newey-West at lag 20. A result: a sub-block that moves payout or CORR at
# |t| >= 2 against the no-block ridge, or that alone accounts for the whole loss.
#
# Protocol = E109's: rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42,
# n_iter 50, target_binned_return_20, era-step 1, ridge search-jobs 8.
#
# Knobs: DRY_RUN=1  ONLY=<regex on run name>  SHARD=k OF=n  SKIP_DONE=0  NO_MAIL=1  REPORT_ONLY=1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs

PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E113; PREFIX=e113; MODEL=ridge
DATA=crypto/v2.0-fenccausal-fmrep
BASELINE=e109_ridge_fenc; FULL=e109_ridge_fenc_fm
WIN=130; REFIT=130; EMB=auto; NSPL=5; VF=0.3; SEED=42; TGT=target_binned_return_20; ERA=1; NITER=50
SHARD="${SHARD:-1}"; OF="${OF:-1}"

# run suffix ; sub-block kept (lookahead body) ; expected feature count (44 + block)
# (';' separates fields because the regexes themselves use '|')
ARMS=(
  "tp1;tp_(mean|median|mode)\$;47"
  "tp14;tp_(mean|median|mode|std|iqr|skew|p[0-4]|disagree)\$;56"
  "tp24;tp_(std|iqr|skew|p[0-4]|disagree)\$;53"
  "emb;tp_emb_pc;52"
  "head;(head_|ctx);51"
  "ts;(tsfm|chr)_;58"
)
SELECTED=()
for a in "${ARMS[@]}"; do
  run="${PREFIX}_${MODEL}_${a%%;*}"
  if [ -n "${ONLY:-}" ] && ! [[ "$run" =~ ${ONLY} ]]; then continue; fi
  SELECTED+=("$a")
done
MINE=(); for k in "${!SELECTED[@]}"; do [ $(( k % OF + 1 )) -eq "$SHARD" ] && MINE+=("${SELECTED[$k]}"); done

report() {
  local REPORT="logs/${PREFIX}_report.txt" PD="$REPO/.claude/skills/crypto-experiment-reporting/scripts/paired_delta.py"
  {
  echo "${EXPERIMENT} -- row ablation of E109's feature_fm_ block, ridge on ${DATA}"
  echo "baseline ${BASELINE} (no block), ceiling ${FULL} (whole block)"
  echo "=============================================================="
  "$PYTHON" - "$BASELINE" "$FULL" "${ARMS[@]}" <<'PY'
import sys
import pandas as pd
from src.prediction.metrics import summarize
from src.analysis.results_table import burn_days
base, full, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [("no block", base), ("whole block", full)] + [(a.split(";")[0], "e113_ridge_" + a.split(";")[0]) for a in arms]
print("%-12s %-22s %8s %8s %8s %8s %7s %5s %4s" % ("arm", "run", "CORR", "MMC", "payout", "c.sharpe", "calmar", "burn", "n"))
for tag, run in rows:
  try: f = pd.read_csv("results/per_period/crypto/ridge/%s.csv" % run)
  except Exception: print("%-12s %-22s (missing)" % (tag, run)); continue
  pay = 0.1 * f["corr"] + f["mmc"]
  s, cs = summarize(pay), summarize(f["corr"])
  dd = s.get("max_drawdown") or 0.0
  print("%-12s %-22s %8.5f %8.5f %8.5f %8.3f %7.3f %5d %4d" % (tag, run, f["corr"].mean(), f["mmc"].mean(), s["mean"],
        cs.get("sharpe") or 0.0, s["mean"] / dd if dd else float("nan"), burn_days(f) or 0, len(f)))
PY
  echo
  for a in "${ARMS[@]}"; do
    run="${PREFIX}_${MODEL}_${a%%;*}"
    echo "=== ${run} vs ${BASELINE} (no block) ==="
    "$PYTHON" "$PD" --model "$MODEL" --baseline "$BASELINE" --arms "$run" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
    echo "=== ${run} vs ${FULL} (whole block) ==="
    "$PYTHON" "$PD" --model "$MODEL" --baseline "$FULL" --arms "$run" 2>/dev/null | grep -E "corr|mmc|payout|verdict" | sed 's/^/    /' || echo "    (n/a)"
  done
  runs=("${MODEL}/${BASELINE}" "${MODEL}/${FULL}"); for a in "${ARMS[@]}"; do runs+=("${MODEL}/${PREFIX}_${MODEL}_${a%%;*}"); done
  "$PYTHON" -m src.analysis.report_html --model "$MODEL" --runs "${runs[@]}" > "logs/${PREFIX}_html.txt" 2>&1 \
    && echo "html: $(grep -o 'results/reports/crypto/[^ ]*\.html' "logs/${PREFIX}_html.txt" | tail -1)" \
    || echo "html report failed -- logs/${PREFIX}_html.txt"
  } 2>&1 | tee "$REPORT"
  if [ "${NO_MAIL:-0}" != "1" ]; then
    "$PYTHON" - "$REPORT" <<'PY' || echo "note: email failed"
import sys; from src.live.notify import send_text
send_text("[crypto] E113 report", open(sys.argv[1]).read())
PY
  fi
}

if [ "${REPORT_ONLY:-0}" = "1" ]; then report; exit 0; fi

# Preflight: the dataset, its meta, the two reused arms, and each regex isolating
# exactly its sub-block (44 base + the block's own width).
"$PYTHON" - "$DATA" "$TGT" "$BASELINE" "$FULL" "${MINE[@]}" <<'PY' || exit 1
import os, sys
import pyarrow.parquet as pq
from src.prediction.functions import filter_features
data, tgt, base, full, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
bad = []
names = pq.ParquetFile("data/%s/train.parquet" % data).schema.names
feats = [c for c in names if c.startswith("feature_")]
if tgt not in names: bad.append("%s lacks %s" % (data, tgt))
if not os.path.exists("data/%s/historical_meta_models.parquet" % data): bad.append("%s lacks meta" % data)
for run in (base, full):
    if not os.path.exists("results/per_period/crypto/ridge/%s.csv" % run): bad.append("reused arm %s missing" % run)
for arm in arms:
    tag, keep, want = arm.split(";")
    kept = filter_features(feats, None, "^feature_fm_(?!%s)" % keep, verbose=False)
    block = [c for c in kept if c.startswith("feature_fm_")]
    if len(kept) != int(want): bad.append("%s keeps %d features, expected %s: %s" % (tag, len(kept), want, block))
    else: print("  %-5s 44 + %2d: %s" % (tag, len(block), ", ".join(c[11:] for c in block)))
if bad: sys.exit("preflight FAILED:\n  " + "\n  ".join(bad))
print("preflight OK: %d arms" % len(arms))
PY

if [ "${NO_MAIL:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ] && [ "$SHARD" = 1 ]; then
  "$PYTHON" -c "import sys;from src.live.notify import send_text;send_text(sys.argv[1],sys.argv[2])" \
"[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname): ${#SELECTED[@]} ridge arms in ${OF} shard(s), row ablation of E109's feature_fm_ block on ${DATA}." 2>/dev/null || echo "note: start email failed"
fi

echo "=== ${EXPERIMENT} shard ${SHARD}/${OF}: ${#MINE[@]} of ${#SELECTED[@]} arms ==="
ran=0; failed=0; FAILED=()
for a in "${MINE[@]}"; do
  IFS=';' read -r tag keep want <<< "$a"; run="${PREFIX}_${MODEL}_${tag}"
  cmd=("$PYTHON" -u -m src.run_forecast crypto "$MODEL" --task regression --metric corr --bmc-reference benchmark
       --cv-scheme rolling --train-window "$WIN" --refit-every "$REFIT" --embargo "$EMB" --n-splits "$NSPL"
       --n-iter "$NITER" --val-frac "$VF" --era-step "$ERA" --seed "$SEED" --search-jobs 8 --n-jobs 1
       --data-version "$DATA" --target "$TGT" --feature-exclude "^feature_fm_(?!${keep})" --run-name "$run")
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%-18s --feature-exclude %s\n' "$run" "^feature_fm_(?!${keep})"; continue; fi
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "results/metrics/crypto/${MODEL}/${run}.json" ]; then echo "  ${run} on disk -- skip"; continue; fi
  ran=$((ran+1)); echo "[${run}] $(date '+%H:%M:%S')"; s=$(date +%s)
  "${cmd[@]}" > "logs/${run}.log" 2>&1; st=$?; el=$(( $(date +%s)-s ))
  if [ "$st" -ne 0 ]; then failed=$((failed+1)); FAILED+=("$run"); echo ">>> ${run} FAILED ($st) after $((el/60))m"
  else echo ">>> ${run} done $((el/60))m$((el%60))s"; fi
done
[ "${DRY_RUN:-0}" = "1" ] && exit 0
echo "shard ${SHARD}/${OF}: ran ${ran}, failed ${failed} ${FAILED[*]:-}"

done_n=0; for a in "${ARMS[@]}"; do [ -s "results/metrics/crypto/${MODEL}/${PREFIX}_${MODEL}_${a%%;*}.json" ] && done_n=$((done_n+1)); done
[ "$done_n" -eq "${#ARMS[@]}" ] && report
exit $([ "${failed}" -gt 0 ] && echo 1 || echo 0)
