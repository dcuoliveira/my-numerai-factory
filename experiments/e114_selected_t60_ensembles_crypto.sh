#!/usr/bin/env bash
#
# E114 -- 60-day twins of the three E112 picks, then 3/4-member ensembles with the book.
#
#   lgb_ktz      crypto/v2.0-allmom-fillmask-tenc (58f)  -> e114_lgb_ktz_tenc60
#   catboost_ktz crypto/v2.0-allmom-fillmask-tenc (58f)  -> e114_catboost_ktz_tenc60
#   mlp_ktz      crypto/v2.0-allmom-fillmask-roll (228f) -> e114_mlp_ktz_roll60
#
# Same protocol as E112/E110 (rolling 130/130, embargo auto -> 60, 5-fold, val_frac 0.3,
# seed 42, n_iter 50) on target_binned_return_60. The three fits run concurrently (8+6+8
# cores; never add side runs -- see the E112 thrash). Then experiments/e114_ensembles.py
# writes logs/e114_report.txt and results/reports/crypto/e114/ensembles.csv; mailed.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
TGT=target_binned_return_60
ARMS=("lgb_ktz crypto/v2.0-allmom-fillmask-tenc e114_lgb_ktz_tenc60 4 2"
      "catboost_ktz crypto/v2.0-allmom-fillmask-tenc e114_catboost_ktz_tenc60 1 6"
      "mlp_ktz crypto/v2.0-allmom-fillmask-roll e114_mlp_ktz_roll60 8 1")

mail() {  # <subject> <file-or-text>; send_text returns False on a transient SMTP error: retry 3x
  "$PYTHON" - "$1" "$2" <<'PY' || echo "note: email failed"
import os, sys, time
from src.live.notify import send_text
body = open(sys.argv[2]).read() if os.path.isfile(sys.argv[2]) else sys.argv[2]
for k in range(3):
    if send_text(sys.argv[1], body):
        break
    time.sleep(30)
else:
    print("note: email failed 3x")
PY
}

t0=$(date +%s)
mail "[crypto] E114 started" "E114 started $(date '+%F %T') on $(hostname)
  60-day twins of the E112 picks (lgb/catboost on -tenc, mlp on -roll), then per model every
  3/4-member ensemble of {plain20, resid20, plain60, resid60, CU_SA_IA_BO_02, CU_SA_IA_BO_MA_R_01}.
  runner experiments/e114_selected_t60_ensembles_crypto.sh; report logs/e114_report.txt"

declare -A PID
for a in "${ARMS[@]}"; do
  read -r mdl data run sj nj <<<"$a"
  if [ -s "results/metrics/crypto/${mdl}/${run}.json" ]; then echo "${run} on disk -- skip"; continue; fi
  "$PYTHON" -u -m src.run_forecast crypto "$mdl" --task regression --metric corr --bmc-reference benchmark \
    --cv-scheme rolling --train-window 130 --refit-every 130 --embargo auto --n-splits 5 --n-iter 50 \
    --val-frac 0.3 --era-step 1 --seed 42 --search-jobs "$sj" --n-jobs "$nj" --data-version "$data" \
    --target "$TGT" --run-name "$run" > "logs/${run}.log" 2>&1 &
  PID[$run]=$!; echo "$(date '+%T') ${run} pid ${PID[$run]}"
done
bad=()
for r in "${!PID[@]}"; do
  if wait "${PID[$r]}"; then echo ">>> ${r} done $(date '+%T')"; else echo ">>> ${r} FAILED"; bad+=("$r"); fi
done

REPORT=logs/e114_report.txt
{
  echo "E114 fits finished in $(( ($(date +%s)-t0)/60 ))m; failed: ${bad[*]:-none}"
  echo
  "$PYTHON" experiments/e114_ensembles.py
} 2>&1 | grep -v "Warning" | tee "$REPORT"
mail "[crypto] E114 finished -- $([ ${#bad[@]} -gt 0 ] && echo "WITH FAILURES (${bad[*]})" || echo SUCCESS)" "$REPORT"
exit $([ ${#bad[@]} -gt 0 ] && echo 1 || echo 0)
