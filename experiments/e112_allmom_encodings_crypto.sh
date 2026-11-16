#!/usr/bin/env bash
#
# E112 -- which per-symbol encoding adds to the all-momentum set?
#         crypto/v2.0-allmom-fillmask (57f, base, E108 arms REUSED) vs four supersets
#         that differ from it ONLY by the encoding:
#
#   fenc  crypto/v2.0-allmom-fillmask-fenccausal  114f  +57  feature_<f>_symmean
#         causal expanding mean of each feature over the symbol's strictly earlier
#         dates, shrunk toward the overall mean (m=50)
#   tenc  crypto/v2.0-allmom-fillmask-tenc         58f  +1   feature_sym_tenc
#         mean of the symbol's own 20d target over resolved dates <= t-20, shrunk (m=50)
#   roll  crypto/v2.0-allmom-fillmask-roll        228f  +171 feature_<f>_rm{5,20,60}
#         per-symbol trailing means over the last 5/20/60 dates, today included
#   all   crypto/v2.0-allmom-fillmask-all         286f  all three
#
# 4 datasets x 5 models from scratch at the shared protocol (rolling 130/130, embargo
# auto=20, 5-fold, val_frac 0.3, seed 42, n_iter 50, target_binned_return_20) = 20
# arms. The four lanes run CONCURRENTLY, each through mommask_core.sh (its per-lane
# 3-way report -- raw v2.0 / allmom / variant -- goes to logs/e112_report_<v>.txt);
# then experiments/e112_report.py writes the combined report (every variant paired vs
# its model's allmom twin, NW lag 20, with calmar) to logs/e112_report.txt and mails it.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p logs
export PYTHON="${PYTHON:-/home/danielco/miniconda3/bin/python3}"
EXPERIMENT=E112; PREFIX=e112
LANES=("fenc crypto/v2.0-allmom-fillmask-fenccausal 114"
       "tenc crypto/v2.0-allmom-fillmask-tenc 58"
       "roll crypto/v2.0-allmom-fillmask-roll 228"
       "all crypto/v2.0-allmom-fillmask-all 286")

mail() {  # <subject> <file-or-text>; send_text returns False (never raises) on a transient SMTP error: retry 3x
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
[ "${NO_MAIL:-0}" = "1" ] || mail "[crypto] ${EXPERIMENT} started" "${EXPERIMENT} started $(date '+%F %T') on $(hostname)
  symbol encodings on top of crypto/v2.0-allmom-fillmask (57f, base e108_*_allmom reused)
  lanes (concurrent): fenc 114f | tenc 58f | roll 228f | all 286f, 5 models each = 20 arms
  n_iter=50, rolling 130/130, embargo auto, 5 folds, seed 42, target_binned_return_20
  runner experiments/e112_allmom_encodings_crypto.sh; report logs/e112_report.txt"

declare -A PID
for lane in "${LANES[@]}"; do
  read -r v data nf <<<"$lane"
  EXPERIMENT=$EXPERIMENT PREFIX=$PREFIX VARIANT=$v MM_DATA=$data MM_FEATS=$nf \
  PARENT_PREFIX=e108 PARENT_SUFFIX=allmom PARENT_LABEL=allmom V0_PREFIX=e72 V0_SUFFIX=base \
  NO_MAIL=1 REPORT="logs/${PREFIX}_report_${v}.txt" SCOUT="${SCOUT:-0}" \
    bash experiments/mommask_core.sh > "logs/${PREFIX}_lane_${v}.log" 2>&1 &
  PID[$v]=$!; echo "lane ${v} (${data}, ${nf}f) pid ${PID[$v]}"
done

bad=()
for v in "${!PID[@]}"; do
  wait "${PID[$v]}" || bad+=("$v")
done
el=$(( ($(date +%s)-t0)/60 ))

REPORT="logs/${PREFIX}_report.txt"
{
  echo "${EXPERIMENT} finished in ${el}m; lanes with a failed arm: ${bad[*]:-none}"
  grep -h ">>> .* FAILED" logs/${PREFIX}_lane_*.log 2>/dev/null || true
  echo
  "$PYTHON" experiments/e112_report.py
} 2>&1 | tee "$REPORT"

# the interactive document: every E112 arm + its allmom base, ranked against the book
"$PYTHON" -m src.analysis.report_html --runs "ridge/${PREFIX}_ridge_*" "lgb_ktz/${PREFIX}_lgb_ktz_*" \
  "catboost_ktz/${PREFIX}_catboost_ktz_*" "rf_ktz/${PREFIX}_rf_ktz_*" "mlp_ktz/${PREFIX}_mlp_ktz_*" \
  ridge/e108_ridge_allmom lgb_ktz/e108_lgb_ktz_allmom catboost_ktz/e108_catboost_ktz_allmom \
  rf_ktz/e108_rf_ktz_allmom mlp_ktz/e108_mlp_ktz_allmom \
  --sort-by payout --title "E112 symbol encodings on allmom" --out "results/reports/crypto/e112_encodings.html" \
  > "logs/${PREFIX}_report_html.log" 2>&1 && echo "html: results/reports/crypto/e112_encodings.html" | tee -a "$REPORT" \
  || echo "note: report_html failed (see logs/${PREFIX}_report_html.log)" | tee -a "$REPORT"

[ "${NO_MAIL:-0}" = "1" ] || mail "[crypto] ${EXPERIMENT} finished -- $([ ${#bad[@]} -gt 0 ] && echo "WITH FAILURES (${bad[*]})" || echo SUCCESS)" "$REPORT"
exit $([ ${#bad[@]} -gt 0 ] && echo 1 || echo 0)
