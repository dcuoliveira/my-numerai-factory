#!/usr/bin/env bash
# E200: fit the searched mlp_ktz arms here (minutes), wait for the searched lgb_ktz lanes, then score + PDF.
# The searched ridge arms were fitted at registration (seconds; picks in preds/ridge/*_picks.csv).
set -euo pipefail
cd /home/danielco/Downloads/my-numerai-agent
export E198_VARIANT=e200
python3 experiments/e198_cs_eval.py --fit-pooled --model ridge --n-jobs 1
python3 experiments/e198_cs_eval.py --fit-pooled --model mlp_ktz --n-jobs 1
P=results/reports/crypto/e200_cs_tuned/preds/lgb_ktz
while true; do
  missing=0; for a in B_ALL B_COV F_COV; do for s in 0 1 2; do [ -f "$P/${a}_s$s.parquet" ] || missing=$((missing+1)); done; done
  [ "$missing" -eq 0 ] && break
  echo "$(date '+%F %T') waiting on $missing lgb prediction files"; sleep 300
done
python3 experiments/e198_cs_eval.py --score --workers 2
python3 -m src.analysis.html_pdf results/reports/crypto/e200_cs_tuned/E200_cs_report.html
