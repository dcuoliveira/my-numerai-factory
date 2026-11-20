#!/usr/bin/env bash
# E198: wait until every fit has written its predictions, then score, render the report and print the PDF.
set -euo pipefail
cd /home/danielco/Downloads/my-numerai-agent
P=results/reports/crypto/e198_cs/preds
need() {
  for m in lgb_ktz mlp_ktz; do for a in B_ALL B_COV F_COV; do for s in 0 1 2; do echo "$P/$m/${a}_s$s.parquet"; done; done; done
  for a in B_ALL B_COV F_COV; do echo "$P/ridge/${a}_s0.parquet"; done
  for a in TN_B TN_F; do for s in 0 1 2; do echo "$P/tabnet/${a}_s$s.parquet"; done; done
}
while true; do
  missing=0; for f in $(need); do [ -f "$f" ] || missing=$((missing+1)); done
  [ "$missing" -eq 0 ] && break
  echo "$(date '+%F %T') waiting on $missing prediction files"; sleep 300
done
python3 experiments/e198_cs_eval.py --score --workers 8
python3 -m src.analysis.html_pdf results/reports/crypto/e198_cs/E198_cs_report.html
