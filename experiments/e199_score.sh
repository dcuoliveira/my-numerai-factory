#!/usr/bin/env bash
# E199: wait for the six refitted per-coin TabNet prediction files, then score against E198's pooled arms.
set -euo pipefail
cd /home/danielco/Downloads/my-numerai-agent
export E198_VARIANT=e199
P=results/reports/crypto/e199_tabnet_percoin/preds/tabnet
while true; do
  missing=0; for a in TN_B TN_F; do for s in 0 1 2; do [ -f "$P/${a}_s$s.parquet" ] || missing=$((missing+1)); done; done
  [ "$missing" -eq 0 ] && break
  echo "$(date '+%F %T') waiting on $missing prediction files"; sleep 300
done
python3 experiments/e198_cs_eval.py --score --workers 8
python3 -m src.analysis.html_pdf results/reports/crypto/e199_tabnet_percoin/E199_cs_report.html
