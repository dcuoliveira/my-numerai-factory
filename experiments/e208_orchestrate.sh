#!/bin/bash
# E208: wait for the six 60-day TabNet fits, then build ensembles, report and PDF.
cd /home/danielco/Downloads/my-numerai-agent
D=results/reports/crypto/e208_tabnet_t60/preds/tabnet
while [ "$(ls $D/TN_?_s?.parquet 2>/dev/null | wc -l)" -lt 6 ]; do sleep 120; done
/home/danielco/miniconda3/bin/python3 experiments/e208_tabnet_ensembles.py && \
/home/danielco/miniconda3/bin/python3 -m src.analysis.html_pdf results/reports/crypto/e208_tabnet_t60/E208_report.html
