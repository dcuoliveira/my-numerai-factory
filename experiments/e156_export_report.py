"""Export completed E156 analysis on a host with working Chrome.

This recovery path never retrains, changes the frozen study or rescores data.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/reports/crypto/e156_rawvol_addback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="lovelace")
    args = ap.parse_args()
    states = list((ROOT / "batch_status").glob("*.json"))
    assert len(states) == 12 and all(json.loads(p.read_text())["state"] == "complete" for p in states)
    registration = json.loads((ROOT / "registration.json").read_text())
    runner = REPO / "experiments/e156_rawvol_addback.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == registration["runner_sha256"]
    headline = pd.read_csv(ROOT / "headline.csv")
    paired = pd.read_csv(ROOT / "paired_deltas.csv")
    mmc = paired[paired.arm.eq("tuneFULL") & paired.metric.eq("mmc")]
    source = ROOT / "E156_does_adding_raw_volatility_help_DEVELOPMENT.html"
    document = source.read_text()
    note = ('<p><b>Headline metric: mean CORR.</b> Proxy MMC also increases in all three models, '
            f'but its paired t-statistics range from {mmc.t.min():.2f} to {mmc.t.max():.2f}, '
            'below the |t| = 2 threshold. CORR and MMC use different numbers of dates; '
            'each paired comparison uses the same dates for both feature sets.</p>')
    marker = '<p>Development-only experiment.'
    if '<b>Headline metric: mean CORR.</b>' not in document:
        document = document.replace(marker, note + marker, 1)
        source.write_text(document)
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host,
                    sys.executable, str(REPO / "src/analysis/html_pdf.py"), str(source)], check=True)
    verdict = "PROMISING_DEVELOPMENT" if (headline.t >= 2).all() else "DEVELOPMENT"
    target = ROOT / ("E156_does_adding_raw_volatility_help_" + verdict)
    for ext in ("html", "pdf"):
        destination = target.with_suffix("." + ext)
        original = source.with_suffix("." + ext)
        if original != destination:
            shutil.copyfile(original, destination)
    scores = pd.read_parquet(ROOT / "scores_by_date_seed.parquet", columns=["model", "arm", "seed", "date"])
    complete = dict(experiment="E156", completed=time.time(), report=target.with_suffix(".pdf").name,
                    report_sha256=hashlib.sha256(target.with_suffix(".pdf").read_bytes()).hexdigest(),
                    scoring_rows=len(scores), development_dates=int(scores.date.nunique()),
                    training_batches=12, headline="Searched FULL minus searched EXRAWVOL; positive means raw volatility helps",
                    source_runner_unchanged=True, report_export_host=args.host,
                    export_recovery="Local snap Chromium failed; remote Chrome successfully exported the completed analysis")
    (ROOT / "COMPLETE.json").write_text(json.dumps(complete, indent=2))
    print(target.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
