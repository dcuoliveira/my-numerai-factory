"""
Score a standalone TimesFM forecast frame into the pipeline's output shape.

    python -m src.analysis.timesfm_score --run-name e25_target

The zero-shot inference (src.analysis.timesfm_zeroshot, run in the tfm venv)
writes only symbol,date,target,prediction. BMC/MMC need the benchmark and meta
model columns per row; those are identical across models, so they are merged from
an already-scored run on (symbol, date). Then summarize_predictions /
per_period_table -- the same functions run_forecast uses -- produce the metrics
JSON and per_period CSV, so paired_delta and the report tools treat the timesfm
arm like any other.
"""

import argparse
import json
import os

import pandas as pd

from src.prediction.metrics import summarize_predictions, per_period_table

TARGET = "target_binned_return_20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="timesfm",
                    help="forecast-tree the standalone run lives under (timesfm, tempo, ...)")
    ap.add_argument("--run-name", default="e38_target")
    ap.add_argument("--meta-from", default="tabpfn/e19_fenccausal",
                    help="run to borrow reference_prediction/meta_model columns from")
    ap.add_argument("--data-version", default="crypto/v2.0")
    args = ap.parse_args()

    fc = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (args.model, args.run_name),
                     parse_dates=["date"])
    fc["symbol"] = fc["symbol"].astype(str)              # ucid, e.g. '1','2'
    # The meta model series lives in historical_meta_models, keyed on `id` (the
    # ucid this forecast uses), not the ticker `symbol` the pipeline's forecast
    # frames key on. For crypto the BMC benchmark and the meta model coincide
    # (run_forecast reports BMC == MMC), so one series serves both.
    meta = pd.read_parquet("data/%s/historical_meta_models.parquet"
                           % args.data_version, columns=["id", "date", "prediction"])
    meta = meta.rename(columns={"id": "symbol", "prediction": "meta_model"})
    meta["symbol"] = meta["symbol"].astype(str)
    df = fc.merge(meta, on=["symbol", "date"], how="left")
    df["reference_prediction"] = df["meta_model"]

    summ = summarize_predictions(df, "prediction", TARGET, "date",
                                 reference_col="reference_prediction",
                                 meta_model_col="meta_model")
    table = per_period_table(df, "prediction", TARGET, "date",
                             reference_col="reference_prediction",
                             meta_model_col="meta_model")

    n = int(table["corr"].notna().sum())
    summ["n_periods"] = n
    summ["config"] = {"data_version": args.data_version, "n_features": 1,
                      "target": TARGET, "model": args.model,
                      "note": "zero-shot univariate, one symbol at a time"}
    summ["meta_model"] = {"column": "meta_model",
                          "row_coverage": float(df["meta_model"].notna().mean())}

    pp_dest = "results/per_period/crypto/%s/%s.csv" % (args.model, args.run_name)
    mt_dest = "results/metrics/crypto/%s/%s.json" % (args.model, args.run_name)
    os.makedirs(os.path.dirname(pp_dest), exist_ok=True)
    os.makedirs(os.path.dirname(mt_dest), exist_ok=True)
    table.reset_index().rename(columns={"index": "date"}).to_csv(pp_dest, index=False)
    with open(mt_dest, "w") as handle:
        json.dump(summ, handle, indent=2, default=float)

    payout = 0.1 * summ["corr"]["mean"] + summ["mmc"]["mean"]
    print("scored %s: CORR %.6f  MMC %.6f  payout %.6f  (n=%d, rows=%d)"
          % (args.run_name, summ["corr"]["mean"], summ["mmc"]["mean"],
             payout, n, len(df)), flush=True)
    print("  wrote %s\n  wrote %s" % (pp_dest, mt_dest))


if __name__ == "__main__":
    main()
