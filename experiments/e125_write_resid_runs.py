"""
Persist E119/E125's residualised and 20-day re-scored series as ordinary runs, so
the house board (report_html) ranks and correlates them beside the book.

    python3 experiments/e125_write_resid_runs.py

  meta_resid/e119_w<N>_s050, meta_resid/e125_w<N>_s050   residualised (live transform,
      strength 0.5, meta >= 5 days old), scored on target_binned_return_20
  rescore20/e125_w<N>                                    fit-60 raw, re-scored on the 20-day label

Same Residualiser and scorer as src.analysis.e125_resid_report, so the board's
numbers are that report's numbers.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.analysis.cross_target import score_on_payable_target
from src.analysis.meta_resid_rescore import Residualiser
from src.prediction.metrics import summarize

WINDOWS = (5, 10, 15, 20, 22, 65, 130)
TARGET = "target_binned_return_20"


def write(tree, name, prediction, extras, config):
    table, _ = score_on_payable_target(prediction, extras)
    results = {"n_periods": int(table["corr"].notna().sum())}
    for family in ("corr", "bmc", "mmc"):
        results[family] = summarize(table[family].dropna())
        if family != "corr":
            results["%s_n_periods" % family] = int(table[family].notna().sum())
    results["config"] = dict(config, model=tree, tournament="crypto", task="regression", target=TARGET,
                             assembled_by="experiments/e125_write_resid_runs.py")
    for kind in ("forecast", "per_period", "metrics"):
        os.makedirs(os.path.join("results", kind, "crypto", tree), exist_ok=True)
    base = os.path.join("results", "%s", "crypto", tree, name)
    rows = extras.join(prediction.rename("prediction"), how="inner").dropna(subset=["prediction"])
    rows.reset_index()[["symbol", "date", TARGET, "prediction", "reference_prediction", "meta_model"]] \
        .to_csv(base % "forecast" + ".csv", index=False)
    table.to_csv(base % "per_period" + ".csv")
    with open(base % "metrics" + ".json", "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    pay = 0.1 * table["corr"] + table["mmc"]
    print("%-12s %-16s n %d  payout %.5f" % (tree, name, len(table), pay.mean()))


def main():
    res = Residualiser()
    for prefix, study in (("e119", "E119"), ("e125", "E125")):
        for w in WINDOWS:
            run = "%s_w%d" % (prefix, w)
            forecast = pd.read_csv(os.path.join("results", "forecast", "crypto", "mlp", run + ".csv"),
                                   usecols=["symbol", "date", "prediction"], parse_dates=["date"]) \
                .set_index(["symbol", "date"])
            source = {"source_run": "mlp/" + run, "study": study}
            write("meta_resid", run + "_s050", res.residualised(forecast), res.extras,
                  dict(source, strength=res.strength, meta_lag_days=res.lag))
            if prefix == "e125":
                write("rescore20", run, forecast.groupby(level=1)["prediction"].rank(pct=True), res.extras,
                      dict(source, note="fit on target_binned_return_60, scored on the 20-day label"))


if __name__ == "__main__":
    main()
