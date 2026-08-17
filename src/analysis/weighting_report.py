"""
Rank weighting configurations against 1/N, paired and lag-corrected.

    python3 -m src.analysis.weighting_report
    python3 -m src.analysis.weighting_report --scheme shrunk_tangency

Two things this does that reading the metrics files does not.

PAIRED DIFFERENCES. Every configuration blends the same five members over the
same 510 periods, so the arms are near-copies and the difference series has a
far smaller standard error than either arm alone. Comparing means side by side
would find nothing; the first sweep's significant results (|t| up to 2.57) only
appear when the comparison is paired.

THE RIGHT BASELINE. Each configuration is compared against `equal` AT ITS OWN
CADENCE AND WINDOW, not against a single global baseline. 1/N does not actually
depend on either, so those baselines should all be identical -- and the report
checks that rather than assuming it. If the equal-weight arms disagree, the
harness is not neutral and no comparison here means anything.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

METRICS = "results/metrics/crypto"
PER_PERIOD = "results/per_period/crypto"
NEWEY_WEST_LAG = 20


def payout(frame):
    """Crypto pays 0.1*CORR + 1*MMC."""
    return 0.1 * frame["corr"] + frame["mmc"]


def newey_west(series, lag=NEWEY_WEST_LAG):
    """(mean, se, t) for the mean of an autocorrelated series."""
    x = np.asarray(series, dtype="float64")
    n = len(x)
    mean = x.mean()
    e = x - mean
    variance = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(variance, 0.0) / n)
    return mean, se, (mean / se if se > 0 else np.nan)


def summarise(series):
    x = np.asarray(series, dtype="float64")
    mean, sd = x.mean(), x.std(ddof=0)
    cumulative = x.cumsum()
    return mean, (mean / sd if sd > 0 else np.nan), \
        (np.maximum.accumulate(cumulative) - cumulative).max()


def read_per_period(path):
    """
    Load a per-period table, tolerating files written before the date was kept.

    run_weighted_ensemble used to write these with index=False, which dropped
    the date. Those files can still be paired -- every sweep arm blends the same
    members over the same periods in the same order, so row i is the same date
    in every arm -- but ONLY positionally, and only against other arms of the
    same length. The date index is what makes that a fact rather than a hope, so
    the writer now keeps it and this fallback exists to read the old files.

    :param path: per-period csv
    :type path: str
    :return: frame indexed by date, or by position for a legacy file
    :rtype: pd.DataFrame
    """
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        return frame.set_index("date")
    return frame


def collect(prefix="w6r6_swp_"):
    """Every completed sweep configuration, with its config block."""
    rows = []
    for path in sorted(glob.glob(os.path.join(METRICS, "*", prefix + "*.json"))):
        run = os.path.basename(path)[:-5]
        model = path.split(os.sep)[-2]
        per_period = os.path.join(PER_PERIOD, model, run + ".csv")
        if not os.path.exists(per_period):
            continue
        with open(path) as handle:
            metrics = json.load(handle)
        frame = read_per_period(per_period)
        config = metrics.get("config", {})
        mean, sharpe, drawdown = summarise(payout(frame))
        rows.append({
            "run": run, "name": run[len(prefix):],
            "scheme": config.get("scheme"),
            "cadence": config.get("update_every"), "window": config.get("window"),
            "shrinkage": config.get("shrinkage"),
            "subset_size": config.get("subset_size"),
            "keep_fraction": config.get("keep_fraction"),
            "constrained": not config.get("gr_unconstrained", False),
            "series": payout(frame), "corr": frame["corr"].mean(),
            "mmc": frame["mmc"].mean(),
            "payout": mean, "sharpe": sharpe, "maxdd": drawdown,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rank weighting configurations.")
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    rows = collect()
    if not rows:
        print("no sweep runs found -- run experiments/weighting_sweep_crypto.sh")
        return

    # Baseline per (cadence, window). 1/N is invariant to both, so these must
    # agree; a disagreement means the harness is doing something to the arms.
    baselines = {(r["cadence"], r["window"]): r
                 for r in rows if r["scheme"] == "equal"}
    if baselines:
        spread = max(r["payout"] for r in baselines.values()) - \
            min(r["payout"] for r in baselines.values())
        print("neutrality check: %d equal-weight arms, payout spread %.2e" % (
            len(baselines), spread))
        if spread > 1e-9:
            print("  WARNING: 1/N should not depend on cadence or window. "
                  "The harness is not neutral; treat every comparison below "
                  "as suspect.")
        print()

    fallback = next(iter(baselines.values())) if baselines else None
    for row in rows:
        base = baselines.get((row["cadence"], row["window"]), fallback)
        if base is None or base["run"] == row["run"]:
            row["d_payout"], row["t"] = 0.0, np.nan
            continue
        joined = pd.concat([base["series"], row["series"]], axis=1,
                           join="inner", keys=["base", "arm"])
        mean, _, t = newey_west(joined["arm"] - joined["base"])
        row["d_payout"], row["t"] = mean, t

    shown = [r for r in rows if not args.scheme or r["scheme"] == args.scheme]
    shown.sort(key=lambda r: -r["payout"])

    print("%-26s %-19s %5s %6s %10s %8s %8s %11s %7s" % (
        "config", "scheme", "cad", "win", "payout", "sharpe", "maxdd",
        "d vs 1/N", "t"))
    for row in shown[:args.top]:
        print("%-26s %-19s %5s %6s %10.6f %8.4f %8.4f %+11.6f %7s" % (
            row["name"], row["scheme"], row["cadence"], row["window"],
            row["payout"], row["sharpe"], row["maxdd"], row["d_payout"],
            "%.2f" % row["t"] if np.isfinite(row["t"]) else "-"))

    beat = [r for r in rows if r["scheme"] != "equal" and r["d_payout"] > 0]
    strong = [r for r in beat if np.isfinite(r["t"]) and r["t"] > 2]
    print("\n%d of %d estimated configurations beat 1/N on the point estimate; "
          "%d at t > 2" % (len(beat), len([r for r in rows if r["scheme"] != "equal"]),
                           len(strong)))
    if not strong:
        print("None clears t > 2. With ~25 independent windows for 4 free "
              "weights this is the forecast combination puzzle behaving exactly "
              "as documented, and 1/N stands.")


if __name__ == "__main__":
    main()
