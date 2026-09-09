"""
E20's report: does exponential recency weighting help, dataset by dataset?

    python3 -m src.analysis.e20_report            # after the runner finishes
    python3 -m src.analysis.e20_report scout      # the downsampled rehearsal

Every row is a PAIRED comparison on one dataset: the same estimator, the same
alpha grid, the same folds and the same seed, differing only in W. So the delta
measures the weighting and nothing else, and it is formed per period and
summarised once -- never rebuilt from the two stored means, which would combine
two summaries of correlated series into a plausible wrong number.

The `half_life` column is the value the search actually chose at each refit. It
is the arm's own diagnostic: a treatment arm that keeps selecting the longest
half-life on the grid is telling you the decay is being switched off, and that
is a different finding from "decay hurts".
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

PER_PERIOD = os.path.join("results", "per_period", "crypto", "gen_ridge")
METRICS = os.path.join("results", "metrics", "crypto", "gen_ridge")
LAG = 20

#: tag -> (features, the submitted slots this dataset feeds)
DATASETS = [
    ("raw", 22, "CURUPIRA_02, SACI_04"),
    ("tenc", 23, "CURUPIRA_01"),
    ("interact", 36, "SACI_01"),
    ("fenccausal", 44, "IA_01"),
    ("cumlag3", 88, "SACI_05"),
    ("tencohot", 763, "SACI_03"),
    ("onehotroll", 786, "CURUPIRA_03"),
    ("allfeat", 788, "SACI_02"),
    ("allfeatpp", 812, "CURUPIRA_04"),
]


def load(run):
    """The run's per-period frame and metrics, or None if it did not finish."""
    per = os.path.join(PER_PERIOD, run + ".csv")
    met = os.path.join(METRICS, run + ".json")
    if not (os.path.exists(per) and os.path.exists(met)):
        return None
    return pd.read_csv(per).set_index("date"), json.load(open(met))


def payout(frame):
    """Per-period payout, formed here rather than from the stored summaries."""
    return 0.1 * frame["corr"] + frame["mmc"]


def summarise(series):
    """mean, sharpe and POSITIVE max drawdown, the repo's conventions."""
    cumulative = series.cumsum()
    return (float(series.mean()),
            float(series.mean() / series.std(ddof=0)),
            float((cumulative.expanding(min_periods=1).max() - cumulative).max()))


def chosen_half_lives(metrics):
    """The half-life the search picked at each refit, as a compact string."""
    picked = []
    for refit in metrics.get("refits", []):
        value = (refit.get("best_params") or {}).get("half_life")
        if value is not None:
            picked.append(int(round(float(value))))
    if not picked:
        return "-"
    return "/".join(str(v) for v in picked)


def main():
    scout = len(sys.argv) > 1 and sys.argv[1].startswith("scout")
    prefix = "e20scout_" if scout else "e20_"

    print("\n" + "=" * 116)
    print("E20 -- exponential recency weighting against uniform, on every dataset "
          "behind a submitted model")
    print("  same estimator, same alpha grid, same folds, same seed. W is the "
          "only thing that differs.")
    print("=" * 116)
    print("  %-12s %5s %8s %10s %10s %8s %8s | %11s %7s %5s | %s"
          % ("dataset", "feat", "arm", "CORR", "payout", "p.shp", "dd(pay)",
             "d payout", "t", "n", "half-life chosen"))

    rows = []
    for tag, n_features, slots in DATASETS:
        control = load(prefix + "unif_" + tag)
        decay = load(prefix + "exp_" + tag)
        if control is None or decay is None:
            print("  %-12s %5d  (not finished: %s)"
                  % (tag, n_features,
                     ", ".join(n for n, v in (("uniform", control), ("exp_decay", decay))
                               if v is None)))
            continue
        control_frame, control_metrics = control
        decay_frame, decay_metrics = decay

        control_pay = payout(control_frame)
        decay_pay = payout(decay_frame)
        joined = pd.concat([control_pay, decay_pay], axis=1, join="inner",
                           keys=["base", "arm"]).dropna()
        difference = joined["arm"] - joined["base"]
        mean, _, t = newey_west(difference.to_numpy(), lag=LAG)

        for label, frame, metrics in (("uniform", control_frame, control_metrics),
                                      ("exp_decay", decay_frame, decay_metrics)):
            pay = payout(frame)
            level, sharpe, drawdown = summarise(pay)
            if label == "uniform":
                print("  %-12s %5d %8s %10.6f %10.6f %8.2f %8.2f | %11s %7s %5s | %s"
                      % (tag, n_features, label, metrics["corr"]["mean"], level,
                         sharpe, drawdown, "-", "-", "-", ""))
            else:
                print("  %-12s %5s %8s %10.6f %10.6f %8.2f %8.2f | %+11.6f %7.2f %5d | %s"
                      % ("", "", label, metrics["corr"]["mean"], level, sharpe,
                         drawdown, mean, t, len(joined),
                         chosen_half_lives(metrics)))
        rows.append((tag, mean, t, len(joined), slots))

    if not rows:
        print("\n  nothing to report yet")
        return

    print("\n" + "=" * 116)
    strong = [r for r in rows if abs(r[2]) > 2]
    print("  %d of %d datasets show a payout difference at |t| > 2"
          % (len(strong), len(rows)))
    for tag, mean, t, _, slots in strong:
        print("    %-12s %+.6f  t %+.2f   feeds %s" % (tag, mean, t, slots))
    if not strong:
        low = min(r[1] for r in rows)
        high = max(r[1] for r in rows)
        print("    None does. Payout deltas span %+.6f to %+.6f across %d datasets,"
              % (low, high, len(rows)))
        print("    inside the noise of ~26 effective independent windows. That is")
        print("    the finding, not an inconclusive run.")
        print("")
        print("    Read it as evidence about WEIGHTING, not about staleness: this")
        print("    study holds --refit-every at 130. Weighting the end of a")
        print("    training block sharpens the fit for the START of a block that")
        print("    then goes stale for another 130 days, so if recency pays at all")
        print("    the more direct lever is a shorter refit cadence, which is not")
        print("    varied here.")

    positive = sum(1 for r in rows if r[1] > 0)
    print("\n  sign count: %d of %d datasets positive (a coin flip is %.1f)"
          % (positive, len(rows), len(rows) / 2.0))


if __name__ == "__main__":
    main()
