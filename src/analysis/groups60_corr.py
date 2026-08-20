"""
Correlate the 60-day feature-group arms against every CURUPIRA slot.

    python3 -m src.analysis.groups60_corr

Run after experiments/feature_groups_target60_crypto.sh. Answers one question:
is there a narrow, single-group 60-day model that decorrelates the deployed book
STABLY, rather than on average?

------------------------------------------------------------------------------
Why the per-date distribution, not just the mean
------------------------------------------------------------------------------

CURUPIRA_05_60 entered the book on a reported correlation of 0.394-0.430 with
the 20-day slots -- the lowest figure in the study. That number was the mean of
a bimodal per-date distribution: sd 0.388, a quarter of days below 0.094, and
45% of days ABOVE 0.90. Diversification present on half the rounds is worth far
less than the mean suggests, and on the first live round the pair came in at
0.950.

So every pair here reports mean, sd, and the fraction of days above 0.90. A
candidate is only interesting if its mean is low AND its p90 fraction is small;
a low mean with a fat upper tail is the trap that has already been walked into
once.

Everything is scored and correlated on the periods the two arms share, and the
count is printed -- the 60-day arms cover 470 dates against the 20-day arms' 510.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

FORECAST = "results/forecast/crypto"
SYMBOL, PERIOD = "symbol", "date"

# The deployed book. CURUPIRA_06 and _07 are functions of the others, so they
# are included for completeness but flagged: a low correlation to a blend that
# contains the candidate would be double counting.
CURUPIRAS = [
    ("C01", "ridge_glmnet", "w6r6_glmnet_reg_tenc"),
    ("C02", "ridge", "w6r6_fixed_reg"),
    ("C03", "ridge_glmnet", "w6r6_glmnet_reg_onehot_rollsel"),
    ("C04", "ridge_glmnet", "w6r6_glmnet_reg_allfeatpp"),
    ("C05", "ridge_glmnet", "w6r6_blend_quintet2"),
    ("C05_60", "ridge_glmnet", "t60_c05_blend"),
]


def load(model, run):
    path = os.path.join(FORECAST, model, run + ".csv")
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, usecols=[SYMBOL, PERIOD, "prediction"],
                        parse_dates=[PERIOD])
    frame["rank"] = frame.groupby(PERIOD)["prediction"].rank(pct=True)
    return frame.set_index([SYMBOL, PERIOD])["rank"]


def per_date_rho(a, b):
    """
    (mean, sd, fraction above 0.90, n_dates) of the daily rank correlation.

    Computed with groupby aggregates rather than groupby.apply. Both inputs are
    already within-date percentile ranks, so a per-date Pearson correlation of
    them IS the Spearman correlation -- which makes the whole thing three
    vectorised group sums instead of a Python callback per date. On 84 pairs x
    470 dates that is the difference between seconds and minutes, and it avoids
    apply's habit of reshaping what it hands the callback.
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    if joined.empty:
        return np.nan, np.nan, np.nan, 0

    dates = joined.index.get_level_values(PERIOD)
    grouped = joined.groupby(dates)
    centred = joined - grouped.transform("mean")
    products = centred["a"] * centred["b"]
    squares = centred.pow(2)

    numerator = products.groupby(dates).sum()
    denominator = np.sqrt(squares["a"].groupby(dates).sum()
                          * squares["b"].groupby(dates).sum())
    rho = (numerator / denominator.replace(0.0, np.nan)).dropna()
    if rho.empty:
        return np.nan, np.nan, np.nan, 0
    return rho.mean(), rho.std(), float((rho > 0.90).mean()), len(rho)


def main():
    group_runs = sorted(glob.glob(os.path.join(FORECAST, "*", "t60_grp_*.csv")))
    if not group_runs:
        print("no t60_grp_* runs found -- run "
              "experiments/feature_groups_target60_crypto.sh first")
        raise SystemExit(1)

    book = {}
    for name, model, run in CURUPIRAS:
        series = load(model, run)
        if series is not None:
            book[name] = series
    if not book:
        print("no CURUPIRA forecasts found")
        raise SystemExit(1)

    candidates = {}
    for path in group_runs:
        # .csv is four characters. [:-5] is correct for the .json metrics files
        # and silently truncates a csv name by one letter, which then fails to
        # load and surfaces much later as a None in the correlation loop.
        run = os.path.basename(path)[:-len(".csv")]
        model = path.split(os.sep)[-2]
        candidates[run.replace("t60_grp_", "")] = load(model, run)

    # Standalone score of each candidate, for context: a decorrelated model that
    # predicts nothing is not a diversifier, it is noise.
    print("\n=== 60-day group arms: own score, on the 60-day target ===")
    print("%-24s %10s %9s %9s" % ("arm", "CORR", "sharpe", "maxdd"))
    for run in sorted(candidates):
        for model in ("ridge_glmnet", "ridge"):
            path = os.path.join("results/metrics/crypto", model,
                                "t60_grp_%s.json" % run)
            if os.path.exists(path):
                with open(path) as handle:
                    d = json.load(handle)
                print("%-24s %10.6f %9.4f %9.4f" % (
                    run, d["corr"]["mean"], d["corr"]["sharpe"],
                    d["corr"]["max_drawdown"]))
                break

    print("\n=== mean per-date rank correlation vs the book ===")
    header = "%-24s" % "arm" + "".join("%9s" % n for n in book)
    print(header)
    detail = []
    for run in sorted(candidates):
        row = "%-24s" % run
        for name in book:
            mean, sd, p90, n = per_date_rho(candidates[run], book[name])
            row += "%9.3f" % mean if np.isfinite(mean) else "%9s" % "-"
            detail.append((run, name, mean, sd, p90, n))
        print(row)

    print("\n=== stability: the pairs with the lowest MEAN correlation ===")
    print("a low mean with a fat upper tail is not diversification")
    print("%-24s %-8s %8s %8s %10s %6s" % (
        "arm", "vs", "mean", "sd", "frac>0.90", "dates"))
    for run, name, mean, sd, p90, n in sorted(
            [d for d in detail if np.isfinite(d[2])], key=lambda d: d[2])[:15]:
        print("%-24s %-8s %8.3f %8.3f %10.3f %6d" % (run, name, mean, sd, p90, n))

    stable = [d for d in detail
              if np.isfinite(d[2]) and d[2] < 0.5 and d[4] < 0.15]
    print("\ncandidates with mean rho < 0.50 AND fewer than 15%% of days above "
          "0.90: %d" % len(stable))
    for run, name, mean, sd, p90, n in sorted(stable, key=lambda d: d[2])[:10]:
        print("   %-24s vs %-8s mean %.3f  sd %.3f  frac>0.90 %.3f"
              % (run, name, mean, sd, p90))
    if not stable:
        print("   none -- every 60-day group arm either tracks the book or "
              "decorrelates only intermittently, which is the same pattern "
              "CURUPIRA_05_60 showed.")


if __name__ == "__main__":
    main()
