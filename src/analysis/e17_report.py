"""
E17's report: cumulative per-symbol differences, on ridge_glmnet.

    python3 -m src.analysis.e17_report            # after the runner finishes
    python3 -m src.analysis.e17_report _scout     # the downsampled rehearsal

Three questions, three tables, all paired against `e17_raw` -- the control
refitted on the CURRENT snapshot, so no delta here confounds the feature set
with the data refresh.

    1. do the difference columns work ALONE            e17_diff<j>
    2. do they add anything BESIDE the raw 22          e17_rawdiff<j>
    3. same span, different basis: differences vs lags e17_featdiff<j> / e17_lag<j>

The arithmetic is ablation_report.newey_west and prediction.metrics.summarize,
imported rather than reimplemented, so these numbers are the report tools'.
Payout is formed PER PERIOD and summarised once -- never rebuilt from the two
stored means, which would combine two summaries of correlated series and produce
a plausible wrong number.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from src.analysis.ablation_report import newey_west

PER_PERIOD = os.path.join("results", "per_period", "crypto", "ridge_glmnet")
METRICS = os.path.join("results", "metrics", "crypto", "ridge_glmnet")
LAG = 20

#: The ladder, and what each rung means in calendar terms. 261 dates is one year
#: on this panel, measured (1,714 dates over 2,399 calendar days), not assumed
#: from a 252 trading-day convention.
ORDERS = [1, 2, 3, 5, 10, 21, 63, 130, 261]
HORIZON = ["1d", "2d", "3d", "1wk", "2wk", "1mo", "1qtr", "6mo", "1yr"]

#: Share of rows with a real predecessor at each order, from build_cumdiff's own
#: coverage report. Printed beside every arm because a mostly-filled column is a
#: mostly-constant column, and a flat deep rung must not be read as "a year of
#: change carries nothing" when it is silent on two rows in five.
COVERAGE = {1: 99.3, 2: 99.0, 3: 98.8, 5: 98.3, 10: 97.2,
            21: 94.8, 63: 86.7, 130: 76.0, 261: 59.9}

BASIS_LEVELS = [3, 6, 9]


def load(run):
    """
    One run's per-period frame and its config, or None if it did not finish.

    :param run: run name
    :type run: str
    :return: (per-period frame, metrics dict) or None
    :rtype: (pd.DataFrame, dict) or None
    """
    per = os.path.join(PER_PERIOD, run + ".csv")
    met = os.path.join(METRICS, run + ".json")
    if not (os.path.exists(per) and os.path.exists(met)):
        return None
    frame = pd.read_csv(per).set_index("date")
    return frame, json.load(open(met))


def payout(frame):
    """Per-period payout. Formed here, never from the stored summaries."""
    return 0.1 * frame["corr"] + frame["mmc"]


def summarise(series):
    """mean, sharpe and POSITIVE max drawdown, the repo's conventions."""
    cumulative = series.cumsum()
    return (float(series.mean()),
            float(series.mean() / series.std(ddof=0)),
            float((cumulative.expanding(min_periods=1).max() - cumulative).max()))


def delta(base_pay, arm_pay):
    """Paired Newey-West delta of payout on the inner join of periods."""
    joined = pd.concat([base_pay, arm_pay], axis=1, join="inner",
                       keys=["base", "arm"]).dropna()
    if len(joined) < 30:
        return np.nan, np.nan, len(joined)
    difference = joined["arm"] - joined["base"]
    mean, _, t = newey_west(difference.to_numpy(), lag=LAG)
    return mean, t, len(joined)


def table(title, rows, control, note=None):
    """Print one block. `rows` is [(label, order-or-None, run)]."""
    loaded = load(control)
    if loaded is None:
        print("\n%s\n  control %s has not finished -- nothing to compare against"
              % (title, control))
        return []
    base_frame, base_met = loaded
    base_pay = payout(base_frame)
    b_mean, b_sharpe, b_dd = summarise(base_pay)

    print("\n" + "=" * 104)
    print(title)
    if note:
        print(note)
    print("=" * 104)
    print("  %-18s %5s %5s %8s %10s %10s %10s %8s %8s %11s %7s"
          % ("arm", "feat", "cov", "horizon", "CORR", "MMC", "payout",
             "p.shp", "dd(pay)", "d payout", "t"))
    print("  %-18s %5d %5s %8s %10.6f %10.6f %10.6f %8.2f %8.2f %11s %7s"
          % ("control", base_met["config"]["n_features"], "-", "-",
             base_met["corr"]["mean"], base_met["mmc"]["mean"],
             b_mean, b_sharpe, b_dd, "-", "-"))

    out = []
    for label, order, run in rows:
        loaded = load(run)
        if loaded is None:
            print("  %-18s %5s %5s %8s   (not finished)" % (label, "-", "-", "-"))
            continue
        frame, met = loaded
        pay = payout(frame)
        mean, sharpe, dd = summarise(pay)
        d, t, n = delta(base_pay, pay)
        cov = "%.0f%%" % COVERAGE[order] if order in COVERAGE else "-"
        horizon = HORIZON[ORDERS.index(order)] if order in ORDERS else "-"
        print("  %-18s %5d %5s %8s %10.6f %10.6f %10.6f %8.2f %8.2f %+11.6f %7.2f"
              % (label, met["config"]["n_features"], cov, horizon,
                 met["corr"]["mean"], met["mmc"]["mean"], mean, sharpe, dd, d, t))
        out.append((label, d, t))
    return out


def verdict(rows, what):
    """State the finding, including when it is a null -- especially then."""
    if not rows:
        print("\n  no %s arms finished" % what)
        return
    strong = [r for r in rows if abs(r[2]) > 2]
    print("\n  %d of %d %s arms differ from the control at |t| > 2"
          % (len(strong), len(rows), what))
    for label, d, t in strong:
        print("    %-18s %+.6f  t %+.2f" % (label, d, t))
    if not strong:
        low = min(r[1] for r in rows)
        high = max(r[1] for r in rows)
        print("    None does. Payout deltas span %+.6f to %+.6f, inside the noise"
              % (low, high))
        print("    of ~26 effective independent windows. That is the finding, not")
        print("    an inconclusive run.")


def resolve(name, scout):
    """
    A run's artifact name. Scouts are PREFIXED e17scout_, never suffixed:
    experiment_db resolves a run to its study by prefix, so an _scout suffix
    would file every rehearsal as an extra full arm.
    """
    return ("e17scout_" + name[len("e17_"):]) if scout else name


def main():
    scout = len(sys.argv) > 1 and sys.argv[1] in ("_scout", "scout", "--scout")
    control = resolve("e17_raw", scout)

    levels = list(range(1, len(ORDERS) + 1))

    alone = table(
        "1. THE DIFFERENCES ALONE -- do changes replace levels?",
        [("e17_diff%d" % j, ORDERS[j - 1], resolve("e17_diff%d" % j, scout))
         for j in levels],
        control,
        note="  level j carries difference orders %s, cumulative. 'cov' is the\n"
             "  share of rows with a real predecessor at the DEEPEST order in\n"
             "  the arm; the rest are the 0.0 fill."
             % ",".join(str(o) for o in ORDERS))
    verdict(alone, "differences-alone")

    beside = table(
        "2. THE DIFFERENCES BESIDE THE RAW 22 -- do changes add to levels?",
        [("e17_rawdiff%d" % j, ORDERS[j - 1], resolve("e17_rawdiff%d" % j, scout))
         for j in levels],
        control)
    verdict(beside, "raw+differences")

    rows = []
    for j in BASIS_LEVELS:
        rows.append(("e17_featdiff%d" % j, ORDERS[j - 1],
                     resolve("e17_featdiff%d" % j, scout)))
        rows.append(("e17_lag%d" % j, ORDERS[j - 1], resolve("e17_lag%d" % j, scout)))
    table(
        "3. SAME SPAN, DIFFERENT BASIS -- differences against lagged levels",
        rows, control,
        note="  featdiff<j> and lag<j> carry the SAME COLUMN SPAN and the same\n"
             "  feature count; they differ only in which basis spans it. An\n"
             "  unpenalised fit could not tell them apart. Ridge can, because\n"
             "  ||b||^2 is not invariant under a change of basis.")

    # The basis question is a PAIRED one between two arms, not against the
    # control, so it gets its own comparison rather than a difference of two
    # deltas -- which would double the standard error for no reason.
    print("\n  the basis delta itself, featdiff minus lag, paired:")
    print("    %-10s %11s %7s %7s" % ("level", "d payout", "t", "n"))
    for j in BASIS_LEVELS:
        a = load(resolve("e17_featdiff%d" % j, scout))
        b = load(resolve("e17_lag%d" % j, scout))
        if a is None or b is None:
            print("    %-10d   (not finished)" % j)
            continue
        d, t, n = delta(payout(b[0]), payout(a[0]))
        print("    %-10d %+11.6f %7.2f %7d" % (j, d, t, n))
    print("\n    A positive t here says the DIFFERENCE basis shrinks better than")
    print("    the lagged-level basis at identical span -- the study's hypothesis.")
    print("    A null says the basis does not matter and the span is what was")
    print("    already measured null by E05, E06, E13 and E14.")

    # Attribution, and the snapshot price.
    print("\n" + "=" * 104)
    print("4. ATTRIBUTION AND CONTROLS")
    print("=" * 104)
    for label, run in (("target diffs only", resolve("e17_tgtdiff6", scout)),
                       ("old-snapshot control", "w6r6_glmnet_reg")):
        loaded = load(run)
        base = load(control)
        if loaded is None or base is None:
            print("  %-22s (not available)" % label)
            continue
        d, t, n = delta(payout(base[0]), payout(loaded[0]))
        print("  %-22s %-22s d %+.6f  t %+.2f  on %d shared periods"
              % (label, run, d, t, n))
    print("\n  The old-snapshot row prices the DATA REFRESH, not a feature set:")
    print("  w6r6_glmnet_reg is this control's exact config on 1,701 dates and")
    print("  1,309 symbols against today's 1,714 and 1,317. Read it before")
    print("  reading any delta in this study against a number from before it.")


if __name__ == "__main__":
    main()
