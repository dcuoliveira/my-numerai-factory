"""
Per-period series and their summaries, computed on ONE series at a time.

Exists because of a reporting bug worth not repeating: several tables printed
`corr.max_drawdown` in a column sitting next to `payout`. Both numbers were
correct, but they describe DIFFERENT series -- the drawdown of CORR against the
mean of 0.1*CORR + MMC -- and the payout drawdown runs 1.8-3.4x larger. A reader
takes the two adjacent columns as one story, so the pairing is the bug even
though neither figure is wrong on its own.

Rule: a drawdown belongs to the series named beside it. Label it, or print both.
"""
import os
import numpy as np
import pandas as pd


def per_period(run, model="lgb", tournament="crypto"):
    """The per-period table for a run, indexed by period."""
    path = "results/per_period/%s/%s/%s.csv" % (tournament, model, run)
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    return frame.set_index("date") if "date" in frame.columns else frame


def payout(frame):
    """Crypto pays 0.1*CORR + 1*MMC."""
    return 0.1 * frame["corr"] + frame["mmc"]


def drawdown(series):
    """
    Largest peak-to-trough drop of the CUMULATIVE score, returned positive.

    Same convention as metrics.summarize(): bigger is worse. Cumulative, not
    per-period -- a drawdown is about a run of bad periods, not one bad period.
    """
    cumulative = np.asarray(series, dtype="float64").cumsum()
    return float((np.maximum.accumulate(cumulative) - cumulative).max())


def calmar(series):
    """
    Mean payout per period divided by the cumulative drawdown -- return per unit
    of worst-case pain, the drawdown analogue of a Sharpe ratio.

    Sharpe divides by the VOLATILITY of the series; this divides by its deepest
    peak-to-trough run. They rank arms differently and both are worth showing: a
    series can be steady period to period (good Sharpe) and still grind out one
    long losing stretch (bad Calmar), which is what a stake actually feels.

    Note the units are not a Sharpe's. The numerator is per-period and the
    denominator is cumulative, so the number scales with 1/n_periods and is only
    comparable ACROSS ARMS MEASURED ON THE SAME WINDOW. Never compare a
    510-period arm's Calmar to a 470-period one.
    """
    dd = drawdown(series)
    if dd <= 0:
        return float("inf")
    return float(np.asarray(series, dtype="float64").mean() / dd)


def summary(series):
    """(mean, sharpe, max_drawdown) for one series."""
    x = pd.Series(np.asarray(series, dtype="float64"))
    sd = x.std(ddof=0)
    return float(x.mean()), float(x.mean() / sd) if sd > 0 else float("nan"), drawdown(x)


def newey_west(series, lag=20):
    """(mean, t) for the mean of an autocorrelated series."""
    x = np.asarray(series, dtype="float64"); n = len(x)
    mean = x.mean(); e = x - mean; var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * ((e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(var, 0.0) / n)
    return mean, (mean / se if se > 0 else float("nan"))
