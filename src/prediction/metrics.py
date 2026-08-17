"""
Scoring for the Numerai Classic and Crypto tournaments.

Everything here is built on Numerai's own open-source scoring package
(``numerai_tools.scoring``), which is what the official example notebooks and the
tournament itself use. We never re-implement the correlation ourselves.

Two families of score, both computed *per period* (an era in Classic, a date in
Crypto) and then summarised across periods:

    CORR  -- numerai_corr(prediction, target).
             How well the model predicts the target.

    BMC   -- correlation_contribution(prediction, reference, target).
             How much the model adds *on top of* a reference prediction series.
             This is the same function Numerai uses for MMC; the only difference
             is what you pass as the reference. See load_reference() in
             data_mani/utils.py for which reference each tournament uses.

The summary statistics follow the formulas in the official `hello_numerai`
notebook exactly -- see summarize() for the details and the one gotcha.
"""

import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr, correlation_contribution


# --------------------------------------------------------------------------- #
# Per-period scores
# --------------------------------------------------------------------------- #

def per_period_corr(df, pred_col, target_col, period_col):
    """
    Numerai correlation between predictions and the target, one score per period.

    :param df: frame holding predictions, target and the period column
    :type df: pd.DataFrame
    :param pred_col: name of the prediction column
    :type pred_col: str
    :param target_col: name of the target column
    :type target_col: str
    :param period_col: name of the period column ("era" or "date")
    :type period_col: str
    :return: one correlation per period, indexed by period and sorted
    :rtype: pd.Series
    """
    # Rows with a missing prediction or target cannot be scored.
    sub = df.dropna(subset=[pred_col, target_col])

    # We loop explicitly rather than using groupby().apply(). It is a little
    # slower but much easier to read, and it side-steps the pandas deprecation
    # around operating on the grouping columns.
    scores = {}
    for period, group in sub.groupby(period_col):
        scores[period] = _safe_corr(group, pred_col, target_col)

    return _sorted_series(scores)


def per_period_bmc(df, pred_col, reference_col, target_col, period_col):
    """
    Contribution of the prediction over a reference series, one score per period.

    This is Numerai's `correlation_contribution`. Feed it the benchmark model and
    you get BMC; feed it the meta model and you get MMC. The maths is identical.

    Periods where the reference is missing are skipped entirely -- see the
    coverage note in data_mani.utils.attach_reference(), because neither
    tournament's reference spans its whole history.

    :param df: frame holding predictions, reference, target and period
    :type df: pd.DataFrame
    :param pred_col: name of the prediction column
    :type pred_col: str
    :param reference_col: name of the reference prediction column
    :type reference_col: str
    :param target_col: name of the target column
    :type target_col: str
    :param period_col: name of the period column ("era" or "date")
    :type period_col: str
    :return: one contribution score per period, indexed by period and sorted
    :rtype: pd.Series
    """
    sub = df.dropna(subset=[pred_col, reference_col, target_col])

    scores = {}
    for period, group in sub.groupby(period_col):
        scores[period] = _safe_bmc(group, pred_col, reference_col, target_col)

    return _sorted_series(scores)


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

def summarize(scores):
    """
    Reduce a series of per-period scores to the four numbers Numerai reports.

    The formulas are lifted verbatim from the `hello_numerai` notebook:

        mean          -- long-run performance
        std           -- volatility, using ddof=0
        sharpe        -- mean / std, i.e. consistency
        max_drawdown  -- largest peak-to-trough drop of the cumulative score

    Two conventions worth stating, because the official notebooks are not
    self-consistent about either:

    * ddof=0. `hello_numerai` and `feature_neutralization.metrics()` use ddof=0;
      `target_ensemble` uses the pandas default of ddof=1. We follow ddof=0,
      which also matches Numerai's own scoring helpers.

    * max_drawdown is returned POSITIVE, so a bigger number is always a worse
      drawdown. `hello_numerai` returns it positive, `feature_neutralization`
      negates it. Callers that optimise this must flip the sign themselves --
      see prediction/functions.py:build_scorer.

    :param scores: per-period scores
    :type scores: pd.Series
    :return: {"mean", "std", "sharpe", "max_drawdown"}
    :rtype: dict
    """
    scores = pd.Series(scores).dropna()

    # An empty or single-point series has no meaningful statistics.
    if scores.empty:
        return {"mean": np.nan, "std": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan}

    mean = float(scores.mean())
    std = float(scores.std(ddof=0))

    # A zero standard deviation makes the sharpe undefined rather than infinite.
    sharpe = float(mean / std) if std > 0 else np.nan

    cumsum = scores.cumsum()
    running_max = cumsum.expanding(min_periods=1).max()
    max_drawdown = float((running_max - cumsum).max())

    return {"mean": mean, "std": std,
            "sharpe": sharpe, "max_drawdown": max_drawdown}


def summarize_predictions(df, pred_col, target_col, period_col,
                          reference_col=None, meta_model_col=None):
    """
    Score a prediction frame end to end: per-period CORR, BMC and MMC.

    All three families are reported whenever the data allows, no matter which
    metric drove the hyperparameter search, so two runs optimised for different
    objectives stay directly comparable.

    :param df: frame with predictions, target, period and optionally a reference
    :type df: pd.DataFrame
    :param pred_col: name of the prediction column
    :type pred_col: str
    :param target_col: name of the target column
    :type target_col: str
    :param period_col: name of the period column
    :type period_col: str
    :param reference_col: reference column for BMC, or None to skip BMC
    :type reference_col: str or None
    :param meta_model_col: real meta model column for MMC, or None to skip MMC
    :type meta_model_col: str or None
    :return: corr / bmc / mmc summaries plus the period count behind each
    :rtype: dict
    """
    corr_scores = per_period_corr(df, pred_col, target_col, period_col)
    out = {"corr": summarize(corr_scores),
           "n_periods": int(corr_scores.notna().sum())}

    # BMC and MMC are the same function -- correlation_contribution -- differing
    # only in what they contribute over:
    #
    #   BMC  vs the configured benchmark or the Ridge meta proxy, which covers
    #        the whole history.
    #   MMC  vs the real published meta model, which exists only for periods
    #        that have actually been published. This is the metric that gets
    #        paid, so report it even when coverage is thin -- but always
    #        alongside its period count, because a mean over 16 periods is a
    #        very different claim from a mean over 500.
    for family, column in (("bmc", reference_col), ("mmc", meta_model_col)):
        if column is not None and column in df.columns:
            scores = per_period_bmc(df, pred_col, column, target_col, period_col)
            out[family] = summarize(scores)
            out["{}_n_periods".format(family)] = int(scores.notna().sum())
        else:
            out[family] = None
            out["{}_n_periods".format(family)] = 0

    return out


def period_portfolio_returns(df, pred_col, target_col, period_col):
    """
    Per-period return of a theoretical market-neutral portfolio.

    Construction, one period at a time:

        1. rank the predictions to [0, 1] within the period
        2. centre them:  w = rank - 0.5   ->  [-0.5, +0.5]
        3. normalise:    w = w / sum|w|   ->  dollar-neutral, gross exposure 1
        4. return = sum(w * centred target)

    So the top-ranked names are held long and the bottom-ranked short, in
    proportion to how extreme their rank is, with longs and shorts always
    balancing. Centring the target too means a period where everything rose
    contributes nothing -- only cross-sectional skill pays.

    Units: the target is a *binned* return (five buckets on [0, 1]), not a price
    return, so this is a theoretical PnL in target units. It is monotone in real
    performance and fine for ranking strategies against each other, but it is
    not a dollar figure and should not be read as one.

    :param df: frame with predictions, target and period
    :type df: pd.DataFrame
    :param pred_col: name of the prediction column
    :type pred_col: str
    :param target_col: name of the target column
    :type target_col: str
    :param period_col: name of the period column
    :type period_col: str
    :return: one portfolio return per period, indexed by period and sorted
    :rtype: pd.Series
    """
    sub = df.dropna(subset=[pred_col, target_col])

    returns = {}
    for period, group in sub.groupby(period_col):
        if len(group) < 2:
            continue

        weights = group[pred_col].rank(pct=True) - 0.5
        gross = weights.abs().sum()
        if gross == 0:                 # every prediction identical
            continue
        weights = weights / gross

        centred_target = group[target_col] - group[target_col].mean()
        returns[period] = float((weights * centred_target).sum())

    return _sorted_series(returns)


def per_period_table(df, pred_col, target_col, period_col,
                     reference_col=None, meta_model_col=None):
    """
    Every per-period series a report needs, in one tidy frame.

    Columns: corr, bmc, mmc, pnl -- one row per period. Cumulative curves are
    just .cumsum() on any of these, which is how the official notebooks plot
    them.

    :return: frame indexed by period
    :rtype: pd.DataFrame
    """
    table = pd.DataFrame({
        "corr": per_period_corr(df, pred_col, target_col, period_col),
        "pnl": period_portfolio_returns(df, pred_col, target_col, period_col),
    })

    for name, column in (("bmc", reference_col), ("mmc", meta_model_col)):
        if column is not None and column in df.columns:
            table[name] = per_period_bmc(
                df, pred_col, column, target_col, period_col
            )
        else:
            table[name] = np.nan

    table.index.name = period_col
    return table[["corr", "bmc", "mmc", "pnl"]]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _safe_corr(group, pred_col, target_col):
    """Return numerai_corr for one period, or NaN if it cannot be computed."""
    # A period with fewer than two rows, or with a constant prediction, has no
    # defined correlation. numerai_corr raises rather than returning NaN, so we
    # catch it and let the caller drop the period.
    try:
        return float(numerai_corr(group[[pred_col]], group[target_col]).iloc[0])
    except Exception:
        return np.nan


def _safe_bmc(group, pred_col, reference_col, target_col):
    """Return correlation_contribution for one period, or NaN if undefined."""
    try:
        return float(
            correlation_contribution(
                group[[pred_col]], group[reference_col], group[target_col]
            ).iloc[0]
        )
    except Exception:
        return np.nan


def _sorted_series(scores):
    """
    Build a Series from a {period: score} dict, sorted chronologically.

    Classic eras are zero-padded strings ("0001"), Crypto periods are dates.
    Both sort correctly with a plain sort_index(), but we sort numerically when
    the labels are integer-like so that unpadded eras cannot mis-order.
    """
    series = pd.Series(scores, dtype="float64")
    if series.empty:
        return series

    try:
        order = sorted(series.index, key=lambda x: int(x))
        return series.loc[order]
    except (TypeError, ValueError):
        return series.sort_index()
