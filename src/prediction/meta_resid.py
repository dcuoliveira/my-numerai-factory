"""
Meta-model residualisation -- the E106/E107 transform, shared by research and live.

Per period, with p a model's predictions and m the meta model as it stood at
least LAG calendar days earlier:

    gp = gauss(rank(p))           gm = gauss(rank(m))      (missing m -> 0)
    beta = gp.gm / gm.gm
    r    = gp - STRENGTH * beta * gm

and what is submitted (or scored) is rank(r). Removing half the projection onto
the meta model raised payout for every one of the 38 models E107 tried, because
MMC pays for what the meta model does not already say.

The lag is a causality rule, not a tuning knob: the meta model for a date is
published after that date's round closes, so a backtest that used the same-day
meta model would be residualising on information no live round ever has. Five
days is what E106 chose and E107 kept.

A period with no snapshot at or before date - LAG, or whose snapshot covers none
of the period's symbols (gm.gm == 0), passes through unchanged: r = gp. That is
the E107 rule and it is kept here so research and live cannot disagree on it.

experiments/e107_residualise_everything_crypto.py carries its own copy of these
three functions because the study predates this module; test_meta_resid asserts
the two agree, so the copy cannot drift.
"""

import numpy as np
import pandas as pd
from scipy import stats

STRENGTH = 0.5
LAG_DAYS = 5


def gauss_rank(values):
    """gauss(tie-averaged rank(x)) on the non-missing entries, 0 where x is missing."""
    s = pd.Series(np.asarray(values, dtype="float64"))
    u = (s.rank(method="average") - 0.5) / s.count()
    return np.nan_to_num(stats.norm.ppf(u.to_numpy()), nan=0.0)


def residualise(prediction, meta, strength=STRENGTH):
    """
    Residualise one period's predictions on the lagged meta model.

    :param prediction: one period's raw predictions (any scale; only rank matters)
    :param meta: the lagged meta model aligned to the same rows, NaN where a
                 symbol had no meta prediction; None or all-NaN means no snapshot
    :return: (r, beta) where r is gauss-scale and beta is NaN on a pass-through
    """
    gp = gauss_rank(prediction)
    if meta is None:
        return gp, np.nan
    meta = np.asarray(meta, dtype="float64")
    if not np.isfinite(meta).any():
        return gp, np.nan
    gm = gauss_rank(meta)
    denom = float(gm @ gm)
    if denom == 0:
        return gp, np.nan
    beta = float(gp @ gm) / denom
    return gp - strength * beta * gm, beta


def snapshot_for(meta_dates, date, lag=LAG_DAYS):
    """The latest meta date <= date - lag days, or None."""
    dates = np.sort(pd.to_datetime(pd.Series(meta_dates).unique()).to_numpy())
    want = np.datetime64(pd.Timestamp(date) - pd.Timedelta(days=lag))
    pos = np.searchsorted(dates, want, side="right") - 1
    return None if pos < 0 else pd.Timestamp(dates[pos])


def lagged_meta(meta, symbols, date, lag=LAG_DAYS):
    """
    Each symbol's meta prediction on the latest meta date <= date - lag days.

    :param meta: frame with date, symbol, prediction (historical_meta_models)
    :return: (values aligned to `symbols`, NaN where missing; snapshot date or None)
    """
    meta = meta[["date", "symbol", "prediction"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    snapshot = snapshot_for(meta["date"], date, lag)
    if snapshot is None:
        return np.full(len(symbols), np.nan), None
    day = meta[meta["date"] == snapshot].drop_duplicates("symbol", keep="last")
    values = day.set_index("symbol")["prediction"].reindex(list(symbols))
    return values.to_numpy(dtype="float64"), snapshot
