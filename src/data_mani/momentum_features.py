"""
Crypto momentum signals from raw spot prices, drawn from the crypto momentum
literature. PURE functions: they turn a (symbol, date, close) panel into raw
momentum columns and nothing else. Binning onto Numerai's scale is a separate
step (``src.data_mani.binning``); merging into the tournament panel is a third
(``src.data_mani.build_momentum``). Keeping the three apart is what lets the
LIVE path reuse this exact computation on ``live_history`` later without dragging
in any offline file layout.

------------------------------------------------------------------------------
What each signal is, and where it comes from
------------------------------------------------------------------------------

Every signal is a trailing function of a symbol's own close series -- it uses
only prices at or before date t, so it is observable at t and needs no
resolution lag (contrast the forward-return targets, which do). The horizons are
the ones the crypto papers actually test; the crypto literature lives at
DAYS-TO-WEEKS, not the 12-1-month equity horizon, so most windows here are short.

    feature_mom_ret_7d     1-week cumulative return    Liu & Tsyvinski (2021, RFS);
                                                       Tzouvanas et al. (2019, EL, J=7)
    feature_mom_ret_14d    14-day cumulative return    Begusic & Kostanjcar (2019, arXiv);
                                                       Tzouvanas et al. (2019, J=15)
    feature_mom_ret_21d    3-week cumulative return    Liu, Tsyvinski & Wu (2022, JF) -- the
                                                       CMOM factor's formation signal
    feature_mom_ret_28d    4-week cumulative return    Liu & Tsyvinski (2021); Tzouvanas (J=30)
    feature_mom_ret_56d    8-week cumulative return    Liu, Tsyvinski & Wu (2022)
    feature_mom_ret_112d   16-week cumulative return   Liu, Tsyvinski & Wu (2022)
    feature_mom_ret_12_1   12-1: return t-360..t-30    Grobys & Sapkota (2019, EL), in days
    feature_mom_max1_28d   MAX(1): largest 1-day       Li, Urquhart, Wang & Zhang (2021, IRFA)
                           return over the past month
    feature_mom_max5_28d   MAX(5): mean of the 5       Li, Urquhart, Wang & Zhang (2021)
                           largest daily returns / month
    feature_mom_volscaled_28d  4-week return / its     Tzouvanas et al. (2019, Adj.MOM);
                           realised daily-return vol   Barroso & Santa-Clara (2015, JFE)
    feature_mom_trend_{7,14,   robust trend signal      Tzotchev (2018) robust
      30,60,120,240}d          2*N(sqrt(T)*mean/std)-1  trend-following
                               per horizon T (days)
    feature_mom_trend_robust   the sum of the six       Tzotchev (2018) -- the paper's
                               per-horizon signals      normalised composite

Two design choices worth stating:

* **Continuous, not sign.** The papers' time-series-momentum SIGN strategy
  (``sign(past return)``) collapses to two values and cannot be rank-binned into
  five. We instead carry the underlying cumulative return, whose cross-sectional
  RANK preserves the same winner/loser ordering while filling all five bins.

* **Windows are in observations, not calendar days.** Crypto trades daily, so a
  symbol's consecutive observations are ~1 calendar day apart and a positional
  window of w rows ~= w calendar days -- the same convention build_rolling_features
  uses. Where a token has gaps (delisted then relisted) the window spans the gap,
  which slightly ages that symbol's signal; this is the documented, minor cost of
  a positional window and it never introduces look-ahead.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

``ret`` at t uses close(t-skip) and close(t-skip-w), both <= t. ``max`` and
``volscaled`` use the daily returns over (t-w, t], all <= t. A full window is
required (``min_periods = w``), so a symbol with fewer than w observations behind
date t yields NaN there -- which becomes the neutral middle bin at load, exactly
as the shipped 60-day indicators are NaN for tokens younger than 60 days. No
signal ever reads a future price.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.special import ndtr  # standard normal CDF, vectorised

# One entry per momentum feature. `kind`:
#   "ret"        cumulative return close(t-skip)/close(t-skip-window) - 1
#   "max"        mean of the `n` largest daily returns over the trailing window
#   "volscaled"  window return divided by realised daily-return vol (ddof=0)
# `paper` is the literature attribution, carried so a reviewer can trace a column
# back to the study that motivated it.
MOMENTUM_SPECS = OrderedDict([
    ("feature_mom_ret_7d",
     {"kind": "ret", "window": 7, "skip": 0,
      "paper": "Liu & Tsyvinski 2021 (1wk TS-mom); Tzouvanas 2019 (J=7)"}),
    ("feature_mom_ret_14d",
     {"kind": "ret", "window": 14, "skip": 0,
      "paper": "Begusic & Kostanjcar 2019 (14d UMD); Tzouvanas 2019 (J=15)"}),
    ("feature_mom_ret_21d",
     {"kind": "ret", "window": 21, "skip": 0,
      "paper": "Liu, Tsyvinski & Wu 2022 (3wk, CMOM signal)"}),
    ("feature_mom_ret_28d",
     {"kind": "ret", "window": 28, "skip": 0,
      "paper": "Liu & Tsyvinski 2021 (4wk); Tzouvanas 2019 (J=30)"}),
    ("feature_mom_ret_56d",
     {"kind": "ret", "window": 56, "skip": 0,
      "paper": "Liu, Tsyvinski & Wu 2022 (8wk)"}),
    ("feature_mom_ret_112d",
     {"kind": "ret", "window": 112, "skip": 0,
      "paper": "Liu, Tsyvinski & Wu 2022 (16wk)"}),
    ("feature_mom_ret_12_1",
     {"kind": "ret", "window": 330, "skip": 30,
      "paper": "Grobys & Sapkota 2019 (12-1-1, in days)"}),
    ("feature_mom_max1_28d",
     {"kind": "max", "window": 28, "n": 1,
      "paper": "Li, Urquhart, Wang & Zhang 2021 (MAX(1))"}),
    ("feature_mom_max5_28d",
     {"kind": "max", "window": 28, "n": 5,
      "paper": "Li, Urquhart, Wang & Zhang 2021 (MAX(5))"}),
    ("feature_mom_volscaled_28d",
     {"kind": "volscaled", "window": 28,
      "paper": "Tzouvanas 2019 (Adj.MOM); Barroso & Santa-Clara 2015"}),
])

# Tzotchev (2018) "robust trend-following" signal, at crypto-relevant horizons.
# For a lookback T, S_{t,T} = 2*N( sqrt(T) * mean / std of the daily returns over
# T ) - 1 -- a bounded, vol-adjusted trend t-statistic (N = standard normal CDF).
# The paper's convention is to SUM these across horizons and normalise; here the
# per-horizon signals and the composite sum are both carried, and the final
# cross-sectional rank-bin does the normalising. The equity paper uses
# 32/64/126/252/504 trading days; crypto trend lives shorter and trades every
# day, so we anchor at a week and roughly double to ~8 months, capped below the
# 12-month coverage cliff (feature_mom_ret_12_1 is mostly empty).
#
# NOTE (rank-binned pipeline): after per-date rank-binning, a SINGLE-horizon
# robust signal ranks identically to a vol-adjusted return at that horizon -- the
# CDF squashing is monotone and vanishes under ranking. The composite is where it
# earns its keep: bounding each horizon to [-1,1] before summing stops one
# horizon's extreme t-stat from dominating the multi-horizon aggregate.
ROBUST_TREND_WINDOWS = (7, 14, 30, 60, 120, 240)
for _T in ROBUST_TREND_WINDOWS:
    MOMENTUM_SPECS["feature_mom_trend_%dd" % _T] = {
        "kind": "robust_trend", "window": _T,
        "paper": "Tzotchev 2018 robust trend (%dd)" % _T}
MOMENTUM_SPECS["feature_mom_trend_robust"] = {
    "kind": "robust_trend_composite", "windows": list(ROBUST_TREND_WINDOWS),
    "paper": "Tzotchev 2018 composite (sum over %s d)"
             % "/".join(str(w) for w in ROBUST_TREND_WINDOWS)}


def _top_n_mean(n):
    """Return a raw-array reducer: mean of the n largest finite values, else NaN."""
    def reducer(a):
        finite = a[~np.isnan(a)]
        if finite.size < n:
            return np.nan
        return np.sort(finite)[-n:].mean()
    return reducer


def _robust_trend(gdret, T):
    """
    Tzotchev (2018) robust trend signal over a trailing window of T daily
    returns, aligned back to the frame's index.

    S = 2 * N( sqrt(T) * mean / std ) - 1, where mean and std are of the daily
    returns over the trailing T observations and N is the standard normal CDF.
    The argument is the t-statistic of the mean return (mean / (std/sqrt(T))), so
    a strong, low-noise up-trend saturates toward +1 and a down-trend toward -1.
    A full window is required (min_periods=T) and std==0 yields NaN, so no signal
    reads a future return.

    :param gdret: the per-symbol groupby of the daily-return column
    :param T: lookback in observations
    :return: the signal S as a Series aligned to the ungrouped frame index
    :rtype: pd.Series
    """
    mean = gdret.rolling(T, min_periods=T).mean().reset_index(level=0, drop=True).sort_index()
    std = gdret.rolling(T, min_periods=T).std(ddof=1).reset_index(level=0, drop=True).sort_index()
    tstat = np.sqrt(T) * mean / std.replace(0.0, np.nan)
    return pd.Series(2.0 * ndtr(tstat.to_numpy()) - 1.0, index=tstat.index)


def compute_momentum_signals(close_panel, specs=None, min_symbol_rows=1):
    """
    Compute the raw (un-binned) momentum signals for every (symbol, date).

    :param close_panel: long panel with columns ``symbol``, ``date``, ``close``.
        Multiple rows per symbol; one close per (symbol, date).
    :type close_panel: pd.DataFrame
    :param specs: feature spec map; defaults to ``MOMENTUM_SPECS``
    :type specs: dict or None
    :param min_symbol_rows: symbols with fewer rows than this are skipped whole
        (they cannot support any window); 1 keeps everyone
    :type min_symbol_rows: int
    :return: frame keyed ``symbol``, ``date`` with one float column per spec,
        NaN wherever a symbol lacks a full window behind that date
    :rtype: pd.DataFrame
    """
    specs = specs or MOMENTUM_SPECS
    required = {"symbol", "date", "close"}
    if not required.issubset(close_panel.columns):
        raise KeyError("close_panel needs columns %s; got %s"
                       % (sorted(required), list(close_panel.columns)))

    df = close_panel[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    if min_symbol_rows > 1:
        counts = df.groupby("symbol")["close"].transform("size")
        df = df[counts >= min_symbol_rows].reset_index(drop=True)

    g = df.groupby("symbol", sort=False)
    logclose = np.log(df["close"])
    df["_logclose"] = logclose
    df["_dret"] = g["close"].pct_change()
    glog = df.groupby("symbol", sort=False)["_logclose"]
    gdret = df.groupby("symbol", sort=False)["_dret"]

    for name, spec in specs.items():
        kind = spec["kind"]
        w = int(spec["window"]) if "window" in spec else None  # composite has none
        if kind == "ret":
            skip = int(spec.get("skip", 0))
            # close(t-skip)/close(t-skip-w) - 1, per symbol, no future price.
            near = glog.shift(skip)
            far = glog.shift(skip + w)
            df[name] = np.expm1(near - far)
        elif kind == "max":
            n = int(spec["n"])
            if n == 1:
                rolled = gdret.rolling(w, min_periods=w).max()
            else:
                rolled = gdret.rolling(w, min_periods=w).apply(
                    _top_n_mean(n), raw=True)
            df[name] = rolled.reset_index(level=0, drop=True).sort_index()
        elif kind == "volscaled":
            r = np.expm1(df["_logclose"] - glog.shift(w))
            vol = gdret.rolling(w, min_periods=w).std(ddof=0)
            vol = vol.reset_index(level=0, drop=True).sort_index()
            df[name] = r / vol.replace(0.0, np.nan)
        elif kind == "robust_trend":
            df[name] = _robust_trend(gdret, w)
        elif kind == "robust_trend_composite":
            # Sum the per-horizon robust signals (the paper's convention). Strict:
            # NaN if ANY horizon lacks a full window, so a coin needs the longest
            # window of history for the composite -> neutral bin otherwise.
            acc = None
            for horizon in spec["windows"]:
                s = _robust_trend(gdret, int(horizon))
                acc = s if acc is None else acc + s
            df[name] = acc
        else:
            raise ValueError("unknown momentum kind %r for %s" % (kind, name))

    cols = list(specs.keys())
    return df[["symbol", "date"] + cols]


def attach_signals_asof(panel, signals, cols, lag_days=1, max_staleness_days=4,
                        period_col="date", symbol_col="symbol"):
    """
    Merge externally-sourced signals onto the tournament panel POINT-IN-TIME, so
    a feature on round date D uses only prices that are already known when we
    must submit round D.

    Why this is not an exact same-date join. Our momentum comes from an external
    daily spot feed (Yahoo/Binance), whose bar for calendar day D does not close
    until the end of day D -- but a Numerai Crypto round opens and is submitted
    DURING day D. So the newest spot close we can legitimately use for round D is
    the last COMPLETED daily bar, date D - lag_days (lag_days=1). Attaching
    close(D) to row D would be look-ahead: it reads a price that is not final at
    submission time. The offline builder therefore reproduces exactly what the
    live path can see, so the backtest is not optimistic about information it
    would not actually have.

    Freshness, not just safety. Within that constraint we take the MOST RECENT
    usable signal (an as-of backward join), so a Monday round uses the weekend's
    prices rather than Friday's -- crypto trades every day and the weekend move is
    real, known information by Monday. ``max_staleness_days`` caps how old that
    signal may be, so a delisted token does not carry a months-stale value
    forward onto a recent date; beyond it the row is left NaN (neutral bin at
    load).

    :param panel: the tournament panel, unique on (symbol, date)
    :type panel: pd.DataFrame
    :param signals: per (symbol, date) signals, where a row dated d was
        computed from prices at or before d
    :type signals: pd.DataFrame
    :param cols: the signal columns to attach
    :type cols: list[str]
    :param lag_days: submission lag -- row D may only use signal dates <= D - lag
    :type lag_days: int
    :param max_staleness_days: the attached signal date must be within this many
        days of D (i.e. D - date <= max_staleness_days); else NaN
    :type max_staleness_days: int
    :param period_col: the date column on both frames
    :param symbol_col: the symbol column on both frames
    :return: a copy of ``panel`` with ``cols`` attached, original rows/order kept
    :rtype: pd.DataFrame
    """
    if max_staleness_days < lag_days:
        raise ValueError("max_staleness_days (%d) must be >= lag_days (%d)"
                         % (max_staleness_days, lag_days))

    left = panel.copy()
    left["_row"] = np.arange(len(left))
    # The cutoff is the newest spot date this row is allowed to see.
    left["_cutoff"] = pd.to_datetime(left[period_col]) - pd.Timedelta(days=lag_days)

    right = signals[[symbol_col, period_col] + cols].copy()
    right = right.rename(columns={period_col: "_sig_date"})
    right["_sig_date"] = pd.to_datetime(right["_sig_date"])
    right = right.dropna(subset=["_sig_date"]).sort_values("_sig_date")

    left_sorted = left.sort_values("_cutoff")
    # tolerance is on (cutoff - sig_date); cutoff = D - lag, so
    # cutoff - sig_date <= max_staleness - lag  <=>  D - sig_date <= max_staleness.
    tol = pd.Timedelta(days=max_staleness_days - lag_days)
    merged = pd.merge_asof(
        left_sorted, right,
        left_on="_cutoff", right_on="_sig_date", by=symbol_col,
        direction="backward", allow_exact_matches=True,
        tolerance=(tol if tol > pd.Timedelta(0) else None),
    )
    if tol == pd.Timedelta(0):
        # exact cutoff only: keep matches where sig_date == cutoff
        stale = merged["_sig_date"] != merged["_cutoff"]
        merged.loc[stale, cols] = np.nan

    merged = merged.sort_values("_row")
    merged = merged.drop(columns=["_row", "_cutoff", "_sig_date"])
    merged.index = panel.index
    return merged
