"""
Macro / cross-asset features, SHARED across every crypto on a date.

Gold, silver, US Treasury yields (1/2/5/10/30yr), yield-curve slopes and S&P
momentum are properties of the WORLD on a given day, not of a particular coin, so
their value is the same for every symbol in that date's cross-section. That single
fact drives every design choice here and makes them unlike the per-symbol momentum
block (momentum_features.py).

------------------------------------------------------------------------------
Why these are NOT cross-sectionally rank-binned
------------------------------------------------------------------------------

Every shipped Numerai feature is rank-binned WITHIN each date. That is impossible
for a feature that is constant across the cross-section: if gold's 20-day return is
the same number for all 300 coins today, its within-date rank is a tie for
everyone, so cross-sectional binning would map the whole column to the neutral
0.5 -- destroying it.

So a shared feature must be normalised across TIME instead: where does today's
value sit in its OWN recent history? We use a CAUSAL TRAILING PERCENTILE -- the
fraction of the last ~year of observations at or below today's -- which lands in
[0, 1] with 0.5 = the running median, matching the pipeline's neutral-fill
convention, and reads only dates <= today so it never looks ahead. A TRAILING
window (not expanding-since-2010) is deliberate: yields and slopes are near
unit-root, so an expanding rank pegs a multi-year high near 1.0 and freezes; a
trailing window keeps the signal regime-relative (a 1-year high is 1.0, and resets
as the regime moves). The macro store is fetched from 2010 so the window is full
long before the crypto panel starts in 2020.

------------------------------------------------------------------------------
The honest caveat: a shared feature has ~zero standalone within-era IC
------------------------------------------------------------------------------

numerai_corr is a WITHIN-DATE rank correlation. A feature that is constant within a
date contributes nothing to that date's ranking, so on its own it scores ~0 corr
and is inert for a linear model's within-era ranking. Its value is REGIME
CONDITIONING: a tree or net can gate per-symbol signals on the macro state ("in an
inverted-curve, risk-off regime, momentum behaves differently"), and explicit
macro x per-symbol interaction features can make it usable for linear models too.
These are added as honest inputs; expect them to help via interactions, not alone.

------------------------------------------------------------------------------
Point-in-time
------------------------------------------------------------------------------

Same submission rule as the spot features: a round on date D may use only macro
values dated <= D-1 (the last completed business day). Macro series trade on
BUSINESS days while crypto trades 24/7, so the as-of join needs a wider staleness
window (default 7 days) to bridge weekends and holidays -- a Saturday round still
gets Friday's macro rather than a null. The normalisation is causal, the attach is
lagged, so backtest == live by construction.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

# One entry per macro feature.
#   kind "ret"   : window-day return of `series` (gold/silver/sp500), a momentum
#   kind "level" : the raw level of `series` (a yield), a rate-regime signal
#   kind "slope" : `long` yield minus `short` yield, a curve-shape signal
# Every feature is then normalised across time (causal expanding percentile).
MACRO_SPECS = OrderedDict([
    ("feature_macro_gold_ret_20d",   {"kind": "ret", "series": "gold",   "window": 20}),
    ("feature_macro_gold_ret_60d",   {"kind": "ret", "series": "gold",   "window": 60}),
    ("feature_macro_silver_ret_20d", {"kind": "ret", "series": "silver", "window": 20}),
    ("feature_macro_silver_ret_60d", {"kind": "ret", "series": "silver", "window": 60}),
    ("feature_macro_snp_ret_5d",     {"kind": "ret", "series": "sp500",  "window": 5}),
    ("feature_macro_snp_ret_20d",    {"kind": "ret", "series": "sp500",  "window": 20}),
    ("feature_macro_snp_ret_60d",    {"kind": "ret", "series": "sp500",  "window": 60}),
    # US dollar index (DXY) momentum -- a moderate NEGATIVE correlate of crypto
    # (dollar strength = risk-off / tighter global liquidity). Same horizons as
    # the S&P, the other broad daily macro index.
    ("feature_macro_dxy_ret_5d",     {"kind": "ret", "series": "dxy",    "window": 5}),
    ("feature_macro_dxy_ret_20d",    {"kind": "ret", "series": "dxy",    "window": 20}),
    ("feature_macro_dxy_ret_60d",    {"kind": "ret", "series": "dxy",    "window": 60}),
    ("feature_macro_y1",   {"kind": "level", "series": "y1"}),
    ("feature_macro_y2",   {"kind": "level", "series": "y2"}),
    ("feature_macro_y5",   {"kind": "level", "series": "y5"}),
    ("feature_macro_y10",  {"kind": "level", "series": "y10"}),
    ("feature_macro_y30",  {"kind": "level", "series": "y30"}),
    # The curve is ~3 factors (level/slope/curvature); more slopes are collinear
    # noise. 2s10s = canonical slope; 1s10s = a 10y-3m recession-spread PROXY
    # (we have no 3mo tenor); 5s30s = the long-end / term-premium slope, additive.
    # 2s5s / 2s30s are spanned by these, so omitted.
    ("feature_macro_slope_2s10s", {"kind": "slope", "long": "y10", "short": "y2"}),
    ("feature_macro_slope_1s10s", {"kind": "slope", "long": "y10", "short": "y1"}),
    ("feature_macro_slope_5s30s", {"kind": "slope", "long": "y30", "short": "y5"}),
])

# Inflation block: break-even inflation LEVELS (nominal - TIPS real yield, the
# market-implied expected inflation) and the MOMENTUM of those levels (their change
# over N business days). Shared/macro, so same causal-percentile normalisation as
# the rest of the macro block -- there is no short end (< 5yr, no TIPS), and the
# break-even carries a liquidity premium, but this is the standard daily gauge.
# Needs the real yields r5/r10/r30 in the macro store alongside the nominal y*.
INFLATION_SPECS = OrderedDict([
    # levels
    ("feature_macro_bei_5y",   {"kind": "breakeven", "nom": "y5",  "real": "r5"}),
    ("feature_macro_bei_10y",  {"kind": "breakeven", "nom": "y10", "real": "r10"}),
    ("feature_macro_bei_30y",  {"kind": "breakeven", "nom": "y30", "real": "r30"}),
    ("feature_macro_bei_5y5y", {"kind": "forward5y5y"}),
    # momentum of the levels (change over 20 / 60 business days)
    ("feature_macro_bei_5y_chg20",   {"kind": "breakeven_change", "nom": "y5",  "real": "r5",  "window": 20}),
    ("feature_macro_bei_5y_chg60",   {"kind": "breakeven_change", "nom": "y5",  "real": "r5",  "window": 60}),
    ("feature_macro_bei_10y_chg20",  {"kind": "breakeven_change", "nom": "y10", "real": "r10", "window": 20}),
    ("feature_macro_bei_10y_chg60",  {"kind": "breakeven_change", "nom": "y10", "real": "r10", "window": 60}),
    ("feature_macro_bei_30y_chg20",  {"kind": "breakeven_change", "nom": "y30", "real": "r30", "window": 20}),
    ("feature_macro_bei_30y_chg60",  {"kind": "breakeven_change", "nom": "y30", "real": "r30", "window": 60}),
    ("feature_macro_bei_5y5y_chg20", {"kind": "forward5y5y_change", "window": 20}),
    ("feature_macro_bei_5y5y_chg60", {"kind": "forward5y5y_change", "window": 60}),
])

# Trailing window (observations) for the percentile, and the history required
# before it emits. ~1 business year window, warmed up over ~6 months. The panel
# starts in 2020 with a decade of macro behind it, so the window is always full.
DEFAULT_WINDOW = 252
DEFAULT_MIN_PERIODS = 126


def causal_trailing_percentile(series, window=DEFAULT_WINDOW,
                               min_periods=DEFAULT_MIN_PERIODS):
    """
    Map a time series to its causal TRAILING percentile in [0, 1].

    Value at date d = fraction of the last ``window`` observations (ending at d,
    inclusive) that are <= the value at d. Uses only dates <= d, so it never looks
    ahead; 0.5 is the window median. NaN until ``min_periods`` observations exist.
    ``window=None`` falls back to an expanding window.

    :param series: a value series indexed by date (may contain NaN)
    :type series: pd.Series
    :param window: trailing lookback in observations, or None for expanding
    :type window: int or None
    :param min_periods: observations required before a percentile is emitted
    :type min_periods: int
    :return: the percentile series, same index, NaN where undefined
    :rtype: pd.Series
    """
    present = series.dropna()
    win = len(present) if window is None else window
    # percentile rank of the window's last element among the window
    pct = present.rolling(win, min_periods=min_periods).apply(
        lambda a: float((a <= a[-1]).mean()), raw=True)
    return pct.reindex(series.index)


def compute_macro_signals(macro_daily, specs=None, normalize=True,
                          window=DEFAULT_WINDOW, min_periods=DEFAULT_MIN_PERIODS):
    """
    Build the macro feature columns from the raw daily macro store.

    :param macro_daily: frame with a ``date`` column and the raw series columns
        (gold, silver, sp500, y1, y2, y5, y10, y30)
    :type macro_daily: pd.DataFrame
    :param specs: feature spec map; defaults to ``MACRO_SPECS``
    :param normalize: apply the causal trailing percentile (leave False to get the
        raw returns/levels/slopes, e.g. for inspection)
    :type normalize: bool
    :param window: trailing window for the percentile (None = expanding)
    :param min_periods: history for the percentile
    :return: frame keyed ``date`` with one column per spec, one row per macro date
    :rtype: pd.DataFrame
    """
    specs = specs or MACRO_SPECS
    df = macro_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date").set_index("date")

    raw = {}
    for name, spec in specs.items():
        kind = spec["kind"]
        if kind == "ret":
            s = df[spec["series"]].dropna()
            raw[name] = (s / s.shift(int(spec["window"])) - 1.0).reindex(df.index)
        elif kind == "level":
            raw[name] = df[spec["series"]].astype("float64")
        elif kind == "slope":
            raw[name] = (df[spec["long"]] - df[spec["short"]]).astype("float64")
        elif kind == "breakeven":
            # break-even inflation LEVEL = nominal yield - TIPS real yield, the
            # market-implied expected inflation at that maturity.
            raw[name] = (df[spec["nom"]] - df[spec["real"]]).astype("float64")
        elif kind == "forward5y5y":
            # 5y5y forward break-even (the Fed's favourite gauge):
            # (10*BEI10 - 5*BEI5) / 5.
            bei5 = df["y5"] - df["r5"]
            bei10 = df["y10"] - df["r10"]
            raw[name] = ((10.0 * bei10 - 5.0 * bei5) / 5.0).astype("float64")
        elif kind == "breakeven_change":
            # MOMENTUM of the break-even level: its change over `window` business
            # days (in percentage points).
            bei = (df[spec["nom"]] - df[spec["real"]]).dropna()
            raw[name] = (bei - bei.shift(int(spec["window"]))).reindex(df.index)
        elif kind == "forward5y5y_change":
            fwd = ((10.0 * (df["y10"] - df["r10"])
                    - 5.0 * (df["y5"] - df["r5"])) / 5.0).dropna()
            raw[name] = (fwd - fwd.shift(int(spec["window"]))).reindex(df.index)
        else:
            raise ValueError("unknown macro kind %r for %s" % (kind, name))

    out = pd.DataFrame(raw, index=df.index)
    if normalize:
        for c in out.columns:
            out[c] = causal_trailing_percentile(out[c], window=window,
                                                min_periods=min_periods)
    return out.reset_index().rename(columns={"index": "date"})


def attach_macro_asof(panel, macro_feats, cols, lag_days=1, max_staleness_days=7,
                      period_col="date"):
    """
    Broadcast the per-date macro features onto the panel POINT-IN-TIME.

    A macro value is shared across all symbols on a date, so this is a DATE-level
    as-of join (no ``by`` symbol): every panel row on date D receives the freshest
    macro row dated <= D - lag_days, the same value for all symbols that day. The
    wider ``max_staleness_days`` (vs the spot features) bridges the business-day /
    24-7 mismatch so a weekend or holiday round still gets the last business day.

    :param panel: the tournament panel (many rows per date)
    :type panel: pd.DataFrame
    :param macro_feats: per-date macro features (one row per date)
    :type macro_feats: pd.DataFrame
    :param cols: the macro columns to attach
    :type cols: list[str]
    :param lag_days: row D may use macro dates <= D - lag_days
    :param max_staleness_days: attached macro date must be within this of D
    :param period_col: the date column on both frames
    :return: a copy of ``panel`` with ``cols`` attached, rows/order preserved
    :rtype: pd.DataFrame
    """
    if max_staleness_days < lag_days:
        raise ValueError("max_staleness_days (%d) must be >= lag_days (%d)"
                         % (max_staleness_days, lag_days))

    # Reindex the macro features onto a COMPLETE daily calendar and forward-fill
    # each column INDEPENDENTLY, capped at the staleness window. Per-column matters
    # because the series run on different calendars: Treasury follows the SIFMA
    # bond calendar (closed Columbus/Veterans Day) while metals/S&P follow the
    # equity calendar, so a single shared row would wrongly null yields on a day
    # the equity market traded. ffill carries only PAST values forward, so this is
    # causal; the cap turns a genuinely dead feed into neutral rather than a frozen
    # level. With a complete daily index, "limit" in rows == calendar days.
    mf = macro_feats[[period_col] + cols].copy()
    mf[period_col] = pd.to_datetime(mf[period_col])
    mf = mf.dropna(subset=[period_col]).drop_duplicates(period_col).sort_values(period_col)
    full = pd.date_range(mf[period_col].min(), mf[period_col].max(), freq="D")
    mf = mf.set_index(period_col).reindex(full).ffill(limit=max_staleness_days)
    mf.index.name = "_lookup"
    mf = mf.reset_index()

    # Row D looks up the filled value at D - lag_days (exact); the same macro row
    # broadcasts to every symbol on D. merge_asof on the daily index would work
    # too, but the exact lookup on the pre-filled grid is simpler and cheaper.
    out = panel.copy()
    lookup = pd.to_datetime(out[period_col]) - pd.Timedelta(days=lag_days)
    joined = pd.DataFrame({"_lookup": lookup.to_numpy()}).merge(mf, on="_lookup", how="left")
    for c in cols:
        out[c] = joined[c].to_numpy()
    return out
