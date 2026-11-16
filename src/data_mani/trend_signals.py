"""
Trend-following signals from raw spot prices, in the Carver / Clenow tradition.
Per-symbol, backward-looking, continuous -- they go through the SAME point-in-time
attach + cross-sectional rank-bin as the crypto momentum block
(momentum_features.py), so a value on date D uses only prices <= D-1 and lands on
Numerai's {0,.25,.5,.75,1} scale.

------------------------------------------------------------------------------
The signals
------------------------------------------------------------------------------

    feature_tf_ewmac_<f>_<s>   EWMAC: (EMA_f - EMA_s) divided by price volatility,
                               Carver's exponential moving-average crossover.
                               Pairs (2,8),(4,16),(8,32),(16,64),(32,128).
    feature_tf_breakout_<N>    Carver breakout: where the price sits in its N-day
                               range, (P - mid) / (max - min), EMA-smoothed span N/4.
                               N in 20,40,80,160.
    feature_tf_bbz_<n>         Bollinger z-score: (P - SMA_n) / std_n, the distance
                               from the mean in standard deviations. n in 20,40,80.
    feature_tf_newhigh20in5    the simple rule: how many of the last 5 days were a
                               new 20-day high (0-5) -- "bought a 20d breakout".
    feature_tf_clenow_<N>      Clenow momentum: annualised slope of an exponential
                               (log-price) regression over N days, times its R^2.
                               N in 60,90.

Why these are NOT redundant with the existing momentum block: raw returns and the
robust-trend t-stat rank on cumulative return and on mean/std; EWMAC ranks on an
EMA gap, breakout and Bollinger on the price's position in a range / in vol units
(mean-reversion-flavoured), and Clenow on trend SMOOTHNESS (slope x R^2) -- distinct
orderings. Normalised momentum (return/vol) and plain cross-sectional momentum are
deliberately NOT re-added: the first is the trend block, the second is what the
cross-sectional rank-bin already does to every one of these.

Causality: every signal reads only the trailing window (<= t); a full window is
required, so an early row is NaN -> neutral bin at load.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

# EWMAC (fast, slow) pairs and the vol lookback that normalises them (Carver uses
# an EW vol; span 32 business days here).
EWMAC_PAIRS = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128)]
EWMAC_VOL_SPAN = 32
BREAKOUT_WINDOWS = [20, 40, 80, 160]
BBZ_WINDOWS = [20, 40, 80]
CLENOW_WINDOWS = [60, 90]
CLENOW_ANNUALISATION = 252  # trading-day-style annualisation of the daily log slope

TREND_SPECS = OrderedDict()
for _f, _s in EWMAC_PAIRS:
    TREND_SPECS["feature_tf_ewmac_%d_%d" % (_f, _s)] = {"kind": "ewmac", "fast": _f, "slow": _s}
for _n in BREAKOUT_WINDOWS:
    TREND_SPECS["feature_tf_breakout_%d" % _n] = {"kind": "breakout", "window": _n}
for _n in BBZ_WINDOWS:
    TREND_SPECS["feature_tf_bbz_%d" % _n] = {"kind": "bbz", "window": _n}
TREND_SPECS["feature_tf_newhigh20in5"] = {"kind": "newhigh", "high_window": 20, "recent": 5}
for _n in CLENOW_WINDOWS:
    TREND_SPECS["feature_tf_clenow_%d" % _n] = {"kind": "clenow", "window": _n}


def _groll(df, col, window, how, min_periods=None):
    """A per-symbol rolling reduce, realigned to the frame index."""
    mp = window if min_periods is None else min_periods
    r = df.groupby("symbol", sort=False)[col].rolling(window, min_periods=mp)
    r = getattr(r, how)()
    return r.reset_index(level=0, drop=True).sort_index()


def _clenow(df, N):
    """
    Vectorised rolling exponential-regression momentum: annualised slope x R^2 of
    log(close) on time over the trailing N observations, per symbol.

    Uses rolling sums (no per-window polyfit): with local x = 0..N-1,
        slope = Sxy_c / Sxx_c,   R^2 = Sxy_c^2 / (Sxx_c * Syy_c),
    where the _c terms are the mean-centred sums; Sxx_c is constant in N.
    """
    y = df["_y"]
    Sy = _groll(df, "_y", N, "sum")
    Sgy = _groll(df, "_gy", N, "sum")
    Sy2 = _groll(df, "_y2", N, "sum")
    start = df["_g"] - (N - 1)                 # global index of the window start
    Sx = N * (N - 1) / 2.0                      # sum of local x = 0..N-1
    Sxx = (N - 1) * N * (2 * N - 1) / 6.0       # sum of x^2
    Sxx_c = Sxx - Sx * Sx / N
    Sxy = Sgy - start * Sy                      # sum local_x * y
    Sxy_c = Sxy - Sx * Sy / N
    Syy_c = (Sy2 - Sy * Sy / N).where(lambda s: s > 0)
    slope = Sxy_c / Sxx_c
    r2 = (Sxy_c * Sxy_c) / (Sxx_c * Syy_c)
    annualised = np.expm1(slope * CLENOW_ANNUALISATION)
    return annualised * r2


def compute_trend_signals(close_panel, specs=None, vol_span=EWMAC_VOL_SPAN):
    """
    Compute the raw (un-binned) trend-following signals for every (symbol, date).

    :param close_panel: long panel with columns symbol, date, close
    :param specs: spec map; defaults to TREND_SPECS
    :param vol_span: EW vol span used to normalise EWMAC
    :return: frame keyed symbol, date with one column per spec
    :rtype: pd.DataFrame
    """
    specs = specs or TREND_SPECS
    df = close_panel[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["_g"] = df.groupby("symbol", sort=False).cumcount()
    df["_y"] = np.log(df["close"])
    df["_gy"] = df["_g"] * df["_y"]
    df["_y2"] = df["_y"] * df["_y"]
    df["_dret"] = df.groupby("symbol", sort=False)["close"].pct_change()

    for name, spec in specs.items():
        kind = spec["kind"]
        if kind == "ewmac":
            f, s = int(spec["fast"]), int(spec["slow"])
            ema_f = df.groupby("symbol", sort=False)["close"].transform(
                lambda x: x.ewm(span=f, min_periods=s).mean())
            ema_s = df.groupby("symbol", sort=False)["close"].transform(
                lambda x: x.ewm(span=s, min_periods=s).mean())
            ret_vol = df.groupby("symbol", sort=False)["_dret"].transform(
                lambda x: x.ewm(span=vol_span, min_periods=vol_span).std())
            price_vol = (df["close"] * ret_vol).replace(0.0, np.nan)
            df[name] = (ema_f - ema_s) / price_vol
        elif kind == "breakout":
            N = int(spec["window"])
            hi = _groll(df, "close", N, "max")
            lo = _groll(df, "close", N, "min")
            rng = (hi - lo).replace(0.0, np.nan)
            raw = (df["close"] - 0.5 * (hi + lo)) / rng
            # Carver smooths the breakout with an EMA of span N/4.
            span = max(1, N // 4)
            sm = pd.DataFrame({"symbol": df["symbol"], "raw": raw}).groupby(
                "symbol", sort=False)["raw"].transform(
                lambda x: x.ewm(span=span, min_periods=1).mean())
            df[name] = sm
        elif kind == "bbz":
            n = int(spec["window"])
            sma = _groll(df, "close", n, "mean")
            std = _groll(df, "close", n, "std").replace(0.0, np.nan)
            df[name] = (df["close"] - sma) / std
        elif kind == "newhigh":
            hw, rec = int(spec["high_window"]), int(spec["recent"])
            roll_max = _groll(df, "close", hw, "max")
            is_high = (df["close"] >= roll_max).astype("float64")
            is_high[roll_max.isna()] = np.nan
            df["_ishigh"] = is_high
            df[name] = _groll(df, "_ishigh", rec, "sum")
        elif kind == "clenow":
            df[name] = _clenow(df, int(spec["window"]))
        else:
            raise ValueError("unknown trend kind %r for %s" % (kind, name))

    cols = list(specs.keys())
    return df[["symbol", "date"] + cols]
