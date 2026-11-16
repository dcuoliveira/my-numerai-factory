"""
Mean-reversion / contrarian signals from raw spot OHLCV, drawn from the crypto
asset-pricing and technical-trading literature. PURE functions: they turn an
(symbol, date, open, high, low, close, volume) panel into raw reversion columns
and nothing else. Binning onto Numerai's scale is a separate step
(``src.data_mani.binning``); merging into the tournament panel is a third
(``src.data_mani.build_reversion``) -- the same three-seam split as the momentum /
volatility / trend blocks, so the LIVE path could reuse this computation later.

------------------------------------------------------------------------------
The signals, and where they come from
------------------------------------------------------------------------------

Cross-sectional asset-pricing reversal:
    feature_rev_str_{1,7,14}d   Short-term reversal: the prior cumulative return
                                over 1/7/14 days. Liu, Tsyvinski & Wu (2022, JF)
                                document 1-week reversal; equity origin Jegadeesh
                                (1990), Lehmann (1990).
    feature_rev_max1_28d        MAX(1): the largest single daily return over the
    feature_rev_max5_28d        past month; MAX(5): mean of the 5 largest. The
                                lottery effect -- high MAX -> LOW future return.
                                Li, Urquhart, Wang & Zhang (2021, IRFA).
    feature_rev_ivol_28d        Idiosyncratic volatility: residual std of a single-
                                factor (equal-weight crypto market) model over the
                                past month. High IVOL -> lower returns. Ang,
                                Hodrick, Xing & Zhang (2006); crypto IVOL work.
    feature_rev_skew_28d        Return skewness over the past month; positive skew
                                (lottery) -> lower future return. Harvey & Siddique
                                (2000); crypto skewness-premium work.
    feature_rev_amihud_28d      Amihud (2002) illiquidity: mean |ret| / dollar
                                volume. Reversal is stronger in illiquid coins.

Technical oscillators (crypto "profitability of TA" literature -- Shen, Urquhart &
Wang; Hudson & Urquhart; Gerritsen et al.):
    feature_rev_rsi_14d         Relative Strength Index(14): >70 overbought, <30
                                oversold. 100 - 100/(1+RS), RS = avg gain/avg loss.
    feature_rev_bbz_20d         Bollinger z-score: (close - SMA20)/std20 -- distance
                                from the mean in standard deviations.
    feature_rev_stoch_14d       Stochastic %K(14): (close - min low)/(max high -
                                min low) over 14 days; near 1 overbought.
    feature_rev_cci_20d         Commodity Channel Index(20): (TP - SMA(TP)) /
                                (0.015 * mean-abs-dev(TP)), TP = (H+L+C)/3.

Distance-from-reference (reversion to a level):
    feature_rev_dist_ma_{20,50}d  close / SMA_n - 1: how stretched price is above
                                (or below) its moving average.

------------------------------------------------------------------------------
Sign convention and causality
------------------------------------------------------------------------------

Each column carries the signal in its NATURAL form (e.g. the prior return itself,
not its negation); the cross-sectional rank-bin and the model learn the sign, and
the crypto evidence is that most of these predict REVERSAL (the recent winner /
overbought / high-MAX / high-IVOL name underperforms). This mirrors
``momentum_features``, whose returns are likewise carried un-flipped. Every signal
is a trailing function of a symbol's own data (and, for IVOL, the contemporaneous
cross-section), using only values at or before date t, so it is observable at t and
needs no resolution lag; a full window is required (``min_periods = w``), so an
early row is NaN -> neutral bin at load. The submission-time lag (row D may only use
data <= D-1) is applied later, in ``attach_signals_asof``.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

REVERSION_SPECS = OrderedDict([
    ("feature_rev_str_1d",     {"kind": "str", "window": 1}),
    ("feature_rev_str_7d",     {"kind": "str", "window": 7}),
    ("feature_rev_str_14d",    {"kind": "str", "window": 14}),
    ("feature_rev_max1_28d",   {"kind": "max", "window": 28, "n": 1}),
    ("feature_rev_max5_28d",   {"kind": "max", "window": 28, "n": 5}),
    ("feature_rev_ivol_28d",   {"kind": "ivol", "window": 28}),
    ("feature_rev_skew_28d",   {"kind": "skew", "window": 28}),
    ("feature_rev_amihud_28d", {"kind": "amihud", "window": 28}),
    ("feature_rev_rsi_14d",    {"kind": "rsi", "window": 14}),
    ("feature_rev_bbz_20d",    {"kind": "bbz", "window": 20}),
    ("feature_rev_stoch_14d",  {"kind": "stoch", "window": 14}),
    ("feature_rev_cci_20d",    {"kind": "cci", "window": 20}),
    ("feature_rev_dist_ma_20d",{"kind": "distma", "window": 20}),
    ("feature_rev_dist_ma_50d",{"kind": "distma", "window": 50}),
])


def _top_n_mean(n):
    def reducer(a):
        finite = a[~np.isnan(a)]
        if finite.size < n:
            return np.nan
        return np.sort(finite)[-n:].mean()
    return reducer


def compute_reversion_signals(ohlcv_panel, specs=None):
    """
    Compute the raw (un-binned) reversion signals for every (symbol, date).

    :param ohlcv_panel: long panel with columns symbol, date, open, high, low,
        close, volume. One bar per (symbol, date).
    :type ohlcv_panel: pd.DataFrame
    :param specs: spec map; defaults to REVERSION_SPECS
    :return: frame keyed symbol, date with one float column per spec, NaN wherever
        a symbol lacks a full window behind that date
    :rtype: pd.DataFrame
    """
    specs = specs or REVERSION_SPECS
    need = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if not need.issubset(ohlcv_panel.columns):
        raise KeyError("panel needs %s; got %s" % (sorted(need), list(ohlcv_panel.columns)))

    df = ohlcv_panel[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    for c in ("open", "high", "low"):
        df.loc[~(df[c] > 0), c] = np.nan
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol", sort=False)
    df["_dret"] = g["close"].pct_change()
    # equal-weight crypto market return per date (the single factor for IVOL)
    df["_mkt"] = df.groupby("date")["_dret"].transform("mean")
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3.0

    def roll(col, w, how, mp=None):
        mp = w if mp is None else mp
        r = df.groupby("symbol", sort=False)[col].rolling(w, min_periods=mp)
        return getattr(r, how)().reset_index(level=0, drop=True).sort_index()

    for name, spec in specs.items():
        kind = spec["kind"]
        w = int(spec["window"])
        if kind == "str":
            df[name] = df["close"] / g["close"].shift(w) - 1.0
        elif kind == "max":
            n = int(spec["n"])
            if n == 1:
                df[name] = roll("_dret", w, "max")
            else:
                r = df.groupby("symbol", sort=False)["_dret"].rolling(w, min_periods=w).apply(
                    _top_n_mean(n), raw=True)
                df[name] = r.reset_index(level=0, drop=True).sort_index()
        elif kind == "ivol":
            # residual std of a single-factor (market) model over the window, via
            # the identity var(resid) = var(x)*(1 - corr(x,mkt)^2), all from rolling
            # means so it is vectorised and causal.
            for a, b in [("_dret", "_dret"), ("_mkt", "_mkt"), ("_dret", "_mkt")]:
                df["_p_%s_%s" % (a, b)] = df[a] * df[b]
            mx = roll("_dret", w, "mean"); my = roll("_mkt", w, "mean")
            mxx = roll("_p__dret__dret", w, "mean"); myy = roll("_p__mkt__mkt", w, "mean")
            mxy = roll("_p__dret__mkt", w, "mean")
            vx = (mxx - mx * mx).clip(lower=0); vy = (myy - my * my).clip(lower=0)
            cov = mxy - mx * my
            denom = (vx * vy).replace(0.0, np.nan)
            rho2 = (cov * cov / denom).clip(0.0, 1.0).fillna(0.0)
            df[name] = np.sqrt((vx * (1.0 - rho2)).clip(lower=0))
            df = df.drop(columns=[c for c in df.columns if c.startswith("_p_")])
        elif kind == "skew":
            df[name] = roll("_dret", w, "skew")
        elif kind == "amihud":
            dollar = (df["volume"] * df["close"]).replace(0.0, np.nan)
            df["_illiq"] = df["_dret"].abs() / dollar
            df[name] = roll("_illiq", w, "mean")
            df = df.drop(columns=["_illiq"])
        elif kind == "rsi":
            delta = g["close"].diff()
            df["_gain"] = delta.clip(lower=0.0); df["_loss"] = (-delta).clip(lower=0.0)
            ag = roll("_gain", w, "mean"); al = roll("_loss", w, "mean")
            rs = ag / al.replace(0.0, np.nan)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            rsi = rsi.where(al != 0, 100.0)          # all-gains window -> RSI 100
            rsi = rsi.where(~((al == 0) & (ag == 0)), 50.0)  # flat window -> neutral 50
            df[name] = rsi.where(ag.notna())
            df = df.drop(columns=["_gain", "_loss"])
        elif kind == "bbz":
            sma = roll("close", w, "mean"); sd = roll("close", w, "std").replace(0.0, np.nan)
            df[name] = (df["close"] - sma) / sd
        elif kind == "stoch":
            hi = roll("high", w, "max"); lo = roll("low", w, "min")
            rng = (hi - lo).replace(0.0, np.nan)
            df[name] = (df["close"] - lo) / rng
        elif kind == "cci":
            smatp = roll("_tp", w, "mean")
            mad = df.groupby("symbol", sort=False)["_tp"].rolling(w, min_periods=w).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            ).reset_index(level=0, drop=True).sort_index()
            df[name] = (df["_tp"] - smatp) / (0.015 * mad.replace(0.0, np.nan))
        elif kind == "distma":
            sma = roll("close", w, "mean")
            df[name] = df["close"] / sma - 1.0
        else:
            raise ValueError("unknown reversion kind %r for %s" % (kind, name))

    cols = list(specs.keys())
    return df[["symbol", "date"] + cols]
