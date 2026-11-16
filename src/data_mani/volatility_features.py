"""
Volatility estimators from raw spot OHLCV, drawn from the realised-volatility
literature. PURE functions: they turn an (symbol, date, open, high, low, close)
panel into raw volatility columns and nothing else. Binning onto Numerai's scale
is a separate step (``src.data_mani.binning``); merging into the tournament panel
is a third (``src.data_mani.build_vol``). Keeping the three apart is what lets the
LIVE path reuse this exact computation on a live OHLCV store later without
dragging in any offline file layout -- the same discipline as
``momentum_features``/``trend_signals``.

This is the FIRST feature module that reads OHLC rather than close alone: two of
the four estimators are range-based and need the daily high/low (Parkinson) or the
full open/high/low/close (Yang-Zhang). The other two need only the close.

------------------------------------------------------------------------------
The four estimators, and their hyperparameter
------------------------------------------------------------------------------

Each is a trailing variance estimate; the hyperparameter is the LOOKBACK, and for
EWMA the decay. All values are per-period VARIANCE (not annualised, not square-
rooted): the cross-sectional rank-bin that follows is invariant to any monotone
increasing transform within a date, so variance and volatility bin identically --
we carry variance because it is what each estimator natively produces.

    feature_vol_sqret_<w>d   Squared-returns realised variance: the canonical
                             unbiased baseline. Mean of squared daily log returns
                             over a trailing window of w days,
                             RV_t = (1/w) sum_{i} r_i^2,  r = dlog(close).
                             Andersen & Bollerslev (1998); the zero-mean estimator.

    feature_vol_ewma_lam<L>  EWMA (RiskMetrics) variance: the standard smoothed
                             estimate. sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r_t^2,
                             an exponentially weighted mean of squared returns.
                             lam in {0.90, 0.94, 0.97, 0.99}; 0.94 is the
                             RiskMetrics daily decay (J.P. Morgan 1996). Named by
                             round(100*lam): lam94 == lambda 0.94.

    feature_vol_park_<w>d    Parkinson (1980): range-based, uses only high/low.
                             (1/(4 ln 2)) * mean over w of (ln(H/L))^2. ~5x more
                             efficient than close-to-close when the range is
                             informative, but biased low under jumps/overnight gaps
                             (it assumes zero-drift GBM with a continuously observed
                             path).

    feature_vol_yz_<w>d      Yang-Zhang (2000): the most robust classical range
                             estimator -- it is the ONLY one that handles both drift
                             and opening gaps. Over a window of n days,
                             V = V_overnight + k*V_openclose + (1-k)*V_rogers-satchell,
                             k = 0.34 / (1.34 + (n+1)/(n-1)). Needs the full OHLC and
                             the PREVIOUS close (for the overnight term).

Two horizons, 20d and 60d, match the shipped ``feature_volatility_{20,60}d`` so a
model can see the same lookbacks it already has, at a different (range-based)
estimator; the shorter and longer windows bracket ~1 week to ~6 months.

------------------------------------------------------------------------------
A note on crypto and the overnight term
------------------------------------------------------------------------------

Crypto trades 24/7, so there is no exchange close: the open of day i is essentially
the close of day i-1, the overnight return ln(O_i / C_{i-1}) ~ 0, and Yang-Zhang's
overnight-variance component is small. That is not a bug -- YZ stays a valid, well-
behaved estimator, it simply collapses toward Rogers-Satchell + open-to-close when
there is no real gap, which is the correct behaviour for a continuously traded
asset. It is kept as a distinct feature because the open/close/high/low still carry
intraday-path information the close-to-close estimators throw away.

------------------------------------------------------------------------------
Causality
------------------------------------------------------------------------------

Every per-day term (r_t, the overnight/open-close/Rogers-Satchell terms, the H/L
range) reads only prices ON day t and, for the overnight term, C_{t-1} -- all <= t.
Each windowed estimate requires a FULL window (``min_periods = w``), so a symbol
with fewer than w observations behind date t yields NaN there, which becomes the
neutral middle bin at load, exactly as the shipped 60-day indicators are NaN for
young tokens. No estimator ever reads a future price. The submission-time lag
(row D may only use closes <= D-1) is applied later, in ``attach_signals_asof``.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

# The windows shared by the three windowed estimators (squared returns, Parkinson,
# Yang-Zhang), in observations ~= calendar days (crypto trades daily). 20 and 60
# match the shipped feature_volatility_{20,60}d lookbacks.
VOL_WINDOWS = (5, 10, 20, 40, 60, 120)

# RiskMetrics-style EWMA decays. 0.94 is J.P. Morgan's daily lambda; 0.97 ~ its
# monthly one; 0.90 and 0.99 bracket a fast and a slow tail.
EWMA_LAMBDAS = (0.90, 0.94, 0.97, 0.99)

_FOUR_LN2 = 4.0 * np.log(2.0)

VOL_SPECS = OrderedDict()
for _w in VOL_WINDOWS:
    VOL_SPECS["feature_vol_sqret_%dd" % _w] = {"kind": "sqret", "window": _w}
for _lam in EWMA_LAMBDAS:
    # min_periods ~ the EWMA's effective sample size 1/(1-lam) so an early estimate
    # is not read off one or two returns: lam 0.90/0.94/0.97/0.99 -> 10/17/33/100.
    VOL_SPECS["feature_vol_ewma_lam%d" % round(100 * _lam)] = {
        "kind": "ewma", "lam": _lam,
        "min_periods": int(round(1.0 / (1.0 - _lam)))}
for _w in VOL_WINDOWS:
    VOL_SPECS["feature_vol_park_%dd" % _w] = {"kind": "parkinson", "window": _w}
for _w in VOL_WINDOWS:
    VOL_SPECS["feature_vol_yz_%dd" % _w] = {"kind": "yangzhang", "window": _w}


def _yz_k(n):
    """Yang-Zhang's weight on the open-to-close variance, k = 0.34/(1.34+(n+1)/(n-1))."""
    if n < 2:
        raise ValueError("Yang-Zhang needs a window of at least 2, got %d" % n)
    return 0.34 / (1.34 + (n + 1.0) / (n - 1.0))


def compute_volatility_signals(ohlc_panel, specs=None):
    """
    Compute the raw (un-binned) volatility estimates for every (symbol, date).

    :param ohlc_panel: long panel with columns ``symbol``, ``date`` and the OHLC
        columns ``open``, ``high``, ``low``, ``close``. Multiple rows per symbol;
        one bar per (symbol, date). Only ``close`` is required for the two close-
        based estimators; the range estimators additionally need open/high/low.
    :type ohlc_panel: pd.DataFrame
    :param specs: feature spec map; defaults to ``VOL_SPECS``
    :type specs: dict or None
    :return: frame keyed ``symbol``, ``date`` with one float column per spec,
        NaN wherever a symbol lacks a full window (or valid OHLC) behind that date
    :rtype: pd.DataFrame
    """
    specs = specs or VOL_SPECS
    required = {"symbol", "date", "open", "high", "low", "close"}
    if not required.issubset(ohlc_panel.columns):
        raise KeyError("ohlc_panel needs columns %s; got %s"
                       % (sorted(required), list(ohlc_panel.columns)))

    df = ohlc_panel[["symbol", "date", "open", "high", "low", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    # A valid bar needs a positive close; the return- and range-based terms below
    # additionally require positive O/H/L, so map any non-positive OHLC to NaN and
    # let min_periods propagate it to a neutral bin rather than a bogus log.
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    for col in ("open", "high", "low"):
        df.loc[~(df[col] > 0), col] = np.nan
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    gb = df.groupby("symbol", sort=False)

    # ---- per-day building blocks (all <= day t, causal) -------------------- #
    logc = np.log(df["close"])
    df["_r"] = logc.groupby(df["symbol"]).diff()             # daily log return
    df["_r2"] = df["_r"] * df["_r"]                          # squared log return
    prev_close = gb["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["_o"] = np.log(df["open"] / prev_close)           # overnight ln(O_t/C_{t-1})
        df["_cc"] = np.log(df["close"] / df["open"])         # open->close ln(C/O)
        u = np.log(df["high"] / df["open"])                  # ln(H/O)
        dn = np.log(df["low"] / df["open"])                  # ln(L/O)
        df["_rs"] = u * (u - df["_cc"]) + dn * (dn - df["_cc"])   # Rogers-Satchell
        df["_park"] = (np.log(df["high"] / df["low"]) ** 2) / _FOUR_LN2  # Parkinson/day
    gb = df.groupby("symbol", sort=False)  # regroup so the new columns are visible

    for name, spec in specs.items():
        kind = spec["kind"]
        if kind == "sqret":
            w = int(spec["window"])
            df[name] = gb["_r2"].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean())
        elif kind == "ewma":
            alpha = 1.0 - float(spec["lam"])
            mp = int(spec["min_periods"])
            df[name] = gb["_r2"].transform(
                lambda s, a=alpha, m=mp:
                    s.ewm(alpha=a, adjust=False, min_periods=m).mean())
        elif kind == "parkinson":
            w = int(spec["window"])
            df[name] = gb["_park"].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean())
        elif kind == "yangzhang":
            n = int(spec["window"])
            k = _yz_k(n)
            v_o = gb["_o"].transform(
                lambda s, n=n: s.rolling(n, min_periods=n).var(ddof=1))
            v_c = gb["_cc"].transform(
                lambda s, n=n: s.rolling(n, min_periods=n).var(ddof=1))
            v_rs = gb["_rs"].transform(
                lambda s, n=n: s.rolling(n, min_periods=n).mean())
            df[name] = v_o + k * v_c + (1.0 - k) * v_rs
        else:
            raise ValueError("unknown volatility kind %r for %s" % (kind, name))

    cols = list(specs.keys())
    return df[["symbol", "date"] + cols]
