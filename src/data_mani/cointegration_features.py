"""
Cointegration / statistical-arbitrage residual signals: regress each coin's
LOG-PRICE on the log-levels of a few factors over a rolling window, and read the
RESIDUAL as a relative-value mean-reversion signal (coin rich/cheap vs the fair
value the factors imply). PURE functions -- binning and the panel merge are the
usual separate seams (build_cointegration).

Factors (log-levels): BTC, an equal-weight crypto MARKET log-price index (cumulated
mean daily log-return), and macro GOLD / DXY / S&P 500. Altcoins are most plausibly
cointegrated with BTC and the market; the macro legs are a weaker relative-value bet.

------------------------------------------------------------------------------
The signals
------------------------------------------------------------------------------

    feature_coint_resid_z_60d    residual / rolling-std(residual): the stat-arb
    feature_coint_resid_z_120d   z-score from a 60d / 120d rolling OLS. High z =
                                 coin rich vs fair value -> expected reversion DOWN.
                                 (Avellaneda & Lee 2010; Gatev et al. 2006.)
    feature_coint_halflife_120d  mean-reversion half-life of the 120d residual, from
                                 its AR(1): -ln2/ln(phi), phi=1+rho. Reversion speed.
    feature_coint_adf_120d       Dickey-Fuller t-stat of the 120d residual (t-stat of
                                 rho in d(resid)=a+rho*resid_{t-1}+e). More negative =
                                 more stationary / more strongly cointegrated.

------------------------------------------------------------------------------
Why the stationarity stat matters (the spurious-regression trap)
------------------------------------------------------------------------------

Log-prices are I(1). Regressing one I(1) series on others is SPURIOUS unless they
are cointegrated -- and if they are not, the residual is itself a random walk with
no mean reversion, so the z-score is noise. ``feature_coint_adf_120d`` (and the
half-life) are exactly the diagnostics that say whether the residual is stationary;
carried alongside the z-score so a model can down-weight non-cointegrated names.

Causality: the rolling OLS betas use only the trailing window (<= t), the residual
at t uses the factors at t (<= t), and the AR(1) stats use only trailing residuals.
A full window is required, so early rows are NaN -> neutral bin at load. The
submission lag (row D uses <= D-1) is applied later in ``attach_signals_asof``.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

REGRESSORS = ["log_btc", "log_mkt", "log_gold", "log_dxy", "log_snp"]
OLS_WINDOWS = (60, 120)       # residual z-score windows
AR1_OLS_WINDOW = 120          # which residual the half-life / ADF describe
AR1_WINDOW = 60               # trailing residuals used to estimate the AR(1)

COINT_SPECS = OrderedDict([
    ("feature_coint_resid_z_60d",   {"kind": "z", "window": 60}),
    ("feature_coint_resid_z_120d",  {"kind": "z", "window": 120}),
    ("feature_coint_halflife_120d", {"kind": "halflife", "window": AR1_OLS_WINDOW}),
    ("feature_coint_adf_120d",      {"kind": "adf", "window": AR1_OLS_WINDOW}),
])


def _ar1_stats(resid, w=AR1_WINDOW):
    """Rolling AR(1) of a residual series -> (half_life, df_tstat) arrays.

    DF regression d(r)_t = a + rho * r_{t-1} + e over a trailing window of w; via
    rolling cross-moments (vectorised, causal). half-life = -ln2/ln(1+rho) when the
    series mean-reverts (0 < 1+rho < 1); df_tstat = rho / se(rho).
    """
    r = pd.Series(np.asarray(resid, dtype="float64"))
    a = r.shift(1)                       # r_{t-1}
    b = r - a                            # d(r)_t
    def rm(s): return s.rolling(w, min_periods=w).mean()
    ma, mb = rm(a), rm(b)
    var_a = rm(a * a) - ma * ma
    cov = rm(a * b) - ma * mb
    var_b = rm(b * b) - mb * mb
    var_a = var_a.where(var_a > 1e-18)
    rho = cov / var_a
    s2 = (var_b - rho * cov) * (w / (w - 2.0))
    se = np.sqrt((s2 / (w * var_a)).clip(lower=0))
    df_t = rho / se.where(se > 1e-18)
    phi = 1.0 + rho
    hl = pd.Series(np.where((phi > 0) & (phi < 1), -np.log(2.0) / np.log(phi.where((phi > 0) & (phi < 1))), np.nan))
    return hl.to_numpy(), df_t.to_numpy()


def build_regressors(spot, macro_daily, max_staleness_days=7):
    """Date-indexed log-level factor frame: BTC, equal-weight market, gold/DXY/S&P."""
    s = spot[["symbol", "date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["close"] > 0].dropna(subset=["close"])
    s = s.sort_values(["symbol", "date"])
    s["_logc"] = np.log(s["close"])
    dates = pd.Index(np.sort(s["date"].unique()))

    log_btc = (s[s["symbol"] == "BTC"].drop_duplicates("date").set_index("date")["_logc"]
               .reindex(dates).ffill(limit=max_staleness_days))
    dl = s.assign(_dl=s.groupby("symbol")["_logc"].diff())
    mkt_daily = dl.groupby("date")["_dl"].mean().reindex(dates).fillna(0.0)
    log_mkt = mkt_daily.cumsum()

    m = macro_daily.copy(); m["date"] = pd.to_datetime(m["date"])
    m = m.drop_duplicates("date").set_index("date").sort_index()
    macro = pd.DataFrame(index=dates)
    macro["log_gold"] = np.log(m["gold"]).reindex(dates).ffill(limit=max_staleness_days)
    macro["log_dxy"] = np.log(m["dxy"]).reindex(dates).ffill(limit=max_staleness_days)
    macro["log_snp"] = np.log(m["sp500"]).reindex(dates).ffill(limit=max_staleness_days)

    R = pd.DataFrame({"log_btc": log_btc, "log_mkt": log_mkt}).join(macro)
    R.index.name = "date"
    return R.reset_index()


def _per_symbol(sub):
    sub = sub.sort_values("date")
    y = sub["_logc"].to_numpy(dtype="float64")
    Xc = sm.add_constant(sub[REGRESSORS].to_numpy(dtype="float64"), has_constant="add")
    out = {}
    resid_primary = None
    n = len(sub)
    for w in OLS_WINDOWS:
        if n >= w and np.isfinite(Xc).all():
            try:
                beta = RollingOLS(y, Xc, window=w, min_nobs=w).fit(params_only=True).params
            except Exception:
                beta = np.full((n, Xc.shape[1]), np.nan)
            fitted = (Xc * beta).sum(axis=1)
            resid = y - fitted
        else:
            resid = np.full(n, np.nan)
        sd = pd.Series(resid).rolling(w, min_periods=w).std()
        sd = sd.where(sd > 1e-18)                       # guard flat-residual windows
        z = resid / sd.to_numpy()
        out["feature_coint_resid_z_%dd" % w] = z
        if w == AR1_OLS_WINDOW:
            resid_primary = resid
    hl, adf = _ar1_stats(resid_primary if resid_primary is not None else np.full(n, np.nan))
    out["feature_coint_halflife_120d"] = hl
    out["feature_coint_adf_120d"] = adf
    return pd.DataFrame(out, index=sub.index)


def compute_cointegration_signals(spot, macro_daily, specs=None):
    """
    Raw (un-binned) cointegration-residual signals for every (symbol, date).

    :param spot: OHLCV panel with symbol, date, close (all symbols; used for the
        per-coin series, the BTC factor and the equal-weight market index)
    :param macro_daily: macro store with date, gold, dxy, sp500 (log-level factors)
    :param specs: defaults to COINT_SPECS
    :return: frame keyed symbol, date with one column per spec
    :rtype: pd.DataFrame
    """
    specs = specs or COINT_SPECS
    df = spot[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0].dropna(subset=["close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["_logc"] = np.log(df["close"])

    R = build_regressors(spot, macro_daily)
    df = df.merge(R, on="date", how="left")

    parts = [_per_symbol(sub) for _, sub in df.groupby("symbol", sort=False)]
    feat = pd.concat(parts).sort_index()
    df = pd.concat([df[["symbol", "date"]], feat], axis=1)
    cols = list(specs.keys())
    out = df[["symbol", "date"] + cols].copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)   # never feed inf to the bin
    return out
