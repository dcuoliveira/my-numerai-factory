"""
Betting-against-beta-under-short-constraints inputs for Numerai Crypto.

The anomaly (Miller 1977; Hong & Sraer 2016, "speculative betas"): when short-
selling is constrained AND beliefs diverge, high-beta assets are bid up by
optimists and become OVERPRICED, so high-beta underperforms -- and the effect
concentrates where shorting is hardest (illiquid names) and disagreement is
highest (high realised vol / turnover). Crypto is a strong setting: shorting the
alt long-tail is costly/constrained, beliefs are heterogeneous and retail-driven.

So the signal is not beta alone but beta INTERACTED with short-constraint and
disagreement proxies. From the spot feed (close + volume; no market cap) we build,
per symbol over trailing windows w (all observations <= t, causal):

    beta_w   = Cov(r_i, r_btc) / Var(r_btc)          market sensitivity
    illiq_w  = mean_{t-w<s<=t} |r_s| / (close_s * volume_s)   Amihud illiquidity
               (dollar-volume illiquidity -- the short-ease / no-borrow proxy)
    rvol_w   = std of daily returns over the window  (disagreement / lottery proxy)

This module returns those RAW continuous inputs per (symbol, date). The builder
(build_bab.py) attaches them point-in-time, forms the interactions as products of
per-date centered percentiles (so "high beta AND illiquid" is extreme), and bins
to Numerai's scale. Pure price math here so the live path can reuse it.
"""
import numpy as np
import pandas as pd

WINDOWS = (30, 60, 90)


def compute_bab_inputs(close_panel, windows=WINDOWS, market_symbol="BTC"):
    """
    Raw (un-binned) beta / Amihud-illiquidity / realised-vol per (symbol, date).

    :param close_panel: long panel with columns ``symbol``, ``date``, ``close``,
        ``volume`` (one row per (symbol, date))
    :type close_panel: pd.DataFrame
    :param windows: trailing window lengths in daily-return observations
    :type windows: tuple[int]
    :param market_symbol: the market factor beta is measured against
    :type market_symbol: str
    :return: frame keyed ``symbol``, ``date`` with ``_beta_{w}``, ``_illiq_{w}``,
        ``_rvol_{w}`` columns; NaN wherever a full window is unavailable
    :rtype: pd.DataFrame
    """
    required = {"symbol", "date", "close", "volume"}
    if not required.issubset(close_panel.columns):
        raise KeyError("close_panel needs %s; got %s"
                       % (sorted(required), list(close_panel.columns)))
    df = close_panel[["symbol", "date", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(["symbol", "date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    mkt = df[df["symbol"] == market_symbol]
    if mkt.empty:
        raise ValueError("market symbol %r not in the spot panel" % market_symbol)
    mret = mkt.set_index("date")["close"].sort_index().pct_change()
    mret = mret[~mret.index.duplicated(keep="last")]

    df["_r"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["_mret"] = df["date"].map(mret)
    dollar_vol = df["close"] * df["volume"]
    df["_illiq_daily"] = np.where(dollar_vol > 0,
                                  df["_r"].abs() / dollar_vol, np.nan)

    out_cols = []
    for w in windows:
        out_cols += ["_beta_%d" % w, "_illiq_%d" % w, "_rvol_%d" % w]

    def _per_symbol(sub):
        ri, rm, il = sub["_r"], sub["_mret"], sub["_illiq_daily"]
        for w in windows:
            cov = ri.rolling(w, min_periods=w).cov(rm)
            var = rm.rolling(w, min_periods=w).var().replace(0.0, np.nan)
            sub["_beta_%d" % w] = cov / var
            sub["_illiq_%d" % w] = il.rolling(w, min_periods=w).mean()
            sub["_rvol_%d" % w] = ri.rolling(w, min_periods=w).std()
        return sub

    df = df.groupby("symbol", sort=False, group_keys=False).apply(_per_symbol)
    return df[["symbol", "date"] + out_cols].reset_index(drop=True)
