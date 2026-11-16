"""
Rolling market-beta features for Numerai Crypto: beta is the projection (OLS
slope) of a coin's daily returns on BTC's daily returns over a trailing window.

For symbol i and date t, over the trailing window of w daily returns ending at t
(all observations <= t):

    beta_{i,t,w} = Cov(r_i, r_btc) / Var(r_btc)          (the single-factor
                 = rolling_cov(r_i, r_btc, w) / rolling_var(r_btc, w)   market model)

This is the coin's sensitivity to the market factor (BTC). beta > 1 is a
high-beta / amplifying coin, beta < 1 defensive, beta < 0 a hedge. It is a
PER-SYMBOL feature (varies cross-sectionally), so it rank-bins normally -- unlike
a shared/macro feature. Pure price math and causal: a full window is required
(min_periods=w), so a coin with too little history, or BTC-missing dates, is NaN
there and becomes the neutral bin at load. The submission-lag timing is handled
by attach_signals_asof (shared with the momentum/vol builders), since the spot
feed is external -- so row D uses returns through D-1, the last completed bar.

Keep this module free of file layout and the Numerai scale (binning lives in
binning.py; the offline wiring in build_beta.py) so the live path can reuse
compute_beta_signals + attach_signals_asof unchanged.
"""
import numpy as np
import pandas as pd

# name -> trailing window length, in daily-return observations (~calendar days)
BETA_SPECS = {
    "feature_beta_btc_30d": {"window": 30},
    "feature_beta_btc_60d": {"window": 60},
    "feature_beta_btc_90d": {"window": 90},
}


def compute_beta_signals(close_panel, specs=None, market_symbol="BTC"):
    """
    Raw (un-binned) rolling market-beta vs ``market_symbol`` for every (symbol,date).

    :param close_panel: long panel with columns ``symbol``, ``date``, ``close``
        (one close per (symbol, date); many rows per symbol)
    :type close_panel: pd.DataFrame
    :param specs: feature spec map {name: {"window": w}}; defaults to BETA_SPECS
    :type specs: dict or None
    :param market_symbol: the market factor whose returns coins are projected on
    :type market_symbol: str
    :return: frame keyed ``symbol``, ``date`` with one float column per spec,
        NaN wherever a symbol lacks a full window of paired returns behind date t
    :rtype: pd.DataFrame
    """
    specs = specs or BETA_SPECS
    required = {"symbol", "date", "close"}
    if not required.issubset(close_panel.columns):
        raise KeyError("close_panel needs columns %s; got %s"
                       % (sorted(required), list(close_panel.columns)))

    df = close_panel[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(["symbol", "date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # market (BTC) daily returns, indexed by date
    mkt = df[df["symbol"] == market_symbol]
    if mkt.empty:
        raise ValueError("market symbol %r not found in the spot panel" % market_symbol)
    mkt_ret = mkt.set_index("date")["close"].sort_index().pct_change()
    mkt_ret = mkt_ret[~mkt_ret.index.duplicated(keep="last")]

    # each coin's own daily returns, and the market return aligned by date
    df["_dret"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["_mret"] = df["date"].map(mkt_ret)

    windows = {int(spec["window"]): name for name, spec in specs.items()}
    out_cols = list(specs.keys())

    def _per_symbol(sub):
        ri = sub["_dret"]
        rm = sub["_mret"]
        for w, name in windows.items():
            cov = ri.rolling(w, min_periods=w).cov(rm)
            var = rm.rolling(w, min_periods=w).var()
            sub[name] = cov / var.replace(0.0, np.nan)
        return sub

    df = df.groupby("symbol", sort=False, group_keys=False).apply(_per_symbol)
    return df[["symbol", "date"] + out_cols].reset_index(drop=True)
