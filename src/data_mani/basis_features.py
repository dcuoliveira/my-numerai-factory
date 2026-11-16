"""
Perp/spot BASIS and funding-carry signals. PURE functions: turn a per-(symbol,date)
frame carrying spot close, perpetual close and daily funding into raw basis/funding
columns. Binning and the panel merge are the usual separate seams (build_basis).

------------------------------------------------------------------------------
What the basis is
------------------------------------------------------------------------------

For a perpetual future, basis = (perp_close - spot_close) / spot_close -- the premium
(perp rich) or discount (perp cheap) of the perp over spot. It is the carry a
perp long/short earns, and it is anchored to spot by the funding rate (longs pay
shorts when the basis is positive). A stretched basis tends to mean-revert; a
persistent one is carry. Both the basis and the funding rate are standard crypto
signals (perpetual-swap carry / funding-rate literature).

    feature_basis_pct        raw percent basis (perp-spot)/spot
    feature_basis_ma_20d     20d mean basis (persistent carry level)
    feature_basis_z_{20,60}d basis z-score (basis - rolling mean)/rolling std -- a
                             stretched basis -> expected mean reversion
    feature_basis_chg_{5,20}d basis momentum (change over 5/20 days)
    feature_funding_rate     daily funding (sum of the 3 intra-day funding payments)
    feature_funding_ma_{7,30}d smoothed funding carry
    feature_funding_z_30d    funding z-score (stretched funding regime)

Causality: the basis at t uses perp(t) and spot(t), both <= t; every rolling stat is
trailing (min_periods = window). The submission lag (<= D-1) is applied later in
attach_signals_asof. CAVEAT: the perp is Binance; the spot is whatever the spot store
holds (Yahoo for most names), so feature_basis_* carries a little cross-venue noise.
Funding is same-venue (Binance) and clean.
"""

import numpy as np
import pandas as pd

BASIS_FEATURES = [
    "feature_basis_pct", "feature_basis_ma_20d",
    "feature_basis_z_20d", "feature_basis_z_60d",
    "feature_basis_chg_5d", "feature_basis_chg_20d",
    "feature_funding_rate", "feature_funding_ma_7d",
    "feature_funding_ma_30d", "feature_funding_z_30d",
]


def compute_basis_signals(panel):
    """
    Raw (un-binned) basis/funding signals for every (symbol, date).

    :param panel: long frame with columns symbol, date, spot_close, perp_close,
        funding (funding may be NaN where unavailable).
    :return: frame keyed symbol, date with one column per BASIS_FEATURES
    :rtype: pd.DataFrame
    """
    need = {"symbol", "date", "spot_close", "perp_close", "funding"}
    if not need.issubset(panel.columns):
        raise KeyError("panel needs %s; got %s" % (sorted(need), list(panel.columns)))
    df = panel[["symbol", "date", "spot_close", "perp_close", "funding"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["spot_close"] > 0]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["_basis"] = df["perp_close"] / df["spot_close"] - 1.0      # NaN where perp missing

    def rm(col, w, how, mp=None):
        mp = w if mp is None else mp
        r = df.groupby("symbol", sort=False)[col].rolling(w, min_periods=mp)
        return getattr(r, how)().reset_index(level=0, drop=True).sort_index()

    g = df.groupby("symbol", sort=False)
    out = {}
    out["feature_basis_pct"] = df["_basis"]
    out["feature_basis_ma_20d"] = rm("_basis", 20, "mean")
    for w in (20, 60):
        mu = rm("_basis", w, "mean"); sd = rm("_basis", w, "std").replace(0.0, np.nan)
        out["feature_basis_z_%dd" % w] = (df["_basis"] - mu) / sd
    for k in (5, 20):
        out["feature_basis_chg_%dd" % k] = df["_basis"] - g["_basis"].shift(k)
    out["feature_funding_rate"] = df["funding"]
    out["feature_funding_ma_7d"] = rm("funding", 7, "mean")
    out["feature_funding_ma_30d"] = rm("funding", 30, "mean")
    mu = rm("funding", 30, "mean"); sd = rm("funding", 30, "std").replace(0.0, np.nan)
    out["feature_funding_z_30d"] = (df["funding"] - mu) / sd

    res = pd.DataFrame(out)
    res.insert(0, "date", df["date"].values)
    res.insert(0, "symbol", df["symbol"].values)
    return res[["symbol", "date"] + BASIS_FEATURES]
