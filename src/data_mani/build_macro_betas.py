"""
Build crypto/v2.0-macrobeta: the 22 shipped v2.0 features plus each coin's ROLLING
BETA to every macro and inflation factor, causal and rank-binned per date.

    python3 -m src.data_mani.build_macro_betas
    python3 -m src.data_mani.build_macro_betas --windows 60,120 --out-suffix=-macrobeta

Run it with:  --data-version crypto/v2.0-macrobeta

Why betas instead of the macro / inflation features themselves
--------------------------------------------------------------
A macro feature is the same for every coin on a date, so it cannot reorder coins:
it is exactly inert in a linear model (E138/E139: ridge F_MACRO and F_INFL equal
v2.0 to 1e-5), and LightGBM used it only to fit a dev-period regime split. A coin's
SENSITIVITY to a macro factor differs across coins, so it is cross-sectional and
can be binned on Numerai's scale like any shipped feature.

The factors (16), as one-day changes on the macro business-day calendar
-------------------------------------------------------------------------
  log returns   gold (GLD), silver (SLV), S&P 500, DXY
  yield changes y1 y2 y5 y10 y30                      (percentage points)
  slope changes 2s10s 1s10s 5s30s                      (same definitions as MACRO_SPECS)
  break-even    bei 5y 10y 30y, 5y5y forward changes   (same as INFLATION_SPECS)

The beta, per coin and factor
-----------------------------
  beta_t = cov(r_coin, dX) / var(dX) over the last W macro business days <= t,
i.e. the slope of the coin's log return on the factor's change, the usual factor
exposure. Univariate, one factor at a time: the yields and slopes are close to
collinear, and a joint regression would give unstable, sign-flipping coefficients.
The coin's return over each interval between consecutive macro dates is its
close-to-close log return between those two calendar dates, so weekend crypto
moves fall into Monday's interval, matched against the factor's change over the
same interval. Crypto closes at 00:00 UTC and US markets at ~21:00 UTC, so each
pair is ~3 hours asynchronous; that biases betas toward zero a little and equally
for every coin.

At least 80% of the window must have both a coin return and a factor change, so
a young or patchy coin is NaN rather than estimated on a few points (warm-up).

Point-in-time
-------------
beta_t uses only data dated <= t. It is attached to panel row D as-of the newest
macro date <= D-1 (attach_signals_asof, per symbol, at most 7 days stale, as the
macro block), so neither the coin's D close nor the factor's D close is used. Then
each beta is rank-binned per date over the panel's own cross-section
(bin_cross_sectional), 5/20/50/20/5 like the shipped features.

Missing values follow the E134-E140 standard: no row dropped, no indicators, NaN in
the file (0.5 at load), and evaluation routes coins without betas to v2.0.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_MASS, bin_cross_sectional, bin_mass
from src.data_mani.build_feed_fillind import SPOT
from src.data_mani.momentum_features import attach_signals_asof
from src.data_mani.utils import download_if_missing, get_api, get_spec, local_path

MACRO = os.path.join("data", "crypto", "macro", "macro_daily.parquet")
LAG, STALE, MIN_SHARE = 1, 7, 0.8
WINDOWS = (60, 120)


def factor_changes(macro_daily):
    """One-day changes of the 16 factors on the macro calendar (index = date)."""
    m = macro_daily.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.drop_duplicates("date").sort_values("date").set_index("date")
    lvl = {}
    for s in ("gold", "silver", "sp500", "dxy"):
        lvl[s] = np.log(m[s].where(m[s] > 0))
    for y in ("y1", "y2", "y5", "y10", "y30"):
        lvl[y] = m[y]
    lvl["slope_2s10s"] = m["y10"] - m["y2"]
    lvl["slope_1s10s"] = m["y10"] - m["y1"]
    lvl["slope_5s30s"] = m["y30"] - m["y5"]
    for k in ("5", "10", "30"):
        lvl["bei_%sy" % k] = m["y" + k] - m["r" + k]
    lvl["bei_5y5y"] = (10.0 * (m["y10"] - m["r10"]) - 5.0 * (m["y5"] - m["r5"])) / 5.0
    names = {"sp500": "snp"}
    return pd.DataFrame({names.get(k, k): v.diff() for k, v in lvl.items()})


def coin_returns(spot, dates):
    """Coins' close-to-close log returns between consecutive macro dates (wide)."""
    s = spot[["symbol", "date", "close"]].dropna()
    s = s[s["close"] > 0].drop_duplicates(["symbol", "date"])
    px = s.pivot(index="date", columns="symbol", values="close").sort_index()
    px = px.reindex(dates)          # the coin's close ON each macro date; NaN if none
    return np.log(px).diff()


def rolling_betas(R, F, windows=WINDOWS, min_share=MIN_SHARE):
    """{column name: wide frame of betas} for every factor x window, causal."""
    out = {}
    for f in F.columns:
        x = F[f].values[:, None]
        ok = R.notna().values & np.isfinite(x)
        y = np.where(ok, R.values, 0.0)
        xx = np.where(ok, np.broadcast_to(x, R.shape), 0.0)
        cum = lambda a: np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])  # noqa: E731
        n, sx, sy, sxy, sxx = cum(ok.astype(float)), cum(xx), cum(y), cum(xx * y), cum(xx * xx)
        for w in windows:
            k = lambda c: c[w:] - c[:-w]  # noqa: E731
            nn, a, b, ab, aa = k(n), k(sx), k(sy), k(sxy), k(sxx)
            with np.errstate(invalid="ignore", divide="ignore"):
                var = aa - a * a / nn
                beta = (ab - a * b / nn) / var
            beta[(nn < min_share * w) | ~(var > 1e-12)] = np.nan
            full = np.full(R.shape, np.nan)
            full[w - 1:] = beta
            out["feature_mbeta_%s_%dd" % (f, w)] = pd.DataFrame(full, index=R.index, columns=R.columns)
    return out


def add_macro_betas(base, spot, macro_daily, windows=WINDOWS, period="date", symbol="symbol"):
    """Return (base + binned beta columns in base's row order, column names)."""
    F = factor_changes(macro_daily)
    R = coin_returns(spot, F.index)
    B = rolling_betas(R, F, windows)
    cols = list(B)
    long = pd.concat({c: B[c].stack(future_stack=True) for c in cols}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long.dropna(how="all").reset_index()
    out = attach_signals_asof(base, long, cols, lag_days=LAG, max_staleness_days=STALE,
                              period_col=period, symbol_col=symbol)
    if len(out) != len(base):
        raise SystemExit("as-of attach changed the row count %d -> %d" % (len(base), len(out)))
    out = bin_cross_sectional(out, cols, period)
    return out[list(base.columns) + cols], cols


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-version", default="crypto/v2.0")
    p.add_argument("--out-suffix", default="-macrobeta")
    p.add_argument("--spot", default=SPOT)
    p.add_argument("--macro", default=MACRO)
    p.add_argument("--windows", default=",".join(map(str, WINDOWS)))
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_spec("crypto")
    api = get_api("crypto")
    windows = tuple(int(w) for w in args.windows.split(","))
    src = download_if_missing(api, "crypto", args.source_version, "train.parquet")
    base = pd.read_parquet(src).reset_index()
    base["date"] = pd.to_datetime(base["date"])
    base["ucid"] = base["ucid"].astype(str)
    n_src = sum(c.startswith("feature_") for c in base)
    print("source: %d rows, %d features (%s)" % (len(base), n_src, args.source_version))
    spot = pd.read_parquet(args.spot, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    out, cols = add_macro_betas(base, spot, pd.read_parquet(args.macro), windows,
                                spec["period_col"], spec["id_col"])
    drift = max(float(np.nanmax(np.abs(bin_mass(out[c]) - BIN_MASS))) for c in cols)
    for c in cols:
        print("   %-34s observed %5.1f%%" % (c, 100 * out[c].notna().mean()))
    print("%d beta features, max bin-mass drift %.4f" % (len(cols), drift))
    version = args.source_version + args.out_suffix
    dest = local_path("crypto", version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index("symbol").to_parquet(dest)
    print("wrote %d rows, %d features (%d shipped + %d betas) -> %s (%.0f MB)"
          % (len(out), n_src + len(cols), n_src, len(cols), dest, os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", args.source_version, filename)
        target = local_path("crypto", version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
    print("run it with:  --data-version %s" % version)


if __name__ == "__main__":
    main()
