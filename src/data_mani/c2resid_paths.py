"""
Daily return paths for E148 (crypto/v2.0-c2resid): raw log returns, BTC+ETH beta residuals, and the
leave-one-out size-peer index. numpy/pandas only, so the chronos venv can import it too.

    python -m src.data_mani.c2resid_paths            # writes /scratch/danielco/c2resid/{paths.npz, rows.parquet}

Per coin i and calendar day t (spot close-to-close; a return needs both closes):
  r_i,t   log(close_t / close_t-1), clipped to [-1, 1] (sub-cent coins and bad prints)
  beta    OLS of r_i on [1, r_BTC, r_ETH] over the 120 days t-120..t-1 (>= 60 joint observations).
          Days strictly before t, so the residual on day t is out of sample for its own beta.
  e_i,t   r_i,t - b_BTC * r_BTC,t - c_ETH * r_ETH,t   (the intercept stays in: it is the coin's own drift)
  BTC (ucid 1) and ETH (ucid 1027) have no residual.

Panel row (coin, D) sees days <= D-1 only: the house submit lag of one day (attach_signals_asof),
and the target runs about close(D+2) -> close(D+30) (E142), i.e. daily returns on days D+3..D+30.
Its size bucket is the coin's shipped feature_market_cap_avg_20d bin on D (5 values), and its peer
index on day t is the mean raw return r_j,t over the OTHER coins j in that bucket on D that have a
return on t. Membership is fixed at D, so the peer path is known at D.
"""
import os

import numpy as np
import pandas as pd

SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
PANEL = os.path.join("data", "crypto", "v2.0-asof-e145", "train.parquet")   # frozen 2026-11-06 copy (E145)
OUT = "/scratch/danielco/c2resid"
BTC, ETH = "1", "1027"          # ucid is a string (the spot file also holds ambiguous "a,b" ids, never matched)
BETA_WIN, BETA_MIN = 120, 60
CLIP = 1.0
CTX = 256
LAG = 1                          # row D sees days <= D - LAG
SIZE = "feature_market_cap_avg_20d"


def returns(spot, cut=None):
    """(days, ucids, r [days, coins]) from the spot file, optionally truncated after day `cut`."""
    s = spot[["ucid", "date", "close"]].dropna().astype({"ucid": str})
    s = s[s["close"] > 0]
    if cut is not None:
        s = s[s["date"] <= cut]
    days = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    c = s.pivot_table(index="date", columns="ucid", values="close", aggfunc="last").reindex(days)
    lr = np.log(c.to_numpy("float64"))
    r = np.full_like(lr, np.nan)
    r[1:] = lr[1:] - lr[:-1]
    return days.to_numpy(), c.columns.to_numpy(), np.clip(r, -CLIP, CLIP)


def residuals(r, ucids):
    """e [days, coins]: BTC+ETH beta residual, beta from the BETA_WIN days strictly before each day."""
    b, x = r[:, list(ucids).index(BTC)], r[:, list(ucids).index(ETH)]
    m = np.isfinite(r) & np.isfinite(b)[:, None] & np.isfinite(x)[:, None]
    y = np.where(m, r, 0.0)
    B, X, M = np.where(m, b[:, None], 0.0), np.where(m, x[:, None], 0.0), m.astype("float64")
    terms = dict(n=M, b=B, x=X, y=y, bb=B * B, xx=X * X, bx=B * X, by=B * y, xy=X * y)

    def roll(a):                                  # sum over t-BETA_WIN .. t-1
        c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        out = np.full_like(a, np.nan)
        t = np.arange(a.shape[0])
        lo = np.maximum(t - BETA_WIN, 0)
        out[:] = c[t] - c[lo]
        return out
    S = {k: roll(v) for k, v in terms.items()}
    A = np.stack([np.stack([S["n"], S["b"], S["x"]], -1), np.stack([S["b"], S["bb"], S["bx"]], -1),
                  np.stack([S["x"], S["bx"], S["xx"]], -1)], -2)          # [days, coins, 3, 3]
    v = np.stack([S["y"], S["by"], S["xy"]], -1)
    ok = (S["n"] >= BETA_MIN) & (np.abs(np.linalg.det(A)) > 1e-12)
    coef = np.full(v.shape, np.nan)
    coef[ok] = np.linalg.solve(A[ok], v[ok][..., None])[..., 0]
    e = r - coef[..., 1] * b[:, None] - coef[..., 2] * x[:, None]
    e[:, [list(ucids).index(BTC), list(ucids).index(ETH)]] = np.nan
    return np.clip(e, -CLIP, CLIP).astype("float32")


def panel_rows(path=PANEL, days=None, ucids=None):
    """Panel rows with the spot column of their coin and the day index of D - LAG."""
    p = pd.read_parquet(path, columns=["symbol", "ucid", "date", SIZE]).reset_index()
    p = p[[c for c in ("symbol", "ucid", "date", SIZE) if c in p]]
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["date", "ucid"]).reset_index(drop=True)
    col = {u: k for k, u in enumerate(ucids)}
    p["col"] = p["ucid"].astype(str).map(col).fillna(-1).astype("int64")
    p["day"] = np.searchsorted(days, (p["date"] - pd.Timedelta(days=LAG)).to_numpy(), side="right") - 1
    p["bucket"] = (p[SIZE] * 4).round().astype("Int64")
    return p


def windows(arr, col, day, ctx=CTX):
    """[len(col), ctx] slices arr[day-ctx+1 .. day, col], NaN-padded at the start, NaN for col < 0."""
    n = len(col)
    out = np.full((n, ctx), np.nan, "float32")
    for k in range(n):
        if col[k] < 0 or day[k] < 0:
            continue
        lo = day[k] - ctx + 1
        a = arr[max(lo, 0):day[k] + 1, col[k]]
        out[k, ctx - len(a):] = a
    return out


def peer_windows(r, rows, ctx=CTX):
    """Leave-one-out size-peer index windows for the rows of ONE date (same day index)."""
    day = int(rows["day"].iloc[0])
    out = np.full((len(rows), ctx), np.nan, "float32")
    lo = day - ctx + 1
    for b, idx in rows.groupby("bucket").indices.items():
        cols = rows["col"].to_numpy()[idx]
        have = cols >= 0
        W = np.full((len(idx), ctx), np.nan)
        if have.any():
            a = r[max(lo, 0):day + 1][:, cols[have]].T
            W[have, ctx - a.shape[1]:] = a
        f = np.isfinite(W)
        S, C = np.where(f, W, 0).sum(0), f.sum(0)
        loo_s, loo_c = S[None, :] - np.where(f, W, 0), C[None, :] - f
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(loo_c > 0, loo_s / np.maximum(loo_c, 1), np.nan)
    return out


def build(spot=None, cut=None):
    spot = spot if spot is not None else pd.read_parquet(SPOT)
    spot["date"] = pd.to_datetime(spot["date"])
    days, ucids, r = returns(spot, cut)
    e = residuals(r, ucids)
    return days, ucids, r.astype("float32"), e


def main():
    os.makedirs(OUT, exist_ok=True)
    days, ucids, r, e = build()
    rows = panel_rows(days=days, ucids=ucids)
    np.savez(os.path.join(OUT, "paths.npz"), days=days, ucids=ucids, r=r, e=e)
    rows.drop(columns=[SIZE]).to_parquet(os.path.join(OUT, "rows.parquet"))
    have = rows["col"] >= 0
    print("%d days x %d coins; panel rows %d, with a spot column %.1f%%; residual defined on %.1f%% of coin-days with a return"
          % (len(days), len(ucids), len(rows), 100 * have.mean(), 100 * np.isfinite(e).sum() / np.isfinite(r).sum()))


if __name__ == "__main__":
    main()
