"""
crypto/v2.0-c2resid (E148): v2.0 + six forecasts of each coin's BTC+ETH beta-residual return over the
target's span (days D+3..D+30), each rank-binned within date with Numerai's rule.

    python -m src.data_mani.build_c2resid --ridge      # causal ridge + residual momentum -> /scratch/.../ridge.parquet
    python -m src.data_mani.build_c2resid --build      # after chronos2_resid has written every part

Columns (NaN where the row has no signal; the loader fills 0.5):
  feature_c2r_ridge   causal rolling ridge on window sums of the same four series C2-cov sees
  feature_c2r_resmom  residual momentum, sum of the last 20 daily residuals (no fit)
  feature_c2r_raw     C2 zero-shot, raw return path alone            (q50 summed over D+3..D+30)
  feature_c2r_res     C2 zero-shot, residual path alone
  feature_c2r_cov     C2 zero-shot, residual + BTC, ETH, leave-one-out size-peer covariates
  feature_c2r_grp     C2 zero-shot, one group per (date, size bucket): members' residual paths + BTC + ETH

The ridge, as C2, reads days <= D-1 only. It is refitted every REFIT panel dates R on the rows of the
trailing TRAIN_DAYS whose label (the residual summed over D'+3..D'+30) resolved by R-1, i.e.
D' <= R - 31 days, with alpha by RidgeCV's efficient leave-one-out on that window alone. Inputs: the
within-date rank of the coin's residual window sums and of its peer-index window sums (8 windows each,
days back 1, 2-5, 6-10, 11-20, 21-40, 41-80, 81-160, 161-256), and residual x market interactions
(residual rank over 1-5 and 6-20 days x BTC and ETH sums over 1-5 and 6-20 days). BTC and ETH sums on
their own are constant within a date, so a linear model can use them only through interactions.
Target: the within-date rank of the forward residual sum, centred.
"""
import argparse
import json
import os
import shutil
from time import time

import numpy as np
import pandas as pd

from src.data_mani import c2resid_paths as RP
from src.data_mani.history_features import bin_ties_preserved

SOURCE = os.path.dirname(RP.PANEL)
DEST = os.path.join("data", "crypto", "v2.0-c2resid")
LAGS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 256)]
REFIT, TRAIN_DAYS, RESOLVE_DAYS = 22, 365, 31
FWD = (4, 32)                       # day offsets from D-1: days D+3 .. D+30
MIN_OBS = 64
SIGNALS = ["ridge", "resmom", "raw", "res", "cov", "grp"]


def wsum(W, lo, hi):
    """Mean over the last-lo..last-hi columns times the width (a NaN-tolerant sum)."""
    a = W[:, RP.CTX - hi:RP.CTX - lo]
    n = np.isfinite(a).sum(1)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, np.nanmean(a, 1) * (hi - lo), np.nan)


def rank0(x):
    s = pd.Series(x)
    return (s.rank(pct=True) - 0.5).fillna(0.0).to_numpy()


def date_features(paths, rows):
    """Ridge inputs, residual momentum, eligibility and the forward label for the rows of one date."""
    r, e, ucids = paths["r"], paths["e"], list(paths["ucids"])
    col, day = rows["col"].to_numpy(), rows["day"].to_numpy()
    d = int(day[0])
    E = RP.windows(e, col, day)
    P = RP.peer_windows(r, rows)
    mk = RP.windows(r, np.array([ucids.index(RP.BTC), ucids.index(RP.ETH)]), np.array([d, d]))
    ok = (np.isfinite(E).sum(1) >= MIN_OBS) & (col >= 0)
    es = {w: wsum(E, *w) for w in LAGS}
    F = {"e_%d_%d" % w: rank0(es[w]) for w in LAGS}
    F.update({"p_%d_%d" % w: rank0(wsum(P, *w)) for w in LAGS})
    for w in [(0, 5), (5, 20)]:
        er = rank0(wsum(E, *w))
        for k, nm in enumerate(("btc", "eth")):
            for m in [(0, 5), (5, 20)]:
                F["e%d_%d_x_%s%d_%d" % (w + (nm,) + m)] = er * float(np.nansum(mk[k, RP.CTX - m[1]:RP.CTX - m[0]]))
    X = pd.DataFrame(F, index=rows.index)
    mom = np.where(np.isfinite(E[:, -20:]).sum(1) >= 10, np.nansum(E[:, -20:], 1), np.nan)
    # forward label: residuals on days D+3..D+30 = indices d+4 .. d+31
    y = np.full(len(rows), np.nan)
    if d + FWD[1] <= e.shape[0]:
        fw = np.where(col[:, None] >= 0, e[d + FWD[0]:d + FWD[1]][:, np.maximum(col, 0)].T, np.nan)
        n = np.isfinite(fw).sum(1)
        y = np.where(n >= 20, np.nanmean(fw, 1) * (FWD[1] - FWD[0]), np.nan)
    return X, mom, ok, y


def ridge_signals():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load(os.path.join(RP.OUT, "paths.npz"), allow_pickle=True)
    paths = dict(days=z["days"], ucids=z["ucids"], r=z["r"], e=z["e"])
    rows = pd.read_parquet(os.path.join(RP.OUT, "rows.parquet"))
    rows = rows[(rows["col"] >= 0) & (rows["day"] >= RP.CTX // 4)]
    t0, Xs, M, OK, Y = time(), [], [], [], []
    for D, g in rows.groupby("date", sort=True):
        X, mom, ok, y = date_features(paths, g)
        Xs.append(X)
        M.append(pd.Series(mom, index=g.index))
        OK.append(pd.Series(ok, index=g.index))
        Y.append(pd.Series(y, index=g.index))
    X, mom, ok, y = pd.concat(Xs), pd.concat(M), pd.concat(OK), pd.concat(Y)
    print("features for %d rows, %.1f min" % (len(X), (time() - t0) / 60), flush=True)
    date = rows.loc[X.index, "date"]
    yr = y.groupby(date).rank(pct=True) - 0.5
    grid = np.sort(date.unique())
    pred = pd.Series(np.nan, index=X.index)
    log = []
    for k in range(0, len(grid), REFIT):
        R = pd.Timestamp(grid[k])
        tr = ok & yr.notna() & (date <= R - pd.Timedelta(days=RESOLVE_DAYS)) & (date > R - pd.Timedelta(days=RESOLVE_DAYS + TRAIN_DAYS))
        te = ok & date.isin(grid[k:k + REFIT])
        if tr.sum() < 5000 or not te.any():
            continue
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))
        m.fit(X[tr].to_numpy(), yr[tr].to_numpy())
        pred[te] = m.predict(X[te].to_numpy())
        log.append(dict(refit=str(R.date()), train_rows=int(tr.sum()), alpha=float(m[-1].alpha_)))
    print("ridge: %d refits, first %s, %d rows predicted" % (len(log), log[0]["refit"] if log else None, pred.notna().sum()), flush=True)
    out = pd.DataFrame({"ridge": pred, "resmom": mom.where(ok), "fwd_resid": y})
    out.to_parquet(os.path.join(RP.OUT, "ridge.parquet"))
    json.dump(log, open(os.path.join(RP.OUT, "ridge_refits.json"), "w"), indent=1)


def build():
    from src.data_mani import chronos2_resid as C
    rows = pd.read_parquet(os.path.join(RP.OUT, "rows.parquet"))
    rd = pd.read_parquet(os.path.join(RP.OUT, "ridge.parquet"))
    C2 = C.merged()
    raw = pd.DataFrame(index=rows.index)
    raw["ridge"], raw["resmom"], raw["fwd_resid"] = rd["ridge"], rd["resmom"], rd["fwd_resid"]
    for a in C.ARMS:
        for c in ("q10", "q50", "q90", "q50_all"):
            raw["%s_%s" % (a, c)] = C2[a][c].reindex(rows.index)
        raw[a] = raw["%s_q50" % a]
    raw = pd.concat([rows[["symbol", "ucid", "date", "bucket"]], raw], axis=1)
    raw.to_parquet(os.path.join(RP.OUT, "signals_raw.parquet"))
    sig = raw[SIGNALS].astype("float64")
    sig.columns = ["feature_c2r_" + s for s in SIGNALS]
    binned = bin_ties_preserved(sig, raw["date"])
    df = pd.read_parquet(os.path.join(SOURCE, "train.parquet")).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    key = raw[["symbol", "date"]].join(binned)
    out = df.merge(key, on=["symbol", "date"], how="left", validate="1:1")
    assert len(out) == len(df)
    for c in binned:
        out[c] = out[c].astype("float32")
    os.makedirs(DEST, exist_ok=True)
    out.to_parquet(os.path.join(DEST, "train.parquet"), index=False)
    shutil.copy(os.path.join(SOURCE, "historical_meta_models.parquet"), DEST)
    mass = {c: binned[c].value_counts(normalize=True).sort_index().round(4).to_dict() for c in binned}
    info = dict(rows=len(out), source=SOURCE, observed={c: float(out[c].notna().mean()) for c in binned}, bin_mass=mass,
                features=int(sum(c.startswith("feature_") for c in out)))
    json.dump(info, open(os.path.join(DEST, "build.json"), "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.ridge:
        ridge_signals()
    if a.build:
        build()


if __name__ == "__main__":
    main()
