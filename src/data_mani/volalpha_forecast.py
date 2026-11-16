"""
E151 volume-alpha -- the out-of-fold VOLUME-forecast layer.

    python3 -m src.data_mani.volalpha_forecast table            # the daily training table (info date s)
    python3 -m src.data_mani.volalpha_forecast fit --quarter 2023Q2 [--n-jobs 4]    # one refit, predicts its quarter
    python3 -m src.data_mani.volalpha_forecast fit --all --workers 12
    python3 -m src.data_mani.volalpha_forecast collect          # -> oof.parquet + eval tables

Rows are (symbol, info date s) on the calendar-daily `ven` series. A row's inputs use data dated <= s.
Targets (never summed from one-day forecasts):
    y1  = eta(s+1)  = logdv(s+1) - mean(logdv(s-4..s))                    the primary auxiliary target
    y5  = mean(logdv(s+1..s+5)) - mean(logdv(s-4..s))                       direct 5-day
    ych = mean(logdv(s+3..s+31)) - mean(logdv(s-4..s))                      direct challenge horizon: Numerai
          date D = s+1, target window D+2..D+30
    lo / hi = logdv(s+1) below / above the coin's own trailing-252-day 10th / 90th percentile at s
Refits are quarterly on an EXPANDING window. The refit for quarter q trains only on rows whose target is
fully observed before q starts (target day <= q - 1: s <= q-2 for y1/lo/hi, q-6 for y5, q-32 for ych) and
predicts every row with s in q. So every stored forecast is out of sample: no row's own target, and no
later target, was seen by its model. Training rows are subsampled uniformly to at most MAX_TRAIN.

Models (fixed configurations, not searched -- a search would need its own nested split and is not what the
study is about): ridge, LightGBM, sklearn MLP (64,32), the paper's 32/16/8 MLP; LightGBM quantile 0.1 / 0.5
/ 0.9; LightGBM classifiers for lo / hi; ridge + LightGBM for y5 and ych.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = "/scratch/danielco/volalpha/forecast"
MAX_TRAIN = 600_000
FIRST_Q = pd.Period("2019Q3", "Q")
SEED = 151
H = {"y1": 1, "lo": 1, "hi": 1, "y5": 5, "ych": 31}


def build_table(fams):
    """The training/prediction table on the ven family."""
    from src.data_mani import volalpha_features as F
    ven = fams["ven"]
    days = pd.date_range(ven["date"].min(), ven["date"].max())
    close, dv, ret, logdv = F.basics(ven, days)
    high, low = F.wide(ven, "high", days), F.wide(ven, "low", days)
    nven, hhi = F.wide(ven, "n_venues", days), F.wide(ven, "hhi", days)
    ma5, ma22, ma252 = F.roll(logdv, 5), F.roll(logdv, 22), F.roll(logdv, 252)
    eta = logdv - ma5.shift(1)
    s5 = eta
    mkt = s5.median(axis=1)
    breadth = (s5 > 0).sum(axis=1) / s5.notna().sum(axis=1).replace(0, np.nan)
    rep = lambda x: pd.DataFrame(np.repeat(x.to_numpy()[:, None], logdv.shape[1], 1), index=days, columns=logdv.columns)
    q10 = logdv.rolling(252, min_periods=120).quantile(0.1)
    q90 = logdv.rolling(252, min_periods=120).quantile(0.9)
    nxt = logdv.shift(-1)
    fwd5 = logdv[::-1].rolling(5, min_periods=4).mean()[::-1].shift(-1)
    fwdch = logdv[::-1].rolling(29, min_periods=18).mean()[::-1].shift(-3)
    X = {"eta0": eta, "eta1": eta.shift(1), "eta2": eta.shift(2), "eta6": eta.shift(6),
         "lvl_ma5": logdv - ma5, "ma5_ma22": ma5 - ma22, "ma22_ma252": ma22 - ma252, "logdv": logdv, "ma5": ma5,
         "ret0": ret, "ret_ma5": F.roll(ret, 5), "ret_ma22": F.roll(ret, 22), "absret0": ret.abs(),
         "range0": (high - low) / close, "dvvol22": F.roll(logdv - logdv.shift(1), 22, "std"),
         "nven": nven, "hhi": hhi, "mkt": rep(mkt), "breadth": rep(breadth),
         "btc": rep(s5["BTC"]) if "BTC" in s5 else rep(mkt * np.nan), "eth": rep(s5["ETH"]) if "ETH" in s5 else rep(mkt * np.nan),
         "wkend": F.weekend_sensitivity(logdv), "q10gap": logdv - q10, "q90gap": logdv - q90,
         "y1": nxt - ma5, "y5": fwd5 - ma5, "ych": fwdch - ma5,
         "lo": (nxt < q10).astype(float).where(nxt.notna() & q10.notna()),
         "hi": (nxt > q90).astype(float).where(nxt.notna() & q90.notna())}
    t = F.long(X)
    t = t[t["ma5"].notna() & t["eta0"].notna()]
    t["dow"] = (t["date"] + pd.Timedelta(days=1)).dt.dayofweek
    return t.reset_index(drop=True)


FEATS = ["eta0", "eta1", "eta2", "eta6", "lvl_ma5", "ma5_ma22", "ma22_ma252", "logdv", "ret0", "ret_ma5", "ret_ma22",
         "absret0", "range0", "dvvol22", "nven", "hhi", "mkt", "breadth", "btc", "eth", "wkend", "q10gap", "q90gap", "dow"]


def _dense(X, med):
    Z = X.fillna(med)
    d = pd.get_dummies(Z["dow"].astype(int), prefix="dow").reindex(columns=["dow_%d" % i for i in range(7)], fill_value=0)
    return np.hstack([Z.drop(columns="dow").to_numpy("float64"), d.to_numpy("float64")])


def fit_quarter(q, n_jobs=4):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    q = pd.Period(q, "Q")
    out_path = os.path.join(ROOT, "oof", "%s.parquet" % q)
    if os.path.exists(out_path):
        return out_path
    t = pd.read_parquet(os.path.join(ROOT, "table.parquet"))
    start, end = q.start_time, (q + 1).start_time
    pred = t[(t["date"] >= start) & (t["date"] < end)].reset_index(drop=True)
    if pred.empty:
        return None
    out = pred[["symbol", "date", "ma5", "eta0", "y1", "y5", "ych", "lo", "hi"]].copy()
    rng = np.random.default_rng(SEED + q.ordinal)
    info = {}
    for tgt in ("y1", "lo", "hi", "y5", "ych"):
        last_s = start - pd.Timedelta(days=H[tgt] + 1)       # target day <= start - 1
        tr = t[(t["date"] <= last_s) & t[tgt].notna()]
        if len(tr) > MAX_TRAIN:
            tr = tr.iloc[np.sort(rng.choice(len(tr), MAX_TRAIN, replace=False))]
        info[tgt] = dict(n_train=len(tr), last_s=str(last_s.date()))
        if len(tr) < 20_000:
            continue
        Xtr, ytr, Xte = tr[FEATS], tr[tgt].to_numpy(), pred[FEATS]
        med = Xtr.median()
        gbm = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, n_jobs=n_jobs, random_state=SEED, verbose=-1)
        if tgt in ("lo", "hi"):
            m = lgb.LGBMClassifier(**gbm).fit(Xtr, ytr.astype(int))
            out["p_" + tgt] = m.predict_proba(Xte)[:, 1]
            continue
        out["%s_lgb" % tgt] = lgb.LGBMRegressor(**gbm).fit(Xtr, ytr).predict(Xte)
        sc = StandardScaler().fit(_dense(Xtr, med))
        A, B = sc.transform(_dense(Xtr, med)), sc.transform(_dense(Xte, med))
        out["%s_ridge" % tgt] = Ridge(alpha=10.0).fit(A, ytr).predict(B)
        if tgt == "y1":
            for a in (0.1, 0.5, 0.9):
                out["y1_q%02d" % round(100 * a)] = lgb.LGBMRegressor(objective="quantile", alpha=a, **gbm).fit(Xtr, ytr).predict(Xte)
            sub = rng.choice(len(A), min(len(A), 300_000), replace=False)
            for name, hid in (("mlp", (64, 32)), ("paper", (32, 16, 8))):
                m = MLPRegressor(hidden_layer_sizes=hid, alpha=1e-4, learning_rate_init=1e-3, batch_size=1024, max_iter=40,
                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)
                out["y1_" + name] = m.fit(A[sub], ytr[sub]).predict(B)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    json.dump(info, open(out_path + ".json", "w"))
    return out_path


def quarters():
    t = pd.read_parquet(os.path.join(ROOT, "table.parquet"), columns=["date"])
    last = pd.Period(t["date"].max(), "Q")
    return [str(q) for q in pd.period_range(FIRST_Q, last, freq="Q")]


def collect():
    parts = [pd.read_parquet(os.path.join(ROOT, "oof", f)) for f in sorted(os.listdir(os.path.join(ROOT, "oof")))
             if f.endswith(".parquet")]
    o = pd.concat(parts, ignore_index=True)
    m1 = [c for c in ("y1_ridge", "y1_lgb", "y1_mlp", "y1_paper") if c in o]
    o["hat_eta_1d"] = o[m1].mean(axis=1)
    o["disagree"] = o[m1].std(axis=1, ddof=0)
    o["hat_logdv_1d"] = o["ma5"] + o["hat_eta_1d"]
    o["hat_logdv_5d"] = o["ma5"] + o[[c for c in ("y5_ridge", "y5_lgb") if c in o]].mean(axis=1)
    o["hat_logdv_ch"] = o["ma5"] + o[[c for c in ("ych_ridge", "ych_lgb") if c in o]].mean(axis=1)
    o["unc"] = o["y1_q90"] - o["y1_q10"]
    o["hat_logdv_q10"] = o["ma5"] + o["y1_q10"]
    o["hat_logdv_down"] = o["hat_logdv_1d"] - (o["y1_q50"] - o["y1_q10"]).clip(lower=0)
    o = o.sort_values(["symbol", "date"])
    # previous realised error, known at s: eta(s) minus the forecast of it made at s-1
    prev = o[["symbol", "date", "hat_eta_1d"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    o = o.merge(prev.rename(columns={"hat_eta_1d": "_prevhat"}), on=["symbol", "date"], how="left")
    o["prev_err"] = o["eta0"] - o["_prevhat"]
    o = o.drop(columns="_prevhat")
    o.to_parquet(os.path.join(ROOT, "oof.parquet"), index=False)
    print("oof: %d rows, %d symbols, %s..%s" % (len(o), o["symbol"].nunique(), o["date"].min().date(), o["date"].max().date()))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "fit", "collect"])
    ap.add_argument("--quarter")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs(ROOT, exist_ok=True)
    if a.cmd == "table":
        from src.data_mani import volalpha_sources as S
        fams, _ = S.families()
        t = build_table(fams)
        t.to_parquet(os.path.join(ROOT, "table.parquet"), index=False)
        print("table: %d rows, %d symbols, targets non-null %s" % (len(t), t["symbol"].nunique(),
              {k: int(t[k].notna().sum()) for k in H}))
    elif a.cmd == "fit":
        qs = quarters() if a.all else [a.quarter]
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            for q, p in zip(qs, ex.map(fit_quarter, qs, [a.n_jobs] * len(qs))):
                print("%s -> %s  (%.0fs)" % (q, p, time.time() - t0), flush=True)
    else:
        collect()


if __name__ == "__main__":
    sys.exit(main())
