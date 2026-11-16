"""
E118's TabPFN passes: what else TabPFN v2 can answer, per (symbol, date), causally.

    CUDA_VISIBLE_DEVICES=0 python -m src.data_mani.fmx_tabpfn --pass attn,nbr,extreme --device cuda:0

foundation_repr (E109) asked TabPFN for its predictive distribution and embedding.
This asks it different QUESTIONS -- a different label, a different context, a
different query -- or reads a different part of the same forward pass:

    attn      x07 x20  last-layer attention of each query row over the context rows:
                       label-weighted mean, entropy, largest weight, effective
                       number of rows, attention-weighted context age
    nbr       x10      context rows' embeddings as a neighbour base: kNN label
                       mean, distance to the nearest, top-vs-bottom prototype gap
    resid     x11      label = y minus the out-of-sample ridge (e118src_ridge_oos)
    crowd     x12      label = Numerai's meta model (exists from 2024-05-07 only)
    t60       x13      label = the 60-day target, behind a 60-date wall
    extreme   x14      label = |y - 0.5|
    now_mom   x15      label = the coin's feature_momentum_20d 20 dates later
    now_vol   x15      label = the coin's feature_volatility_20d 20 dates later
    self      x16      context = the coin's OWN past rows only
    regime    x17      context = past dates in today's BTC regime (trend sign x vol tercile)
    pair      x18      rows = coin pairs (feature differences -> label difference)
    drivers   x19      prediction change when each raw input moves up one bin

------------------------------------------------------------------------------
Causality -- the same wall as foundation_repr, per pass
------------------------------------------------------------------------------
Every pass predicts 20-date blocks from START on, and every block's context is
drawn from dates ending WALL periods before the block's first date:

    context dates  periods[b - WALL - window : b - WALL]   ->   predict periods[b : b + 20]

A context row's label must be KNOWN by date b. For y (20-day), |y - .5|, the
ridge residual and a feature 20 dates ahead that needs WALL = 20; for the 60-day
target WALL = 60. The meta model is published with the round, far inside the
wall. The ridge in `resid` is itself walk-forward out-of-sample (fitted on data
ending 20 periods before each of its own blocks). `regime` reads BTC's spot
closes dated <= D-1, the same submission lag as every spot feature. No row ever
sits in its own context, and nothing in a block reads a later date.

Live: each pass is one TabPFN fit per refresh on rows that are already labelled,
then predict on the round's rows -- the same shape as the backtest.
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd

from src.data_mani.foundation_repr import (CACHE_DIR, EMBARGO, MAX_CONTEXT, N_ESTIMATORS, REFRESH,
                                           SEED, START, TEMPERATURE, WINDOW, refresh_blocks)
from src.data_mani.utils import load_tournament_data
from src.prediction.functions import sorted_unique
from src.prediction.tabpfn_fm import TabPFNSkRegressor

TARGET = "target_binned_return_20"
TARGET60 = "target_binned_return_60"
PERIOD, ID = "date", "symbol"
FENC = "crypto/v2.0-fenccausal"
SPOT = os.path.join("data", "crypto", "spot", "ohlcv.parquet")
META = os.path.join("data", "crypto", "v2.0-fenccausal", "historical_meta_models.parquet")
RIDGE_OOS = os.path.join("results", "forecast", "crypto", "ridge_glmnet", "e118src_ridge_oos.csv")
CHUNK = 10
KNN = 50
SELF_WINDOW, SELF_MIN = 390, 60
REGIME_WINDOW, REGIME_MIN_DATES = 390, 20
PAIRS_PER_DATE, OPPONENTS = 32, 10
NUDGE = 0.25

PASSES = ["attn", "nbr", "resid", "crowd", "t60", "extreme", "now_mom", "now_vol",
          "self", "regime", "pair", "drivers"]


# --------------------------------------------------------------------------- #
# Panel and labels
# --------------------------------------------------------------------------- #
def load_panel():
    df, features = load_tournament_data(tournament="crypto", version=FENC, val_frac=0.3,
                                        embargo=EMBARGO, verbose=False, target=TARGET)
    df = df.sort_values([PERIOD, ID]).reset_index(drop=True)
    periods = sorted_unique(df[PERIOD].to_numpy())
    df["_pos"] = np.searchsorted(periods, df[PERIOD].to_numpy())
    raw = [f for f in features if not f.endswith("_symmean")]
    return df, features, raw, periods


def ahead(df, column, steps=EMBARGO):
    """The coin's own `column` value `steps` periods later (NaN when absent)."""
    key = df[[ID, "_pos", column]].copy()
    key["_pos"] = key["_pos"] - steps
    out = df[[ID, "_pos"]].merge(key, on=[ID, "_pos"], how="left")[column]
    return out.to_numpy()


def label_for(name, df):
    """(label array, wall) for the relabel passes; NaN = not usable as context."""
    y = df[TARGET].to_numpy(dtype="float64")
    if name in ("attn", "nbr", "self", "regime", "pair", "drivers"):
        return y, EMBARGO
    if name == "extreme":
        return np.abs(y - 0.5), EMBARGO
    if name == "t60":
        return df[TARGET60].to_numpy(dtype="float64"), 60
    if name == "now_mom":
        return ahead(df, "feature_momentum_20d"), EMBARGO
    if name == "now_vol":
        return ahead(df, "feature_volatility_20d"), EMBARGO
    if name == "crowd":
        meta = pd.read_parquet(META, columns=["symbol", "date", "prediction"])
        meta["date"] = pd.to_datetime(meta["date"])
        got = df[[ID, PERIOD]].merge(meta.rename(columns={"prediction": "_m"}), on=[ID, PERIOD], how="left")
        return got["_m"].to_numpy(dtype="float64"), EMBARGO
    if name == "resid":
        oos = pd.read_csv(RIDGE_OOS, usecols=[ID, PERIOD, "prediction"], parse_dates=[PERIOD])
        got = df[[ID, PERIOD]].merge(oos, on=[ID, PERIOD], how="left")
        frame = pd.DataFrame({PERIOD: df[PERIOD].to_numpy(), "y": y, "r": got["prediction"].to_numpy()})
        out = np.full(len(frame), np.nan)
        done = frame.dropna()
        for _, idx in done.groupby(PERIOD).indices.items():
            sub = done.iloc[idx]
            rank = sub["r"].rank(pct=True).to_numpy()
            X = np.column_stack([np.ones(len(sub)), rank])
            beta = np.linalg.lstsq(X, sub["y"].to_numpy(), rcond=None)[0]
            out[sub.index.to_numpy()] = sub["y"].to_numpy() - X @ beta
        return out, EMBARGO
    raise KeyError(name)


def regimes(periods):
    """BTC regime per panel date from spot closes dated <= D-1: trend sign x vol tercile.

    The tercile edges are EXPANDING quantiles of past dates' vol only, so a date's
    regime never depends on a later one.
    """
    spot = pd.read_parquet(SPOT, columns=["ucid", "date", "close"])
    btc = spot[spot["ucid"] == "1"].dropna().sort_values("date").set_index("date")["close"]
    logp = np.log(btc)
    ret = logp.diff()
    out = {}
    vols = []
    for d in pd.to_datetime(periods):
        upto = logp[logp.index <= d - pd.Timedelta(days=1)]
        r = ret[ret.index <= d - pd.Timedelta(days=1)]
        if len(upto) < 21:
            out[d] = np.nan
            continue
        trend = 1 if upto.iloc[-1] - upto.iloc[-21] > 0 else 0
        vol = r.iloc[-20:].std()
        tercile = 1 if len(vols) < 60 else int(np.searchsorted(np.quantile(vols, [1 / 3, 2 / 3]), vol))
        vols.append(vol)
        out[d] = trend * 3 + tercile
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def fit(X, y, device, n_estimators=N_ESTIMATORS):
    """E19's TabPFN on an already-chosen context (no further subsampling)."""
    est = TabPFNSkRegressor(n_estimators=n_estimators, max_context=None,
                            softmax_temperature=TEMPERATURE, device=device, random_state=SEED)
    return est.fit(X, y)


def subsample(idx, k=MAX_CONTEXT, seed=SEED):
    """foundation_repr's seeded, order-keeping subsample of context row indices."""
    if len(idx) <= k:
        return idx
    keep = np.random.default_rng(seed).choice(len(idx), size=k, replace=False)
    keep.sort()
    return idx[keep]


class LastLayerAttention:
    """
    Recompute the last layer's test->context attention of every estimator.

    In TabPFN v2 rows attend to rows once per feature column, and test rows read the
    first key/value head only (multi-query). The hook takes the layer's input,
    rebuilds q for the test rows and k for the context rows, and accumulates the
    softmax averaged over columns and heads: one (test rows x context rows) matrix
    per estimator, averaged over estimators at the end.
    """

    def __init__(self, regressor, n_context, row_chunk=512):
        self.n_context, self.row_chunk = n_context, row_chunk
        self.handles, self.calls = [], []
        for model in regressor.models_:
            layer = model.blocks[-1].per_column_attention_between_cells
            self.handles.append(layer.register_forward_hook(self._hook, with_kwargs=True))

    def _hook(self, module, args, kwargs, output):
        import torch
        x = args[0]
        n = kwargs.get("single_eval_pos")
        if n is None or n != self.n_context or x.shape[1] <= n:
            return
        with torch.no_grad():
            Bc, R, _ = x.shape
            acc = torch.zeros(R - n, n, device=x.device)
            # One column at a time: all columns x rows x heads x context at once
            # does not fit an 8 GB card next to the model.
            for c in range(Bc):
                k = module.k_projection(x[c, :n]).view(n, -1, module.head_dim)[:, 0].float()
                for s in range(n, R, self.row_chunk):
                    q = module.q_projection(x[c, s:s + self.row_chunk]).view(-1, module.num_heads, module.head_dim).float()
                    w = torch.softmax(torch.einsum("mhd,nd->mhn", q, k) / module.head_dim ** 0.5, dim=-1)
                    acc[s - n:s - n + q.shape[0]] += w.mean(dim=1)
            self.calls.append((acc / Bc).cpu().numpy())

    def matrix(self, n_rows, n_estimators):
        """(n_rows, n_context) attention, averaged over the estimators."""
        total = sum(len(c) for c in self.calls)
        if total != n_rows * n_estimators:
            raise RuntimeError("attention hook saw %d rows, expected %d x %d" % (total, n_rows, n_estimators))
        per = np.split(np.concatenate(self.calls), n_estimators)
        return np.mean(per, axis=0)

    def close(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #
def run(name, device, df, features, raw, periods, regime_of=None, max_blocks=None):
    label, wall = label_for(name, df)
    first = int(np.searchsorted(periods, np.datetime64(START)))
    pos = df["_pos"].to_numpy()
    rng = np.random.default_rng(SEED)
    by_symbol = df.groupby(ID).indices if name == "self" else None
    frames, started = [], time()

    for n, (_, block) in enumerate(refresh_blocks(periods, first, window=WINDOW, embargo=wall)):
        if max_blocks is not None and n >= max_blocks:
            break
        b = int(np.searchsorted(periods, block[0]))
        qmask = df[PERIOD].isin(block).to_numpy()
        q = df.loc[qmask]
        out = q[[ID, PERIOD]].reset_index(drop=True)
        ok = ~np.isnan(label)

        if name in ("attn", "nbr", "drivers", "resid", "crowd", "t60", "extreme", "now_mom", "now_vol"):
            cidx = np.where(ok & (pos >= b - wall - WINDOW) & (pos < b - wall))[0]
            if len(cidx) < 500:
                continue
            cidx = subsample(cidx)
            Xc, yc = df.loc[cidx, features].to_numpy(np.float32), label[cidx]
            est = fit(Xc, yc, device)
            Xq = q[features].to_numpy(np.float32)
            if name == "attn":
                vals = {k: [] for k in ("att_label", "att_entropy", "att_max", "att_neff", "att_age")}
                age = (b - pos[cidx]).astype("float64")
                for s in range(0, len(Xq), CHUNK * 300):
                    hook = LastLayerAttention(est._model, len(cidx))
                    try:
                        pred = est._model.predict(Xq[s:s + CHUNK * 300])
                        W = hook.matrix(len(Xq[s:s + CHUNK * 300]), est._model.n_estimators_)
                    finally:
                        hook.close()
                    W = W / W.sum(axis=1, keepdims=True)
                    vals["att_label"].append(W @ yc)
                    if s == 0 and n % 5 == 0:
                        print("    check: corr(attention-weighted label, prediction) = %.3f"
                              % np.corrcoef(W @ yc, pred)[0, 1], flush=True)
                    vals["att_entropy"].append(-(W * np.log(np.clip(W, 1e-12, None))).sum(axis=1))
                    vals["att_max"].append(W.max(axis=1))
                    vals["att_neff"].append(1.0 / (W ** 2).sum(axis=1))
                    vals["att_age"].append(W @ age)
                for k, v in vals.items():
                    out[k] = np.concatenate(v).astype("float32")
            elif name == "nbr":
                ec = np.asarray(est._model.get_embeddings(Xc, data_source="test")).mean(axis=0)
                eq = np.asarray(est._model.get_embeddings(Xq, data_source="test")).mean(axis=0)
                # Centre on the context mean first: raw embeddings share one large
                # common direction, which puts every cosine similarity near 1.
                centre = ec.mean(axis=0)
                ec, eq = ec - centre, eq - centre
                ec /= np.linalg.norm(ec, axis=1, keepdims=True)
                eq /= np.linalg.norm(eq, axis=1, keepdims=True)
                sim = eq @ ec.T
                top = np.argpartition(-sim, KNN, axis=1)[:, :KNN]
                out["nbr_label"] = yc[top].mean(axis=1).astype("float32")
                out["nbr_nearest"] = (1 - sim.max(axis=1)).astype("float32")
                hi, lo = ec[yc >= 0.75].mean(axis=0), ec[yc <= 0.25].mean(axis=0)
                hi, lo = hi / np.linalg.norm(hi), lo / np.linalg.norm(lo)
                out["nbr_proto"] = ((1 - eq @ lo) - (1 - eq @ hi)).astype("float32")
            elif name == "drivers":
                base = est.predict(Xq)
                for j, f in enumerate(raw):
                    c = features.index(f)
                    Xn = Xq.copy()
                    up = Xn[:, c] <= 1.0 - NUDGE + 1e-6
                    Xn[:, c] = np.where(up, Xn[:, c] + NUDGE, Xn[:, c] - NUDGE)
                    d = est.predict(Xn) - base
                    out["drv_%02d" % j] = np.where(up, d, -d).astype("float16")
            else:
                out[name] = est.predict(Xq).astype("float32")

        elif name == "self":
            preds = np.full(len(q), np.nan)
            qpos = q.index.to_numpy()
            for sym, rows in q.groupby(ID).indices.items():
                own = by_symbol[sym]
                cidx = own[ok[own] & (pos[own] >= b - wall - SELF_WINDOW) & (pos[own] < b - wall)]
                if len(cidx) < SELF_MIN or np.unique(label[cidx]).size < 2:
                    continue
                est = fit(df.loc[cidx, features].to_numpy(np.float32), label[cidx], device, n_estimators=2)
                preds[rows] = est.predict(df.loc[qpos[rows], features].to_numpy(np.float32))
            out["self"] = preds.astype("float32")

        elif name == "regime":
            reg = regime_of.reindex(pd.to_datetime(periods)).to_numpy()
            preds = np.full(len(q), np.nan)
            qreg = reg[q["_pos"].to_numpy()]
            for r in np.unique(qreg[~np.isnan(qreg)]):
                past = np.where((reg == r) & (np.arange(len(periods)) >= b - wall - REGIME_WINDOW)
                                & (np.arange(len(periods)) < b - wall))[0]
                if len(past) < REGIME_MIN_DATES:
                    continue
                cidx = subsample(np.where(ok & np.isin(pos, past))[0])
                est = fit(df.loc[cidx, features].to_numpy(np.float32), label[cidx], device)
                rows = np.where(qreg == r)[0]
                preds[rows] = est.predict(q.iloc[rows][features].to_numpy(np.float32))
            out["regime"] = preds.astype("float32")

        elif name == "pair":
            ctx_dates = np.arange(b - wall - WINDOW, b - wall)
            Xp, yp = [], []
            for p in ctx_dates:
                idx = np.where(ok & (pos == p))[0]
                if len(idx) < 2:
                    continue
                i = rng.choice(idx, PAIRS_PER_DATE); j = rng.choice(idx, PAIRS_PER_DATE)
                keep = i != j
                Xp.append(df.loc[i[keep], features].to_numpy(np.float32) - df.loc[j[keep], features].to_numpy(np.float32))
                yp.append(label[i[keep]] - label[j[keep]])
            Xp, yp = np.concatenate(Xp), np.concatenate(yp)
            sel = subsample(np.arange(len(yp)))
            est = fit(Xp[sel], yp[sel], device)
            wins = np.full(len(q), np.nan)
            Xq = q[features].to_numpy(np.float32)
            for _, rows in q.reset_index(drop=True).groupby(PERIOD).indices.items():
                opp = rng.choice(rows, (len(rows), OPPONENTS))
                same = opp == rows[:, None]
                opp[same] = rows[(np.searchsorted(rows, opp[same]) + 1) % len(rows)]
                diff = (Xq[rows][:, None, :] - Xq[opp]).reshape(-1, Xq.shape[1])
                wins[rows] = est.predict(diff).reshape(len(rows), OPPONENTS).mean(axis=1)
            out["pair"] = wins.astype("float32")

        out["block"] = n
        frames.append(out)
        if n % 5 == 0:
            print("  %s block %d: %s..%s, %.1f min" % (name, n, block[0], block[-1], (time() - started) / 60), flush=True)

    res = pd.concat(frames, ignore_index=True)
    path = os.path.join(CACHE_DIR, "x_%s%s.parquet" % (name, "_smoke" if max_blocks else ""))
    res.to_parquet(path, index=False)
    print("wrote %s: %d rows x %d cols, %.1f min" % (path, len(res), res.shape[1], (time() - started) / 60), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pass", dest="passes", required=True, help="comma-separated: %s" % ", ".join(PASSES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=None, help="smoke test: stop after N blocks")
    args = parser.parse_args()
    df, features, raw, periods = load_panel()
    regime_of = None
    for name in args.passes.split(","):
        if args.skip_done and os.path.exists(os.path.join(CACHE_DIR, "x_%s.parquet" % name)):
            print("%s on disk -- skip" % name)
            continue
        if name == "regime" and regime_of is None:
            regime_of = regimes(periods)
        print("== %s on %s" % (name, args.device), flush=True)
        run(name, args.device, df, features, raw, periods, regime_of, args.max_blocks)


if __name__ == "__main__":
    main()
