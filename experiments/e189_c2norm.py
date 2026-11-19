#!/usr/bin/env python3
"""
E189 -- how does normalisation change the value of frozen Chronos-2-Synth hidden states? (registered
in experiments/EXPERIMENTS.md and src/analysis/experiment_db.py before launch)

    python3 experiments/e189_c2norm.py --validate                 # pre-run checks -> validation.json
    python3 experiments/e189_c2norm.py --fit --arms N_CS_Z RAW    # fit arms (default: all 38 not yet written)
    python3 experiments/e189_c2norm.py --replay                   # post-fit live-path replay at the check dates
    python3 experiments/e189_c2norm.py --score                    # per-date routed + covered scores
    python3 experiments/e189_c2norm.py --report                   # tables, heatmaps, tests, HTML (then html_pdf)

Inputs: the frozen panel data/crypto/v2.0-asof-e189 and the cached cov-mode 768-d target-patch states in
/scratch/danielco/c2syn (verified bit-exact by experiments/e189_extract_verify.py). Model: the repo's
`ridge` (sklearn Ridge through H.fit_predict, intercept, no scaler), one pooled panel fit per arm and
block, alpha picked per block by H.pick's rule on a common 25-point grid with the registered boundary
expansion. Arms: N_<AXIS>_<TF> (5 x 7), RAW, B_ALL, B_COV.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e133_histfe_factorial as H  # noqa: E402
from src.prediction import embedding_norm as N  # noqa: E402

STUDY = "E189"
ROOT = "results/reports/crypto/e189_c2norm"
SCR = "/scratch/danielco/e189"
DATA = "data/crypto/v2.0-asof-e189/train.parquet"
META = "data/crypto/v2.0-asof-e189/historical_meta_models.parquet"
CACHE = "/scratch/danielco/c2syn"
KEYS = "results/reports/crypto/e132_feed_fillind/preds/keys.parquet"
T20 = H.T20
LAST = pd.Timestamp("2026-08-20")
TEST_START = H.TEST_START
GROUP_COL = "feature_market_cap_avg_20d"
ALPHAS = np.logspace(-3, 9, 25)
EXPAND, MAX_EXPAND = 6, 2
CHECK_DATES = ["2021-06-30", "2023-06-30", "2025-03-14", "2026-04-15"]
EMB_ARMS = ["N_%s_%s" % (a, t) for a in N.AXES for t in N.TRANSFORMS]
ARMS = EMB_ARMS + ["RAW", "B_ALL", "B_COV"]
COVERED_ARMS = EMB_ARMS + ["RAW", "B_COV"]


# ------------------------------------------------------------------------------------------ data
def load():
    """
    The panel <= LAST in (date, ucid) order -- the order the cache's row ids index -- with coverage,
    the row's position in the embedding matrix, the size group and the meta model.
    """
    df = pd.read_parquet(DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ucid"]).reset_index(drop=True)
    keys = pd.read_parquet(os.path.join(CACHE, "rows.parquet"))
    n0 = len(keys)
    assert (df.loc[:n0 - 1, ["symbol", "ucid"]].to_numpy() == keys[["symbol", "ucid"]].to_numpy()).all()
    assert (df.loc[:n0 - 1, "date"].to_numpy() == pd.to_datetime(keys["date"]).to_numpy()).all()
    df = df[df["date"] <= LAST].reset_index(drop=True)
    ids, reps = [], []
    for k in range(35):
        z = np.load(os.path.join(CACHE, "part_%03d.npz" % k))
        ids.append(z["cov_rows"])
        reps.append(z["cov_reps"])
    ids, reps = np.concatenate(ids), np.concatenate(reps)
    assert len(np.unique(ids)) == len(ids) and reps.shape[1] == 768
    keep = ids < len(df)
    ids, reps = ids[keep], reps[keep]
    order = np.argsort(ids)
    ids, emb = ids[order], reps[order].astype(np.float32)
    assert np.isfinite(emb).all()
    df["covered"] = False
    df.loc[ids, "covered"] = True
    df["emb_row"] = -1
    df.loc[ids, "emb_row"] = np.arange(len(ids))
    feats = sorted(c for c in df if c.startswith("feature_"))
    assert len(feats) == 22
    g = df[GROUP_COL].to_numpy()
    assert np.isin(g, N.BIN_LABELS).all()
    df["group"] = np.where(g <= 0.25, 0, np.where(g == 0.5, 1, 2)).astype(np.int8)
    df[feats] = df[feats].fillna(0.5)                      # shipped-feature convention, B_* arms only
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    df = df.merge(meta, on=["symbol", "date"], how="left", validate="1:1")
    return df, emb, feats


def schedule(df):
    """E131's 11 dev blocks + the confirmation block, as (block, train dates, predict dates, kind)."""
    past = np.array(sorted(df.loc[df["date"] < TEST_START, "date"].unique()), dtype="datetime64[ns]")
    k = pd.read_parquet(KEYS, columns=["date", "kind", "block"]).drop_duplicates()
    k["date"] = pd.to_datetime(k["date"])
    test = np.array(sorted(k.loc[k["kind"] == "test", "date"].unique()), dtype="datetime64[ns]")
    assert len(test) == H.BLOCK and test[0] == TEST_START and test[-1] == LAST.to_datetime64()
    B = [(i, tr, te, "dev") for i, tr, te in H.blocks(past)] + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], test, "test")]
    kb = k.set_index("date")["block"]
    for i, _, te, _ in B:                                   # same blocks as every E131-E146 study
        assert (kb.reindex(te).to_numpy() == i).all(), i
    return B


def inner_dates(tr):
    """H.pick's split: fit on train[:-(40 + 22)], validate on the last 40 train dates."""
    return tr[:-(H.INNER_VAL + H.PURGE)], tr[-H.INNER_VAL:]


# ---------------------------------------------------------------------------------- features
class Features:
    """Builds an arm's matrix for any set of rows of one fold, with the arm's reference rule."""

    def __init__(self, arm, df, emb, feats):
        self.arm, self.df, self.emb, self.feats = arm, df, emb, feats
        self.axis = self.tf = None
        self.snap = None
        self.fallback = 0
        if arm.startswith("N_"):
            _, self.axis, self.tf = arm.split("_")
            if self.axis in N.SNAPSHOT_AXES:
                cov = df["covered"].to_numpy()
                X = emb[df.loc[cov, "emb_row"].to_numpy()]
                assert (df.loc[cov].groupby("date").size() >= N.MIN_CS).all()
                out, self.fallback = N.snapshot(self.tf, self.axis, X, df.loc[cov, "date"].to_numpy(),
                                                df.loc[cov, "group"].to_numpy())
                self.snap = np.full((len(emb), emb.shape[1]), np.nan, np.float32)
                self.snap[df.loc[cov, "emb_row"].to_numpy()] = out

    def eligible(self, rows):
        return rows if self.arm == "B_ALL" else rows & self.df["covered"].to_numpy()

    def build(self, fit_rows, apply_rows):
        """
        :param fit_rows: boolean mask of the fold's training rows (eligible only)
        :param apply_rows: boolean mask of the rows to transform (fit rows + val/test rows)
        :return: float64 matrix for apply_rows (in panel order), info
        """
        d = self.df
        if self.arm in ("B_ALL", "B_COV"):
            return d.loc[apply_rows, self.feats].to_numpy(np.float64), {}
        assert d.loc[apply_rows, "covered"].all() and d.loc[fit_rows, "covered"].all()
        er = d.loc[apply_rows, "emb_row"].to_numpy()
        if self.arm == "RAW":
            return self.emb[er].astype(np.float64), {}
        if self.snap is not None:
            return self.snap[er].astype(np.float64), {"reference": "snapshot", "grp_fallback_rows_total": int(self.fallback)}
        fit_in = fit_rows[apply_rows]
        out, info = N.train_referenced(self.tf, self.axis, self.emb[er], d.loc[apply_rows, "ucid"].to_numpy(), fit_in,
                                       np.ones(len(er), bool))
        return out, info


def date_spearman(dates, pred, y):
    """Mean over dates of the per-date Spearman correlation (H.pick's criterion), vectorised."""
    f = pd.DataFrame({"d": dates, "p": pred, "y": y})
    r = f.groupby("d")[["p", "y"]].rank()
    r["d"] = f["d"].to_numpy()
    s = r.groupby("d").corr().xs("p", level=1)["y"]
    return float(s.mean())


def ridge(alpha, Xtr, ytr, Xte):
    return H.fit_predict("ridge", {"alpha": float(alpha), "fit_intercept": True}, 0, 1, Xtr, ytr, Xte)


def tune(Fx, d, tr):
    """H.pick on the arm's own inner fold, references refitted on the inner fit dates only."""
    fd, vd = inner_dates(tr)
    dates = d["date"].to_numpy()
    fit = Fx.eligible(np.isin(dates, fd))
    val = Fx.eligible(np.isin(dates, vd))
    app = fit | val
    X, info = Fx.build(fit, app)
    fi, vi = fit[app], val[app]
    assert d.loc[fit, "date"].max() < d.loc[val, "date"].min() - pd.Timedelta(days=H.PURGE)   # purge holds
    ytr, yv, dv = d.loc[fit, T20].to_numpy(), d.loc[val, T20].to_numpy(), d.loc[val, "date"].to_numpy()
    scores = {}

    def run(grid):
        for a in grid:
            if a not in scores:
                scores[a] = date_spearman(dv, ridge(a, X[fi], ytr, X[vi]), yv)

    grid = list(ALPHAS)
    run(grid)
    expansions = 0
    while True:
        best = max(sorted(scores), key=lambda a: scores[a])        # first max on an ascending grid
        lo, hi = min(scores), max(scores)
        if best not in (lo, hi) or expansions == MAX_EXPAND:
            break
        step = 10 ** 0.5
        new = [lo / step ** k for k in range(1, EXPAND + 1)] if best == lo else [hi * step ** k for k in range(1, EXPAND + 1)]
        run(new)
        expansions += 1
    best = max(sorted(scores), key=lambda a: scores[a])
    meta = dict(alpha=best, boundary=bool(best in (min(scores), max(scores))), expansions=expansions,
                inner_scores={"%.6g" % a: scores[a] for a in sorted(scores)}, inner_fit_dates=[str(fd[0])[:10], str(fd[-1])[:10]],
                inner_val_dates=[str(vd[0])[:10], str(vd[-1])[:10]], n_inner_fit=int(fit.sum()), n_inner_val=int(val.sum()),
                inner_ref_rows=int(info["n_fit"]) if "n_fit" in info else None)
    assert info.get("n_fit", fit.sum()) == fit.sum()                # train references saw the fit rows only
    return meta, info


def save_params(arm, block, fold, info):
    if "pool" not in info:
        return
    path = os.path.join(SCR, "params", arm, "block%02d_%s.npz" % (block, fold))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {"pool_" + k: v for k, v in info["pool"].summary().items()}
    if info["assets"]:
        ucids = sorted(info["assets"])
        out["asset_ucid"] = np.array(ucids)
        for k in info["assets"][ucids[0]].summary():
            out["asset_" + k] = np.stack([info["assets"][u].summary()[k] for u in ucids])
    np.savez_compressed(path, **out)


def fit_arm(arm, df, emb, feats, B):
    path = os.path.join(ROOT, "preds", arm + ".parquet")
    if os.path.exists(path):
        print("skip", arm, flush=True)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    Fx = Features(arm, df, emb, feats)
    dates = df["date"].to_numpy()
    parts, picks = [], []
    for i, tr, te, kind in B:
        t1 = time.time()
        meta, inner_info = tune(Fx, df, tr)
        save_params(arm, i, "inner", inner_info)
        fit = Fx.eligible(np.isin(dates, tr))
        tst = Fx.eligible(np.isin(dates, te))
        app = fit | tst
        X, info = Fx.build(fit, app)
        save_params(arm, i, "outer", info)
        fi, ti = fit[app], tst[app]
        Xtr = X[fi]
        pred = ridge(meta["alpha"], Xtr, df.loc[fit, T20].to_numpy(), X[ti])
        assert np.isfinite(pred).all()
        o = df.loc[np.isin(dates, te), ["symbol", "ucid", "date", T20, "meta_model", "covered", "group"]].copy()
        o["prediction"] = np.nan
        o.loc[tst[np.isin(dates, te)], "prediction"] = pred
        o["block"], o["kind"] = i, kind
        parts.append(o)
        var = Xtr.var(0)
        meta.update(block=i, kind=kind, n_train=int(fit.sum()), n_test=int(tst.sum()), mean_col_var=float(var.mean()),
                    alpha_over_n_var=float(meta["alpha"] / (fit.sum() * max(var.mean(), 1e-300))),
                    ts_fallback_assets=info.get("fallback_assets"), ts_fallback_rows=info.get("fallback_rows"),
                    grp_fallback_rows_total=info.get("grp_fallback_rows_total"), seconds=time.time() - t1)
        picks.append(meta)
        print("%s block %2d alpha %.3g%s  %d/%d rows  %.0fs" % (arm, i, meta["alpha"], " BOUNDARY" if meta["boundary"] else "",
                                                             meta["n_train"], meta["n_test"], meta["seconds"]), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path + ".partial", index=False)
    json.dump({"arm": arm, "picks": picks, "seconds": time.time() - t0, "host": os.uname().nodename},
              open(os.path.join(ROOT, "preds", arm + ".picks.json"), "w"), indent=1, default=float)
    os.replace(path + ".partial", path)
    print("%s done in %.1f min" % (arm, (time.time() - t0) / 60), flush=True)


# ---------------------------------------------------------------------------------- validation
def validate():
    df, emb, feats = load()
    B = schedule(df)
    res = {}
    v = json.load(open(os.path.join(SCR, "verify", "verify.json")))
    res["extraction"] = v
    res["extraction_ok"] = all(r[k]["max_abs"] == 0.0 and r[k]["not_in_cache"] == 0 and r[k]["width"] == 768
                               for r in v["dates"].values() for k in ("full", "cut")) and \
        all(r["same_covered_set"] and r["cut_same_covered_set"] and r["rerun_identical"] for r in v["dates"].values())
    cov = df["covered"].to_numpy()
    res["keys"] = dict(rows=len(df), covered=int(cov.sum()), covered_share=float(cov.mean()), width=int(emb.shape[1]),
                       finite=bool(np.isfinite(emb).all()), one_vector_per_key=bool(len(emb) == cov.sum()),
                       min_covered_per_date_scored=int(df[df["date"] >= B[0][2][0]].groupby("date")["covered"].sum().min()))
    # group sizes and fallback
    gs = df[cov].groupby(["date", "group"]).size()
    res["groups"] = dict(min_cell=int(gs.min()), cells_below_min=int((gs < N.MIN_GROUP).sum()),
                         mean_share={int(k): float(x) for k, x in df[cov].groupby("group").size().div(cov.sum()).items()})
    # cut-date replay of transformed features: states from the cut panel through each axis
    rep = {}
    dates = df["date"].to_numpy()
    for dstr in CHECK_DATES:
        T = df.loc[df["date"] <= dstr, "date"].max()
        z = np.load(os.path.join(SCR, "verify", "cut_%s.npz" % T.date()))
        blk = next(b for b in B if T.to_datetime64() in b[2])
        rows_T = (dates == T.to_datetime64()) & cov
        er = df.loc[rows_T, "emb_row"].to_numpy()
        assert np.array_equal(np.sort(z["ids"]), np.flatnonzero(rows_T))
        cut = z["reps"][np.argsort(z["ids"])].astype(np.float32)
        diffs = {}
        for arm in EMB_ARMS:
            ax, tf = arm.split("_")[1:]
            if ax in N.SNAPSHOT_AXES:
                a = N.snapshot(tf, ax, emb[er], dates[rows_T], df.loc[rows_T, "group"].to_numpy())[0]
                b = N.snapshot(tf, ax, cut, dates[rows_T], df.loc[rows_T, "group"].to_numpy())[0]
            else:
                tr_rows = np.isin(dates, blk[1]) & cov
                Xr = emb[df.loc[tr_rows, "emb_row"].to_numpy()]
                assets = np.r_[df.loc[tr_rows, "ucid"].to_numpy(), df.loc[rows_T, "ucid"].to_numpy()]
                fitm = np.r_[np.ones(len(Xr), bool), np.zeros(len(er), bool)]
                a = N.train_referenced(tf, ax, np.vstack([Xr, emb[er]]), assets, fitm, ~fitm)[0][~fitm]
                b = N.train_referenced(tf, ax, np.vstack([Xr, cut]), assets, fitm, ~fitm)[0][~fitm]
            diffs[arm] = float(np.abs(a - b).max())
        rep[str(T.date())] = dict(block=int(blk[0]), max_abs_diff=max(diffs.values()), per_arm=diffs)
    res["cut_replay_features"] = rep
    # future perturbation on the real panel: scramble every state after T; nothing on or before T may move
    T = pd.Timestamp("2023-06-30")
    blk = next(b for b in B if T.to_datetime64() in b[2])
    rng = np.random.default_rng(189)
    emb2 = emb.copy()
    fut = df.loc[cov & (dates > T.to_datetime64()), "emb_row"].to_numpy()
    emb2[fut] = rng.normal(size=(len(fut), emb.shape[1])).astype(np.float32) * 10
    keep = cov & (dates <= T.to_datetime64()) & (dates >= blk[1][0])
    fp = {}
    for arm in EMB_ARMS:
        ax, tf = arm.split("_")[1:]
        er = df.loc[keep, "emb_row"].to_numpy()
        if ax in N.SNAPSHOT_AXES:
            a = N.snapshot(tf, ax, emb[er], dates[keep], df.loc[keep, "group"].to_numpy())[0]
            b = N.snapshot(tf, ax, emb2[er], dates[keep], df.loc[keep, "group"].to_numpy())[0]
        else:
            fitm = np.isin(dates[keep], blk[1])
            a = N.train_referenced(tf, ax, emb[er], df.loc[keep, "ucid"].to_numpy(), fitm, np.ones(len(er), bool))[0]
            b = N.train_referenced(tf, ax, emb2[er], df.loc[keep, "ucid"].to_numpy(), fitm, np.ones(len(er), bool))[0]
        fp[arm] = bool(np.array_equal(a, b))
    res["future_perturbation_invariant"] = dict(date=str(T.date()), all=all(fp.values()), per_arm=fp)
    # uncovered rows cannot reach a statistic: Features.build refuses them, and CS on a date ignores them
    Fx = Features("N_POOL_Z", df, emb, feats)
    tr = B[3][1]
    try:
        Fx.build(np.isin(dates, tr), np.isin(dates, tr))          # includes uncovered rows -> must refuse
        res["uncovered_refused"] = False
    except AssertionError:
        res["uncovered_refused"] = True
    # inner-fold references exclude validation dates (the purge and split rule)
    res["inner_split"] = [dict(block=i, fit_last=str(inner_dates(tr)[0][-1])[:10], val_first=str(inner_dates(tr)[1][0])[:10],
                               gap_dates=int(np.searchsorted(tr, inner_dates(tr)[1][0]) - np.searchsorted(tr, inner_dates(tr)[0][-1]) - 1))
                          for i, tr, _, _ in B]
    assert all(s["gap_dates"] == H.PURGE for s in res["inner_split"])
    res["schedule"] = [dict(block=i, kind=k, train=[str(tr[0])[:10], str(tr[-1])[:10]], predict=[str(te[0])[:10], str(te[-1])[:10]])
                       for i, tr, te, k in B]
    res["ok"] = bool(res["extraction_ok"] and res["keys"]["finite"] and res["keys"]["one_vector_per_key"]
                     and max(r["max_abs_diff"] for r in rep.values()) == 0.0 and res["future_perturbation_invariant"]["all"]
                     and res["uncovered_refused"])
    os.makedirs(ROOT, exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "validation.json"), "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("extraction_ok", "keys", "groups", "uncovered_refused", "ok")}, indent=1, default=str))
    print("cut replay max diff", {k: r["max_abs_diff"] for k, r in rep.items()})
    print("future perturbation invariant:", res["future_perturbation_invariant"]["all"])


def replay():
    """
    The live prediction path at each check date, from the CUT panel's states: transform, refit the
    block's ridge at its stored alpha on the block's training rows, predict date T, compare with the
    stored historical prediction.
    """
    df, emb, feats = load()
    B = schedule(df)
    dates = df["date"].to_numpy()
    out = {}
    for dstr in CHECK_DATES:
        T = df.loc[df["date"] <= dstr, "date"].max()
        z = np.load(os.path.join(SCR, "verify", "cut_%s.npz" % T.date()))
        cut = z["reps"][np.argsort(z["ids"])].astype(np.float32)
        blk = next(b for b in B if T.to_datetime64() in b[2])
        # what the live path holds at T: the block's training rows and date T's cut-panel states
        sub = df[np.isin(dates, blk[1]) | (dates == T.to_datetime64())].reset_index(drop=True)
        semb = emb[sub.loc[sub["covered"], "emb_row"].to_numpy()].copy()
        sub.loc[sub["covered"], "emb_row"] = np.arange(len(semb))
        sd = sub["date"].to_numpy()
        rows_T = (sd == T.to_datetime64()) & sub["covered"].to_numpy()
        semb[sub.loc[rows_T, "emb_row"].to_numpy()] = cut
        r = {}
        for arm in COVERED_ARMS:
            picks = json.load(open(os.path.join(ROOT, "preds", arm + ".picks.json")))["picks"]
            alpha = next(p["alpha"] for p in picks if p["block"] == blk[0])
            Fx = Features(arm, sub, semb, feats)
            fit = Fx.eligible(np.isin(sd, blk[1]))
            app = fit | rows_T
            X, _ = Fx.build(fit, app)
            pred = ridge(alpha, X[fit[app]], sub.loc[fit, T20].to_numpy(), X[rows_T[app]])
            st = pd.read_parquet(os.path.join(ROOT, "preds", arm + ".parquet"))
            st = st[(st["date"] == T) & st["covered"]].sort_values("ucid")
            mine = pd.Series(pred, index=sub.loc[rows_T, "ucid"].to_numpy()).sort_index()
            r[arm] = float(np.abs(mine.to_numpy() - st["prediction"].to_numpy()).max())
        out[str(T.date())] = dict(block=int(blk[0]), max_abs_diff=max(r.values()), per_arm=r)
        print(T.date(), "max |live replay - stored|", max(r.values()), flush=True)
    json.dump(out, open(os.path.join(ROOT, "replay.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------- scoring
def _score(arm):
    import e131_analysis as E
    import e134_route_feature_sets as R
    b = pd.read_parquet(os.path.join(ROOT, "preds", "B_ALL.parquet"))
    c = pd.read_parquet(os.path.join(ROOT, "preds", arm + ".parquet"))
    assert (b[["symbol", "date"]].to_numpy() == c[["symbol", "date"]].to_numpy()).all()
    g = b.rename(columns={"prediction": "b0"})
    g["cand"] = c["prediction"].to_numpy() if arm != "B_ALL" else b["prediction"].to_numpy()
    if arm != "B_ALL":
        assert c["prediction"].notna().to_numpy().sum() == c["covered"].sum() and c.loc[c["covered"], "prediction"].notna().all()
    rows = []
    for d, h in g.groupby("date", sort=True):
        h = h.reset_index(drop=True)
        cov = h["covered"].to_numpy()
        bb = h["b0"].rank(pct=True).to_numpy()
        if arm == "B_ALL":
            x = bb
        else:
            x = R.slot(bb, h["cand"].to_numpy(), cov)
            assert np.array_equal(x[~cov], bb[~cov])                         # uncovered keep their exact ranks
            assert np.array_equal(np.sort(x[cov]), np.sort(bb[cov]))         # covered slot set preserved
        hr = h.assign(prediction=x).set_index("symbol")
        corr = E.corr_one(hr)
        mmc = E.contrib_one(hr, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
        k = h[cov].assign(prediction=h.loc[cov, "cand"].to_numpy()).set_index("symbol")
        kb = h[cov].assign(prediction=h.loc[cov, "b0"].to_numpy()).set_index("symbol")
        ccorr = E.corr_one(k)
        cmmc = E.contrib_one(k, "meta_model") if k["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(arm=arm, date=d, block=int(h["block"].iloc[0]), kind=h["kind"].iloc[0], n=len(h), n_cov=int(cov.sum()),
                         corr=corr, mmc=mmc, payout=0.1 * corr + mmc if mmc == mmc else np.nan,
                         cov_corr=ccorr, cov_mmc=cmmc, cov_payout=0.1 * ccorr + cmmc if cmmc == cmmc else np.nan,
                         cov_corr_ball=E.corr_one(kb)))
    return pd.DataFrame(rows)


def score(workers):
    missing = [a for a in ARMS if not os.path.exists(os.path.join(ROOT, "preds", a + ".parquet"))]
    assert not missing, missing
    ref = pd.read_parquet(os.path.join(ROOT, "preds", "RAW.parquet"), columns=["symbol", "date", "block", "covered"])
    for a in ARMS:                                       # identical scored keys, folds and coverage everywhere
        x = pd.read_parquet(os.path.join(ROOT, "preds", a + ".parquet"), columns=["symbol", "date", "block", "covered", "prediction"])
        assert x[["symbol", "date", "block", "covered"]].equals(ref), a
        if a in COVERED_ARMS:
            assert (x["prediction"].notna() == x["covered"]).all(), a
    with ProcessPoolExecutor(workers) as ex:
        S = pd.concat(ex.map(_score, ARMS), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores.parquet"), index=False)
    print(S.groupby(["arm", "kind"])[["corr", "payout", "cov_corr"]].mean().unstack().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.validate:
        return validate()
    if a.replay:
        return replay()
    if a.score:
        return score(a.workers)
    if a.report:
        import e189_report
        return e189_report.main()
    if a.fit:
        df, emb, feats = load()
        B = schedule(df)
        for arm in (a.arms or ARMS):
            assert arm in ARMS, arm
            fit_arm(arm, df, emb, feats, B)


if __name__ == "__main__":
    main()
