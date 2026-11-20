#!/usr/bin/env python3
"""
E191 -- cross-asset variants of the frozen Chronos-2-Synth state (registered in experiments/EXPERIMENTS.md and
src/analysis/experiment_db.py before launch). E190 design, own state replaced by:

    G_<T>   2'  ridge   [v2.0 | T(own) | T(leave-one-out mean of other coins in the date x size group)]
    L_*     2'' lgb_ktz [v2.0 | T(own - leave-one-out mean of all other coins on the date)] + lgb controls
    M_<T>   3'  ridge   [v2.0 | T(own) | 16 market PCs | 32 own PCs x 16 market PCs]

    python3 experiments/e191_c2cross.py --fit --arms G_N_TS_MM ...
    python3 experiments/e191_c2cross.py --score
    python3 experiments/e191_c2cross.py --report
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e190_c2norm_v2 as V  # noqa: E402  (patches e189_c2norm -> M)

M = V.M
H = M.H
E190_ROOT = M.ROOT
M.STUDY, M.ROOT, M.SCR = "E191", "results/reports/crypto/e191_c2cross", "/scratch/danielco/e191"
TFS = ["N_TS_MM", "N_GRP_MM", "N_POOL_MM", "RAW"]
G_ARMS = ["G_" + t for t in TFS]
M_ARMS = ["M_" + t for t in TFS]
L_ARMS = ["L_B_COV", "L_OWN_RAW", "L_REL_RAW", "L_REL_N_TS_MM"]
NEW = G_ARMS + L_ARMS + M_ARMS
REUSED = {"B_ALL": V.E189_ROOT, "B_COV": V.E189_ROOT}
REUSED.update({"V_" + t: E190_ROOT for t in TFS})
K_OWN, K_MKT = 32, 16
LGB_THREADS = int(os.environ.get("E191_THREADS", "8"))


# ------------------------------------------------------------------------------ derived states
def loo_mean(emb, df, by):
    """Mean of the OTHER covered rows' raw states in each `by` cell, per covered row (emb_row order)."""
    cov = df[df["covered"]]
    er = cov["emb_row"].to_numpy()
    key = cov.groupby(by, sort=False).ngroup().to_numpy()
    X = emb[er].astype(np.float64)
    S = np.zeros((key.max() + 1, emb.shape[1]))
    np.add.at(S, key, X)
    n = np.bincount(key)
    assert n.min() >= 2
    out = np.empty_like(emb)
    out[er] = ((S[key] - X) / (n[key] - 1)[:, None]).astype(np.float32)
    return out


def date_mean(emb, df):
    cov = df[df["covered"]]
    g = pd.DataFrame(emb[cov["emb_row"].to_numpy()].astype(np.float64)).groupby(cov["date"].to_numpy()).mean()
    return g                                                        # index = date, 768 cols


class Factory:
    """Builds per-arm Features objects; fit_arm calls M.Features(arm, df, emb, feats)."""

    def __init__(self, df, emb):
        self.df, self.emb = df, emb
        self.cache = {}

    def mat(self, kind):
        if kind not in self.cache:
            if kind == "G":
                self.cache[kind] = np.hstack([self.emb, loo_mean(self.emb, self.df, ["date", "group"])])
            elif kind == "REL":
                self.cache[kind] = self.emb - loo_mean(self.emb, self.df, ["date"])
            elif kind == "MKT":
                self.cache[kind] = date_mean(self.emb, self.df)
        return self.cache[kind]

    def __call__(self, arm, df, emb, feats):
        fam, rest = arm.split("_", 1)
        if fam == "G":
            return V.Features("V_" + rest, df, self.mat("G"), feats)
        if fam == "M":
            return MarketFeatures(rest, df, emb, feats, self.mat("MKT"))
        if fam == "L":
            if rest == "B_COV":
                return V.Features("B_COV", df, emb, feats)
            if rest == "OWN_RAW":
                return V.Features("V_RAW", df, emb, feats)
            return V.Features("V_" + rest[4:], df, self.mat("REL"), feats)
        return V.Features(arm, df, emb, feats)


def _pca(X, k):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:k].T
    Z = (X - mu) @ W
    sd = Z.std(0)
    sd[sd == 0] = 1
    return mu, W, sd


class MarketFeatures(V.Features):
    """[v2.0 | T(own) | market PCs | own PCs x market PCs]; both PCAs fitted on the fold's fit rows only."""

    def __init__(self, rest, df, emb, feats, mkt):
        super().__init__("V_" + rest, df, emb, feats)
        self.mkt = mkt

    def build(self, fit_rows, apply_rows):
        X0, info = super().build(fit_rows, apply_rows)
        fi = fit_rows[apply_rows]
        own = X0[:, 22:]
        mu, W, sd = _pca(own[fi], K_OWN)
        P = ((own - mu) @ W) / sd
        dates = self.df.loc[apply_rows, "date"].to_numpy()
        fit_dates = np.unique(dates[fi])
        Mk = self.mkt.loc[fit_dates].to_numpy()
        mm, Wm, _ = _pca(Mk, K_MKT)
        Q = (self.mkt.loc[dates].to_numpy() - mm) @ Wm
        qs = Q[fi].std(0)
        qs[qs == 0] = 1
        Q = Q / qs
        inter = (P[:, :, None] * Q[:, None, :]).reshape(len(P), -1)
        return np.hstack([X0, Q, inter]), info


# ------------------------------------------------------------------------------ lgb fitting
def lgb_arm(arm, df, emb, feats, B):
    path = os.path.join(M.ROOT, "preds", arm + ".parquet")
    if os.path.exists(path):
        print("skip", arm, flush=True)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    Fx = M.Features(arm, df, emb, feats)
    dates = df["date"].to_numpy()
    parts, picks = [], []
    cands = H.candidates("lgb_ktz")
    for i, tr, te, kind in B:
        t1 = time.time()
        fd, vd = M.inner_dates(tr)
        fit, val = Fx.eligible(np.isin(dates, fd)), Fx.eligible(np.isin(dates, vd))
        app = fit | val
        X, _ = Fx.build(fit, app)
        X = X.astype(np.float32)
        scores = []
        for c in cands:
            p = H.fit_predict("lgb_ktz", c, 0, LGB_THREADS, X[fit[app]], df.loc[fit, M.T20].to_numpy(), X[val[app]])
            scores.append(M.date_spearman(df.loc[val, "date"].to_numpy(), p, df.loc[val, M.T20].to_numpy()))
        best = cands[int(np.argmax(scores))]
        fit = Fx.eligible(np.isin(dates, tr))
        tst = Fx.eligible(np.isin(dates, te))
        app = fit | tst
        X, _ = Fx.build(fit, app)
        X = X.astype(np.float32)
        pred = H.fit_predict("lgb_ktz", best, 0, LGB_THREADS, X[fit[app]], df.loc[fit, M.T20].to_numpy(), X[tst[app]])
        o = df.loc[np.isin(dates, te), ["symbol", "ucid", "date", M.T20, "meta_model", "covered", "group"]].copy()
        o["prediction"] = np.nan
        o.loc[tst[np.isin(dates, te)], "prediction"] = pred
        o["block"], o["kind"] = i, kind
        parts.append(o)
        picks.append(dict(block=i, kind=kind, num_leaves=best["num_leaves"], colsample_bytree=best["colsample_bytree"],
                          inner_scores=scores, alpha=float("nan"), seconds=time.time() - t1))
        print("%s block %2d leaves %d col %.2f  %.0fs" % (arm, i, best["num_leaves"], best["colsample_bytree"],
                                                         time.time() - t1), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(path + ".partial", index=False)
    json.dump({"arm": arm, "picks": picks, "seconds": time.time() - t0, "host": os.uname().nodename},
              open(os.path.join(M.ROOT, "preds", arm + ".picks.json"), "w"), indent=1, default=float)
    os.replace(path + ".partial", path)
    print("%s done in %.1f min" % (arm, (time.time() - t0) / 60), flush=True)


def link_reused():
    d = os.path.join(M.ROOT, "preds")
    os.makedirs(d, exist_ok=True)
    for a, root in list(REUSED.items()) + [("RAW", V.E189_ROOT)]:      # RAW: score()'s key reference
        for ext in (".parquet", ".picks.json"):
            dst = os.path.join(d, a + ext)
            if not os.path.lexists(dst):
                try:
                    os.symlink(os.path.abspath(os.path.join(root, "preds", a + ext)), dst)
                except FileExistsError:
                    pass


M.ARMS = NEW + list(REUSED)
M.COVERED_ARMS = [a for a in M.ARMS if a != "B_ALL"]


# ------------------------------------------------------------------------------ report
def report():
    import scipy.stats as st
    import e131_analysis as E
    S = pd.read_parquet(os.path.join(M.ROOT, "scores.parquet"))
    rows, tests = [], []
    pairs = {**{"G_" + t: ["B_COV", "V_" + t] for t in TFS}, **{"M_" + t: ["B_COV", "V_" + t] for t in TFS},
             "L_B_COV": ["B_COV"], "L_OWN_RAW": ["B_COV", "L_B_COV", "V_RAW"],
             "L_REL_RAW": ["B_COV", "L_B_COV", "L_OWN_RAW"], "L_REL_N_TS_MM": ["B_COV", "L_B_COV", "L_OWN_RAW"]}
    for kind in ("dev", "test"):
        s = S[S["kind"] == kind]
        P = s.pivot(index="date", columns="arm", values="payout").dropna()
        C = s.pivot(index="date", columns="arm", values="corr")
        Mm = s.pivot(index="date", columns="arm", values="mmc").loc[P.index]
        for a in M.ARMS:
            cum = P[a].cumsum()
            rows.append(dict(kind=kind, arm=a, corr=C[a].mean(), mmc=Mm[a].mean(), payout=P[a].mean(),
                             pay_sharpe=P[a].mean() / P[a].std(ddof=0), pay_dd=float((cum.cummax() - cum).max()),
                             n_corr=int(C[a].notna().sum()), n_pay=len(P)))
        for a, refs in pairs.items():
            for ref in refs:
                for met, X in (("corr", C), ("payout", P)):
                    d = (X[a] - X[ref]).dropna().to_numpy()
                    m, _, t = E.newey_west(d)
                    tests.append(dict(kind=kind, arm=a, vs=ref, metric=met, delta=m, t=t, n=len(d),
                                      p=float(2 * (1 - st.norm.cdf(abs(t))))))
    T, K = pd.DataFrame(rows), pd.DataFrame(tests)
    K["family"] = np.where(K["vs"] == "B_COV", "vs B_COV", "vs own-state control")
    for _, g in K.groupby(["kind", "family", "metric"]):
        o = g["p"].sort_values()
        K.loc[o.index, "holm"] = np.maximum.accumulate([min(1, p * (len(o) - i)) for i, p in enumerate(o)])
    T.to_csv(os.path.join(M.ROOT, "table.csv"), index=False)
    K.to_csv(os.path.join(M.ROOT, "tests.csv"), index=False)
    f = lambda x: "%.4f" % x  # noqa: E731
    html = ["<html><head><meta charset='utf-8'><title>E191 cross-asset states</title><style>"
            "body{font-family:sans-serif;margin:24px;font-size:12px}table{border-collapse:collapse;margin:8px 0 18px}"
            "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#eee}</style></head><body>",
            "<h1>E191: cross-asset variants of the Chronos-2-Synth state</h1>",
            "<p>G_* = v2.0 + own + leave-one-out size-group mean (ridge). L_* = lgb_ktz: v2.0 only, v2.0+own, "
            "v2.0 + (own - leave-one-out all-coin mean). M_* = v2.0 + own + 16 market PCs + 32x16 own-PC x market-PC "
            "products (ridge). V_* = E190 own-state arms, B_* = E189 baselines. Routed R_SLOT; CORR on all dates, "
            "MMC/payout/sharpe/dd on dates with the meta model. Dev = selection; test 2026-02-20..08-20 = confirmation.</p>"]
    for kind in ("dev", "test"):
        html += ["<h2>%s, sorted by payout</h2>" % kind,
                 T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind").to_html(index=False, float_format=f),
                 "<h3>%s paired Newey-West lag 20, Holm within family</h3>" % kind,
                 K[K["kind"] == kind].drop(columns="kind").to_html(index=False, float_format=f, na_rep="")]
    rd = os.path.join(M.ROOT, "reading.md")
    if os.path.exists(rd):
        html += ["<h2>Reading</h2><pre style='white-space:pre-wrap'>%s</pre>" % open(rd).read()]
    open(os.path.join(M.ROOT, "E191_report.html"), "w").write("\n".join(html + ["</body></html>"]))
    for kind in ("dev", "test"):
        print(kind)
        print(T[T["kind"] == kind].sort_values("payout", ascending=False).to_string(index=False, float_format=f))
    print(K.to_string(index=False, float_format=f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    link_reused()
    if a.score:
        return M.score(a.workers)
    if a.report:
        return report()
    if a.fit:
        df, emb, feats = M.load()
        B = M.schedule(df)
        M.Features = Factory(df, emb)
        for arm in (a.arms or NEW):
            assert arm in NEW, arm
            (lgb_arm if arm.startswith("L_") else M.fit_arm)(arm, df, emb, feats, B)


if __name__ == "__main__":
    main()
