#!/usr/bin/env python3
"""
E192 -- the user's ORIGINAL options 2 and 3 for the Chronos-2-Synth state, fitted ONE RIDGE PER COIN instead
of one pooled ridge (registered in experiments/EXPERIMENTS.md and src/analysis/experiment_db.py before launch).

Pooled, options 2 and 3 cannot change a date's ranking (the date-common part enters every coin with the same
weight). Per coin, each coin weights the date-common part with its own coefficients, so they can.

    python3 experiments/e192_c2percoin.py --fit [--arms ...]   # per-coin arms (default: all not yet written)
    python3 experiments/e192_c2percoin.py --check              # dual solve == sklearn Ridge on sample coins
    python3 experiments/e192_c2percoin.py --score              # routed + covered scores (E189 scorer)
    python3 experiments/e192_c2percoin.py --report             # table, tests, HTML (print with src.analysis.html_pdf)

State blocks (RAW frozen 768-d cov states, E189 cache), per coin i and date d:
    OWN  x_i,d
    LOO  [x_i,d | (S_d - x_i,d) / (n_d - 1)]      S_d, n_d over the covered coins on d   (option 2)
    CAT  [x_1,d | x_2,d | ... ]  one fixed slot per coin, 0 where the coin is absent       (option 3)
Arms C_<block> = state only; CV_<block> = [22 v2.0 features | state]; CV_B = 22 v2.0 features only, per coin.

Model per coin and block: sklearn-equivalent Ridge with intercept, solved in dual form on the coin's own
training rows (centred Gram, eigendecomposition once, every alpha in one pass). CAT's Gram is
K[d,d'] = sum over coins present on both dates of x_c,d . x_c,d', built once for all dates, so the ~1M-column
matrix is never formed. One alpha per arm and block, shared by all coins, picked by E189's rule (inner fit on
train[:-62], validate on the last 40 train dates, mean per-date Spearman over the coins that have a model, same
grid and boundary expansion). A coin needs >= MIN_OUTER covered training rows in the block (MIN_INNER on the
inner fold); otherwise -- and for a coin with no training rows at all -- its covered rows take the pooled twin's
prediction (E189 RAW for C_*, E190 V_RAW for CV_OWN/LOO/CAT, E189 B_COV for CV_B); the share is logged.
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
import e189_c2norm as M  # noqa: E402

H = M.H
E189_ROOT, E190_ROOT = M.ROOT, "results/reports/crypto/e190_c2norm_v2"
M.STUDY, M.ROOT, M.SCR = "E192", "results/reports/crypto/e192_c2percoin", "/scratch/danielco/e192"
NEW = ["C_OWN", "C_LOO", "C_CAT", "CV_B", "CV_OWN", "CV_LOO", "CV_CAT"]
REUSED = {"B_ALL": E189_ROOT, "B_COV": E189_ROOT, "RAW": E189_ROOT, "V_RAW": E190_ROOT}
TWIN = {"C_OWN": "RAW", "C_LOO": "RAW", "C_CAT": "RAW", "CV_B": "B_COV",
        "CV_OWN": "V_RAW", "CV_LOO": "V_RAW", "CV_CAT": "V_RAW"}
M.ARMS = NEW + list(REUSED)
M.COVERED_ARMS = [a for a in M.ARMS if a != "B_ALL"]
MIN_OUTER, MIN_INNER = 30, 20


def link_reused():
    d = os.path.join(M.ROOT, "preds")
    os.makedirs(d, exist_ok=True)
    for a, root in REUSED.items():
        for ext in (".parquet", ".picks.json"):
            dst = os.path.join(d, a + ext)
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(root, "preds", a + ext)), dst)


# ---------------------------------------------------------------------------------- data
class Panel:
    """Covered rows indexed by (coin, date position), the LOO means and the global CAT Gram, built lazily."""

    def __init__(self, df, emb, feats):
        self.df, self.emb, self.feats = df, emb, feats
        self.udates = np.array(sorted(df["date"].unique()), dtype="datetime64[ns]")
        self.dpos = np.searchsorted(self.udates, df["date"].to_numpy())
        self.cov = df["covered"].to_numpy()
        self.ucid = df["ucid"].to_numpy()
        self.V = df[feats].to_numpy(np.float64)
        self._loo = self._kcat = None

    @property
    def loo(self):
        if self._loo is None:
            er = self.df.loc[self.cov, "emb_row"].to_numpy()
            dp = self.dpos[self.cov]
            S = np.zeros((len(self.udates), self.emb.shape[1]), np.float64)
            np.add.at(S, dp, self.emb[er].astype(np.float64))
            n = np.bincount(dp, minlength=len(self.udates)).astype(np.float64)
            assert n[dp].min() >= 2
            L = np.empty(self.emb.shape, np.float32)
            L[er] = (S[dp] - self.emb[er]) / (n[dp] - 1)[:, None]
            self._loo = L
        return self._loo

    @property
    def kcat(self):
        """K[d,d'] = sum_c x_c,d . x_c,d' over coins covered on both dates (date-common, all coins)."""
        if self._kcat is None:
            t0 = time.time()
            nd = len(self.udates)
            K = np.zeros((nd, nd), np.float64)
            c = self.df.loc[self.cov, ["ucid", "emb_row"]].assign(dp=self.dpos[self.cov])
            for _, g in c.groupby("ucid", sort=False):
                X = self.emb[g["emb_row"].to_numpy()].astype(np.float64)
                idx = g["dp"].to_numpy()
                K[np.ix_(idx, idx)] += X @ X.T
            self._kcat = K
            print("CAT Gram %dx%d built in %.0fs" % (nd, nd, time.time() - t0), flush=True)
        return self._kcat

    def gram(self, arm, ra, rb):
        """Uncentred Gram between row-index arrays ra and rb (rows of ONE coin) for an arm's feature map."""
        blk = arm.split("_", 1)[1]
        K = np.zeros((len(ra), len(rb)), np.float64)
        if arm.startswith("CV_"):
            K += self.V[ra] @ self.V[rb].T
        if blk in ("OWN", "LOO"):
            ea, eb = self.df["emb_row"].to_numpy()[ra], self.df["emb_row"].to_numpy()[rb]
            K += self.emb[ea].astype(np.float64) @ self.emb[eb].astype(np.float64).T
            if blk == "LOO":
                K += self.loo[ea].astype(np.float64) @ self.loo[eb].astype(np.float64).T
        elif blk == "CAT":
            K += self.kcat[np.ix_(self.dpos[ra], self.dpos[rb])]
        return K


def dual_ridge(Ktt, Ket, y, alphas):
    """Ridge with an unpenalised intercept in dual form; returns predictions (n_test, n_alpha)."""
    n = len(y)
    rm = Ktt.mean(0)
    tot = rm.mean()
    Kc = Ktt - rm[None, :] - rm[:, None] + tot
    Kec = Ket - rm[None, :] - Ket.mean(1, keepdims=True) + tot
    lam, U = np.linalg.eigh((Kc + Kc.T) / 2)
    lam = np.clip(lam, 0, None)
    ym = y.mean()
    w = U.T @ (y - ym)
    A = Kec @ U
    return A @ (w[:, None] / (lam[:, None] + np.asarray(alphas)[None, :])) + ym


def coin_preds(P, arm, fit_rows, app_rows, alphas, min_rows):
    """Per-coin predictions for app_rows (row indices) at every alpha; NaN for coins below min_rows."""
    y = P.df[M.T20].to_numpy(np.float64)
    out = np.full((len(app_rows), len(alphas)), np.nan)
    pos = pd.Series(np.arange(len(app_rows)), index=app_rows)
    fu = pd.Series(fit_rows).groupby(P.ucid[fit_rows]).agg(list)
    au = pd.Series(app_rows).groupby(P.ucid[app_rows]).agg(list)
    n_model = 0
    for u, ar in au.items():
        fr = np.array(fu.get(u, []), dtype=np.int64)
        if len(fr) < min_rows:
            continue
        ar = np.array(ar, dtype=np.int64)
        out[pos[ar].to_numpy()] = dual_ridge(P.gram(arm, fr, fr), P.gram(arm, ar, fr), y[fr], alphas)
        n_model += 1
    return out, n_model


def tune(P, arm, tr):
    fd, vd = M.inner_dates(tr)
    dates = P.df["date"].to_numpy()
    fit = np.flatnonzero(P.cov & np.isin(dates, fd))
    val = np.flatnonzero(P.cov & np.isin(dates, vd))
    assert dates[fit].max() < dates[val].min() - np.timedelta64(H.PURGE, "D")
    y, dv = P.df[M.T20].to_numpy(np.float64)[val], dates[val]
    scores, models = {}, {}

    def run(grid):
        grid = [a for a in grid if a not in scores]
        if not grid:
            return
        pr, nm = coin_preds(P, arm, fit, val, grid, MIN_INNER)
        ok = ~np.isnan(pr[:, 0])
        for j, a in enumerate(grid):
            scores[a] = M.date_spearman(dv[ok], pr[ok, j], y[ok])
            models[a] = nm

    run(list(M.ALPHAS))
    expansions = 0
    while True:
        best = max(sorted(scores), key=lambda a: scores[a])
        lo, hi = min(scores), max(scores)
        if best not in (lo, hi) or expansions == M.MAX_EXPAND:
            break
        step = 10 ** 0.5
        run([lo / step ** k for k in range(1, M.EXPAND + 1)] if best == lo else [hi * step ** k for k in range(1, M.EXPAND + 1)])
        expansions += 1
    best = max(sorted(scores), key=lambda a: scores[a])
    return dict(alpha=best, boundary=bool(best in (min(scores), max(scores))), expansions=expansions,
                inner_scores={"%.6g" % a: scores[a] for a in sorted(scores)}, inner_coins_modelled=models[best],
                n_inner_fit=len(fit), n_inner_val=len(val))


def fit_arm(arm, P, B):
    path = os.path.join(M.ROOT, "preds", arm + ".parquet")
    if os.path.exists(path):
        print("skip", arm, flush=True)
        return
    t0 = time.time()
    df = P.df
    dates = df["date"].to_numpy()
    twin = pd.read_parquet(os.path.join(M.ROOT, "preds", TWIN[arm] + ".parquet"), columns=["symbol", "date", "prediction"])
    assert (twin[["symbol", "date"]].to_numpy() == df[["symbol", "date"]].to_numpy()[np.isin(dates, np.concatenate([b[2] for b in B]))]).all()
    twin = twin.set_index(["symbol", "date"])["prediction"]
    parts, picks = [], []
    ckdir = os.path.join(M.ROOT, "preds", arm + ".blocks")            # per-block checkpoints: a killed run resumes
    os.makedirs(ckdir, exist_ok=True)
    for i, tr, te, kind in B:
        ck = os.path.join(ckdir, "block%02d" % i)
        if os.path.exists(ck + ".json"):
            parts.append(pd.read_parquet(ck + ".parquet"))
            picks.append(json.load(open(ck + ".json")))
            print("%s block %2d resumed from checkpoint" % (arm, i), flush=True)
            continue
        t1 = time.time()
        meta = tune(P, arm, tr)
        fit = np.flatnonzero(P.cov & np.isin(dates, tr))
        tst = np.flatnonzero(P.cov & np.isin(dates, te))
        pr, nm = coin_preds(P, arm, fit, tst, [meta["alpha"]], MIN_OUTER)
        pr = pr[:, 0]
        o = df.loc[np.isin(dates, te), ["symbol", "ucid", "date", M.T20, "meta_model", "covered", "group"]].copy()
        tw = twin.reindex(pd.MultiIndex.from_frame(df.loc[tst, ["symbol", "date"]])).to_numpy()
        fb = np.isnan(pr)
        assert np.isfinite(tw[fb]).all()
        pr[fb] = tw[fb]
        o["prediction"] = np.nan
        o.loc[tst, "prediction"] = pr
        o["fallback"] = False
        o.loc[tst[fb], "fallback"] = True
        o["block"], o["kind"] = i, kind
        parts.append(o)
        meta.update(block=i, kind=kind, n_train=len(fit), n_test=len(tst), coins_modelled=nm,
                    fallback_rows=int(fb.sum()), fallback_share=float(fb.mean()), seconds=time.time() - t1)
        picks.append(meta)
        o.to_parquet(ck + ".parquet.partial", index=False)
        os.replace(ck + ".parquet.partial", ck + ".parquet")
        with open(ck + ".json.partial", "w") as fh:
            json.dump(meta, fh, default=float)
        os.replace(ck + ".json.partial", ck + ".json")              # json last: it marks the block complete
        print("%s block %2d alpha %.3g%s  coins %d  fallback %.1f%%  %.0fs" % (
            arm, i, meta["alpha"], " BOUNDARY" if meta["boundary"] else "", nm, 100 * fb.mean(), meta["seconds"]), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path + ".partial", index=False)
    json.dump({"arm": arm, "picks": picks, "seconds": time.time() - t0, "host": os.uname().nodename},
              open(os.path.join(M.ROOT, "preds", arm + ".picks.json"), "w"), indent=1, default=float)
    os.replace(path + ".partial", path)
    print("%s done in %.1f min" % (arm, (time.time() - t0) / 60), flush=True)


def check(P, B):
    """The dual solve against sklearn Ridge on explicit features, a few coins, every block type."""
    from sklearn.linear_model import Ridge
    i, tr, te, _ = B[5]
    dates = P.df["date"].to_numpy()
    fit = np.flatnonzero(P.cov & np.isin(dates, tr))
    tst = np.flatnonzero(P.cov & np.isin(dates, te))
    er = P.df["emb_row"].to_numpy()
    res = {}
    for u in pd.Series(P.ucid[fit]).value_counts().index[:3]:
        fr, ar = fit[P.ucid[fit] == u], tst[P.ucid[tst] == u]
        for arm in NEW:
            if arm.endswith("CAT"):        # explicit CAT columns: slots of coins covered in this block only
                rows = np.r_[fr, ar]
                blkrows = np.flatnonzero(P.cov & np.isin(P.dpos, np.unique(P.dpos[rows])))
                coins = np.unique(P.ucid[blkrows])
                X = np.zeros((len(rows), len(coins) * 768))
                rp = {d: k for k, d in enumerate(P.dpos[rows])}
                for r in blkrows:
                    c = np.searchsorted(coins, P.ucid[r])
                    X[rp[P.dpos[r]], c * 768:(c + 1) * 768] = P.emb[er[r]]
            else:
                rows = np.r_[fr, ar]
                X = P.emb[er[rows]].astype(np.float64) if arm != "CV_B" else np.zeros((len(rows), 0))
                if arm.endswith("LOO"):
                    X = np.hstack([X, P.loo[er[rows]]])
            if arm.startswith("CV_"):
                X = np.hstack([P.V[rows], X])
            a = 10.0 ** (8 if arm.endswith("CAT") else 3)
            y64 = P.df[M.T20].to_numpy(np.float64)[fr]          # float64: sklearn on the float32 target is only ~1e-6 exact
            sk = Ridge(alpha=a).fit(X[:len(fr)], y64).predict(X[len(fr):])
            du = dual_ridge(P.gram(arm, fr, fr), P.gram(arm, ar, fr), y64, [a])[:, 0]
            res["%s_%s" % (u, arm)] = float(np.abs(sk - du).max() / max(np.abs(sk - sk.mean()).max(), 1e-12))
    print(json.dumps(res, indent=1))
    worst = max(res.values())
    print("worst relative |dual - sklearn| = %.2e" % worst)
    os.makedirs(M.ROOT, exist_ok=True)
    json.dump(res, open(os.path.join(M.ROOT, "dual_check.json"), "w"), indent=1)
    assert worst < 1e-6


# ---------------------------------------------------------------------------------- report
def report():
    import scipy.stats as st
    import e131_analysis as E
    S = pd.read_parquet(os.path.join(M.ROOT, "scores.parquet"))
    pay0 = S[S["mmc"].notna()]["date"].min()
    rows, tests = [], []
    pairs = [("C_LOO", "C_OWN"), ("C_CAT", "C_OWN"), ("C_OWN", "RAW"), ("C_LOO", "RAW"), ("C_CAT", "RAW"),
             ("CV_LOO", "CV_OWN"), ("CV_CAT", "CV_OWN"), ("CV_OWN", "CV_B"), ("CV_LOO", "CV_B"), ("CV_CAT", "CV_B"),
             ("CV_B", "B_COV"), ("CV_OWN", "V_RAW"), ("CV_LOO", "B_COV"), ("CV_CAT", "B_COV"), ("CV_OWN", "B_COV")]
    for kind in ("dev", "test"):
        s = S[S["kind"] == kind]
        P = s.pivot(index="date", columns="arm", values="payout").dropna()
        C = s.pivot(index="date", columns="arm", values="corr")
        CC = s.pivot(index="date", columns="arm", values="cov_corr")
        Mm = s.pivot(index="date", columns="arm", values="mmc").loc[P.index]
        for a in M.ARMS:
            cum = P[a].cumsum()
            rows.append(dict(kind=kind, arm=a, corr=C[a].mean(), cov_corr=CC[a].mean(), mmc=Mm[a].mean(), payout=P[a].mean(),
                             pay_sharpe=P[a].mean() / P[a].std(ddof=0), pay_dd=float((cum.cummax() - cum).max()),
                             n_corr=int(C[a].notna().sum()), n_pay=len(P)))
        for a, ref in pairs:
            for met, X in (("corr", C), ("payout", P)):
                dlt = (X[a] - X[ref]).dropna()
                m, _, t = E.newey_west(dlt.to_numpy())
                tests.append(dict(kind=kind, arm=a, vs=ref, metric=met, delta=m, t=t, n=len(dlt),
                                  p=float(2 * (1 - st.norm.cdf(abs(t))))))
    T, K = pd.DataFrame(rows), pd.DataFrame(tests)
    for _, g in K.groupby(["kind", "metric"]):                       # Holm over all pairs per kind x metric
        o = g["p"].sort_values()
        K.loc[o.index, "holm"] = np.maximum.accumulate([min(1, p * (len(o) - i)) for i, p in enumerate(o)])
    fb = {a: np.mean([p["fallback_share"] for p in json.load(open(os.path.join(M.ROOT, "preds", a + ".picks.json")))["picks"]])
          for a in NEW}
    alphas = {a: [round(np.log10(p["alpha"]), 1) for p in json.load(open(os.path.join(M.ROOT, "preds", a + ".picks.json")))["picks"]]
              for a in M.ARMS}
    T["fallback_share"] = T["arm"].map(fb)
    T.to_csv(os.path.join(M.ROOT, "table.csv"), index=False)
    K.to_csv(os.path.join(M.ROOT, "tests.csv"), index=False)
    ff = lambda x: "%.4f" % x  # noqa: E731
    html = ["<html><head><meta charset='utf-8'><title>E192 per-coin Chronos-2 states</title><style>"
            "body{font-family:sans-serif;margin:24px;font-size:12px}table{border-collapse:collapse;margin:8px 0 18px}"
            "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#eee}</style></head><body>",
            "<h1>E192: options 1-3 for the Chronos-2-Synth state, one ridge per coin</h1>",
            "<p>E189 design and scorer. C_* = state only, CV_* = 22 v2.0 features + state, CV_B = v2.0 only; all per coin, "
            "alpha shared by coins and picked per block. OWN = own state; LOO = own | mean of the other covered coins on the "
            "date; CAT = every coin's state on the date in fixed slots. Pooled twins: RAW (E189), V_RAW (E190), B_COV / B_ALL "
            "(E189). Routed R_SLOT vs B_ALL; cov_corr = covered rows only. MMC/payout over dates with the meta model (from %s). "
            "Dev = 11 blocks 2020-08..2026-02; test = 2026-02-20..2026-08-20.</p>" % str(pay0)[:10]]
    for kind in ("dev", "test"):
        t = T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind")
        html += ["<h2>%s, sorted by payout</h2>" % kind, t.to_html(index=False, float_format=ff, na_rep="")]
        html += ["<h3>%s paired Newey-West (lag 20), Holm over all pairs</h3>" % kind,
                 K[K["kind"] == kind].drop(columns="kind").to_html(index=False, float_format=ff, na_rep="")]
    html += ["<h2>log10 alpha picked per block</h2>", pd.DataFrame(alphas).T.to_html()]
    rd = os.path.join(M.ROOT, "reading.md")
    if os.path.exists(rd):
        html += ["<h2>Reading</h2><pre style='white-space:pre-wrap'>%s</pre>" % open(rd).read()]
    html.append("</body></html>")
    open(os.path.join(M.ROOT, "E192_report.html"), "w").write("\n".join(html))
    for kind in ("dev", "test"):
        print(kind)
        print(T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind").to_string(index=False, float_format=ff))
    print(K.to_string(index=False, float_format=ff))


def _fit_one(arm):
    df, emb, feats = M.load()
    P = Panel(df, emb, feats)
    fit_arm(arm, P, M.schedule(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    link_reused()
    if a.score:
        return M.score(a.workers)
    if a.report:
        return report()
    df, emb, feats = M.load()
    P = Panel(df, emb, feats)
    B = M.schedule(df)
    if a.check:
        return check(P, B)
    if a.fit:
        arms = [x for x in (a.arms or NEW) if not os.path.exists(os.path.join(M.ROOT, "preds", x + ".parquet"))]
        assert all(x in NEW for x in arms), arms
        if a.workers > 1 and len(arms) > 1:
            del P, df, emb
            with ProcessPoolExecutor(min(a.workers, len(arms))) as ex:
                list(ex.map(_fit_one, arms))
        else:
            for arm in arms:
                fit_arm(arm, P, B)


if __name__ == "__main__":
    main()
