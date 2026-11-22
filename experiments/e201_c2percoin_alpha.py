#!/usr/bin/env python3
"""
E201 -- E192 with an INDEPENDENT ridge per coin: every coin picks its own alpha on its own inner fold, instead of
one alpha per arm and block shared by all coins (registered in experiments/EXPERIMENTS.md and
src/analysis/experiment_db.py before launch).

    python3 experiments/e201_c2percoin_alpha.py --fit [--arms ...]   # 6 arms (default: all not yet written)
    python3 experiments/e201_c2percoin_alpha.py --check              # per-coin alpha pick == brute-force sklearn
    python3 experiments/e201_c2percoin_alpha.py --score              # routed + covered scores (E189 scorer)
    python3 experiments/e201_c2percoin_alpha.py --report             # table, tests, HTML (print with src.analysis.html_pdf)

Arms (RAW frozen Chronos-2-Synth states, E192's feature maps and dual solver):
    A_OWN   own state                    AV_OWN  own state + 22 v2.0 features
    A_LOO   own | mean of other coins     AV_LOO  own | mean of other coins + v2.0
    A_CAT   every coin's state, flat      AV_CAT  every coin's state, flat + v2.0
Per coin and block: inner fit on the coin's rows in train[:-62], validate on its rows in the last 40 train dates
(E189's purged split). A single coin has one row per date, so there is no per-date ranking to score; the coin's
alpha minimises its validation MSE over a 49-point half-decade grid 1e-9..1e15 (the range E192's shared pick
could reach with its boundary expansion). The coin is then refitted on all its block training rows at that
alpha. Coins with >= MIN_OUTER (30) training rows but < MIN_INNER (20) inner-fit or < MIN_VAL (10) validation
rows take the median alpha of the block's tuned coins; coins below MIN_OUTER take the pooled twin's prediction
(as E192). Shared-alpha twins: E192's C_*/CV_* arms; pooled twins RAW/V_RAW/B_COV/B_ALL.
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
import e192_c2percoin as Z  # noqa: E402

M, H = Z.M, Z.H
E192_ROOT = M.ROOT
M.STUDY, M.ROOT, M.SCR = "E201", "results/reports/crypto/e201_c2percoin_alpha", "/scratch/danielco/e201"
NEW = ["A_OWN", "AV_OWN", "A_LOO", "AV_LOO", "A_CAT", "AV_CAT"]
FEATMAP = {"A_OWN": "C_OWN", "AV_OWN": "CV_OWN", "A_LOO": "C_LOO", "AV_LOO": "CV_LOO", "A_CAT": "C_CAT", "AV_CAT": "CV_CAT"}
SHARED = list(FEATMAP.values())                                  # E192 shared-alpha twins
REUSED = {**{a: E192_ROOT for a in SHARED}, "B_ALL": E192_ROOT, "B_COV": E192_ROOT, "RAW": E192_ROOT, "V_RAW": E192_ROOT}
TWIN = {a: Z.TWIN[FEATMAP[a]] for a in NEW}
M.ARMS = NEW + list(REUSED)
M.COVERED_ARMS = [a for a in M.ARMS if a != "B_ALL"]
GRID = np.logspace(-9, 15, 49)
MIN_OUTER, MIN_INNER, MIN_VAL = 30, 20, 10


def link_reused():
    d = os.path.join(M.ROOT, "preds")
    os.makedirs(d, exist_ok=True)
    for a, root in REUSED.items():
        for ext in (".parquet", ".picks.json"):
            dst = os.path.join(d, a + ext)
            src = os.path.realpath(os.path.join(root, "preds", a + ext))
            if not os.path.lexists(dst):
                os.symlink(src, dst)


def coin_alphas(P, arm, tr):
    """{ucid: (alpha, boundary, n_fit, n_val)} for every coin with enough inner rows."""
    fm = FEATMAP[arm]
    fd, vd = M.inner_dates(tr)
    dates = P.df["date"].to_numpy()
    y = P.df[M.T20].to_numpy(np.float64)
    fit = np.flatnonzero(P.cov & np.isin(dates, fd))
    val = np.flatnonzero(P.cov & np.isin(dates, vd))
    assert dates[fit].max() < dates[val].min() - np.timedelta64(H.PURGE, "D")
    fu = pd.Series(fit).groupby(P.ucid[fit]).agg(list)
    vu = pd.Series(val).groupby(P.ucid[val]).agg(list)
    out = {}
    for u, vr in vu.items():
        fr = np.array(fu.get(u, []), dtype=np.int64)
        vr = np.array(vr, dtype=np.int64)
        if len(fr) < MIN_INNER or len(vr) < MIN_VAL:
            continue
        pr = Z.dual_ridge(P.gram(fm, fr, fr), P.gram(fm, vr, fr), y[fr], GRID)
        mse = ((pr - y[vr][:, None]) ** 2).mean(0)
        j = int(np.argmin(mse))                                    # first min on an ascending grid
        out[u] = (float(GRID[j]), j in (0, len(GRID) - 1), len(fr), len(vr))
    return out


def fit_arm(arm, P, B):
    path = os.path.join(M.ROOT, "preds", arm + ".parquet")
    if os.path.exists(path):
        print("skip", arm, flush=True)
        return
    t0 = time.time()
    fm = FEATMAP[arm]
    df = P.df
    dates = df["date"].to_numpy()
    y = df[M.T20].to_numpy(np.float64)
    twin = pd.read_parquet(os.path.join(M.ROOT, "preds", TWIN[arm] + ".parquet"), columns=["symbol", "date", "prediction"])
    twin = twin.set_index(["symbol", "date"])["prediction"]
    ckdir = os.path.join(M.ROOT, "preds", arm + ".blocks")         # per-block checkpoints: a killed run resumes
    os.makedirs(ckdir, exist_ok=True)
    parts, picks = [], []
    for i, tr, te, kind in B:
        ck = os.path.join(ckdir, "block%02d" % i)
        if os.path.exists(ck + ".json"):
            parts.append(pd.read_parquet(ck + ".parquet"))
            picks.append(json.load(open(ck + ".json")))
            print("%s block %2d resumed from checkpoint" % (arm, i), flush=True)
            continue
        t1 = time.time()
        al = coin_alphas(P, arm, tr)
        med = float(np.median([a for a, *_ in al.values()])) if al else float(GRID[len(GRID) // 2])
        fit = np.flatnonzero(P.cov & np.isin(dates, tr))
        tst = np.flatnonzero(P.cov & np.isin(dates, te))
        fu = pd.Series(fit).groupby(P.ucid[fit]).agg(list)
        pr = np.full(len(tst), np.nan)
        pos = pd.Series(np.arange(len(tst)), index=tst)
        n_tuned = n_med = 0
        coin_alpha = {}
        for u, ar in pd.Series(tst).groupby(P.ucid[tst]).agg(list).items():
            fr = np.array(fu.get(u, []), dtype=np.int64)
            if len(fr) < MIN_OUTER:
                continue
            if u in al:
                a = al[u][0]
                n_tuned += 1
            else:
                a, n_med = med, n_med + 1
            coin_alpha[str(u)] = a
            ar = np.array(ar, dtype=np.int64)
            pr[pos[ar].to_numpy()] = Z.dual_ridge(P.gram(fm, fr, fr), P.gram(fm, ar, fr), y[fr], [a])[:, 0]
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
        la = np.log10([v for v in coin_alpha.values()]) if coin_alpha else np.array([np.nan])
        bnd = [b for (a, b, *_ ) in al.values()]
        meta = dict(block=i, kind=kind, n_train=len(fit), n_test=len(tst), coins_tuned=n_tuned, coins_median_alpha=n_med,
                    median_alpha=med, log10_alpha_q=[float(q) for q in np.nanpercentile(la, [10, 25, 50, 75, 90])],
                    boundary_share=float(np.mean(bnd)) if bnd else None, coin_alpha=coin_alpha,
                    fallback_rows=int(fb.sum()), fallback_share=float(fb.mean()), seconds=time.time() - t1)
        picks.append(meta)
        o.to_parquet(ck + ".parquet.partial", index=False)
        os.replace(ck + ".parquet.partial", ck + ".parquet")
        with open(ck + ".json.partial", "w") as fh:
            json.dump(meta, fh, default=float)
        os.replace(ck + ".json.partial", ck + ".json")                # json last: it marks the block complete
        parts.append(o)
        print("%s block %2d  coins tuned %d / median-alpha %d  log10 alpha q10/50/90 %.1f/%.1f/%.1f  boundary %.0f%%  "
              "fallback %.1f%%  %.0fs" % (arm, i, n_tuned, n_med, meta["log10_alpha_q"][0], meta["log10_alpha_q"][2],
                                         meta["log10_alpha_q"][4], 100 * (meta["boundary_share"] or 0), 100 * fb.mean(),
                                         meta["seconds"]), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path + ".partial", index=False)
    with open(os.path.join(M.ROOT, "preds", arm + ".picks.json"), "w") as fh:
        json.dump({"arm": arm, "picks": picks, "seconds": time.time() - t0, "host": os.uname().nodename}, fh, indent=1, default=float)
    os.replace(path + ".partial", path)
    print("%s done in %.1f min" % (arm, (time.time() - t0) / 60), flush=True)


def check(P, B):
    """For 3 coins x 6 arms: the per-coin MSE pick equals a brute-force sklearn Ridge sweep on explicit features."""
    from sklearn.linear_model import Ridge
    i, tr, te, _ = B[5]
    fd, vd = M.inner_dates(tr)
    dates = P.df["date"].to_numpy()
    y = P.df[M.T20].to_numpy(np.float64)
    er = P.df["emb_row"].to_numpy()
    fit = np.flatnonzero(P.cov & np.isin(dates, fd))
    val = np.flatnonzero(P.cov & np.isin(dates, vd))
    res = {}
    for arm in [a for a in NEW if not a.endswith("CAT")]:
        al = coin_alphas(P, arm, tr)
        for u in pd.Series(P.ucid[fit]).value_counts().index[:3]:
            fr, vr = fit[P.ucid[fit] == u], val[P.ucid[val] == u]
            rows = np.r_[fr, vr]
            X = P.emb[er[rows]].astype(np.float64)
            if "LOO" in arm:
                X = np.hstack([X, P.loo[er[rows]]])
            if arm.startswith("AV"):
                X = np.hstack([P.V[rows], X])
            grid = GRID[(GRID >= 1e-6) & (GRID <= 1e9)]       # sklearn is unstable at the grid's extremes
            mse = [((Ridge(alpha=a).fit(X[:len(fr)], y[fr]).predict(X[len(fr):]) - y[vr]) ** 2).mean() for a in grid]
            sk = float(grid[int(np.argmin(mse))])
            mine = al[u][0]
            res["%s_%s" % (u, arm)] = dict(sklearn=sk, dual=mine, same=bool(np.isclose(sk, mine) or not (1e-6 <= mine <= 1e9)))
    print(json.dumps(res, indent=1))
    os.makedirs(M.ROOT, exist_ok=True)
    json.dump(res, open(os.path.join(M.ROOT, "alpha_check.json"), "w"), indent=1)
    assert all(r["same"] for r in res.values())


def report():
    import scipy.stats as st
    import e131_analysis as E
    S = pd.read_parquet(os.path.join(M.ROOT, "scores.parquet"))
    pay0 = S[S["mmc"].notna()]["date"].min()
    rows, tests = [], []
    pairs = [(a, FEATMAP[a]) for a in NEW] + [("A_LOO", "A_OWN"), ("A_CAT", "A_OWN"), ("AV_LOO", "AV_OWN"), ("AV_CAT", "AV_OWN"),
             ("AV_OWN", "A_OWN"), ("AV_OWN", "B_COV"), ("AV_LOO", "B_COV"), ("AV_CAT", "B_COV"), ("A_OWN", "RAW"), ("AV_OWN", "V_RAW")]
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
                tests.append(dict(kind=kind, arm=a, vs=ref, metric=met, delta=m, t=t, n=len(dlt), p=float(2 * (1 - st.norm.cdf(abs(t))))))
    T, K = pd.DataFrame(rows), pd.DataFrame(tests)
    for _, g in K.groupby(["kind", "metric"]):
        o = g["p"].sort_values()
        K.loc[o.index, "holm"] = np.maximum.accumulate([min(1, p * (len(o) - i)) for i, p in enumerate(o)])
    pk = {a: json.load(open(os.path.join(M.ROOT, "preds", a + ".picks.json")))["picks"] for a in NEW}
    T["fallback_share"] = T["arm"].map({a: np.mean([p["fallback_share"] for p in v]) for a, v in pk.items()})
    al = pd.DataFrame({a: ["%.1f / %.1f / %.1f (%.0f%% bnd)" % (p["log10_alpha_q"][0], p["log10_alpha_q"][2], p["log10_alpha_q"][4],
                                                                100 * (p["boundary_share"] or 0)) for p in v] for a, v in pk.items()}).T
    T.to_csv(os.path.join(M.ROOT, "table.csv"), index=False)
    K.to_csv(os.path.join(M.ROOT, "tests.csv"), index=False)
    ff = lambda x: "%.4f" % x  # noqa: E731
    html = ["<html><head><meta charset='utf-8'><title>E201 independent per-coin ridge</title><style>"
            "body{font-family:sans-serif;margin:24px;font-size:12px}table{border-collapse:collapse;margin:8px 0 18px}"
            "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#eee}</style></head><body>",
            "<h1>E201: Chronos-2-Synth state options, an independent ridge per coin (own alpha)</h1>",
            "<p>E192 with each coin choosing its own alpha (validation MSE on its inner fold, grid 1e-9..1e15). A_* = state only, "
            "AV_* = state + 22 v2.0. OWN = own state; LOO = own | mean of the other covered coins; CAT = every coin's state in "
            "fixed slots. Shared-alpha twins C_*/CV_* (E192); pooled RAW/V_RAW/B_COV/B_ALL. Routed R_SLOT vs B_ALL; cov_corr = "
            "covered rows only. MMC/payout over meta-model dates (from %s). Dev = 11 blocks 2020-08..2026-02; test = "
            "2026-02-20..2026-08-20.</p>" % str(pay0)[:10]]
    for kind in ("dev", "test"):
        t = T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind")
        html += ["<h2>%s, sorted by payout</h2>" % kind, t.to_html(index=False, float_format=ff, na_rep="")]
        html += ["<h3>%s paired Newey-West (lag 20), Holm over all pairs</h3>" % kind,
                 K[K["kind"] == kind].drop(columns="kind").to_html(index=False, float_format=ff, na_rep="")]
    html += ["<h2>per-coin log10 alpha, q10 / median / q90 by block (share of tuned coins at the grid edge)</h2>", al.to_html()]
    rd = os.path.join(M.ROOT, "reading.md")
    if os.path.exists(rd):
        html += ["<h2>Reading</h2><pre style='white-space:pre-wrap'>%s</pre>" % open(rd).read()]
    html.append("</body></html>")
    open(os.path.join(M.ROOT, "E201_report.html"), "w").write("\n".join(html))
    for kind in ("dev", "test"):
        print(kind)
        print(T[T["kind"] == kind].sort_values("payout", ascending=False).drop(columns="kind").to_string(index=False, float_format=ff))
    print(K.to_string(index=False, float_format=ff))


def _fit_one(arm):
    df, emb, feats = M.load()
    fit_arm(arm, Z.Panel(df, emb, feats), M.schedule(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    link_reused()
    if a.score:
        return M.score(a.workers)
    if a.report:
        return report()
    df, emb, feats = M.load()
    B = M.schedule(df)
    if a.check:
        return check(Z.Panel(df, emb, feats), B)
    if a.fit:
        arms = [x for x in (a.arms or NEW) if not os.path.exists(os.path.join(M.ROOT, "preds", x + ".parquet"))]
        assert all(x in NEW for x in arms), arms
        if a.workers > 1 and len(arms) > 1:
            from concurrent.futures import ProcessPoolExecutor
            del df, emb
            with ProcessPoolExecutor(min(a.workers, len(arms))) as ex:
                list(ex.map(_fit_one, arms))
        else:
            P = Z.Panel(df, emb, feats)
            for arm in arms:
                fit_arm(arm, P, B)


if __name__ == "__main__":
    main()
