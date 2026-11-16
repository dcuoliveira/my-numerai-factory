#!/usr/bin/env python3
"""
E134 -- do the feed-fillind datasets beat crypto/v2.0 under the rule we would run live?

    python3 experiments/e134_route_feature_sets.py [--check]

E132 gave a coin with no features a neutral prediction. Live, the v2.0 model scores every coin
anyway, so a coin outside the richer dataset keeps its v2.0 prediction. No model is refitted here:
this scores E132's saved predictions under that rule, per date and seed-paired.

  R_SLOT   primary. B0 percentiles over every coin. The candidate reorders only the coins it covers,
           inside the percentile slots those coins already hold, so an uncovered coin keeps B0's
           exact rank and the two arms can differ only on covered coins.
  R_POOL   sensitivity. Candidate percentile among covered coins, B0 percentile among all coins,
           merged and re-ranked. Simpler, but an uncovered coin can move.
  R_HALF   sensitivity. R_SLOT, but a coin counts as covered only if at least half the proposed
           features are observed on that date, so a new listing in warm-up routes to v2.0 as well.
           Warm-up is known at the time. The no-feed part of the mask is not.

Which coins have no feed is decided by today's coverage (E132 section 3), so the covered set is
survivor-biased. Under R_SLOT that cannot move an uncovered coin, but the candidate is still judged
mostly on coins that lived, and live it would also face coins that later die.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402

os.chdir(REPO)
T20 = E.T20
SRC = "results/reports/crypto/e132_feed_fillind/preds"
ROOT = "results/reports/crypto/e134_route"
MODELS = E.MODELS
SEEDS = {"ridge": [0], "lgb_ktz": list(range(5)), "mlp_ktz": list(range(5))}
ARMS = {"L_FULL": "L", "L_M1": "L", "S_FULL": "S", "S_M1": "S", "C_FULL": "C", "C_RAW": "C"}
RULES = ["R_SLOT", "R_POOL", "R_HALF"]
DATA = {"L": "data/crypto/v2.0-litmom-feed-fillind/train.parquet",
        "S": "data/crypto/v2.0-momswap-feed-fillind/train.parquet",
        "C": "data/crypto/v2.0-momswap-macro-carver-feed-fillind/train.parquet"}
TEST0 = pd.Timestamp("2026-02-20")
PROPOSED = {"L": ("feature_isna_mom_",), "S": ("feature_isna_mom_",), "C": ("feature_isna_mom_", "feature_isna_tf_")}


def observed():
    """Share of proposed features observed per key, for R_HALF. Saved so workers read one file."""
    out = pd.read_parquet(os.path.join(SRC, "keys.parquet"))[["symbol", "date"]]
    for k, path in DATA.items():
        d = pd.read_parquet(path).reset_index()
        cols = [c for c in d if c.startswith(PROPOSED[k])]
        assert len(cols) == (32 if k == "C" else 17)
        d["obs_" + k] = 1 - d[cols].mean(axis=1)
        out = out.merge(d[["symbol", "date", "obs_" + k]], on=["symbol", "date"], how="left", validate="1:1")
    out.to_parquet(os.path.join(ROOT, "observed.parquet"), index=False)


def pct(p):
    p = p[["symbol", "date", "prediction"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    return p


def slot(b, c, cov):
    """Put covered coins into the sorted B0 values they occupy, in the candidate's order."""
    out = b.copy()
    i = np.flatnonzero(cov)
    out[i[np.argsort(c[i], kind="stable")]] = np.sort(b[i])
    return out


def route(g, rule):
    b, c = g["b0"].values, g["cand"].values
    if rule == "R_POOL":
        cov = ~np.isnan(c)
        x = b.copy()
        x[cov] = pd.Series(c[cov]).rank(pct=True).values
        return pd.Series(x).rank(pct=True).values, cov
    cov = ~np.isnan(c)
    if rule == "R_HALF":
        cov = cov & (g["obs"].values >= .5)
    return slot(b, c, cov), cov


def frame(model, seed, arm):
    """Every key with B0's percentile, B0F's and the candidate's raw prediction."""
    k = pd.read_parquet(os.path.join(SRC, "keys.parquet"))
    key = ARMS[arm]
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"), columns=["symbol", "date", "obs_" + key]).rename(columns={"obs_" + key: "obs"})
    b = pct(pd.read_parquet(os.path.join(SRC, model, "B0_s%d.parquet" % seed))).rename(columns={"prediction": "b0"})
    f = pct(pd.read_parquet(os.path.join(SRC, model, "B0F_s%d.parquet" % seed))).rename(columns={"prediction": "b0f"})
    c = pct(pd.read_parquet(os.path.join(SRC, model, "%s_s%d.parquet" % (arm, seed)))).rename(columns={"prediction": "cand"})
    g = k.merge(b, on=["symbol", "date"], how="left", validate="1:1").merge(c, on=["symbol", "date"], how="left", validate="1:1") \
        .merge(f, on=["symbol", "date"], how="left", validate="1:1").merge(o, on=["symbol", "date"], how="left", validate="1:1")
    assert g["b0"].notna().all()
    g["b0"] = g.groupby("date")["b0"].rank(pct=True)
    return g


def one(g, col="prediction"):
    gg = g.set_index("symbol")
    return E.corr_one(gg, col)


def score_job(job):
    model, seed, arm = job
    f = frame(model, seed, arm)
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for rule in RULES:
            x, cov = route(g, rule)
            h = g.assign(prediction=x)
            c = one(h)
            m = E.contrib_one(h.set_index("symbol"), "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
            kept = h[cov]
            rows.append(dict(model=model, seed=seed, arm=arm, rule=rule, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0],
                             corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan, routed=1 - cov.mean(),
                             predcorr_b0=pd.Series(x).rank().corr(g["b0"].rank()),
                             cov_arm=one(kept), cov_b0=one(kept, "b0"), cov_b0f=one(kept[kept["b0f"].notna()], "b0f")))
    return pd.DataFrame(rows)


def base_job(job):
    model, seed = job
    f = frame(model, seed, "L_FULL")
    rows = []
    for d, g in f.groupby("date", sort=True):
        h = g.assign(prediction=g["b0"].values)
        c = one(h)
        m = E.contrib_one(h.set_index("symbol"), "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(model=model, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0], corr=c, mmc=m,
                         payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def tuned_job(job):
    """R_SLOT with both legs retuned, CORR and payout on the runs' common dates."""
    model, tag = job
    k = pd.read_parquet(os.path.join(SRC, "keys.parquet"))

    def load(name):
        d = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (model, name), usecols=["symbol", "date", "prediction"], parse_dates=["date"])
        return d[d["date"].isin(k["date"])]
    b = load("e131_%s_tuneB0" % model).rename(columns={"prediction": "b0"})
    c = load("e132_%s_tune%s" % (model, tag)).rename(columns={"prediction": "cand"})
    g = k.merge(b, on=["symbol", "date"], how="inner", validate="1:1").merge(c, on=["symbol", "date"], how="left", validate="1:1")
    g = g[g["date"].isin(c["date"].unique())]
    g["b0"] = g.groupby("date")["b0"].rank(pct=True)
    rows = []
    for d, z in g.groupby("date", sort=True):
        z = z.reset_index(drop=True)
        for name, x in (("arm", slot(z["b0"].values, z["cand"].values, z["cand"].notna().values)), ("b0", z["b0"].values)):
            h = z.assign(prediction=x)
            cr = one(h)
            m = E.contrib_one(h.set_index("symbol"), "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
            rows.append(dict(model=model, tag=tag, date=d, who=name, corr=cr, payout=0.1 * cr + m if m == m else np.nan))
    return pd.DataFrame(rows)


def check():
    """A candidate equal to B0 must reproduce B0 exactly, and uncovered coins must never move."""
    rng = np.random.default_rng(0)
    b = rng.permutation(np.arange(1, 301) / 300)
    cov = rng.random(300) < .9
    assert (slot(b, b, cov) == b).all()
    x = slot(b, rng.random(300), cov)
    assert (x[~cov] == b[~cov]).all() and (np.sort(x) == np.sort(b)).all()
    g = frame("ridge", 0, "L_FULL")
    s = pd.read_parquet("results/reports/crypto/e132_feed_fillind/scores_by_date_seed.parquet", columns=["model", "arm", "seed", "date", "corr"])
    s = s[(s["model"] == "ridge") & (s["arm"] == "B0")].set_index("date")["corr"]
    d = next(iter(g.groupby("date", sort=True)))[1]
    assert abs(one(d.assign(prediction=d["b0"].values)) - s.loc[d["date"].iloc[0]]) < 1e-9
    print("check ok")


def summarise(S, B, T):
    A = S.groupby(["model", "arm", "rule", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout", "routed", "predcorr_b0", "cov_arm", "cov_b0", "cov_b0f"]].mean()
    AB = B.groupby(["model", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout"]].mean()
    lev, dl, by = [], [], []

    def level(g, **key):
        for sc in ("dev", "test"):
            z = g[g["kind"] == sc]
            pw = z.dropna(subset=["payout"])
            dd = E.max_dd(pw["payout"].values)
            lev.append(dict(key, scope=sc, corr=z["corr"].mean(), c_sharpe=z["corr"].mean() / z["corr"].std(ddof=0), dd_corr=E.max_dd(z["corr"].values), mmc=pw["mmc"].mean(),
                            payout=pw["payout"].mean(), p_sharpe=pw["payout"].mean() / pw["payout"].std(ddof=0), dd_pay=dd, calmar=pw["payout"].mean() / dd,
                            burn=int((pw["payout"] < 0).sum()), n=len(z), n_pay=len(pw)))
    for m in MODELS:
        b = AB[AB["model"] == m]
        level(b, model=m, arm="B0", rule="-")
        for (arm, rule), g in A[A["model"] == m].groupby(["arm", "rule"]):
            level(g, model=m, arm=arm, rule=rule)
            z = g.merge(b, on=["date", "kind", "block"], suffixes=("", "_b"))
            z["d"] = z["corr"] - z["corr_b"]
            z["dp"] = z["payout"] - z["payout_b"]
            z["dcov"] = z["cov_arm"] - z["cov_b0"]
            z["dcovf"] = z["cov_arm"] - z["cov_b0f"]
            for sc in ("dev", "test"):
                q = z[z["kind"] == sc]
                p = q["dp"].dropna()
                dl.append(dict(model=m, arm=arm, rule=rule, scope=sc, **E.describe(q["d"]), payout_mean=p.mean(), payout_t=E.newey_west(p.values)[2], payout_n=len(p),
                               routed=q["routed"].mean(), predcorr_b0=q["predcorr_b0"].mean(), cov_mean=q["dcov"].mean(), cov_t=E.newey_west(q["dcov"].values)[2],
                               covf_mean=q["dcovf"].mean(), covf_t=E.newey_west(q["dcovf"].values)[2]))
            for kk, zz in z.groupby("block"):
                by.append(dict(model=m, arm=arm, rule=rule, by="block", key=kk, mean=zz["d"].mean()))
            for kk, zz in z.groupby(z["date"].dt.year):
                by.append(dict(model=m, arm=arm, rule=rule, by="year", key=kk, mean=zz["d"].mean()))
    pd.DataFrame(lev).to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)
    pd.DataFrame(dl).to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(ROOT, "paired_deltas_by_block_year.csv"), index=False)
    p = S.merge(B[["model", "seed", "date", "corr"]], on=["model", "seed", "date"], suffixes=("", "_b"))
    p["d"] = p["corr"] - p["corr_b"]
    rows = []
    for (m, a, r, s, sc), g in p.groupby(["model", "arm", "rule", "seed", "kind"]):
        rows.append(dict(model=m, arm=a, rule=r, seed=s, scope=sc, mean=g["d"].mean(), nw_t=E.newey_west(g["d"].values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"), index=False)
    rows = []
    for (m, tag), g in T.groupby(["model", "tag"]):
        w = g.pivot(index="date", columns="who", values=["corr", "payout"]).dropna(subset=[("corr", "arm"), ("corr", "b0")])
        for sc, mk in (("pre-test", w.index < TEST0), ("test", w.index >= TEST0)):
            d = (w[("corr", "arm")] - w[("corr", "b0")])[mk]
            dp = (w[("payout", "arm")] - w[("payout", "b0")])[mk].dropna()
            rows.append(dict(model=m, dataset=tag, scope=sc, arm_corr=w[("corr", "arm")][mk].mean(), b0_corr=w[("corr", "b0")][mk].mean(), **E.describe(d),
                             payout_mean=dp.mean(), payout_t=E.newey_west(dp.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tuned_equal_budget.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, "observed.parquet")):
        observed()
    check()
    if a.check:
        return
    jobs = [(m, s, x) for m in MODELS for s in SEEDS[m] for x in ARMS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E134_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(score_job, jobs)), ignore_index=True)
        B = pd.concat(list(ex.map(base_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
        T = pd.concat(list(ex.map(tuned_job, [(m, t) for m in MODELS for t in "LSC"])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    B.to_parquet(os.path.join(ROOT, "scores_b0.parquet"), index=False)
    T.to_parquet(os.path.join(ROOT, "scores_tuned.parquet"), index=False)
    summarise(S, B, T)
    json.dump({"jobs": len(jobs), "rules": RULES, "arms": list(ARMS)}, open(os.path.join(ROOT, "analysis_meta.json"), "w"))
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
