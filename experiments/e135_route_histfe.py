#!/usr/bin/env python3
"""
E135 -- does crypto/v2.0-histfe beat crypto/v2.0 under the rule we would run live?

    python3 experiments/e135_route_histfe.py --fit --model lgb_ktz --n-jobs 8
    python3 experiments/e135_route_histfe.py --fit --model lgb_ktz --tune --n-jobs 8
    python3 experiments/e135_route_histfe.py [--check]

E134's rule and bar on E133's dataset. E133 predicted the 11 dev blocks and never read a date from
2026-02-20, so --fit adds the final block with E133's configs, train 130 / purge 22 / predict 130.
Scoring then joins the two, per date and seed-paired.

Every coin has the 22 base features here, so there is no feed mask. A coin lacks history features
only in warm-up, which is known at the time, and E133 fed those cells to the model as 0.5.
"Covered" is therefore read from the arm's OWN proposed columns on that date:

  R_SLOT   primary. At least one proposed feature observed. B0 percentiles over every coin; the
           candidate reorders only covered coins, inside the slots they already hold.
  R_POOL   sensitivity. Same covered set; percentiles merged and re-ranked.
  R_HALF   sensitivity. R_SLOT, but at least half the proposed features observed.

P_HIST has nothing to route: tenure exists for every coin, so it is scored as it stands.
ADDED_ONLY is left out because it drops the base features.
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
import e131_analysis as E  # noqa: E402
import e133_histfe_factorial as H  # noqa: E402
import e134_route_feature_sets as R  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e135_route_histfe"
R.ROOT = ROOT
DEV = "results/reports/crypto/e133_histfe/preds"
TEST = os.path.join(ROOT, "preds")
KEYS = os.path.join(R.SRC, "keys.parquet")
MODELS = E.MODELS
SEEDS = R.SEEDS
ARMS = ["SHORT", "LONG", "COMBO", "TSC", "FULL", "F_RANK", "F_MA", "F_TSMAX", "P_HIST"]
RULES = ["R_SLOT", "R_POOL", "R_HALF"]
LAST = pd.Timestamp("2026-08-20")


def load():
    """E133's load() without its date cut, ending where the 20-day label does."""
    df = pd.read_parquet(H.DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base = [c for c in df if c.startswith("feature_") and not c.startswith("feature_hfe_")]
    new = [c for c in df if c.startswith("feature_hfe_")]
    assert len(base) == 22 and len(new) == 352
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    obs = df[["symbol", "date"]].copy()
    obs[new] = df[new].notna().astype("float32")
    df[base + new + ["feature_hist"]] = df[base + new + ["feature_hist"]].fillna(0.5)
    meta = pd.read_parquet(H.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, new, obs


def observed():
    """Share of each arm's proposed features observed per key. Saved so workers read one file."""
    df, base, new, obs = load()
    A = H.arms(base, new)
    k = pd.read_parquet(KEYS, columns=["symbol", "date"])
    k["date"] = pd.to_datetime(k["date"])
    for arm in ARMS:
        cols = [c for c in A[arm] if c.startswith("feature_hfe_")]
        obs["obs_" + arm] = obs[cols].mean(axis=1) if cols else 1.0
    k.merge(obs[["symbol", "date"] + ["obs_" + a for a in ARMS]], on=["symbol", "date"], how="left", validate="1:1") \
        .to_parquet(os.path.join(ROOT, "observed.parquet"), index=False)


def fit(args):
    df, base, new, _ = load()
    A = H.arms(base, new)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    train = past[-(H.PURGE + H.TRAIN):-H.PURGE]
    key = pd.read_parquet(KEYS, columns=["date", "kind"])
    last = pd.to_datetime(key.loc[key["kind"] == "test", "date"]).unique()
    te = df["date"].isin(last)
    assert len(last) == H.BLOCK and df.loc[te, "date"].min() == H.TEST_START
    tr = df["date"].isin(train)
    seeds = [0] if (args.model == "ridge" or args.tune) else SEEDS[args.model]
    arms = ["B0", "FULL"] if args.tune else ["B0"] + ARMS
    tag = "tune" if args.tune else ""
    keep = ["symbol", "date", H.T20, "meta_model"]
    for arm in arms:
        cols = A[arm]
        for s in seeds:
            path = os.path.join(TEST, args.model, "%s%s_s%d.parquet" % (tag, arm, s))
            if os.path.exists(path):
                print("skip", path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            t1 = time.time()
            params = H.pick(args.model, df, cols, train, s, args.n_jobs) if args.tune else H.FIXED[args.model]
            o = df.loc[te, keep].copy()
            o["prediction"] = H.fit_predict(args.model, params, s, args.n_jobs, df.loc[tr, cols].to_numpy(np.float32),
                                            df.loc[tr, H.T20].to_numpy(), df.loc[te, cols].to_numpy(np.float32))
            o["block"] = 11
            tmp = path + ".partial"
            o.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            print("%s %-8s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


def preds(model, name):
    """E133's dev predictions followed by the final block."""
    out = []
    for root in (DEV, TEST):
        p = pd.read_parquet(os.path.join(root, model, name + ".parquet"), columns=["symbol", "date", "prediction"])
        p["date"] = pd.to_datetime(p["date"])
        out.append(p)
    return pd.concat(out, ignore_index=True)


def frame(model, seed, arm, tag=""):
    k = pd.read_parquet(KEYS)
    k["date"] = pd.to_datetime(k["date"])
    b = preds(model, "%sB0_s%d" % (tag, seed)).rename(columns={"prediction": "b0"})
    c = preds(model, "%s%s_s%d" % (tag, arm, seed)).rename(columns={"prediction": "cand"})
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"), columns=["symbol", "date", "obs_" + arm]).rename(columns={"obs_" + arm: "obs"})
    g = k.merge(b, on=["symbol", "date"], how="left", validate="1:1").merge(c, on=["symbol", "date"], how="left", validate="1:1") \
        .merge(o, on=["symbol", "date"], how="left", validate="1:1")
    assert g[["b0", "cand", "obs"]].notna().all().all()
    g["b0"] = g.groupby("date")["b0"].rank(pct=True)
    return g


def route(g, rule):
    b, c = g["b0"].values, g["cand"].values
    cov = g["obs"].values >= .5 if rule == "R_HALF" else g["obs"].values > 0
    if rule == "R_POOL":
        x = b.copy()
        x[cov] = pd.Series(c[cov]).rank(pct=True).values
        return pd.Series(x).rank(pct=True).values, cov
    return R.slot(b, c, cov), cov


def priced(h):
    c = R.one(h)
    m = E.contrib_one(h.set_index("symbol"), "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
    return c, m


def score_job(job):
    model, seed, arm = job
    rows = []
    for d, g in frame(model, seed, arm).groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for rule in RULES:
            x, cov = route(g, rule)
            h = g.assign(prediction=x)
            c, m = priced(h)
            kept = h[cov]
            # no dropped-row baseline here, so cov_b0f repeats cov_b0 for E134's summariser
            rows.append(dict(model=model, seed=seed, arm=arm, rule=rule, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0],
                             corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan, routed=1 - cov.mean(),
                             predcorr_b0=pd.Series(x).rank().corr(g["b0"].rank()),
                             cov_arm=R.one(kept), cov_b0=R.one(kept, "b0"), cov_b0f=R.one(kept, "b0")))
    return pd.DataFrame(rows)


def base_job(job):
    model, seed = job
    rows = []
    for d, g in frame(model, seed, "FULL").groupby("date", sort=True):
        c, m = priced(g.assign(prediction=g["b0"].values))
        rows.append(dict(model=model, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0], corr=c, mmc=m,
                         payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def tuned_job(model):
    """R_SLOT with both legs retuned under the same grid, CORR and payout."""
    rows = []
    for d, g in frame(model, 0, "FULL", "tune").groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for name, x in (("arm", route(g, "R_SLOT")[0]), ("b0", g["b0"].values)):
            c, m = priced(g.assign(prediction=x))
            rows.append(dict(model=model, tag="H", date=d, who=name, corr=c, payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def check():
    """B0 must be the same model as E132's, on the final block as well as on dev."""
    rng = np.random.default_rng(0)
    b = rng.permutation(np.arange(1, 301) / 300)
    cov = rng.random(300) < .9
    assert (R.slot(b, b, cov) == b).all()
    x = R.slot(b, rng.random(300), cov)
    assert (x[~cov] == b[~cov]).all() and (np.sort(x) == np.sort(b)).all()
    for model in MODELS:
        p = pd.read_parquet(os.path.join(R.SRC, model, "B0_s0.parquet"), columns=["symbol", "date", "prediction"]).rename(columns={"prediction": "e132"})
        p["date"] = pd.to_datetime(p["date"])
        z = preds(model, "B0_s0").merge(p, on=["symbol", "date"], validate="1:1")
        t = z[z["date"] >= H.TEST_START]
        assert len(z) == 468000 and len(t) == 39000, (len(z), len(t))
        c = t.groupby("date").apply(lambda g: g["prediction"].corr(g["e132"]), include_groups=False).min()
        print("B0 vs E132 %-8s max |diff| %.2e  lowest per-date corr %.6f" % (model, (t["prediction"] - t["e132"]).abs().max(), c))
        assert c > .999 if model != "lgb_ktz" else c > .99
    print("check ok")


def tenure(S):
    """Each routed arm against the tenure control. E133's lgb F_RANK gain tied P_HIST on dev."""
    A = S.groupby(["model", "arm", "rule", "date", "kind"], as_index=False)[["corr", "payout"]].mean()
    rows = []
    for m in MODELS:
        c = A[(A["model"] == m) & (A["arm"] == "P_HIST") & (A["rule"] == "R_SLOT")]
        for arm in ("FULL", "F_RANK"):
            for rule in ("R_SLOT", "R_HALF"):
                z = A[(A["model"] == m) & (A["arm"] == arm) & (A["rule"] == rule)].merge(c, on=["date", "kind"], suffixes=("", "_c"))
                for sc in ("dev", "test"):
                    q = z[z["kind"] == sc]
                    p = (q["payout"] - q["payout_c"]).dropna()
                    rows.append(dict(model=m, arm=arm, rule=rule, scope=sc, **E.describe(q["corr"] - q["corr_c"]),
                                     payout_mean=p.mean(), payout_t=E.newey_west(p.values)[2], payout_n=len(p)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "vs_tenure.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        fit(a)
        return
    if not os.path.exists(os.path.join(ROOT, "observed.parquet")):
        observed()
    check()
    if a.check:
        return
    jobs = [(m, s, x) for m in MODELS for s in SEEDS[m] for x in ARMS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E135_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(score_job, jobs)), ignore_index=True)
        B = pd.concat(list(ex.map(base_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
        T = pd.concat(list(ex.map(tuned_job, MODELS)), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    B.to_parquet(os.path.join(ROOT, "scores_b0.parquet"), index=False)
    T.to_parquet(os.path.join(ROOT, "scores_tuned.parquet"), index=False)
    R.summarise(S, B, T)
    tenure(S)
    json.dump({"jobs": len(jobs), "rules": RULES, "arms": ARMS}, open(os.path.join(ROOT, "analysis_meta.json"), "w"))
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
