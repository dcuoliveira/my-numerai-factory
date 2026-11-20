#!/usr/bin/env python3
"""E198 -- do cross-sectional statistics of the raw v2.0 features (crypto/v2.0-cs) add value?

    python3 experiments/e198_cs_eval.py --fit-pooled --model lgb_ktz --n-jobs 4
    python3 experiments/e198_cs_eval.py --fit-tabnet --arm TN_F --seed 0 --workers 8
    python3 experiments/e198_cs_eval.py --score                      # after every fit; HTML + PDF

Dataset src/data_mani/build_cs_moments.py: v2.0 + 15 feature_cs_* columns, NaN where not covered,
mask cov_cs_all. Rows <= 2026-08-20. Blocks: E131's 11 development blocks (train 130 dates, purge 22,
predict 130) plus the held-out test block 11 (2026-02-20..2026-08-20, trained on the last 130 dates
before the purge), the same schedule as every routed feature-set study (E134-E159).

Arms, missing-value protocol (.claude/skills/numerai-crypto-missing-values):
  B_ALL   pooled, the 22 shipped features, every row (NaN -> 0.5)            ridge, lgb_ktz, mlp_ktz
  B_COV   pooled, the 22, trained and predicted ONLY on cov_cs_all rows      ridge, lgb_ktz, mlp_ktz
  F_COV   pooled, 22 + 15 CS columns, the same rows as B_COV                 ridge, lgb_ktz, mlp_ktz
  TN_B    ONE TABNET PER COIN, the 22, cov_cs_all rows                       (the requested baseline)
  TN_F    ONE TABNET PER COIN, 22 + 15, the same rows
Pooled configs are E131's fixed configs (never searched on these columns); seeds: ridge 0, others 0-2.
TabNet: one pinned config (below), TV epochs from its own size, seeds 0-2. A coin with fewer than
MIN_ROWS covered training rows in a block takes the ridge B_COV prediction in BOTH TabNet arms.

Scoring: every covered arm is routed R_SLOT into its model's B_ALL ranking (TabNet into ridge's):
the covered coins are reordered by the arm, inside the rank slots they hold in B_ALL; uncovered coins
keep B_ALL's rank. CORR = numerai_corr, MMC = correlation_contribution against the meta model,
payout = 0.1 CORR + MMC, per date, averaged over seeds per date. Paired per-date deltas, Newey-West
lag 20. Primary: F_COV - B_COV per model, TN_F - TN_B. Development and test reported separately.
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

os.chdir(REPO)
sys.path.insert(0, REPO)

STUDY = "E198"
DIR = "data/crypto/v2.0-cs"
ROOT = "results/reports/crypto/e198_cs"
PRED = os.path.join(ROOT, "preds")
LAST = pd.Timestamp("2026-08-20")
T20 = H.T20
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
SEEDS = {"ridge": [0], "lgb_ktz": [0, 1, 2], "mlp_ktz": [0, 1, 2], "tabnet": [0, 1, 2]}
POOLED = ["B_ALL", "B_COV", "F_COV"]
TABNET = ["TN_B", "TN_F"]
MIN_ROWS = 30
TABNET_CFG = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2, n_shared=2, lambda_sparse=1e-3,
                  learning_rate=0.02, patience=10, batch_size=256, virtual_batch_size=64, device="cpu")
PAIRS = [("F_COV", "B_COV", m) for m in MODELS] + [("TN_F", "TN_B", "tabnet"),
                                                     ("TN_B", "B_COV", "tabnet|ridge"), ("TN_F", "F_COV", "tabnet|ridge")]


def load():
    df = pd.read_parquet(os.path.join(DIR, "train.parquet")).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base = [c for c in df if c.startswith("feature_") and not c.startswith("feature_cs_")]
    cs = [c for c in df if c.startswith("feature_cs_")]
    assert len(base) == 22 and len(cs) == 15, (len(base), len(cs))
    df["cov"] = df["cov_cs_all"].astype(bool)
    assert df.loc[df["cov"], cs].notna().all().all()
    df[base] = df[base].fillna(0.5)            # shipped NaN -> neutral, identically in every arm
    meta = pd.read_parquet(os.path.join(DIR, "historical_meta_models.parquet"))[["symbol", "date", "prediction"]]
    meta = meta.rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    df = df.merge(meta, on=["symbol", "date"], how="left", validate="1:1")
    return df, base, cs


def schedule(df):
    dates = np.array(sorted(df["date"].unique()))
    past = dates[dates < H.TEST_START]
    test = dates[dates >= H.TEST_START]
    return H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], test)]


def cols_of(arm, base, cs):
    return base + cs if arm in ("F_COV", "TN_F") else base


def out_path(model, arm, seed):
    return os.path.join(PRED, model, "%s_s%d.parquet" % (arm, seed))


def save(parts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.concat(parts).to_parquet(path + ".partial", index=False)
    os.replace(path + ".partial", path)


# ------------------------------------------------------------------ pooled
def fit_pooled(model, n_jobs, arms):
    df, base, cs = load()
    B = schedule(df)
    for arm in arms:
        cols = cols_of(arm, base, cs)
        for s in SEEDS[model]:
            path = out_path(model, arm, s)
            if os.path.exists(path):
                print("skip", path)
                continue
            parts, t1 = [], time.time()
            for i, tr, te in B:
                trm, tem = df["date"].isin(tr), df["date"].isin(te)
                if arm != "B_ALL":
                    trm, tem = trm & df["cov"], tem & df["cov"]
                if not tem.any() or trm.sum() < 1000:    # block 0: no covered history yet -> B_ALL there
                    continue
                pred = H.fit_predict(model, H.FIXED[model], s, n_jobs, df.loc[trm, cols].to_numpy(np.float32),
                                     df.loc[trm, T20].to_numpy(), df.loc[tem, cols].to_numpy(np.float32))
                parts.append(df.loc[tem, ["symbol", "date"]].assign(prediction=pred, block=i))
            save(parts, path)
            print("%s %-6s seed %d  %d cols  %.0fs" % (model, arm, s, len(cols), time.time() - t1), flush=True)


# ------------------------------------------------------------------ tabnet per coin
_G = {}


def _init(arm, seed):
    import torch
    torch.set_num_threads(1)
    df, base, cs = load()
    _G.update(df=df[df["cov"]].reset_index(drop=True), cols=cols_of(arm, base, cs), seed=seed)


def _coin_block(task):
    from src.prediction.tabnet_tv import TabNetTVRegressor
    ucid, i, tr0, tr1, te0, te1 = task
    d, cols = _G["df"], _G["cols"]
    g = d[d["ucid"] == ucid]
    tr = g[(g["date"] >= tr0) & (g["date"] <= tr1)]
    te = g[(g["date"] >= te0) & (g["date"] <= te1)]
    if te.empty:
        return None
    out = te[["symbol", "date"]].assign(block=i, n_train=len(tr))
    if len(tr) < MIN_ROWS:
        return out.assign(prediction=np.nan)
    est = TabNetTVRegressor(seed=_G["seed"], **TABNET_CFG)
    est.fit(tr[cols].to_numpy(np.float32), tr[T20].to_numpy(np.float32))
    return out.assign(prediction=np.asarray(est.predict(te[cols].to_numpy(np.float32))).ravel())


def fit_tabnet(arm, seed, workers):
    path = out_path("tabnet", arm, seed)
    if os.path.exists(path):
        print("skip", path)
        return
    df, _, _ = load()
    cov = df[df["cov"]]
    tasks = []
    for i, tr, te in schedule(df):
        coins = cov.loc[cov["date"].isin(te), "ucid"].unique()
        tasks += [(u, i, tr[0], tr[-1], te[0], te[-1]) for u in coins]
    t1 = time.time()
    parts = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(arm, seed)) as ex:
        for k, r in enumerate(ex.map(_coin_block, tasks, chunksize=4)):
            if r is not None:
                parts.append(r)
            if k % 500 == 0:
                print("tabnet %s seed %d  %d/%d tasks  %.0fs" % (arm, seed, k, len(tasks), time.time() - t1), flush=True)
    save(parts, path)
    p = pd.concat(parts)
    print("tabnet %s seed %d done: %d rows, %.1f%% fallback, %.0fs" % (arm, seed, len(p), 100 * p["prediction"].isna().mean(),
                                                                       time.time() - t1), flush=True)


# ------------------------------------------------------------------ scoring
def read(model, arm, seed):
    p = pd.read_parquet(out_path(model, arm, seed))
    p["date"] = pd.to_datetime(p["date"])
    return p


def routed(keys, model, arm, seed):
    """Per-date CORR/MMC of arm routed R_SLOT into its model's B_ALL (TabNet into ridge's)."""
    bmodel = "ridge" if model == "tabnet" else model
    bseed = 0 if model == "tabnet" else seed
    b = read(bmodel, "B_ALL", bseed)[["symbol", "date", "prediction"]].rename(columns={"prediction": "b0"})
    k = keys.merge(b, on=["symbol", "date"], how="inner", validate="1:1")
    if arm != "B_ALL":
        c = read(model, arm, seed)[["symbol", "date", "prediction"]].rename(columns={"prediction": "cand"})
        if model == "tabnet":       # coins below MIN_ROWS take ridge B_COV in both TabNet arms
            fb = read("ridge", "B_COV", 0)[["symbol", "date", "prediction"]].rename(columns={"prediction": "fb"})
            c = c.merge(fb, on=["symbol", "date"], how="left", validate="1:1")
            c["cand"] = c["cand"].fillna(c.pop("fb"))
        k = k.merge(c, on=["symbol", "date"], how="left", validate="1:1")
    rows = []
    for d, g in k.groupby("date", sort=True):
        x = g["b0"].rank(pct=True).to_numpy()
        if arm != "B_ALL":
            cand = g["cand"].to_numpy()
            cov = ~np.isnan(cand)
            i = np.flatnonzero(cov)
            x = x.copy()
            x[i[np.argsort(cand[i], kind="stable")]] = np.sort(x[i])
        h = g.assign(prediction=x).set_index("symbol")
        c = E.corr_one(h)
        m = E.contrib_one(h, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(date=d, corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan,
                         covered=float(cov.mean()) if arm != "B_ALL" else 0.0))
    return pd.DataFrame(rows).assign(model=model, arm=arm, seed=seed)


def _score_job(job):
    keys = pd.read_parquet(os.path.join(ROOT, "keys.parquet"))
    return routed(keys, *job)


def summarize(s):
    out = {}
    for k in ("corr", "mmc", "payout"):
        out[k.upper()] = s[k].mean()
    p = s["payout"].dropna()
    out["p.sharpe"] = p.mean() / p.std(ddof=0) if p.std(ddof=0) > 0 else np.nan
    out["dd(pay)"] = E.max_dd(p.values)
    out["calmar"] = p.mean() / out["dd(pay)"] if out["dd(pay)"] > 0 else np.nan
    out["burn(d)"] = int((p < 0).sum())
    out["covered"] = s["covered"].mean()
    out["n"] = len(s)
    return out


def score(workers):
    df, _, _ = load()
    B = schedule(df)
    kind = {d: ("test" if i == 11 else "dev") for i, _, te in B for d in te}
    keys = df.loc[df["date"].isin(list(kind)), ["symbol", "date", T20, "meta_model"]]
    os.makedirs(ROOT, exist_ok=True)
    keys.to_parquet(os.path.join(ROOT, "keys.parquet"), index=False)
    jobs = [(m, a, s) for m in MODELS for a in POOLED for s in SEEDS[m]] + \
           [("tabnet", a, s) for a in TABNET for s in SEEDS["tabnet"]]
    missing = [j for j in jobs if not os.path.exists(out_path(*j))]
    assert not missing, "fits missing: %s" % missing
    with ProcessPoolExecutor(workers) as ex:
        per = pd.concat(ex.map(_score_job, jobs))
    per["kind"] = per["date"].map(kind)
    per.to_parquet(os.path.join(ROOT, "per_date.parquet"), index=False)
    daily = per.groupby(["model", "arm", "kind", "date"])[["corr", "mmc", "payout", "covered"]].mean().reset_index()
    table = pd.DataFrame([dict(model=m, arm=a, kind=k, **summarize(g)) for (m, a, k), g in daily.groupby(["model", "arm", "kind"])])
    deltas = []
    for a, b, spec in PAIRS:
        ma, mb = (spec.split("|") + spec.split("|"))[:2] if "|" in spec else (spec, spec)
        for k in ("dev", "test"):
            x = daily[(daily.model == ma) & (daily.arm == a) & (daily.kind == k)].set_index("date")
            y = daily[(daily.model == mb) & (daily.arm == b) & (daily.kind == k)].set_index("date")
            j = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
            r = dict(arm="%s:%s" % (ma, a), base="%s:%s" % (mb, b), kind=k, n=len(j))
            for met in ("corr", "mmc", "payout"):
                mean, se, t = E.newey_west((j[met + "_a"] - j[met + "_b"]).values)
                r["d" + met] = mean
                r["t_" + met] = t
            deltas.append(r)
    deltas = pd.DataFrame(deltas)
    table.to_csv(os.path.join(ROOT, "table.csv"), index=False)
    deltas.to_csv(os.path.join(ROOT, "deltas.csv"), index=False)
    pd.set_option("display.width", 220)
    for k in ("dev", "test"):
        print("\n== %s, sorted by payout" % k)
        print(table[table.kind == k].drop(columns="kind").sort_values("PAYOUT", ascending=False).round(5).to_string(index=False))
    print("\n== paired deltas (Newey-West lag 20)")
    print(deltas.round(5).to_string(index=False))
    report(table, deltas, daily)


def report(table, deltas, daily):
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = []
    for k in ("dev", "test"):
        fig, ax = plt.subplots(figsize=(9, 4))
        for (m, a), g in daily[daily.kind == k].groupby(["model", "arm"]):
            if a == "B_ALL" and m != "ridge":
                continue
            ax.plot(g["date"], g["payout"].fillna(0).cumsum(), label="%s %s" % (m, a), lw=1)
        ax.set_title("Cumulative routed payout, %s" % k)
        ax.legend(fontsize=6, ncol=3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        figs.append(base64.b64encode(buf.getvalue()).decode())
    fmt = lambda f: f.to_html(index=False, float_format=lambda x: "%.5f" % x, border=0)
    tabs = "".join("<h2>%s block, sorted by payout</h2>%s" % (k, fmt(table[table.kind == k].drop(columns="kind")
                                                                        .sort_values("PAYOUT", ascending=False)))
                   for k in ("dev", "test"))
    html = """<!doctype html><html><head><meta charset="utf-8"><title>E198 CS statistics</title>
<style>body{font-family:sans-serif;font-size:11px;margin:24px}table{border-collapse:collapse}
td,th{padding:2px 6px;border-bottom:1px solid #ddd;text-align:right}h1{font-size:18px}h2{font-size:14px}</style></head><body>
<h1>E198: cross-sectional statistics of raw v2.0 features (crypto/v2.0-cs)</h1>
<p>Target target_binned_return_20. Covered arms routed R_SLOT into B_ALL (TabNet into ridge B_ALL); metrics per date,
averaged over seeds. Development = E131 blocks 0-10; test = block 11 (2026-02-20..2026-08-20). calmar and burn(d)
compare only rows with the same n. Pooled configs fixed (E131); TabNet one pinned config, one model per coin,
coins with &lt; %d covered training rows take ridge B_COV in both TabNet arms.</p>
%s<h2>Paired deltas, Newey-West lag 20</h2>%s
<img style="width:100%%" src="data:image/png;base64,%s"><img style="width:100%%" src="data:image/png;base64,%s">
</body></html>""" % (MIN_ROWS, tabs, fmt(deltas), figs[0], figs[1])
    path = os.path.join(ROOT, "E198_cs_report.html")
    with open(path, "w") as h:
        h.write(html)
    print("report:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-pooled", action="store_true")
    ap.add_argument("--fit-tabnet", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arms", nargs="+", default=POOLED)
    ap.add_argument("--arm", choices=TABNET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    with open(os.path.join(ROOT, "config.json"), "w") as h:
        json.dump(dict(study=STUDY, seeds=SEEDS, tabnet=TABNET_CFG, min_rows=MIN_ROWS, fixed=H.FIXED), h, indent=1, default=str)
    if a.fit_pooled:
        fit_pooled(a.model, a.n_jobs, a.arms)
    elif a.fit_tabnet:
        fit_tabnet(a.arm, a.seed, a.workers)
    elif a.score:
        score(a.workers)


if __name__ == "__main__":
    main()
