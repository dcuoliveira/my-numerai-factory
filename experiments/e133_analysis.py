#!/usr/bin/env python3
"""
E133 analysis: score every arm on the dev blocks, paired against B0, plus redundancy,
coverage and blends. Reads predictions and dev-period rows only; dates from 2026-02-20
are dropped on load and never scored. Statistics and per-date scoring are E131's.
MMC is a proxy against the historical meta model and only exists from 2024-05-07.

    python3 experiments/e133_analysis.py --workers 8
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.binning import BIN_LABELS  # noqa: E402
from src.data_mani.history_features import history_columns  # noqa: E402

T20 = E.T20
ROOT = "results/reports/crypto/e133_histfe"
PRED = os.path.join(ROOT, "preds")
DATA = "data/crypto/v2.0-histfe/train.parquet"
TEST0 = pd.Timestamp("2026-02-20")
MODELS = E.MODELS
COMPARISONS = [("SHORT", "B0", "short horizons"), ("LONG", "B0", "long horizons"), ("COMBO", "B0", "short + long"),
               ("TSC", "B0", "days since change"), ("FULL", "B0", "all 352"), ("F_RANK", "B0", "own-history rank"),
               ("F_MA", "B0", "MA differences"), ("F_TSMAX", "B0", "days since max"), ("P_HIST", "B0", "tenure control"),
               ("COMBO", "SHORT", "long, given short"), ("COMBO", "LONG", "short, given long"),
               ("FULL", "COMBO", "days since change, given horizons"), ("FULL", "P_HIST", "block beyond tenure"),
               ("ADDED_ONLY", "B0", "added features alone")]


def score_job(job):
    model, arm, seed, tag, path, b0 = job
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"])
    assert f["date"].max() < TEST0
    f = f.merge(pd.read_parquet(b0, columns=["symbol", "date", "prediction"]).rename(columns={"prediction": "b0"})
                .assign(date=lambda d: pd.to_datetime(d["date"])), on=["symbol", "date"], validate="1:1")
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        c = E.corr_one(g)
        m = E.contrib_one(g, "meta_model") if g["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(model=model, arm=arm, seed=seed, tune=tag, date=d, block=int(g["block"].iloc[0]), corr=c, mmc=m,
                         payout=0.1 * c + m if m == m else np.nan,
                         predcorr_b0=g["prediction"].rank().corr(g["b0"].rank())))
    return pd.DataFrame(rows)


def blend_job(job):
    model, seed, b0, added = job
    a = pd.read_parquet(b0)
    a["date"] = pd.to_datetime(a["date"])
    b = pd.read_parquet(added, columns=["symbol", "date", "prediction"]).rename(columns={"prediction": "added"})
    b["date"] = pd.to_datetime(b["date"])
    f = a.merge(b, on=["symbol", "date"], validate="1:1")
    f["prediction"] = f.groupby("date")["prediction"].rank(pct=True)
    f["added"] = f.groupby("date")["added"].rank(pct=True)
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        for w in E.W:
            g["bl"] = (1 - w) * g["prediction"] + w * g["added"]
            rows.append(dict(model=model, seed=seed, date=d, block=int(g["block"].iloc[0]), w=w, corr=E.corr_one(g, "bl")))
    return pd.DataFrame(rows)


def levels(S):
    rows = []
    for (model, arm, tag), g in S.groupby(["model", "arm", "tune"]):
        p = g.groupby("date")[["corr", "mmc", "payout"]].mean(numeric_only=True)
        pay = p["payout"].dropna()
        dd = E.max_dd(pay.values)
        rows.append(dict(model=model, arm=tag + arm, CORR=p["corr"].mean(), MMC=p["mmc"].mean(),
                         c_sharpe=p["corr"].mean() / p["corr"].std(ddof=0), dd_corr=E.max_dd(p["corr"].values),
                         payout=pay.mean(), p_sharpe=pay.mean() / pay.std(ddof=0), dd_pay=dd, calmar=pay.mean() / dd,
                         burn=int((pay < 0).sum()), n=len(p), n_pay=len(pay),
                         predcorr_b0=g["predcorr_b0"].mean(), seeds=g["seed"].nunique()))
    return pd.DataFrame(rows)


def deltas(S):
    rows = []
    for model in MODELS:
        for metric in ("corr", "mmc", "payout"):
            w = S[(S["model"] == model) & (S["tune"] == "")].pivot_table(index="date", columns="arm", values=metric)
            for a, b, label in COMPARISONS:
                if a not in w or b not in w:
                    continue
                d = (w[a] - w[b]).dropna()
                r = E.describe(d)
                rows.append(dict(model=model, metric=metric, arm=a, vs=b, label=label, delta=r["mean"], t=r["nw_t"],
                                 boot_lo=r["boot_lo"], boot_hi=r["boot_hi"], n=r["n"]))
            t = S[(S["model"] == model) & (S["tune"] == "tune")].pivot_table(index="date", columns="arm", values=metric)
            if {"B0", "FULL"} <= set(t):
                r = E.describe((t["FULL"] - t["B0"]).dropna())
                rows.append(dict(model=model, metric=metric, arm="tuneFULL", vs="tuneB0", label="equal-budget retune",
                                 delta=r["mean"], t=r["nw_t"], boot_lo=r["boot_lo"], boot_hi=r["boot_hi"], n=r["n"]))
    return pd.DataFrame(rows)


def stability(S):
    rows = []
    fixed = S[S["tune"] == ""]
    for model in MODELS:
        w = fixed[fixed["model"] == model]
        by = w.pivot_table(index=["block", "date"], columns="arm", values="corr")
        for arm in ("SHORT", "LONG", "COMBO", "FULL"):
            if arm not in by:
                continue
            d = (by[arm] - by["B0"]).groupby("block").mean()
            rows += [dict(model=model, arm=arm, cut="block", key=int(k), delta=v) for k, v in d.items()]
        seeds = w.pivot_table(index=["seed", "date"], columns="arm", values="corr")
        for arm in ("SHORT", "LONG", "COMBO", "FULL"):
            if arm in seeds:
                d = (seeds[arm] - seeds["B0"]).groupby("seed").mean()
                rows += [dict(model=model, arm=arm, cut="seed", key=int(k), delta=v) for k, v in d.items()]
    return pd.DataFrame(rows)


def blends(B):
    """Weight picked on the OTHER blocks, then scored on the held-out block."""
    rows = []
    for model, g in B.groupby("model"):
        p = g.pivot_table(index=["block", "date"], columns="w", values="corr")
        means = p.groupby("block").mean()
        chosen = []
        for blk in p.index.get_level_values("block").unique():
            w = means.drop(index=blk).mean().idxmax()
            chosen.append((p.xs(blk, level="block")[w] - p.xs(blk, level="block")[0.0]).rename(blk))
        d = pd.concat(chosen)
        r = E.describe(d.values)
        rows.append(dict(model=model, w_full=means.mean().idxmax(), delta_cv=r["mean"], t=r["nw_t"], n=r["n"],
                         **{"w%.2f" % w: p[w].mean() for w in E.W}))
    return pd.DataFrame(rows)


def redundancy():
    """Dev rows only, 40 sampled dates. Correlations are per-date Spearman, averaged."""
    d = pd.read_parquet(DATA).reset_index()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] < TEST0]
    base = [c for c in d if c.startswith("feature_") and not c.startswith("feature_hfe_")]
    new = [c for c in d if c.startswith("feature_hfe_")]
    fam = history_columns(base)
    per = []
    for f in new:
        g = d[["date", f, T20]].dropna()
        x = g.groupby("date").apply(lambda z: z[f].rank().corr(z[T20].rank()), include_groups=False).dropna()
        m, _, t = E.newey_west(x.values)
        per.append(dict(feature=f, ic=m, t=t, present=d[f].notna().mean(), mid_bin=(d[f] == 0.5).sum() / d[f].notna().sum(),
                        top_bin_mass=max(((d[f] == b).sum() / d[f].notna().sum()) for b in BIN_LABELS)))
    per = pd.DataFrame(per).set_index("feature")
    for k, cols in fam.items():
        per.loc[cols, "family"] = k
    every = sorted(d["date"].unique())
    days = every[::max(1, len(every) // 40)]
    corr = sum(d[d["date"] == x][base + new].fillna(0.5).corr(method="spearman").fillna(0).to_numpy() for x in days) / len(days)
    corr = pd.DataFrame(corr, index=base + new, columns=base + new)
    per["max_abs_corr_base"] = corr.loc[new, base].abs().max(axis=1)
    per["parent_corr"] = [corr.loc[f, "feature_" + re.sub(r"_(rank\d+|tsmax\d+|ma\d+_\d+|tschg)$", "", f[len("feature_hfe_"):])] for f in new]
    nn = corr.loc[new, new].abs().to_numpy().copy()
    np.fill_diagonal(nn, 0)
    per["max_abs_corr_block"] = nn.max(axis=1)
    eig = np.sort(np.linalg.eigvalsh(corr.loc[new, new].to_numpy()))[::-1]
    share = np.cumsum(eig) / eig.sum()
    summary = dict(n_new=len(new), pairs_gt_95=int((np.triu(nn, 1) > .95).sum()), pairs_gt_80=int((np.triu(nn, 1) > .8).sum()),
                   comps_for_90=int(np.searchsorted(share, .90) + 1), comps_for_99=int(np.searchsorted(share, .99) + 1),
                   sig_t2=int((per["t"].abs() >= 2).sum()), bonf=int((per["t"].abs() >= 3.8).sum()), dates=len(days))
    return per, summary


def coverage():
    d = pd.read_parquet(DATA).reset_index()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] < TEST0]
    new = [c for c in d if c.startswith("feature_hfe_")]
    tail = pd.Series({c: c.rsplit("_", 1)[-1] for c in new})
    tail = tail.where(~tail.str.fullmatch(r"\d+"), [c.rsplit("_", 2)[-2] + "_" + c.rsplit("_", 1)[-1] for c in new])
    return d.groupby(d["date"].dt.year)[new].apply(lambda g: g.notna().mean()).T.groupby(tail).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs, bjobs = [], []
    for model in MODELS:
        for path in sorted(glob.glob(os.path.join(PRED, model, "*_s*.parquet"))):
            stem = os.path.basename(path)[:-len(".parquet")]
            arm, seed = stem.rsplit("_s", 1)
            tag = "tune" if arm.startswith("tune") else ""
            arm = arm[len(tag):]
            jobs.append((model, arm, int(seed), tag, path, os.path.join(PRED, model, "%sB0_s%s.parquet" % (tag, seed))))
            if arm == "ADDED_ONLY":
                bjobs.append((model, int(seed), os.path.join(PRED, model, "B0_s%s.parquet" % seed), path))
    print(len(jobs), "prediction files;", len(bjobs), "blend pairs", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        S = pd.concat(ex.map(score_job, jobs), ignore_index=True)
        B = pd.concat(ex.map(blend_job, bjobs), ignore_index=True)
    assert S["date"].max() < TEST0
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_columns", 40)
    L = levels(S)
    D = deltas(S)
    L.to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)
    D.to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    stability(S).to_csv(os.path.join(ROOT, "stability.csv"), index=False)
    BL = blends(B)
    BL.to_csv(os.path.join(ROOT, "blend_summary.csv"), index=False)
    per, summ = redundancy()
    per.to_csv(os.path.join(ROOT, "feature_summary.csv"))
    json.dump(summ, open(os.path.join(ROOT, "redundancy.json"), "w"), indent=1)
    coverage().to_csv(os.path.join(ROOT, "coverage_by_year.csv"))
    print(L.round(4).to_string(index=False))
    print(D[D["metric"] == "corr"].round(4).to_string(index=False))
    print(D[D["metric"] == "payout"].round(4).to_string(index=False))
    print(BL.round(4).to_string(index=False))
    print(summ)
    print(per.groupby("family")[["ic", "present", "mid_bin", "max_abs_corr_base", "max_abs_corr_block"]].mean().round(3))
    print(per.assign(a=per["t"].abs()).sort_values("a", ascending=False).head(12)[["family", "ic", "t", "present", "mid_bin", "parent_corr"]].round(3))


if __name__ == "__main__":
    main()
