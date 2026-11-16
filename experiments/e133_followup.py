#!/usr/bin/env python3
"""
E133 follow-ups to e133_analysis.py, on the same dev predictions:
  - lgb F_RANK against the tenure control, per metric, per seed, per payout block;
  - per-block level of the tuned arms, to see what the MLP retune did;
  - B0 + w*ADDED_ONLY blends on PAYOUT, weight picked leave-one-block-out.

    python3 experiments/e133_followup.py --workers 8
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
ROOT = "results/reports/crypto/e133_histfe"
PRED = os.path.join(ROOT, "preds")
PAIRS = [("F_RANK", "B0"), ("F_RANK", "P_HIST"), ("SHORT", "B0"), ("SHORT", "P_HIST"), ("FULL", "P_HIST")]


def blend_pay(job):
    model, seed = job
    a = pd.read_parquet(os.path.join(PRED, model, "B0_s%d.parquet" % seed))
    b = pd.read_parquet(os.path.join(PRED, model, "ADDED_ONLY_s%d.parquet" % seed), columns=["symbol", "date", "prediction"])
    f = a.merge(b.rename(columns={"prediction": "added"}), on=["symbol", "date"], validate="1:1")
    f = f[f["meta_model"].notna()]
    f["date"] = pd.to_datetime(f["date"])
    f = f[f.groupby("date")["meta_model"].transform("count") >= 10]
    f["prediction"] = f.groupby("date")["prediction"].rank(pct=True)
    f["added"] = f.groupby("date")["added"].rank(pct=True)
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        for w in E.W:
            o = g.assign(prediction=(1 - w) * g["prediction"] + w * g["added"])
            rows.append(dict(model=model, seed=seed, date=d, block=int(g["block"].iloc[0]), w=w,
                             payout=0.1 * E.corr_one(o) + E.contrib_one(o, "meta_model")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"))
    fixed = S[S["tune"] == ""]

    rows = []
    for model in E.MODELS:
        w = fixed[fixed["model"] == model]
        for metric in ("corr", "mmc", "payout"):
            p = w.pivot_table(index="date", columns="arm", values=metric)
            for a, b in PAIRS:
                d = (p[a] - p[b]).dropna()
                r = E.describe(d)
                rows.append(dict(model=model, metric=metric, arm=a, vs=b, delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "followup_pairs.csv"), index=False)

    # CORR and payout on the SAME 468 dates, because the headline CORR covers 1430
    window = fixed[fixed["mmc"].notna()]["date"].unique()
    w = fixed[fixed["date"].isin(window)]
    lev = w.groupby(["model", "arm", "date"])[["corr", "mmc", "payout"]].mean().groupby(["model", "arm"]).mean()
    lev.to_csv(os.path.join(ROOT, "levels_payout_window.csv"))

    g = fixed[(fixed["model"] == "lgb_ktz") & fixed["arm"].isin(["B0", "F_RANK", "P_HIST"])]
    by_seed = g.pivot_table(index=["seed", "date"], columns="arm", values="payout").dropna()
    by_block = g.pivot_table(index=["block", "date"], columns="arm", values="payout").dropna()
    seed = pd.DataFrame({"F_RANK-B0": (by_seed["F_RANK"] - by_seed["B0"]).groupby("seed").mean(),
                         "F_RANK-P_HIST": (by_seed["F_RANK"] - by_seed["P_HIST"]).groupby("seed").mean()})
    block = pd.DataFrame({"F_RANK-B0": (by_block["F_RANK"] - by_block["B0"]).groupby("block").mean(),
                          "F_RANK-P_HIST": (by_block["F_RANK"] - by_block["P_HIST"]).groupby("block").mean(),
                          "n": by_block.groupby("block").size()})
    seed.to_csv(os.path.join(ROOT, "lgb_frank_by_seed.csv"))
    block.to_csv(os.path.join(ROOT, "lgb_frank_by_block.csv"))

    t = S[S["tune"] == "tune"].pivot_table(index=["model", "block"], columns="arm", values="corr", aggfunc="mean")
    t.to_csv(os.path.join(ROOT, "tuned_corr_by_block.csv"))
    picks = {}
    for model in E.MODELS:
        for arm in ("B0", "FULL"):
            picks["%s %s" % (model, arm)] = json.load(open(os.path.join(PRED, model, "tune%s_s0.parquet.picks.json" % arm)))
    json.dump(picks, open(os.path.join(ROOT, "tuned_picks.json"), "w"), indent=1)

    jobs = [("ridge", 0)] + [(m, s) for m in ("lgb_ktz", "mlp_ktz") for s in range(5)]
    with ProcessPoolExecutor(args.workers) as ex:
        B = pd.concat(ex.map(blend_pay, jobs), ignore_index=True)
    out = []
    for model, g in B.groupby("model"):
        p = g.pivot_table(index=["block", "date"], columns="w", values="payout")
        means = p.groupby("block").mean()
        chosen = []
        for blk in means.index:
            pick = means.drop(index=blk).mean().idxmax()
            x = p.xs(blk, level="block")
            chosen.append(x[pick] - x[0.0])
        r = E.describe(pd.concat(chosen).values)
        out.append(dict(model=model, w_full=means.mean().idxmax(), delta_cv=r["mean"], t=r["nw_t"], n=r["n"],
                        **{"w%.2f" % w: p[w].mean() for w in E.W}))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(ROOT, "blend_payout.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(lev.round(4).to_string())
    print(seed.round(4).to_string())
    print(block.round(4).to_string())
    print(t.round(4).to_string())
    for k, v in picks.items():
        print(k, [list(x.values()) for x in v] if "mlp" in k else "")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
