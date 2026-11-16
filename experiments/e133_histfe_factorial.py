#!/usr/bin/env python3
"""
E133 -- do own-history features (crypto/v2.0-histfe) add out-of-sample value over the 22
shipped features, and at which horizons?

    python3 experiments/e133_histfe_factorial.py --model lgb_ktz --arms B0 FULL --seeds 0 1 2 3 4
    python3 experiments/e133_histfe_factorial.py --model ridge --tune --arms B0 FULL

Design (registered in experiments/EXPERIMENTS.md, E133, before launch):
  rows    every row of crypto/v2.0-histfe dated before 2026-02-20. Later dates are dropped
          at load: the final test period is neither fitted nor scored here.
  label   target_binned_return_20
  blocks  E131's 11 dev blocks: each trains on 130 dates, purges 22 (the target overlaps
          20) and predicts the next 130, so predictions run 2020-09 to 2026-02-19.
  configs the E131 fixed configs, all searched on raw v2.0, never on these features:
          ridge alpha 65.79, lgb_ktz and mlp_ktz modal picks.
  seeds   ridge is deterministic; lgb_ktz and mlp_ktz get 5.
  arms    B0 the 22 shipped features; SHORT / LONG add the short / long horizons, split
          at 30 days; COMBO adds both; TSC adds days-since-change (it has no horizon);
          FULL adds all 352; ADDED_ONLY is the 352 alone; F_RANK / F_MA / F_TSMAX add one
          family at all horizons; P_HIST adds tenure, as the control for "the block is
          only proxying coin age".
  --tune  equal-budget retune: the SAME candidates on B0 and FULL, picked per block on
          the last 40 train dates after a 22-date purge, refit on the whole window.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.history_features import (bin_ties_preserved, history_columns,  # noqa: E402
                                            is_short)
from src.prediction.models import build_wrapper  # noqa: E402

T20 = "target_binned_return_20"
DATA = "data/crypto/v2.0-histfe/train.parquet"
META = "data/crypto/v2.0-histfe/historical_meta_models.parquet"
OUT = "results/reports/crypto/e133_histfe/preds"
TEST_START = pd.Timestamp("2026-02-20")     # nothing on or after this date is read
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172
INNER_VAL = 40

FIXED = {
    "ridge": {"alpha": 65.79332246575683, "fit_intercept": True},
    "lgb_ktz": {"learning_rate": 0.002848035868435802, "colsample_bytree": 0.17, "num_leaves": 4,
                "n_estimators": 4000, "max_depth": -1, "min_child_samples": 100, "subsample": 0.8,
                "subsample_freq": 1, "reg_lambda": 10.0, "deterministic": True, "force_row_wise": True,
                "verbose": -1},
    "mlp_ktz": {"mlp__hidden_layer_sizes": (32, 16), "mlp__activation": "identity",
                "mlp__learning_rate_init": 0.05011872336272722, "mlp__alpha": 0.01,
                "mlp__batch_size": 1024},
}
GRID = {
    "ridge": {"alpha": [1, 10, 65.79332246575683, 316, 1000, 10000]},
    "lgb_ktz": {"num_leaves": [4, 8], "colsample_bytree": [0.1, 0.17, 0.3]},
    "mlp_ktz": {"mlp__alpha": [0.01, 1.0, 100.0], "mlp__learning_rate_init": [0.01, 0.05011872336272722]},
}


def load():
    keep = pd.read_parquet(DATA, columns=["date"])["date"]
    df = pd.read_parquet(DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < TEST_START].reset_index(drop=True)
    assert df["date"].max() < TEST_START and len(keep) > len(df)
    base = [c for c in df if c.startswith("feature_") and not c.startswith("feature_hfe_")]
    new = [c for c in df if c.startswith("feature_hfe_")]
    assert len(base) == 22 and len(new) == 352
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    # tenure: the token's EARLIER panel dates, rank-binned per date like the block
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    df[base + new + ["feature_hist"]] = df[base + new + ["feature_hist"]].fillna(0.5)
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, new


def arms(base, new):
    fam = history_columns(base)
    short = [c for c in new if is_short(c) is True]
    long_ = [c for c in new if is_short(c) is False]
    return {"B0": base, "SHORT": base + short, "LONG": base + long_, "COMBO": base + short + long_,
            "TSC": base + fam["tschg"], "FULL": base + new, "ADDED_ONLY": new,
            "F_RANK": base + fam["rank"], "F_MA": base + fam["ma"], "F_TSMAX": base + fam["tsmax"],
            "P_HIST": base + ["feature_hist"]}


def blocks(dates):
    n = len(dates)
    starts = list(range(FIRST, n - BLOCK + 1, BLOCK))
    assert len(starts) == 11 and starts[-1] + BLOCK == n, (starts, n)
    return [(i, dates[b - PURGE - TRAIN:b - PURGE], dates[b:b + BLOCK]) for i, b in enumerate(starts)]


def fit_predict(model, params, seed, n_jobs, Xtr, ytr, Xte):
    est = build_wrapper(model, "regression", seed=seed, n_jobs=n_jobs).ModelClass
    p = dict(params)
    if model == "lgb_ktz":
        p.update(random_state=seed, n_jobs=n_jobs)
    elif model == "mlp_ktz":
        p.update({"mlp__random_state": seed})
    est.set_params(**p)
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def candidates(model):
    keys = list(GRID[model])
    return [dict(FIXED[model], **dict(zip(keys, v))) for v in itertools.product(*GRID[model].values())]


def pick(model, df, cols, train, seed, n_jobs):
    """Best candidate on the last INNER_VAL train dates, after a PURGE-date gap."""
    fit_d, val_d = train[:-(INNER_VAL + PURGE)], train[-INNER_VAL:]
    fit, val = df[df["date"].isin(fit_d)], df[df["date"].isin(val_d)]
    best, score = None, -np.inf
    for params in candidates(model):
        pred = fit_predict(model, params, seed, n_jobs, fit[cols].to_numpy(np.float32), fit[T20].to_numpy(),
                           val[cols].to_numpy(np.float32))
        s = val.assign(p=pred).groupby("date").apply(lambda g: g["p"].corr(g[T20], method="spearman")).mean()
        if s > score:
            best, score = params, s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FIXED))
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("E133_JOBS", "4")))
    ap.add_argument("--layout-only", action="store_true")
    args = ap.parse_args()

    df, base, new = load()
    A = arms(base, new)
    dates = np.array(sorted(df["date"].unique()))
    B = blocks(dates)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "layout.json"), "w") as h:
        json.dump({"arms": A, "test_start": str(TEST_START.date()), "train": TRAIN, "purge": PURGE, "block": BLOCK,
                   "blocks": [{"block": i, "train": [str(t[0])[:10], str(t[-1])[:10]],
                               "predict": [str(p[0])[:10], str(p[-1])[:10]]} for i, t, p in B]}, h, indent=1)
    if args.layout_only:
        for i, t, p in B:
            print("%2d train %s..%s predict %s..%s" % (i, str(t[0])[:10], str(t[-1])[:10], str(p[0])[:10], str(p[-1])[:10]))
        print({k: len(v) for k, v in A.items()})
        return
    seeds = [0] if (args.model == "ridge" or args.tune) else args.seeds
    keep = ["symbol", "date", T20, "meta_model"]
    tag = "tune" if args.tune else ""
    for arm in (args.arms or list(A)):
        cols = A[arm]
        for s in seeds:
            path = os.path.join(OUT, args.model, "%s%s_s%d.parquet" % (tag, arm, s))
            if os.path.exists(path):
                print("skip", path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, picks, t1 = [], [], time.time()
            for i, tr, te in B:
                trm, tem = df["date"].isin(tr), df["date"].isin(te)
                params = pick(args.model, df, cols, tr, s, args.n_jobs) if args.tune else FIXED[args.model]
                picks.append({k: str(params.get(k)) for k in GRID[args.model]})
                pred = fit_predict(args.model, params, s, args.n_jobs, df.loc[trm, cols].to_numpy(np.float32),
                                   df.loc[trm, T20].to_numpy(), df.loc[tem, cols].to_numpy(np.float32))
                o = df.loc[tem, keep].copy()
                o["prediction"], o["block"] = pred, i
                parts.append(o)
            tmp = path + ".partial"
            pd.concat(parts).to_parquet(tmp, index=False)
            os.replace(tmp, path)
            if args.tune:
                with open(path + ".picks.json", "w") as h:
                    json.dump(picks, h)
            print("%s %-10s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


if __name__ == "__main__":
    main()
