#!/usr/bin/env python3
"""
E131 -- does crypto/v2.0-litmom-fillind add out-of-sample value over crypto/v2.0?

Fixed-configuration factorial: model family x feature-set arm, refit from scratch on
every chronological block, predictions saved per (model, arm, seed) for the analysis
script (experiments/e131_analysis.py).

    python3 experiments/e131_litmom_fillind_factorial.py --model lgb_ktz --arms B0 FULL --seeds 0 1 2 3 4

Design (pre-registered in experiments/EXPERIMENTS.md, E131):
  rows    the 513,496 (symbol, date) keys of crypto/v2.0-litmom-fillind -- a strict
          date-prefix of crypto/v2.0 with identical targets and base values (audit/schema.md)
  label   target_binned_return_20
  blocks  dates sorted; block start b in 172, 302, ..., 1472 (11 DEV blocks) and the TEST
          block b = N-130 = 1602 (2026-02-20..2026-08-20). Each block trains on the 130 dates
          [b-152, b-22) -- a 22-date purge -- and predicts the 130 dates [b, b+130).
          Rows of one date never straddle train/predict. Every prediction is out of sample.
          The TEST block was already scored by E65 (litmom vs base); it is untouched with
          respect to every E131 choice (indicators, fills, blend weights), not w.r.t. E65.
  configs ONE fixed previously-searched configuration per family (audit/provenance.md):
          "ridge_ktz" does not exist -> ridge (alpha 65.79, the modal raw-v2.0 pick);
          "gbm_ktz" does not exist -> lgb_ktz (latest-refit pick of e29/e64/e65/e72);
          mlp_ktz modal pick. All three were SEARCHED ON B0 (raw v2.0), never on FULL.
  seeds   ridge is deterministic (one fit); lgb_ktz and mlp_ktz get --seeds.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.binning import bin_cross_sectional  # noqa: E402
from src.prediction.models import build_wrapper  # noqa: E402

T20 = "target_binned_return_20"
CAND = "data/crypto/v2.0-litmom-fillind/train.parquet"
PARENT = "data/crypto/v2.0-litmom/train.parquet"        # pre-fill values (for the fallback-fill arm)
V20 = "data/crypto/v2.0/train.parquet"                  # the original baseline file (B0)
META = "data/crypto/v2.0-litmom-fillind/historical_meta_models.parquet"
COVERAGE = "data/crypto/spot/coverage.csv"
OUT = "results/reports/crypto/e131_litmom_fillind/preds"
TRAIN, PURGE, BLOCK, FIRST = 130, 22, 130, 172

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

RET = ["ret_7d", "ret_14d", "ret_21d", "ret_28d", "ret_56d", "ret_112d", "ret_12_1"]
MAXV = ["max1_28d", "max5_28d", "volscaled_28d"]
TREND = ["trend_7d", "trend_14d", "trend_30d", "trend_60d", "trend_120d", "trend_240d", "trend_robust"]
FAMILIES = {"ret": RET, "maxvol": MAXV, "trend": TREND}
BASE_IND_SRC = ["bollinger_60d", "momentum_60d", "relative_strength_index_20d", "relative_strength_index_60d"]


def groups(cols):
    base = [c for c in cols if c.startswith("feature_") and not c.startswith(("feature_mom_", "feature_isna_"))]
    mom = ["feature_mom_" + x for x in RET + MAXV + TREND]
    mom_ind = ["feature_isna_mom_" + x for x in RET + MAXV + TREND]
    base_ind = ["feature_isna_" + x for x in BASE_IND_SRC]
    assert len(base) == 22 and all(c in cols for c in mom + mom_ind + base_ind), "schema drift"
    assert len(base) + len(mom) == 39 and len(base_ind) + len(mom_ind) == 21
    return base, mom, mom_ind, base_ind


def load():
    """One frame on the candidate's keys with every column any arm needs."""
    c = pd.read_parquet(CAND).reset_index()
    base, mom, mom_ind, base_ind = groups(list(c.columns))
    assert not c.duplicated(["symbol", "date"]).any()
    # B0: the ORIGINAL v2.0 base values on the same keys, with the loader's own 0.5 fill
    v = pd.read_parquet(V20).reset_index()[["symbol", "date", T20] + base]
    v = v.rename(columns={b: "v20__" + b for b in base}).rename(columns={T20: "v20__t20"})
    c = c.merge(v, on=["symbol", "date"], how="left", validate="1:1")
    assert c["v20__t20"].notna().all() and (c["v20__t20"] == c[T20]).all(), "B0 row/target mismatch"
    for b in base:
        c["v20__" + b] = c["v20__" + b].fillna(0.5).astype("float32")
    # alternative fill (M1FB): a missing mom value takes the nearest SHORTER-window value of
    # its own family from the same row (all inputs are already point-in-time), else 0.5
    p = pd.read_parquet(PARENT).reset_index()[["symbol", "date"] + mom]
    p = p.rename(columns={m: "fb__" + m for m in mom})
    c = c.merge(p, on=["symbol", "date"], how="left", validate="1:1")
    for fam in (RET, TREND):
        for i, x in enumerate(fam):
            col = "fb__feature_mom_" + x
            for y in reversed(fam[:i]):
                c[col] = c[col].fillna(c["fb__feature_mom_" + y])
    c["fb__feature_mom_volscaled_28d"] = c["fb__feature_mom_volscaled_28d"].fillna(c["fb__feature_mom_ret_28d"])
    for m in mom:
        c["fb__" + m] = c["fb__" + m].fillna(0.5).astype("float32")
    # placebo P_NM: the spot-feed match flag, assembled 2026-09-22 -- LOOK-AHEAD by construction;
    # it measures how much of any gain the survivorship channel alone explains (audit/leakage.md)
    cov = pd.read_csv(COVERAGE)[["symbol", "matched"]].drop_duplicates("symbol")
    c = c.merge(cov, on="symbol", how="left")
    c["feature_never_matched"] = (c["matched"].fillna(0) == 0).astype("float32")
    # control P_HIST: causal panel tenure (count of the symbol's EARLIER panel dates), rank-binned per date
    c = c.sort_values(["date", "symbol"]).reset_index(drop=True)
    c["feature_hist"] = c.groupby("symbol").cumcount().astype("float64")
    c = bin_cross_sectional(c, ["feature_hist"], period_col="date")
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    c = c.merge(meta, on=["symbol", "date"], how="left")
    return c, (base, mom, mom_ind, base_ind)


def arms(g):
    base, mom, mom_ind, base_ind = g
    v20 = ["v20__" + b for b in base]
    a = {
        "B0": v20,                                   # original crypto/v2.0 values
        "B0C": base,                                 # the candidate file's copy of them
        "B1": base + base_ind,
        "M1": base + mom,
        "M2": base + mom + mom_ind,
        "FULL": base + mom + mom_ind + base_ind,
        "ADDED_ONLY": mom + mom_ind,
        "M1FB": base + ["fb__" + m for m in mom],     # alternative fill, no indicators
        "P_NM": base + ["feature_never_matched"],    # survivorship placebo (look-ahead)
        "P_HIST": base + ["feature_hist"],           # causal tenure control
        "MISS_ONLY": mom_ind + base_ind,             # missingness-only inputs
    }
    for fam, xs in FAMILIES.items():                 # leave-one-family-out from M1
        a["M1_no_" + fam] = base + [m for m in mom if m.split("feature_mom_")[1] not in xs]
    return a


def blocks(dates):
    n = len(dates)
    starts = list(range(FIRST, n - BLOCK - BLOCK + 1, BLOCK))
    test = n - BLOCK
    assert starts[-1] + BLOCK == test, (starts[-1], test)
    out = [(i, b, "dev") for i, b in enumerate(starts)] + [(len(starts), test, "test")]
    return [(i, dates[b - PURGE - TRAIN:b - PURGE], dates[b - PURGE:b], dates[b:b + BLOCK], kind)
            for i, b, kind in out]


def fit_predict(model, params, seed, n_jobs, Xtr, ytr, Xte):
    w = build_wrapper(model, "regression", seed=seed, n_jobs=n_jobs)
    est = w.ModelClass
    p = dict(params)
    if model == "lgb_ktz":
        p.update(random_state=seed, n_jobs=n_jobs)
    elif model == "mlp_ktz":
        p.update({"mlp__random_state": seed})
    est.set_params(**p)
    est.fit(Xtr, ytr)
    return est.predict(Xte), est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FIXED))
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("E131_JOBS", "4")))
    ap.add_argument("--layout-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df, g = load()
    A = arms(g)
    dates = np.array(sorted(df["date"].unique()))
    B = blocks(dates)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "layout.json"), "w") as h:
        json.dump({"n_dates": len(dates), "blocks": [
            {"block": i, "kind": k, "train": [str(tr[0])[:10], str(tr[-1])[:10]], "purge": [str(pu[0])[:10], str(pu[-1])[:10]],
             "predict": [str(te[0])[:10], str(te[-1])[:10]]} for i, tr, pu, te, k in B],
            "arms": {k: v for k, v in A.items()}, "fixed": {m: {k: (list(x) if isinstance(x, tuple) else x) for k, x in d.items()} for m, d in FIXED.items()},
            "train_dates": TRAIN, "purge_dates": PURGE, "block_dates": BLOCK}, h, indent=1, default=str)
    if args.layout_only:
        for i, tr, pu, te, k in B:
            print("%2d %-4s train %s..%s purge %s..%s predict %s..%s" % (i, k, str(tr[0])[:10], str(tr[-1])[:10],
                  str(pu[0])[:10], str(pu[-1])[:10], str(te[0])[:10], str(te[-1])[:10]))
        print({k: len(v) for k, v in A.items()})
        return
    seeds = [0] if args.model == "ridge" else args.seeds
    keep = ["symbol", "date", T20, "meta_model", "matched", "feature_hist"]
    for arm in (args.arms or list(A)):
        cols = A[arm]
        for s in seeds:
            path = os.path.join(OUT, args.model, "%s_s%d.parquet" % (arm, s))
            if os.path.exists(path):
                print("skip", path); continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, t1 = [], time.time()
            for i, tr, pu, te, kind in B:
                trm, tem = df["date"].isin(tr), df["date"].isin(te)
                pred, _ = fit_predict(args.model, FIXED[args.model], s, args.n_jobs,
                                      df.loc[trm, cols].to_numpy(np.float32), df.loc[trm, T20].to_numpy(),
                                      df.loc[tem, cols].to_numpy(np.float32))
                o = df.loc[tem, keep].copy()
                o["prediction"], o["block"], o["kind"] = pred, i, kind
                parts.append(o)
            pd.concat(parts).to_parquet(path, index=False)
            print("%s %-12s seed %d  %d cols  %.0fs" % (args.model, arm, s, len(cols), time.time() - t1), flush=True)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
