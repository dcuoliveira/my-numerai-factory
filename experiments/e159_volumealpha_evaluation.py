"""E159 -- does the volume-alpha block of crypto/v2.0-volumealpha-causal help v2.0?

    python3 experiments/e159_volumealpha_evaluation.py --fit --model lgb_ktz --n-jobs 4 [--arms FULL]
    python3 experiments/e159_volumealpha_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    E159_WORKERS=8 python3 experiments/e159_volumealpha_evaluation.py           # score, after every fit

Dataset: E152's inferred reconstruction (built 2026-11-13, point_in_time_verified false, 2-day observation
lag): v2.0 + 70 feature_va_* columns, each binned per date, NaN where unavailable. Families from its manifest:
  technical 8 (the paper's lagged MA of return and log dollar volume), liquidity 13, fundamentals 17
  (size, surprise, leverage, valuation, beta, age), cross_asset 11, forecasts 11 (OOF volume forecasts),
  costs 7 + 3 participation scenarios.

This is the routed feature-set protocol of E149 (E138/E139 machinery, test = block 11 as in every
feature-set study), NOT E152's registered forward test, which has no data yet.

Arms (each routed on its own columns' coverage):
  FULL 22+70   ADDED_ONLY 70   F_TECH  F_LIQ  F_FUND  F_XASSET  F_FCST  F_COST (22 + one family)
  F_RAW (tech+liq+fund+xasset)  F_MODEL (forecasts+costs)
  FULL_H vs P_HIST (tenure), P_WARM (binned share observed), P_NM (coin never observed, look-ahead),
  tuneFULL vs tuneB0.
"""
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e135_route_histfe as Q  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e159_volumealpha"
DIR = "data/crypto/v2.0-volumealpha-causal"
MANIFEST = json.load(open(os.path.join(DIR, "manifest.json")))
FAM = MANIFEST["families"]
P.STUDY, P.ROOT, P.DATA, P.META = "E159", ROOT, os.path.join(DIR, "train.parquet"), os.path.join(DIR, "historical_meta_models.parquet")
P.REUSE = {}
BLOCKS = ["F_TECH", "F_LIQ", "F_FUND", "F_XASSET", "F_FCST", "F_COST", "F_RAW", "F_MODEL"]
P.FITS = ["FULL", "ADDED_ONLY"] + BLOCKS + ["FULL_H", "P_WARM", "P_NM"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL"] + BLOCKS + ["FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "F_RAW", "forecasts and costs, given the raw block"), ("FULL", "F_MODEL", "raw block, given forecasts and costs"),
             ("F_MODEL", "F_RAW", "forecasts and costs against raw"), ("F_FCST", "F_TECH", "volume forecasts against their inputs"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"),
             ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_TECH", "B0"), ("F_FCST", "B0"), ("ADDED_ONLY", "B0"),
           ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]


def groups(new):
    fund = sum((FAM[k] for k in ("size", "surprise", "leverage", "valuation", "beta", "age")), [])
    G = {"F_TECH": FAM["technical"], "F_LIQ": FAM["liquidity"], "F_FUND": fund, "F_XASSET": FAM["cross_asset"],
         "F_FCST": FAM["forecasts"], "F_COST": FAM["costs"] + sum((FAM[k] for k in FAM if k.startswith("scenario_")), [])}
    G["F_RAW"] = G["F_TECH"] + G["F_LIQ"] + G["F_FUND"] + G["F_XASSET"]
    G["F_MODEL"] = G["F_FCST"] + G["F_COST"]
    assert sorted(G["F_RAW"] + G["F_MODEL"]) == sorted(new), (len(G["F_RAW"] + G["F_MODEL"]), len(new))
    return G


def split(cols):
    new = [c for c in cols if c.startswith("feature_va_")]
    base = [c for c in cols if c.startswith("feature_") and c not in new]
    assert len(base) == 22 and len(new) == 70, (len(base), len(new))
    return base, new


def load():
    df = pd.read_parquet(P.DATA)
    if "symbol" not in df or "date" not in df:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base, new = split(list(df))
    G = groups(new)
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    share = df[new].notna().mean(axis=1).astype("float64").to_frame("feature_warm")
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    ever = df[new].notna().any(axis=1).groupby(df["ucid"]).transform("any")
    df["feature_nofeed"] = (~ever).astype("float32")
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[new].notna().mean(axis=1).astype("float32").values
    for a, cols in G.items():
        obs["obs_" + a] = df[cols].notna().mean(axis=1).astype("float32").values
    obs["no_feed"] = df["feature_nofeed"].values
    fill = base + new + ["feature_hist", "feature_warm"]
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, new, [], obs


def arms(base, new, _):
    G = groups(new)
    A = {"B0": base, "B0CHK": base, "FULL": base + new, "ADDED_ONLY": new, "FULL_H": base + new + ["feature_hist"],
         "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"]}
    for a, cols in G.items():
        A[a] = base + cols
    assert len(A["FULL"]) == 92 and len(A["F_TECH"]) == 30
    return A


def coverage(arm, o):
    return o["obs_" + arm] if "obs_" + arm in o else o["obs_tf"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    P.patch()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        if not a.arms and not a.tune:
            a.arms = P.FITS + (["B0CHK"] if a.model == "ridge" else [])
        T.fit(a)
        return
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path):
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        k.merge(load()[4], on=["symbol", "date"], how="left", validate="1:1").to_parquet(obs_path, index=False)
    P.score(int(os.environ.get("E159_WORKERS", "8")))


P.load, P.arms, P.coverage = load, arms, coverage
if __name__ == "__main__":
    main()
