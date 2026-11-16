"""E149 -- does the proposed realised-volatility block of crypto/v2.0-vol-causal help v2.0?

    python3 experiments/e149_vol_causal_evaluation.py --fit --model lgb_ktz --n-jobs 4 [--arms FULL]
    python3 experiments/e149_vol_causal_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    E149_WORKERS=8 python3 experiments/e149_vol_causal_evaluation.py           # score, after every fit

Dataset (src/data_mani/build_vol_causal.py, 2026-11-12): v2.0 + 154 PROPOSED columns feature_vol_*:
  22 levels   close-to-close (sqret 5..120d), EWMA (lambda .90..99), Parkinson and Yang-Zhang (5..120d)
  132 ranks   each level ranked against the same coin's own daily history over 7..365 days (_tsr<w>)
attached as-of D-1, behind a causal wrong-coin identity gate, binned per date with Numerai's rule.

The candidate is ONLY feature_vol_*. The shipped feature_volatility_20d / _60d (and feature_volume_*)
are part of the 22 base features: they stay in B0 and in every arm, never in the candidate block.
F_SWAP is the one arm that drops them, to ask whether the proposed block can stand in for them.

Arms (E146's protocol through E138/E139's machinery; each routed on its own block's coverage):
  FULL        22 + 154                       ADDED_ONLY   the 154 alone
  F_LVL       22 + the 22 levels              F_TSR        22 + the 132 own-history ranks
  F_SQRET / F_EWMA / F_PARK / F_YZ            22 + one estimator family (levels + its ranks)
  F_SHORT     22 + short arms (levels <= 20d / lambda .90-.94, ranks over 7-30 days)
  F_LONG      22 + long arms  (levels >= 40d / lambda .97-.99, ranks over 90-365 days)
  F_SWAP      FULL minus the two shipped volatility features (20 + 154)
  FULL_H vs P_HIST (tenure), P_WARM (binned share of the block observed), P_NM (look-ahead: the
  coin never passes the gate anywhere in the file), tuneFULL vs tuneB0 (mlp grid capped at alpha 1).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e135_route_histfe as Q  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e149_vol_causal"
DIR = "data/crypto/v2.0-vol-causal"
MANIFEST = json.load(open(os.path.join(DIR, "manifest.json")))
FAMS = ("sqret", "ewma", "park", "yz")
OLD_VOL = ["feature_volatility_20d", "feature_volatility_60d"]   # shipped: base, never candidate
P.STUDY, P.ROOT, P.DATA, P.META = "E149", ROOT, os.path.join(DIR, "train.parquet"), os.path.join(DIR, "historical_meta_models.parquet")
P.REUSE = {}
BLOCKS = ["F_LVL", "F_TSR"] + ["F_" + f.upper() for f in FAMS] + ["F_SHORT", "F_LONG", "F_SWAP"]
P.FITS = ["FULL", "ADDED_ONLY"] + BLOCKS + ["FULL_H", "P_WARM", "P_NM"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL"] + BLOCKS + ["FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "F_LVL", "own-history ranks, given the levels"), ("FULL", "F_TSR", "levels, given the ranks"),
             ("F_TSR", "F_LVL", "ranks against levels"), ("F_LONG", "F_SHORT", "long against short"),
             ("F_SWAP", "FULL", "dropping the shipped volatility"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"),
             ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_LVL", "B0"), ("F_TSR", "B0"), ("ADDED_ONLY", "B0"), ("F_SWAP", "B0"),
           ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]


def groups(new):
    lvl = [c for a in MANIFEST["arms"].values() for c in a]
    tsr = [c for a in MANIFEST["tsr_arms"].values() for c in a]
    assert sorted(lvl + tsr) == sorted(new) and len(lvl) == 22 and len(tsr) == 132
    G = {"F_LVL": lvl, "F_TSR": tsr}
    for f in FAMS:
        G["F_" + f.upper()] = MANIFEST["arms"][f + "_short"] + MANIFEST["arms"][f + "_long"] + \
            MANIFEST["tsr_arms"]["tsr_%s_short" % f] + MANIFEST["tsr_arms"]["tsr_%s_long" % f]
    for h in ("short", "long"):
        G["F_" + h.upper()] = [c for f in FAMS for c in MANIFEST["arms"]["%s_%s" % (f, h)] + MANIFEST["tsr_arms"]["tsr_%s_%s" % (f, h)]]
    assert sorted(sum((G["F_" + f.upper()] for f in FAMS), [])) == sorted(new)
    assert sorted(G["F_SHORT"] + G["F_LONG"]) == sorted(new)
    return G


def split(cols):
    new = [c for c in cols if c.startswith("feature_vol_")]          # the proposed block; NOT feature_volatility_*
    base = [c for c in cols if c.startswith("feature_") and c not in new]
    assert len(base) == 22 and len(new) == 154 and all(c in base for c in OLD_VOL), (len(base), len(new))
    return base, new


def load():
    df = pd.read_parquet(P.DATA).reset_index()
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
         "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"],
         "F_SWAP": [c for c in base if c not in OLD_VOL] + new}
    for a, cols in G.items():
        A[a] = base + cols
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_LVL", "F_TSR", "F_SQRET", "F_EWMA", "F_SHORT", "F_SWAP")] == \
        [176, 154, 44, 154, 64, 50, 22 + len(G["F_SHORT"]), 174]
    return A


def coverage(arm, o):
    if arm == "F_SWAP":
        return o["obs_tf"]
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
    P.score(int(os.environ.get("E149_WORKERS", "8")))


P.load, P.arms, P.coverage = load, arms, coverage
if __name__ == "__main__":
    main()
