#!/usr/bin/env python3
"""
E141 -- the feature-set evaluation procedure for crypto/v2.0-macrobeta.

    python3 experiments/e141_macrobeta_evaluation.py --fit --model lgb_ktz --n-jobs 6 [--arms FULL]
    python3 experiments/e141_macrobeta_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e141_macrobeta_evaluation.py --audit
    python3 experiments/e141_macrobeta_evaluation.py

22 shipped features + 32 causal rolling betas of each coin's log return on the one-day change of
16 macro / inflation factors (gold, silver, S&P, DXY; y1..y30; 3 slopes; 4 break-evens) over 60 and
120 macro business days, rank-binned per date (src/data_mani/build_macro_betas.py). E138's protocol,
routing and bar, through E138/E139's machinery (module globals patched below).

  FULL        22 + 32 betas
  ADDED_ONLY  the 32 betas
  F_PRICE     + 8 betas to gold / silver / S&P / DXY
  F_RATES     + 16 betas to yields and slopes
  F_INFL      + 8 betas to break-evens
  W60, W120   + one window's 16 betas
  NO_PRICE NO_RATES NO_INFL   FULL with one factor group left out
  FULL_H      FULL + tenure, against P_HIST
  P_WARM      B0 + binned share of the beta block observed (causal)
  P_NM        B0 + "this coin has no beta on any row of the file" (LOOK-AHEAD by construction)
  tuneFULL    vs tuned B0 (mlp alpha capped at 1; mlp tuned B0 refitted)
Routing covers a coin when its beta block is observed that date.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e141_macrobeta"
DATA = "data/crypto/v2.0-macrobeta/train.parquet"
GROUPS = {"PRICE": ("gold_", "silver_", "snp_", "dxy_"), "RATES": ("y1_", "y2_", "y5_", "y10_", "y30_", "slope_"), "INFL": ("bei_",)}
P.STUDY, P.ROOT, P.DATA, P.META = "E141", ROOT, DATA, "data/crypto/v2.0-macrobeta/historical_meta_models.parquet"
P.REUSE = {}
P.FITS = ["FULL", "ADDED_ONLY"] + ["F_" + g for g in GROUPS] + ["W60", "W120"] + ["NO_" + g for g in GROUPS] + ["FULL_H", "P_WARM", "P_NM"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL"] + ["F_" + g for g in GROUPS] + ["W60", "W120"] + ["NO_" + g for g in GROUPS] + ["FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "NO_" + g, "%s betas, given the rest" % g.lower()) for g in GROUPS] + \
            [("FULL", "W60", "120-day window, given 60"), ("FULL", "W120", "60-day window, given 120"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"), ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_PRICE", "B0"), ("F_RATES", "B0"), ("F_INFL", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]


def fam(cols, keys):
    return [c for c in cols if c[len("feature_mbeta_"):].startswith(keys)]


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw[raw["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    mb = [c for c in df if c.startswith("feature_mbeta_")]
    base = [c for c in df if c.startswith("feature_") and c not in mb]
    assert len(base) == 22 and len(mb) == 32
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    share = df[mb].notna().mean(axis=1).astype("float64").to_frame("feature_warm")
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    whole = raw[mb].notna().any(axis=1).groupby([raw["symbol"], raw["ucid"]]).any().rename("ever").reset_index()
    df = df.merge(whole, on=["symbol", "ucid"], how="left", validate="m:1")
    df["feature_nofeed"] = (~df.pop("ever")).astype("float32")
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[mb].notna().mean(axis=1).astype("float32")      # the routable block, under T's column name
    obs["obs_macro"] = obs["obs_tf"]
    obs["no_feed"] = df["feature_nofeed"]
    fill = base + mb + ["feature_hist", "feature_warm"]
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, mb, [], obs


def arms(base, mb, _):
    A = {"B0": base, "B0CHK": base, "FULL": base + mb, "ADDED_ONLY": mb, "FULL_H": base + mb + ["feature_hist"],
         "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"],
         "W60": base + [c for c in mb if c.endswith("_60d")], "W120": base + [c for c in mb if c.endswith("_120d")]}
    for g, k in GROUPS.items():
        A["F_" + g] = base + fam(mb, k)
        A["NO_" + g] = [c for c in base + mb if c not in set(fam(mb, k))]
    assert [len(A["F_" + g]) for g in GROUPS] == [30, 38, 30] and len(A["W60"]) == 38
    return A


def audit():
    """Rebuild the betas from panel, spot and macro cut at T and compare row T; redundancy."""
    from src.data_mani.build_feed_fillind import SPOT
    from src.data_mani.build_macro_betas import MACRO, add_macro_betas
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    v = pd.read_parquet("data/crypto/v2.0/train.parquet").reset_index()
    v["date"] = pd.to_datetime(v["date"])
    v["ucid"] = v["ucid"].astype(str)
    spot = pd.read_parquet(SPOT, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    mac = pd.read_parquet(MACRO)
    mac["date"] = pd.to_datetime(mac["date"])
    mb = [c for c in ds if c.startswith("feature_mbeta_")]
    rows = []
    for t in ("2021-06-30", "2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        t = pd.Timestamp(t)
        cut, _ = add_macro_betas(v[v["date"] <= t].reset_index(drop=True), spot[spot["date"] <= t], mac[mac["date"] <= t])
        a = cut.loc[cut["date"] == t].sort_values("symbol")[mb].to_numpy("float64")
        b = ds.loc[ds["date"] == t].sort_values("symbol")[mb].to_numpy("float64")
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(t.date()), rows=len(a), cells=int(eq.size), differ=int((~eq).sum())))
        print(rows[-1], flush=True)
    k = N.keys()
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    base = [c for c in q if c.startswith("feature_") and c not in mb]
    C = q[mb].corr(method="spearman").abs().to_numpy()
    np.fill_diagonal(C, np.nan)
    S = q[mb + base].corr(method="spearman").loc[mb, base].abs()
    y = z.groupby([z["date"].dt.year, "kind"]).agg(tf_any=("obs_tf", lambda s: (s > 0).mean()), tf_half=("obs_tf", lambda s: (s >= .5).mean()), no_feed=("no_feed", "mean")).reset_index()
    out = dict(truncation=rows,
               coverage=dict(keys=len(z), tf_any=float((z["obs_tf"] > 0).mean()), tf_half=float((z["obs_tf"] >= .5).mean()), macro_any=float((z["obs_tf"] > 0).mean()),
                             no_feed=float(z["no_feed"].mean()), warm_given_feed=float(((z["obs_tf"] == 0) & (z["no_feed"] == 0)).mean())),
               coverage_by_year=y.to_dict("records"),
               mass_tf={str(a): float(b) for a, b in q[mb].stack().value_counts(normalize=True).sort_index().items()},
               macro_distinct_per_date=int(q.groupby("date")[mb].nunique().min().min()),
               beta_redundancy=dict(mean_abs_rho=float(np.nanmean(C)), pairs_over_08=int((C > .8).sum() // 2)),
               beta_vs_base={c: dict(base=S.loc[c].idxmax(), rho=float(S.loc[c].max())) for c in mb})
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    print({x: out[x] for x in ("truncation", "coverage", "beta_redundancy")})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
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
    if a.audit:
        audit()
        return
    P.score(int(os.environ.get("E141_WORKERS", "16")))


P.load, P.arms, P.coverage = load, arms, (lambda arm, o: o["obs_tf"])
if __name__ == "__main__":
    main()
