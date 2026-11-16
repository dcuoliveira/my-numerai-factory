#!/usr/bin/env python3
"""
E143 -- the feature-set evaluation procedure for crypto/v2.0-c2syn.

    python3 experiments/e143_c2syn_evaluation.py --fit --model lgb_ktz --n-jobs 6 [--arms FULL]
    python3 experiments/e143_c2syn_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e143_c2syn_evaluation.py --audit
    python3 experiments/e143_c2syn_evaluation.py

22 shipped features + 32 Chronos-2-Synth pre-head representation PCs: each coin's panel history
(label + 22 features on the panel-date grid) run zero-shot through autogluon/chronos-2-synth, the
encoder state the forecasting head reads for date D's step, labels only from dates <= D-35 days
(src/data_mani/chronos2_syn_repr.py), reduced to 16 PCs per mode on a past-only burn-in basis and
rank-binned per date (src/data_mani/build_c2syn.py). E138's protocol, routing and bar, through
E138/E139's machinery (module globals patched below).

  FULL        22 + 16 cov + 16 ar
  ADDED_ONLY  the 32 PCs
  F_COV       + 16 cov PCs (label with the 22 features as covariates; = FULL minus ar)
  F_AR        + 16 ar PCs (label alone, temporal only; = FULL minus cov)
  FULL_H      FULL + tenure, against P_HIST
  P_WARM      B0 + binned share of the block observed (causal)
  P_NM        B0 + "this coin has no state on any row of the file" (LOOK-AHEAD by construction)
  tuneFULL    vs tuned B0 (mlp alpha capped at 1; mlp tuned B0 refitted)
Routing covers a coin when its block is observed that date.
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

ROOT = "results/reports/crypto/e143_c2syn"
DATA = "data/crypto/v2.0-c2syn/train.parquet"
P.STUDY, P.ROOT, P.DATA, P.META = "E143", ROOT, DATA, "data/crypto/v2.0-c2syn/historical_meta_models.parquet"
P.REUSE = {}
P.FITS = ["FULL", "ADDED_ONLY", "F_COV", "F_AR", "FULL_H", "P_WARM", "P_NM"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL", "F_COV", "F_AR", "FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "F_AR", "cov block, given ar"), ("FULL", "F_COV", "ar block, given cov"),
             ("F_COV", "F_AR", "with covariates vs temporal only"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"),
             ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_COV", "B0"), ("F_AR", "B0"), ("ADDED_ONLY", "B0"), ("F_COV", "F_AR"),
           ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw[raw["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    blk = [c for c in df if c.startswith("feature_c2")]
    base = [c for c in df if c.startswith("feature_") and c not in blk]
    assert len(base) == 22 and len(blk) == 32
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    share = df[blk].notna().mean(axis=1).astype("float64").to_frame("feature_warm")
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    whole = raw[blk].notna().any(axis=1).groupby([raw["symbol"], raw["ucid"]]).any().rename("ever").reset_index()
    df = df.merge(whole, on=["symbol", "ucid"], how="left", validate="m:1")
    df["feature_nofeed"] = (~df.pop("ever")).astype("float32")
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[blk].notna().mean(axis=1).astype("float32")      # the routable block, under T's column name
    obs["obs_macro"] = obs["obs_tf"]
    obs["no_feed"] = df["feature_nofeed"]
    fill = base + blk + ["feature_hist", "feature_warm"]
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, blk, [], obs


def arms(base, blk, _):
    cov = [c for c in blk if c.startswith("feature_c2cov_")]
    ar = [c for c in blk if c.startswith("feature_c2ar_")]
    A = {"B0": base, "B0CHK": base, "FULL": base + blk, "ADDED_ONLY": blk, "F_COV": base + cov, "F_AR": base + ar,
         "FULL_H": base + blk + ["feature_hist"], "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"]}
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_COV", "F_AR")] == [54, 32, 38, 38]
    return A


def audit():
    """Point-in-time rebuild (build_c2syn --audit), coverage, bin mass and redundancy."""
    from src.data_mani import build_c2syn
    cut = build_c2syn.audit()
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    blk = [c for c in ds if c.startswith("feature_c2")]
    k = N.keys()
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    base = [c for c in q if c.startswith("feature_") and c not in blk]
    C = q[blk].corr(method="spearman").abs().to_numpy()
    np.fill_diagonal(C, np.nan)
    S = q[blk + base].corr(method="spearman").loc[blk, base].abs()
    cov = [c for c in blk if "_c2cov_" in c]
    X = q[blk + ["target_binned_return_20"]].dropna()
    ic = X.groupby(q.loc[X.index, "date"])[blk + ["target_binned_return_20"]].corr(method="spearman") \
        .xs("target_binned_return_20", level=1)[blk]
    y = z.groupby([z["date"].dt.year, "kind"]).agg(observed=("obs_tf", lambda s: (s > 0).mean()), no_feed=("no_feed", "mean")).reset_index()
    out = dict(truncation=cut,
               coverage=dict(keys=len(z), observed=float((z["obs_tf"] > 0).mean()), no_feed=float(z["no_feed"].mean())),
               coverage_by_year=y.to_dict("records"),
               mass={str(a): float(b) for a, b in q[blk].stack().value_counts(normalize=True).sort_index().items()},
               redundancy=dict(mean_abs_rho=float(np.nanmean(C)), pairs_over_08=int((C > .8).sum() // 2),
                               cov_vs_ar_max=float(q[blk].corr(method="spearman").loc[cov, [c for c in blk if c not in cov]].abs().max().max())),
               vs_base={c: dict(base=S.loc[c].idxmax(), rho=float(S.loc[c].max())) for c in blk},
               date_ic={c: dict(mean=float(ic[c].mean()), t=float(ic[c].mean() / ic[c].std() * np.sqrt(len(ic) / 20))) for c in blk})
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    print({x: out[x] for x in ("truncation", "coverage", "redundancy")})


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
    P.score(int(os.environ.get("E143_WORKERS", "16")))


P.load, P.arms, P.coverage = load, arms, (lambda arm, o: o["obs_tf"])
if __name__ == "__main__":
    main()
