#!/usr/bin/env python3
"""
E142 -- the feature-set evaluation procedure on crypto/v2.0-future-trends, a LOOK-AHEAD positive control.

    python3 experiments/e142_future_trends_evaluation.py --fit --model lgb_ktz --n-jobs 6
    python3 experiments/e142_future_trends_evaluation.py --audit
    python3 experiments/e142_future_trends_evaluation.py          # score, after every fit exists

22 shipped features + feature_future_ret_20d, each coin's spot return over the NEXT 20 calendar
days, rank-binned per date (src/data_mani/build_future_trends.py). The column knows the future, so
this is a sanity check of the procedure, not a candidate: it must come out far ahead of B0, by more
than any honest block has, and the verdict has to say LEAKAGE rather than ADOPT.

  FULL        22 + the future return
  ADDED_ONLY  the future return alone
Routing covers a coin when its future return is observed that date. No retune arm: there is
nothing to tune for. E138's protocol, seeds, blocks and bar, through E138/E139's machinery.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e133_histfe_factorial as H  # noqa: E402
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.build_future_trends import COLUMN, SPOT, future_return  # noqa: E402

ROOT = "results/reports/crypto/e142_future_trends"
DATA = "data/crypto/v2.0-future-trends/train.parquet"
P.STUDY, P.ROOT, P.DATA, P.META = "E142", ROOT, DATA, "data/crypto/v2.0-future-trends/historical_meta_models.parquet"
P.REUSE = {}
P.FITS = ["FULL", "ADDED_ONLY"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED]
P.PAIRS = [("FULL", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0")]


def load():
    df = pd.read_parquet(DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base = [c for c in df if c.startswith("feature_") and c != COLUMN]
    assert len(base) == 22
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[COLUMN].notna().astype("float32")      # the routable block, under T's column name
    obs["obs_macro"] = obs["obs_tf"]
    obs["no_feed"] = np.float32(np.nan)
    df[base + [COLUMN]] = df[base + [COLUMN]].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, [COLUMN], [], obs


def arms(base, fut, _):
    return {"B0": base, "B0CHK": base, "FULL": base + fut, "ADDED_ONLY": fut}


def audit():
    """Rebuild the column from the spot file and the panel, and measure what it knows."""
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    spot = pd.read_parquet(SPOT, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    from src.data_mani.history_features import bin_ties_preserved
    again = bin_ties_preserved(future_return(ds, spot).to_frame(COLUMN), ds["date"])[COLUMN].to_numpy("float64")
    have = ds[COLUMN].to_numpy("float64")
    same = (again == have) | (np.isnan(again) & np.isnan(have))
    k = N.keys()
    q = ds.merge(k[["symbol", "date", "kind"]], on=["symbol", "date"])
    rho = q.groupby("date")[[COLUMN, H.T20]].corr(method="spearman").xs(COLUMN, level=1)[H.T20]
    kind = q.groupby("date")["kind"].first()
    base = [c for c in q if c.startswith("feature_") and c != COLUMN]
    out = dict(rebuild_cells=int(same.size), rebuild_differ=int((~same).sum()),
               observed=float(q[COLUMN].notna().mean()),
               observed_by_kind=q.groupby("kind")[COLUMN].apply(lambda s: float(s.notna().mean())).to_dict(),
               spearman_with_target=dict(mean=float(rho.mean()), median=float(rho.median()), min=float(rho.min()),
                                         dev=float(rho[kind == "dev"].mean()), test=float(rho[kind == "test"].mean())),
               mass={str(a): float(b) for a, b in q[COLUMN].value_counts(normalize=True).sort_index().items()},
               max_abs_spearman_with_base=float(q[[COLUMN] + base].corr(method="spearman").loc[COLUMN, base].abs().max()))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1)
    print(out)


def score(workers):
    T.check()
    named = ["B0", "P_HIST"] + P.SCORED
    jobs = [(m, x, s) for m in T.MODELS for x in named for s in T.SEEDS[m]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        S = pd.concat(list(ex.map(N.score_job, jobs, chunksize=2)), ignore_index=True)
        Z = pd.concat(list(ex.map(N.blend_job, [(m, s) for m in T.MODELS for s in T.SEEDS[m]])), ignore_index=True)
        X = pd.concat(list(ex.map(T.route_job, [(m, s, x) for m in T.MODELS for s in T.SEEDS[m] for x in P.ROUTED])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    Z.to_parquet(os.path.join(ROOT, "blend_by_date.parquet"), index=False)
    N.summarise(S)
    N.blends(Z)
    T.routing(S, X)
    T.windows(S)
    print("wrote", ROOT)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    a.tune = False
    P.patch()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        a.arms = a.arms or (P.FITS + (["B0CHK"] if a.model == "ridge" else []))
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
    score(int(os.environ.get("E142_WORKERS", "16")))


P.load, P.arms, P.coverage = load, arms, (lambda arm, o: o["obs_tf"])
if __name__ == "__main__":
    main()
