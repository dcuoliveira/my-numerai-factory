#!/usr/bin/env python3
"""
E139 -- the feature-set evaluation procedure for crypto/v2.0-carver-macro-infl.

    python3 experiments/e139_carver_macro_infl_evaluation.py --fit --model lgb_ktz --n-jobs 4 [--arms FULL]
    python3 experiments/e139_carver_macro_infl_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e139_carver_macro_infl_evaluation.py --audit
    python3 experiments/e139_carver_macro_infl_evaluation.py          # score, after E138's fits exist

crypto/v2.0-carver-macro + 12 shared break-even inflation features (4 levels, 8 changes), built by
build_inflation onto E138's dataset, so every other cell is E138's. Same protocol, routing and bar
as E138, whose machinery this imports (fit / route / summarise read the module globals patched below).

  CM          E138's FULL (22 + 15 tf + 18 macro), reused: the parent, so FULL vs CM is inflation's margin
  FULL        CM + 12 inflation (67)
  ADDED_ONLY  the 45 non-shipped columns
  F_INFL      22 shipped + 12 inflation
  NO_BEILVL / NO_BEICHG   FULL without the 4 levels / the 8 changes
  FULL_H      FULL + tenure, against P_HIST
  P_WARM, P_NM  reused from E138 (same columns, same rows)
  tuneFULL    vs tuned B0, mlp grid capped at alpha 1

A within-date-constant column cannot reorder coins in a linear model, so F_INFL is expected to be
inert on ridge and the linear mlp; it can matter only through tree splits (regime gating).
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
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

STUDY = "E139"
ROOT = "results/reports/crypto/e139_carver_macro_infl"
DATA = "data/crypto/v2.0-carver-macro-infl/train.parquet"
META = "data/crypto/v2.0-carver-macro-infl/historical_meta_models.parquet"
PARENT = "CM"
REUSE = {"CM": (T.PRED, "FULL"), "P_WARM": (T.PRED, "P_WARM"), "P_NM": (T.PRED, "P_NM")}
FITS = ["FULL", "ADDED_ONLY", "F_INFL", "NO_BEILVL", "NO_BEICHG", "FULL_H"]
SCORED = FITS + list(REUSE)
ROUTED = ["CM", "FULL", "F_INFL", "NO_BEILVL", "NO_BEICHG", "FULL_H"]
COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + SCORED if a != "FULL_H"] + \
          [("FULL", "CM", "inflation, given carver-macro"), ("FULL", "NO_BEILVL", "break-even levels, given the rest"),
           ("FULL", "NO_BEICHG", "break-even changes, given the rest"), ("FULL_H", "P_HIST", "block beyond tenure"),
           ("tuneFULL", "tuneB0", "equal-budget retune")]
PAIRS = [("FULL", "B0"), ("CM", "B0"), ("FULL", "CM"), ("F_INFL", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0"), ("FULL_H", "P_HIST")]


def split(df):
    tf = [c for c in df if c.startswith("feature_tf_")]
    inf = [c for c in df if c.startswith("feature_macro_bei_")]
    mc = [c for c in df if c.startswith("feature_macro_") and c not in inf]
    base = [c for c in df if c.startswith("feature_") and c not in tf + mc + inf]
    assert (len(base), len(tf), len(mc), len(inf)) == (22, 15, 18, 12)
    return base, tf, mc, inf


def load():
    df = pd.read_parquet(DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base, tf, mc, inf = split(df)
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[tf].notna().mean(axis=1).astype("float32")
    obs["obs_macro"] = df[mc + inf].notna().mean(axis=1).astype("float32")
    obs["no_feed"] = np.float32(np.nan)
    fill = base + tf + mc + inf + ["feature_hist"]
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    # T.fit unpacks (df, base, tf, mc, obs) and calls arms(base, tf, mc): mc carries inflation too
    return df.merge(meta, on=["symbol", "date"], how="left"), base, tf, mc + inf, obs


def arms(base, tf, mci):
    inf = [c for c in mci if c.startswith("feature_macro_bei_")]
    lvl = [c for c in inf if "_chg" not in c]
    full = base + tf + mci
    A = {"B0": base, "B0CHK": base, "FULL": full, "ADDED_ONLY": tf + mci, "F_INFL": base + inf, "FULL_H": full + ["feature_hist"],
         "NO_BEILVL": [c for c in full if c not in lvl], "NO_BEICHG": [c for c in full if c in lvl or c not in inf]}
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_INFL", "NO_BEILVL", "NO_BEICHG")] == [67, 45, 34, 63, 59]
    return A


def coverage(arm, o):
    return o["obs_macro"] if arm == "F_INFL" else o["obs_tf"]


def preds(model, arm, seed):
    name = "%s_s%d" % (arm, seed)
    if arm in ("B0", "P_HIST") or (arm == "tuneB0" and model != "mlp_ktz"):
        return Q.preds(model, name)
    root, src = REUSE.get(arm, (T.PRED, None))
    path = os.path.join(root, model, "%s_s%d.parquet" % (src, seed)) if src else os.path.join(ROOT, "preds", model, name + ".parquet")
    p = pd.read_parquet(path, columns=["symbol", "date", "prediction"])
    p["date"] = pd.to_datetime(p["date"])
    return p


def patch():
    """Point E138's machinery at this study."""
    T.ROOT, T.PRED, T.DATA, T.META = ROOT, os.path.join(ROOT, "preds"), DATA, META
    T.load, T.arms, T.coverage, T.preds, T.FITS, T.ROUTED, T.PAIRS = load, arms, coverage, preds, FITS, ROUTED, PAIRS
    N.preds = preds
    N.ROOT, N.COMPARE, N.OLD, N.FITS = ROOT, COMPARE, ["B0", "P_HIST"], [a for a in SCORED if a != "ADDED_ONLY"]


def audit():
    """Rebuild the inflation block from the macro store cut at T and compare row T."""
    from src.data_mani.macro_features import INFLATION_SPECS, attach_macro_asof, compute_macro_signals
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    mac = pd.read_parquet("data/crypto/macro/macro_daily.parquet")
    mac["date"] = pd.to_datetime(mac["date"])
    cols = list(INFLATION_SPECS)
    rows = []
    for t in ("2021-06-30", "2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        t = pd.Timestamp(t)
        day = ds.loc[ds["date"] == t, ["symbol", "date"] + cols].sort_values("symbol")
        f = compute_macro_signals(mac[mac["date"] <= t], INFLATION_SPECS, normalize=True, min_periods=252)
        a = attach_macro_asof(day[["symbol", "date"]], f, cols, lag_days=1, max_staleness_days=7)[cols].to_numpy("float64")
        b = day[cols].to_numpy("float64")
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(t.date()), rows=len(a), cells=int(eq.size), differ=int((~eq).sum())))
        print(rows[-1], flush=True)
    parent = pd.read_parquet(T.DATA0).reset_index()
    parent["date"] = pd.to_datetime(parent["date"])
    other = [c for c in parent if c.startswith("feature_")]
    z = ds[["symbol", "date"] + other].merge(parent[["symbol", "date"] + other], on=["symbol", "date"], suffixes=("", "_p"))
    x, y = z[other].to_numpy("float64"), z[[c + "_p" for c in other]].to_numpy("float64")
    k = N.keys()
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    out = dict(truncation=rows, parent_cells=int(x.size), parent_differ=int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum()),
               distinct_per_date=int(q.groupby("date")[cols].nunique().max().max()), observed=float(q[cols].notna().mean().mean()))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1)
    print(out)


def score(workers):
    T.check()
    named = ["B0", "P_HIST"] + SCORED
    jobs = [(m, x, s) for m in T.MODELS for x in named for s in T.SEEDS[m]] + [(m, x, 0) for m in T.MODELS for x in ("tuneB0", "tuneFULL")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        S = pd.concat(list(ex.map(N.score_job, jobs, chunksize=2)), ignore_index=True)
        Z = pd.concat(list(ex.map(N.blend_job, [(m, s) for m in T.MODELS for s in T.SEEDS[m]])), ignore_index=True)
        X = pd.concat(list(ex.map(T.route_job, [(m, s, x) for m in T.MODELS for s in T.SEEDS[m] for x in ROUTED])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    Z.to_parquet(os.path.join(ROOT, "blend_by_date.parquet"), index=False)
    N.summarise(S)
    N.blends(Z)
    T.routing(S, X)
    T.windows(S)
    print("wrote", ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args(argv)
    T.DATA0 = T.DATA
    patch()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        if not a.arms and not a.tune:
            a.arms = FITS + (["B0CHK"] if a.model == "ridge" else [])
        T.fit(a)
        return
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path):
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        o = load()[4].drop(columns="no_feed")
        nf = pd.read_parquet(os.path.join(T.ROOT0, "observed.parquet"), columns=["symbol", "date", "no_feed"])
        k.merge(o, on=["symbol", "date"], how="left", validate="1:1").merge(nf, on=["symbol", "date"], validate="1:1").to_parquet(obs_path, index=False)
    if a.audit:
        audit()
        return
    score(int(os.environ.get("E139_WORKERS", "16")))


T.ROOT0 = T.ROOT
if __name__ == "__main__":
    main()
