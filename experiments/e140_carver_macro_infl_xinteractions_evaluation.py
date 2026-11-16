#!/usr/bin/env python3
"""
E140 -- the feature-set evaluation procedure for crypto/v2.0-carver-macro-infl-xinteractions.

    python3 experiments/e140_carver_macro_infl_xinteractions_evaluation.py --fit --model lgb_ktz --n-jobs 3 [--arms FULL]
    python3 experiments/e140_carver_macro_infl_xinteractions_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e140_carver_macro_infl_xinteractions_evaluation.py --audit
    python3 experiments/e140_carver_macro_infl_xinteractions_evaluation.py   # score, after E138 and E139's fits exist

crypto/v2.0-carver-macro-infl + 270 binned interactions of the 15 raw trend signals with the 18
signed macro momentum signals (10 metals / S&P / DXY returns, 8 break-even changes), multiplied raw
and binned after (src/data_mani/build_trend_xinteractions.py). E139's runner and E138's machinery,
with this study's data, arms and reuse map.

  CMI         E139's FULL (67), reused: the parent, so FULL vs CMI is the interactions' margin
  FULL        CMI + 270 interactions (337)
  ADDED_ONLY  the 315 non-shipped columns
  F_IX        22 shipped + 270 interactions
  IX_RET      CMI + the 150 interactions with metals / S&P / DXY returns
  IX_BEI      CMI + the 120 interactions with break-even changes
  FULL_H      FULL + tenure, against P_HIST
  P_WARM, P_NM  reused from E138
  tuneFULL    vs tuned B0, mlp grid capped at alpha 1
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

P.STUDY = "E140"
P.ROOT = "results/reports/crypto/e140_carver_macro_infl_xinteractions"
P.DATA = "data/crypto/v2.0-carver-macro-infl-xinteractions/train.parquet"
P.META = "data/crypto/v2.0-carver-macro-infl-xinteractions/historical_meta_models.parquet"
P.PARENT = "CMI"
P.REUSE = {"CMI": ("results/reports/crypto/e139_carver_macro_infl/preds", "FULL"), "P_WARM": (T.PRED, "P_WARM"), "P_NM": (T.PRED, "P_NM")}
P.FITS = ["FULL", "ADDED_ONLY", "F_IX", "IX_RET", "IX_BEI", "FULL_H"]
P.SCORED = P.FITS + list(P.REUSE)
P.ROUTED = ["CMI", "FULL", "F_IX", "IX_RET", "IX_BEI", "FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("FULL", "CMI", "interactions, given carver-macro-infl"), ("IX_RET", "CMI", "return interactions, given the parent"),
             ("IX_BEI", "CMI", "break-even interactions, given the parent"), ("FULL_H", "P_HIST", "block beyond tenure"),
             ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("CMI", "B0"), ("FULL", "CMI"), ("F_IX", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0"), ("FULL_H", "P_HIST")]


def load():
    df = pd.read_parquet(P.DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    ix = [c for c in df if c.startswith("feature_ix_")]
    base, tf, mc, inf = P.split(df.drop(columns=ix))
    assert len(ix) == 270
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[tf].notna().mean(axis=1).astype("float32")
    obs["obs_macro"] = df[mc + inf].notna().mean(axis=1).astype("float32")
    obs["no_feed"] = np.float32(np.nan)
    fill = base + tf + mc + inf + ix + ["feature_hist"]
    df[fill] = df[fill].fillna(0.5).astype("float32")
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, tf, mc + inf + ix, obs


def arms(base, tf, rest):
    ix = [c for c in rest if c.startswith("feature_ix_")]
    mci = [c for c in rest if c not in ix]
    bei = [c for c in ix if "_X_bei_" in c]
    ret = [c for c in ix if c not in bei]
    parent = base + tf + mci
    A = {"B0": base, "B0CHK": base, "FULL": parent + ix, "ADDED_ONLY": tf + mci + ix, "F_IX": base + ix,
         "IX_RET": parent + ret, "IX_BEI": parent + bei, "FULL_H": parent + ix + ["feature_hist"]}
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_IX", "IX_RET", "IX_BEI")] == [337, 315, 292, 217, 187]
    return A


def coverage(arm, o):
    return o["obs_tf"]


def audit():
    """Rebuild the interactions from panel, spot and macro cut at T and compare row T."""
    from src.data_mani.build_feed_fillind import SPOT
    from src.data_mani.build_trend_xinteractions import MACRO, add_interactions
    ds = pd.read_parquet(P.DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    src = pd.read_parquet("data/crypto/v2.0-carver-macro-infl/train.parquet").reset_index()
    src["date"] = pd.to_datetime(src["date"])
    src["ucid"] = src["ucid"].astype(str)
    spot = pd.read_parquet(SPOT, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    mac = pd.read_parquet(MACRO)
    mac["date"] = pd.to_datetime(mac["date"])
    ix = [c for c in ds if c.startswith("feature_ix_")]
    rows = []
    for t in ("2021-06-30", "2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        t = pd.Timestamp(t)
        cut, _ = add_interactions(src[src["date"] <= t].reset_index(drop=True), spot[spot["date"] <= t], mac[mac["date"] <= t])
        a = cut.loc[cut["date"] == t].sort_values("symbol")[ix].to_numpy("float64")
        b = ds.loc[ds["date"] == t].sort_values("symbol")[ix].to_numpy("float64")
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(t.date()), rows=len(a), cells=int(eq.size), differ=int((~eq).sum())))
        print(rows[-1], flush=True)
    k = N.keys()
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    tf = [c for c in q if c.startswith("feature_tf_")]
    # the sign-flip identity: on a date with macro > 0 the interaction's ranking equals its trend signal's
    sp = []
    for d in sorted(q["date"].unique())[::60]:
        g = q[q["date"] == d]
        for c in ix[::18]:
            t_ = "feature_tf_" + c[len("feature_ix_"):].split("_X_")[0]
            r = g[c].corr(g[t_], method="spearman")
            if r == r:
                sp.append(abs(r))
    out = dict(truncation=rows, observed=float(q[ix].notna().mean().mean()), tf_observed=float(q[tf].notna().mean().mean()),
               abs_spearman_ix_vs_own_trend=dict(mean=float(np.mean(sp)), min=float(np.min(sp)), n=len(sp)))
    json.dump(out, open(os.path.join(P.ROOT, "audit.json"), "w"), indent=1)
    print(out)


P.load, P.arms, P.coverage, P.audit = load, arms, coverage, audit
if __name__ == "__main__":
    P.main()
