#!/usr/bin/env python3
"""
E132 -- do the three *-feed-fillind datasets add out-of-sample value over crypto/v2.0, when a
coin with no features is held at a neutral prediction instead of dropped from the score?

    python3 experiments/e132_feed_fillind_factorial.py --model lgb_ktz --arms B0 L_FULL --seeds 0 1 2 3 4

Same protocol as E131 (feature-set-evaluation skill): the same 513,496 (symbol, date) keys, the
same 12 blocks (train 130 / purge 22 / predict 130), the same three fixed configs, 5 seeds for
lgb_ktz and mlp_ktz. What is new is the universe:

  scored   every key, always. A candidate fits and predicts on the rows its dataset has; the
           rows it dropped (coins with no price on any panel row, 6.4% of keys) are given the
           neutral prediction by e132_analysis.py: per-date rank, then 0.5.
  B0       raw v2.0 on every key, the baseline a candidate must beat.
  B0F      raw v2.0 fitted and predicted on the feed rows, neutral elsewhere. B0F - B0 is what
           the row handling alone does; X - B0F is what the features alone do.
  P_DROP   raw v2.0 + the no-feed flag on every key. The flag is LOOK-AHEAD by construction (a
           coin is dropped because no feed exists today), so this measures the survivorship
           channel by itself.
  P_HIST   raw v2.0 + causal panel tenure on the feed rows.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_litmom_fillind_factorial as E  # noqa: E402

from src.data_mani.binning import bin_cross_sectional  # noqa: E402

os.chdir(REPO)
T20 = E.T20
KEYS = "data/crypto/v2.0-litmom-fillind/train.parquet"    # E131's 513,496 keys
DATA = {"L": "data/crypto/v2.0-litmom-feed-fillind/train.parquet",
        "S": "data/crypto/v2.0-momswap-feed-fillind/train.parquet",
        "C": "data/crypto/v2.0-momswap-macro-carver-feed-fillind/train.parquet"}
OUT = "results/reports/crypto/e132_feed_fillind/preds"


def load():
    u = pd.read_parquet(KEYS).reset_index()[["symbol", "date", T20]]
    assert not u.duplicated(["symbol", "date"]).any()
    v = pd.read_parquet(E.V20).reset_index()
    base22 = [c for c in v if c.startswith("feature_")]
    assert len(base22) == 22
    r = v[["symbol", "date", T20] + base22].rename(columns={T20: "v20_t"})
    r = r.rename(columns={b: "v20__" + b for b in base22})
    u = u.merge(r, on=["symbol", "date"], how="left", validate="1:1")
    assert (u["v20_t"] == u[T20]).all(), "target drifted from E131's keys"
    u[["v20__" + b for b in base22]] = u[["v20__" + b for b in base22]].fillna(0.5).astype("float32")
    cols = {}
    for k, path in DATA.items():
        d = pd.read_parquet(path).reset_index()
        fs = [c for c in d if c.startswith("feature_")]
        d = d[["symbol", "date", T20] + fs].rename(columns={T20: "t_" + k, **{c: k + "__" + c for c in fs}})
        d = d[d["date"] <= u["date"].max()]
        d["in_" + k] = True
        u = u.merge(d, on=["symbol", "date"], how="left", validate="1:1")
        assert u["in_" + k].sum() == len(d), "%s has keys outside the universe" % k
        assert (u.loc[u["in_" + k].notna(), "t_" + k] == u.loc[u["in_" + k].notna(), T20]).all()
        u["in_" + k] = u["in_" + k].notna() & (u["in_" + k] == True)  # noqa: E712
        u = u.drop(columns="t_" + k)
        cols[k] = fs
    u["no_feed"] = (~u["in_L"]).astype("float32")
    u = u.sort_values(["date", "symbol"]).reset_index(drop=True)
    u["feature_hist"] = u.groupby("symbol").cumcount().astype("float64")
    u = bin_cross_sectional(u, ["feature_hist"], period_col="date")
    meta = pd.read_parquet(E.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    u = u.merge(meta, on=["symbol", "date"], how="left")
    return u, base22, cols


def arms(base22, cols):
    v20 = ["v20__" + b for b in base22]
    p = lambda k, xs: [k + "__" + x for x in xs]  # noqa: E731
    pick = lambda k, s: [c for c in cols[k] if c.startswith(s)]  # noqa: E731
    fam = lambda names: ["feature_mom_" + x for x in names]  # noqa: E731
    fam_ind = lambda names: ["feature_isna_mom_" + x for x in names]  # noqa: E731

    mom = pick("L", "feature_mom_")
    mom_ind = pick("L", "feature_isna_mom_")
    base_ind = [c for c in pick("L", "feature_isna_") if c not in mom_ind]
    a = {"B0": ("all", v20), "B0F": ("L", v20),
         "P_DROP": ("all", v20 + ["no_feed"]), "P_HIST": ("L", v20 + ["feature_hist"])}
    b = p("L", base22)
    a.update({"L_B1": ("L", b + p("L", base_ind)), "L_M1": ("L", b + p("L", mom)),
              "L_M2": ("L", b + p("L", mom + mom_ind)), "L_FULL": ("L", p("L", cols["L"])),
              "L_ADDED": ("L", p("L", mom + mom_ind))})
    for f, xs in E.FAMILIES.items():
        a["L_no_" + f] = ("L", b + p("L", [m for m in mom if m not in fam(xs)]))
    s0 = [c for c in cols["S"] if not c.startswith(("feature_mom_", "feature_isna_"))]
    assert len(s0) == 14
    sind = pick("S", "feature_isna_")
    a.update({"S_S0": ("S", p("S", s0)), "S_M1": ("S", p("S", s0 + mom)),
              "S_FULL": ("S", p("S", cols["S"])), "S_ADDED": ("S", p("S", mom + sind))})
    macro, tf = pick("C", "feature_macro_"), pick("C", "feature_tf_")
    ctf = pick("C", "feature_isna_tf_")
    cmi = pick("C", "feature_isna_mom_")
    assert len(macro) == 18 and len(tf) == 15 and len(ctf) == 15 and len(cmi) == 17
    s0c = s0
    a.update({"C_MACRO": ("C", p("C", s0c + macro)), "C_TF": ("C", p("C", s0c + tf)),
              "C_RAW": ("C", p("C", s0c + mom + macro + tf)), "C_FULL": ("C", p("C", cols["C"])),
              "C_ADDED": ("C", p("C", mom + macro + tf + cmi + ctf)),
              "C_no_mom": ("C", p("C", s0c + macro + tf + ctf)),
              "C_no_macro": ("C", p("C", s0c + mom + tf + cmi + ctf)),
              "C_no_tf": ("C", p("C", s0c + mom + macro + cmi))})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E.FIXED))
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--layout-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df, base22, cols = load()
    A = arms(base22, cols)
    dates = np.array(sorted(df["date"].unique()))
    B = E.blocks(dates)
    os.makedirs(OUT, exist_ok=True)
    if args.layout_only:
        print(len(dates), {k: (r, len(c)) for k, (r, c) in A.items()})
        print({k: int(df["in_" + k].sum()) for k in DATA}, (df["in_L"] == df["in_S"]).all(), (df["in_L"] == df["in_C"]).all())
        keep = ["symbol", "date", T20, "meta_model", "no_feed", "feature_hist", "in_L", "in_S", "in_C"]
        k = df[keep].copy()
        k["block"], k["kind"] = -1, ""
        for i, _, _, te, kind in B:
            m = k["date"].isin(te)
            k.loc[m, "block"], k.loc[m, "kind"] = i, kind
        k[k["block"] >= 0].to_parquet(os.path.join(OUT, "keys.parquet"), index=False)
        json.dump({a: {"rows": r, "cols": c} for a, (r, c) in A.items()}, open(os.path.join(OUT, "arms.json"), "w"), indent=1)
        return
    seeds = [0] if args.model == "ridge" else args.seeds
    for arm in (args.arms or list(A)):
        rows, cs = A[arm]
        use = df if rows == "all" else df[df["in_" + rows]]
        for s in seeds:
            path = os.path.join(OUT, args.model, "%s_s%d.parquet" % (arm, s))
            if os.path.exists(path):
                print("skip", path); continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, t1 = [], time.time()
            for i, tr, _, te, kind in B:
                trm, tem = use["date"].isin(tr), use["date"].isin(te)
                pred, _ = E.fit_predict(args.model, E.FIXED[args.model], s, args.n_jobs,
                                        use.loc[trm, cs].to_numpy(np.float32), use.loc[trm, T20].to_numpy(),
                                        use.loc[tem, cs].to_numpy(np.float32))
                o = use.loc[tem, ["symbol", "date"]].copy()
                o["prediction"] = pred
                parts.append(o)
            pd.concat(parts).to_parquet(path, index=False)
            print("%s %-10s seed %d  %d cols  %.0fs" % (args.model, arm, s, len(cs), time.time() - t1), flush=True)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
