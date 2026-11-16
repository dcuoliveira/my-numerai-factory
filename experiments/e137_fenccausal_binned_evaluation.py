#!/usr/bin/env python3
"""
E137 -- the feature-set evaluation procedure for crypto/v2.0-fenccausal-binned.

    python3 experiments/e137_fenccausal_binned_evaluation.py --fit --model lgb_ktz --n-jobs 6
    python3 experiments/e137_fenccausal_binned_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e137_fenccausal_binned_evaluation.py --audit
    python3 experiments/e137_fenccausal_binned_evaluation.py

Base features and targets equal E133's panel on every key, so B0, P_HIST and tuned B0 are reused
from E133 and E135. Ridge B0 is refitted as B0CHK to prove that.

  FULL        22 shipped + 22 binned _symmean
  ADDED_ONLY  the 22 means alone
  CONT        22 shipped + the same means unbinned. v2.0-fenccausal stops at 2026-07-08, so they
              are recomputed here with the builder's own function. FULL vs CONT is binning's cost.
  F_*, NO_*   one group of means added, or left out of FULL
  H20, H60    one horizon of means
  FULL_H      FULL + tenure, read against P_HIST
  P_GLOBAL    binned whole-sample means, look-ahead by construction. If FULL's gain is real but
              small and this one is large, the difference is what can never be had live.
  tuneFULL    the equal-budget retune, against E133/E135's tuned B0
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402
import e133_histfe_factorial as H  # noqa: E402
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.build_feature_encoding import causal_feature_means, global_feature_means  # noqa: E402
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e137_fenccausal_binned"
PRED = os.path.join(ROOT, "preds")
DATA = "data/crypto/v2.0-fenccausal-binned/train.parquet"
MODELS = E.MODELS
SEEDS = Q.SEEDS
GROUPS = {"PRICE": ("close_",), "CAP": ("market_cap_",), "VOL": ("volume_",),
          "SIGNAL": ("bollinger_", "momentum_", "relative_strength_", "sharpe_", "volatility_")}
FITS = ["FULL", "ADDED_ONLY", "CONT"] + ["F_" + g for g in GROUPS] + ["NO_" + g for g in GROUPS] + ["H20", "H60", "FULL_H", "P_GLOBAL"]
SHARED = ("B0", "P_HIST", "tuneB0")
RULES = ["R_SLOT", "R_HALF"]
COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + FITS if a != "FULL_H"] + \
          [("FULL", "CONT", "binned against continuous"), ("FULL_H", "P_HIST", "block beyond tenure"), ("P_GLOBAL", "FULL", "look-ahead gap"),
           ("tuneFULL", "tuneB0", "equal-budget retune")] + \
          [("FULL", "NO_" + g, "%s means, given the rest" % g.lower()) for g in GROUPS]
PAIRS = [("FULL", "B0"), ("CONT", "B0"), ("FULL", "CONT"), ("P_GLOBAL", "B0"), ("P_HIST", "B0"), ("F_SIGNAL", "B0"), ("F_CAP", "B0"), ("F_VOL", "B0"), ("F_PRICE", "B0"), ("FULL_H", "P_HIST")]
N.ROOT, N.COMPARE, N.OLD, N.FITS = ROOT, COMPARE, ["B0", "P_HIST"], [a for a in FITS if a != "ADDED_ONLY"]


def group(col, g):
    return col[len("feature_"):].startswith(GROUPS[g])


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["date", "symbol"]).reset_index(drop=True)
    sm = [c for c in raw if c.endswith("_symmean")]
    base = [c for c in raw if c.startswith("feature_") and c not in sm]
    assert len(base) == 22 and len(sm) == 22
    cont = causal_feature_means(raw, base, "symbol", "date", 50.0)[0].add_prefix("cont_")
    glob = bin_ties_preserved(global_feature_means(raw, base, "symbol", 50.0)[0], raw["date"]).add_prefix("glob_")
    df = pd.concat([raw, cont, glob], axis=1)
    df = df[df["date"] <= Q.LAST].reset_index(drop=True)
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    obs = df[["symbol", "date"]].assign(obs_FULL=df[sm].notna().mean(axis=1))
    fill = base + sm + list(cont) + list(glob)
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet("data/crypto/v2.0-fenccausal-binned/historical_meta_models.parquet")[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, sm, obs


def arms(base, sm):
    A = {"B0CHK": base, "FULL": base + sm, "ADDED_ONLY": sm, "CONT": base + ["cont_" + c for c in sm], "FULL_H": base + sm + ["feature_hist"],
         "P_GLOBAL": base + ["glob_" + c for c in sm], "H20": base + [c for c in sm if "_20d_" in c], "H60": base + [c for c in sm if "_60d_" in c]}
    for g in GROUPS:
        A["F_" + g] = base + [c for c in sm if group(c, g)]
        A["NO_" + g] = base + [c for c in sm if not group(c, g)]
    assert len(A["H20"]) == 33 and len(A["F_SIGNAL"]) == 32 and len(A["F_CAP"]) == 26 and len(A["NO_SIGNAL"]) == 34
    return A


def fit(args):
    df, base, sm, _ = load()
    A = arms(base, sm)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    last = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(last.loc[last["kind"] == "test", "date"]).unique()
    B = H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)]
    keep = ["symbol", "date", H.T20, "meta_model"]
    if args.tune:
        names, seeds = ["FULL"], [0]
    else:
        names, seeds = FITS + (["B0CHK"] if args.model == "ridge" else []), ([0] if args.model == "ridge" else SEEDS[args.model])
    tag = "tune" if args.tune else ""
    for arm in names:
        cols = A[arm]
        for s in seeds:
            path = os.path.join(PRED, args.model, "%s%s_s%d.parquet" % (tag, arm, s))
            if os.path.exists(path):
                print("skip", path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, picks, t1 = [], [], time.time()
            for i, tr, te in B:
                trm, tem = df["date"].isin(tr), df["date"].isin(te)
                params = H.pick(args.model, df, cols, tr, s, args.n_jobs) if args.tune else H.FIXED[args.model]
                picks.append({k: str(params.get(k)) for k in H.GRID[args.model]})
                pred = H.fit_predict(args.model, params, s, args.n_jobs, df.loc[trm, cols].to_numpy(np.float32),
                                     df.loc[trm, H.T20].to_numpy(), df.loc[tem, cols].to_numpy(np.float32))
                o = df.loc[tem, keep].copy()
                o["prediction"], o["block"] = pred, i
                parts.append(o)
            tmp = path + ".partial"
            pd.concat(parts).to_parquet(tmp, index=False)
            os.replace(tmp, path)
            if args.tune:
                json.dump(picks, open(path + ".picks.json", "w"))
            print("%s %-10s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


def preds(model, arm, seed):
    name = "%s_s%d" % (arm, seed)
    if arm in SHARED:
        return Q.preds(model, name)
    p = pd.read_parquet(os.path.join(PRED, model, name + ".parquet"), columns=["symbol", "date", "prediction"])
    p["date"] = pd.to_datetime(p["date"])
    return p


N.preds = preds


def route_job(job):
    model, seed = job
    k = N.keys().merge(preds(model, "B0", seed).rename(columns={"prediction": "b0"}), on=["symbol", "date"], validate="1:1") \
        .merge(preds(model, "FULL", seed).rename(columns={"prediction": "cand"}), on=["symbol", "date"], validate="1:1") \
        .merge(pd.read_parquet(os.path.join(ROOT, "observed.parquet")).rename(columns={"obs_FULL": "obs"}), on=["symbol", "date"], validate="1:1")
    k["b0"] = k.groupby("date")["b0"].rank(pct=True)
    rows = []
    for d, g in k.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for rule in RULES:
            x, cov = Q.route(g, rule)
            c, m = N.priced(g.assign(prediction=x))
            rows.append(dict(model=model, seed=seed, rule=rule, date=d, kind=g["kind"].iloc[0], corr=c, payout=0.1 * c + m if m == m else np.nan, routed=1 - cov.mean()))
    return pd.DataFrame(rows)


def routing(S, X):
    a = X.groupby(["model", "rule", "date", "kind"], as_index=False)[["corr", "payout", "routed"]].mean()
    b = S[S["arm"] == "B0"].groupby(["model", "date"], as_index=False)[["corr", "payout"]].mean()
    z = a.merge(b, on=["model", "date"], suffixes=("", "_b"))
    rows = []
    for (m, r, sc), g in z.groupby(["model", "rule", "kind"]):
        c, p = E.describe(g["corr"] - g["corr_b"]), E.describe((g["payout"] - g["payout_b"]).dropna())
        rows.append(dict(model=m, rule=r, scope=sc, corr_delta=c["mean"], corr_t=c["nw_t"], payout_delta=p["mean"], payout_t=p["nw_t"], routed=g["routed"].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "routing.csv"), index=False)


def windows(S):
    """CORR on payout's own dates, as E136 found the two windows can disagree."""
    A = S.groupby(["model", "arm", "date", "kind"], as_index=False)[["corr", "payout"]].mean()
    rows = []
    for m in MODELS:
        p = A[A["model"] == m]
        for a, b in PAIRS:
            z = p[p["arm"] == a].merge(p[p["arm"] == b], on=["date", "kind"], suffixes=("", "_b"))
            z = z[z["payout"].notna() & z["payout_b"].notna()]
            for sc in ("dev", "test"):
                q = z[z["kind"] == sc]
                c, y = E.describe(q["corr"] - q["corr_b"]), E.describe(q["payout"] - q["payout_b"])
                rows.append(dict(model=m, arm=a, vs=b, scope=sc, corr_delta=c["mean"], corr_t=c["nw_t"], payout_delta=y["mean"], payout_t=y["nw_t"], payout_lo=y["boot_lo"], payout_hi=y["boot_hi"], n=len(q)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "payout_window.csv"), index=False)


def check():
    """B0 reused from E133/E135 must be what these rows give."""
    a = preds("ridge", "B0CHK", 0).merge(preds("ridge", "B0", 0), on=["symbol", "date"], validate="1:1", suffixes=("", "_old"))
    assert len(a) == 468000
    print("B0CHK vs reused B0, ridge: max |diff| %.2e" % (a["prediction"] - a["prediction_old"]).abs().max())
    assert (a["prediction"] - a["prediction_old"]).abs().max() < 1e-4


def audit():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["date", "symbol"]).reset_index(drop=True)
    sm = [c for c in raw if c.endswith("_symmean")]
    base = [c for c in raw if c.startswith("feature_") and c not in sm]
    out = {}
    full = bin_ties_preserved(causal_feature_means(raw, base, "symbol", "date", 50.0)[0], raw["date"])
    same = (full[sm].to_numpy("float32") == raw[sm].to_numpy("float32")) | (full[sm].isna().to_numpy() & raw[sm].isna().to_numpy())
    out["rebuild"] = dict(cells=int(same.size), differ=int((~same).sum()))
    rows = []
    for T in ("2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        T = pd.Timestamp(T)
        cut = raw[raw["date"] <= T].reset_index(drop=True)
        enc = bin_ties_preserved(causal_feature_means(cut, base, "symbol", "date", 50.0)[0], cut["date"])
        a = enc[cut["date"] == T][sm].to_numpy("float32")
        b = raw.loc[raw["date"] == T, sm].to_numpy("float32")
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(T.date()), cells=int(eq.size), differ=int((~eq).sum())))
        print(rows[-1], flush=True)
    out["truncation"] = rows
    old = pd.read_parquet("data/crypto/v2.0-fenccausal/train.parquet").reset_index()
    old["date"] = pd.to_datetime(old["date"])
    cont = causal_feature_means(raw, base, "symbol", "date", 50.0)[0]
    z = raw[["symbol", "date"]].join(cont).merge(old[["symbol", "date"] + sm], on=["symbol", "date"], suffixes=("", "_o"))
    out["continuous"] = dict(rows=len(z), max_diff=float(max((z[c] - z[c + "_o"]).abs().max() for c in sm)), mean_diff=float(np.mean([(z[c] - z[c + "_o"]).abs().mean() for c in sm])))
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
    k["date"] = pd.to_datetime(k["date"])
    ids = raw[["symbol", "date", "ucid"]].merge(k, on=["symbol", "date"])
    n = raw.groupby("symbol")["ucid"].nunique()
    reused = set(n[n > 1].index)
    # keys whose ticker had already served another coin by that date: their means mix two coins
    r = raw[["symbol", "date", "ucid"]].sort_values("date")
    r["seen"] = (r.groupby("symbol")["ucid"].transform(lambda s: pd.factorize(s)[0]).values > 0) | (r.groupby("symbol")["ucid"].transform(lambda s: s.ne(s.iloc[0]).cummax()).values)
    hit = r.merge(k, on=["symbol", "date"])
    out["tickers"] = dict(reused_symbols=len(reused), keys_on_reused=float(ids["symbol"].isin(reused).mean()), keys_mixing_coins=float(hit["seen"].mean()), keys=len(ids))
    q = raw.merge(k, on=["symbol", "date"])
    out["mass"] = {str(a): float(b) for a, b in q[sm].stack().value_counts(normalize=True).sort_index().items()}
    out["nan_cells"] = float(q[sm].isna().mean().mean())
    same_f = pd.Series({c: q[c].corr(q[c[:-len("_symmean")]], method="spearman") for c in sm})
    out["corr_with_own_base"] = dict(mean=float(same_f.mean()), max=float(same_f.max()), min=float(same_f.min()))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1)
    print(json.dumps({x: out[x] for x in ("rebuild", "continuous", "tickers", "corr_with_own_base")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        fit(a)
        return
    if a.audit:
        audit()
        return
    check()
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path):
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        k.merge(load()[3], on=["symbol", "date"], how="left", validate="1:1").to_parquet(obs_path, index=False)
    named = ["B0", "P_HIST"] + FITS
    jobs = [(m, x, s) for m in MODELS for x in named for s in SEEDS[m]] + [(m, x, 0) for m in MODELS for x in ("tuneB0", "tuneFULL")]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E137_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(N.score_job, jobs, chunksize=2)), ignore_index=True)
        Z = pd.concat(list(ex.map(N.blend_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
        X = pd.concat(list(ex.map(route_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    Z.to_parquet(os.path.join(ROOT, "blend_by_date.parquet"), index=False)
    N.summarise(S)
    N.blends(Z)
    routing(S, X)
    windows(S)
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
