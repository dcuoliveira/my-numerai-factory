#!/usr/bin/env python3
"""
E138 -- the feature-set evaluation procedure for crypto/v2.0-carver-macro.

    python3 experiments/e138_carver_macro_evaluation.py --fit --model lgb_ktz --n-jobs 6 [--arms FULL F_TF]
    python3 experiments/e138_carver_macro_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e138_carver_macro_evaluation.py --audit
    python3 experiments/e138_carver_macro_evaluation.py

The dataset (src/data_mani/build_carver_macro.py) is all 22 shipped features + 15 Carver /
Clenow trend signals recomputed from the current spot file + 18 shared macro features, every v2.0
row kept, missing cells left NaN (0.5 at load), no indicator columns. Its 22 base features and
target equal E133's panel on all 468,000 keys, so B0, P_HIST and tuned B0 (ridge, lgb) are reused
from E133/E135, and a ridge B0 refit (B0CHK) proves it.

  FULL         22 + 15 tf + 18 macro
  ADDED_ONLY   the 33 new columns alone
  F_TF, F_MACRO  one block added (F_MACRO is also FULL minus the trend block, and F_TF FULL minus macro)
  NO_EWMAC NO_BREAKOUT NO_BBZ NO_CLENOW   one trend family left out of FULL (CLENOW includes newhigh20in5)
  NO_RATES NO_RISK                        yields+slopes / metals+S&P+DXY left out of FULL
  FULL_H       FULL + tenure, read against P_HIST
  P_WARM       B0 + binned share of the trend block observed: causal, depth of history
  P_NM         B0 + "this coin has no trend value on ANY panel row": LOOK-AHEAD by construction
               (a coin has no feed because none exists today). Measures the survivorship channel.
  tuneFULL     equal-budget retune vs tuned B0. The mlp grid is capped at alpha 1 (E137: alpha 100
               collapses the net to a constant), so mlp's tuned B0 is refitted here on that grid.

Scoring: plain on all 300 coins per date, and ROUTED (E134 rule, the deployment score): v2.0 ranks
every coin and the candidate reorders only the coins whose proposed block is observed that date,
inside the slots they already hold. R_SLOT = any proposed feature observed, R_HALF = at least half.
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
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e138_carver_macro"
PRED = os.path.join(ROOT, "preds")
VERSION = "crypto/v2.0-carver-macro"
DATA = "data/crypto/v2.0-carver-macro/train.parquet"
META = "data/crypto/v2.0-carver-macro/historical_meta_models.parquet"
MODELS = E.MODELS
SEEDS = Q.SEEDS
TF_FAM = {"EWMAC": ("ewmac_",), "BREAKOUT": ("breakout_",), "BBZ": ("bbz_",), "CLENOW": ("clenow_", "newhigh")}
MACRO_FAM = {"RATES": ("y1", "y2", "y5", "y10", "y30", "slope_"), "RISK": ("gold_", "silver_", "snp_", "dxy_")}
FITS = ["FULL", "ADDED_ONLY", "F_TF", "F_MACRO"] + ["NO_" + f for f in TF_FAM] + ["NO_" + f for f in MACRO_FAM] + ["FULL_H", "P_WARM", "P_NM"]
ROUTED = ["FULL", "F_TF", "F_MACRO"] + ["NO_" + f for f in TF_FAM] + ["NO_" + f for f in MACRO_FAM] + ["FULL_H"]
RULES = ["R_SLOT", "R_HALF"]
# mlp's grid capped: E137 showed alpha 100 collapses the linear net to a constant
H.GRID["mlp_ktz"] = {"mlp__alpha": [0.01, 0.1, 1.0], "mlp__learning_rate_init": [0.01, 0.05011872336272722]}
COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + FITS if a != "FULL_H"] + \
          [("FULL", "F_MACRO", "trend block, given macro"), ("FULL", "F_TF", "macro block, given trend"),
           ("FULL_H", "P_HIST", "block beyond tenure"), ("F_TF", "P_WARM", "trend beyond warm-up"),
           ("tuneFULL", "tuneB0", "equal-budget retune")] + \
          [("FULL", "NO_" + f, "%s, given the rest" % f.lower()) for f in list(TF_FAM) + list(MACRO_FAM)]
PAIRS = [("FULL", "B0"), ("F_TF", "B0"), ("F_MACRO", "B0"), ("ADDED_ONLY", "B0"), ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST"), ("F_TF", "P_WARM")]
N.ROOT, N.COMPARE, N.OLD, N.FITS = ROOT, COMPARE, ["B0", "P_HIST"], [a for a in FITS if a != "ADDED_ONLY"]


def fam(cols, prefix, keys):
    return [c for c in cols if c[len(prefix):].startswith(keys)]


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw[raw["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    tf = [c for c in df if c.startswith("feature_tf_")]
    mc = [c for c in df if c.startswith("feature_macro_")]
    base = [c for c in df if c.startswith("feature_") and c not in tf + mc]
    assert len(base) == 22 and len(tf) == 15 and len(mc) == 18
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    share = df[tf].notna().mean(axis=1).astype("float64").to_frame("feature_warm")
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    # look-ahead flag: a coin (symbol, ucid) with no trend value on any row of the WHOLE file
    whole = raw[tf].notna().any(axis=1).groupby([raw["symbol"], raw["ucid"]]).any().rename("ever").reset_index()
    df = df.merge(whole, on=["symbol", "ucid"], how="left", validate="m:1")
    df["feature_nofeed"] = (~df.pop("ever")).astype("float32")
    obs = df[["symbol", "date"]].copy()
    obs["obs_tf"] = df[tf].notna().mean(axis=1).astype("float32")
    obs["obs_macro"] = df[mc].notna().mean(axis=1).astype("float32")
    obs["no_feed"] = df["feature_nofeed"]
    fill = base + tf + mc + ["feature_hist", "feature_warm"]
    df[fill] = df[fill].fillna(0.5)
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, tf, mc, obs


def arms(base, tf, mc):
    A = {"B0CHK": base, "B0": base, "FULL": base + tf + mc, "ADDED_ONLY": tf + mc, "F_TF": base + tf, "F_MACRO": base + mc,
         "FULL_H": base + tf + mc + ["feature_hist"], "P_WARM": base + ["feature_warm"], "P_NM": base + ["feature_nofeed"]}
    for f, k in TF_FAM.items():
        A["NO_" + f] = [c for c in A["FULL"] if c not in set(fam(tf, "feature_tf_", k))]
    for f, k in MACRO_FAM.items():
        A["NO_" + f] = [c for c in A["FULL"] if c not in set(fam(mc, "feature_macro_", k))]
    assert [len(A["NO_" + f]) for f in TF_FAM] == [50, 51, 52, 52], [len(A["NO_" + f]) for f in TF_FAM]
    assert [len(A["NO_" + f]) for f in MACRO_FAM] == [47, 45]
    return A


def coverage(arm, o):
    """Share of the arm's proposed (routable) features observed on each key."""
    return o["obs_tf"] if arm != "F_MACRO" else o["obs_macro"]


def fit(args):
    df, base, tf, mc, _ = load()
    A = arms(base, tf, mc)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    last = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(last.loc[last["kind"] == "test", "date"]).unique()
    B = H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)]
    keep = ["symbol", "date", H.T20, "meta_model"]
    if args.tune:
        names = ["FULL"] + (["B0"] if args.model == "mlp_ktz" else [])
        seeds = [0]
    else:
        names = args.arms or (FITS + (["B0CHK"] if args.model == "ridge" else []))
        seeds = [0] if args.model == "ridge" else SEEDS[args.model]
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
            print("%s %-12s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(cols), time.time() - t1), flush=True)


def preds(model, arm, seed):
    name = "%s_s%d" % (arm, seed)
    local = os.path.join(PRED, model, name + ".parquet")
    if arm in ("B0", "P_HIST") or (arm == "tuneB0" and model != "mlp_ktz"):
        return Q.preds(model, name)
    p = pd.read_parquet(local, columns=["symbol", "date", "prediction"])
    p["date"] = pd.to_datetime(p["date"])
    return p


N.preds = preds


def route_job(job):
    model, seed, arm = job
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    o["obs"] = coverage(arm, o)
    k = N.keys().merge(preds(model, "B0", seed).rename(columns={"prediction": "b0"}), on=["symbol", "date"], validate="1:1") \
        .merge(preds(model, arm, seed).rename(columns={"prediction": "cand"}), on=["symbol", "date"], validate="1:1") \
        .merge(o[["symbol", "date", "obs", "no_feed"]], on=["symbol", "date"], validate="1:1")
    k["b0"] = k.groupby("date")["b0"].rank(pct=True)
    rows = []
    for d, g in k.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for rule in RULES:
            x, cov = Q.route(g, rule)
            c, m = N.priced(g.assign(prediction=x))
            kept = g.assign(prediction=x)[cov]
            rows.append(dict(model=model, seed=seed, arm=arm, rule=rule, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0], corr=c, mmc=m,
                             payout=0.1 * c + m if m == m else np.nan, routed=1 - cov.mean(), nofeed=g["no_feed"].mean(),
                             cov_arm=E.corr_one(kept.set_index("symbol")) if len(kept) > 1 else np.nan,       # no covered coin: E144 blocks 0-1
                             cov_b0=E.corr_one(kept.set_index("symbol"), "b0") if len(kept) > 1 else np.nan))
    return pd.DataFrame(rows)


def routing(S, X):
    """Routed arm against B0 (seed-averaged per date), dev and test, CORR and payout."""
    X.to_parquet(os.path.join(ROOT, "routed_by_date_seed.parquet"), index=False)
    a = X.groupby(["model", "arm", "rule", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout", "routed", "cov_arm", "cov_b0"]].mean()
    b = S[S["arm"] == "B0"].groupby(["model", "date"], as_index=False)[["corr", "mmc", "payout"]].mean()
    z = a.merge(b, on=["model", "date"], suffixes=("", "_b"))
    rows, lev = [], []
    for (m, arm, r, sc), g in z.groupby(["model", "arm", "rule", "kind"]):
        c, p = E.describe(g["corr"] - g["corr_b"]), E.describe((g["payout"] - g["payout_b"]).dropna())
        cv = E.describe(g["cov_arm"] - g["cov_b0"])
        rows.append(dict(model=m, arm=arm, rule=r, scope=sc, corr_delta=c["mean"], corr_t=c["nw_t"], corr_lo=c["boot_lo"], corr_hi=c["boot_hi"],
                         payout_delta=p["mean"], payout_t=p["nw_t"], payout_lo=p["boot_lo"], payout_hi=p["boot_hi"], routed=g["routed"].mean(),
                         covered_corr_delta=cv["mean"], covered_corr_t=cv["nw_t"], n=len(g), n_pay=p["n"]))
        q = g.dropna(subset=["payout"])
        dd = E.max_dd(q["payout"].values)
        lev.append(dict(model=m, arm=arm, rule=r, scope=sc, corr=g["corr"].mean(), mmc=q["mmc"].mean(), c_sharpe=g["corr"].mean() / g["corr"].std(ddof=0),
                        dd_corr=E.max_dd(g["corr"].values), payout=q["payout"].mean(), p_sharpe=q["payout"].mean() / q["payout"].std(ddof=0), dd_pay=dd,
                        calmar=q["payout"].mean() / dd, burn=int((q["payout"] < 0).sum()), n=len(g), n_pay=len(q)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "routing.csv"), index=False)
    pd.DataFrame(lev).to_csv(os.path.join(ROOT, "routing_levels.csv"), index=False)
    p = X.merge(S[S["arm"] == "B0"][["model", "seed", "date", "corr", "payout"]], on=["model", "seed", "date"], suffixes=("", "_b"))
    rows = []
    for (m, arm, r, s, sc), g in p.groupby(["model", "arm", "rule", "seed", "kind"]):
        rows.append(dict(model=m, arm=arm, rule=r, seed=s, scope=sc, corr_delta=(g["corr"] - g["corr_b"]).mean(), corr_t=E.newey_west((g["corr"] - g["corr_b"]).values)[2],
                         payout_delta=(g["payout"] - g["payout_b"]).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "routing_by_seed.csv"), index=False)


def windows(S):
    """CORR on payout's own dates: dev CORR starts in 2020, payout in May 2024 (E136)."""
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
    """Rebuild both blocks from the panel, spot and macro store cut at T; compare row T with the dataset."""
    from src.data_mani.build_carver_macro import MACRO, build
    from src.data_mani.build_feed_fillind import SPOT
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    v = pd.read_parquet("data/crypto/v2.0/train.parquet").reset_index()
    v["date"] = pd.to_datetime(v["date"])
    v["ucid"] = v["ucid"].astype(str)
    spot = pd.read_parquet(SPOT, columns=["symbol", "date", "close"])
    spot["date"] = pd.to_datetime(spot["date"])
    mac = pd.read_parquet(MACRO)
    mac["date"] = pd.to_datetime(mac["date"])
    new = [c for c in ds if c.startswith(("feature_tf_", "feature_macro_"))]
    out = {}
    rows = []
    for T in ("2021-06-30", "2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        T = pd.Timestamp(T)
        cut, _ = build(v[v["date"] <= T].reset_index(drop=True), spot[spot["date"] <= T], mac[mac["date"] <= T])
        a = cut.loc[cut["date"] == T].sort_values("symbol")[new].to_numpy("float64")
        b = ds.loc[ds["date"] == T].sort_values("symbol")[new].to_numpy("float64")
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(T.date()), rows=len(a), cells=int(eq.size), differ=int((~eq).sum())))
        print(rows[-1], flush=True)
    out["truncation"] = rows
    k = N.keys()
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    out["coverage"] = dict(keys=len(z), tf_any=float((z["obs_tf"] > 0).mean()), tf_half=float((z["obs_tf"] >= .5).mean()), macro_any=float((z["obs_macro"] > 0).mean()),
                           no_feed=float(z["no_feed"].mean()), warm_given_feed=float(((z["obs_tf"] == 0) & (z["no_feed"] == 0)).mean()))
    y = z.groupby([z["date"].dt.year, "kind"]).agg(tf_any=("obs_tf", lambda s: (s > 0).mean()), tf_half=("obs_tf", lambda s: (s >= .5).mean()), no_feed=("no_feed", "mean")).reset_index()
    out["coverage_by_year"] = y.to_dict("records")
    q = ds.merge(k[["symbol", "date"]], on=["symbol", "date"])
    tf = [c for c in new if c.startswith("feature_tf_")]
    mc = [c for c in new if c.startswith("feature_macro_")]
    out["mass_tf"] = {str(a): float(b) for a, b in q[tf].stack().value_counts(normalize=True).sort_index().items()}
    out["macro_distinct_per_date"] = int(q.groupby("date")[mc].nunique().max().max())
    base = [c for c in q if c.startswith("feature_") and c not in new]
    sp = q[tf + base].corr(method="spearman").loc[tf, base].abs()
    out["tf_max_abs_spearman_with_base"] = {c: dict(base=sp.loc[c].idxmax(), rho=float(sp.loc[c].max())) for c in tf}
    old = pd.read_parquet("data/crypto/v2.0-momswap-macro-carver/train.parquet").reset_index()
    old["date"] = pd.to_datetime(old["date"])
    w = q[["symbol", "date"] + new].merge(old[["symbol", "date"] + new], on=["symbol", "date"], suffixes=("", "_o"))
    x, xo = w[new].to_numpy("float64"), w[[c + "_o" for c in new]].to_numpy("float64")
    out["vs_momswap_macro_carver"] = dict(rows=len(w), differ=int((~((x == xo) | (np.isnan(x) & np.isnan(xo)))).sum()), gained=int((~np.isnan(x) & np.isnan(xo)).sum()), lost=int((np.isnan(x) & ~np.isnan(xo)).sum()))
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    print(json.dumps({x: out[x] for x in ("truncation", "coverage", "vs_momswap_macro_carver")}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path) and not a.fit:
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        k.merge(load()[4], on=["symbol", "date"], how="left", validate="1:1").to_parquet(obs_path, index=False)
    if a.fit:
        fit(a)
        return
    if a.audit:
        audit()
        return
    check()
    named = ["B0", "P_HIST"] + FITS
    jobs = [(m, x, s) for m in MODELS for x in named for s in SEEDS[m]] + [(m, x, 0) for m in MODELS for x in ("tuneB0", "tuneFULL")]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E138_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(N.score_job, jobs, chunksize=2)), ignore_index=True)
        Z = pd.concat(list(ex.map(N.blend_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
        X = pd.concat(list(ex.map(route_job, [(m, s, x) for m in MODELS for s in SEEDS[m] for x in ROUTED])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    Z.to_parquet(os.path.join(ROOT, "blend_by_date.parquet"), index=False)
    N.summarise(S)
    N.blends(Z)
    routing(S, X)
    windows(S)
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
