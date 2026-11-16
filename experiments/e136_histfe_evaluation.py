#!/usr/bin/env python3
"""
E136 -- finish the feature-set evaluation procedure for crypto/v2.0-histfe.

    python3 experiments/e136_histfe_evaluation.py --audit
    python3 experiments/e136_histfe_evaluation.py --fit --model lgb_ktz --arm NO_RANK --n-jobs 4
    python3 experiments/e136_histfe_evaluation.py

E133 ran the factorial on dev and E135 added the final block, so those predictions are reused.
This runs what neither did:

  NO_RANK NO_MA NO_TSMAX NO_TSC  leave one family out of FULL
  FULL_H F_RANK_H TSC_H          block + tenure, read against P_HIST
  P_WARM                         B0 + the binned share of history features observed. Causal, and
                                 a placebo for how much of a gain is only depth of history.
                                 histfe has no look-ahead flag, so there is no P_NM arm.
  ADDED_ONLY                     its final block
  blends                         rank B0 + w*ADDED_ONLY, w chosen on dev, scored on test
  audit                          the block rebuilt on the truncated real panel

Scoring is plain, on all 300 coins. The routed reading is E135's.
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

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.history_features import add_history_features, bin_ties_preserved, history_columns  # noqa: E402

ROOT = "results/reports/crypto/e136_histfe_evaluation"
NEW = os.path.join(ROOT, "preds")
MODELS = E.MODELS
SEEDS = Q.SEEDS
OLD = ["B0", "SHORT", "LONG", "COMBO", "TSC", "FULL", "F_RANK", "F_MA", "F_TSMAX", "P_HIST"]
FITS = ["NO_RANK", "NO_MA", "NO_TSMAX", "NO_TSC", "FULL_H", "F_RANK_H", "TSC_H", "P_WARM"]
COMPARE = [(a, "B0", "against v2.0") for a in OLD[1:] + ["ADDED_ONLY", "P_WARM"]] + \
          [("FULL", "NO_RANK", "own-history rank, given the rest"), ("FULL", "NO_MA", "MA differences, given the rest"),
           ("FULL", "NO_TSMAX", "days since max, given the rest"), ("FULL", "NO_TSC", "days since change, given the rest"),
           ("FULL_H", "P_HIST", "whole block beyond tenure"), ("F_RANK_H", "P_HIST", "rank beyond tenure"),
           ("TSC_H", "P_HIST", "days since change beyond tenure"), ("P_HIST", "P_WARM", "tenure beyond warm-up"),
           ("F_RANK", "P_WARM", "rank beyond warm-up"), ("tuneFULL", "tuneB0", "equal-budget retune")]


def arms(base, new):
    A = H.arms(base, new)
    fam = history_columns(base)
    without = lambda k: base + [c for c in new if c not in set(fam[k])]  # noqa: E731
    return dict(A, NO_RANK=without("rank"), NO_MA=without("ma"), NO_TSMAX=without("tsmax"), NO_TSC=without("tschg"),
                FULL_H=A["FULL"] + ["feature_hist"], F_RANK_H=A["F_RANK"] + ["feature_hist"], TSC_H=A["TSC"] + ["feature_hist"],
                P_WARM=base + ["feature_warm"])


def fit(args):
    df, base, new, obs = Q.load()
    share = obs[new].mean(axis=1).astype("float64").to_frame("feature_warm")
    del obs
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"].fillna(0.5).values
    cols = arms(base, new)[args.arm]
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    last = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(last.loc[last["kind"] == "test", "date"]).unique()
    final = (11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)
    B = [final] if args.arm == "ADDED_ONLY" else H.blocks(past) + [final]
    keep = ["symbol", "date", H.T20, "meta_model"]
    seeds = [0] if args.model == "ridge" else SEEDS[args.model]
    for s in seeds:
        path = os.path.join(NEW, args.model, "%s_s%d.parquet" % (args.arm, s))
        if os.path.exists(path):
            print("skip", path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        parts, t1 = [], time.time()
        for i, tr, te in B:
            trm, tem = df["date"].isin(tr), df["date"].isin(te)
            pred = H.fit_predict(args.model, H.FIXED[args.model], s, args.n_jobs, df.loc[trm, cols].to_numpy(np.float32),
                                 df.loc[trm, H.T20].to_numpy(), df.loc[tem, cols].to_numpy(np.float32))
            o = df.loc[tem, keep].copy()
            o["prediction"], o["block"] = pred, i
            parts.append(o)
        tmp = path + ".partial"
        pd.concat(parts).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        print("%s %-10s seed %d  %d cols  %.0fs" % (args.model, args.arm, s, len(cols), time.time() - t1), flush=True)


def preds(model, arm, seed):
    name = "%s_s%d" % (arm, seed)
    if arm == "ADDED_ONLY":
        paths = [os.path.join(Q.DEV, model, name + ".parquet"), os.path.join(NEW, model, name + ".parquet")]
    elif arm in FITS:
        paths = [os.path.join(NEW, model, name + ".parquet")]
    else:
        paths = [os.path.join(Q.DEV, model, name + ".parquet"), os.path.join(Q.TEST, model, name + ".parquet")]
    p = pd.concat([pd.read_parquet(x, columns=["symbol", "date", "prediction"]) for x in paths], ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    return p


def keys():
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date", H.T20, "meta_model", "block", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    return k


def priced(h):
    h = h.set_index("symbol")
    c = E.corr_one(h)
    m = E.contrib_one(h, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
    return c, m


def score_job(job):
    model, arm, seed = job
    b0 = "tuneB0" if arm.startswith("tune") else "B0"
    f = keys().merge(preds(model, arm, seed), on=["symbol", "date"], how="left", validate="1:1") \
        .merge(preds(model, b0, seed).rename(columns={"prediction": "b0"}), on=["symbol", "date"], how="left", validate="1:1")
    assert f[["prediction", "b0"]].notna().all().all(), (model, arm, seed)
    rows = []
    for d, g in f.groupby("date", sort=True):
        c, m = priced(g)
        rows.append(dict(model=model, arm=arm, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0], corr=c, mmc=m,
                         payout=0.1 * c + m if m == m else np.nan, predcorr_b0=g["prediction"].rank().corr(g["b0"].rank())))
    return pd.DataFrame(rows)


def blend_job(job):
    model, seed = job
    f = keys().merge(preds(model, "B0", seed), on=["symbol", "date"], validate="1:1") \
        .merge(preds(model, "ADDED_ONLY", seed).rename(columns={"prediction": "added"}), on=["symbol", "date"], validate="1:1")
    f["prediction"] = f.groupby("date")["prediction"].rank(pct=True)
    f["added"] = f.groupby("date")["added"].rank(pct=True)
    rows = []
    for d, g in f.groupby("date", sort=True):
        for w in E.W:
            c, m = priced(g.assign(prediction=(1 - w) * g["prediction"] + w * g["added"]))
            rows.append(dict(model=model, seed=seed, date=d, kind=g["kind"].iloc[0], w=w, corr=c, payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def blends(Z):
    """The weight comes from dev alone, once by CORR and once by payout."""
    A = Z.groupby(["model", "date", "kind", "w"], as_index=False)[["corr", "payout"]].mean()
    rows = []
    for m in MODELS:
        z = A[A["model"] == m]
        for metric in ("corr", "payout"):
            lv = z[z["kind"] == "dev"].groupby("w")[metric].mean()
            w = float(lv.idxmax())
            p = z.pivot_table(index=["date", "kind"], columns="w", values=metric).dropna()
            for sc in ("dev", "test"):
                q = p.xs(sc, level="kind")
                d = q[w] - q[0.0]
                r = E.describe(d) if w > 0 else dict(mean=0.0, nw_t=np.nan, boot_lo=np.nan, boot_hi=np.nan, n=len(d))
                rows.append(dict(model=m, metric=metric, w=w, scope=sc, level=q[w].mean(), b0=q[0.0].mean(), delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "blends.csv"), index=False)


def summarise(S):
    A = S.groupby(["model", "arm", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout", "predcorr_b0"]].mean()
    lev, dl, by = [], [], []
    for (m, a, sc), g in A.groupby(["model", "arm", "kind"]):
        p = g.dropna(subset=["payout"])
        dd = E.max_dd(p["payout"].values)
        lev.append(dict(model=m, arm=a, scope=sc, corr=g["corr"].mean(), mmc=p["mmc"].mean(), c_sharpe=g["corr"].mean() / g["corr"].std(ddof=0), dd_corr=E.max_dd(g["corr"].values),
                        payout=p["payout"].mean(), p_sharpe=p["payout"].mean() / p["payout"].std(ddof=0), dd_pay=dd, calmar=p["payout"].mean() / dd, burn=int((p["payout"] < 0).sum()),
                        n=len(g), n_pay=len(p), predcorr_b0=g["predcorr_b0"].mean()))
    for m in MODELS:
        w = A[A["model"] == m]
        for a, b, label in COMPARE:
            x, y = w[w["arm"] == a], w[w["arm"] == b]
            z = x.merge(y, on=["date", "kind", "block"], suffixes=("", "_b"))
            for metric in ("corr", "mmc", "payout"):
                for sc in ("dev", "test"):
                    d = (z.loc[z["kind"] == sc, metric] - z.loc[z["kind"] == sc, metric + "_b"]).dropna()
                    r = E.describe(d)
                    dl.append(dict(model=m, arm=a, vs=b, label=label, metric=metric, scope=sc, delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
            z["d"] = z["corr"] - z["corr_b"]
            for k, q in z.groupby("block"):
                by.append(dict(model=m, arm=a, vs=b, block=k, mean=q["d"].mean()))
    pd.DataFrame(lev).to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)
    pd.DataFrame(dl).to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(ROOT, "paired_deltas_by_block.csv"), index=False)
    rows = []
    p = S.pivot_table(index=["model", "seed", "date", "kind"], columns="arm", values="corr")
    for m in MODELS[1:]:
        for a in [x for x in OLD[1:] + FITS + ["ADDED_ONLY"]]:
            q = p.xs(m, level="model")
            for (s, sc), g in q.groupby(level=["seed", "kind"]):
                d = (g[a] - g["B0"]).dropna()
                rows.append(dict(model=m, arm=a, seed=s, scope=sc, mean=d.mean(), t=E.newey_west(d.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"), index=False)


def extra():
    """CORR on payout's own dates, because dev CORR starts in 2020 and dev payout in May 2024."""
    S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"))
    A = S.groupby(["model", "arm", "date", "kind"], as_index=False)[["corr", "payout"]].mean()
    rows = []
    pairs = [("F_RANK", "B0"), ("TSC", "B0"), ("SHORT", "B0"), ("FULL", "B0"), ("P_HIST", "B0"), ("P_WARM", "B0"), ("F_RANK_H", "B0"), ("TSC_H", "B0"),
             ("F_RANK_H", "P_HIST"), ("TSC_H", "P_HIST"), ("F_RANK", "P_WARM")]
    for m in MODELS:
        p = A[A["model"] == m]
        for a, b in pairs:
            z = p[p["arm"] == a].merge(p[p["arm"] == b], on=["date", "kind"], suffixes=("", "_b"))
            z = z[z["payout"].notna() & z["payout_b"].notna()]
            for sc in ("dev", "test"):
                q = z[z["kind"] == sc]
                c, y = E.describe(q["corr"] - q["corr_b"]), E.describe(q["payout"] - q["payout_b"])
                rows.append(dict(model=m, arm=a, vs=b, scope=sc, corr_delta=c["mean"], corr_t=c["nw_t"], payout_delta=y["mean"], payout_t=y["nw_t"], payout_lo=y["boot_lo"], payout_hi=y["boot_hi"], n=len(q)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "payout_window.csv"), index=False)
    rows = []
    w = S[S["model"] == "lgb_ktz"].pivot_table(index=["seed", "date", "kind"], columns="arm", values="payout").dropna(subset=["B0"])
    for (s, sc), g in w.groupby(level=["seed", "kind"]):
        for a, b in (("F_RANK_H", "P_HIST"), ("F_RANK_H", "B0"), ("F_RANK", "B0"), ("F_RANK", "P_WARM")):
            d = (g[a] - g[b]).dropna()
            rows.append(dict(seed=s, scope=sc, arm=a, vs=b, payout_delta=d.mean(), t=E.newey_west(d.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "lgb_payout_by_seed.csv"), index=False)


def audit():
    """Rebuild the block on the real panel cut at T and compare T with the dataset, then coverage overlap."""
    df = pd.read_parquet(H.DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    base = [c for c in df if c.startswith("feature_") and not c.startswith("feature_hfe_")]
    new = [c for c in df if c.startswith("feature_hfe_")]
    rows = []
    for T in ("2023-06-30", "2025-12-31", "2026-02-19", "2026-08-20"):
        T = pd.Timestamp(T)
        cut = df.loc[df["date"] <= T, ["symbol", "date", "ucid"] + base].reset_index(drop=True)
        out = add_history_features(cut, base)[new]
        a = out[cut["date"] == T].to_numpy(dtype="float64")
        b = df.loc[df["date"] == T, new].to_numpy(dtype="float64")
        assert a.shape == b.shape
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append(dict(date=str(T.date()), rows=len(a), cells=int(same.size), differ=int((~same).sum())))
        print(rows[-1], flush=True)
    k = pd.read_parquet(Q.KEYS, columns=["symbol", "date", "no_feed", "kind"])
    k["date"] = pd.to_datetime(k["date"])
    o = pd.read_parquet(os.path.join(Q.ROOT, "observed.parquet"), columns=["symbol", "date", "obs_FULL"])
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    z["warm"] = z["obs_FULL"] < .5
    ov = dict(keys=len(z), no_feed=float(z["no_feed"].mean()), warm=float(z["warm"].mean()), warm_given_no_feed=float(z.loc[z["no_feed"].astype(bool), "warm"].mean()),
              warm_given_feed=float(z.loc[~z["no_feed"].astype(bool), "warm"].mean()), obs_no_feed=float(z.loc[z["no_feed"].astype(bool), "obs_FULL"].mean()),
              obs_feed=float(z.loc[~z["no_feed"].astype(bool), "obs_FULL"].mean()), corr=float(z["warm"].astype(float).corr(z["no_feed"].astype(float))))
    json.dump(dict(truncation=rows, overlap=ov), open(os.path.join(ROOT, "audit.json"), "w"), indent=1)
    print(ov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arm", choices=FITS + ["ADDED_ONLY"])
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        fit(a)
        return
    if a.audit:
        audit()
        return
    if a.extra:
        extra()
        return
    jobs = [(m, x, s) for m in MODELS for x in OLD + FITS + ["ADDED_ONLY"] for s in SEEDS[m]] + [(m, x, 0) for m in MODELS for x in ("tuneB0", "tuneFULL")]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E136_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(score_job, jobs, chunksize=2)), ignore_index=True)
        Z = pd.concat(list(ex.map(blend_job, [(m, s) for m in MODELS for s in SEEDS[m]])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    Z.to_parquet(os.path.join(ROOT, "blend_by_date.parquet"), index=False)
    summarise(S)
    blends(Z)
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
