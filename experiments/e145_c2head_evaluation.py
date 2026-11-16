#!/usr/bin/env python3
"""
E145 -- Chronos-2-Synth FORECAST-head features: the feature-set evaluation, rules frozen in
experiments/e145_registration.md before any fit.

    python3 experiments/e145_c2head_evaluation.py --fit --model lgb_ktz --n-jobs 6 [--arms ZS_COV_HEAD ...]
    python3 experiments/e145_c2head_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    python3 experiments/e145_c2head_evaluation.py --audit
    python3 experiments/e145_c2head_evaluation.py --score           # writes the CSVs the report reads
    E145_VARIANT=bin python3 experiments/e145_c2head_evaluation.py ...   # the five-bin robustness copy

Features: crypto/v2.0-c2head (continuous within-date percentile rank) and -bin (Numerai's five
bins), built by src/data_mani/build_c2head.py from src/data_mani/chronos2_head.py. Six branches of
five summaries: zscov, zsar, zsdelta (zero-shot), ftcov, ftar, ftdelta (fine-tuned per block 2-11;
blocks 0-1 have none). The ft values DIFFER BY BLOCK, so each block is fitted on its own values from
blocks.parquet (as E144); routing and the direct legs read train.parquet, where each row carries the
block that predicts it.

Protocol and machinery are E133-E144's: walk-forward 130/22/130, 12 blocks, fixed configs
(ridge -- there is no ridge_ktz --, lgb_ktz, mlp_ktz), 5 seeds for lgb/mlp, B0 / P_HIST / tuned B0
reused from E133/E135 on the same keys, R_SLOT routing (a coin without the arm's forecasts keeps
v2.0's rank), Newey-West lag 20. Primary dev window: blocks 2-10.
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
import e134_route_feature_sets as R  # noqa: E402
import e135_route_histfe as Q  # noqa: E402
import e136_histfe_evaluation as N  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

VARIANT = os.environ.get("E145_VARIANT", "rank")
assert VARIANT in ("rank", "bin")
BASE_ROOT = "results/reports/crypto/e145_c2head"
ROOT = BASE_ROOT if VARIANT == "rank" else os.path.join(BASE_ROOT, "bin")
DS = "data/crypto/v2.0-c2head" + ("" if VARIANT == "rank" else "-bin")
DATA, BLOCKS, META = DS + "/train.parquet", DS + "/blocks.parquet", DS + "/historical_meta_models.parquet"
MODELS = E.MODELS
SEEDS = Q.SEEDS
BR = ("zscov", "zsar", "zsdelta", "ftcov", "ftar", "ftdelta")
FAM = {"ZS_COV": "zscov", "ZS_DELTA": "zsdelta", "FT_COV": "ftcov", "FT_DELTA": "ftdelta"}
CAND = ["ZS_COV_HEAD", "FT_COV_HEAD", "ZS_DELTA_HEAD", "FT_DELTA_HEAD", "ZS_FULL_HEAD", "FT_FULL_HEAD", "ALL_HEAD"]
TUNED = ["ZS_COV_HEAD", "FT_COV_HEAD", "ALL_HEAD"]
LEGS = {"L_ZS_COV": "feature_zscov_median_mean", "L_FT_COV": "feature_ftcov_median_mean",
        "L_ZS_DELTA": "feature_zsdelta_median_mean", "L_FT_DELTA": "feature_ftdelta_median_mean"}
W = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
if VARIANT == "rank":
    FITS = CAND + ["HEAD_ONLY", "ZS_AR_HEAD", "FT_AR_HEAD", "ALL_H"] + ["ALL_NO_" + f for f in FAM] + [f + "_ONLY" for f in FAM] + \
        ["P_WARM", "P_OBS", "P_NM"]
else:
    FITS = list(CAND)
NO_ROUTE = {"HEAD_ONLY", "P_WARM", "P_OBS", "P_NM"} | {f + "_ONLY" for f in FAM}


def cols(br):
    return ["feature_%s_%s" % (br, s) for s in ("median_mean", "median_terminal", "median_slope", "uncertainty_mean", "asymmetry_mean")]


def load():
    raw = pd.read_parquet(DATA).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw[raw["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    head = [c for br in BR for c in cols(br)]
    base = [c for c in df if c.startswith("feature_") and c not in head]
    assert len(base) == 22 and all(c in df for c in head)
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    df["feature_obs"] = bin_ties_preserved(df[["ctrl_ctx_share"]].rename(columns={"ctrl_ctx_share": "feature_obs"}), df["date"])["feature_obs"]
    ever = raw[cols("zscov")[0]].notna().groupby([raw["symbol"], raw["ucid"]]).any().rename("ever").reset_index()
    df = df.merge(ever, on=["symbol", "ucid"], how="left", validate="m:1")
    df["feature_nofeed"] = (~df.pop("ever")).astype("float32")          # LOOK-AHEAD placebo: whole file
    df[base] = df[base].fillna(0.5)                                     # the loader's fill; head columns are filled per block
    obs = df[["symbol", "date"]].copy()
    obs["obs_zs"] = df[cols("zscov")[0]].notna().astype("float32")
    obs["obs_ft"] = df[cols("ftcov")[0]].notna().astype("float32")
    obs["obs_all"] = obs["obs_zs"] * obs["obs_ft"]
    obs["no_feed"] = df["feature_nofeed"]
    obs["ctx_share"] = df["ctrl_ctx_share"].astype("float32")
    meta = pd.read_parquet(META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, head, obs


def arms(base):
    c = {f: cols(b) for f, b in FAM.items()}
    allh = c["ZS_COV"] + c["ZS_DELTA"] + c["FT_COV"] + c["FT_DELTA"]
    A = {"B0": base, "B0CHK": base,
         "ZS_COV_HEAD": base + c["ZS_COV"], "FT_COV_HEAD": base + c["FT_COV"],
         "ZS_DELTA_HEAD": base + c["ZS_DELTA"], "FT_DELTA_HEAD": base + c["FT_DELTA"],
         "ZS_FULL_HEAD": base + c["ZS_COV"] + c["ZS_DELTA"], "FT_FULL_HEAD": base + c["FT_COV"] + c["FT_DELTA"],
         "ALL_HEAD": base + allh, "HEAD_ONLY": allh, "ALL_H": base + allh + ["feature_hist"],
         "ZS_AR_HEAD": base + cols("zsar"), "FT_AR_HEAD": base + cols("ftar"),
         "P_WARM": base + ["feature_warm"], "P_OBS": base + ["feature_obs"], "P_NM": base + ["feature_nofeed"]}
    for f in FAM:
        A["ALL_NO_" + f] = base + [x for x in allh if x not in c[f]]
        A[f + "_ONLY"] = c[f]
    assert len(A["ALL_HEAD"]) == 42 and len(A["HEAD_ONLY"]) == 20 and len(A["ALL_NO_FT_COV"]) == 37
    return A


def coverage(arm):
    """The observed column a routed arm needs: its forecasts exist for the row."""
    if arm.startswith(("ZS_", "tuneZS_")):
        return "obs_zs"
    if arm.startswith(("FT_", "tuneFT_")):
        return "obs_ft"
    return "obs_all"


def windows(past, last):
    return H.blocks(past) + [(11, past[-(H.PURGE + H.TRAIN):-H.PURGE], last)]


def block_frames(df, B):
    """{block: its train+test rows with that block's own ft values, P_WARM recomputed, head NaN -> 0.5}."""
    ftc = [c for br in ("ftcov", "ftar", "ftdelta") for c in cols(br)]
    cand = [c for f in FAM.values() for c in cols(f)]
    L = pd.read_parquet(BLOCKS, columns=["block", "ucid", "date"] + ftc)
    L["date"] = pd.to_datetime(L["date"])
    out = {}
    for i, tr, te in B:
        rows = df["date"].isin(tr) | df["date"].isin(te)
        d = df.loc[rows].drop(columns=ftc).merge(L.loc[L["block"] == i].drop(columns="block"), on=["ucid", "date"], how="left", validate="1:1")
        assert len(d) == rows.sum()
        share = d[cand].notna().mean(axis=1).astype("float64").to_frame("feature_warm")
        d["feature_warm"] = bin_ties_preserved(share, d["date"])["feature_warm"].fillna(0.5)
        print("block %d: %d rows, zs on %.1f%%, own ft on %.1f%% of train / %.1f%% of test rows" % (
            i, len(d), 100 * d[cols("zscov")[0]].notna().mean(), 100 * d.loc[d["date"].isin(tr), ftc[0]].notna().mean(),
            100 * d.loc[d["date"].isin(te), ftc[0]].notna().mean()), flush=True)
        fill = [c for br in BR for c in cols(br)] + ["feature_hist", "feature_obs"]
        d[fill] = d[fill].fillna(0.5)
        out[i] = d.reset_index(drop=True)
    return out


def fit(args):
    df, base, head, _ = load()
    A = arms(base)
    past = np.array(sorted(df.loc[df["date"] < H.TEST_START, "date"].unique()))
    last = pd.read_parquet(Q.KEYS, columns=["date", "kind"])
    last = pd.to_datetime(last.loc[last["kind"] == "test", "date"]).unique()
    B = windows(past, last)
    F = block_frames(df, B)
    keep = ["symbol", "date", H.T20, "meta_model"]
    if args.tune:
        names, seeds = TUNED + (["B0"] if args.model == "mlp_ktz" else []), [0]
    else:
        names = args.arms or (FITS + (["B0CHK"] if args.model == "ridge" and VARIANT == "rank" else []))
        seeds = [0] if args.model == "ridge" else SEEDS[args.model]
    tag = "tune" if args.tune else ""
    for arm in names:
        c = A[arm]
        for s in seeds:
            path = os.path.join(ROOT, "preds", args.model, "%s%s_s%d.parquet" % (tag, arm, s))
            if os.path.exists(path):
                print("skip", path)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, picks, t1 = [], [], time.time()
            for i, tr, te in B:
                d = F[i]
                trm, tem = d["date"].isin(tr), d["date"].isin(te)
                params = H.pick(args.model, d, c, tr, s, args.n_jobs) if args.tune else H.FIXED[args.model]
                picks.append({k: str(params.get(k)) for k in H.GRID[args.model]})
                pred = H.fit_predict(args.model, params, s, args.n_jobs, d.loc[trm, c].to_numpy(np.float32),
                                     d.loc[trm, H.T20].to_numpy(), d.loc[tem, c].to_numpy(np.float32))
                o = d.loc[tem, keep].copy()
                o["prediction"], o["block"] = pred, i
                parts.append(o)
            p = pd.concat(parts)
            p["spread"] = p.groupby("date")["prediction"].transform("std")
            p.to_parquet(path + ".partial", index=False)
            os.replace(path + ".partial", path)
            if args.tune:
                json.dump(picks, open(path + ".picks.json", "w"))
            print("%s %-14s %s seed %d  %d cols  %.0fs" % (args.model, arm, tag, s, len(c), time.time() - t1), flush=True)


# ------------------------------------------------------------------ scoring
def preds(model, arm, seed):
    """B0, P_HIST and (except mlp) tuned B0 reused from E133/E135 on the same keys; the rest from this study."""
    name = "%s_s%d" % (arm, seed)
    if arm in ("B0", "P_HIST") or (arm == "tuneB0" and model != "mlp_ktz"):
        return Q.preds(model, name)
    root = os.path.join(BASE_ROOT, "preds") if arm in ("tuneB0",) else os.path.join(ROOT, "preds")
    p = pd.read_parquet(os.path.join(root, model, name + ".parquet"), columns=["symbol", "date", "prediction"])
    p["date"] = pd.to_datetime(p["date"])
    return p


def frame(model, arm, seed):
    b0 = "tuneB0" if arm.startswith("tune") else "B0"
    k = N.keys().merge(preds(model, arm, seed), on=["symbol", "date"], how="left", validate="1:1") \
        .merge(preds(model, b0, seed).rename(columns={"prediction": "b0"}), on=["symbol", "date"], how="left", validate="1:1") \
        .merge(pd.read_parquet(os.path.join(ROOT, "observed.parquet")), on=["symbol", "date"], how="left", validate="1:1")
    assert k[["prediction", "b0"]].notna().all().all(), (model, arm, seed)
    k["b0"] = k.groupby("date")["b0"].rank(pct=True)
    return k


def score_job(job):
    """Per date: plain and (for arms holding v2.0) R_SLOT-routed CORR / proxy MMC / payout."""
    model, arm, seed = job
    k = frame(model, arm, seed)
    route = arm not in NO_ROUTE and arm.replace("tune", "") not in ("B0", "P_HIST")
    ob = coverage(arm.replace("tune", ""))
    rows = []
    for d, g in k.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        base = dict(model=model, arm=arm, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0])
        c, m = N.priced(g)
        rows.append(dict(base, mode="plain", corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan,
                         predcorr_b0=g["prediction"].rank().corr(g["b0"]), routed=0.0))
        if route:
            cov = g[ob].to_numpy() > 0
            x = R.slot(g["b0"].to_numpy(), g["prediction"].to_numpy(), cov)
            c, m = N.priced(g.assign(prediction=x))
            rows.append(dict(base, mode="route", corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan,
                             predcorr_b0=pd.Series(x).rank().corr(g["b0"]), routed=1 - cov.mean()))
    return pd.DataFrame(rows)


def leg_job(job):
    """Direct legs: the leg alone (plain, NaN -> 0.5) and its rank blend with B0 at each w, routed on the leg's coverage."""
    model, seed = job
    ds = pd.read_parquet(DATA, columns=["date"] + list(LEGS.values())).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    k = N.keys().merge(preds(model, "B0", seed).rename(columns={"prediction": "b0"}), on=["symbol", "date"], validate="1:1") \
        .merge(ds, on=["symbol", "date"], how="left", validate="1:1")
    k["b0"] = k.groupby("date")["b0"].rank(pct=True)
    rows = []
    for d, g in k.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        base = dict(model=model, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0])
        for leg, col in LEGS.items():
            v = g[col].to_numpy()
            cov = np.isfinite(v)
            alone = pd.Series(np.where(cov, v, 0.5)).rank(pct=True).to_numpy()
            c, m = N.priced(g.assign(prediction=alone))
            rows.append(dict(base, leg=leg, w=-1.0, corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan, covered=cov.mean()))
            lv = pd.Series(np.where(cov, v, np.nan)).rank(pct=True).to_numpy()
            for w in W:
                x = g["b0"].to_numpy() if w == 0 else R.slot(g["b0"].to_numpy(), (1 - w) * g["b0"].to_numpy() + w * np.nan_to_num(lv, nan=0.5), cov)
                c, m = N.priced(g.assign(prediction=x))
                rows.append(dict(base, leg=leg, w=w, corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan, covered=cov.mean()))
    return pd.DataFrame(rows)


def window_of(df):
    return np.where(df["kind"] == "test", "test", np.where(df["block"] >= 2, "dev2", "dev01"))


def levels(S):
    """Level table per model x arm x mode x window (seed-averaged per date first)."""
    A = S.groupby(["model", "arm", "mode", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout", "predcorr_b0", "routed"]].mean()
    A["win"] = window_of(A)
    out = []
    for win_name, sel in (("dev2", A["win"] == "dev2"), ("dev", A["kind"] == "dev"), ("test", A["win"] == "test")):
        for (m, a, md), g in A[sel].groupby(["model", "arm", "mode"]):
            p = g.dropna(subset=["payout"])
            dd = E.max_dd(p["payout"].values) if len(p) else np.nan
            out.append(dict(model=m, arm=a, mode=md, window=win_name, corr=g["corr"].mean(), mmc=p["mmc"].mean(),
                            c_sharpe=g["corr"].mean() / g["corr"].std(ddof=0), dd_corr=E.max_dd(g["corr"].values),
                            payout=p["payout"].mean(), p_sharpe=p["payout"].mean() / p["payout"].std(ddof=0) if len(p) else np.nan,
                            dd_pay=dd, calmar=p["payout"].mean() / dd if dd else np.nan, burn=int((p["payout"] < 0).sum()),
                            n=len(g), n_pay=len(p), predcorr_b0=g["predcorr_b0"].mean(), routed=g["routed"].mean()))
    return pd.DataFrame(out)


def deltas(S, pairs):
    """
    Paired per-date deltas, seed-matched (B0 and P_HIST carry the same seeds as every arm; tuned arms
    and ridge are seed 0), then averaged over seeds per date; NW lag 20 + block bootstrap, per window.
    """
    P = S.pivot_table(index=["model", "mode", "seed", "date", "block", "kind"], columns="arm", values=["corr", "mmc", "payout"])
    rows, byseed, byblock = [], [], []
    for a, b, amode, bmode, label in pairs:
        for m in MODELS:
            try:
                x = P.xs((m, amode), level=("model", "mode"))
                y = P.xs((m, bmode), level=("model", "mode"))
            except KeyError:
                continue
            for metric in ("corr", "mmc", "payout"):
                if (metric, a) not in x or (metric, b) not in y:
                    continue
                d0 = (x[(metric, a)] - y[(metric, b)]).dropna()
                if not len(d0):
                    continue
                d = d0.groupby(level=["date", "block", "kind"]).mean()
                win = window_of(d.index.to_frame(index=False))
                for wn, sel in (("dev2", win == "dev2"), ("dev", win != "test"), ("test", win == "test")):
                    r = E.describe(d[sel])
                    rows.append(dict(model=m, arm=a, vs=b, amode=amode, bmode=bmode, label=label, metric=metric, window=wn,
                                     delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
                if metric != "corr":
                    continue
                for blk, q in d.groupby(level="block"):
                    byblock.append(dict(model=m, arm=a, vs=b, amode=amode, block=blk, mean=q.mean(), n=len(q)))
                for sd, q in d0.groupby(level="seed"):
                    qw = window_of(q.index.to_frame(index=False))
                    for wn in ("dev2", "test"):
                        v = q[qw == wn]
                        if len(v):
                            byseed.append(dict(model=m, arm=a, vs=b, amode=amode, seed=sd, window=wn, mean=v.mean(), t=E.newey_west(v.values)[2]))
    return pd.DataFrame(rows), pd.DataFrame(byseed), pd.DataFrame(byblock)


def pairs():
    r = lambda a: "route" if a not in NO_ROUTE else "plain"
    P = []
    for a in FITS:
        if a in ("ALL_H",):
            continue
        P.append((a, "B0", "plain", "plain", "against v2.0, unrouted"))
        if a not in NO_ROUTE:
            P.append((a, "B0", "route", "plain", "against v2.0, routed"))
    if VARIANT == "rank":
        P += [("P_HIST", "B0", "plain", "plain", "tenure control"),
              ("ALL_H", "P_HIST", "plain", "plain", "head beyond tenure"), ("ALL_H", "P_HIST", "route", "plain", "head beyond tenure, routed")]
        for f in FAM:
            P += [("ALL_HEAD", "ALL_NO_" + f, "route", "route", "%s inside ALL_HEAD" % f), ("ALL_HEAD", "ALL_NO_" + f, "plain", "plain", "%s inside ALL_HEAD, unrouted" % f)]
        P += [("ZS_COV_HEAD", "ZS_AR_HEAD", "route", "route", "covariates over AR, zero-shot"),
              ("FT_COV_HEAD", "FT_AR_HEAD", "route", "route", "covariates over AR, fine-tuned"),
              ("FT_COV_HEAD", "ZS_COV_HEAD", "route", "route", "fine-tuned over zero-shot (cov)"),
              ("FT_FULL_HEAD", "ZS_FULL_HEAD", "route", "route", "fine-tuned over zero-shot (full)"),
              ("ZS_AR_HEAD", "B0", "route", "plain", "AR control against v2.0"), ("FT_AR_HEAD", "B0", "route", "plain", "AR control against v2.0")]
        for a in CAND:
            for ctl in ("P_WARM", "P_OBS"):
                P.append((a, ctl, "plain", "plain", "beyond %s" % ctl))
            P.append((a, "P_HIST", "plain", "plain", "beyond P_HIST"))
        for a in TUNED:
            P += [("tune" + a, "tuneB0", "plain", "plain", "equal-budget retune"), ("tune" + a, "tuneB0", "route", "plain", "equal-budget retune, routed")]
    return P


def legs_summary(L):
    """Blend weight chosen on dev2 routed CORR; dev2/test deltas against B0; the leg alone."""
    A = L.groupby(["model", "leg", "w", "date", "block", "kind"], as_index=False)[["corr", "mmc", "payout"]].mean()
    A["win"] = window_of(A)
    rows = []
    for (m, leg), z in A.groupby(["model", "leg"]):
        lv = z[(z["win"] == "dev2") & (z["w"] >= 0)].groupby("w")["corr"].mean()
        wbest = float(lv.idxmax())
        alone = z[z["w"] == -1.0]
        for metric in ("corr", "payout"):
            p = z.pivot_table(index=["date", "win"], columns="w", values=metric)
            for wn in ("dev2", "test"):
                q = p.xs(wn, level="win")
                for w in sorted(set([wbest] + [x for x in W if x > 0])):
                    r = E.describe(q[w] - q[0.0]) if w > 0 else dict(mean=0.0, nw_t=np.nan, boot_lo=np.nan, boot_hi=np.nan, n=len(q))
                    rows.append(dict(model=m, leg=leg, metric=metric, window=wn, w=w, chosen=w == wbest, level=q[w].mean(), b0=q[0.0].mean(),
                                     delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
                a = alone[alone["win"] == wn].dropna(subset=[metric])
                r = E.describe(a[metric])
                rows.append(dict(model=m, leg=leg, metric=metric, window=wn, w=-1.0, chosen=False, level=r["mean"], b0=np.nan,
                                 delta=np.nan, t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
    return pd.DataFrame(rows)


def check():
    missing = []
    for m in MODELS:
        for a in FITS:
            for s in ([0] if m == "ridge" else SEEDS[m]):
                if not os.path.exists(os.path.join(ROOT, "preds", m, "%s_s%d.parquet" % (a, s))):
                    missing.append((m, a, s))
        if VARIANT == "rank":
            for a in TUNED + (["B0"] if m == "mlp_ktz" else []):
                if not os.path.exists(os.path.join(ROOT, "preds", m, "tune%s_s0.parquet" % a)):
                    missing.append((m, "tune" + a, 0))
    assert not missing, missing


def score(workers):
    check()
    named = ["B0"] + (["P_HIST"] if VARIANT == "rank" else []) + FITS
    jobs = [(m, a, s) for m in MODELS for a in named for s in ([0] if m == "ridge" else SEEDS[m])]
    if VARIANT == "rank":
        jobs += [(m, "tune" + a, 0) for m in MODELS for a in ["B0"] + TUNED]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        S = pd.concat(list(ex.map(score_job, jobs, chunksize=1)), ignore_index=True)
        L = pd.concat(list(ex.map(leg_job, [(m, s) for m in MODELS for s in ([0] if m == "ridge" else SEEDS[m])]))) if VARIANT == "rank" else None
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    levels(S).to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)
    d, ds, db = deltas(S, pairs())
    d.to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    ds.to_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"), index=False)
    db.to_csv(os.path.join(ROOT, "paired_deltas_by_block.csv"), index=False)
    if L is not None:
        L.to_parquet(os.path.join(ROOT, "legs_by_date_seed.parquet"), index=False)
        legs_summary(L).to_csv(os.path.join(ROOT, "legs.csv"), index=False)
    print("wrote", ROOT)


def spreads():
    """Within-date prediction spread per model x arm: a tuned MLP that collapses to a constant is unreadable (E137)."""
    rows = []
    for m in MODELS:
        d = os.path.join(ROOT, "preds", m)
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                p = pd.read_parquet(os.path.join(d, f), columns=["date", "block", "spread"])
                b = p.groupby("block")["spread"].median()
                rows.append(dict(model=m, file=f[:-8], min_block_spread=float(b.min()), collapsed_blocks=int((b < 1e-8).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "spreads.csv"), index=False)


def audit():
    """Point-in-time + fp16-adapter checks (build_c2head --audit), coverage by year, raw IC, redundancy, missingness causality."""
    from src.data_mani import build_c2head
    pit = build_c2head.audit()
    ds = pd.read_parquet(DATA).reset_index()
    ds["date"] = pd.to_datetime(ds["date"])
    head = [c for br in BR for c in cols(br)]
    k = N.keys()
    q = ds.merge(k[["symbol", "date", "block", "kind"]], on=["symbol", "date"])
    o = pd.read_parquet(os.path.join(ROOT, "observed.parquet"))
    z = k.merge(o, on=["symbol", "date"], validate="1:1")
    X = q[head + [H.T20, "date"]]
    ic = X.groupby("date").apply(lambda g: g[head].corrwith(g[H.T20], method="spearman"))
    blk = q.groupby("date")["block"].first()
    icd = {c: dict(mean_dev2=float(ic.loc[blk[ic.index] .between(2, 10).values, c].mean()),
                   t_dev2=float(E.newey_west(ic.loc[blk[ic.index].between(2, 10).values, c].dropna().values)[2]),
                   mean_test=float(ic.loc[(blk[ic.index] == 11).values, c].mean())) for c in head}
    C = q[[c for f in FAM.values() for c in cols(f)]].corr(method="spearman").abs()
    base = [c for c in q if c.startswith("feature_") and c not in head]
    Sb = q[head + base].corr(method="spearman").loc[head, base].abs()
    # missingness depends only on history: a zs forecast exists iff the context holds >= 32 resolved labels
    need = (z["ctx_share"] * 256).round() >= 32
    out = dict(truncation=pit, keys=len(z), same_keys_every_arm=True,
               coverage=dict(zs=float(z["obs_zs"].mean()), ft=float(z["obs_ft"].mean()), both=float(z["obs_all"].mean()), no_feed=float(z["no_feed"].mean())),
               coverage_by_year=z.groupby([z["date"].dt.year, "kind"]).agg(zs=("obs_zs", "mean"), ft=("obs_ft", "mean")).reset_index().to_dict("records"),
               missing_rule=dict(agree=float((need == (z["obs_zs"] > 0)).mean()), rows=len(z)),
               build=json.load(open(os.path.join(DS, "build.json"))),
               date_ic=icd, redundancy=dict(mean_abs_rho=float(np.nanmean(C.where(~np.eye(len(C), dtype=bool)).to_numpy())),
                                            pairs_over_08=int(((C > .8).sum().sum() - len(C)) // 2)),
               vs_base={c: dict(base=Sb.loc[c].idxmax(), rho=float(Sb.loc[c].max())) for c in head})
    json.dump(out, open(os.path.join(ROOT, "audit.json"), "w"), indent=1, default=str)
    print(json.dumps({x: out[x] for x in ("coverage", "missing_rule", "redundancy")}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if a.fit:
        fit(a)
        return
    obs_path = os.path.join(ROOT, "observed.parquet")
    if not os.path.exists(obs_path):
        k = pd.read_parquet(Q.KEYS, columns=["symbol", "date"])
        k["date"] = pd.to_datetime(k["date"])
        k.merge(load()[3], on=["symbol", "date"], how="left", validate="1:1").to_parquet(obs_path, index=False)
    if a.audit:
        audit()
    if a.score:
        score(int(os.environ.get("E145_WORKERS", "16")))
        spreads()


if __name__ == "__main__":
    main()
