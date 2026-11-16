"""E148 -- c2resid: does a forecast of the BTC+ETH beta-residual return path help, and where does any gain come from?

    python3 experiments/e148_c2resid_evaluation.py --standalone          # Part A: each signal ranked, alone
    python3 experiments/e148_c2resid_evaluation.py --fit --model lgb_ktz --n-jobs 4 [--arms FULL]
    python3 experiments/e148_c2resid_evaluation.py --fit --model lgb_ktz --tune --n-jobs 3
    E148_WORKERS=8 python3 experiments/e148_c2resid_evaluation.py         # Part B: score, after every fit

Dataset crypto/v2.0-c2resid (src/data_mani/build_c2resid.py): v2.0 + six binned signals, each a
forecast of the coin's residual return over days D+3..D+30 from days <= D-1:
  RIDGE   causal rolling ridge on window sums of the residual, BTC, ETH and leave-one-out peer paths
  RESMOM  residual momentum, last 20 days (no fit)
  RAW     C2 zero-shot, raw return path alone
  RES     C2 zero-shot, residual path alone
  COV     C2 zero-shot, residual + BTC, ETH, LOO size-peer index as past covariates (headline arm)
  GRP     C2 zero-shot, one group per (date, size bucket): members' residual paths + BTC + ETH

Part A (standalone) ranks each RAW signal within date against target_binned_return_20 and reads the
decomposition on the rows where every signal exists:
  RES - RAW      residualising the input
  COV - RES      adding BTC / ETH / peer-index covariates
  GRP - RES      grouped peer context;  GRP - COV  peers' own paths against their average
  COV - RIDGE    encoding with C2 against a linear model on the same inputs (also GRP - RIDGE)
  RIDGE - RESMOM a fitted linear read against the one-line residual momentum
Part B (marginal) is E146's protocol through E138/E139's machinery: 12 walk-forward blocks, fixed
ridge / lgb_ktz / mlp_ktz (linear), 5 seeds, routed R_SLOT deployment score with each arm routed on
its own signal's coverage, NW lag 20 + block bootstrap, proxy MMC.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402
import e133_histfe_factorial as H  # noqa: E402
import e135_route_histfe as Q  # noqa: E402
import e138_carver_macro_evaluation as T  # noqa: E402
import e139_carver_macro_infl_evaluation as P  # noqa: E402

os.chdir(REPO)
from src.data_mani.history_features import bin_ties_preserved  # noqa: E402

ROOT = "results/reports/crypto/e148_c2resid"
DIR = "data/crypto/v2.0-c2resid"
RAWSIG = "/scratch/danielco/c2resid/signals_raw.parquet"
SIG = ["ridge", "resmom", "raw", "res", "cov", "grp"]
NAME = {s: s.upper() for s in SIG}
E144 = "results/reports/crypto/e144_c2ft/preds"
P.STUDY, P.ROOT, P.DATA, P.META = "E148", ROOT, os.path.join(DIR, "train.parquet"), os.path.join(DIR, "historical_meta_models.parquet")
P.REUSE = {}                                     # mlp tuned B0 is refitted here (T.fit --tune); ridge/lgb tuned B0 come from E133/E135
P.FITS = ["FULL", "ADDED_ONLY"] + ["F_" + NAME[s] for s in SIG] + ["FULL_H", "P_WARM", "P_NM"]
P.SCORED = list(P.FITS)
P.ROUTED = ["FULL"] + ["F_" + NAME[s] for s in SIG] + ["FULL_H"]
P.COMPARE = [(a, "B0", "against v2.0") for a in ["P_HIST"] + P.SCORED if a != "FULL_H"] + \
            [("F_RES", "F_RAW", "residualising the input"), ("F_COV", "F_RES", "BTC / ETH / peer covariates"),
             ("F_GRP", "F_RES", "grouped peer context"), ("F_GRP", "F_COV", "peers' paths vs their average"),
             ("F_COV", "F_RIDGE", "C2 encoding vs ridge, same inputs"), ("F_GRP", "F_RIDGE", "C2 grouped vs ridge"),
             ("F_RIDGE", "F_RESMOM", "fitted linear vs residual momentum"),
             ("FULL_H", "P_HIST", "block beyond tenure"), ("FULL", "P_WARM", "block beyond warm-up"), ("tuneFULL", "tuneB0", "equal-budget retune")]
P.PAIRS = [("FULL", "B0"), ("F_COV", "B0"), ("F_RES", "F_RAW"), ("F_COV", "F_RES"), ("F_GRP", "F_RES"), ("F_COV", "F_RIDGE"),
           ("ADDED_ONLY", "B0"), ("P_HIST", "B0"), ("P_WARM", "B0"), ("P_NM", "B0"), ("FULL_H", "P_HIST")]
DECOMP = [("res", "raw", "residualising the input"), ("cov", "res", "BTC / ETH / peer covariates"), ("grp", "res", "grouped peer context"),
          ("grp", "cov", "peers' paths vs their average"), ("cov", "ridge", "C2 encoding vs ridge, same inputs"),
          ("grp", "ridge", "C2 grouped vs ridge"), ("ridge", "resmom", "fitted linear vs residual momentum")]


def split(cols):
    new = ["feature_c2r_" + s for s in SIG]
    base = [c for c in cols if c.startswith("feature_") and c not in new]
    assert len(base) == 22 and all(c in cols for c in new), (len(base), new)
    return base, new


def load():
    df = pd.read_parquet(P.DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= Q.LAST].sort_values(["date", "symbol"]).reset_index(drop=True)
    base, new = split(list(df))
    tenure = df.groupby("ucid").cumcount().astype("float64").to_frame("feature_hist")
    df["feature_hist"] = bin_ties_preserved(tenure, df["date"])["feature_hist"]
    ever = df.groupby("ucid")["feature_c2r_cov"].transform(lambda s: s.notna().any())
    df["feature_nofeed"] = (~ever).astype("float32")
    share = df[new].notna().mean(axis=1).astype("float64").to_frame("feature_warm")   # causal: how much of the block is observed
    df["feature_warm"] = bin_ties_preserved(share, df["date"])["feature_warm"]
    obs = df[["symbol", "date"]].copy()
    for s in SIG:
        obs["obs_" + s] = df["feature_c2r_" + s].notna().astype("float32").values
    obs["obs_tf"] = df[new].notna().mean(axis=1).astype("float32").values     # T's name: the arm-agnostic coverage
    obs["no_feed"] = df["feature_nofeed"].values
    df[base + new + ["feature_hist", "feature_warm"]] = df[base + new + ["feature_hist", "feature_warm"]].fillna(0.5)
    meta = pd.read_parquet(P.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    return df.merge(meta, on=["symbol", "date"], how="left"), base, new, [], obs


def arms(base, new, _):
    A = {"B0": base, "B0CHK": base, "FULL": base + new, "ADDED_ONLY": new, "FULL_H": base + new + ["feature_hist"],
         "P_NM": base + ["feature_nofeed"], "P_WARM": base + ["feature_warm"]}
    for s in SIG:
        A["F_" + NAME[s]] = base + ["feature_c2r_" + s]
    assert [len(A[k]) for k in ("FULL", "ADDED_ONLY", "F_COV")] == [28, 6, 23]
    return A


def coverage(arm, o):
    s = arm[2:].lower() if arm.startswith("F_") else None
    return o["obs_" + s] if s in SIG else o["obs_tf"]


# ------------------------------------------------------------------ Part A: standalone
def standalone_job(job):
    """Per-date scores of one raw signal (and its sign-flip check), on all its rows and on the common rows."""
    s, = job
    N = P.N
    k = N.keys()
    raw = pd.read_parquet(RAWSIG, columns=["symbol", "date"] + SIG)
    raw["date"] = pd.to_datetime(raw["date"])
    f = k.merge(raw, on=["symbol", "date"], how="left", validate="1:1")
    f["common"] = f[SIG].notna().all(axis=1)
    rows = []
    for d, g in f.groupby("date", sort=True):
        for rs, x in (("own", g[g[s].notna()]), ("common", g[g["common"]])):
            if len(x) < 10 or x[s].nunique() < 2:
                continue
            h = x.assign(prediction=x[s].rank(pct=True))
            c, m = N.priced(h)
            rows.append(dict(signal=s, rows=rs, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0], n_coins=len(x),
                             corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan))
    return pd.DataFrame(rows)


def standalone():
    os.makedirs(ROOT, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        S = pd.concat(list(ex.map(standalone_job, [(s,) for s in SIG])), ignore_index=True)
    S.to_parquet(os.path.join(ROOT, "standalone_by_date.parquet"), index=False)
    lev, dl = [], []
    for (s, rs, sc), x in S.groupby(["signal", "rows", "kind"]):
        p = x.dropna(subset=["payout"])
        dd = E.max_dd(p["payout"].values)
        rc = E.describe(x["corr"])
        lev.append(dict(signal=s, rows=rs, scope=sc, corr=x["corr"].mean(), corr_t=rc["nw_t"], corr_lo=rc["boot_lo"], corr_hi=rc["boot_hi"],
                        mmc=p["mmc"].mean(), c_sharpe=x["corr"].mean() / x["corr"].std(ddof=0), payout=p["payout"].mean(),
                        p_sharpe=p["payout"].mean() / p["payout"].std(ddof=0), dd_pay=dd, calmar=p["payout"].mean() / dd if dd > 0 else np.nan,
                        burn=int((p["payout"] < 0).sum()), n=len(x), n_pay=len(p), coins=x["n_coins"].mean()))
    C = S[S["rows"] == "common"]
    for a, b, label in DECOMP:
        z = C[C.signal == a].merge(C[C.signal == b], on=["date", "kind"], suffixes=("", "_b"))
        for metric in ("corr", "mmc", "payout"):
            for sc in ("dev", "test"):
                q = z[z["kind"] == sc]
                r = E.describe((q[metric] - q[metric + "_b"]).dropna())
                dl.append(dict(arm=a, vs=b, label=label, metric=metric, scope=sc, delta=r["mean"], t=r["nw_t"], lo=r["boot_lo"], hi=r["boot_hi"], n=r["n"]))
    pd.DataFrame(lev).to_csv(os.path.join(ROOT, "standalone_levels.csv"), index=False)
    pd.DataFrame(dl).to_csv(os.path.join(ROOT, "standalone_deltas.csv"), index=False)
    raw = pd.read_parquet(RAWSIG, columns=["date"] + SIG)
    rc = raw.dropna().groupby("date")[SIG].corr(method="spearman").groupby(level=1).mean().loc[SIG, SIG]
    rc.to_csv(os.path.join(ROOT, "signal_rank_corr.csv"))
    print(pd.DataFrame(lev).round(4).to_string())


# ------------------------------------------------------------------ CPU half of the audit
def audit_cpu(dates=("2021-12-31", "2023-06-30", "2026-02-20")):
    """Ridge inputs, residual momentum and the size-peer index rebuilt from spot cut at D-1, against the full paths."""
    import json
    sys.path.insert(0, REPO)
    from src.data_mani import build_c2resid as B, c2resid_paths as RP
    spot = pd.read_parquet(RP.SPOT)
    spot["date"] = pd.to_datetime(spot["date"])
    full = dict(zip(("days", "ucids", "r", "e"), RP.build(spot.copy())))
    out = []
    for D in dates:
        D = pd.Timestamp(D)
        cut = dict(zip(("days", "ucids", "r", "e"), RP.build(spot.copy(), cut=D - pd.Timedelta(days=RP.LAG))))
        res = {}
        for nm, pth in (("full", full), ("cut", cut)):
            rows = RP.panel_rows(days=pth["days"], ucids=pth["ucids"])
            rows = rows[rows["date"] == D]
            X, mom, ok, _ = B.date_features(pth, rows)
            peer = RP.peer_windows(pth["r"], rows)
            res[nm] = (X, mom, ok, peer)
        a, b = res["full"], res["cut"]
        x1, x2 = a[0].to_numpy(), b[0].to_numpy()
        eq = np.isclose(x1, x2, rtol=0, atol=1e-6) | (np.isnan(x1) & np.isnan(x2))
        pe = np.isclose(a[3], b[3], rtol=0, atol=1e-6) | (np.isnan(a[3]) & np.isnan(b[3]))
        me = np.isclose(a[1], b[1], rtol=0, atol=1e-6) | (np.isnan(a[1]) & np.isnan(b[1]))
        out.append(dict(date=str(D.date()), rows=len(x1), ridge_input_cells=int(eq.size), ridge_input_differ=int((~eq).sum()),
                        peer_cells=int(pe.size), peer_differ=int((~pe).sum()), resmom_differ=int((~me).sum()), eligible_differ=int((a[2] != b[2]).sum())))
        print(out[-1], flush=True)
    json.dump(out, open(os.path.join(ROOT, "audit_cpu.json"), "w"), indent=1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--audit-cpu", action="store_true")
    ap.add_argument("--model", choices=T.MODELS)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    if a.standalone:
        standalone()
        return
    if a.audit_cpu:
        os.makedirs(ROOT, exist_ok=True)
        audit_cpu()
        return
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
    P.score(int(os.environ.get("E148_WORKERS", "8")))


P.load, P.arms, P.coverage = load, arms, coverage
if __name__ == "__main__":
    main()
