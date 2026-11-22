#!/usr/bin/env python3
"""
E202 -- the four best E193-E197 arms on the 60-day label, residualised, blended with three live slots
(registered in experiments/EXPERIMENTS.md and src/analysis/experiment_db.py before launch).

    python3 experiments/e202_tsfm_t60.py --fit --arms B_ALL60 T60_timestral   # 60-day ridge fits (default: all missing)
    python3 experiments/e202_tsfm_t60.py --score                              # routed per-date scores on the 20-day label
    python3 experiments/e202_tsfm_t60.py --export                             # 8 routed runs + 8 residualised -> results/
    python3 experiments/e202_tsfm_t60.py --ensembles                          # prediction-level blends with the live slots
    python3 experiments/e202_tsfm_t60.py --report                             # study HTML (then src.analysis.html_pdf)

The 60-day arms are E189/E193's N_TS_MM arm with two changes: the frozen states come from
/scratch/danielco/tsfmstates60/<model> (label series = target_binned_return_60, wall D-91d, written by
src/data_mani/tsfm_states.py --target 60), and the ridge is tuned and fitted on target_binned_return_60.
Every 60-day label in a block's training window must be resolved when the block starts: the label
lags the 20-day one by 56 days in the frozen panel, so the outer window (130 dates, as E131) ends
PURGE60 = 22 + 56 dates before the block instead of 22. The inner split is E189's (last 40 training
dates validate, 22-date purge). Scoring is E193's: covered rows slotted into B_ALL's ranks, payable
20-day label, the E131 scorer. The 20-day twins are the E193-E197 N_TS_MM predictions, unchanged.
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e133_histfe_factorial as H  # noqa: E402
import e189_c2norm as M  # noqa: E402

STUDY = "E202"
ROOT = "results/reports/crypto/e202_tsfm_t60"
E189_ROOT = "results/reports/crypto/e189_c2norm"
STATES60 = "/scratch/danielco/tsfmstates60"
PANEL = "data/crypto/v2.0-asof-e189/train.parquet"
T20, T60 = "target_binned_return_20", "target_binned_return_60"
EXTRA = 56                                   # days the 60-day label lags the 20-day one in the frozen panel
PURGE60 = H.PURGE + EXTRA                    # dates between a 60-day training window and its block
WALL60_DAYS = 91
ARM_TF = "N_TS_MM"
MODELS = {"timestral": ("e197", "Timestral"), "tabpfnts": ("e194", "TabPFN-TS"),
          "toto": ("e193", "Toto 2.0"), "t0": ("e196", "t0")}
ARMS = ["T60_%s" % m for m in MODELS] + ["B_ALL60"]
TREE = "tsfm_states"
SLOTS = ["CU_SA_IA_BO_MA_R_01", "CU_SA_IA_BO_MA_01", "CU_SA_IA_BO_02"]
SPLIT = pd.Timestamp("2026-02-20")           # blends are picked on common dates before this, read after


def twin_path(m):
    sid = MODELS[m][0]
    return "results/reports/crypto/%s_%snorm/preds/%s.parquet" % (sid, m, ARM_TF)


# ------------------------------------------------------------------------------------------ data
def load(model):
    """
    The panel <= LAST in (date, ucid) order with the model's 60-day states. Coverage is the 60-day
    rule, recomputed here by tsfm_states.expected_covered and asserted equal to the cache's row set.
    model None: no states (B_ALL60), every row eligible.
    """
    from src.data_mani import chronos2_syn_repr as C
    from src.data_mani import tsfm_states as TS
    C.TARGET, C.WALL_DAYS = T60, WALL60_DAYS
    full, _, grid = C.load_panel(PANEL)
    rule = TS.expected_covered(full, grid, C.wall_steps(grid))
    df = full[full["date"] <= M.LAST].reset_index(drop=True)
    rule = rule[rule < len(df)]
    assert (df["date"].iloc[rule] <= M.LAST).all()
    feats = sorted(c for c in df if c.startswith("feature_"))
    emb = np.zeros((0, 1), np.float32)
    df["covered"] = False
    df["emb_row"] = -1
    if model is None:
        df["covered"] = True
    else:
        cache = os.path.join(STATES60, model)
        files = sorted(f for f in os.listdir(cache) if f.startswith("part_") and f.endswith(".npz"))
        ids = np.concatenate([np.load(os.path.join(cache, f))["cov_rows"] for f in files])
        reps = np.concatenate([np.load(os.path.join(cache, f))["cov_reps"] for f in files])
        keep = ids < len(df)
        ids, reps = ids[keep], reps[keep]
        assert np.array_equal(np.sort(ids), rule), "60-day state rows differ from the 60-day covered rule"
        order = np.argsort(ids)
        ids, emb = ids[order], reps[order].astype(np.float32)
        assert np.isfinite(emb).all()
        df.loc[ids, "covered"] = True
        df.loc[ids, "emb_row"] = np.arange(len(ids))
    g = df[M.GROUP_COL].to_numpy()
    df["group"] = np.where(g <= 0.25, 0, np.where(g == 0.5, 1, 2)).astype(np.int8)
    df[feats] = df[feats].fillna(0.5)
    meta = pd.read_parquet(M.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    df = df.merge(meta, on=["symbol", "date"], how="left", validate="1:1")
    return df, emb, feats


def schedule60(df):
    """E189's blocks with each training window moved back so it ends PURGE60 dates before the block."""
    B = M.schedule(df)
    past = np.array(sorted(df.loc[df["date"] < M.TEST_START, "date"].unique()), dtype="datetime64[ns]")
    out = []
    for i, tr, te, kind in B:
        b = np.searchsorted(past, te[0]) if kind == "dev" else len(past)
        tr60 = past[max(0, b - PURGE60 - H.TRAIN):b - PURGE60]
        gap_days = (pd.Timestamp(te[0]) - pd.Timestamp(tr60[-1])).days
        assert gap_days >= WALL60_DAYS - 5, (i, gap_days)     # >= the 20-day schedule's 30 days + 56
        out.append((i, tr60, te, kind))
    return out


class F60(M.Features):
    """E189's Features; eligible fit / validation rows also need a resolved 60-day label."""

    def __init__(self, arm, df, emb, feats):
        super().__init__(arm if arm != "B_ALL60" else "B_ALL", df, emb, feats)
        self.lab = np.isfinite(df[T60].to_numpy())

    def eligible(self, rows):
        return super().eligible(rows) & self.lab

    def predictable(self, rows):
        return super().eligible(rows)


def fit_arm(arm, df, emb, feats, B):
    path = os.path.join(ROOT, "preds", arm + ".parquet")
    if os.path.exists(path):
        print("skip", arm, flush=True)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    M.T20 = T60                                    # M.tune reads its label from this module global
    t0 = time.time()
    Fx = F60(ARM_TF if arm.startswith("T60_") else arm, df, emb, feats)
    dates = df["date"].to_numpy()
    y60 = df[T60].to_numpy()
    parts, picks = [], []
    for i, tr, te, kind in B:
        t1 = time.time()
        meta, _ = M.tune(Fx, df, tr)
        fit = Fx.eligible(np.isin(dates, tr))
        tst = Fx.predictable(np.isin(dates, te))
        app = fit | tst
        X, info = Fx.build(fit, app)
        fi, ti = fit[app], tst[app]
        assert np.isfinite(y60[fit]).all()
        assert df.loc[fit, "date"].max() + pd.Timedelta(days=WALL60_DAYS - 5) <= pd.Timestamp(te[0])
        pred = M.ridge(meta["alpha"], X[fi], y60[fit], X[ti])
        assert np.isfinite(pred).all()
        o = df.loc[np.isin(dates, te), ["symbol", "ucid", "date", T20, "meta_model", "covered", "group"]].copy()
        o["prediction"] = np.nan
        o.loc[tst[np.isin(dates, te)], "prediction"] = pred
        o["block"], o["kind"] = i, kind
        parts.append(o)
        meta.update(block=i, kind=kind, n_train=int(fit.sum()), n_test=int(tst.sum()), train=[str(tr[0])[:10], str(tr[-1])[:10]],
                    ts_fallback_assets=info.get("fallback_assets"), seconds=time.time() - t1)
        picks.append(meta)
        print("%s block %2d alpha %.3g%s  %d/%d rows  train %s..%s  %.0fs" % (
            arm, i, meta["alpha"], " BOUNDARY" if meta["boundary"] else "", meta["n_train"], meta["n_test"],
            meta["train"][0], meta["train"][1], meta["seconds"]), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path + ".partial", index=False)
    json.dump({"arm": arm, "target_fit": T60, "purge60": PURGE60, "picks": picks, "seconds": time.time() - t0,
               "host": os.uname().nodename}, open(os.path.join(ROOT, "preds", arm + ".picks.json"), "w"), indent=1, default=float)
    os.replace(path + ".partial", path)
    print("%s done in %.1f min" % (arm, (time.time() - t0) / 60), flush=True)


# ---------------------------------------------------------------------------------- scoring
def arm_paths():
    """Every arm scored here: the 60-day fits, their 20-day twins, the references."""
    p = {"T20_%s" % m: twin_path(m) for m in MODELS}
    p.update({a: os.path.join(ROOT, "preds", a + ".parquet") for a in ARMS})
    p.update({"B_ALL": os.path.join(E189_ROOT, "preds", "B_ALL.parquet"), "B_COV": os.path.join(E189_ROOT, "preds", "B_COV.parquet")})
    return p


def routed(arm, path):
    """Per-row routed prediction (E193's R_SLOT into B_ALL's per-date ranks), keyed like B_ALL."""
    import e134_route_feature_sets as R
    b = pd.read_parquet(arm_paths()["B_ALL"])
    c = pd.read_parquet(path)
    assert (b[["symbol", "date"]].to_numpy() == c[["symbol", "date"]].to_numpy()).all(), arm
    cov = c["covered"].to_numpy() if "covered" in c else np.ones(len(c), bool)
    if arm != "B_ALL":
        assert (c["prediction"].notna().to_numpy() == cov).all(), arm
    x = np.empty(len(b))
    for _, idx in b.groupby("date", sort=True).indices.items():
        bb = b["prediction"].iloc[idx].rank(pct=True).to_numpy()
        if arm == "B_ALL":
            x[idx] = bb
        else:
            cv = cov[idx]
            x[idx] = R.slot(bb, c["prediction"].to_numpy()[idx], cv)
            assert np.array_equal(x[idx][~cv], bb[~cv])
    out = b[["symbol", "date", T20, "meta_model", "block", "kind"]].copy()
    out["prediction"] = x
    out["covered"] = cov
    return out


def _score(item):
    import e131_analysis as E
    arm, path = item
    r = routed(arm, path)
    rows = []
    for d, h in r.groupby("date", sort=True):
        hr = h.set_index("symbol")
        corr = E.corr_one(hr)
        mmc = E.contrib_one(hr, "meta_model") if h["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(arm=arm, date=d, block=int(h["block"].iloc[0]), kind=h["kind"].iloc[0], n=len(h),
                         n_cov=int(h["covered"].sum()), corr=corr, mmc=mmc, payout=0.1 * corr + mmc if mmc == mmc else np.nan))
    return pd.DataFrame(rows)


def score(workers):
    P = arm_paths()
    missing = [a for a, p in P.items() if not os.path.exists(p)]
    assert not missing, missing
    with ProcessPoolExecutor(workers) as ex:
        S = pd.concat(ex.map(_score, P.items()), ignore_index=True)
    # the 20-day twins must reproduce E193-E197's stored scores (same routing, scorer, rows)
    for m, (sid, _) in MODELS.items():
        old = pd.read_parquet("results/reports/crypto/%s_%snorm/scores.parquet" % (sid, m))
        old = old[old["arm"] == ARM_TF].set_index("date")["payout"]
        new = S[S["arm"] == "T20_%s" % m].set_index("date")["payout"]
        assert np.allclose(old.reindex(new.index), new, atol=1e-12, equal_nan=True), m
    S.to_parquet(os.path.join(ROOT, "scores.parquet"), index=False)
    print(S.groupby(["arm", "kind"])["payout"].mean().unstack().round(4).to_string())


def table():
    import e131_analysis as E
    S = pd.read_parquet(os.path.join(ROOT, "scores.parquet"))
    rows = []
    for a, x in S.groupby("arm"):
        d, t = x[x["kind"] == "dev"], x[x["kind"] == "test"]
        dp, tp = E.describe(d["payout"]), E.describe(t["payout"])
        rows.append(dict(arm=a, payout=dp["mean"], sharpe_p=dp["sharpe"], maxdd_p=dp["max_dd"], corr=d["corr"].mean(),
                         mmc=d["mmc"].mean(), n=dp["n"], test_payout=tp["mean"], test_sharpe_p=tp["sharpe"]))
    T = pd.DataFrame(rows).sort_values("payout", ascending=False).reset_index(drop=True)
    pairs = []
    for m in MODELS:
        w = S[S["arm"].isin(["T60_" + m, "T20_" + m])].pivot(index="date", columns="arm", values="payout")
        k = S.drop_duplicates("date").set_index("date")["kind"]
        for kind in ("dev", "test"):
            dd = (w["T60_" + m] - w["T20_" + m])[k.reindex(w.index) == kind]
            mu, se, t = E.newey_west(dd.values)
            pairs.append(dict(model=MODELS[m][1], window=kind, d_payout=mu, nw_t=t, n=int(dd.notna().sum())))
    T.to_csv(os.path.join(ROOT, "table_by_payout.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(ROOT, "t60_vs_t20.csv"), index=False)
    return T, pd.DataFrame(pairs)


# ---------------------------------------------------------------------------------- export + residualise
def lagged_meta_matrix(frame, meta):
    """Per row, the symbol's meta prediction on the latest meta date <= date - LAG_DAYS (meta_resid's rule)."""
    from src.prediction import meta_resid as MR
    wide = meta.pivot_table(index="date", columns="symbol", values="prediction", aggfunc="last").sort_index()
    md = wide.index.to_numpy()
    out = np.full(len(frame), np.nan)
    snap = {}
    for d, idx in frame.groupby("date").indices.items():
        pos = np.searchsorted(md, np.datetime64(pd.Timestamp(d) - pd.Timedelta(days=MR.LAG_DAYS)), side="right") - 1
        if pos < 0:
            snap[d] = None
            continue
        snap[d] = pd.Timestamp(md[pos])
        out[idx] = wide.iloc[pos].reindex(frame["symbol"].to_numpy()[idx]).to_numpy(float)
    return out, snap


def write_run(name, frame, config):
    from src.prediction.metrics import per_period_table, summarize
    table = per_period_table(frame, "prediction", T20, "date", meta_model_col="meta_model")
    res = {"n_periods": int(table["corr"].notna().sum())}
    for fam in ("corr", "mmc"):
        res[fam] = summarize(table[fam])
        if fam != "corr":
            res["%s_n_periods" % fam] = int(table[fam].notna().sum())
    res["config"] = dict(config, model=TREE, tournament="crypto", task="regression", study=STUDY, target=T20,
                         assembled_by="experiments/e202_tsfm_t60.py")
    for kind in ("forecast", "per_period", "metrics"):
        os.makedirs(os.path.join("results", kind, "crypto", TREE), exist_ok=True)
    base = os.path.join("results", "%s", "crypto", TREE, name)
    frame[["symbol", "date", T20, "meta_model", "prediction"]].to_csv(base % "forecast" + ".csv", index=False)
    table.to_csv(base % "per_period" + ".csv")
    json.dump(res, open(base % "metrics" + ".json", "w"), indent=2, sort_keys=True, default=str)
    return table


def export():
    """The 8 routed models as runs, and each residualised on the >= 5-day-old meta model (E107 transform)."""
    from src.prediction import meta_resid as MR
    meta = pd.read_parquet(M.META, columns=["date", "symbol", "prediction"])
    meta["date"] = pd.to_datetime(meta["date"])
    P = arm_paths()
    first = True
    diag = []
    for m in MODELS:
        for lab in ("T20", "T60"):
            arm = "%s_%s" % (lab, m)
            r = routed(arm, P[arm]).sort_values(["date", "symbol"]).reset_index(drop=True)
            name = "e202_%s_%s" % (m, lab.lower())
            cfg = dict(source=P[arm], state_label=T20 if lab == "T20" else T60, ridge_label=T20 if lab == "T20" else T60,
                       arm=ARM_TF, routing="R_SLOT into B_ALL ranks on covered rows")
            write_run(name, r, cfg)
            lm, snap = lagged_meta_matrix(r, meta)
            if first:                                 # the vectorised lag == meta_resid.lagged_meta on a few dates
                for d in list(snap)[::150]:
                    idx = np.flatnonzero(r["date"].to_numpy() == np.datetime64(d))
                    v, s = MR.lagged_meta(meta, r["symbol"].to_numpy()[idx], d)
                    assert s == snap[d] and np.allclose(v, lm[idx], equal_nan=True), d
                first = False
            res = np.empty(len(r))
            for d, idx in r.groupby("date").indices.items():
                rr, beta = MR.residualise(r["prediction"].to_numpy()[idx], lm[idx] if snap[d] is not None else None)
                v = pd.Series(rr)
                res[idx] = ((v.rank(method="average") - 0.5) / v.count()).to_numpy()
                diag.append(dict(run=name, date=d, beta=beta))
            write_run(name + "_r", r.assign(prediction=res),
                      dict(cfg, source_run="%s/%s" % (TREE, name), strength=MR.STRENGTH, meta_lag_days=MR.LAG_DAYS,
                           transform="rank(gauss(rank(p)) - s * beta * gauss(rank(m_lag))), per day", meta_source=M.META))
            print("exported", name, name + "_r", flush=True)
    pd.DataFrame(diag).to_parquet(os.path.join(ROOT, "resid_beta.parquet"), index=False)


def runs():
    return ["e202_%s_%s%s" % (m, lab, r) for m in MODELS for lab in ("t20", "t60") for r in ("", "_r")]


# ---------------------------------------------------------------------------------- ensembles
_G = {}


def _blend_one(spec):
    from src.prediction.metrics import per_period_table
    members, weights = spec
    R, base = _G["R"], _G["base"]
    p = sum(w * R[k] for k, w in zip(members, weights))
    t = per_period_table(base.assign(prediction=p), "prediction", T20, "date", meta_model_col="meta_model")
    return spec, (0.1 * t["corr"] + t["mmc"]).rename(None), t["corr"], t["mmc"]


def ensembles(workers):
    """
    Prediction-level per-date rank averages over the 16 new runs + the three live slots on their common
    (symbol, date) rows. Every equal-weight 2- and 3-member blend, plus slot x new run at 0.8/0.2 and 0.5/0.5.
    """
    from src.analysis import slots as SL
    srcs = {}
    for n in runs():
        srcs[n] = os.path.join("results/forecast/crypto", TREE, n + ".csv")
    for s in SLOTS:
        model, run, _ = SL.slot_run(s)
        srcs[s] = "results/forecast/crypto/%s/%s.csv" % (model, run)
    frames = {}
    for k, path in srcs.items():
        f = pd.read_csv(path, usecols=lambda c: c in ("symbol", "date", T20, "meta_model", "prediction"), parse_dates=["date"])
        frames[k] = f.dropna(subset=["prediction"]).drop_duplicates(["symbol", "date"])
    keys = None
    for k, f in frames.items():
        kk = f[["symbol", "date"]]
        keys = kk if keys is None else keys.merge(kk, on=["symbol", "date"])
    base = keys.merge(frames[runs()[0]][["symbol", "date", T20, "meta_model"]], on=["symbol", "date"])
    base = base.sort_values(["date", "symbol"]).reset_index(drop=True)
    agree = {}
    for s in SLOTS:                                    # the slots' stored labels vs the frozen panel's
        j = base.merge(frames[s][["symbol", "date", T20]], on=["symbol", "date"], suffixes=("", "_s"))
        agree[s] = float(np.isclose(j[T20], j[T20 + "_s"], equal_nan=True).mean())
    R = {}
    for k, f in frames.items():
        x = base[["symbol", "date"]].merge(f[["symbol", "date", "prediction"]], on=["symbol", "date"], how="left")
        assert x["prediction"].notna().all()
        R[k] = x.groupby("date")["prediction"].rank(pct=True).to_numpy()
    new = runs()
    specs = [((k,), (1.0,)) for k in srcs]
    pool = list(srcs)
    for size in (2, 3):
        specs += [(c, tuple([1.0 / size] * size)) for c in itertools.combinations(pool, size)]
    for s in SLOTS:
        for n in new:
            specs.append(((s, n), (0.8, 0.2)))
    _G.update(R=R, base=base)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_blend_one, specs, chunksize=8))
    pay = pd.DataFrame({"+".join("%s@%.3g" % (k, w) for k, w in zip(*sp)): p for sp, p, _, _ in out})
    corr = pd.DataFrame({"+".join("%s@%.3g" % (k, w) for k, w in zip(*sp)): c for sp, _, c, _ in out})
    mmc = pd.DataFrame({"+".join("%s@%.3g" % (k, w) for k, w in zip(*sp)): x for sp, _, _, x in out})
    pay.to_parquet(os.path.join(ROOT, "ens_payout.parquet"))
    corr.to_parquet(os.path.join(ROOT, "ens_corr.parquet"))
    mmc.to_parquet(os.path.join(ROOT, "ens_mmc.parquet"))
    info = dict(rows=len(base), dates=int(base["date"].nunique()), first=str(base["date"].min().date()),
                last=str(base["date"].max().date()), label_agreement=agree, n_blends=len(specs),
                rows_per_source={k: len(f) for k, f in frames.items()})
    json.dump(info, open(os.path.join(ROOT, "ens_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


def ens_tables():
    import e131_analysis as E
    pay = pd.read_parquet(os.path.join(ROOT, "ens_payout.parquet"))
    corr = pd.read_parquet(os.path.join(ROOT, "ens_corr.parquet"))
    mmc = pd.read_parquet(os.path.join(ROOT, "ens_mmc.parquet"))
    pay.index = pd.to_datetime(pay.index)
    sel, rd = pay.index < SPLIT, pay.index >= SPLIT

    def row(k):
        s, r, a = E.describe(pay.loc[sel, k]), E.describe(pay.loc[rd, k]), E.describe(pay[k])
        return dict(blend=k, sel_payout=s["mean"], sel_sharpe_p=s["sharpe"], read_payout=r["mean"], read_sharpe_p=r["sharpe"],
                    payout=a["mean"], sharpe_p=a["sharpe"], maxdd_p=a["max_dd"], corr=corr[k].mean(), mmc=mmc[k].mean())
    T = pd.DataFrame([row(k) for k in pay.columns])
    T["members"] = T["blend"].str.count(r"\+") + 1
    T["has_slot"] = T["blend"].apply(lambda b: any(s in b for s in SLOTS))
    T["n_new"] = T["blend"].apply(lambda b: sum(1 for part in b.split("+") if part.startswith("e202_")))
    vs = []
    for s in SLOTS:
        alone = "%s@1" % s
        cands = T[T["blend"].str.contains(s, regex=False) & (T["n_new"] > 0)].sort_values("sel_payout", ascending=False)
        for k in cands["blend"].head(3):
            for w, m in (("read", rd), ("all", slice(None))):
                d = pay.loc[m, k] - pay.loc[m, alone]
                mu, se, t = E.newey_west(d.values)
                vs.append(dict(slot=s, blend=k, window=w, d_payout=mu, nw_t=t, n=int(d.notna().sum())))
    return T, pd.DataFrame(vs)


# ---------------------------------------------------------------------------------- report
def report():
    import html
    T, pairs = table()
    ET, vs = ens_tables()
    info = json.load(open(os.path.join(ROOT, "ens_info.json")))
    from src.analysis.results_table import payout_series  # noqa: F401  (same payout definition as the slots)
    exp = []
    for n in runs():
        pp = pd.read_csv(os.path.join("results/per_period/crypto", TREE, n + ".csv"), parse_dates=["date"]).set_index("date")
        p = 0.1 * pp["corr"] + pp["mmc"]
        exp.append((n, p))
    S = pd.read_parquet(os.path.join(ROOT, "scores.parquet"))
    kind = S.drop_duplicates("date").set_index("date")["kind"]
    import e131_analysis as E
    xr = []
    for n, p in exp:
        k = kind.reindex(p.index)
        d, t = E.describe(p[k == "dev"]), E.describe(p[k == "test"])
        pp = pd.read_csv(os.path.join("results/per_period/crypto", TREE, n + ".csv"), parse_dates=["date"]).set_index("date")
        xr.append(dict(run=n, payout=d["mean"], sharpe_p=d["sharpe"], maxdd_p=d["max_dd"], corr=pp.loc[k == "dev", "corr"].mean(),
                       mmc=pp.loc[k == "dev", "mmc"].mean(), n=d["n"], test_payout=t["mean"], test_sharpe_p=t["sharpe"]))
    XR = pd.DataFrame(xr).sort_values("payout", ascending=False)
    XR.to_csv(os.path.join(ROOT, "resid_table.csv"), index=False)
    ET.to_csv(os.path.join(ROOT, "ens_table.csv"), index=False)
    vs.to_csv(os.path.join(ROOT, "ens_vs_slot.csv"), index=False)

    f4 = "{:.4f}".format
    f3 = "{:.3f}".format
    fm = {c: f4 for c in ("payout", "corr", "mmc", "test_payout", "sel_payout", "read_payout", "d_payout")}
    fm.update({c: f3 for c in ("sharpe_p", "maxdd_p", "test_sharpe_p", "sel_sharpe_p", "read_sharpe_p", "nw_t")})

    def tab(df):
        return df.to_html(index=False, formatters={k: v for k, v in fm.items() if k in df}, classes="t", border=0)

    singles = ET[ET["members"] == 1].sort_values("payout", ascending=False)
    top = ET[ET["members"] > 1].sort_values("sel_payout", ascending=False).head(15)
    top_slot = ET[(ET["members"] > 1) & ET["has_slot"] & (ET["n_new"] > 0)].sort_values("sel_payout", ascending=False).head(15)
    cols = ["blend", "sel_payout", "sel_sharpe_p", "read_payout", "read_sharpe_p", "payout", "sharpe_p", "maxdd_p", "corr", "mmc"]
    css = ("body{font-family:Helvetica,Arial,sans-serif;font-size:10px;margin:18px;color:#111}h1{font-size:16px}"
           "h2{font-size:13px;margin-top:22px}.m{color:#333}table.t{border-collapse:collapse;margin:6px 0 12px}"
           "table.t td,table.t th{padding:2px 6px;text-align:right;border-bottom:1px solid #ddd}"
           "table.t td:first-child,table.t th:first-child{text-align:left}")
    doc = """<!doctype html><html><head><meta charset=utf-8><title>E202 TSFM 60-day</title><style>%s</style></head><body>
<h1>E202: top-four TSFM states on the 60-day label, residualised, blended with live slots</h1>
<p class=m>The four best E193-E197 arms (N_TS_MM of Timestral, TabPFN-TS, Toto 2.0, t0). <b>T20_*</b> = the E193-E197
predictions (20-day label series in the state, ridge on the 20-day label). <b>T60_*</b> = states re-extracted with
target_binned_return_60 as the label series (wall D-91d, 417,199 covered rows) and the ridge tuned and fitted on the 60-day
label (training window ending %d dates before each block so every 60-day label is resolved). Everything is SCORED on the
payable 20-day label, routed into B_ALL's ranks on covered rows (E193's scorer). Dev = E131's 11 blocks (468 periods);
test = 2026-02-20..2026-08-20 (never used for a choice). B_ALL60 = v2.0 ridge fitted on the 60-day label, same schedule.
payout = mean of 0.1&middot;corr + mmc per period; sharpe_p = its mean/std; maxdd_p = max drawdown of cumulative payout
(positive, bigger is worse). Same n for every row, so maxdd_p is comparable.</p>
<h2>1. 60-day vs 20-day, routed, scored on the 20-day label (sorted by dev payout)</h2>%s
<p class=m>Paired Newey-West (lag 20), T60 minus T20 per model:</p>%s
<h2>2. The eight models, plain and meta-residualised</h2>
<p class=m>E107 transform (src.prediction.meta_resid): rank(gauss(rank p) - 0.5&middot;beta&middot;gauss(rank m)), m = meta model
&ge; 5 days old, per date. Scored with per_period_table (the scorer behind the slots' series), so levels differ slightly from
section 1's E131 scorer. Suffix _r = residualised.</p>%s
<h2>3. Ensembles with the live slots</h2>
<p class=m>Prediction-level per-date rank averages on the rows every source covers: %d rows, %d dates, %s..%s. Sources:
the 16 runs of section 2 and the registry backtest forecasts of %s. Every equal-weight 1-, 2- and 3-member blend
plus each slot at 0.8 with each new run at 0.2 (%d blends). <b>sel</b> = common dates before 2026-02-20 (where blends are
ranked), <b>read</b> = 2026-02-20 onward (%d dates, out of sample for the choice only: the slots themselves were built with
data through 2026-05). Labels: slot forecasts agree with the frozen panel's 20-day label on %s of rows.</p>
<h3>Singles on the common window</h3>%s
<h3>Top 15 blends containing a slot and at least one new run, ranked on sel payout</h3>%s
<h3>Top 15 blends overall, ranked on sel payout</h3>%s
<h3>Best three slot+new blends per slot vs the slot alone (paired Newey-West, lag 20)</h3>%s
</body></html>""" % (
        css, PURGE60, tab(T.drop(columns=["n"])), tab(pairs), tab(XR.drop(columns=["n"])),
        info["rows"], info["dates"], info["first"], info["last"], ", ".join(SLOTS), info["n_blends"], int((pd.to_datetime(
            pd.read_parquet(os.path.join(ROOT, "ens_payout.parquet")).index) >= SPLIT).sum()),
        ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in info["label_agreement"].items()),
        tab(singles[cols]), tab(top_slot[cols]), tab(top[cols]), tab(vs))
    out = "results/reports/crypto/e202_tsfm_t60.html"
    open(out, "w").write(doc)
    print(T.to_string(index=False, float_format="%.4f"))
    print(pairs.to_string(index=False, float_format="%.4f"))
    print(XR.to_string(index=False, float_format="%.4f"))
    print(top_slot[cols].head(10).to_string(index=False, float_format="%.4f"))
    print(vs.to_string(index=False, float_format="%.4f"))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--ensembles", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check-schedule", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.check_schedule:
        df, _, _ = load(None)
        for i, tr, te, k in schedule60(df):
            print(i, k, "train %s..%s (%d)  predict %s..%s  gap %d days" % (
                str(tr[0])[:10], str(tr[-1])[:10], len(tr), str(te[0])[:10], str(te[-1])[:10],
                (pd.Timestamp(te[0]) - pd.Timestamp(tr[-1])).days))
        return
    if a.score:
        return score(a.workers)
    if a.export:
        return export()
    if a.ensembles:
        return ensembles(a.workers)
    if a.report:
        return report()
    if a.fit:
        for arm in (a.arms or ARMS):
            assert arm in ARMS, arm
            if os.path.exists(os.path.join(ROOT, "preds", arm + ".parquet")):
                print("skip", arm)
                continue
            df, emb, feats = load(None if arm == "B_ALL60" else arm[4:])
            fit_arm(arm, df, emb, feats, schedule60(df))


if __name__ == "__main__":
    main()
