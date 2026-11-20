#!/usr/bin/env python3
"""
E193-E197 -- E189 rerun with another frozen time-series foundation model's states (registered in
experiments/EXPERIMENTS.md and src/analysis/experiment_db.py before launch).

    python3 experiments/e193_tsfm_norm.py --tsfm toto --setup               # copy E189's B_ALL/B_COV, check states
    python3 experiments/e193_tsfm_norm.py --tsfm toto --fit --arms N_CS_Z RAW
    python3 experiments/e193_tsfm_norm.py --tsfm toto --score
    python3 experiments/e193_tsfm_norm.py --tables                          # one table per model + PDF

Everything is experiments/e189_c2norm.py; only the state cache (/scratch/danielco/tsfmstates/<model>,
written by src/data_mani/tsfm_states.py on E189's covered rows) and the output root change.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e189_c2norm as M  # noqa: E402

STUDIES = {"toto": ("E193", "Toto 2.0 (Toto-2.0-313m)"), "tabpfnts": ("E194", "TabPFN-TS (TabPFN-3 TS checkpoint)"),
           "tempopfn": ("E195", "TempoPFN 38M (univariate)"), "t0": ("E196", "t0 (t0-gamma)"),
           "timestral": ("E197", "Timestral-209M")}
E189_ROOT = "results/reports/crypto/e189_c2norm"
STATES = "/scratch/danielco/tsfmstates"
TABLES = "results/reports/crypto/e193_e197_tsfm_tables"


def configure(m):
    sid = STUDIES[m][0]
    M.STUDY = sid
    M.ROOT = "results/reports/crypto/%s_%snorm" % (sid.lower(), m)
    M.SCR = "/scratch/danielco/%s" % sid.lower()
    M.CACHE = os.path.join(STATES, m)
    M.load = load


def load():
    """e189_c2norm.load with the model's cache: any width, any number of part files."""
    df = pd.read_parquet(M.DATA).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ucid"]).reset_index(drop=True)
    keys = pd.read_parquet("/scratch/danielco/c2syn/rows.parquet")          # row ids index this order
    n0 = len(keys)
    assert (df.loc[:n0 - 1, ["symbol", "ucid"]].to_numpy() == keys[["symbol", "ucid"]].to_numpy()).all()
    df = df[df["date"] <= M.LAST].reset_index(drop=True)
    files = sorted(glob.glob(os.path.join(M.CACHE, "part_*.npz")))
    ids = np.concatenate([np.load(f)["cov_rows"] for f in files])
    reps = np.concatenate([np.load(f)["cov_reps"] for f in files])
    c2 = np.concatenate([np.load(f)["cov_rows"] for f in sorted(glob.glob("/scratch/danielco/c2syn/part_*.npz"))])
    assert np.array_equal(np.sort(ids), np.sort(c2[c2 < len(df)])), "state rows differ from E189's covered set"
    order = np.argsort(ids)
    ids, emb = ids[order], reps[order].astype(np.float32)
    assert np.isfinite(emb).all()
    df["covered"] = False
    df.loc[ids, "covered"] = True
    df["emb_row"] = -1
    df.loc[ids, "emb_row"] = np.arange(len(ids))
    feats = sorted(c for c in df if c.startswith("feature_"))
    g = df[M.GROUP_COL].to_numpy()
    df["group"] = np.where(g <= 0.25, 0, np.where(g == 0.5, 1, 2)).astype(np.int8)
    df[feats] = df[feats].fillna(0.5)
    meta = pd.read_parquet(M.META)[["symbol", "date", "prediction"]].rename(columns={"prediction": "meta_model"})
    meta["date"] = pd.to_datetime(meta["date"])
    df = df.merge(meta, on=["symbol", "date"], how="left", validate="1:1")
    return df, emb, feats


def setup():
    """Copy E189's baselines (same rows, schedule, model, coverage) and check the cache is complete."""
    os.makedirs(os.path.join(M.ROOT, "preds"), exist_ok=True)
    for a in ("B_ALL", "B_COV"):
        for ext in (".parquet", ".picks.json"):
            shutil.copy2(os.path.join(E189_ROOT, "preds", a + ext), os.path.join(M.ROOT, "preds", a + ext))
    df, emb, _ = load()
    e = pd.read_parquet(os.path.join(E189_ROOT, "preds", "RAW.parquet"), columns=["symbol", "date", "covered"])
    mine = df[df["date"].isin(e["date"].unique())][["symbol", "date", "covered"]]
    m = e.merge(mine, on=["symbol", "date"], suffixes=("", "_new"))
    assert len(m) == len(e) and (m["covered"] == m["covered_new"]).all()
    info = dict(rows=len(df), covered=int(df["covered"].sum()), width=int(emb.shape[1]),
                manifest=json.load(open(os.path.join(M.CACHE, "manifest.json"))))
    json.dump(info, open(os.path.join(M.ROOT, "setup.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------ tables
def table(m):
    import e131_analysis as E
    configure(m)
    S = pd.read_parquet(os.path.join(M.ROOT, "scores.parquet"))
    rows = []
    for a, x in S.groupby("arm"):
        d, t = x[x["kind"] == "dev"], x[x["kind"] == "test"]
        dp, tp = E.describe(d["payout"]), E.describe(t["payout"])
        rows.append(dict(arm=a, corr=d["corr"].mean(), mmc=d["mmc"].mean(), payout=dp["mean"], sharpe_p=dp["sharpe"],
                         maxdd_p=dp["max_dd"], n=dp["n"], test_payout=tp["mean"], test_sharpe_p=tp["sharpe"]))
    T = pd.DataFrame(rows).sort_values("payout", ascending=False).reset_index(drop=True)
    T.to_csv(os.path.join(M.ROOT, "table_by_payout.csv"), index=False)
    return T


def tables():
    import html
    done = [m for m in STUDIES if os.path.exists(os.path.join("results/reports/crypto/%s_%snorm" % (STUDIES[m][0].lower(), m),
                                                               "scores.parquet"))]
    os.makedirs(os.path.dirname(TABLES), exist_ok=True)
    e189 = pd.read_csv(os.path.join(E189_ROOT, "table_by_payout.csv")).set_index("arm")
    parts = []
    for m in done:
        T = table(m)
        sid, name = STUDIES[m]
        man = json.load(open(os.path.join(STATES, m, "manifest.json")))
        print("\n%s  %s  (dim %s)" % (sid, name, man["dim"]))
        print(T[["arm", "corr", "mmc", "payout", "sharpe_p", "maxdd_p", "test_payout"]].to_string(index=False, float_format="%.4f"))
        best = T[~T["arm"].isin(["B_ALL", "B_COV"])].iloc[0]
        f = {c: "{:.4f}".format for c in ("corr", "mmc", "payout", "test_payout")}
        f.update(sharpe_p="{:.3f}".format, maxdd_p="{:.3f}".format, test_sharpe_p="{:.3f}".format)
        parts.append("<h2>%s &mdash; %s</h2><p class=m>state width %s &middot; checkpoint <code>%s</code><br>%s<br>"
                     "best state arm <b>%s</b>: dev payout %.4f vs B_COV %.4f (E189 Chronos-2-Synth best state arm "
                     "%s %.4f)</p>%s" % (
                         sid, html.escape(name), man["dim"], html.escape(man["checkpoint"]), html.escape(man.get("notes", "")),
                         best["arm"], best["payout"], T.set_index("arm").loc["B_COV", "payout"],
                         e189.drop(["B_ALL", "B_COV"])["dev_payout"].idxmax(), e189.drop(["B_ALL", "B_COV"])["dev_payout"].max(),
                         T.drop(columns=["n"]).to_html(index=False, formatters=f, classes="t", border=0)))
    css = ("body{font-family:Helvetica,Arial,sans-serif;font-size:10px;margin:18px;color:#111}h1{font-size:16px}"
           "h2{font-size:13px;margin-top:22px;page-break-before:auto}.m{color:#444}table.t{border-collapse:collapse}"
           "table.t td,table.t th{padding:2px 7px;text-align:right;border-bottom:1px solid #ddd}"
           "table.t td:first-child,table.t th:first-child{text-align:left}code{font-size:9px}")
    doc = ("<!doctype html><html><head><meta charset=utf-8><title>E193-E197 TSFM states</title><style>%s</style></head><body>"
           "<h1>E193&ndash;E197: E189 rerun with five other time-series foundation models</h1><p class=m>Frozen hidden "
           "states on E189's 449,471 covered rows, E189's 35 normalisation arms + RAW, repo ridge, E131 blocks; routed "
           "(R_SLOT vs B_ALL) scores over the 11 dev blocks. corr / mmc / payout = per-period means, payout = 0.1&middot;corr + "
           "mmc per period; sharpe_p = payout mean/std; maxdd_p = max drawdown of cumulative payout; test_* = held-out block "
           "2026-02-20..2026-08-20 (not used for any choice). B_ALL / B_COV = v2.0 ridge on all / covered rows, copied from "
           "E189. Sorted by dev payout, descending. Not shown: %s.</p>%s</body></html>"
           % (css, ", ".join(m for m in STUDIES if m not in done) or "none", "".join(parts)))
    open(TABLES + ".html", "w").write(doc)
    print("\nwrote", TABLES + ".html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsfm", choices=sorted(STUDIES))
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.tables:
        return tables()
    configure(a.tsfm)
    if a.setup:
        return setup()
    if a.score:
        return M.score(a.workers)
    if a.fit:
        df, emb, feats = load()
        B = M.schedule(df)
        for arm in (a.arms or M.ARMS):
            assert arm in M.ARMS, arm
            M.fit_arm(arm, df, emb, feats, B)


if __name__ == "__main__":
    main()
