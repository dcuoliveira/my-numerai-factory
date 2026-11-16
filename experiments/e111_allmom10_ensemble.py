#!/usr/bin/env python3
"""
E111 addendum -- one ensemble of ALL ten all-momentum models (both targets).

    python3 experiments/e111_allmom10_ensemble.py

Members: the 5 allmom models fit on the 20-day label (E108, e108_<m>_allmom) and the
5 fit on the 60-day label (E110, e110_<m>_allmom60). Equal-weight average of each
member's per-day prediction rank, re-ranked, scored on the PAYABLE 20-day label
(a 60-day forecast carries only the 60-day label; the 20-day one is joined from
crypto/v2.0). Residualised version = the finished blend put through E107's
deployable transform (s = 0.5, meta model >= 5 days old) -- what you would submit.

Context arms on the SAME common window: the 20-day half (5), the 60-day half (5),
every member alone, and the book (CU_SA_IA_BO_02), each plain and residualised.
All comparisons paired, Newey-West lag 20; payout and calmar formed per period.

Artefacts: run_ensemble writes lgb_ktz/e111_ens_allmom10 (plain, scored 20d);
the residualised blend is written to meta_resid/e111_ens_allmom10_s050.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e111_allmom_resid_ensembles_crypto as E  # noqa: E402
from e107_residualise_everything_crypto import gauss_rank, lagged_meta  # noqa: E402
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.prediction.metrics import per_period_bmc, per_period_corr, summarize  # noqa: E402

T20 = E.T20
MODELS = E.MODELS
SH = E.SHORT
M20 = [("%s/e108_%s_allmom" % (m, m), "allmom20_%s" % SH[m]) for m in MODELS]
M60 = [("%s/e110_%s_allmom60" % (m, m), "allmom60_%s" % SH[m]) for m in MODELS]

# ---- 1. materialise the plain 10-model blend with the repo's own tool ---------- #
members = [p for p, _ in M20] + [p for p, _ in M60]          # a 20d member FIRST: label from it
cmd = [sys.executable, "-m", "src.run_ensemble", "crypto", "lgb_ktz"] + members + [
    "--target", T20, "--run-name", "e111_ens_allmom10"]
print("run_ensemble:", " ".join(cmd[3:6]), "... 10 members", flush=True)
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


# ---- 2. member frames: prediction rank, meta, payable label -------------------- #
def member_frame(path):
    model, run = path.split("/")
    f = pd.read_csv(os.path.join("results", "forecast", "crypto", model, run + ".csv"), parse_dates=["date"])
    if T20 not in f.columns:
        f = f.merge(E.t20_frame(), on=["symbol", "date"], how="left")
    f = f[f["meta_model"].notna() & f[T20].notna()]
    f["r"] = f.groupby("date")["prediction"].rank(pct=True)
    return f[["symbol", "date", T20, "meta_model", "r"]]


frames = {slug: member_frame(p) for p, slug in M20 + M60}


def resid(f):
    """E107's deployable transform on a frame with a 'prediction' column."""
    f = f.sort_values(["date", "symbol"]).reset_index(drop=True)
    m_raw, m_src = lagged_meta(f, E.LAG)
    s0, s5 = np.empty(len(f)), np.empty(len(f))
    for _, idx in f.groupby("date").indices.items():
        p = gauss_rank(f["prediction"].to_numpy()[idx])
        m = gauss_rank(m_raw[idx])
        r = p if (pd.isna(m_src[idx[0]]) or m @ m == 0) else p - E.S * ((p @ m) / (m @ m)) * m
        s0[idx], s5[idx] = E.rank01(p), E.rank01(r)
    return f.assign(pred_s0=s0, pred_s5=s5)


def blend(slugs):
    base = None
    for k in slugs:
        f = frames[k][["symbol", "date", "r"]].rename(columns={"r": k})
        base = f if base is None else base.merge(f, on=["symbol", "date"])
    lab = frames[slugs[0]][["symbol", "date", T20, "meta_model"]]
    base = base.merge(lab, on=["symbol", "date"])
    base["prediction"] = base[slugs].mean(axis=1)
    return resid(base[["symbol", "date", T20, "meta_model", "prediction"]])


def score(f, col):
    return pd.DataFrame({"corr": per_period_corr(f, col, T20, "date"),
                         "mmc": per_period_bmc(f, col, "meta_model", T20, "date")})


arms = {}
all10 = blend([s for _, s in M20 + M60])
arms["ALL 10 (20d+60d)"] = all10
arms["20d half (5)"] = blend([s for _, s in M20])
arms["60d half (5)"] = blend([s for _, s in M60])
for _, s in M20 + M60:
    arms[s] = resid(frames[s].rename(columns={"r": "prediction"}))
book = E.load_and_residualise({"slug": "book", "group": "book", "model": "lgb",
                               "run": "e47_cusaiabo02", "native": T20})
arms["book (CU_SA_IA_BO_02)"] = book[book[T20].notna()]

tabs = {}
for name, f in arms.items():
    tabs[(name, 0)] = score(f, "pred_s0")
    tabs[(name, 5)] = score(f, "pred_s5")

days = None
for t in tabs.values():
    have = set(t.dropna().index)
    days = have if days is None else days & have
days = sorted(days)


def summ(t):
    t = t.loc[t.index.isin(days)]
    pay = 0.1 * t["corr"] + t["mmc"]
    s, cs = summarize(pay), summarize(t["corr"])
    dd = s.get("max_drawdown") or 0.0
    return dict(payout=s["mean"], calmar=s["mean"] / dd if dd else np.nan, CORR=t["corr"].mean(),
                MMC=t["mmc"].mean(), ddpay=dd, burn=int((pay < 0).sum()), n=int(pay.notna().sum()))


def paired(a, b):
    pa = (0.1 * a["corr"] + a["mmc"]).reindex(days)
    pb = (0.1 * b["corr"] + b["mmc"]).reindex(days)
    m, se, t = newey_west((pa - pb).dropna(), lag=20)
    return m, t


rows = []
for name in arms:
    for s, tag in ((0, ""), (5, " ·r")):
        rows.append(dict(arm=name + tag, **summ(tabs[(name, s)])))
lb = pd.DataFrame(rows)

single = [s for _, s in M20 + M60]
best_plain = max(single, key=lambda k: summ(tabs[(k, 0)])["payout"])
best_res = max(single, key=lambda k: summ(tabs[(k, 5)])["payout"])
A = "ALL 10 (20d+60d)"
tests = [
    ("ALL 10 vs best single member (%s), plain" % best_plain, tabs[(A, 0)], tabs[(best_plain, 0)]),
    ("ALL 10 ·r vs best single member ·r (%s)" % best_res, tabs[(A, 5)], tabs[(best_res, 5)]),
    ("ALL 10 vs 20d half (5)", tabs[(A, 0)], tabs[("20d half (5)", 0)]),
    ("ALL 10 vs 60d half (5)", tabs[(A, 0)], tabs[("60d half (5)", 0)]),
    ("ALL 10 ·r vs ALL 10 (residualisation)", tabs[(A, 5)], tabs[(A, 0)]),
    ("ALL 10 vs book, both plain", tabs[(A, 0)], tabs[("book (CU_SA_IA_BO_02)", 0)]),
    ("ALL 10 ·r vs book ·r", tabs[(A, 5)], tabs[("book (CU_SA_IA_BO_02)", 5)]),
]

# residualised blend as a run (meta_resid tree), like E111's write_run
name = "e111_ens_allmom10_s050"
for kind in ("forecast", "per_period", "metrics"):
    os.makedirs(os.path.join("results", kind, "crypto", "meta_resid"), exist_ok=True)
all10.assign(prediction=all10["pred_s5"])[["symbol", "date", T20, "prediction", "meta_model"]].to_csv(
    os.path.join("results", "forecast", "crypto", "meta_resid", name + ".csv"), index=False)
tabs[(A, 5)].to_csv(os.path.join("results", "per_period", "crypto", "meta_resid", name + ".csv"))
with open(os.path.join("results", "metrics", "crypto", "meta_resid", name + ".json"), "w") as h:
    json.dump({"n_periods": int(tabs[(A, 5)]["corr"].notna().sum()),
               "corr": summarize(tabs[(A, 5)]["corr"]), "mmc": summarize(tabs[(A, 5)]["mmc"]),
               "config": {"model": "meta_resid", "study": "E111", "target": T20, "strength": E.S,
                          "meta_lag_days": E.LAG, "source_run": "lgb_ktz/e111_ens_allmom10",
                          "members": members, "assembled_by": "experiments/e111_allmom10_ensemble.py"}},
              h, indent=2, default=str)

out = os.path.join("results", "reports", "crypto", "e111")
os.makedirs(out, exist_ok=True)
lb.to_csv(os.path.join(out, "allmom10_ensemble.csv"), index=False)
print("COMMON_DAYS %d %s %s" % (len(days), days[0].date(), days[-1].date()))
for key in ("payout", "calmar"):
    print("\n=== by %s ===" % key)
    for _, r in lb.sort_values(key, ascending=False).iterrows():
        print("%-28s pay %.5f cal %.3f CORR %.4f MMC %.4f dd %.3f burn %d n %d" % (
            r["arm"], r["payout"], r["calmar"], r["CORR"], r["MMC"], r["ddpay"], r["burn"], r["n"]))
print("\n=== paired (NW lag 20) ===")
for label, a, b in tests:
    m, t = paired(a, b)
    print("%-52s d(payout) %+.5f  t %+.2f" % (label, m, t))
print("DONE")
