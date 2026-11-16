#!/usr/bin/env python3
"""
E112 combined report -- the four symbol-encoding supersets of allmom against the base.

    python3 experiments/e112_report.py            # prints; the runner tees + mails it

Per model: the base (e108_<m>_allmom) and the four variants on that model's common
window, CORR / MMC / payout / c.sharpe / calmar / burn / n, then each variant paired
against the base (Newey-West lag 20) on CORR, MMC and payout. Then one leaderboard of
every arm plus the book (CU_SA_IA_BO_02) on a single common window, by payout and by
calmar. Payout is formed per period; calmar = payout mean / max drawdown of the
cumulative payout, comparable only at equal n (printed). Arms not on disk are NAMED.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.prediction.metrics import summarize  # noqa: E402

MODELS = ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"]
SHORT = {"ridge": "rdg", "lgb_ktz": "lgb", "catboost_ktz": "cat", "rf_ktz": "rf", "mlp_ktz": "mlp"}
VARIANTS = [("allmom", "e108", "base 57f"), ("fenc", "e112", "+symmean 114f"),
            ("tenc", "e112", "+tenc 58f"), ("roll", "e112", "+rm 228f"), ("all", "e112", "+all 286f")]
BOOK = ("lgb", "e47_cusaiabo02")


def load(model, run):
    p = os.path.join("results", "per_period", "crypto", model, run + ".csv")
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p, parse_dates=["date"]).set_index("date")[["corr", "mmc"]].dropna()
    return f


def summ(f, days):
    f = f.loc[f.index.isin(days)]
    pay = 0.1 * f["corr"] + f["mmc"]
    s, cs = summarize(pay), summarize(f["corr"])
    dd = s.get("max_drawdown") or 0.0
    return dict(CORR=f["corr"].mean(), MMC=f["mmc"].mean(), payout=s["mean"],
                csharpe=cs.get("sharpe") or 0.0, calmar=s["mean"] / dd if dd else np.nan,
                ddpay=dd, burn=int((pay < 0).sum()), n=len(f))


def paired(a, b, days, col):
    if col == "payout":
        x = (0.1 * a["corr"] + a["mmc"]) - (0.1 * b["corr"] + b["mmc"])
    else:
        x = a[col] - b[col]
    m, _, t = newey_west(x.reindex(days).dropna(), lag=20)
    return m, t


def common(frames):
    days = None
    for f in frames:
        days = set(f.index) if days is None else days & set(f.index)
    return sorted(days)


arms, missing = {}, []
for m in MODELS:
    for v, pre, _ in VARIANTS:
        run = "%s_%s_%s" % (pre, m, v)
        f = load(m, run)
        if f is None:
            missing.append("%s/%s" % (m, run))
        else:
            arms[(m, v)] = f

print("E112 -- symbol encodings on top of crypto/v2.0-allmom-fillmask (57f)")
print("each variant differs from the base ONLY by the encoding; base = e108_<m>_allmom (reused)")
print("payout = 0.1*CORR + MMC per period; calmar comparable only at equal n; paired NW lag 20")
if missing:
    print("NOT ON DISK (left out): " + ", ".join(missing))
print("=" * 96)

for m in MODELS:
    have = [(v, lbl) for v, _, lbl in VARIANTS if (m, v) in arms]
    if ("allmom" not in [v for v, _ in have]) or len(have) < 2:
        print("\n%s: base or every variant missing -- skipped" % m)
        continue
    days = common([arms[(m, v)] for v, _ in have])
    print("\n=== %s  (%d common days, %s -> %s) ===" % (m, len(days), days[0].date(), days[-1].date()))
    print("  %-8s %-14s %8s %8s %8s %7s %7s %6s %5s %4s" % (
        "arm", "dataset", "CORR", "MMC", "payout", "c.shrp", "calmar", "dd", "burn", "n"))
    for v, lbl in have:
        r = summ(arms[(m, v)], days)
        print("  %-8s %-14s %8.5f %8.5f %8.5f %7.3f %7.3f %6.3f %5d %4d" % (
            v, lbl, r["CORR"], r["MMC"], r["payout"], r["csharpe"], r["calmar"], r["ddpay"], r["burn"], r["n"]))
    print("  -- paired vs allmom base --   d(CORR)      t |   d(MMC)      t | d(payout)      t")
    base = arms[(m, "allmom")]
    for v, lbl in have:
        if v == "allmom":
            continue
        cells = [paired(arms[(m, v)], base, days, c) for c in ("corr", "mmc", "payout")]
        flag = " *" if abs(cells[2][1]) >= 2 else ""
        print("  %-8s %-18s %+9.5f %+6.2f | %+8.5f %+6.2f | %+9.5f %+6.2f%s" % (
            v, lbl, cells[0][0], cells[0][1], cells[1][0], cells[1][1], cells[2][0], cells[2][1], flag))

# ---- one leaderboard: every arm + the book on ONE common window --------------- #
book = load(*BOOK)
pool = dict(((("%s_%s" % (SHORT[m], v)), f) for (m, v), f in arms.items()))
if book is not None:
    pool["BOOK cu_sa_ia_bo_02"] = book
else:
    print("\nbook per-period series missing -- leaderboard without it")
days = common(pool.values())
rows = pd.DataFrame([dict(arm=k, **summ(f, days)) for k, f in pool.items()])
print("\n" + "=" * 96)
print("LEADERBOARD -- %d arms on ONE common window: %d days, %s -> %s" % (
    len(rows), len(days), days[0].date(), days[-1].date()))
for key in ("payout", "calmar"):
    print("\n--- by %s ---" % key)
    print("  %-22s %8s %8s %8s %7s %7s %5s %4s" % ("arm", "payout", "calmar", "CORR", "MMC", "c.shrp", "burn", "n"))
    for _, r in rows.sort_values(key, ascending=False).iterrows():
        print("  %-22s %8.5f %8.3f %8.5f %7.4f %7.3f %5d %4d" % (
            r["arm"], r["payout"], r["calmar"], r["CORR"], r["MMC"], r["csharpe"], r["burn"], r["n"]))
if book is not None:
    best = rows[rows["arm"] != "BOOK cu_sa_ia_bo_02"].sort_values("payout", ascending=False).iloc[0]["arm"]
    m_, t_ = paired(pool[best], book, days, "payout")
    print("\nbest arm by payout (%s) vs book: d(payout) %+.5f t %+.2f" % (best, m_, t_))

out = os.path.join("results", "reports", "crypto", "e112")
os.makedirs(out, exist_ok=True)
rows.to_csv(os.path.join(out, "leaderboard.csv"), index=False)
print("\nwrote %s/leaderboard.csv" % out)
