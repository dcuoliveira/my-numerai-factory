#!/usr/bin/env python3
"""
E114 report -- 60-day twins of the E112 picks, then 3/4-member ensembles with the book.

    python3 experiments/e114_ensembles.py

Part A: each e114_<m>_*60 arm in-target (60-day label) against its E110 allmom60 twin,
paired NW lag 20, with calmar.

Part B, per model: six items -- the model's 20-day forecast plain and residualised, its
60-day forecast plain and residualised (E107 transform: s = 0.5, meta model >= 5 days old,
imported via E111 so the arithmetic is identical), CU_SA_IA_BO_02 and CU_SA_IA_BO_MA_R_01
(already residualised) -- and every 3- and 4-member equal-weight rank average of them
(20 + 15 = 35). Everything is scored on the PAYABLE 20-day label against the real meta
model over ONE common window shared by every series of every model, payout formed per
period, calmar = payout mean / max drawdown of the cumulative payout. Each combo is paired
against CU_SA_IA_BO_02, CU_SA_IA_BO_MA_R_01 and its best member. Gate: each slot alone
reproduces its registry backtest payout on its own window.
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e111_allmom_resid_ensembles_crypto as E  # noqa: E402
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.prediction.metrics import per_period_bmc, per_period_corr, summarize  # noqa: E402

T20, T60 = E.T20, E.T60
PICKS = [  # label, model tree, 20-day run (E112), 60-day run (E114), in-target 60-day twin (E110)
    ("lgb+tenc", "lgb_ktz", "e112_lgb_ktz_tenc", "e114_lgb_ktz_tenc60", "e110_lgb_ktz_allmom60"),
    ("cat+tenc", "catboost_ktz", "e112_catboost_ktz_tenc", "e114_catboost_ktz_tenc60", "e110_catboost_ktz_allmom60"),
    ("mlp+roll", "mlp_ktz", "e112_mlp_ktz_roll", "e114_mlp_ktz_roll60", "e110_mlp_ktz_allmom60"),
]
SLOTS = {"BO_02": 0.106350, "MA_R_01": 0.118694}   # registry backtest payouts (470 periods)
OUT = os.path.join("results", "reports", "crypto", "e114")


def summ(t, days):
    t = t.reindex(days)
    pay = 0.1 * t["corr"] + t["mmc"]
    s = summarize(pay)
    dd = s["max_drawdown"]
    return dict(payout=s["mean"], calmar=s["mean"] / dd if dd else np.nan, CORR=t["corr"].mean(),
                MMC=t["mmc"].mean(), dd=dd, burn=int((pay < 0).sum()), n=int(pay.notna().sum()))


def dpay(a, b, days):
    pa = (0.1 * a["corr"] + a["mmc"]).reindex(days)
    pb = (0.1 * b["corr"] + b["mmc"]).reindex(days)
    m, _, t = newey_west((pa - pb).dropna(), lag=20)
    return m, t


# ---- Part A: in-target 60-day arms vs their allmom60 twins ------------------- #
def part_a():
    print("PART A -- 60-day fits, scored IN-TARGET (60-day label), vs the E110 allmom60 twin")
    print("  %-10s %-26s %8s %8s %8s %7s %5s | %9s %6s" % (
        "pick", "run", "CORR", "MMC", "payout", "calmar", "n", "d(payout)", "t"))
    for lbl, m, _, r60, twin in PICKS:
        fs = {}
        for r in (r60, twin):
            p = os.path.join("results", "per_period", "crypto", m, r + ".csv")
            fs[r] = pd.read_csv(p, parse_dates=["date"]).set_index("date")[["corr", "mmc"]].dropna() if os.path.exists(p) else None
        if fs[r60] is None or fs[twin] is None:
            print("  %-10s missing %s" % (lbl, [r for r in fs if fs[r] is None]))
            continue
        days = sorted(set(fs[r60].index) & set(fs[twin].index))
        for r in (twin, r60):
            s = summ(fs[r], days)
            d = dpay(fs[r60], fs[twin], days) if r == r60 else None
            print("  %-10s %-26s %8.5f %8.5f %8.5f %7.3f %5d | %s" % (
                lbl, r, s["CORR"], s["MMC"], s["payout"], s["calmar"], s["n"],
                "%+9.5f %+6.2f" % d if d else "%9s %6s" % ("-", "-")))


# ---- Part B: items and ensembles --------------------------------------------- #
def load_items():
    book = E.load_and_residualise({"slug": "book", "group": "book", "model": "lgb",
                                   "run": "e47_cusaiabo02", "native": T20})
    book = book[book[T20].notna()]
    lab = book[["symbol", "date", T20, "meta_model"]]
    mar = pd.read_csv("results/forecast/crypto/meta_resid/e107px_cu_sa_cu_sa.csv", parse_dates=["date"])
    mar = mar[mar["meta_model"].notna() & mar[T20].notna()].copy()
    slots = {"BO_02": book[["symbol", "date", "pred_s0"]].rename(columns={"pred_s0": "r"}),
             "MA_R_01": mar.assign(r=mar.groupby("date")["prediction"].rank(pct=True))[["symbol", "date", "r"]]}
    picks = {}
    for lbl, m, r20, r60, _ in PICKS:
        it = {}
        for tag, run, native in (("20", r20, T20), ("60", r60, T60)):
            f = E.load_and_residualise({"slug": run, "group": "x", "model": m, "run": run, "native": native})
            it["plain" + tag] = f[["symbol", "date", "pred_s0"]].rename(columns={"pred_s0": "r"})
            it["resid" + tag] = f[["symbol", "date", "pred_s5"]].rename(columns={"pred_s5": "r"})
        picks[lbl] = it
    return lab, slots, picks


LAB, SLOT_FR, PICK_FR = None, None, None


def score_combo(spec):
    pick, members = spec
    items = dict(PICK_FR[pick], **SLOT_FR)
    g = LAB
    for k in members:
        g = g.merge(items[k].rename(columns={"r": k}), on=["symbol", "date"], how="inner")
    g = g.copy()
    g["e"] = g[list(members)].mean(axis=1)
    g["e"] = g.groupby("date")["e"].rank(pct=True)
    return spec, pd.DataFrame({"corr": per_period_corr(g, "e", T20, "date"),
                               "mmc": per_period_bmc(g, "e", "meta_model", T20, "date")})


def main():
    global LAB, SLOT_FR, PICK_FR
    print("E114 -- 60-day twins of the E112 picks + 3/4-member ensembles with the book")
    print("=" * 100)
    part_a()
    LAB, SLOT_FR, PICK_FR = load_items()
    names = ["plain20", "resid20", "plain60", "resid60", "BO_02", "MA_R_01"]
    specs = []
    for lbl, *_ in PICKS:
        specs += [(lbl, (k,)) for k in names]
        specs += [(lbl, c) for n in (3, 4) for c in itertools.combinations(names, n)]
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = dict(ex.map(score_combo, specs))

    print("\nGATE (each slot alone, own window, payable 20d):")
    for s, reg in SLOTS.items():
        t = res[(PICKS[0][0], (s,))]
        p = (0.1 * t["corr"] + t["mmc"]).dropna()
        print("  %-8s n=%d payout %.6f  registry %.6f  diff %+.1e" % (s, len(p), p.mean(), reg, p.mean() - reg))

    days = None
    for t in res.values():
        have = set(t.dropna().index)
        days = have if days is None else days & have
    days = sorted(days)
    print("\nPART B -- every series on ONE common window: %d days, %s -> %s (payable 20-day label)" % (
        len(days), days[0].date(), days[-1].date()))

    rows = []
    for (pick, members), t in res.items():
        s = summ(t, days)
        r = dict(pick=pick, k=len(members), members="+".join(members), **s)
        if len(members) > 1:
            singles = {m: res[(pick, (m,))] for m in members}
            best = max(singles, key=lambda m: summ(singles[m], days)["payout"])
            r["best_member"] = best
            r["d_best"], r["t_best"] = dpay(t, singles[best], days)
            r["d_bo02"], r["t_bo02"] = dpay(t, res[(pick, ("BO_02",))], days)
            r["d_mar"], r["t_mar"] = dpay(t, res[(pick, ("MA_R_01",))], days)
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "ensembles.csv"), index=False)

    ref = {s: summ(res[(PICKS[0][0], (s,))], days) for s in SLOTS}
    print("  slots alone:  BO_02 payout %.5f calmar %.3f | MA_R_01 payout %.5f calmar %.3f" % (
        ref["BO_02"]["payout"], ref["BO_02"]["calmar"], ref["MA_R_01"]["payout"], ref["MA_R_01"]["calmar"]))
    hdr = "  %-36s %8s %7s %8s %7s %5s %5s | %9s %6s | %9s %6s | %9s %6s" % (
        "members", "payout", "calmar", "CORR", "MMC", "dd", "burn", "d(BO_02)", "t", "d(MA_R)", "t", "d(best)", "t")
    for lbl, *_ in PICKS:
        d = df[df["pick"] == lbl]
        print("\n=== %s ===" % lbl)
        print("  singles:")
        for _, r in d[d["k"] == 1].iterrows():
            print("  %-36s %8.5f %7.3f %8.5f %7.4f %5.2f %5d" % (
                r["members"], r["payout"], r["calmar"], r["CORR"], r["MMC"], r["dd"], r["burn"]))
        for key in ("payout", "calmar"):
            print("  top 8 ensembles by %s:" % key)
            print(hdr)
            for _, r in d[d["k"] > 1].sort_values(key, ascending=False).head(8).iterrows():
                print("  %-36s %8.5f %7.3f %8.5f %7.4f %5.2f %5d | %+9.5f %+6.2f | %+9.5f %+6.2f | %+9.5f %+6.2f" % (
                    r["members"], r["payout"], r["calmar"], r["CORR"], r["MMC"], r["dd"], r["burn"],
                    r["d_bo02"], r["t_bo02"], r["d_mar"], r["t_mar"], r["d_best"], r["t_best"]))
        e = d[d["k"] > 1]
        print("  of 35: payout > MA_R_01 alone: %d (t>=2: %d) | calmar > BO_02 alone: %d | calmar > MA_R_01 alone: %d" % (
            (e["d_mar"] > 0).sum(), (e["t_mar"] >= 2).sum(),
            (e["calmar"] > ref["BO_02"]["calmar"]).sum(), (e["calmar"] > ref["MA_R_01"]["calmar"]).sum()))
    print("\n105 combos screened; equal weights only; calmar comparable here (one window, equal n).")
    print("wrote %s/ensembles.csv" % OUT)


if __name__ == "__main__":
    main()
