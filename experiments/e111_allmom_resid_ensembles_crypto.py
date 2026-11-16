#!/usr/bin/env python3
"""
E111 -- meta residualisation, then ensembles, of the all-momentum models (20d and 60d).

    python3 experiments/e111_allmom_resid_ensembles_crypto.py            # full run (needs E108 + E110)
    python3 experiments/e111_allmom_resid_ensembles_crypto.py --only-20  # 20-day sources only (E110 not needed)
    NO_MAIL=1 ...                                                         # skip the email

WHAT IT ASKS
  1. Does E106/E107's meta residualisation lift the all-momentum models, at both
     horizons -- and does allmom still trail / beat raw once both are residualised?
  2. Does any 2/3/4-model ensemble of the residualised allmom models (20d and 60d,
     all scored on the payable 20-day label) beat its best single member, and does
     anything beat -- or add to -- the residualised book (CU_SA_IA_BO_02)?

WHAT IT RUNS (no training)
  Residualisation is E107's deployable transform, imported from
  experiments/e107_residualise_everything_crypto.py so the arithmetic is identical:
  per day, p = gauss(rank(prediction)), m = gauss(rank(meta model >= 5 days old)),
  beta = p.m / m.m, r = p - 0.5 * beta * m, submitted = rank(r). Scored against the
  SAME-day meta model (what Numerai pays on). s = 0 is the forecast unchanged.
  GATE: the s = 0 score must equal the repo scorer on the stored forecast (1e-9).

  Sources (held-out forecasts already on disk):
    allmom20  e108_<m>_allmom      crypto/v2.0-allmom-fillmask, fit+scored 20d
    raw20     e72_<m>_base         crypto/v2.0, fit+scored 20d
    allmom60  e110_<m>_allmom60    allmom, fit 60d -- scored 60d AND payable 20d
    raw60     e110_<m>_raw60       raw, fit 60d -- scored 60d AND payable 20d
    book      lgb/e47_cusaiabo02   CU_SA_IA_BO_02, 20d (the reference)
  A 60-day forecast file carries only the 60-day label; the 20-day label is joined
  from crypto/v2.0 on (symbol, date) to score it on what Numerai pays.

  Ensembles: pool = the 10 allmom models (5 x allmom20, 5 x allmom60-on-20d), each
  RESIDUALISED; every 2/3/4-member equal-weight average of the members' per-day
  rank, re-ranked, scored on the 20-day label on the pool's common days. Each is
  paired (Newey-West lag 20) vs its best single member and vs the residualised book.
  Then book + best ensemble at 10/20%: does it add to the residualised book?

  Every comparison is on a common window and paired; payout and calmar are formed
  per period (calmar comparable only at equal n -- printed).

ARTEFACTS
  results/{forecast,per_period,metrics}/crypto/meta_resid/e111_<slug>_s050
  results/reports/crypto/e111/{residualise.csv,ensembles.csv,book_blend.csv}
  logs/e111_report.txt (+ email)
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from e107_residualise_everything_crypto import gauss_rank, lagged_meta  # noqa: E402
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.prediction.metrics import (per_period_bmc, per_period_corr,  # noqa: E402
                                    per_period_table, summarize)

T20, T60 = "target_binned_return_20", "target_binned_return_60"
S, LAG = 0.5, 5
TREE = "meta_resid"
MODELS = ["ridge", "lgb_ktz", "catboost_ktz", "rf_ktz", "mlp_ktz"]
SHORT = {"ridge": "rdg", "lgb_ktz": "lgb", "catboost_ktz": "cat", "rf_ktz": "rf", "mlp_ktz": "mlp"}
OUT = os.path.join("results", "reports", "crypto", "e111")


def sources(only20):
    out = []
    for m in MODELS:
        out.append({"slug": "allmom20_%s" % SHORT[m], "group": "allmom20", "model": m,
                    "run": "e108_%s_allmom" % m, "native": T20})
        out.append({"slug": "raw20_%s" % SHORT[m], "group": "raw20", "model": m,
                    "run": "e72_%s_base" % m, "native": T20})
        if not only20:
            out.append({"slug": "allmom60_%s" % SHORT[m], "group": "allmom60", "model": m,
                        "run": "e110_%s_allmom60" % m, "native": T60})
            out.append({"slug": "raw60_%s" % SHORT[m], "group": "raw60", "model": m,
                        "run": "e110_%s_raw60" % m, "native": T60})
    out.append({"slug": "book", "group": "book", "model": "lgb", "run": "e47_cusaiabo02", "native": T20})
    return out


_T20 = None


def t20_frame():
    global _T20
    if _T20 is None:
        d = pd.read_parquet("data/crypto/v2.0/train.parquet", columns=["date", T20]).reset_index()
        d["date"] = pd.to_datetime(d["date"])
        _T20 = d[["symbol", "date", T20]]
    return _T20


def rank01(v):
    s = pd.Series(v)
    return ((s.rank(method="average") - 0.5) / s.count()).to_numpy()


def load_and_residualise(src):
    """Forecast -> rows with meta, both labels where available, and the s=0 / s=0.5 predictions."""
    path = os.path.join("results", "forecast", "crypto", src["model"], src["run"] + ".csv")
    f = pd.read_csv(path, parse_dates=["date"])
    if src["native"] != T20:
        f = f.merge(t20_frame(), on=["symbol", "date"], how="left")
    f = f[f["meta_model"].notna()].sort_values(["date", "symbol"]).reset_index(drop=True)
    m_raw, m_src = lagged_meta(f, LAG)
    s0, s5 = np.empty(len(f)), np.empty(len(f))
    for _, idx in f.groupby("date").indices.items():
        p = gauss_rank(f["prediction"].to_numpy()[idx])
        m = gauss_rank(m_raw[idx])
        if pd.isna(m_src[idx[0]]) or m @ m == 0:
            r = p
        else:
            r = p - S * ((p @ m) / (m @ m)) * m
        s0[idx], s5[idx] = rank01(p), rank01(r)
    f["pred_s0"], f["pred_s5"] = s0, s5
    return f


def score(f, col, target):
    g = f[f[target].notna()]
    return pd.DataFrame({"corr": per_period_corr(g, col, target, "date"),
                         "mmc": per_period_bmc(g, col, "meta_model", target, "date")})


def summ(tab, days):
    t = tab.loc[tab.index.isin(days)]
    pay = 0.1 * t["corr"] + t["mmc"]
    s, cs = summarize(pay), summarize(t["corr"])
    dd = s.get("max_drawdown") or 0.0
    return {"CORR": t["corr"].mean(), "MMC": t["mmc"].mean(), "payout": s["mean"],
            "c.sharpe": cs.get("sharpe") or 0.0, "calmar": s["mean"] / dd if dd else np.nan,
            "n": int(pay.notna().sum())}


def paired(a, b, days):
    pa = (0.1 * a["corr"] + a["mmc"]).reindex(days)
    pb = (0.1 * b["corr"] + b["mmc"]).reindex(days)
    d = (pa - pb).dropna()
    mean, se, t = newey_west(d, lag=20)
    return mean, t


def write_run(src, f, tab, target):
    name = "e111_%s_s050" % src["slug"]
    res = {"n_periods": int(tab["corr"].notna().sum()),
           "corr": summarize(tab["corr"]), "mmc": summarize(tab["mmc"]),
           "config": {"model": TREE, "tournament": "crypto", "study": "E111",
                      "source_run": "%s/%s" % (src["model"], src["run"]), "strength": S,
                      "meta_lag_days": LAG, "target": target,
                      "transform": "rank(gauss(rank(p)) - s * beta * gauss(rank(m_lag))), per day",
                      "assembled_by": "experiments/e111_allmom_resid_ensembles_crypto.py"}}
    for kind in ("forecast", "per_period", "metrics"):
        os.makedirs(os.path.join("results", kind, "crypto", TREE), exist_ok=True)
    base = os.path.join("results", "%s", "crypto", TREE, name)
    f.assign(prediction=f["pred_s5"])[["symbol", "date", target, "prediction", "reference_prediction",
                                       "meta_model"]].to_csv(base % "forecast" + ".csv", index=False)
    tab.to_csv(base % "per_period" + ".csv")
    with open(base % "metrics" + ".json", "w") as h:
        json.dump(res, h, indent=2, sort_keys=True, default=str)


# ---- ensemble worker state (fork-shared) ------------------------------------ #
_POOL = {}


def _ens_task(combo):
    base = _POOL["base"]
    v = sum(_POOL["preds"][k] for k in combo) / len(combo)
    g = base.assign(e=v)
    g["e"] = g.groupby("date")["e"].rank(method="average", pct=True)
    tab = pd.DataFrame({"corr": per_period_corr(g, "e", T20, "date"),
                        "mmc": per_period_bmc(g, "e", "meta_model", T20, "date")})
    return combo, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-20", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--sizes", default="2,3,4")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    srcs = sources(args.only_20)

    # ---- PART A: residualise every source ------------------------------------ #
    frames, tabs = {}, {}
    gate = []
    for s in srcs:
        f = load_and_residualise(s)
        frames[s["slug"]] = f
        targets = [s["native"]] + ([T20] if s["native"] != T20 else [])
        for tgt in targets:
            plain = per_period_table(f[f[tgt].notna()], "prediction", tgt, "date",
                                     reference_col="reference_prediction", meta_model_col="meta_model")
            t0s, t5s = score(f, "pred_s0", tgt), score(f, "pred_s5", tgt)
            gate.append(max(float((t0s["corr"] - plain["corr"]).abs().max()),
                            float((t0s["mmc"] - plain["mmc"]).abs().max())))
            tabs[(s["slug"], tgt, 0)] = t0s
            tabs[(s["slug"], tgt, 5)] = t5s
        write_run(s, f, tabs[(s["slug"], s["native"], 5)], s["native"])
        print("  residualised %-16s (%d rows)" % (s["slug"], len(f)), flush=True)
    gmax = max(gate)
    print("GATE s=0 vs repo scorer: max |diff| = %.2e  %s" % (gmax, "OK" if gmax < 1e-9 else "FAILED"))
    if gmax >= 1e-9:
        sys.exit("gate failed -- s=0 does not reproduce the scorer")

    def days_of(keys):
        d = None
        for k in keys:
            have = set(tabs[k].dropna().index)
            d = have if d is None else d & have
        return sorted(d)

    rowsA = []
    views = [("20d set (fit+score 20d)", ["allmom20", "raw20", "book"], T20)]
    if not args.only_20:
        views += [("60d set, scored IN-TARGET 60d", ["allmom60", "raw60"], T60),
                  ("60d set, scored on PAYABLE 20d", ["allmom60", "raw60"], T20)]
    textA = []
    for title, groups, tgt in views:
        ss = [s for s in srcs if s["group"] in groups]
        keys = [(s["slug"], tgt, k) for s in ss for k in (0, 5)]
        days = days_of(keys)
        textA.append("\n--- %s : %d common days ---" % (title, len(days)))
        textA.append("%-14s %8s %8s %8s %7s %7s | %8s %7s | %9s %6s" % (
            "source", "CORR", "MMC", "payout", "c.shrp", "calmar", "payout", "calmar", "d(resid)", "t"))
        textA.append("%-14s %s | %s | %s" % ("", "------------- before (s=0) ------------",
                                              "after s=0.5", "after-before"))
        for s in ss:
            b = summ(tabs[(s["slug"], tgt, 0)], days)
            a = summ(tabs[(s["slug"], tgt, 5)], days)
            dm, dt = paired(tabs[(s["slug"], tgt, 5)], tabs[(s["slug"], tgt, 0)], days)
            textA.append("%-14s %8.5f %8.5f %8.5f %7.3f %7.3f | %8.5f %7.3f | %+9.5f %+6.2f" % (
                s["slug"], b["CORR"], b["MMC"], b["payout"], b["c.sharpe"], b["calmar"],
                a["payout"], a["calmar"], dm, dt))
            rowsA.append(dict(view=title, source=s["slug"], target=tgt, n=b["n"],
                              **{"before_" + k: v for k, v in b.items() if k != "n"},
                              **{"after_" + k: v for k, v in a.items() if k != "n"},
                              resid_delta=dm, resid_t=dt))
        # allmom vs raw after residualisation, per model
        am = [g for g in groups if g.startswith("allmom")][0]
        rw = [g for g in groups if g.startswith("raw")][0]
        textA.append("  allmom vs raw, both residualised (paired):")
        for m in MODELS:
            dm, dt = paired(tabs[("%s_%s" % (am, SHORT[m]), tgt, 5)],
                            tabs[("%s_%s" % (rw, SHORT[m]), tgt, 5)], days)
            textA.append("    %-13s d(payout) %+.5f  t %+.2f" % (m, dm, dt))
    pd.DataFrame(rowsA).to_csv(os.path.join(OUT, "residualise.csv"), index=False)

    # ---- PART B: ensembles of residualised allmom models, payable 20d -------- #
    pool = [s["slug"] for s in srcs if s["group"] in ("allmom20", "allmom60")]
    keys = [(k, T20, 5) for k in pool] + [("book", T20, 5)]
    days = days_of(keys)
    base = None
    preds = {}
    for k in pool:
        f = frames[k][["symbol", "date", T20, "meta_model", "pred_s5"]]
        f = f[f["date"].isin(days) & f[T20].notna()]
        base = f[["symbol", "date"]] if base is None else base.merge(f[["symbol", "date"]], on=["symbol", "date"])
    fb = frames[pool[0]]
    base = base.merge(fb[["symbol", "date", T20, "meta_model"]], on=["symbol", "date"]).sort_values(
        ["date", "symbol"]).reset_index(drop=True)
    for k in pool:
        preds[k] = base[["symbol", "date"]].merge(
            frames[k][["symbol", "date", "pred_s5"]], on=["symbol", "date"], how="left")["pred_s5"].to_numpy()
    _POOL["base"], _POOL["preds"] = base, preds

    single = {k: score(base.assign(e=preds[k]), "e", T20) for k in pool}
    book5 = tabs[("book", T20, 5)]
    edays = sorted(set(days) & set(base["date"].unique()))
    best_k = max(pool, key=lambda k: summ(single[k], edays)["payout"])
    sizes = [int(x) for x in args.sizes.split(",")]
    combos = [c for n in sizes for c in itertools.combinations(pool, n)]
    print("ensembles: %d combos over %d residualised members, %d common days" % (
        len(combos), len(pool), len(edays)), flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs) as p:
        res = p.map(_ens_task, combos, chunksize=4)
    rowsB = []
    for combo, tab in res:
        a = summ(tab, edays)
        best_member = max(combo, key=lambda k: summ(single[k], edays)["payout"])
        dm_b, t_b = paired(tab, single[best_member], edays)
        dm_k, t_k = paired(tab, book5, edays)
        rowsB.append(dict(members="+".join(combo), size=len(combo), **a,
                          d_vs_best_member=dm_b, t_vs_best_member=t_b,
                          d_vs_book_resid=dm_k, t_vs_book_resid=t_k))
    ens = pd.DataFrame(rowsB).sort_values("payout", ascending=False)
    ens.to_csv(os.path.join(OUT, "ensembles.csv"), index=False)

    # ---- PART C: does the best ensemble ADD to the residualised book? -------- #
    top = ens.iloc[0]["members"].split("+")
    bookf = frames["book"][["symbol", "date", "pred_s5"]].rename(columns={"pred_s5": "b"})
    bb = base.merge(bookf, on=["symbol", "date"], how="inner")
    ev = sum(bb[["symbol", "date"]].merge(frames[k][["symbol", "date", "pred_s5"]],
                                          on=["symbol", "date"], how="left")["pred_s5"].to_numpy()
             for k in top) / len(top)
    bb["ens"] = pd.Series(ev).groupby(bb["date"].to_numpy()).rank(pct=True).to_numpy()
    rowsC = []
    for w in (0.1, 0.2, 0.3):
        bb["mix"] = (1 - w) * bb["b"] + w * bb["ens"]
        bb["mix"] = bb.groupby("date")["mix"].rank(pct=True)
        tab = pd.DataFrame({"corr": per_period_corr(bb, "mix", T20, "date"),
                            "mmc": per_period_bmc(bb, "mix", "meta_model", T20, "date")})
        a = summ(tab, edays)
        dm, dt = paired(tab, book5, edays)
        rowsC.append(dict(weight_ensemble=w, **a, d_vs_book_resid=dm, t_vs_book_resid=dt))
    pd.DataFrame(rowsC).to_csv(os.path.join(OUT, "book_blend.csv"), index=False)

    # ---- report ---------------------------------------------------------------- #
    bk = summ(book5, edays)
    L = []
    L.append("E111 -- meta residualisation (s=0.5, meta >= 5d old) + ensembles of the all-momentum models")
    L.append("finished in %dm. GATE s=0 == repo scorer: max |diff| %.1e (OK). payout+calmar per period;"
             " calmar comparable only at equal n." % ((time.time() - t0) / 60, gmax))
    L.append("=" * 72)
    L.append("PART A -- residualisation, before vs after, per source")
    L.extend(textA)
    L.append("")
    L.append("PART B -- ensembles of RESIDUALISED allmom models, scored on PAYABLE 20d (%d common days)" % len(edays))
    L.append("  residualised book (CU_SA_IA_BO_02): payout %.5f  calmar %.3f  CORR %.5f  MMC %.5f  n=%d"
             % (bk["payout"], bk["calmar"], bk["CORR"], bk["MMC"], bk["n"]))
    sb = summ(single[best_k], edays)
    L.append("  best single residualised member: %s  payout %.5f  calmar %.3f" % (best_k, sb["payout"], sb["calmar"]))
    nb = int(((ens["d_vs_best_member"] > 0) & (ens["t_vs_best_member"] > 2)).sum())
    nk = int((ens["d_vs_book_resid"] > 0).sum())
    nk2 = int(((ens["d_vs_book_resid"] > 0) & (ens["t_vs_book_resid"] > 2)).sum())
    L.append("  %d combos: beat their best member at t>2: %d | beat resid book: %d (t>2: %d)" % (len(ens), nb, nk, nk2))
    L.append("  %-44s %8s %7s %8s %8s | %9s %6s | %9s %6s" % (
        "top 15 by payout", "payout", "calmar", "CORR", "MMC", "d_best", "t", "d_book", "t"))
    for _, r in ens.head(15).iterrows():
        L.append("  %-44s %8.5f %7.3f %8.5f %8.5f | %+9.5f %+6.2f | %+9.5f %+6.2f" % (
            r["members"][:44], r["payout"], r["calmar"], r["CORR"], r["MMC"],
            r["d_vs_best_member"], r["t_vs_best_member"], r["d_vs_book_resid"], r["t_vs_book_resid"]))
    L.append("")
    L.append("PART C -- residualised book + best ensemble (%s)" % "+".join(top))
    for r in rowsC:
        L.append("  book %.0f%% + ens %.0f%%: payout %.5f calmar %.3f | d vs resid book %+.5f t %+.2f" % (
            100 * (1 - r["weight_ensemble"]), 100 * r["weight_ensemble"], r["payout"], r["calmar"],
            r["d_vs_book_resid"], r["t_vs_book_resid"]))
    text = "\n".join(L)
    open(os.path.join("logs", "e111_report.txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)

    if os.environ.get("NO_MAIL") != "1":
        from src.live.notify import send_text
        for _ in range(3):
            if send_text("[crypto] E111 finished -- allmom residualisation + ensembles", text):
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
