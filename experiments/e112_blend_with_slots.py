"""Can an E112 arm improve CU_SA_IA_BO_02 or CU_SA_IA_BO_MA_R_01 as a blend leg?

Prediction-level blends: per date, (1-w)*rank(slot) + w*rank(arm), re-ranked, scored on the
payable 20-day label against the real meta model (payout = 0.1*CORR + MMC per period).
Arms enter plain (s=0) or residualised (E107 transform, s=0.5, meta >= 5 days old) -- the
latter is how a leg with "resid": 0.5 enters CU_SA_IA_BO_MA_R_01 live. Everything on ONE
common window; each blend paired (NW lag 20) vs its slot alone.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/danielco/Downloads/my-numerai-agent/experiments")
import e111_allmom_resid_ensembles_crypto as E  # noqa: E402  (chdirs to the repo)
from src.analysis.ablation_report import newey_west  # noqa: E402
from src.prediction.metrics import per_period_bmc, per_period_corr, summarize  # noqa: E402

T20 = E.T20
ARMS = {"lgb+tenc": ("lgb_ktz", "e112_lgb_ktz_tenc"),
        "cat+tenc": ("catboost_ktz", "e112_catboost_ktz_tenc"),
        "mlp+roll": ("mlp_ktz", "e112_mlp_ktz_roll")}
W = [0.1, 0.2, 0.3, 0.4, 0.5]


def load_arm(model, run):
    f = E.load_and_residualise({"slug": run, "group": "x", "model": model, "run": run, "native": T20})
    return f[["symbol", "date", "pred_s0", "pred_s5"]]


book02 = E.load_and_residualise({"slug": "book", "group": "book", "model": "lgb",
                                 "run": "e47_cusaiabo02", "native": T20})
book02 = book02[book02[T20].notna()][["symbol", "date", T20, "meta_model", "pred_s0"]].rename(columns={"pred_s0": "slot"})
mar = pd.read_csv("results/forecast/crypto/meta_resid/e107px_cu_sa_cu_sa.csv", parse_dates=["date"])
mar = mar[mar["meta_model"].notna() & mar[T20].notna()].copy()
mar["slot"] = mar.groupby("date")["prediction"].rank(pct=True)
mar = mar[["symbol", "date", T20, "meta_model", "slot"]]
SLOTS = {"CU_SA_IA_BO_02": book02, "CU_SA_IA_BO_MA_R_01": mar}

arms = {k: load_arm(*v) for k, v in ARMS.items()}
pair = arms["lgb+tenc"].merge(arms["cat+tenc"], on=["symbol", "date"], suffixes=("_l", "_c"))
for s in ("pred_s0", "pred_s5"):
    pair[s] = (pair[s + "_l"] + pair[s + "_c"]) / 2
arms["lgb+cat tenc (50/50)"] = pair[["symbol", "date", "pred_s0", "pred_s5"]]


def score(f, col):
    return pd.DataFrame({"corr": per_period_corr(f, col, T20, "date"),
                         "mmc": per_period_bmc(f, col, "meta_model", T20, "date")})


def task(spec):
    slot, arm, ver, w = spec
    f = SLOTS[slot]
    if arm is None:
        return spec, score(f, "slot")
    a = arms[arm][["symbol", "date", ver]]
    g = f.merge(a, on=["symbol", "date"], how="inner")
    g["arm"] = g.groupby("date")[ver].rank(pct=True)
    g["b"] = (1 - w) * g["slot"] + w * g["arm"]
    g["b"] = g.groupby("date")["b"].rank(pct=True)
    return spec, score(g, "b")


specs = [(s, None, None, 0.0) for s in SLOTS]
specs += [(s, a, v, w) for s in SLOTS for a in arms for v in ("pred_s0", "pred_s5") for w in W]
with ProcessPoolExecutor(max_workers=16) as ex:
    res = dict(ex.map(task, specs))

# gate: each slot alone on its OWN window reproduces the registry backtest payout
for s, reg in (("CU_SA_IA_BO_02", 0.106350), ("CU_SA_IA_BO_MA_R_01", 0.118694)):
    t = res[(s, None, None, 0.0)]
    p = (0.1 * t["corr"] + t["mmc"]).dropna()
    print("GATE %-20s own window n=%d payout %.6f (registry %.6f)" % (s, len(p), p.mean(), reg))

days = None
for t in res.values():
    have = set(t.dropna().index)
    days = have if days is None else days & have
days = sorted(days)
print("COMMON %d days %s -> %s" % (len(days), days[0].date(), days[-1].date()))


def summ(t):
    t = t.reindex(days)
    pay = 0.1 * t["corr"] + t["mmc"]
    s = summarize(pay)
    dd = s["max_drawdown"]
    return dict(payout=s["mean"], calmar=s["mean"] / dd, CORR=t["corr"].mean(), MMC=t["mmc"].mean(),
                dd=dd, burn=int((pay < 0).sum())), pay


rows = []
for spec, t in res.items():
    slot, arm, ver, w = spec
    r, pay = summ(t)
    base_r, base_pay = summ(res[(slot, None, None, 0.0)])
    m, _, tt = newey_west((pay - base_pay).dropna(), lag=20) if arm else (0.0, 0.0, 0.0)
    mc, _, tc = newey_west((t.reindex(days)["corr"] - res[(slot, None, None, 0.0)].reindex(days)["corr"]).dropna(), lag=20) if arm else (0.0, 0.0, 0.0)
    rows.append(dict(slot=slot, arm=arm or "(slot alone)", leg={"pred_s0": "plain", "pred_s5": "resid", None: ""}[ver],
                     w=w, **r, d_pay=m, t_pay=tt, d_corr=mc, t_corr=tc, d_calmar=r["calmar"] - base_r["calmar"]))
df = pd.DataFrame(rows)
out = "results/reports/crypto/e112/blend_with_slots.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 250)
for s in SLOTS:
    d = df[df["slot"] == s].sort_values(["arm", "leg", "w"])
    print("\n=== %s ===" % s)
    print(d.drop(columns="slot").to_string(index=False, float_format=lambda x: "%.5f" % x))
print("WROTE", out)
