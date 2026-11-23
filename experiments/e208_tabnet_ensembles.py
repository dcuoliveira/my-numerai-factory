#!/usr/bin/env python3
"""
E208 -- per-coin TabNet fitted on the 20- and 60-day labels, residualised, and rank-blended with the
CU_SA_IA_BO book (registered in experiments/EXPERIMENTS.md and src/analysis/experiment_db.py).

    python3 experiments/e208_tabnet_ensembles.py
    python3 -m src.analysis.html_pdf results/reports/crypto/e208_tabnet_t60/E208_report.html

TabNet members (seed-averaged predictions, routed R_SLOT into ridge B_ALL ranks as E199 scores them):
  TN_B20 / TN_F20   E199 fits on target_binned_return_20
  TN_B60 / TN_F60   E208 fits on target_binned_return_60 (purge 60)
  *_r               the same, residualised on the meta model lagged >= 5 days (E107, strength 0.5)
Pool A = the 4 residualised TabNets + IA_02, CURUPIRA_05_60, CU_SA_IA_BO_01, MAPINGUARI_01.
Pool B = the 4 raw TabNets + CU_SA_IA_BO_02, CU_SA_IA_BO_MA_01, CU_SA_IA_BO_MA_R_01.
Every 2..5-member subset, equal-weight mean of per-date percentile ranks, on the slots' row set
(300 coins x 470 dates). Corr = E131 scorer on the 20-day label; MMC against the forecasts' meta_model.
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e131_analysis as E  # noqa: E402
from src.analysis import slots  # noqa: E402
from src.prediction import meta_resid as MR  # noqa: E402
from src.prediction.metrics import summarize  # noqa: E402

OUT = "results/reports/crypto/e208_tabnet_t60"
E198P = "results/reports/crypto/e198_cs/preds/ridge"
TNDIR = {20: "results/reports/crypto/e199_tabnet_percoin/preds/tabnet", 60: os.path.join(OUT, "preds/tabnet")}
KEYS = "results/reports/crypto/e199_tabnet_percoin/keys.parquet"
META = "data/crypto/v2.0-cs/historical_meta_models.parquet"
T20 = "target_binned_return_20"
SEEDS = [0, 1, 2]
TN = ["TN_B20", "TN_F20", "TN_B60", "TN_F60"]
TNR = [t + "_r" for t in TN]
HEAD = ["CU_SA_IA_BO_02", "CU_SA_IA_BO_MA_01", "CU_SA_IA_BO_MA_R_01"]
UNDER = ["IA_02", "CURUPIRA_05_60", "CU_SA_IA_BO_01", "MAPINGUARI_01"]
POOLS = {"A": TNR + UNDER, "B": TN + HEAD}
REF = "CU_SA_IA_BO_MA_R_01"
SPLIT = pd.Timestamp("2026-02-20")


def _pq(path):
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    return p


def tabnet_routed(keys, name):
    """Seed-averaged routed percentile ranks of one TabNet arm on the E199 key universe."""
    arm, fit = name[:4], int(name[4:])
    b = _pq(os.path.join(E198P, "B_ALL_s0.parquet"))[["symbol", "date", "prediction"]].rename(columns={"prediction": "b0"})
    fb = _pq(os.path.join(E198P, "B_COV_s0.parquet"))[["symbol", "date", "prediction"]].rename(columns={"prediction": "fb"})
    k = keys[["symbol", "date"]].merge(b, on=["symbol", "date"], how="inner", validate="1:1")
    acc = np.zeros(len(k))
    for s in SEEDS:
        c = _pq(os.path.join(TNDIR[fit], "%s_s%d.parquet" % (arm, s)))[["symbol", "date", "prediction"]]
        c = c.merge(fb, on=["symbol", "date"], how="left", validate="1:1")
        c["cand"] = c["prediction"].fillna(c["fb"])            # < MIN_ROWS / diverged fits -> ridge B_COV
        kk = k.merge(c[["symbol", "date", "cand"]], on=["symbol", "date"], how="left", validate="1:1")
        x = np.empty(len(kk))
        for _, idx in kk.groupby("date").indices.items():
            xx = kk["b0"].iloc[idx].rank(pct=True).to_numpy()
            cand = kk["cand"].iloc[idx].to_numpy()
            i = np.flatnonzero(~np.isnan(cand))
            xx[i[np.argsort(cand[i], kind="stable")]] = np.sort(xx[i])
            x[idx] = xx
        acc += x
    k[name] = acc / len(SEEDS)
    k[name] = k.groupby("date")[name].rank(pct=True)
    return k[["symbol", "date", name]]


def residualise(P, cols):
    meta = _pq(META)[["date", "symbol", "prediction"]]
    out = {c + "_r": np.full(len(P), np.nan) for c in cols}
    for d, idx in P.groupby("date").indices.items():
        m, _ = MR.lagged_meta(meta, P["symbol"].iloc[idx].to_numpy(), d)
        for c in cols:
            r, _ = MR.residualise(P[c].iloc[idx].to_numpy(), m)
            out[c + "_r"][idx] = r
    for k, v in out.items():
        P[k] = v
    return P


def _score_date(args):
    d, h, arms = args
    rows = []
    for a in arms:
        hr = h.assign(prediction=h[a].to_numpy()).set_index("symbol")
        c = E.corr_one(hr)
        m = E.contrib_one(hr, "meta_model")
        rows.append((d, a, c, m, 0.1 * c + m if m == m else np.nan))
    return rows


def score(R, arms, workers=8):
    jobs = [(d, h[["symbol", T20, "meta_model"] + arms], arms) for d, h in R.groupby("date", sort=True)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_score_date, jobs, chunksize=4) for r in part]
    return pd.DataFrame(rows, columns=["date", "arm", "corr", "mmc", "payout"])


def stats(S, arms):
    rows = []
    for a in arms:
        g = S[S.arm == a].set_index("date").sort_index()
        r = {"arm": a}
        for k in ("corr", "mmc", "payout"):
            s = summarize(g[k])
            r.update({k: s["mean"], k[0] + ".sharpe": s["sharpe"], k[0] + ".maxdd": s["max_drawdown"]})
        p = g["payout"].dropna()
        r["calmar"] = r["payout"] / r["p.maxdd"]
        r["burn(d)"] = int((p < 0).sum())
        r["n(pay)"] = len(p)
        r["n(corr)"] = int(g["corr"].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)


def full_period(keys):
    """The four raw TabNets on E199's full universe and window (1560 dates), 60-day fit vs 20-day twin."""
    P = keys.copy()
    for n in TN:
        P = P.merge(tabnet_routed(keys, n), on=["symbol", "date"], how="left", validate="1:1")
    S = score(P, TN)
    T = stats(S, TN)
    D = []
    piv = {k: S.pivot(index="date", columns="arm", values=k) for k in ("corr", "mmc", "payout")}
    for a, b in (("TN_B60", "TN_B20"), ("TN_F60", "TN_F20"), ("TN_F20", "TN_B20"), ("TN_F60", "TN_B60")):
        r = {"arm": a, "vs": b}
        for k, X in piv.items():
            d = (X[a] - X[b]).dropna().to_numpy()
            m, _, t = E.newey_west(d)
            r.update({"d" + k: m, "t " + k: t, "n " + k: len(d)})
        D.append(r)
    return T, pd.DataFrame(D)


def slot_rowset(keys):
    P = None
    for n in HEAD + UNDER:
        m, r, _ = slots.slot_run(n)
        f = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (m, r), comment="#", parse_dates=["date"])
        keep = ["symbol", "date", "prediction"] + ([T20, "meta_model"] if P is None else [])
        f = f[keep].rename(columns={"prediction": n})
        P = f if P is None else P.merge(f, on=["symbol", "date"], how="inner", validate="1:1")
    for n in TN:
        P = P.merge(tabnet_routed(keys, n), on=["symbol", "date"], how="inner", validate="1:1")
    P = P.dropna(subset=[T20]).sort_values(["date", "symbol"]).reset_index(drop=True)
    for c in HEAD + UNDER + TN:
        P[c] = P.groupby("date")[c].rank(pct=True)
    P = residualise(P, TN)
    for c in TNR:
        P[c] = P.groupby("date")[c].rank(pct=True)
    return P


def blends(P):
    names = {}
    for pool, mem in POOLS.items():
        for k in (2, 3, 4, 5):
            for combo in itertools.combinations(mem, k):
                nm = "%s: %s" % (pool, " + ".join(combo))
                P[nm] = P[list(combo)].mean(axis=1)
                names[nm] = (pool, k)
    return P, names


def main():
    os.makedirs(OUT, exist_ok=True)
    keys = _pq(KEYS)
    T1, D1 = full_period(keys)
    P = slot_rowset(keys)
    P, names = blends(P)
    singles = HEAD + UNDER + TN + TNR
    arms = singles + list(names)
    S = score(P, arms)
    S.to_parquet(os.path.join(OUT, "scores.parquet"), index=False)
    T = stats(S, arms)
    T["kind"] = T["arm"].map(lambda a: "blend %s%d" % names[a] if a in names else
                             ("slot" if a in HEAD else ("member" if a in UNDER else "tabnet")))
    piv = S.pivot(index="date", columns="arm", values="payout")
    for a in arms:
        if a == REF:
            continue
        m, _, t = E.newey_west((piv[a] - piv[REF]).dropna().to_numpy())
        T.loc[T.arm == a, "dpay vs r_01"] = m
        T.loc[T.arm == a, "t"] = t
    h1, h2 = piv[piv.index < SPLIT], piv[piv.index >= SPLIT]
    T["pay H1"] = T["arm"].map(h1.mean())
    T["pay H2"] = T["arm"].map(h2.mean())
    T = T.sort_values("payout", ascending=False)
    T.to_csv(os.path.join(OUT, "all_arms.csv"), index=False)
    T1.to_csv(os.path.join(OUT, "full_period_singles.csv"), index=False)
    D1.to_csv(os.path.join(OUT, "full_period_deltas.csv"), index=False)
    # selection check: in each pool and size, pick the best on H1 and read it on H2
    sel = []
    for (pool, k) in sorted(set(names.values())):
        g = T[T.arm.map(lambda a: names.get(a) == (pool, k))]
        b = g.sort_values("pay H1", ascending=False).iloc[0]
        sel.append({"pool": pool, "size": k, "picked on H1": b.arm, "pay H1": b["pay H1"], "pay H2": b["pay H2"],
                    "%s pay H2" % REF: T.loc[T.arm == REF, "pay H2"].iloc[0],
                    "rank of pick on H2 (of %d)" % len(g): int((g["pay H2"] > b["pay H2"]).sum()) + 1})
    sel = pd.DataFrame(sel)
    sel.to_csv(os.path.join(OUT, "selection_check.csv"), index=False)
    report(T, T1, D1, sel, names, P)


def report(T, T1, D1, sel, names, P):
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = set(T[T.kind.str.startswith("blend")].groupby("kind").head(1).arm)
    for pool in POOLS:
        best |= set(T[T.kind.str.startswith("blend " + pool)].head(10).arm)
    main_t = T[T.arm.isin(HEAD) | T.arm.isin(best)].copy()
    cols = ["arm", "kind", "corr", "mmc", "payout", "p.sharpe", "p.maxdd", "calmar", "burn(d)", "c.sharpe", "c.maxdd",
            "dpay vs r_01", "t", "pay H1", "pay H2", "n(pay)"]
    sg = T[T.kind.isin(["slot", "member", "tabnet"])][cols]
    S = pd.read_parquet(os.path.join(OUT, "scores.parquet"))
    piv = S.pivot(index="date", columns="arm", values="payout")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    top = list(main_t.arm[:6]) + [a for a in HEAD if a not in list(main_t.arm[:6])]
    for a in top:
        ax.plot(piv.index, piv[a].cumsum(), lw=2 if a in HEAD else 1.1, ls="-" if a in HEAD else "--",
                label=a if len(a) < 70 else a[:67] + "...")
    ax.set_title("cumulative payout, slot row set"); ax.grid(alpha=.3); ax.legend(fontsize=6.5)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130)
    png = base64.b64encode(buf.getvalue()).decode()
    Q = P.groupby("date")[HEAD + UNDER + TN + TNR].corr().groupby(level=1).mean().loc[HEAD + UNDER + TN + TNR, HEAD + UNDER + TN + TNR]
    f4 = lambda x: "%.4f" % x  # noqa: E731
    d0, d1 = str(piv.index.min())[:10], str(piv.index.max())[:10]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>E208 TabNet ensembles</title>
<style>body{{font-family:sans-serif;margin:22px;font-size:11.5px}} table{{border-collapse:collapse;font-size:9.5px;margin:6px 0 16px}}
td,th{{padding:2px 5px;border-bottom:1px solid #ddd;text-align:right}} td:first-child{{text-align:left;max-width:420px}} th{{background:#f2f2f2}}</style></head><body>
<h1>E208: per-coin TabNet on the 20/60-day label, residualised, blended with the CU_SA_IA_BO book</h1>
<p>TabNet members are E199's per-coin TabNets (TN_B = 22 v2.0 features, TN_F = + 15 CS columns), fitted on the 20-day label (E199) or the 60-day label (E208, purge 60). Each has its 3 seeds averaged and is routed into ridge B_ALL's rank slots. <i>_r</i> = residualised on the meta model lagged &ge; 5 days (E107, strength 0.5). Every score is on the payable 20-day label. Payout = 0.1&middot;corr + mmc per date.</p>
<h2>1. 60-day fit evaluated on the 20-day label, full backtest</h2>
<p>E199 universe and window (1560 dates, MMC/payout on the 588 with a meta model), seed-averaged predictions.</p>
{T1.to_html(index=False, float_format=f4, border=0)}
<p>Paired Newey-West, lag 20:</p>{D1.to_html(index=False, float_format=f4, border=0)}
<h2>2. Slots + highest-payout ensembles, sorted by payout</h2>
<p>Row set: the slots' backtest universe (300 coins) &times; {len(piv)} dates, {d0} to {d1}. Pool A (210 subsets) = 4 residualised TabNets + IA_02, CURUPIRA_05_60, CU_SA_IA_BO_01, MAPINGUARI_01. Pool B (112 subsets) = 4 raw TabNets + the three headline slots. Ensembles are equal-weight means of per-date percentile ranks over every 2-, 3-, 4- and 5-member subset. The table shows the top 10 of each pool plus the best of each size. <i>dpay vs r_01</i> / <i>t</i> = paired Newey-West against CU_SA_IA_BO_MA_R_01. <b>Picking the maximum over 322 subsets inflates the top rows</b>, and the slots' own backtests were built on this history. See the selection check in section 3.</p>
{main_t[cols].to_html(index=False, float_format=f4, border=0)}
<img src="data:image/png;base64,{png}" style="width:100%">
<h2>3. Selection check: pick on dates &lt; {SPLIT.date()}, read on the rest</h2>
{sel.to_html(index=False, float_format=f4, border=0)}
<h2>4. Every member on the slot row set</h2>
{sg.to_html(index=False, float_format=f4, border=0)}
<h2>5. Mean per-date Spearman between members</h2>
{Q.to_html(float_format=lambda x: "%.2f" % x, border=0)}
</body></html>"""
    open(os.path.join(OUT, "E208_report.html"), "w").write(html)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(T1.round(4).to_string(index=False)); print(D1.round(4).to_string(index=False))
    print(main_t[cols].round(4).to_string(index=False)); print(sel.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
