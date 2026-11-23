"""E198+E199 full-backtest report: dev and test pooled, one table of full statistics.

    python3 experiments/e199_full_report.py
Reads results/reports/crypto/e199_tabnet_percoin/per_date.parquet; writes E199_full_report.html beside it
(then print with src.analysis.html_pdf). Seeds are averaged per date before any statistic.
"""
import base64, io, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.prediction.metrics import summarize
from src.analysis.ablation_report import newey_west

OUT = "results/reports/crypto/e199_tabnet_percoin"
NAME = {"ridge": "ridge", "lgb_ktz": "lgb", "mlp_ktz": "mlp", "tabnet": "tabnet (per-coin)"}
PAIRS = [("ridge", "F_COV", "ridge", "B_COV"), ("lgb_ktz", "F_COV", "lgb_ktz", "B_COV"),
         ("mlp_ktz", "F_COV", "mlp_ktz", "B_COV"), ("tabnet", "TN_F", "tabnet", "TN_B"),
         ("tabnet", "TN_B", "ridge", "B_COV"), ("tabnet", "TN_F", "ridge", "F_COV"),
         ("ridge", "B_COV", "ridge", "B_ALL")]

per = pd.read_parquet(os.path.join(OUT, "per_date.parquet"))
daily = per.groupby(["model", "arm", "date"])[["corr", "mmc", "payout", "covered"]].mean().reset_index()
ser = {(m, a): g.set_index("date").sort_index() for (m, a), g in daily.groupby(["model", "arm"])}

rows = []
for (m, a), g in ser.items():
    r = {"model": NAME[m], "arm": a, "seeds": per[(per.model == m) & (per.arm == a)].seed.nunique()}
    for k in ("corr", "mmc", "payout"):
        s = summarize(g[k])
        r.update({k + ".mean": s["mean"], k + ".std": s["std"], k + ".sharpe": s["sharpe"], k + ".maxdd": s["max_drawdown"]})
    p = g["payout"].dropna()
    r["calmar"] = r["payout.mean"] / r["payout.maxdd"]
    r["burn(d)"] = int((p < 0).sum()); r["burn%"] = (p < 0).mean()
    r["covered"] = g["covered"].mean(); r["n(corr)"] = int(g["corr"].notna().sum()); r["n(pay)"] = len(p)
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("payout.mean", ascending=False)

drows = []
for ma, a, mb, b in PAIRS:
    j = ser[(ma, a)].join(ser[(mb, b)], lsuffix="_a", rsuffix="_b", how="inner")
    r = {"arm": "%s %s" % (NAME[ma], a), "baseline": "%s %s" % (NAME[mb], b)}
    for k in ("corr", "mmc", "payout"):
        d = (j[k + "_a"] - j[k + "_b"]).dropna().to_numpy()
        mean, _, t = newey_west(d, lag=20)
        r["Δ" + k] = mean; r["t " + k] = t; r["n " + k] = len(d)
    drows.append(r)
dl = pd.DataFrame(drows)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for (m, a), g in ser.items():
    for i, k in enumerate(("corr", "payout")):
        s = g[k].dropna().cumsum()
        ax[i].plot(s.index, s.values, label="%s %s" % (NAME[m], a), lw=1.6 if m in ("ridge", "tabnet") else 0.9,
                   ls="-" if a in ("F_COV", "TN_F") else ("--" if a in ("B_COV", "TN_B") else ":"))
for i, k in enumerate(("cumulative CORR (1560 dates)", "cumulative payout (dates with a meta model)")):
    ax[i].set_title(k); ax[i].grid(alpha=.3)
ax[1].legend(fontsize=7, ncol=2); fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); png = base64.b64encode(buf.getvalue()).decode()

def fmt(df):
    return df.to_html(index=False, float_format=lambda x: "%.4f" % x, border=0)

lo, hi = daily.date.min().date(), daily.date.max().date()
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>E198/E199 full backtest</title>
<style>body{{font-family:sans-serif;margin:24px;font-size:12px}} table{{border-collapse:collapse;font-size:10.5px}}
td,th{{padding:3px 6px;border-bottom:1px solid #ddd;text-align:right}} th{{background:#f2f2f2}} h2{{margin-top:26px}}</style></head><body>
<h1>E198 / E199: CS features, full backtest</h1>
<p>Full walk-forward backtest, {lo} to {hi}. The 11 dev blocks and the test block are pooled into one series, with no train/test split. Every row is out of sample: block <i>i</i> is predicted by a fit on the 130 dates before it, after a purge. E198's ridge, lgb_ktz and mlp_ktz use fixed configs. The per-coin TabNet arms are E199's (batch 32). Seeds are averaged per date before any statistic is taken. CORR covers all 1560 dates; MMC and payout (0.1·CORR + MMC) only exist on the 588 dates that have a meta model.</p>
<p><b>Arms:</b> B_ALL = 22 v2.0 features, all rows. B_COV = the same 22 features on the rows where every CS column exists (~74%). F_COV = B_COV + 15 CS columns. TN_B and TN_F are the per-coin TabNet versions of B_COV and F_COV. Covered coins are re-ranked inside the rank slots they hold in B_ALL, so every arm is scored on the full universe.</p>
<h2>Full statistics, sorted by mean payout</h2>
<p>std is ddof=0; sharpe = mean/std; maxdd = largest fall of the cumulative series (positive, bigger is worse); calmar = payout.mean / payout.maxdd; burn = number of dates with negative payout.</p>
{fmt(tab)}
<h2>Paired deltas, Newey-West lag 20, full period</h2>
{fmt(dl)}
<h2>Cumulative series</h2><img src="data:image/png;base64,{png}" style="width:100%">
</body></html>"""
p = os.path.join(OUT, "E199_full_report.html"); open(p, "w").write(html)
pd.set_option("display.width", 250)
print(tab.round(4).to_string(index=False)); print(); print(dl.round(4).to_string(index=False)); print(p)
