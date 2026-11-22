#!/usr/bin/env python3
"""
E204 -- fixed-weight rank blends of the three E191 LightGBM arms with three live slots
(registered in experiments/EXPERIMENTS.md and src/analysis/experiment_db.py before launch).

    python3 experiments/e204_blend_e191_slots.py            # build, score, table, tests, HTML
    python3 -m src.analysis.html_pdf results/reports/crypto/e204_blend_e191_slots/E204_report.html

Members, all scored on ONE row set (the slots' 300-coin universe x the 470 dates they share with E191):
  L_REL_N_TS_MM, L_REL_RAW, L_OWN_RAW  E191 lgb arms, routed exactly as E191 scored them (R_SLOT into B_ALL ranks)
  CU_SA_IA_BO_MA_R_01, CU_SA_IA_BO_02, CU_R_IA_MA_60_01   backtest forecasts from results/forecast/crypto/
Blends are per-date means of member percentile ranks with weights fixed in advance (no weight is fitted).
Scores: corr = E131 corr_one on target_binned_return_20, mmc = contrib_one against the frozen E189 meta model.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import e131_analysis as E  # noqa: E402
import e134_route_feature_sets as RT  # noqa: E402
from src.analysis import slots  # noqa: E402

ROOT = "results/reports/crypto/e204_blend_e191_slots"
E191 = "results/reports/crypto/e191_c2cross/preds"
L = ["L_REL_N_TS_MM", "L_REL_RAW", "L_OWN_RAW"]
SL = {"cu_sa_ia_bo_ma_r_01": "CU_SA_IA_BO_MA_R_01", "cu_sa_ia_bo_02": "CU_SA_IA_BO_02",
      "cu_r_ia_ma_60_01": "CU_R_IA_MA_60_01"}
BOOK = list(SL)
BLENDS = {
    "BOOK3": {s: 1 for s in BOOK},
    "ALL6": {m: 1 for m in BOOK + L},
    "BOOK3+L_REL_N_TS_MM": {**{s: 1 for s in BOOK}, "L_REL_N_TS_MM": 1},
    "BOOK3+L_REL_RAW": {**{s: 1 for s in BOOK}, "L_REL_RAW": 1},
    "BOOK3+L_OWN_RAW": {**{s: 1 for s in BOOK}, "L_OWN_RAW": 1},
    "BOOK3+L3 (50/50)": {**{s: 1 / 3 for s in BOOK}, **{m: 1 / 3 for m in L}},
}
for w in (0.1, 0.2, 0.3):                    # sweep reported in full, not a pick
    BLENDS["BOOK3 %.0f%% + L_REL_N_TS_MM %.0f%%" % (100 - 100 * w, 100 * w)] = {
        **{s: (1 - w) / 3 for s in BOOK}, "L_REL_N_TS_MM": w}


def members():
    b = pd.read_parquet(os.path.join(E191, "B_ALL.parquet"))
    P = b[["symbol", "date", "target_binned_return_20", "meta_model", "covered"]].copy()
    for a in L:
        c = pd.read_parquet(os.path.join(E191, a + ".parquet"))
        assert (c[["symbol", "date"]].to_numpy() == b[["symbol", "date"]].to_numpy()).all()
        x = np.empty(len(b))
        for d, idx in b.groupby("date").indices.items():
            bb = b["prediction"].iloc[idx].rank(pct=True).to_numpy()
            x[idx] = RT.slot(bb, c["prediction"].iloc[idx].to_numpy(), b["covered"].iloc[idx].to_numpy())
        P[a] = x
    for lab, n in SL.items():
        m, r, _ = slots.slot_run(n)
        f = pd.read_csv("results/forecast/crypto/%s/%s.csv" % (m, r), usecols=["symbol", "date", "prediction"],
                        parse_dates=["date"]).rename(columns={"prediction": lab})
        P = P.merge(f, on=["symbol", "date"], how="inner", validate="1:1")
    P = P.dropna(subset=["target_binned_return_20"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    return P


def score(P):
    cols = L + BOOK
    R = P[["symbol", "date", "target_binned_return_20", "meta_model"]].copy()
    rk = P.groupby("date")[cols].rank(pct=True)
    for c in cols:
        R[c] = rk[c]
    for name, w in BLENDS.items():
        R[name] = sum(v * rk[k] for k, v in w.items()) / sum(w.values())
    arms = cols + list(BLENDS)
    rows = []
    for d, h in R.groupby("date", sort=True):
        for a in arms:
            hr = h.assign(prediction=h[a].to_numpy()).set_index("symbol")
            c = E.corr_one(hr)
            mm = E.contrib_one(hr, "meta_model")
            rows.append(dict(date=d, arm=a, corr=c, mmc=mm, payout=0.1 * c + mm))
    return pd.DataFrame(rows), R, arms


def report(S, R, arms):
    from scipy.stats import norm
    P = S.pivot(index="date", columns="arm", values="payout")[arms]
    C = S.pivot(index="date", columns="arm", values="corr")[arms]
    M = S.pivot(index="date", columns="arm", values="mmc")[arms]
    rows = []
    for a in arms:
        p = P[a]
        cum = p.cumsum()
        rows.append({"arm": a, "kind": "blend" if a in BLENDS else ("slot" if a in BOOK else "E191"),
                     "corr": C[a].mean(), "mmc": M[a].mean(), "payout": p.mean(),
                     "sharpe.p": p.mean() / p.std(ddof=0), "maxdd.p": float((cum.cummax() - cum).max()),
                     "calmar": p.mean() / float((cum.cummax() - cum).max()), "burn(d)": int((p < 0).sum()), "n": len(p)})
    T = pd.DataFrame(rows).sort_values("payout", ascending=False)
    tests = []
    for a in BLENDS:
        for ref in ("BOOK3", "cu_sa_ia_bo_ma_r_01"):
            if a == ref:
                continue
            for met, X in (("payout", P), ("corr", C), ("mmc", M)):
                m, _, t = E.newey_west((X[a] - X[ref]).to_numpy())
                tests.append(dict(arm=a, vs=ref, metric=met, delta=m, t=t, p=float(2 * (1 - norm.cdf(abs(t))))))
    K = pd.DataFrame(tests)
    for _, g in K.groupby(["vs", "metric"]):
        o = g["p"].sort_values()
        K.loc[o.index, "holm"] = np.maximum.accumulate([min(1, p * (len(o) - i)) for i, p in enumerate(o)])
    os.makedirs(ROOT, exist_ok=True)
    T.to_csv(os.path.join(ROOT, "table.csv"), index=False)
    K.to_csv(os.path.join(ROOT, "tests.csv"), index=False)
    S.to_parquet(os.path.join(ROOT, "scores.parquet"), index=False)
    Q = R.groupby("date")[arms[:6]].corr().groupby(level=1).mean().loc[arms[:6], arms[:6]]
    plots(P, T, Q)
    fmt = lambda x: "%.4f" % x  # noqa: E731
    w = pd.DataFrame(BLENDS).T.fillna(0)
    w = w.div(w.sum(axis=1), axis=0)
    html = ["<html><head><meta charset='utf-8'><title>E204 blends E191 + slots</title><style>"
            "body{font-family:sans-serif;margin:24px;font-size:12px}table{border-collapse:collapse;margin:8px 0 18px}"
            "td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#eee}img{max-width:100%}</style></head><body>",
            "<h1>E204: rank blends of E191 LightGBM arms with live slots</h1>",
            "<p>All rows on one row set: the slots' backtest universe (300 coins per date) x %d dates %s to %s. "
            "Corr on target_binned_return_20, MMC against the frozen E189 meta model, payout = 0.1 corr + mmc per date. "
            "Blend weights fixed in advance (table below); the 10/20/30%% sweep is reported in full.</p>"
            % (len(P), str(P.index.min())[:10], str(P.index.max())[:10]),
            "<h2>Sorted by payout</h2>", T.to_html(index=False, float_format=fmt),
            "<img src='E204_cumpay_corr.png'>",
            "<h2>Paired Newey-West (lag 20) vs BOOK3 and vs cu_sa_ia_bo_ma_r_01; Holm within (reference, metric)</h2>",
            K.to_html(index=False, float_format=fmt),
            "<h2>Blend weights</h2>", w.to_html(float_format=lambda x: "%.3f" % x),
            "<h2>Mean per-date Spearman of member predictions</h2>", Q.to_html(float_format=lambda x: "%.2f" % x)]
    rd = os.path.join(ROOT, "reading.md")
    if os.path.exists(rd):
        html += ["<h2>Reading</h2><pre style='white-space:pre-wrap'>%s</pre>" % open(rd).read()]
    html.append("</body></html>")
    open(os.path.join(ROOT, "E204_report.html"), "w").write("\n".join(html))
    print(T.to_string(index=False, float_format=fmt))
    print(K.to_string(index=False, float_format=fmt))
    print(Q.round(2).to_string())


def plots(P, T, Q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["BOOK3", "ALL6", "BOOK3+L_REL_N_TS_MM", "BOOK3+L3 (50/50)", "cu_sa_ia_bo_ma_r_01", "cu_sa_ia_bo_02",
            "cu_r_ia_ma_60_01", "L_REL_N_TS_MM", "L_REL_RAW", "L_OWN_RAW"]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for a in T["arm"]:
        if a not in show:
            continue
        ls = "-" if a in BLENDS else (":" if a.startswith("cu_") else "--")
        ax[0].plot(P.index, P[a].cumsum(), ls, lw=2.2 if a in BLENDS else 1.4, label="%s (%.4f)" % (a, P[a].mean()))
    ax[0].set_title("E204 cumulative payout, %s to %s, n=%d" % (str(P.index[0])[:10], str(P.index[-1])[:10], len(P)))
    ax[0].legend(title="series (mean payout)", fontsize=8.5)
    ax[0].grid(alpha=.3)
    K = P[show].corr()
    im = ax[1].imshow(K, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(show)))
    ax[1].set_xticklabels(show, rotation=50, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(show)))
    ax[1].set_yticklabels(show, fontsize=8)
    for i in range(len(show)):
        for j in range(len(show)):
            ax[1].text(j, i, "%.2f" % K.iloc[i, j], ha="center", va="center", fontsize=7,
                       color="w" if abs(K.iloc[i, j]) > .6 else "k")
    ax[1].set_title("correlation of per-date payout")
    fig.colorbar(im, ax=ax[1], fraction=.046)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "E204_cumpay_corr.png"), dpi=120, bbox_inches="tight")


def main():
    P = members()
    print("rows", len(P), "dates", P["date"].nunique(), P["date"].min(), P["date"].max(), flush=True)
    S, R, arms = score(P)
    report(S, R, arms)


if __name__ == "__main__":
    main()
