#!/usr/bin/env python3
"""
E132 analysis: score every arm on all keys, with dropped rows at the neutral prediction.

    python3 experiments/e132_analysis.py

Each arm's predictions become a per-date percentile; a key the arm has no prediction for takes
0.5. Scoring, statistics and blends are E131's (e131_analysis), on that completed vector. corr_kept
is the same prediction scored on the feed rows only. MMC is a proxy against the historical meta
model, from 2024-05-07 on, and is never official.
"""
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import e131_analysis as E  # noqa: E402

os.chdir(REPO)
import matplotlib.pyplot as plt  # noqa: E402

T20 = E.T20
ROOT = "results/reports/crypto/e132_feed_fillind"
PRED = os.path.join(ROOT, "preds")
FIG = os.path.join(ROOT, "figures")
MODELS = E.MODELS
ARMS = list(json.load(open(os.path.join(PRED, "arms.json")))) + ["B0N"]
DATASETS = {"L": "litmom-feed", "S": "momswap-feed", "C": "momswap-macro-carver-feed"}
COMPARISONS = [("B0F", "B0", "row handling: refit on feed rows + neutral"),
               ("B0N", "B0", "neutral only: B0 with dropped rows at 0.5"),
               ("B0F", "B0N", "refit on feed rows"),
               ("P_DROP", "B0", "survivorship placebo (look-ahead)"),
               ("P_HIST", "B0F", "causal tenure control")]
for k, n in DATASETS.items():
    COMPARISONS += [("%s_FULL" % k, "B0", n + ": full candidate vs v2.0"),
                    ("%s_FULL" % k, "B0F", n + ": features alone"),
                    ("%s_ADDED" % k, "B0F", n + ": added features alone")]
COMPARISONS += [("L_M1", "B0F", "litmom raw momentum"), ("L_M2", "L_M1", "momentum indicators"),
                ("L_B1", "B0F", "base indicators"), ("L_FULL", "L_M1", "all indicators"),
                ("L_M1", "L_no_ret", "returns family (LOFO)"), ("L_M1", "L_no_maxvol", "max/vol family (LOFO)"),
                ("L_M1", "L_no_trend", "trend family (LOFO)"),
                ("S_S0", "B0F", "drop the 8 base momentum proxies"), ("S_M1", "S_S0", "swap in literature momentum"),
                ("S_FULL", "S_M1", "momswap indicators"),
                ("C_MACRO", "S_S0", "macro block"), ("C_TF", "S_S0", "carver trend block"),
                ("C_RAW", "S_M1", "macro + trend on momswap"), ("C_FULL", "C_RAW", "carver indicators"),
                ("C_FULL", "C_no_mom", "momentum block (LOBO)"), ("C_FULL", "C_no_macro", "macro block (LOBO)"),
                ("C_FULL", "C_no_tf", "trend block (LOBO)")]
HEAD = ["B0", "B0F", "L_FULL", "S_FULL", "C_FULL"]
BLEND = {"L": "L_ADDED", "S": "S_ADDED", "C": "C_ADDED"}
COL = {"B0": "#52514e", "B0F": "#b0aea6", "L_FULL": "#2a78d6", "S_FULL": "#eb6834", "C_FULL": "#1baf7a"}
TEST0 = pd.Timestamp("2026-02-20")


def complete(path):
    """Per-date percentile of an arm's predictions over ALL keys, 0.5 where it has none."""
    k = pd.read_parquet(os.path.join(PRED, "keys.parquet"))
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"].isin(k["date"])]
    p["prediction"] = p.groupby("date")["prediction"].rank(pct=True)
    f = k.merge(p[["symbol", "date", "prediction"]], on=["symbol", "date"], how="left", validate="1:1")
    f["prediction"] = f["prediction"].fillna(0.5)
    return f


def apply_neutral(f):
    f = f.copy()
    f.loc[f["no_feed"] == 1, "prediction"] = 0.5
    return f


def score_job(job):
    model, arm, seed, path, b0, b0f = job
    f = complete(path)
    if arm == "B0N":
        f = apply_neutral(f)
    f["b0"] = complete(b0)["prediction"].values
    f["b0f"] = complete(b0f)["prediction"].values
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        kept = g[g["no_feed"] == 0]
        c = E.corr_one(g)
        m = E.contrib_one(g, "meta_model") if g["meta_model"].notna().sum() >= 10 else np.nan
        rows.append(dict(model=model, arm=arm, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0],
                         n_symbols=len(g), corr=c, mmc=m, payout=0.1 * c + m if m == m else np.nan,
                         contrib_b0=E.contrib_one(g, "b0") if arm != "B0" else 0.0,
                         predcorr_b0=g["prediction"].rank().corr(g["b0"].rank()),
                         predcorr_b0f=g["prediction"].rank().corr(g["b0f"].rank()),
                         corr_kept=E.corr_one(kept)))
    return pd.DataFrame(rows)


def blend_job(job):
    """(1-w)*B0 + w*ADDED, both completed percentiles, per date."""
    model, seed, key, b0, added = job
    f = complete(b0)
    f["added"] = complete(added)["prediction"].values
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        for w in E.W:
            g["bl"] = (1 - w) * g["prediction"] + w * g["added"]
            rows.append(dict(model=model, seed=seed, dataset=key, date=d, kind=g["kind"].iloc[0], block=int(g["block"].iloc[0]),
                             w=w, corr=E.corr_one(g, "bl")))
    return pd.DataFrame(rows)


def tuned(model, tag):
    """A tuned run's per-date CORR on all keys, dropped rows neutral, on the run's own dates."""
    p = "results/forecast/crypto/%s/%s.csv" % (model, tag)
    if not os.path.exists(p):
        return None
    k = pd.read_parquet(os.path.join(PRED, "keys.parquet"))
    d = pd.read_csv(p, usecols=["symbol", "date", "prediction"], parse_dates=["date"])
    d = d[d["date"].isin(k["date"])]
    tmp = os.path.join(PRED, "_t_%s.parquet" % tag)
    d.to_parquet(tmp, index=False)
    f = complete(tmp)
    os.remove(tmp)
    f = f[f["date"].isin(d["date"].unique())]
    return pd.Series({dd: E.corr_one(g.set_index("symbol")) for dd, g in f.groupby("date", sort=True)})


def tuned_v20(model):
    """E131's search on every key: the budget-matched B0."""
    p = "results/per_period/crypto/%s/e131_%s_tuneB0.csv" % (model, model)
    return pd.read_csv(p, parse_dates=["date"]).set_index("date")["corr"]


def main():
    os.makedirs(FIG, exist_ok=True)
    jobs, missing = [], []
    for m in MODELS:
        seeds = [0] if m == "ridge" else [0, 1, 2, 3, 4]
        for s in seeds:
            b0 = os.path.join(PRED, m, "B0_s%d.parquet" % s)
            b0f = os.path.join(PRED, m, "B0F_s%d.parquet" % s)
            for arm in ARMS:
                p = b0 if arm == "B0N" else os.path.join(PRED, m, "%s_s%d.parquet" % (arm, s))
                if os.path.exists(p):
                    jobs.append((m, arm, s, p, b0, b0f))
                else:
                    missing.append((m, arm, s))
    print("scoring %d files; missing %s" % (len(jobs), missing or "none"))
    bjobs = [(m, s, k, os.path.join(PRED, m, "B0_s%d.parquet" % s), os.path.join(PRED, m, "%s_s%d.parquet" % (a, s)))
             for m in MODELS for s in ([0] if m == "ridge" else range(5)) for k, a in BLEND.items()
             if os.path.exists(os.path.join(PRED, m, "%s_s%d.parquet" % (a, s)))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E132_WORKERS", "16"))) as ex:
        S = pd.concat(list(ex.map(score_job, jobs)), ignore_index=True)
        BL = pd.concat(list(ex.map(blend_job, bjobs)), ignore_index=True)
    S["date"] = pd.to_datetime(S["date"])
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)

    num = ["corr", "mmc", "payout", "contrib_b0", "predcorr_b0", "predcorr_b0f", "corr_kept"]
    A = S.groupby(["model", "arm", "date", "block", "kind"], as_index=False)[num].mean()
    masks = {"dev": A["kind"] == "dev", "test": A["kind"] == "test"}

    rows = []
    for (m, arm), g in A.groupby(["model", "arm"]):
        for sc, mk in masks.items():
            gg = g[mk.loc[g.index]]
            r = dict(model=m, arm=arm, scope=sc, **E.describe(gg["corr"]))
            r.update(corr_kept=gg["corr_kept"].mean(), contrib_b0=gg["contrib_b0"].mean(), predcorr_b0=gg["predcorr_b0"].mean(),
                     predcorr_b0f=gg["predcorr_b0f"].mean())
            pw = gg.dropna(subset=["payout"])
            if len(pw):
                dd = E.max_dd(pw["payout"].values)
                r.update(window_n=len(pw), window_corr=pw["corr"].mean(), mmc=pw["mmc"].mean(), payout=pw["payout"].mean(),
                         p_sharpe=pw["payout"].mean() / pw["payout"].std(ddof=0), dd_pay=dd,
                         calmar=pw["payout"].mean() / dd if dd > 0 else np.nan, burn=int((pw["payout"] < 0).sum()))
            rows.append(r)
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)

    piv = {v: A.pivot_table(index=["model", "date", "kind", "block"], columns="arm", values=v).reset_index()
           for v in ("corr", "payout", "corr_kept")}
    rows, by = [], []
    for m in MODELS:
        g = piv["corr"][piv["corr"]["model"] == m]
        gp = piv["payout"][piv["payout"]["model"] == m]
        gk = piv["corr_kept"][piv["corr_kept"]["model"] == m]
        for x, y, what in COMPARISONS:
            if x not in g or y not in g:
                continue
            for sc in ("dev", "test"):
                mk = g["kind"] == sc
                d = g.loc[mk, x] - g.loc[mk, y]
                if d.notna().sum() == 0:
                    continue
                r = dict(model=m, comparison="%s - %s" % (x, y), meaning=what, scope=sc, **E.describe(d))
                dk = gk.loc[gk["kind"] == sc, x] - gk.loc[gk["kind"] == sc, y]
                dp = (gp.loc[gp["kind"] == sc, x] - gp.loc[gp["kind"] == sc, y]).dropna()
                r.update(kept_mean=dk.mean(), kept_t=E.newey_west(dk.values)[2], payout_mean=dp.mean(),
                         payout_t=E.newey_west(dp.values)[2], payout_n=len(dp))
                rows.append(r)
            dd = pd.DataFrame({"date": g["date"], "block": g["block"], "d": g[x] - g[y]})
            for b, z in dd.groupby("block"):
                by.append(dict(model=m, comparison="%s - %s" % (x, y), by="block", key=b, mean=z["d"].mean()))
            for yr, z in dd.groupby(dd["date"].dt.year):
                by.append(dict(model=m, comparison="%s - %s" % (x, y), by="year", key=yr, mean=z["d"].mean()))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(ROOT, "paired_deltas_by_block_year.csv"), index=False)

    sp = S.pivot_table(index=["model", "seed", "date", "kind"], columns="arm", values="corr").reset_index()
    rows = []
    for (m, s), g in sp.groupby(["model", "seed"]):
        for x, y, what in COMPARISONS:
            if x in g and y in g:
                for sc in ("dev", "test"):
                    d = (g[x] - g[y])[g["kind"] == sc]
                    rows.append(dict(model=m, seed=s, comparison="%s - %s" % (x, y), scope=sc, mean=d.mean(), nw_t=E.newey_west(d.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"), index=False)

    rows = []
    for x, y, what in COMPARISONS:
        P = {}
        for m in MODELS:
            g = piv["corr"][(piv["corr"]["model"] == m) & (piv["corr"]["kind"] == "dev")]
            if x in g and y in g:
                P[m] = pd.Series((g[x] - g[y]).values, index=g["date"].values)
        if len(P) == 3:
            P = pd.DataFrame(P).dropna()
            rows.append(dict(comparison="%s - %s" % (x, y), meaning=what, **{"avg_" + k: v for k, v in E.describe(P.mean(axis=1)).items()}))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "cross_model.csv"), index=False)

    BA = BL.groupby(["model", "dataset", "date", "kind", "block", "w"], as_index=False)["corr"].mean()
    rows = []
    for (m, k), g in BA.groupby(["model", "dataset"]):
        dev = g[g["kind"] == "dev"].groupby("w")["corr"].mean()
        lobo = [g[(g["kind"] == "dev") & (g["block"] != b)].groupby("w")["corr"].mean().idxmax() for b in sorted(g.loc[g["kind"] == "dev", "block"].unique())]
        w = float(dev.idxmax())
        ser = lambda kind, ww: g[(g["kind"] == kind) & (g["w"] == ww)].set_index("date")["corr"]  # noqa: E731
        t = ser("test", w) - ser("test", 0.0)
        dv = ser("dev", w) - ser("dev", 0.0)
        rows.append(dict(model=m, dataset=k, w_star=w, lobo=",".join("%.2f" % x for x in lobo), dev_delta=dv.mean(),
                         dev_t=E.newey_west(dv.values)[2], test_w0=ser("test", 0.0).mean(), test_wstar=ser("test", w).mean(),
                         test_delta=t.mean(), test_t=E.newey_west(t.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "blend_summary.csv"), index=False)

    rows = []
    for m in MODELS:
        ser = {"B0": tuned_v20(m)}
        for a in ("B0F", "L", "S", "C"):
            s = tuned(m, "e132_%s_tune%s" % (m, a))
            if s is not None:
                ser[a] = s
            else:
                rows.append(dict(model=m, comparison=a, note="tuned run missing"))
        P = pd.DataFrame(ser)
        for a in ("L", "S", "C"):
            for ref in ("B0", "B0F"):
                if a in P and ref in P:
                    Q = P[[a, ref]].dropna()
                    for sc, mk in (("pre-test", Q.index < TEST0), ("test", Q.index >= TEST0)):
                        d = (Q[a] - Q[ref])[mk]
                        rows.append(dict(model=m, comparison="%s - %s" % (a, ref), scope=sc, arm=Q.loc[mk, a].mean(), ref=Q.loc[mk, ref].mean(), **E.describe(d)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tuned_equal_budget.csv"), index=False)

    audit()
    figures(A, piv["corr"], BA, L, D)
    json.dump({"scored_files": len(jobs), "missing": missing}, open(os.path.join(ROOT, "analysis_meta.json"), "w"), default=str)
    print("wrote", ROOT)


def audit():
    """Who the neutral rows are, and whether the indicators still carry the future."""
    k = pd.read_parquet(os.path.join(PRED, "keys.parquet"))
    k["year"] = k["date"].dt.year
    a = k.groupby("year").agg(rows=("no_feed", "size"), dropped=("no_feed", "mean"))
    t = k.groupby(["year", "no_feed"])[T20].mean().unstack()
    a["target_kept"], a["target_dropped"] = t[0.0], t[1.0]
    a.to_csv(os.path.join(ROOT, "audit_dropped_rows.csv"))
    last = k["date"].max()
    live = set(k.loc[k["date"] == last, "symbol"])
    d = k[k["no_feed"] == 1]
    s = dict(dropped_rows=len(d), share=len(d) / len(k), symbols=d["symbol"].nunique(),
             symbols_alive_on_last_date=len(set(d["symbol"]) & live), target_dropped=d[T20].mean(),
             target_kept=k.loc[k["no_feed"] == 0, T20].mean())
    json.dump(s, open(os.path.join(ROOT, "audit_summary.json"), "w"), indent=1, default=float)
    # indicator survivorship inside the kept rows, by group of coin
    f = pd.read_parquet("data/crypto/v2.0-litmom-feed-fillind/train.parquet").reset_index()
    f = f[f["date"].isin(k["date"])]
    rows = []
    for c in [c for c in f if c.startswith("feature_isna_")]:
        for sc, g in (("dev", f[f["date"] < TEST0]), ("test", f[f["date"] >= TEST0])):
            on = g[g[c] == 1]
            rows.append(dict(indicator=c, scope=sc, share=(g[c] == 1).mean(), target_on=on[T20].mean(), target_off=g.loc[g[c] == 0, T20].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "audit_indicators.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    m = k.groupby("date")["no_feed"].mean()
    ax.plot(m.index, m.values, color="#2a78d6", lw=1.4)
    ax.set_ylabel("share of keys at neutral"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "audit_neutral_share.png"), dpi=130); plt.close(fig)


def _save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=130); plt.close(fig)


def figures(A, piv, BA, L, D):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        for arm in HEAD:
            g = A[(A["model"] == m) & (A["arm"] == arm)].sort_values("date")
            ax.plot(g["date"], g["corr"].cumsum(), lw=1.5, color=COL[arm], label=arm)
        ax.axvline(TEST0, color="k", ls=":", lw=1); ax.set_title(E.LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("cumulative CORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "01_cumulative_corr.png")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, m in zip(axs, MODELS):
        g = piv[piv["model"] == m].sort_values("date")
        for x in HEAD[1:]:
            ax.plot(g["date"], (g[x] - g["B0"]).cumsum(), color=COL[x], lw=1.5, label=x + " - B0")
        ax.axhline(0, color="k", lw=.8); ax.axvline(TEST0, color="k", ls=":", lw=1); ax.set_title(E.LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("cumulative paired ΔCORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "02_cumulative_delta_vs_B0.png")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, m in zip(axs, MODELS):
        for arm in HEAD:
            g = A[(A["model"] == m) & (A["arm"] == arm)].sort_values("date")
            ax.plot(g["date"], g["corr"].rolling(60).mean(), color=COL[arm], lw=1.3, label=arm)
        ax.set_title(E.LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("60-date rolling mean CORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "03_rolling_corr.png")
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    for ax, m in zip(axs, MODELS):
        g = piv[(piv["model"] == m) & (piv["kind"] == "dev")]
        for x in HEAD[2:]:
            ax.hist(g[x] - g["B0F"], bins=60, histtype="step", color=COL[x], label=x + " - B0F")
        ax.axvline(0, color="k", lw=.8); ax.set_title(E.LABEL[m] + " (dev)")
    axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "04_delta_distribution.png")
    S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"))
    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    for j, m in enumerate(MODELS):
        g = S[S["model"] == m]
        pb = g.pivot_table(index="block", columns="arm", values="corr", aggfunc="mean")
        pb[HEAD[2:]].sub(pb["B0F"], axis=0).plot.bar(ax=axs[0, j], color=[COL[x] for x in HEAD[2:]], legend=j == 0)
        axs[0, j].axhline(0, color="k", lw=.8); axs[0, j].set_title(E.LABEL[m] + ": arm - B0F by block (11 = test)")
        ps = g[g["kind"] == "dev"].pivot_table(index="seed", columns="arm", values="corr", aggfunc="mean")
        ps[HEAD[2:]].sub(ps["B0F"], axis=0).plot.bar(ax=axs[1, j], color=[COL[x] for x in HEAD[2:]], legend=False)
        axs[1, j].axhline(0, color="k", lw=.8); axs[1, j].set_title("by seed (dev): arm - B0F")
    _save(fig, "05_by_fold_and_seed.png")
    lv = L[L["scope"] == "dev"].pivot_table(index="model", columns="arm", values="mean")
    hm = lv.sub(lv["B0"], axis=0)[[a for a in ARMS if a in lv]].reindex(MODELS)
    fig, ax = plt.subplots(figsize=(17, 3.4))
    v = np.nanmax(np.abs(hm.values))
    im = ax.imshow(hm.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([E.LABEL[m] for m in MODELS])
    for i in range(hm.shape[0]):
        for k in range(hm.shape[1]):
            ax.text(k, i, "%+.3f" % hm.values[i, k], ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="dev CORR - B0")
    _save(fig, "06_model_feature_heatmap.png")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, m in zip(axs, MODELS):
        for k in DATASETS:
            g = BA[(BA["model"] == m) & (BA["kind"] == "dev") & (BA["dataset"] == k)].groupby("w")["corr"].mean()
            if len(g):
                ax.plot(g.index, g.values - g.loc[0.0], marker="o", label=DATASETS[k])
        ax.axhline(0, color="k", lw=.8); ax.set_xlabel("w on ADDED"); ax.set_title(E.LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("dev CORR - w=0"); axs[0].legend(frameon=False)
    _save(fig, "07_blend_sensitivity.png")
    shutil.copy(os.path.join(ROOT, "audit_neutral_share.png"), os.path.join(FIG, "08_missingness_time.png"))
    fig, axs = plt.subplots(1, 3, figsize=(17, 8), sharex=True)
    for ax, m in zip(axs, MODELS):
        g = D[(D["model"] == m) & (D["scope"] == "dev")].iloc[::-1]
        ax.errorbar(g["mean"], range(len(g)), xerr=[g["mean"] - g["nw_lo"], g["nw_hi"] - g["mean"]], fmt="o", color="#2a78d6", ms=3)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g["comparison"], fontsize=7); ax.axvline(0, color="k", lw=.8)
        ax.set_title(E.LABEL[m] + " (dev, NW 95% CI)"); ax.grid(alpha=.3)
    _save(fig, "09_ablation_ci.png")
    pc = L[L["scope"] == "dev"].pivot_table(index="arm", columns="model", values="predcorr_b0")[MODELS]
    fig, ax = plt.subplots(figsize=(9, 7))
    pc.drop(index="B0", errors="ignore").plot.barh(ax=ax); ax.set_xlabel("mean per-date rank corr with B0 (dev)"); ax.grid(alpha=.3)
    _save(fig, "10_predcorr_with_B0.png")


if __name__ == "__main__":
    main()
