#!/usr/bin/env python3
"""
E131 analysis: score the factorial's out-of-sample predictions and write every table/figure.

    python3 experiments/e131_analysis.py

Reads results/reports/crypto/e131_litmom_fillind/preds/<model>/<arm>_s<seed>.parquet
(written by e131_litmom_fillind_factorial.py) and the six equal-budget searched runs
results/per_period/crypto/<model>/e131_<model>_tune{B0,FULL}.csv.

Scoring (per date): CORR = numerai_tools.scoring.numerai_corr(prediction, target_binned_return_20);
"custom contribution relative to B0" = numerai_tools.scoring.correlation_contribution(prediction,
B0 prediction of the same model and seed, target) -- NOT official MMC; MMC against the historical
Crypto meta model is reported separately, where that meta model exists (from 2024-05-07).
Seeds: per-date scores are averaged over seeds for the headline series; each seed is also kept.
Uncertainty: Newey-West (lag 20, Bartlett) and moving-block bootstrap (block 20, 2000 draws) on
per-date paired deltas -- adjacent dates share 19/20 of the target window.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numerai_tools.scoring import correlation_contribution, numerai_corr  # noqa: E402

T20 = "target_binned_return_20"
ROOT = "results/reports/crypto/e131_litmom_fillind"
PRED = os.path.join(ROOT, "preds")
FIG = os.path.join(ROOT, "figures")
MODELS = ["ridge", "lgb_ktz", "mlp_ktz"]
LABEL = {"ridge": "ridge_ktz -> ridge", "lgb_ktz": "gbm_ktz -> lgb_ktz", "mlp_ktz": "mlp_ktz"}
ARMS = ["B0", "B0C", "B1", "M1", "M2", "FULL", "ADDED_ONLY", "M1FB", "P_NM", "P_HIST", "MISS_ONLY",
        "M1_no_ret", "M1_no_maxvol", "M1_no_trend"]
COMPARISONS = [("M1", "B0", "raw momentum"), ("M2", "M1", "momentum indicators"),
               ("B1", "B0", "base indicators"), ("FULL", "M1", "all indicators"),
               ("FULL", "B0", "full candidate"), ("B0C", "B0", "preprocessing / base copy"),
               ("M1FB", "M1", "fallback fill vs 0.5 fill"), ("P_NM", "B0", "survivorship placebo"),
               ("P_HIST", "B0", "causal tenure control"), ("FULL", "P_NM", "full vs placebo"),
               ("M1", "M1_no_ret", "returns family (LOFO)"), ("M1", "M1_no_maxvol", "max/vol family (LOFO)"),
               ("M1", "M1_no_trend", "trend family (LOFO)"), ("ADDED_ONLY", "B0", "added-only vs base"),
               ("MISS_ONLY", "B0", "missingness-only vs base")]
W = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
LAG, BLK, NBOOT = 20, 20, 2000


# ---------------------------------------------------------------- statistics
def newey_west(x, lag=LAG):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    m = x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 0) / n)
    return m, se, (m / se if se > 0 else np.nan)


def block_bootstrap_ci(x, block=BLK, reps=NBOOT, seed=131):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block * 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def max_dd(x):
    c = np.cumsum(np.nan_to_num(np.asarray(x, float)))
    return float(np.max(np.maximum.accumulate(np.r_[0, c])[1:] - c)) if len(c) else np.nan


def describe(s):
    s = pd.Series(s).dropna()
    m, se, t = newey_west(s.values)
    lo, hi = block_bootstrap_ci(s.values)
    return dict(mean=s.mean(), median=s.median(), std=s.std(ddof=0),
                sharpe=s.mean() / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan,
                pos_rate=(s > 0).mean(), cum=s.sum(), max_dd=max_dd(s.values), worst=s.min(),
                nw_se=se, nw_t=t, nw_lo=m - 1.96 * se, nw_hi=m + 1.96 * se, boot_lo=lo, boot_hi=hi, n=len(s))


# ---------------------------------------------------------------- per-date scoring
def corr_one(g, col="prediction"):
    return float(numerai_corr(g[[col]], g[T20]).iloc[0])


def contrib_one(g, ref):
    gg = g.dropna(subset=[ref])
    if len(gg) < 10:
        return np.nan
    return float(correlation_contribution(gg[["prediction"]], gg[ref], gg[T20].copy()).iloc[0])


def score_file(job):
    model, arm, seed, path, b0path = job
    f = pd.read_parquet(path)
    b0 = pd.read_parquet(b0path)[["symbol", "date", "prediction"]].rename(columns={"prediction": "b0"})
    f = f.merge(b0, on=["symbol", "date"], how="left", validate="1:1")
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        r = dict(model=model, arm=arm, seed=seed, date=d, block=int(g["block"].iloc[0]), kind=g["kind"].iloc[0],
                 n_symbols=len(g), corr=corr_one(g))
        r["contrib_b0"] = contrib_one(g, "b0") if arm != "B0" else 0.0
        r["predcorr_b0"] = g["prediction"].rank().corr(g["b0"].rank())
        r["mmc"] = contrib_one(g, "meta_model") if g["meta_model"].notna().sum() >= 10 else np.nan
        gm = g[g["matched"] == 1]
        r["corr_matched"] = corr_one(gm) if len(gm) >= 10 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def blend_scores(job):
    """Per date CORR of (1-w)*rank(B0) + w*rank(ADDED_ONLY) for every w, one model and seed."""
    model, seed = job
    b = pd.read_parquet(os.path.join(PRED, model, "B0_s%d.parquet" % seed))
    a = pd.read_parquet(os.path.join(PRED, model, "ADDED_ONLY_s%d.parquet" % seed))[["symbol", "date", "prediction"]]
    f = b.merge(a.rename(columns={"prediction": "added"}), on=["symbol", "date"], validate="1:1")
    f["rb"] = f.groupby("date")["prediction"].rank(pct=True)
    f["ra"] = f.groupby("date")["added"].rank(pct=True)
    rows = []
    for d, g in f.groupby("date", sort=True):
        g = g.set_index("symbol")
        for w in W:
            g["bl"] = (1 - w) * g["rb"] + w * g["ra"]
            rows.append(dict(model=model, seed=seed, date=d, kind=g["kind"].iloc[0], block=int(g["block"].iloc[0]),
                             w=w, corr=corr_one(g, "bl")))
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG, exist_ok=True)
    jobs = []
    for m in MODELS:
        for p in sorted(glob.glob(os.path.join(PRED, m, "*_s*.parquet"))):
            arm, seed = os.path.basename(p)[:-8].rsplit("_s", 1)
            b0 = os.path.join(PRED, m, "B0_s%s.parquet" % seed)
            if arm in ARMS and os.path.exists(b0):
                jobs.append((m, arm, int(seed), p, b0))
    have = pd.DataFrame([(j[0], j[1], j[2]) for j in jobs], columns=["model", "arm", "seed"])
    missing = [(m, a) for m in MODELS for a in ARMS if not ((have.model == m) & (have.arm == a)).any()]
    print("scoring %d (model, arm, seed) files; missing (model, arm): %s" % (len(jobs), missing or "none"))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("E131_WORKERS", "12"))) as ex:
        S = pd.concat(list(ex.map(score_file, jobs)), ignore_index=True)
        bl_jobs = [(m, s) for m in MODELS for s in sorted(have[(have.model == m) & (have.arm == "ADDED_ONLY")].seed)]
        BL = pd.concat(list(ex.map(blend_scores, bl_jobs)), ignore_index=True)
    S["date"] = pd.to_datetime(S["date"])
    S.to_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"), index=False)
    S.to_csv(os.path.join(ROOT, "scores_by_date_seed.csv.gz"), index=False)

    # seed-averaged per-date series
    num = ["corr", "contrib_b0", "predcorr_b0", "mmc", "corr_matched"]
    A = S.groupby(["model", "arm", "date", "block", "kind"], as_index=False)[num].mean()
    A["year"] = A["date"].dt.year
    scopes = {"dev": A["kind"] == "dev", "dev_2021+": (A["kind"] == "dev") & (A["date"] >= "2021-01-01"),
              "test": A["kind"] == "test", "dev_mmc_window": (A["kind"] == "dev") & A["mmc"].notna()}

    # 1. level summaries
    rows = []
    for (m, arm), g in A.groupby(["model", "arm"]):
        for sc, mask in scopes.items():
            gg = g[mask.loc[g.index]]
            if gg.empty:
                continue
            r = dict(model=m, arm=arm, scope=sc, **{k: v for k, v in describe(gg["corr"]).items()})
            r.update(contrib_b0=gg["contrib_b0"].mean(), predcorr_b0=gg["predcorr_b0"].mean(),
                     mmc=gg["mmc"].mean(), corr_matched=gg["corr_matched"].mean(),
                     worst_block_mean=gg.groupby("block")["corr"].mean().min(),
                     recent_130=g[g["kind"] == "dev"].sort_values("date")["corr"].tail(130).mean())
            rows.append(r)
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(ROOT, "summary_levels.csv"), index=False)

    # 2. paired deltas (seed-averaged per date, same model)
    piv = A.pivot_table(index=["model", "date", "kind", "block"], columns="arm", values="corr").reset_index()
    pivm = A.pivot_table(index=["model", "date", "kind", "block"], columns="arm", values="corr_matched").reset_index()
    rows, by = [], []
    for m in MODELS:
        g, gm = piv[piv["model"] == m], pivm[pivm["model"] == m]
        for x, y, what in COMPARISONS:
            if x not in g or y not in g:
                continue
            for sc in ("dev", "dev_2021+", "test"):
                mk = (g["kind"] == ("test" if sc == "test" else "dev")) & ((g["date"] >= "2021-01-01") if sc == "dev_2021+" else True)
                d = (g.loc[mk, x] - g.loc[mk, y])
                dm = (gm.loc[mk, x] - gm.loc[mk, y]) if x in gm and y in gm else pd.Series(dtype=float)
                r = dict(model=m, comparison="%s - %s" % (x, y), meaning=what, scope=sc, **describe(d))
                r["mean_matched_only"] = dm.mean()
                r["nw_t_matched_only"] = newey_west(dm.values)[2]
                rows.append(r)
            dd = pd.DataFrame({"date": g["date"], "block": g["block"], "kind": g["kind"], "d": g[x] - g[y]})
            for b, z in dd.groupby("block"):
                by.append(dict(model=m, comparison="%s - %s" % (x, y), by="block", key=b, mean=z["d"].mean(), n=len(z)))
            for yr, z in dd.groupby(dd["date"].dt.year):
                by.append(dict(model=m, comparison="%s - %s" % (x, y), by="year", key=yr, mean=z["d"].mean(), n=len(z)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(ROOT, "paired_deltas.csv"), index=False)
    pd.DataFrame(by).to_csv(os.path.join(ROOT, "paired_deltas_by_block_year.csv"), index=False)

    # per-seed paired deltas (seed stability)
    sp = S.pivot_table(index=["model", "seed", "date", "kind"], columns="arm", values="corr").reset_index()
    rows = []
    for (m, s), g in sp.groupby(["model", "seed"]):
        for x, y, what in COMPARISONS:
            if x in g and y in g:
                for sc in ("dev", "test"):
                    d = (g[x] - g[y])[g["kind"] == sc]
                    rows.append(dict(model=m, seed=s, comparison="%s - %s" % (x, y), scope=sc, mean=d.mean(),
                                     nw_t=newey_west(d.values)[2]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "paired_deltas_by_seed.csv"), index=False)
    # model x arm x block x seed detail table
    S.groupby(["model", "arm", "block", "kind", "seed"]).agg(corr=("corr", "mean"), contrib_b0=("contrib_b0", "mean"),
                                                              mmc=("mmc", "mean"), n_dates=("corr", "size")).reset_index() \
        .to_csv(os.path.join(ROOT, "detail_model_arm_block_seed.csv"), index=False)

    # 3. cross-model: average effect and heterogeneity (dev)
    rows = []
    for x, y, what in COMPARISONS:
        per = {}
        for m in MODELS:
            g = piv[(piv["model"] == m) & (piv["kind"] == "dev")]
            if x in g and y in g:
                per[m] = pd.Series((g[x] - g[y]).values, index=g["date"].values)
        if len(per) < 3:
            continue
        P = pd.DataFrame(per).dropna()
        r = dict(comparison="%s - %s" % (x, y), meaning=what, **{"avg_" + k: v for k, v in describe(P.mean(axis=1)).items()})
        for a_, b_ in (("ridge", "lgb_ktz"), ("ridge", "mlp_ktz"), ("lgb_ktz", "mlp_ktz")):
            mm, _, t = newey_west((P[a_] - P[b_]).values)
            r["het_%s_vs_%s" % (a_, b_)] = mm
            r["het_t_%s_vs_%s" % (a_, b_)] = t
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "cross_model.csv"), index=False)

    # 4. blend: choose w on DEV (max mean dev CORR, seed-averaged), evaluate once on TEST
    BL.to_csv(os.path.join(ROOT, "blend_by_date_seed_w.csv.gz"), index=False)
    BA = BL.groupby(["model", "date", "kind", "block", "w"], as_index=False)["corr"].mean()
    rows = []
    for m, g in BA.groupby("model"):
        dev = g[g["kind"] == "dev"].groupby("w")["corr"].mean()
        # leave-one-block-out choice, to show how stable the CV choice is
        lobo = [g[(g["kind"] == "dev") & (g["block"] != b)].groupby("w")["corr"].mean().idxmax()
                for b in sorted(g.loc[g["kind"] == "dev", "block"].unique())]
        w_star = float(dev.idxmax())
        t0 = g[(g["kind"] == "test") & (g["w"] == 0.0)].set_index("date")["corr"]
        tw = g[(g["kind"] == "test") & (g["w"] == w_star)].set_index("date")["corr"]
        dd = (tw - t0)
        mm, se, tt = newey_west(dd.values)
        r = dict(model=m, w_star=w_star, lobo_choices=",".join("%.2f" % x for x in lobo),
                 **{"dev_w%.2f" % w: dev[w] for w in W},
                 test_w0=t0.mean(), test_wstar=tw.mean(), test_delta=mm, test_delta_t=tt)
        dv0 = g[(g["kind"] == "dev") & (g["w"] == 0.0)].set_index("date")["corr"]
        dvs = g[(g["kind"] == "dev") & (g["w"] == w_star)].set_index("date")["corr"]
        r["dev_delta_t_at_wstar"] = newey_west((dvs - dv0).values)[2]
        rows.append(r)
    BLS = pd.DataFrame(rows)
    BLS.to_csv(os.path.join(ROOT, "blend_summary.csv"), index=False)

    # 5. equal-budget searched runs (secondary)
    rows = []
    for m in MODELS:
        ser = {}
        for v in ("B0", "FULL"):
            p = "results/per_period/crypto/%s/e131_%s_tune%s.csv" % (m, m, v)
            if os.path.exists(p):
                ser[v] = pd.read_csv(p, parse_dates=["date"]).set_index("date")["corr"]
        if len(ser) < 2:
            rows.append(dict(model=m, note="tuned runs missing: %s" % sorted(set(("B0", "FULL")) - set(ser))))
            continue
        P = pd.DataFrame(ser).dropna()
        for sc, mk in (("pre-test (2024-08..2026-02-19)", P.index < "2026-02-20"), ("test block", P.index >= "2026-02-20")):
            d = (P["FULL"] - P["B0"])[mk]
            rows.append(dict(model=m, scope=sc, B0=P.loc[mk, "B0"].mean(), FULL=P.loc[mk, "FULL"].mean(), **describe(d)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tuned_equal_budget.csv"), index=False)

    figures(A, piv, BA, L)
    json.dump({"scored_files": len(jobs), "missing": missing}, open(os.path.join(ROOT, "analysis_meta.json"), "w"), default=str)
    print("wrote tables + figures under", ROOT)


# ---------------------------------------------------------------- figures
COL = {"B0": "#52514e", "M1": "#2a78d6", "M2": "#eb6834", "FULL": "#1baf7a", "B1": "#8a5cd1", "B0C": "#b0aea6",
       "ADDED_ONLY": "#d6a72a", "P_NM": "#c94a4a", "P_HIST": "#6b9e2f"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=130)
    plt.close(fig)


def figures(A, piv, BA, L):
    dev = A[A["kind"].isin(["dev", "test"])]
    # 1 cumulative CORR by feature set, per model
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        for arm in ["B0", "M1", "M2", "FULL", "B1", "P_NM"]:
            g = dev[(dev["model"] == m) & (dev["arm"] == arm)].sort_values("date")
            ax.plot(g["date"], g["corr"].cumsum(), lw=1.6, color=COL[arm], label=arm)
        ax.axvline(pd.Timestamp("2026-02-20"), color="k", ls=":", lw=1)
        ax.set_title(LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("cumulative CORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "01_cumulative_corr.png")
    # 2 cumulative FULL-B0 (and M1-B0, P_NM-B0)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, m in zip(axs, MODELS):
        g = piv[piv["model"] == m].sort_values("date")
        for x, c in (("FULL", COL["FULL"]), ("M1", COL["M1"]), ("P_NM", COL["P_NM"])):
            ax.plot(g["date"], (g[x] - g["B0"]).cumsum(), color=c, lw=1.6, label="%s - B0" % x)
        ax.axhline(0, color="k", lw=.8); ax.axvline(pd.Timestamp("2026-02-20"), color="k", ls=":", lw=1)
        ax.set_title(LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("cumulative paired ΔCORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "02_cumulative_delta_vs_B0.png")
    # 3 rolling 60-date mean CORR
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, m in zip(axs, MODELS):
        for arm in ["B0", "M1", "FULL"]:
            g = dev[(dev["model"] == m) & (dev["arm"] == arm)].sort_values("date")
            ax.plot(g["date"], g["corr"].rolling(60).mean(), color=COL[arm], lw=1.4, label=arm)
        ax.set_title(LABEL[m]); ax.grid(alpha=.3)
    axs[0].set_ylabel("60-date rolling mean CORR"); axs[0].legend(frameon=False, fontsize=8)
    _save(fig, "03_rolling_corr.png")
    # 4 distribution of per-date FULL-B0
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    for ax, m in zip(axs, MODELS):
        g = piv[(piv["model"] == m) & (piv["kind"] == "dev")]
        ax.hist(g["FULL"] - g["B0"], bins=60, color=COL["FULL"], alpha=.8)
        ax.axvline(0, color="k", lw=.8); ax.set_title(LABEL[m] + ": FULL - B0 per date (dev)")
    _save(fig, "04_delta_distribution.png")
    # 5 by fold (block) and seed
    S = pd.read_parquet(os.path.join(ROOT, "scores_by_date_seed.parquet"))
    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    for j, m in enumerate(MODELS):
        g = S[S["model"] == m]
        pb = g.pivot_table(index="block", columns="arm", values="corr", aggfunc="mean")
        (pb["FULL"] - pb["B0"]).plot.bar(ax=axs[0, j], color=COL["FULL"]); axs[0, j].axhline(0, color="k", lw=.8)
        axs[0, j].set_title(LABEL[m] + ": FULL - B0 by block (11 = test)")
        ps = g[g["kind"] == "dev"].pivot_table(index="seed", columns="arm", values="corr", aggfunc="mean")
        (ps[["M1", "M2", "FULL"]].sub(ps["B0"], axis=0)).plot.bar(ax=axs[1, j]); axs[1, j].axhline(0, color="k", lw=.8)
        axs[1, j].set_title("by seed (dev): arm - B0")
    _save(fig, "05_by_fold_and_seed.png")
    # 6 model x feature heatmap (dev mean CORR minus B0)
    lv = L[L["scope"] == "dev"].pivot_table(index="model", columns="arm", values="mean")
    hm = lv.sub(lv["B0"], axis=0)[[a for a in ARMS if a in lv]].loc[MODELS]
    fig, ax = plt.subplots(figsize=(14, 3.2))
    v = np.nanmax(np.abs(hm.values))
    im = ax.imshow(hm.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=40, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([LABEL[m] for m in MODELS])
    for i in range(hm.shape[0]):
        for k in range(hm.shape[1]):
            ax.text(k, i, "%+.4f" % hm.values[i, k], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="dev mean CORR - B0")
    _save(fig, "06_model_feature_heatmap.png")
    # 7 blend weight sensitivity (dev)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in MODELS:
        g = BA[(BA["model"] == m) & (BA["kind"] == "dev")].groupby("w")["corr"].mean()
        ax.plot(g.index, g.values - g.loc[0.0], marker="o", label=LABEL[m])
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("w on ADDED_ONLY"); ax.set_ylabel("dev mean CORR - w=0")
    ax.legend(frameon=False); ax.grid(alpha=.3)
    _save(fig, "07_blend_sensitivity.png")
    # 8 missingness and coverage through time (copied from the audit if present)
    src = os.path.join(ROOT, "audit", "missingness_time.png")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(FIG, "08_missingness_time.png"))
    # 9 feature-group ablation (dev ΔCORR with NW 95% CI)
    D = pd.read_csv(os.path.join(ROOT, "paired_deltas.csv"))
    D = D[D["scope"] == "dev"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, m in zip(axs, MODELS):
        g = D[D["model"] == m].iloc[::-1]
        ax.errorbar(g["mean"], range(len(g)), xerr=[g["mean"] - g["nw_lo"], g["nw_hi"] - g["mean"]], fmt="o", color="#2a78d6")
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g["comparison"], fontsize=8); ax.axvline(0, color="k", lw=.8)
        ax.set_title(LABEL[m] + " (dev, NW 95% CI)"); ax.grid(alpha=.3)
    _save(fig, "09_ablation_ci.png")
    # 10 prediction correlation with B0
    pc = L[L["scope"] == "dev"].pivot_table(index="arm", columns="model", values="predcorr_b0")[MODELS]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pc.drop(index="B0", errors="ignore").plot.barh(ax=ax); ax.set_xlabel("mean per-date rank corr with B0 (dev)")
    ax.grid(alpha=.3)
    _save(fig, "10_predcorr_with_B0.png")


if __name__ == "__main__":
    main()
