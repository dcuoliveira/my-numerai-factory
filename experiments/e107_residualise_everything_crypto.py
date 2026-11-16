"""
Experiment E107: residualise every model on a 5-day-old meta model, at s = 0.5.

    python3 experiments/e107_residualise_everything_crypto.py --jobs 30
    python3 experiments/e107_residualise_everything_crypto.py --html --notes <reading.md>

WHAT IT ASKS
  E106 showed that moving the -CORR network off a lagged meta model raises its payout.
  If EVERY model -- the walk-forward ridge, the MSE and -CORR networks, every 20-day
  slot in the book -- lands on the same payout once residualised, the model no longer
  matters: the choice is settled and the work moves to the residualisation / portfolio
  layer. If the spread survives residualisation, model choice still matters.

WHAT IT RUNS
  No training. For each source's stored held-out forecast, on the days every source
  shares (2024-07-25 .. 2026-05-13: 470 days, all with a meta model):

      p = gauss(rank(prediction))        m = gauss(rank(meta model >= 5 days old))
      beta = p.m / m.m (per day)         r = p - 0.5 * beta * m
      submitted = rank(r)

  The meta model is the latest one published at least 5 calendar days before the day,
  from data/crypto/v2.0/historical_meta_models.parquet -- the deployable setting of E106
  (the meta model appears about 2 days after a round). Every arm is SCORED against the
  same-day meta model, which is what Numerai pays on. s = 0 is the source unchanged.

  Gate: s = 0 must equal the repo's scorer applied to the same stored forecast (1e-9),
  i.e. the pipeline is the identity before any residualising. The stored per-period
  series is NOT the reference: for some blends it is not reproduced by rescoring the
  blend's own stored forecast (CURUPIRA_05: up to 1.6e-3 a day, 4e-6 on the mean -- the
  forecast and the series were evidently written at different times, and a rank average
  is full of near-ties). Every such source is listed with its deviation; every number
  E107 compares is the stored forecast rescored, before and after, by the same code.

SOURCES
  ridge     the walk-forward ridge on fenccausal (alpha 1, the P3 baseline of the
            network reports), ridge/wfdiag_ridge_full_a1
  networks  the from-scratch identity 64-32-16 on MSE and on -CORR, seeds 42 and 43
            (torch_scratch/torchscratch_{mse,corr}_500ep_wf_s4X_ep500). The two seeds of
            one recipe are the 'same model' noise floor for the pair tests.
  book      every slot in src.live.registry.ORDER scored on target_binned_return_20,
            resolved by src.analysis.slots.slot_run. SACI_03 and CURUPIRA_05_60 are
            scored on the 60-day label and are left out, named.

WHAT WOULD COUNT AS CONVERGENCE
  After residualisation the cross-model spread of payout collapses, and the share of
  model pairs whose paired payout difference clears |t| >= 2 (Newey-West, lag 20) falls
  to the level of two seeds of one network. If the before/after rankings stay correlated
  and pairs stay separated, model choice still matters.

ARTEFACTS
  results/{forecast,per_period,metrics}/crypto/meta_resid/e107_<model>_s050
  results/reports/crypto/e107_residualise_everything/{table.csv,pairs.csv,figure.png}
  results/reports/crypto/e107_residualise_everything.html
"""
import argparse
import base64
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "target_binned_return_20"
TREE = "meta_resid"
STRENGTH, LAG = 0.5, 5
META_FILE = os.path.join("data", "crypto", "v2.0", "historical_meta_models.parquet")
OUT_DIR = os.path.join("results", "reports", "crypto", "e107_residualise_everything")
RESEARCH = [
    ("ridge (walk-forward)", "ridge_wf", "ridge", "wfdiag_ridge_full_a1"),
    ("MSE net s42", "mse_s42", "torch_scratch", "torchscratch_mse_500ep_wf_s42_ep500"),
    ("MSE net s43", "mse_s43", "torch_scratch", "torchscratch_mse_500ep_wf_s43_ep500"),
    ("-CORR net s42", "corr_s42", "torch_scratch", "torchscratch_corr_500ep_wf_s42_ep500"),
    ("-CORR net s43", "corr_s43", "torch_scratch", "torchscratch_corr_500ep_wf_s43_ep500"),
]
SEED_PAIRS = [("mse_s42", "mse_s43"), ("corr_s42", "corr_s43")]


def sources():
    """Every source as a dict, the research arms first, then the book in registry order."""
    from src.analysis import slots
    from src.analysis.results_table import target_of
    from src.live import registry
    out, dropped = [], []
    for label, slug, model, run in RESEARCH:
        out.append({"label": label, "slug": slug, "model": model, "run": run, "group": "research",
                    "staked": False})
    staked = slots._staked()
    for name in registry.ORDER:
        model, run, note = slots.slot_run(name)
        if not model:
            dropped.append((name, note))
            continue
        # The slots this study gave rise to (CU_SA_IA_BO_MA_R_01, CU_R_IA_MA_60_01)
        # are already residualised; residualising them again is not an arm, and
        # skipping them keeps a re-run at the 38 the study reported.
        if model == TREE:
            dropped.append((name, "already residualised (a product of this study)"))
            continue
        target = target_of("crypto", model, run)
        if target != TARGET:
            dropped.append((name, "scored on %s" % target))
            continue
        spec = registry.get(name)
        group = ("blend" if registry.is_slot_blend(spec) or registry.is_blend(spec) else "single")
        out.append({"label": name, "slug": name.lower(), "model": model, "run": run, "group": group,
                    "staked": name in staked})
    return out, dropped


def forecast_path(src):
    return os.path.join("results", "forecast", "crypto", src["model"], src["run"] + ".csv")


def common_days(srcs):
    days = None
    for s in srcs:
        f = pd.read_csv(forecast_path(s), usecols=["date", "meta_model"], parse_dates=["date"])
        have = set(f.loc[f["meta_model"].notna(), "date"].unique())
        days = have if days is None else days & have
    return sorted(days)


def gauss_rank(values):
    """gauss(tie_kept_rank(x)) on the non-missing entries, 0 where x is missing."""
    s = pd.Series(values)
    u = (s.rank(method="average") - 0.5) / s.count()
    return np.nan_to_num(stats.norm.ppf(u.to_numpy()), nan=0.0)


def lagged_meta(frame, lag):
    """Per row: the symbol's meta model on the latest meta date <= date - lag days."""
    meta = pd.read_parquet(META_FILE, columns=["date", "symbol", "prediction"])
    meta["date"] = pd.to_datetime(meta["date"])
    wide = meta.pivot(index="date", columns="symbol", values="prediction")
    dates = wide.index.to_numpy()
    want = (frame["date"] - pd.Timedelta(days=lag)).to_numpy()
    pos = np.searchsorted(dates, want, side="right") - 1
    src = np.where(pos >= 0, dates[np.clip(pos, 0, None)], np.datetime64("NaT"))
    stacked = wide.stack().rename("m").reset_index()
    out = pd.DataFrame({"date": src, "symbol": frame["symbol"].to_numpy()}).merge(
        stacked, on=["date", "symbol"], how="left")
    return out["m"].to_numpy(), src


def process(task):
    """Worker: one source, unchanged (s = 0) and residualised, scored on the common days."""
    from src.prediction.metrics import per_period_table
    src, days = task
    f = pd.read_csv(forecast_path(src), parse_dates=["date"])
    f = f[f["date"].isin(days)].sort_values(["date", "symbol"]).reset_index(drop=True)
    m_raw, m_src = lagged_meta(f, LAG)
    out = {0.0: np.empty(len(f)), STRENGTH: np.empty(len(f))}
    gp_all, gr_all, diag = np.empty(len(f)), np.empty(len(f)), []
    for day, idx in f.groupby("date").indices.items():
        p = gauss_rank(f["prediction"].to_numpy()[idx])
        m = gauss_rank(m_raw[idx])
        gm_today = gauss_rank(f["meta_model"].to_numpy()[idx])
        if pd.isna(m_src[idx[0]]) or m @ m == 0:
            beta, r = np.nan, p
        else:
            beta = (p @ m) / (m @ m)
            r = p - STRENGTH * beta * m
        for s, values in ((0.0, p), (STRENGTH, r)):
            v = pd.Series(values)
            out[s][idx] = ((v.rank(method="average") - 0.5) / v.count()).to_numpy()
        gr = gauss_rank(r)
        gp_all[idx], gr_all[idx] = p, gr
        diag.append({"date": day, "beta": beta, "stale_days": (day - pd.Timestamp(m_src[idx[0]])).days
                     if not pd.isna(m_src[idx[0]]) else np.nan,
                     "meta_corr_before": np.corrcoef(p, gm_today)[0, 1],
                     "meta_corr_after": np.corrcoef(gr, gm_today)[0, 1]})
    tables = {}
    for s, pred in list(out.items()) + [("plain", f["prediction"].to_numpy())]:
        frame = f[["symbol", "date", TARGET, "reference_prediction", "meta_model"]].assign(prediction=pred)
        tables[s] = per_period_table(frame, "prediction", TARGET, "date",
                                     reference_col="reference_prediction", meta_model_col="meta_model")
    return {"slug": src["slug"], "tables": tables, "pred": out[STRENGTH], "keys": f[["date", "symbol"]],
            "gauss_before": gp_all, "gauss_after": gr_all, "diag": pd.DataFrame(diag),
            "rows": f[["symbol", "date", TARGET, "reference_prediction", "meta_model"]]}


def write_run(src, res):
    from src.prediction.metrics import summarize
    name = "e107_%s_s050" % src["slug"]
    table = res["tables"][STRENGTH]
    results = {"n_periods": int(table["corr"].notna().sum())}
    for family in ("corr", "bmc", "mmc"):
        results[family] = summarize(table[family])
        if family != "corr":
            results["%s_n_periods" % family] = int(table[family].notna().sum())
    results["config"] = {
        "model": TREE, "tournament": "crypto", "task": "regression", "study": "E107",
        "source_run": "%s/%s" % (src["model"], src["run"]), "source_label": src["label"],
        "strength": STRENGTH, "meta_lag_days": LAG, "target": TARGET,
        "transform": "rank(gauss(rank(p)) - s * beta * gauss(rank(m_lag))), per day",
        "meta_source": META_FILE, "assembled_by": "experiments/e107_residualise_everything_crypto.py"}
    for kind in ("forecast", "per_period", "metrics"):
        os.makedirs(os.path.join("results", kind, "crypto", TREE), exist_ok=True)
    base = os.path.join("results", "%s", "crypto", TREE, name)
    res["rows"].assign(prediction=res["pred"])[["symbol", "date", TARGET, "prediction",
                                                 "reference_prediction", "meta_model"]].to_csv(
        base % "forecast" + ".csv", index=False)
    table.to_csv(base % "per_period" + ".csv")
    with open(base % "metrics" + ".json", "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    return name


def summarise(srcs, results, days):
    """One row per source, before and after; the pair tests; the convergence numbers."""
    from src.analysis import results_table as rt
    from src.analysis.ablation_report import newey_west
    pay = {}
    rows = []
    for s in srcs:
        r = results[s["slug"]]
        t0, t5 = r["tables"][0.0].loc[days], r["tables"][STRENGTH].loc[days]
        b, a = rt.row_from_frame(s["label"], t0[["corr", "mmc"]]), rt.row_from_frame(s["label"], t5[["corr", "mmc"]])
        p0, p5 = 0.1 * t0["corr"] + t0["mmc"], 0.1 * t5["corr"] + t5["mmc"]
        pay[s["slug"]] = (p0.to_numpy(), p5.to_numpy())
        d, se, t = newey_west((p5 - p0).to_numpy(), lag=20)
        rows.append({"model": s["label"], "slug": s["slug"], "group": s["group"], "staked": s["staked"],
                     "n": len(days), "CORR": b["CORR"], "MMC": b["MMC"], "payout": b["payout"],
                     "calmar": b["calmar"], "dd(pay)": b["dd(pay)"],
                     "CORR_r": a["CORR"], "MMC_r": a["MMC"], "payout_r": a["payout"], "calmar_r": a["calmar"],
                     "dd(pay)_r": a["dd(pay)"], "burn(d)_r": a["burn(d)"], "dpay": d, "t_dpay": t,
                     "beta_median": float(r["diag"]["beta"].median()),
                     "meta_corr_before": float(r["diag"]["meta_corr_before"].mean()),
                     "meta_corr_after": float(r["diag"]["meta_corr_after"].mean())})
    table = pd.DataFrame(rows)
    slugs = list(table["slug"])
    pairs = []
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            a, b = pay[slugs[i]], pay[slugs[j]]
            pairs.append({"a": slugs[i], "b": slugs[j],
                          "d_before": float(np.mean(a[0] - b[0])), "t_before": newey_west(a[0] - b[0], lag=20)[2],
                          "d_after": float(np.mean(a[1] - b[1])), "t_after": newey_west(a[1] - b[1], lag=20)[2]})
    pairs = pd.DataFrame(pairs)
    return table, pairs


def prediction_correlations(srcs, results):
    """Mean per-day correlation between every two models' predictions, before and after."""
    keys = None
    cols = {}
    for s in srcs:
        r = results[s["slug"]]
        k = r["keys"].assign(**{"b_" + s["slug"]: r["gauss_before"], "a_" + s["slug"]: r["gauss_after"]})
        keys = k if keys is None else keys.merge(k, on=["date", "symbol"], how="inner")
    out = {}
    for when in ("b", "a"):
        names = ["%s_%s" % (when, s["slug"]) for s in srcs]
        mats = [np.corrcoef(g[names].to_numpy().T) for _, g in keys.groupby("date")]
        out[when] = pd.DataFrame(np.nanmean(np.stack(mats), axis=0), index=[s["slug"] for s in srcs],
                                 columns=[s["slug"] for s in srcs])
    return out, len(keys)


def convergence(table, pairs, corrs):
    seed_floor = pairs[pairs.apply(lambda r: (r["a"], r["b"]) in SEED_PAIRS or (r["b"], r["a"]) in SEED_PAIRS,
                                   axis=1)]
    off = lambda m: m.to_numpy()[np.triu_indices(len(m), 1)]
    return {
        "n_models": int(len(table)), "n_pairs": int(len(pairs)),
        "range_before": float(table["payout"].max() - table["payout"].min()),
        "range_after": float(table["payout_r"].max() - table["payout_r"].min()),
        "std_before": float(table["payout"].std(ddof=0)), "std_after": float(table["payout_r"].std(ddof=0)),
        "pairs_sig_before": float((pairs["t_before"].abs() >= 2).mean()),
        "pairs_sig_after": float((pairs["t_after"].abs() >= 2).mean()),
        "rank_corr_before_after": float(stats.spearmanr(table["payout"], table["payout_r"])[0]),
        "seed_pairs": seed_floor[["a", "b", "d_before", "t_before", "d_after", "t_after"]].to_dict("records"),
        "pred_corr_before": float(np.mean(off(corrs["b"]))), "pred_corr_after": float(np.mean(off(corrs["a"]))),
        "gain_min": float(table["dpay"].min()), "gain_max": float(table["dpay"].max()),
        "models_gain_t2": int((table["t_dpay"] >= 2).sum()),
    }


def plot(table, pairs, conv, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"research": "#eb6834", "single": "#2a78d6", "blend": "#1baf7a"}
    t = table.sort_values("payout_r").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.27 * len(t) + 2.5)), facecolor=surface,
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.28, "left": 0.16,
                                          "right": 0.98, "top": 0.88, "bottom": 0.09})
    ax = axes[0]
    for i, r in t.iterrows():
        ax.plot([r["payout"], r["payout_r"]], [i, i], color=grid, linewidth=2, zorder=1)
        ax.scatter([r["payout"]], [i], s=28, facecolor=surface, edgecolor=color[r["group"]], linewidth=1.6, zorder=2)
        ax.scatter([r["payout_r"]], [i], s=36, color=color[r["group"]], edgecolor=surface, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([("$ " if r["staked"] else "") + r["model"] for _, r in t.iterrows()], fontsize=8)
    ax.set_xlabel("payout = 0.1·CORR + MMC, mean per day", color=muted)
    ax.set_title("A · Payout before (hollow) and after residualising (filled)", loc="left", fontsize=11,
                 fontweight="bold", color=ink, pad=24)
    ax.text(0, 1.01, "range %.4f → %.4f · std %.4f → %.4f · $ = staked" % (
        conv["range_before"], conv["range_after"], conv["std_before"], conv["std_after"]),
        transform=ax.transAxes, fontsize=8.5, color=muted)
    for g, c in color.items():
        ax.scatter([], [], s=36, color=c, label={"research": "ridge and networks", "single": "book: single model",
                                                  "blend": "book: blend"}[g])
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    bins = np.linspace(0, 8, 33)                 # |t|: which model of a pair is listed first is arbitrary
    ax.hist(pairs["t_before"].abs().clip(0, 8), bins=bins, color="#b4b2a9", alpha=0.8, label="before (s = 0)")
    ax.hist(pairs["t_after"].abs().clip(0, 8), bins=bins, histtype="step", color="#2a78d6", linewidth=2,
            label="after (s = 0.5, meta ≥ 5 days old)")
    ax.axvline(2, color=muted, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlabel("|t| of the paired payout difference of two models (NW, lag 20)", color=muted)
    ax.set_ylabel("model pairs", color=muted)
    ax.set_title("B · Do models still differ from each other?", loc="left", fontsize=11, fontweight="bold",
                 color=ink, pad=24)
    ax.text(0, 1.01, "pairs at |t| ≥ 2: %.0f%% → %.0f%% of %d · seed pairs after: %s" % (
        100 * conv["pairs_sig_before"], 100 * conv["pairs_sig_after"], conv["n_pairs"],
        ", ".join("t %+.1f" % p["t_after"] for p in conv["seed_pairs"])),
        transform=ax.transAxes, fontsize=8.5, color=muted)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    for a in axes:
        a.set_facecolor(surface)
        a.grid(color=grid, linewidth=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("E107 · Every model residualised on a 5-day-old meta model, s = 0.5 (%d days, %s)" % (
        int(table["n"].iloc[0]), "2024-07-25 → 2026-05-13"), x=0.01, ha="left", fontsize=13,
        fontweight="bold", color=ink)
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)


def html(table, conv, dropped, stored_gap, notes_path, png_path, out_path):
    import markdown
    show = table.copy()
    show["model"] = np.where(show["staked"], "$ " + show["model"], show["model"])
    cols = ["model", "group", "n", "CORR", "MMC", "payout", "calmar", "dd(pay)", "CORR_r", "MMC_r", "payout_r",
            "calmar_r", "dd(pay)_r", "burn(d)_r", "dpay", "t_dpay", "beta_median", "meta_corr_before",
            "meta_corr_after"]
    show = show.sort_values("payout_r", ascending=False)[cols]
    fmt = lambda x: ("%+.4f" % x) if isinstance(x, float) and abs(x) < 1 else ("%.2f" % x if isinstance(x, float)
                                                                               else str(x))
    table_html = show.to_html(index=False, border=0, formatters={c: fmt for c in cols if c not in ("model", "group")})
    notes = markdown.markdown(open(notes_path).read(), extensions=["tables"]) if notes_path else ""
    png = base64.b64encode(open(png_path, "rb").read()).decode()
    left = "".join("<li><code>%s</code>: %s</li>" % (n, why) for n, why in dropped)
    left += "".join("<li><code>%s</code> is in, but its stored per-period series is not reproduced by rescoring its "
                    "own stored forecast (per day up to %.1e on %d days, %+.1e on the mean); E107 uses the rescored "
                    "forecast throughout</li>" % (g["model"], g["max_day_diff"], g["days_differing"], g["mean_diff"])
                    for g in stored_gap)
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E107 residualise everything</title><style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--rule:#e4e3df}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37}}
:root[data-theme="dark"]{--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37}
body{background:var(--surface);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1240px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin-top:28px}h3{font-size:14px;margin:18px 0 6px}
.sub{color:var(--muted)}img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fcfcfb}
table{border-collapse:collapse;font:12px/1.4 ui-monospace,monospace;display:block;overflow-x:auto}
th,td{padding:3px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}
</style></head><body>
<h1>E107 · Every model residualised on a 5-day-old meta model, at s = 0.5</h1>
<p class="sub">per day: rank(gauss(rank(p)) − 0.5·β·gauss(rank(m))), m = the latest meta model at least 5 days old · scored against the same-day meta model · 2024-07-25 → 2026-05-13, the days every model shares · paired Newey-West lag 20</p>
@@NOTES@@
<h2>Before and after, every model</h2><img alt="E107 before and after" src="data:image/png;base64,@@PNG@@">
<h2>The table</h2>
<p class="sub">Columns without a suffix are the model unchanged; _r is residualised. dpay and t_dpay are the residualised payout minus the unchanged one, per day, paired. beta_median is the model's per-day projection on the lagged meta model; meta_corr is its mean per-day correlation with the same-day meta model, before and after. $ = staked. Notes on the sources:</p>
<ul>@@LEFT@@</ul>
@@TABLE@@
</body></html>"""
    page = (page.replace("@@NOTES@@", notes).replace("@@PNG@@", png).replace("@@LEFT@@", left)
            .replace("@@TABLE@@", table_html))
    with open(out_path, "w") as handle:
        handle.write(page)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--html", action="store_true", help="only rebuild the report from the saved tables")
    p.add_argument("--notes", default=None)
    args = p.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {k: os.path.join(OUT_DIR, k) for k in ("table.csv", "pairs.csv", "figure.png", "summary.json")}
    srcs, dropped = sources()

    if not args.html:
        days = common_days(srcs)
        print("%d sources; %d common days with a meta model, %s .. %s; left out: %s" % (
            len(srcs), len(days), days[0].date(), days[-1].date(), "; ".join("%s (%s)" % d for d in dropped)))
        with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            results = {r["slug"]: r for r in pool.map(process, [(s, days) for s in srcs])}
        worst, stored_gap = 0.0, []
        for s in srcs:
            tabs = results[s["slug"]]["tables"]
            for col in ("corr", "mmc"):          # gate: s = 0 is the plain scorer on the same forecast
                gap = float((tabs[0.0][col] - tabs["plain"][col]).abs().max())
                worst = max(worst, gap)
                if gap > 1e-9:
                    raise SystemExit("gate: %s %s at s=0 differs from the plain scorer by %g" % (s["label"], col, gap))
            stored = pd.read_csv(os.path.join("results", "per_period", "crypto", s["model"], s["run"] + ".csv"),
                                 index_col=0, parse_dates=True)
            d = (tabs["plain"]["corr"] - stored.loc[tabs["plain"].index, "corr"])
            if d.abs().max() > 1e-6:
                stored_gap.append({"model": s["label"], "max_day_diff": float(d.abs().max()),
                                   "days_differing": int((d.abs() > 1e-6).sum()), "mean_diff": float(d.mean())})
        print("gate: at s = 0 every source equals the plain scorer on its own forecast (max |diff| %.1e)" % worst)
        for g in stored_gap:
            print("  stored series not reproduced by its own forecast: %(model)s -- per day up to %(max_day_diff).1e "
                  "on %(days_differing)d days, mean %(mean_diff)+.1e" % g)
        names = [write_run(s, results[s["slug"]]) for s in srcs]
        print("wrote %d runs under results/*/crypto/%s/ (e107_*_s050)" % (len(names), TREE))
        idx = pd.DatetimeIndex(days)
        table, pairs = summarise(srcs, results, idx)
        corrs, n_rows = prediction_correlations(srcs, results)
        conv = convergence(table, pairs, corrs)
        conv["n_rows_correlation"] = n_rows
        table.to_csv(paths["table.csv"], index=False)
        pairs.to_csv(paths["pairs.csv"], index=False)
        corrs["b"].to_csv(os.path.join(OUT_DIR, "pred_corr_before.csv"))
        corrs["a"].to_csv(os.path.join(OUT_DIR, "pred_corr_after.csv"))
        with open(paths["summary.json"], "w") as handle:
            json.dump({"convergence": conv, "dropped": dropped, "days": [str(days[0].date()), str(days[-1].date())],
                       "n_days": len(days), "stored_series_not_reproduced": stored_gap}, handle, indent=1,
                      default=float)
    table = pd.read_csv(paths["table.csv"])
    pairs = pd.read_csv(paths["pairs.csv"])
    summary = json.load(open(paths["summary.json"]))
    conv, dropped = summary["convergence"], summary["dropped"]
    stored_gap = summary.get("stored_series_not_reproduced", [])
    plot(table, pairs, conv, paths["figure.png"])
    if args.html:
        out = os.path.join("results", "reports", "crypto", "e107_residualise_everything.html")
        html(table, conv, dropped, stored_gap, args.notes, paths["figure.png"], out)
        print("wrote %s" % out)
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_rows", 100):
        print(table.sort_values("payout_r", ascending=False)[
            ["model", "group", "staked", "CORR", "MMC", "payout", "calmar", "CORR_r", "MMC_r", "payout_r",
             "calmar_r", "dd(pay)_r", "dpay", "t_dpay", "beta_median", "meta_corr_before", "meta_corr_after"]
        ].round(4).to_string(index=False))
    print("\nconvergence: %s" % json.dumps(conv, indent=1, default=float))


if __name__ == "__main__":
    main()
