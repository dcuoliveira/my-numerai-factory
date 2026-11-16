"""
Experiment E106: residualise the -CORR network's predictions on the meta model.

    python3 experiments/e106_meta_residualise_crypto.py                # sweep + figure
    python3 experiments/e106_meta_residualise_crypto.py --plot-only    # redraw from the csv
    python3 experiments/e106_meta_residualise_crypto.py --html --notes <reading.md>

WHAT IT RUNS
  No training. The input is the stitched held-out forecast of the from-scratch -CORR
  network (identity 64-32-16, alpha 0, epoch 500, 12 full-layout refits;
  torch_scratch/torchscratch_corr_500ep_wf_s{42,43}_ep500, built by
  src.analysis.assemble_walkforward). On every day the meta model exists (547 days,
  2024-05-07 .. 2026-07-08), the prediction is moved away from the meta model:

      p = gauss(rank(prediction))         m = gauss(rank(meta model))
      beta = p.m / m.m                     (Numerai's orthogonalize, no intercept)
      r_s  = p - s * beta * m              s = 0, 0.05, ..., 1
      submitted = rank(r_s) in (0, 1)      (the rescale; the scorer re-ranks anyway)

  s = 0 is the network unchanged; s = 1 removes its whole linear projection on m.
  This is the space MMC itself is computed in (numerai_tools.correlation_contribution
  rank-gaussianises both and orthogonalises), so s = 1 hands the scorer a prediction
  whose meta component it has nothing left to subtract.

THE ORACLE PROBLEM, AND THE CONTROL FOR IT
  The meta model for a day is built from that day's submissions and published
  after the round closes, so residualising on the SAME day's meta model cannot be
  done live: `same` is an oracle, the upper bound. The deployable arms residualise
  on the most recent meta model at least L calendar days old (L = 1, 5, 20, 30;
  the latest meta date <= t - L, so a weekend or the June 2024 gap makes it
  staler). A symbol absent from that older cross-section gets m = 0 (no
  adjustment); a day with no meta model L days back is passed through unchanged
  (s acts as 0 there) and counted. Every arm is SCORED against the same-day meta
  model, which is what Numerai pays on.

WHAT IT IS MEASURED AGAINST
  s = 0 of the same seed, paired per day, Newey-West lag 20, on the 547 days. Gate:
  s = 0 must reproduce the source run's per-period CORR and MMC to 1e-12, and the
  local gauss-rank must equal numerai_tools' gaussian(tie_kept_rank(.)).

WHAT WOULD COUNT AS A RESULT
  A deployable (lagged) arm whose payout delta over s = 0 clears |t| >= 2 on BOTH
  seeds. The oracle's gain alone says only how much is there to take.

ARTEFACTS
  results/{forecast,per_period,metrics}/crypto/torch_scratch/e106_<variant>_s<SSS>_s<seed>
  (s = 0 once per seed as e106_base_s000_s<seed>; forecasts only for --forecast-at),
  results/reports/crypto/e106_meta_residualise/{sweep.csv,frontier.png}.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "target_binned_return_20"
TREE = "torch_scratch"
SOURCE = "torchscratch_corr_500ep_wf_s{seed}_ep500"
OUT_DIR = os.path.join("results", "reports", "crypto", "e106_meta_residualise")
VARIANTS = ["same", "lag1", "lag5", "lag20", "lag30"]       # order = colour order
LABELS = {"same": "same-day meta (oracle)", "lag1": "meta >= 1 day old",
          "lag5": "meta >= 5 days old", "lag20": "meta >= 20 days old",
          "lag30": "meta >= 30 days old"}
COLORS = {"same": "#2a78d6", "lag1": "#eb6834", "lag5": "#1baf7a",
          "lag20": "#eda100", "lag30": "#e87ba4"}


def run_name(variant, strength, seed):
    if strength == 0:
        return "e106_base_s000_s%d" % seed
    return "e106_%s_s%03d_s%d" % (variant, round(strength * 100), seed)


def gauss_rank(values):
    """gauss(tie_kept_rank(x)) on the non-missing entries, 0 where x is missing."""
    s = pd.Series(values)
    u = (s.rank(method="average") - 0.5) / s.count()
    return np.nan_to_num(stats.norm.ppf(u.to_numpy()), nan=0.0)


def load_source(seed):
    path = os.path.join("results", "forecast", "crypto", TREE, SOURCE.format(seed=seed) + ".csv")
    f = pd.read_csv(path, parse_dates=["date"])
    # The source run was scored with the target as float32 (val_rows.parquet); the CSV
    # reads it back as float64, which moves numerai_corr by ~4e-9 on identical ranks.
    f[TARGET] = f[TARGET].astype("float32")
    return f[f["meta_model"].notna()].sort_values(["date", "symbol"]).reset_index(drop=True)


def lagged_meta(frame, lag):
    """Per row: the symbol's meta model on the latest meta date <= date - lag days."""
    if lag == 0:
        return frame["meta_model"].to_numpy(), frame["date"].to_numpy()
    wide = frame.pivot(index="date", columns="symbol", values="meta_model")
    dates = wide.index.to_numpy()
    want = (frame["date"] - pd.Timedelta(days=lag)).to_numpy()
    pos = np.searchsorted(dates, want, side="right") - 1          # latest meta date <= want
    src = np.where(pos >= 0, dates[np.clip(pos, 0, None)], np.datetime64("NaT"))
    stacked = wide.stack().rename("m").reset_index()
    out = pd.DataFrame({"date": src, "symbol": frame["symbol"].to_numpy()}).merge(
        stacked, on=["date", "symbol"], how="left")
    return out["m"].to_numpy(), src


def residualise(frame, variant, strengths):
    """(rows x strengths) submitted predictions, plus per-day diagnostics."""
    lag = 0 if variant == "same" else int(variant[3:])
    m_raw, src = lagged_meta(frame, lag)
    out = np.empty((len(frame), len(strengths)))
    days, passed = [], 0
    for day, idx in frame.groupby("date").indices.items():
        p = gauss_rank(frame["prediction"].to_numpy()[idx])
        if pd.isna(src[idx[0]]):                 # no meta model that far back: unchanged
            passed += 1
            for j in range(len(strengths)):
                out[idx, j] = p
            continue
        m = gauss_rank(m_raw[idx])
        if m @ m == 0:                           # no symbol of today in that older day
            passed += 1
            for j in range(len(strengths)):
                out[idx, j] = p
            continue
        beta = (p @ m) / (m @ m)
        for j, s in enumerate(strengths):
            out[idx, j] = p - s * beta * m
        days.append({"date": day, "beta": beta, "stale_days": (day - pd.Timestamp(src[idx[0]])).days,
                     "symbols_missing": float(np.isnan(m_raw[idx]).mean())})
    for day, idx in frame.groupby("date").indices.items():          # the rescale
        for j in range(len(strengths)):
            r = pd.Series(out[idx, j])
            out[idx, j] = ((r.rank(method="average") - 0.5) / r.count()).to_numpy()
    return out, pd.DataFrame(days), passed


def score(task):
    """Worker: per-period table of one (seed, variant, strength) arm."""
    from src.prediction.metrics import per_period_table
    seed, variant, strength, pred = task
    frame = load_source(seed)
    frame["prediction"] = pred
    table = per_period_table(frame, "prediction", TARGET, "date",
                             reference_col="reference_prediction", meta_model_col="meta_model")
    spearman = frame.groupby("date")[["prediction", "meta_model"]].apply(
        lambda g: g["prediction"].rank().corr(g["meta_model"].rank()))
    return seed, variant, strength, table, float(spearman.mean())


def write_run(name, seed, variant, strength, table, frame, pred, write_forecast, extra):
    from src.prediction.metrics import summarize
    results = {"n_periods": int(table["corr"].notna().sum())}
    for family in ("corr", "bmc", "mmc"):
        results[family] = summarize(table[family])
        if family != "corr":
            results["%s_n_periods" % family] = int(table[family].notna().sum())
    results["config"] = {
        "model": TREE, "tournament": "crypto", "task": "regression", "study": "E106",
        "source_run": "%s/%s" % (TREE, SOURCE.format(seed=seed)), "seed": seed,
        "meta_variant": variant, "strength": strength, "target": TARGET,
        "transform": "rank(gauss(rank(p)) - s * beta * gauss(rank(m))), per day",
        "assembled_by": "experiments/e106_meta_residualise_crypto.py", **extra}
    paths = {}
    for kind in ("forecast", "per_period", "metrics"):
        directory = os.path.join("results", kind, "crypto", TREE)
        os.makedirs(directory, exist_ok=True)
        paths[kind] = os.path.join(directory, name + (".json" if kind == "metrics" else ".csv"))
    if write_forecast:
        f = frame[["symbol", "date", TARGET]].copy()
        f["prediction"] = pred
        f["reference_prediction"] = frame["reference_prediction"].to_numpy()
        f["meta_model"] = frame["meta_model"].to_numpy()
        f.to_csv(paths["forecast"], index=False)
    table.to_csv(paths["per_period"])
    with open(paths["metrics"], "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)


def gate(seeds):
    """Local gauss-rank == numerai_tools, and s = 0 reproduces the source run exactly."""
    from numerai_tools.scoring import gaussian, tie_kept_rank
    frame = load_source(seeds[0])
    day = frame[frame["date"] == frame["date"].iloc[0]].reset_index(drop=True)
    ref = gaussian(tie_kept_rank(day[["prediction"]]))["prediction"].to_numpy()
    assert np.allclose(gauss_rank(day["prediction"].to_numpy()), ref, atol=1e-12), "gauss_rank drifted"
    for seed in seeds:
        src = pd.read_csv(os.path.join("results", "per_period", "crypto", TREE,
                                       SOURCE.format(seed=seed) + ".csv"), index_col=0, parse_dates=True)
        mine = pd.read_csv(os.path.join("results", "per_period", "crypto", TREE,
                                        run_name("same", 0, seed) + ".csv"), index_col=0, parse_dates=True)
        for col in ("corr", "mmc"):
            gap = (mine[col] - src.loc[mine.index, col]).abs().max()
            assert gap < 1e-12, "seed %d %s: s=0 differs from the source by %g" % (seed, col, gap)
    print("gate: gauss_rank matches numerai_tools; s=0 reproduces the source per-period CORR/MMC (<1e-12)")


def summarise(seeds, strengths, spearman, passed):
    from src.analysis import results_table as rt
    from src.analysis.ablation_report import newey_west
    rows = []
    for seed in seeds:
        base = pd.read_csv(os.path.join("results", "per_period", "crypto", TREE,
                                        run_name("same", 0, seed) + ".csv"), index_col=0, parse_dates=True)
        base_pay = 0.1 * base["corr"] + base["mmc"]
        for variant in VARIANTS:
            for s in strengths:
                name = run_name(variant, s, seed)
                t = pd.read_csv(os.path.join("results", "per_period", "crypto", TREE, name + ".csv"),
                                index_col=0, parse_dates=True)
                r = rt.row_from_frame(name, t[["corr", "mmc"]])
                pay = 0.1 * t["corr"] + t["mmc"]
                d = {k: newey_west((x - y).loc[base.index].to_numpy(), lag=20) for k, (x, y) in {
                    "corr": (t["corr"], base["corr"]), "mmc": (t["mmc"], base["mmc"]),
                    "pay": (pay, base_pay)}.items()}
                rows.append({"seed": seed, "variant": variant, "strength": s, "run": name,
                             "n": int(t["mmc"].notna().sum()), "CORR": r["CORR"], "MMC": r["MMC"],
                             "payout": r["payout"], "c.sharpe": r["c.sharpe"], "p.sharpe": r["p.sharpe"],
                             "dd(pay)": r["dd(pay)"], "calmar": r["calmar"], "burn(d)": r["burn(d)"],
                             "dCORR": d["corr"][0], "t_dCORR": d["corr"][2], "dMMC": d["mmc"][0],
                             "t_dMMC": d["mmc"][2], "dpay": d["pay"][0], "se_dpay": d["pay"][1],
                             "t_dpay": d["pay"][2], "spearman_meta": spearman[(seed, variant, s)],
                             "days_unadjusted": passed[(seed, variant)]})
    return pd.DataFrame(rows)


def plot(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), facecolor=surface)
    seed = sweep["seed"].min()
    one = sweep[sweep["seed"] == seed]
    other = sweep[sweep["seed"] != seed]
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]

    ax = axes[0, 0]
    lo_m, hi_m = one["MMC"].min(), one["MMC"].max()
    lo_c, hi_c = one["CORR"].min(), one["CORR"].max()
    pad_m, pad_c = (hi_m - lo_m) * 0.08, (hi_c - lo_c) * 0.08
    xs = np.linspace(lo_m - pad_m, hi_m + pad_m, 50)
    top = hi_c + pad_c
    for level in np.arange(0.03, 0.20, 0.01):
        ax.plot(xs, (level - xs) / 0.1, color=grid, lw=1, zorder=0)
        x_top = level - 0.1 * top                 # where this iso-payout line meets the top
        if lo_m - pad_m < x_top < hi_m + pad_m:
            ax.text(x_top, top, " payout %.2f" % level, color=muted, fontsize=7.5, ha="left", va="top")
    ax.text(1.0, 1.035, "grey lines: equal payout", color=muted, fontsize=7.5, ha="right",
            transform=ax.transAxes)
    for v in VARIANTS:
        d = one[one["variant"] == v].sort_values("strength")
        ax.plot(d["MMC"], d["CORR"], color=COLORS[v], lw=2, label=LABELS[v])
        k = d[d["strength"].round(2).isin(marks)]
        ax.scatter(k["MMC"], k["CORR"], s=36, color=COLORS[v], edgecolor=surface, linewidth=2, zorder=3)
    d = one[one["variant"] == "same"]
    for s in marks:
        hit = d[np.isclose(d["strength"], s)]
        if not len(hit):
            continue
        row = hit.iloc[0]
        ax.annotate("s=%g" % s, (row["MMC"], row["CORR"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=muted)
    ax.set_xlim(lo_m - pad_m, hi_m + pad_m)
    ax.set_ylim(lo_c - pad_c, hi_c + pad_c)
    ax.set_xlabel("MMC (mean per day)")
    ax.set_ylabel("CORR (mean per day)")
    ax.set_title("A · The frontier: CORR against MMC as s goes 0 → 1")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax = axes[0, 1]
    for v in VARIANTS:
        d = one[one["variant"] == v].sort_values("strength")
        ax.plot(d["strength"], d["payout"], color=COLORS[v], lw=2, label=LABELS[v])
        e = other[other["variant"] == v].sort_values("strength")
        if len(e):
            ax.plot(e["strength"], e["payout"], color=COLORS[v], lw=1, ls=(0, (3, 2)))
    ax.set_xlabel("strength s  (0 = network unchanged, 1 = whole meta projection removed)")
    ax.set_ylabel("payout = 0.1·CORR + MMC (mean per day)")
    ax.set_title("B · Payout by strength  (solid seed %d, dashed seed %d)" % (seed, other["seed"].min())
                 if len(other) else "B · Payout by strength")

    ax = axes[1, 0]
    ax.axhspan(-2, 2, color=grid, alpha=0.6, lw=0)
    ax.axhline(0, color=muted, lw=0.8)
    for v in VARIANTS:
        d = one[one["variant"] == v].sort_values("strength")
        d = d[d["strength"] > 0]
        ax.plot(d["strength"], d["t_dpay"], color=COLORS[v], lw=2, label=LABELS[v])
        e = other[(other["variant"] == v) & (other["strength"] > 0)].sort_values("strength")
        if len(e):
            ax.plot(e["strength"], e["t_dpay"], color=COLORS[v], lw=1, ls=(0, (3, 2)))
    ax.text(0.02, 1.9, "|t| < 2", color=muted, fontsize=8, va="top", transform=ax.get_yaxis_transform())
    ax.set_xlabel("strength s")
    ax.set_ylabel("t of the payout gain over s = 0 (paired, Newey-West lag 20)")
    ax.set_title("C · Is the payout gain real?")

    ax = axes[1, 1]
    for v in VARIANTS:
        d = one[one["variant"] == v].sort_values("strength")
        ax.plot(d["strength"], d["spearman_meta"], color=COLORS[v], lw=2, label=LABELS[v])
    ax.axhline(0, color=muted, lw=0.8)
    ax.set_xlabel("strength s")
    ax.set_ylabel("per-day Spearman with the same-day meta model (mean)")
    ax.set_title("D · How much of the meta model is left in the prediction")

    for ax in axes.flat:
        ax.set_facecolor(surface)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("E106 · Residualising the −CORR network on the meta model "
                 "(547 meta days, 2024-05-07 → 2026-07-08)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def html(sweep, notes_path, png_path, out_path):
    import markdown
    seeds = sorted(sweep["seed"].unique())
    keep = sweep[sweep["strength"].round(2).isin([0.0, 0.25, 0.5, 0.75, 1.0])].copy()
    cols = ["variant", "strength", "n", "CORR", "MMC", "payout", "p.sharpe", "dd(pay)", "calmar",
            "burn(d)", "dCORR", "t_dCORR", "dMMC", "t_dMMC", "dpay", "t_dpay", "spearman_meta"]
    tables = ""
    for seed in seeds:
        t = keep[keep["seed"] == seed][cols].copy()
        t["variant"] = t["variant"].map(LABELS)
        tables += "<h3>Seed %d</h3>" % seed + t.to_html(index=False, float_format=lambda x: "%+.4f" % x
                                                        if abs(x) < 1 else "%.2f" % x, border=0)
    notes = markdown.markdown(open(notes_path).read(), extensions=["tables"]) if notes_path else ""
    png = base64.b64encode(open(png_path, "rb").read()).decode()
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E106 meta residualisation</title><style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--rule:#e4e3df}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37}}
:root[data-theme="dark"]{--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37}
body{background:var(--surface);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin-top:28px}h3{font-size:14px;margin:18px 0 6px}
.sub{color:var(--muted)}img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fcfcfb}
table{border-collapse:collapse;font:12px/1.4 ui-monospace,monospace;display:block;overflow-x:auto}
th,td{padding:3px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}
</style></head><body>
<h1>E106 · Residualising the −CORR network on the meta model</h1>
<p class="sub">torch_scratch/torchscratch_corr_500ep_wf_s{42,43}_ep500 · per day: rank(gauss(rank(p)) − s·β·gauss(rank(m))) · scored against the same-day meta model · paired Newey-West lag 20 against s = 0</p>
@@NOTES@@
<h2>The frontier</h2><img alt="E106 frontier" src="data:image/png;base64,@@PNG@@">
<h2>Sweep at s = 0, 0.25, 0.5, 0.75, 1</h2>
<p class="sub">Deltas and t are against s = 0 of the same seed. spearman_meta is the mean per-day rank correlation with the same-day meta model. The full 21-step sweep is in sweep.csv.</p>
@@TABLES@@
</body></html>"""
    page = page.replace("@@NOTES@@", notes).replace("@@PNG@@", png).replace("@@TABLES@@", tables)
    with open(out_path, "w") as handle:
        handle.write(page)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--steps", type=int, default=20, help="strengths 0..1 in this many steps")
    p.add_argument("--forecast-at", default="0.25,0.5,0.75,1",
                   help="strengths whose row-level forecast is written (seed 42 only); disk")
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--html", action="store_true")
    p.add_argument("--notes", default=None)
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    strengths = [round(i / args.steps, 4) for i in range(args.steps + 1)]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path, png_path = os.path.join(OUT_DIR, "sweep.csv"), os.path.join(OUT_DIR, "frontier.png")

    if not (args.plot_only or args.html):
        forecast_at = {float(x) for x in args.forecast_at.split(",")}
        tasks, frames, passed, diag = [], {}, {}, []
        for seed in seeds:
            frames[seed] = load_source(seed)
            for variant in VARIANTS:
                preds, days, n_pass = residualise(frames[seed], variant, strengths)
                passed[(seed, variant)] = n_pass
                days["seed"], days["variant"] = seed, variant
                diag.append(days)
                for j, s in enumerate(strengths):
                    if s == 0 and variant != "same":
                        continue                          # s = 0 is one run per seed
                    tasks.append((seed, variant, s, preds[:, j]))
                print("seed %d %-6s: %d days, %d passed through unadjusted, beta median %.3f, "
                      "meta staleness median %.0f d" % (seed, variant, frames[seed]["date"].nunique(),
                                                         n_pass, days["beta"].median(),
                                                         days["stale_days"].median()), flush=True)
        pd.concat(diag).to_csv(os.path.join(OUT_DIR, "per_day_beta.csv"), index=False)
        spearman = {}
        with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            for seed, variant, s, table, rho in pool.map(score, tasks, chunksize=2):
                pred = next(t[3] for t in tasks if t[:3] == (seed, variant, s))
                write_run(run_name(variant, s, seed), seed, variant, s, table, frames[seed], pred,
                          write_forecast=(seed == seeds[0] and (s in forecast_at or s == 0)),
                          extra={"days_unadjusted": passed[(seed, variant)]})
                for v in (VARIANTS if s == 0 else [variant]):
                    spearman[(seed, v, s)] = rho
        gate(seeds)
        sweep = summarise(seeds, strengths, spearman, passed)
        sweep.to_csv(csv_path, index=False)
        print("wrote %s (%d arms)" % (csv_path, len(sweep)))
    sweep = pd.read_csv(csv_path)
    plot(sweep, png_path)
    print("wrote %s" % png_path)
    if args.html:
        out = os.path.join("results", "reports", "crypto", "e106_meta_residualise.html")
        html(sweep, args.notes, png_path, out)
        print("wrote %s" % out)
    show = sweep[sweep["strength"].round(2).isin([0.0, 0.25, 0.5, 0.75, 1.0])]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(show[["seed", "variant", "strength", "CORR", "MMC", "payout", "dd(pay)", "calmar",
                    "burn(d)", "dCORR", "t_dCORR", "dMMC", "t_dMMC", "dpay", "t_dpay",
                    "spearman_meta"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
