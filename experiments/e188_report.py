"""E188 reports: per-run training pages, scaling-law fits and the cross-run report.

    python3 experiments/e188_report.py [--partial] [--boot 1000] [--rep-fold 3] [--rep-seed 42]

Writes into results/e188_iara_scaling/:
    epochs.parquet           every run's per-epoch metrics (deliverable 1)
    summary.parquet          one row per run (deliverable 2)
    size_points.parquet      one row per (loss, size): the scaling-law input (seed mean, folds pooled)
    fits.csv, ridge.csv      fitted laws with bootstrap CIs, ridge baselines
    e188_runs.pdf            one page per (size, loss), representative run (fold/seed below)
    e188_scaling.pdf         cross-run scaling-law report
--partial renders whatever has finished (marked PARTIAL on every page).

Scaling-law points: for each loss and size, each seed's retained-checkpoint test
predictions are scored on all 520 test eras (4 folds pooled); the point is the mean
over seeds and the seed band is their min..max. Compute on the x-axis is
6 P D E / 8.64e19 PetaFLOP-days with E the full epoch budget (fold-0 loss B trains on
D = 12,900 rows; the point uses the fold-mean D).
"""
import argparse
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import e188_iara_scaling as E  # noqa: E402

OUT = E.OUT
INK, MUTED, GRID = "#142b44", "#4e6378", "#dbe6ed"
LOSS_COLOR = {"A": "#1b4978", "B": "#147d73"}
LOSS_NAME = {"A": "MSE + L2", "B": "MMC"}
METRICS = {  # column, label, form: +1 means y = inf + A c^-a (decreasing), -1 means y = max - A c^-a
    "rmse": ("Test RMSE", +1),
    "r2": ("Test R² (vs 0.5)", -1),
    "payout": ("Test payout (0.1·CORR + MMC)", -1),
    "sharpe": ("Test era-Sharpe", -1),
}


# --------------------------------------------------------------------------- load

def load(partial):
    summaries = [json.loads(p.read_text()) for p in sorted(E.RUNS.glob("*/summary.json"))]
    if not summaries:
        raise SystemExit("no finished runs yet")
    s = pd.DataFrame(summaries)
    s.to_parquet(OUT / "summary.parquet", index=False)
    eras = pd.concat([pd.read_parquet(E.RUNS / i / "test_eras.parquet") for i in s.id], ignore_index=True)
    grid = s.loc[s.kind.isin(["A", "B"])]
    if not partial:
        expect = len(E.SIZES) * E.FOLDS * len(E.SEEDS)
        counts = grid.groupby("loss").size()
        assert counts.eq(expect).all() and len(counts) == 2, f"grid incomplete: {counts.to_dict()} of {expect}"
    return s, eras


def write_epochs(s):
    parts = [pd.read_parquet(E.RUNS / i / "epochs.parquet") for i in s.id]
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "epochs.parquet", index=False)


def size_points(s, eras):
    grid = s.loc[s.kind.isin(["A", "B"])]
    e = eras.merge(grid[["id", "kind"]], on="id")
    e["mse"] = e.rmse ** 2
    per_seed = e.groupby(["loss", "P", "seed"]).agg(
        payout=("payout", "mean"), corr=("corr", "mean"), mmc=("mmc", "mean"), cwmm=("cwmm", "mean"),
        mse=("mse", "mean"), pay_std=("payout", lambda x: x.std(ddof=0)), n_eras=("payout", "size"),
        n_folds=("fold", "nunique")).reset_index()
    per_seed["rmse"] = np.sqrt(per_seed.mse)
    per_seed["sharpe"] = per_seed.payout / per_seed.pay_std
    # R^2 vs 0.5 pooled: 1 - SSE / SS0; per-era rmse^2 * 300 = SSE_era. SS0 from the cached targets.
    ss0 = sum(float(np.sum((E.load_fold(f)["test_y"] - .5) ** 2)) for f in range(E.FOLDS))
    per_seed["r2"] = 1 - per_seed.mse * E.ERA * 520 / ss0
    meta = grid.groupby(["loss", "P"]).agg(w=("w", "first"), epochs=("epochs", "first"),
                                            steps=("steps_per_epoch", "max"), D=("n_rows_train", "mean"),
                                            retained_epoch=("retained_epoch", "median"),
                                            retained_min=("retained_epoch", "min"),
                                            retained_max=("retained_epoch", "max"),
                                            runs=("id", "size")).reset_index()
    meta["c"] = 6 * meta.P * meta.D * meta.epochs / 8.64e19
    meta["P_over_N"] = meta.P / 39000
    agg = per_seed.groupby(["loss", "P"]).agg(
        **{f"{m}": (m, "mean") for m in ("rmse", "r2", "payout", "sharpe", "corr", "mmc", "cwmm")},
        **{f"{m}_lo": (m, "min") for m in ("rmse", "r2", "payout", "sharpe")},
        **{f"{m}_hi": (m, "max") for m in ("rmse", "r2", "payout", "sharpe")},
        seeds=("seed", "nunique"), complete=("n_folds", "min")).reset_index()
    pts = meta.merge(agg, on=["loss", "P"])
    pts.to_parquet(OUT / "size_points.parquet", index=False)
    return pts, per_seed


# --------------------------------------------------------------------------- fits

ALPHAS = np.logspace(-3, 0.7, 500)


def weights(c):
    lc = np.log(c)
    w = lc - lc.min() + 1
    return w / w.sum()


def varpro(c, y, form):
    """y = inf + form * A * c^-a with A >= 0: closed-form WLS in (inf, A) per a, 1-D search over a."""
    w = weights(c)
    best = None
    for a in ALPHAS:
        x = c ** (-a)
        X = np.column_stack([np.ones_like(x), form * x])
        W = np.sqrt(w)
        theta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        if theta[1] < 0:                                   # wrong-sign curvature: constant fit
            theta = np.array([np.sum(w * y), 0.0])
        sse = float(np.sum(w * (y - X @ theta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, theta)
    sse, a, (inf, A) = best
    ybar = np.sum(w * y)
    r2 = 1 - sse / float(np.sum(w * (y - ybar) ** 2))
    c0 = A ** (1 / a) if A > 0 else np.nan
    return dict(asymptote=inf, alpha=a, A=A, c0=c0, r2=r2)


def bootstrap(c, y, form, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c))
        if len(np.unique(c[i])) < 4:
            continue
        out.append(varpro(c[i], y[i], form))
    return pd.DataFrame(out)


def two_sided(draws, null=0.0):
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    return float(min(1.0, 2 * min((draws <= null).mean(), (draws >= null).mean())))


def fit_all(pts, n_boot):
    rows, boots = [], {}
    for loss in ("A", "B"):
        p = pts.loc[pts.loss.eq(loss)].sort_values("c")
        if p.P.nunique() < 5:
            continue
        for m, (label, form) in METRICS.items():
            c, y = p.c.to_numpy(), p[m].to_numpy()
            est = varpro(c, y, form)
            b = bootstrap(c, y, form, n_boot)
            boots[(loss, m)] = b
            rec = dict(loss=loss, metric=m, n_points=len(p), **est)
            for k in ("asymptote", "alpha", "A"):
                lo, hi = np.nanpercentile(b[k], [2.5, 97.5])
                rec[f"{k}_lo"], rec[f"{k}_hi"] = lo, hi
                rec[f"{k}_p"] = two_sided(b[k])
            rows.append(rec)
    fits = pd.DataFrame(rows, columns=['loss', 'metric', 'n_points', 'asymptote', 'alpha', 'A', 'c0', 'r2'] +
                        [f'{k}_{q}' for k in ('asymptote', 'alpha', 'A') for q in ('lo', 'hi', 'p')]) if not rows else pd.DataFrame(rows)
    diffs = []
    for m in METRICS:
        if ("A", m) in boots and ("B", m) in boots:
            for k in ("asymptote", "alpha"):
                a_, b_ = boots[("A", m)][k].to_numpy(), boots[("B", m)][k].to_numpy()
                n = min(len(a_), len(b_))
                d = b_[:n] - a_[:n]
                est = fits.set_index(["loss", "metric"])[k]
                diffs.append(dict(metric=m, param=k, B_minus_A=est[("B", m)] - est[("A", m)],
                                  lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=two_sided(d)))
    return fits, pd.DataFrame(diffs), boots


# --------------------------------------------------------------------------- ridge

def ridge_baseline():
    from sklearn.linear_model import Ridge, LinearRegression
    rows = []
    for lam in (0, 1e-2, 1e-1, 1, 10, 1e2, 1e3):
        pay, mse, n = [], [], 0
        for fold in range(E.FOLDS):
            f = E.load_fold(fold)
            mu, sd = f["train_x"].mean(0), f["train_x"].std(0)
            sd[sd == 0] = 1
            x, xt = (f["train_x"] - mu) / sd, (f["test_x"] - mu) / sd
            model = LinearRegression() if lam == 0 else Ridge(alpha=lam)
            model.fit(x, f["train_y"] - .5)
            pred = model.predict(xt)
            k = len(pred) // E.ERA
            _, _, p_, _ = E.era_scores(pred.reshape(k, E.ERA), f["test_y"].reshape(k, E.ERA),
                                       f["test_meta"].reshape(k, E.ERA))
            pay.append(p_)
            mse.append((pred - (f["test_y"] - .5)) ** 2)
        pay = np.concatenate(pay)
        mse = np.concatenate(mse)
        rows.append(dict(penalty=lam, rmse=float(np.sqrt(mse.mean())), payout=float(pay.mean()),
                         sharpe=float(pay.mean() / pay.std(ddof=0))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- pages

def page(title, subtitle, partial, number=None):
    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    fig.text(.05, .955, "NUMERAI CRYPTO  ·  E188" + ("  ·  PARTIAL" if partial else ""), color="#147d73",
             fontsize=10, weight="bold")
    fig.text(.05, .915, title, color=INK, fontsize=20, weight="bold")
    fig.text(.05, .885, subtitle, color=MUTED, fontsize=9.5)
    if number:
        fig.text(.95, .02, number, color=MUTED, fontsize=8, ha="right")
    return fig


def style(ax, title, xlog=True, ylog=False):
    ax.set_title(title, loc="left", fontsize=8.5, color=INK, weight="bold")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(alpha=.35, color=GRID)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)


def run_page(pdf, run_id, partial, number):
    d = E.RUNS / run_id
    s = json.loads((d / "summary.json").read_text())
    ep = pd.read_parquet(d / "epochs.parquet")
    dg = pd.read_parquet(d / "diag.parquet")
    loss = s["loss"]
    fig = page(f"{LOSS_NAME[loss]} · P = {s['P']:,} (w = {s['w']})",
               f"{run_id} · {s['epochs']:,} epochs × {s['steps_per_epoch']} steps · {s['pflop_days']:.2e} PFLOP-days · "
               f"P/N = {s['P_over_N']:.3f} · retained epoch {s['retained_epoch']} · test payout {s['test_payout']:.4f}",
               partial, number)
    axes = [fig.add_axes([.05 + .235 * (k % 4), .63 - .265 * (k // 4), .19, .19]) for k in range(12)]
    ret = ep.loc[ep.epoch.eq(s["retained_epoch"])].iloc[0]
    x = ep.cum_pflop_days
    # 1 loss vs compute
    ax = axes[0]
    ax.plot(x, ep.train_data_loss, color=LOSS_COLOR[loss], lw=1, label="train")
    val = ep.val_mse if loss == "A" else -ep.val_mmc
    ax.plot(x, val, color="#a6742a", lw=1, label="validation")
    ax.axvline(ret.cum_pflop_days, color="#c0392b", lw=.8, ls="--")
    style(ax, "1 · Loss vs compute (" + ("MSE" if loss == "A" else "−MMC") + ")")
    ax.legend(fontsize=6, frameon=False)
    # 2 metric decomposition
    ax = axes[1]
    for col, c_ in (("corr", "#1b4978"), ("mmc", "#147d73"), ("cwmm", "#a6742a")):
        ax.plot(x, ep[f"val_{col}"], color=c_, lw=1, label=f"val {col.upper()}")
        if f"train_{col}" in dg:
            ax.plot(dg.cum_pflop_days, dg[f"train_{col}"], color=c_, lw=.8, ls=":")
    style(ax, "2 · CORR/MMC/CWMM (dotted: train)")
    ax.legend(fontsize=5.5, frameon=False)
    # 3 per-era validation distribution
    ax = axes[2]
    ax.plot(x, ep.val_payout, color=INK, lw=1, label="mean")
    ax.plot(x, ep.val_era_std, color=MUTED, lw=1, label="std")
    ax2 = ax.twinx()
    ax2.plot(x, ep.val_hit, color="#147d73", lw=.8, label="hit rate")
    ax2.plot(x, ep.val_sharpe, color="#a6742a", lw=.8, label="era-Sharpe")
    ax2.tick_params(labelsize=6, colors=MUTED)
    style(ax, "3 · Val eras: mean, std | hit, Sharpe")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax2.legend(fontsize=5.5, frameon=False, loc="lower right")
    # 4 train saturation. Loss A: train R^2 / Pearson -> 1 means interpolation. Loss B is scale-free, so R^2
    # means nothing there; its saturation measure is the train CORR and MMC.
    ax = axes[3]
    tail = dg.loc[dg.epoch >= .8 * s["epochs"]]
    if loss == "A":
        series = (("train_r2", "train R²", LOSS_COLOR["A"]), ("train_pearson", "train Pearson", "#a6742a"))
    else:
        series = (("train_corr", "train CORR", LOSS_COLOR["B"]), ("train_mmc", "train MMC", "#a6742a"))
    for col, lab, colr in series:
        ax.plot(dg.epoch, dg[col], color=colr, lw=1, label=lab)
    ax.axvspan(.8 * s["epochs"], s["epochs"], color=GRID, alpha=.6)
    key = series[0][0]
    interp = bool(len(tail) and (tail[key].max() > .99 or (loss == "A" and tail.train_pearson.max() > .99)))
    style(ax, "4 · Train saturation (final 20% shaded)")
    msg = "no tail diagnostics" if not len(tail) else ("INTERPOLATED" if interp else
          f"not interpolated: final-20% {series[0][1]} ≤ {tail[key].max():.3f}")
    ax.text(.02, .92, msg, transform=ax.transAxes, fontsize=6, color="#c0392b" if interp else INK)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    # 5 gradients
    ax = axes[4]
    ax.fill_between(ep.epoch, ep.grad_norm_min, ep.grad_norm_max, color=GRID, lw=0)
    ax.plot(ep.epoch, ep.grad_norm_mean, color=INK, lw=.8)
    style(ax, "5a · Gradient L2, batch band", ylog=True)
    ax = axes[5]
    for i in range(6):
        if f"update_ratio_{i}" in dg:
            ax.plot(dg.step, dg[f"update_ratio_{i}"], lw=.8, label=f"L{i}")
    ax.axhline(1e-3, color="#c0392b", lw=.6, ls="--")
    style(ax, "5b · ‖ΔW‖/‖W‖ per layer", ylog=True)
    ax.legend(fontsize=5, frameon=False, ncol=3)
    # 6 weight norms
    ax = axes[6]
    total = np.zeros(len(dg))
    for i in range(6):
        if f"wnorm_{i}" in dg:
            ax.plot(dg.step, dg[f"wnorm_{i}"], lw=.8, label=f"L{i}")
            total += dg[f"wnorm_{i}"].to_numpy() ** 2
    rising = len(dg) > 5 and pd.Series(total).is_monotonic_increasing
    flat = (ep.val_score.iloc[len(ep) // 2:].std() < .1 * max(abs(ep.val_score).max(), 1e-9))
    style(ax, "6 · Weight norm per layer vs step", ylog=True)
    if rising and flat:
        ax.text(.02, .9, "FLAG: monotone ‖W‖ growth with flat metric", transform=ax.transAxes, fontsize=6,
                color="#c0392b")
    # 7 prediction spread
    ax = axes[7]
    ax.plot(ep.step, ep.val_pred_std, color=INK, lw=1, label="std")
    ax.plot(ep.step, ep.val_pred_era_std, color="#a6742a", lw=1, label="mean per-era std")
    style(ax, "7 · Prediction spread (validation)", ylog=True)
    ax.legend(fontsize=5.5, frameon=False)
    # 8 churn
    ax = axes[8]
    ax.plot(ep.epoch, ep.churn, color=INK, lw=.8)
    style(ax, "8 · Churn: rank corr, consecutive epochs")
    # 9 estimability
    ax = axes[9]
    est_path = d / "estimability.parquet"
    if loss == "B" and est_path.exists():
        es = pd.read_parquet(est_path)
        sc = ax.scatter(es.full_epoch_mmc, es.batch_mmc, c=np.log10(es.epoch), s=4, cmap="viridis")
        lim = [min(es.full_epoch_mmc.min(), es.batch_mmc.min()), max(es.full_epoch_mmc.max(), es.batch_mmc.max())]
        ax.plot(lim, lim, color=MUTED, lw=.6)
        g = es.groupby("epoch").grad_cosine.first()
        ax.text(.02, .9, f"grad cosine (2 batches): median {g.median():.2f}", transform=ax.transAxes, fontsize=6)
        style(ax, "9 · Batch vs full-epoch MMC", xlog=False)
    else:
        ax.text(.5, .5, "loss A: not applicable", ha="center", fontsize=7, color=MUTED, transform=ax.transAxes)
        style(ax, "9 · Loss estimability", xlog=False)
    # 10 snapshots
    snap_path = d / "snapshots.parquet"
    if snap_path.exists():
        sn = pd.read_parquet(snap_path)
        steps = sorted(sn.step.unique())
        pick = [steps[0], steps[len(steps) // 2], steps[-1]] if len(steps) >= 3 else steps
        for k, (ax, col) in enumerate(((axes[10], "target"), (axes[11], "meta"))):
            for j, st in enumerate(pick):
                q = sn.loc[sn.step.eq(st)]
                z = (q.pred - q.pred.mean()) / (q.pred.std() + 1e-12)
                ax.scatter(q[col] + (j - 1) * (.05 if col == "target" else 0), z + 4 * j, c=q.era, s=1.2,
                           cmap="tab10", vmin=0, vmax=30)
                ax.text(1.0, 4 * j, f"step {st}", fontsize=5, color=MUTED, transform=ax.get_yaxis_transform())
            style(ax, f"10 · Snapshots: pred vs {col}", xlog=False)
            ax.set_yticks([])
    pdf.savefig(fig)
    plt.close(fig)
    return interp


def scaling_pages(pdf, pts, per_seed, fits, diffs, ridge, partial, s):
    sub = (f"{pts.groupby('loss').P.nunique().to_dict()} sizes · points = seed mean of retained-checkpoint test "
           f"metrics over 520 eras · bars = seed range · compute = full epoch budget")
    # scaling-law figure
    fig = page("Scaling laws: retained checkpoint vs compute", sub, partial, "1")
    for i, (m, (label, form)) in enumerate(METRICS.items()):
        for j, loss in enumerate(("A", "B")):
            ax = fig.add_axes([.06 + .235 * i, .5 - .4 * j, .19, .3])
            p = pts.loc[pts.loss.eq(loss)].sort_values("c")
            if p.empty:
                continue
            ax.errorbar(p.c, p[m], yerr=[p[m] - p[f"{m}_lo"], p[f"{m}_hi"] - p[m]], fmt="none", ecolor=GRID, lw=.8)
            sc = ax.scatter(p.c, p[m], c=np.log10(p.P), cmap="viridis", s=12, zorder=3)
            f = fits.loc[fits.loss.eq(loss) & fits.metric.eq(m)]
            if len(f):
                f = f.iloc[0]
                cc = np.logspace(np.log10(p.c.min()), np.log10(p.c.max()), 200)
                ax.plot(cc, f.asymptote + form * f.A * cc ** (-f.alpha), color="#c0392b", lw=1)
                ax.axhline(f.asymptote, color="#c0392b", lw=.6, ls=":")
                ax.text(.02, .03, f"asym {f.asymptote:.4f} α {f.alpha:.2f} R² {f.r2:.2f}", transform=ax.transAxes,
                        fontsize=6.5)
            if m in ("rmse", "payout", "sharpe") and len(ridge):
                best = ridge.loc[ridge.payout.idxmax()]
                ax.axhline(best[m], color=MUTED, lw=.7, ls="--")
            style(ax, f"{LOSS_NAME[loss]} · {label}")
            ax.set_xlabel("PetaFLOP-days", fontsize=7, color=MUTED)
    cb = fig.add_axes([.955, .15, .01, .6])
    fig.colorbar(sc, cax=cb).set_label("log10 P", fontsize=7)
    fig.text(.06, .045, "Red: fitted law (dotted = asymptote). Grey dashed: best ridge (by test payout). "
             "Fit: variable projection, log-compute weights.", fontsize=8, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)

    # seed band / training risk + double descent view
    fig = page("Training risk and double descent", "Seed spread per size and test RMSE against P/N", partial, "2")
    for j, loss in enumerate(("A", "B")):
        p = pts.loc[pts.loss.eq(loss)].sort_values("P")
        if p.empty:
            continue
        ax = fig.add_axes([.06 + .47 * j, .5, .4, .3])
        ax.fill_between(p.c, p.payout_lo, p.payout_hi, color=LOSS_COLOR[loss], alpha=.2, lw=0)
        ax.plot(p.c, p.payout, color=LOSS_COLOR[loss], lw=1.2)
        style(ax, f"{LOSS_NAME[loss]} · test payout, seed mean and range")
        ax = fig.add_axes([.06 + .47 * j, .1, .4, .3])
        ax.fill_between(p.P_over_N, p.rmse_lo, p.rmse_hi, color=LOSS_COLOR[loss], alpha=.2, lw=0)
        ax.plot(p.P_over_N, p.rmse, color=LOSS_COLOR[loss], lw=1.2, marker="o", ms=2.5)
        for v in (1, 10):
            ax.axvline(v, color="#c0392b", lw=.6, ls="--")
        style(ax, f"{LOSS_NAME[loss]} · test RMSE vs P/N (dashed at 1 and 10)")
    pdf.savefig(fig)
    plt.close(fig)

    # tables
    for loss in ("A", "B"):
        p = pts.loc[pts.loss.eq(loss)].sort_values("P")
        if p.empty:
            continue
        fig = page(f"Per-size table · {LOSS_NAME[loss]}", "Seed mean over 5 seeds, 4 folds pooled (520 test eras)",
                   partial, "3" if loss == "A" else "4")
        cols = ["P", "w", "epochs", "steps", "c", "P_over_N", "retained_epoch", "rmse", "payout", "sharpe", "seeds"]
        t = p[cols].copy()
        t["c"] = t.c.map(lambda v: f"{v:.2e}")
        t["P_over_N"] = t.P_over_N.map(lambda v: f"{v:.3f}")
        for c_ in ("rmse", "payout", "sharpe"):
            t[c_] = t[c_].map(lambda v: f"{v:.4f}")
        t["retained_epoch"] = [f"{int(a)} [{int(lo)}–{int(hi)}]" for a, lo, hi in
                               zip(p.retained_epoch, p.retained_min, p.retained_max)]
        ax = fig.add_axes([.05, .05, .9, .8])
        ax.axis("off")
        tab = ax.table(cellText=t.values, colLabels=["P", "w", "epochs", "steps/ep", "PFLOP-days", "P/N",
                                                     "retained epoch (med [min–max])", "test RMSE", "test payout",
                                                     "era-Sharpe", "seeds"], loc="upper center", cellLoc="right")
        tab.auto_set_font_size(False)
        tab.set_fontsize(6.8)
        tab.scale(1, 1.05)
        pdf.savefig(fig)
        plt.close(fig)

    # fits, A vs B, ridge
    fig = page("Fitted parameters, A vs B, ridge baselines", "95% percentile intervals from a pairs bootstrap over "
               "size points; p-values two-sided against zero (A vs B: against no difference)", partial, "5")
    ax = fig.add_axes([.05, .5, .9, .35])
    ax.axis("off")
    if len(fits):
        t = fits.apply(lambda r: [LOSS_NAME[r.loss], METRICS[r.metric][0], f"{r.asymptote:.4f} [{r.asymptote_lo:.4f}, "
                                  f"{r.asymptote_hi:.4f}]", f"{r.alpha:.3f} [{r.alpha_lo:.3f}, {r.alpha_hi:.3f}]",
                                  f"{r.A:.3g}", f"{r.c0:.3g}", f"{r.r2:.3f}", f"{r.asymptote_p:.3f}"], axis=1)
        tab = ax.table(cellText=list(t), colLabels=["loss", "metric", "asymptote [95%]", "alpha [95%]", "A", "c0",
                                                    "fit R²", "p(asym)"], loc="upper center", cellLoc="right")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)
    ax = fig.add_axes([.05, .27, .42, .2])
    ax.axis("off")
    if len(diffs):
        t = diffs.apply(lambda r: [METRICS[r.metric][0], r.param, f"{r.B_minus_A:+.4f}", f"[{r.lo:+.4f}, {r.hi:+.4f}]",
                                   f"{r.p:.3f}"], axis=1)
        tab = ax.table(cellText=list(t), colLabels=["metric", "param", "B − A", "95%", "p"], loc="upper center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)
    ax = fig.add_axes([.53, .27, .42, .2])
    ax.axis("off")
    if len(ridge):
        t = ridge.apply(lambda r: [f"{r.penalty:g}", f"{r.rmse:.4f}", f"{r.payout:.4f}", f"{r.sharpe:.3f}"], axis=1)
        tab = ax.table(cellText=list(t), colLabels=["ridge penalty", "test RMSE", "test payout", "era-Sharpe"],
                       loc="upper center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)
    pdf.savefig(fig)
    plt.close(fig)


def sanity(s, pts):
    grid = s.loc[s.kind.isin(["A", "B"])]
    checks = []
    checks.append(("steps_per_epoch >= 32 for every run", bool(grid.steps_per_epoch.ge(32).all()),
                   f"min {grid.steps_per_epoch.min()}"))
    final = grid.retained_epoch.eq(grid.epochs)
    first = grid.retained_epoch.le(3)
    checks.append(("retained checkpoint not always the final epoch", not final.all(),
                   f"{final.mean():.0%} of runs retain the final epoch"))
    checks.append(("retained checkpoint not always epoch 1-3 (first smoothable)", not first.all(),
                   f"{first.mean():.0%} of runs retain epoch <= 3"))
    span = np.log10(pts.c.max() / pts.c.min()) if len(pts) else 0
    checks.append(("compute spans >= 4 orders of magnitude", span >= 4, f"{span:.1f} decades"))
    big = sorted(pts.loc[pts.P_over_N >= 1, "P"].unique())
    near10 = sorted(pts.loc[pts.P_over_N.between(5, 20), "P"].unique())
    checks.append(("at least one model with P/N >= 1", len(big) > 0, f"P/N>=1 from P={big[0]:,}" if big else "none"))
    checks.append(("note: models near P/N ~ 10", True, ", ".join(f"{p:,}" for p in near10) or "none"))
    leak = []
    for fold in range(E.FOLDS):
        f = E.load_fold(fold)
        tr, va, te = (set(f[f"{k}_dates"]) for k in ("train", "val", "test"))
        leak.append(bool(tr & va or tr & te or va & te))
    checks.append(("no era in both a training fold and its own validation/test window", not any(leak), "per fold"))
    return pd.DataFrame(checks, columns=["check", "pass", "detail"])


def sanity_page(pdf, checks, alpha, partial):
    fig = page("Sanity checks", "Section 10 of the spec; results are only trusted when all pass", partial, "0")
    ax = fig.add_axes([.05, .35, .9, .5])
    ax.axis("off")
    tab = ax.table(cellText=[[c, "PASS" if p else "FAIL", d] for c, p, d in checks.values],
                   colLabels=["check", "result", "detail"], loc="upper center", cellLoc="left",
                   colWidths=[.5, .1, .4])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    fig.text(.05, .3, textwrap.fill("Loss A alpha: " + json.dumps(alpha), 140), fontsize=8.5, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rep-fold", type=int, default=3)
    ap.add_argument("--rep-seed", type=int, default=42)
    ap.add_argument("--skip-runs", action="store_true", help="skip the per-run pages")
    args = ap.parse_args()
    s, eras = load(args.partial)
    write_epochs(s)
    pts, per_seed = size_points(s, eras)
    fits, diffs, _ = fit_all(pts, args.boot)
    fits.to_csv(OUT / "fits.csv", index=False)
    diffs.to_csv(OUT / "fits_A_vs_B.csv", index=False)
    ridge = ridge_baseline()
    ridge.to_csv(OUT / "ridge.csv", index=False)
    # One-sided: does each financial asymptote exceed the best ridge?
    checks = sanity(s, pts)
    alpha_path = OUT / "alpha.json"
    alpha = json.loads(alpha_path.read_text()) if alpha_path.exists() else "not chosen yet"
    with PdfPages(OUT / "e188_scaling.pdf") as pdf:
        sanity_page(pdf, checks, alpha, args.partial)
        scaling_pages(pdf, pts, per_seed, fits, diffs, ridge, args.partial, s)
    if not args.skip_runs:
        grid = s.loc[s.kind.isin(["A", "B"]) & s.fold.eq(args.rep_fold) & s.seed.eq(args.rep_seed)]
        with PdfPages(OUT / "e188_runs.pdf") as pdf:
            for k, (_, r) in enumerate(grid.sort_values(["loss", "P"]).iterrows(), 1):
                run_page(pdf, r.id, args.partial, f"{k} / {len(grid)}")
    print(checks.to_string(index=False))
    if len(fits):
        print(fits[["loss", "metric", "asymptote", "asymptote_lo", "asymptote_hi", "alpha", "alpha_lo", "alpha_hi",
                    "r2"]].round(5).to_string(index=False))
    print(ridge.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
