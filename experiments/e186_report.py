"""E186 report: torch MLP on an MMC loss vs the sklearn recipes of E185.

    python3 experiments/e186_report.py

Reads results/e186_ia_torch_mmc/ and results/ia_mlp_10x/daily_scores.csv (E185) and
writes results/e186_ia_torch_mmc/e186_report.pdf plus comparison.csv.

Torch arms carry five seeds. Their row in a table is the mean over seeds of each
date's score (seed-averaged scores, NOT a seed-averaged prediction); the seed range
of the 520-date mean payout is reported beside it. Paired intervals are 40-date
circular block bootstraps (5,000 draws) on the per-date delta, as in E185.
"""
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.volatility_proxy_report import table, max_drawdown  # noqa: E402

OUT = ROOT / "results" / "e186_ia_torch_mmc"
E185 = ROOT / "results" / "ia_mlp_10x" / "daily_scores.csv"
PDF = OUT / "e186_report.pdf"
MODELS = ("IA_01", "IA_02", "IA_03", "IA_04")
ARMS = ("before", "reference", "torch_mse", "torch_mmc")
LABELS = dict(zip(ARMS, ("sklearn, current recipe", "sklearn, TV epochs, MSE",
                         "torch, TV epochs, MSE", "torch, TV epochs, MMC")))
COLORS = dict(zip(ARMS, ("#66788d", "#9c6b24", "#1b4978", "#147d73")))
INK, MUTED, GRID = "#142b44", "#4e6378", "#dbe6ed"
PAGES = 4


def page(number, heading, subtitle):
    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    fig.text(.07, .94, "NUMERAI CRYPTO  ·  E186", color=COLORS["torch_mmc"], fontsize=11, weight="bold")
    fig.text(.07, .881, heading, color=INK, fontsize=27, weight="bold")
    note(fig, .837, subtitle)
    fig.add_artist(plt.Line2D([.07, .93], [.06, .06], color=GRID, lw=1))
    fig.text(.07, .032, "Torch MLP, sklearn init, MMC loss, Timmermann-Vulicevic epochs", color=MUTED, fontsize=9)
    fig.text(.93, .032, f"{number} / {PAGES}", color=MUTED, fontsize=9, ha="right")
    return fig


def note(fig, y, text, title=None, width=120):
    if title:
        fig.text(.07, y, title, fontsize=13, weight="bold", color=INK, va="top")
        y -= .039
    fig.text(.07, y, textwrap.fill(text, width), fontsize=10.5, color=MUTED, va="top", linespacing=1.45)


def interval(delta, block=40):
    rng = np.random.default_rng(42)
    values = np.asarray(delta.dropna())
    n = len(values)
    starts = rng.integers(0, n, (5000, int(np.ceil(n / block)), 1))
    idx = ((starts + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
    return np.quantile(values[idx].mean(1), [.025, .975])


def load():
    torch = pd.read_csv(OUT / "daily_scores.csv", parse_dates=["date"])
    seeds = torch.seed.nunique()
    counts = torch.groupby(["model", "arm", "seed"]).size()
    assert counts.eq(520).all() and len(counts) == len(MODELS) * 2 * seeds, counts
    sk = pd.read_csv(E185, parse_dates=["date"])
    sk = sk.loc[sk.arm.isin(["before", "reference"])]
    metrics = ["corr", "mmc", "payout", "mse"]
    per_seed = torch.groupby(["model", "arm", "seed"])[metrics].mean()
    torch_mean = torch.groupby(["model", "date", "arm"], as_index=False)[metrics].mean()
    daily = pd.concat([sk[["model", "date", "arm", *metrics]], torch_mean], ignore_index=True)
    # Paired: every arm on the same 520 dates for each model.
    assert daily.groupby(["model", "arm"]).size().eq(520).all()
    assert daily.groupby("model").date.nunique().eq(520).all()
    return daily, per_seed, torch, seeds


def compare(daily, per_seed):
    rows = []
    for name in MODELS:
        wide = {m: daily.loc[daily.model.eq(name)].pivot(index="date", columns="arm", values=m)
                for m in ("corr", "mmc", "payout", "mse")}
        for arm in ARMS:
            pay = wide["payout"][arm]
            rec = dict(model=name, arm=arm, corr=wide["corr"][arm].mean(), mmc=wide["mmc"][arm].mean(),
                       payout=pay.mean(), mse=wide["mse"][arm].mean(), cumulative=pay.sum(),
                       drawdown=max_drawdown(pay), sharpe=pay.mean() / pay.std(ddof=0))
            for base in ("before", "torch_mse"):
                d = pay - wide["payout"][base]
                lo, hi = interval(d)
                rec.update({f"dpay_vs_{base}": d.mean(), f"lo_vs_{base}": lo, f"hi_vs_{base}": hi})
            dm = wide["mmc"][arm] - wide["mmc"]["before"]
            rec["dmmc_vs_before"] = dm.mean()
            rec["dmmc_lo"], rec["dmmc_hi"] = interval(dm)
            if arm.startswith("torch"):
                seeds = per_seed.loc[(name, arm), "payout"]
                rec.update(seed_min=seeds.min(), seed_max=seeds.max())
            rows.append(rec)
    return pd.DataFrame(rows).set_index(["model", "arm"])


def build():
    daily, per_seed, torch, n_seeds = load()
    comp = compare(daily, per_seed)
    comp.to_csv(OUT / "comparison.csv")
    fits = pd.read_csv(OUT / "fits.csv")
    curves = pd.read_csv(OUT / "curves.csv")

    with PdfPages(PDF, metadata={"Title": "E186 torch MLP on an MMC loss"}) as pdf:
        # ---- page 1: headline
        fig = page(1, "Training IA_01–IA_04 on MMC",
                   f"Four rolling holdouts · 520 scored dates · {daily.date.min():%d %b %Y} to "
                   f"{daily.date.max():%d %b %Y} · torch arms averaged over {n_seeds} seeds")
        head = []
        for name in MODELS:
            b, m = comp.loc[(name, "before")], comp.loc[(name, "torch_mmc")]
            head.append([name, f"{b['corr']:.4f} → {m['corr']:.4f}", f"{b.mmc:.4f} → {m.mmc:.4f}",
                         f"{b.payout:.4f} → {m.payout:.4f}", f"{m.dpay_vs_before:+.4f}",
                         f"[{m.lo_vs_before:+.4f}, {m.hi_vs_before:+.4f}]",
                         f"{m.seed_min:.4f}–{m.seed_max:.4f}"])
        table(fig, pd.DataFrame(head, columns=["Model", "CORR", "Proxy MMC", "Proxy payout", "Δ payout",
                                               "95% paired interval", "Seed range"]),
              [.07, .565, .86, .215], widths=[.07, .17, .15, .15, .1, .2, .16], font=10.5)
        s = comp.xs("torch_mmc", level="arm")
        up = list(s.index[s.dpay_vs_before > 0])
        clear_up = list(s.index[s.lo_vs_before > 0])
        clear_dn = list(s.index[s.hi_vs_before < 0])
        vs_mse_up = list(s.index[s.lo_vs_torch_mse > 0])
        vs_mse_dn = list(s.index[s.hi_vs_torch_mse < 0])
        text = (f"Against the deployed sklearn recipe, the MMC-trained torch net raised mean proxy payout for "
                f"{', '.join(up) if up else 'none of the four'}. The paired interval is above zero for "
                f"{', '.join(clear_up) if clear_up else 'none'} and below zero for "
                f"{', '.join(clear_dn) if clear_dn else 'none'}. Holding the torch loop fixed and changing only the "
                f"loss from MSE to MMC, payout is clearly higher for {', '.join(vs_mse_up) if vs_mse_up else 'none'}"
                f" and clearly lower for {', '.join(vs_mse_dn) if vs_mse_dn else 'none'}.")
        note(fig, .51, text, title="What changed")
        note(fig, .37, "Arrows read deployed sklearn recipe → torch MMC arm. The torch arm's score on a date is the "
             "mean of its five seeds' scores on that date. Seed range is the spread of the five seeds' 520-date "
             "mean payout. Proxy payout is MMC + 0.1 × CORR on rows with a historical Numerai meta score; CORR is on "
             "the full cross-section. Intervals are 40-date circular block bootstraps of the paired per-date delta.")
        note(fig, .235, "Design. Same features, folds and pinned hyperparameters as E185. The network is "
             "TorchMLPRegressor's builder: sklearn Glorot-uniform initialisation of weights and biases, float64, "
             "Adam with sklearn's settings and sklearn's L2 term. Each net trains for exactly the "
             "Timmermann-Vulicevic reference epochs E = floor(0.2 P^0.8) + 10 (IA_01 78, IA_02 426, IA_03 72, "
             "IA_04 402), with no early stopping, on batches of whole dates. The MMC loss is the per-date negative "
             "BMC of the network output against the real meta model, a Pearson surrogate of the rank-based score.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: full board
        fig = page(2, "All four recipes",
                   "Same 520 dates and asset universe per model. Δ columns are paired against the named baseline.")
        rows = []
        for name in MODELS:
            for arm in ARMS:
                r = comp.loc[(name, arm)]
                rows.append([name, LABELS[arm], f"{r['corr']:.4f}", f"{r.mmc:.4f}", f"{r.payout:.4f}",
                             f"{r.sharpe:.3f}", f"{r.drawdown:.3f}",
                             "" if arm == "before" else f"{r.dpay_vs_before:+.4f}",
                             "" if arm == "torch_mse" or not arm.startswith("torch") else f"{r.dpay_vs_torch_mse:+.4f}"])
        table(fig, pd.DataFrame(rows, columns=["Model", "Recipe", "CORR", "Proxy MMC", "Payout", "p.sharpe",
                                               "dd(pay)", "Δ vs current", "Δ vs torch MSE"]),
              [.07, .30, .86, .49], widths=[.075, .2, .085, .1, .09, .09, .085, .11, .125], font=9.5)
        note(fig, .255, "sklearn rows come from E185 (seed 42, row-shuffled minibatches; the current recipe stops "
             "on the training-loss rule with a 200-epoch cap). The torch rows differ from 'sklearn, TV epochs, MSE' "
             "in batching (whole dates) and seed averaging as well as framework, so the clean loss contrast is "
             "torch MMC against torch MSE.")
        fold0 = fits.loc[fits.fold.eq(0), "train_meta_coverage"].iloc[0]
        note(fig, .165, f"Fold 0 trains on dates before the meta model is published: only {fold0:.0%} of its training "
             "rows have a meta score. On the others the reference is constant and the MMC loss reduces to "
             "per-date CORR. Folds 1-3 have full coverage.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: cumulative payout
        fig = page(3, "Cumulative proxy payout", "Running sum of daily proxy payout, every recipe on the same dates.")
        tot = []
        cum = {}
        for name in MODELS:
            w = daily.loc[daily.model.eq(name)].pivot(index="date", columns="arm", values="payout")[list(ARMS)]
            c = w.sort_index().cumsum()
            c.loc[c.index[0] - pd.Timedelta(days=1)] = 0
            cum[name] = c.sort_index()
            last = cum[name].iloc[-1]
            tot.append([name, *[f"{last[a]:.2f}" for a in ARMS], f"{last['torch_mmc'] - last['before']:+.2f}"])
        table(fig, pd.DataFrame(tot, columns=["Model", *[LABELS[a] for a in ARMS], "MMC − current"]),
              [.07, .66, .86, .13], font=9.5)
        lo = min(c.min().min() for c in cum.values())
        hi = max(c.max().max() for c in cum.values())
        for j, name in enumerate(MODELS):
            ax = fig.add_axes([.095 + .445 * (j % 2), .385 - .23 * (j // 2), .39, .17])
            for arm in ARMS:
                ax.plot(cum[name].index, cum[name][arm], color=COLORS[arm], lw=1.5, label=LABELS[arm])
            ax.set_ylim(lo - 1, hi + 2)
            ax.set_title(name, loc="left", color=INK, fontsize=11, weight="bold")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.grid(axis="y", color=GRID, lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8, colors=MUTED)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .615), ncol=4,
                   frameon=False, fontsize=9.5)
        note(fig, .095, "Totals in score units over 520 dates. All four plots share one vertical scale.")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: training dynamics on validation
        fig = page(4, "Training dynamics",
                   "Validation window (30 dates before each test block), mean over folds and seeds, per epoch.")
        panels = [("val_mmc", "Validation proxy MMC"), ("val_corr", "Validation CORR"),
                  ("val_pred_std", "Prediction std (validation)"), ("weight_norm", "Weight norm ‖W‖")]
        for j, (col, title) in enumerate(panels):
            ax = fig.add_axes([.078 + .447 * (j % 2), .53 - .27 * (j // 2), .382, .19])
            for k, name in enumerate(MODELS):
                for loss, ls in (("mmc", "-"), ("mse", ":")):
                    c = curves.loc[curves.model.eq(name) & curves.loss.eq(loss)].groupby("epoch")[col].mean()
                    ax.plot(c.index, c.values, ls=ls, lw=1.3,
                            color=("#1b4978", "#147d73", "#a6742a", "#80579e")[k],
                            label=f"{name} {loss.upper()}")
            ax.set_xscale("log")
            if col in ("val_pred_std", "weight_norm"):
                ax.set_yscale("log")
            ax.set_title(title, loc="left", fontsize=10, color=INK, weight="bold")
            ax.set_xlabel("Epoch", fontsize=9, color=MUTED)
            ax.grid(alpha=.35, color=GRID)
            ax.tick_params(labelsize=8, colors=MUTED)
            ax.spines[["top", "right"]].set_visible(False)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .79), ncol=8,
                   frameon=False, fontsize=8.5)
        peaks = []
        for name in MODELS:
            for loss in ("mmc", "mse"):
                c = curves.loc[curves.model.eq(name) & curves.loss.eq(loss)].groupby("epoch").val_mmc.mean()
                peaks.append(f"{name} {loss.upper()} peak at epoch {c.idxmax()} ({c.max():.3f}, final {c.iloc[-1]:.3f})")
        note(fig, .205, "Mean validation MMC: " + "; ".join(peaks) + ".", width=140)
        note(fig, .115, "Solid lines are the MMC loss, dotted the MSE control. Validation never chooses anything: the "
             "scored weights are the final epoch's, as the TV schedule prescribes.", width=140)
        pdf.savefig(fig)
        plt.close(fig)
    print(comp[["corr", "mmc", "payout", "cumulative", "dpay_vs_before", "lo_vs_before", "hi_vs_before",
                "dpay_vs_torch_mse", "lo_vs_torch_mse", "hi_vs_torch_mse"]].round(4).to_string())


if __name__ == "__main__":
    build()
