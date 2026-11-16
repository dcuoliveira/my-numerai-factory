"""Create the individual volatility proxy PDFs and sweep tables from completed runs."""

import argparse
import json
from pathlib import Path
import textwrap
from zoneinfo import ZoneInfo
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.analysis.volatility_experiment import ANCHORS
from src.data_mani.volatility_proxies import FAMILIES, VIEWS

TITLES = {
    "ewma": "EWMA volatility",
    "semivariance": "Downside and upside semivariance",
    "idiosyncratic": "Idiosyncratic volatility",
    "parkinson": "Parkinson volatility",
    "garman_klass": "Garman–Klass volatility",
    "rogers_satchell": "Rogers–Satchell volatility",
    "yang_zhang": "Yang–Zhang volatility",
}
DEFINITIONS = {
    "ewma": (
        "EWMA applies exponential weights to squared close-to-close returns around zero. "
        "The recursive version initializes at the first squared return after a gap. The normalized "
        "version divides finite exponential weights by their sum. The 11 sweep values are half-lives "
        "and minimum contiguous return counts. The fixed RiskMetrics decay of .94 uses 20 returns "
        "of warm-up and has a half-life of about 11.2 bars.",
        "Recursive log returns · normalized log returns · recursive simple returns · RiskMetrics .94",
        "RiskMetrics Technical Document (1996)",
        "https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a",
    ),
    "semivariance": (
        "Downside and upside accumulate squared deviations on either side of a threshold. "
        "Zero-log and zero-simple use zero as the threshold and divide by all returns in the window. "
        "The rolling-mean variant uses the contemporaneously available trailing mean of log returns. "
        "The side-count variant divides by the number of returns on each side and assigns zero "
        "when a complete window has no returns on that side. Each is fit downside only, upside "
        "only, and jointly as two separate inputs.",
        "Zero threshold, log returns · zero threshold, simple returns · window mean · side count",
        "Barndorff-Nielsen, Kinnebrock & Shephard, Measuring downside risk",
        "https://www.nuff.ox.ac.uk/economics/papers/2008/w2/downside.pdf",
    ),
    "idiosyncratic": (
        "Estimate a rolling regression of asset log returns on benchmark log returns with an "
        "intercept. Divide trailing residual sum of squares by n − k − 1, where k is the number of "
        "benchmarks. The benchmark assets themselves are uncovered in this experiment. A complete "
        "window and a numerically identifiable benchmark covariance matrix are required.",
        "BTC factor · BTC and ETH factors · ETH factor",
        "Ang, Hodrick, Xing & Zhang, The Cross-Section of Volatility and Expected Returns",
        "https://www.ruf.rice.edu/~yxing/vol.pdf",
    ),
    "parkinson": (
        "Daily variance is log(H/L)² divided by 4 log(2). The standard estimator averages these "
        "daily values over n bars. The exponential variant uses normalized weights, half-life n "
        "and a contiguous n-bar warm-up. Weekday estimates use only weekday intrabar ranges.",
        "Equal-weight mean of ranges · exponentially weighted mean of ranges",
        "Parkinson, The Extreme Value Method for Estimating the Variance of the Rate of Return",
        "https://doi.org/10.1086/296071",
    ),
    "garman_klass": (
        "Let r = log(H/L), c = log(C/O), h = log(H/O), l = log(L/O), and g = log(O/Cprev). "
        "The standard daily term is 0.5r² − (2 log(2) − 1)c². The four-term version uses "
        "0.511r² − 0.019[c(h + l) − 2hl] − 0.383c². The gap-added version adds g² to "
        "the standard term as a sensitivity. Each term is averaged over n bars.",
        "Standard two-term · four-term · standard plus squared close-to-open gap",
        "Garman & Klass, On the Estimation of Security Price Volatilities from Historical Data",
        "https://www-2.rotman.utoronto.ca/~kan/3032/pdf/FinancialAssetReturns/Garman_Klass_JB_1980.pdf",
    ),
    "rogers_satchell": (
        "The daily term is log(H/O) log(H/C) + log(L/O) log(L/C). Standard Rogers–Satchell "
        "uses a trailing arithmetic mean. The gap-added version adds log(O/Cprev)² before "
        "averaging. The exponential sensitivity uses normalized weights, half-life n and "
        "a contiguous n-bar warm-up.",
        "Equal-weight mean · squared gap added · exponentially weighted mean",
        "Rogers & Satchell, Estimating Variance From High, Low and Closing Prices",
        "https://doi.org/10.1214/aoap/1177005835",
    ),
    "yang_zhang": (
        "Yang–Zhang combines the sample variance of log(O/Cprev), k times the sample "
        "variance of log(C/O), and (1 − k) times mean Rogers–Satchell. The n-bar coefficient "
        "is k = 0.34 / [1.34 + (n + 1)/(n − 1)]. The gap-free variant omits the first term. "
        "The gap RMS variant replaces the first term with the mean squared gap. In weekday "
        "calculations Monday's gap spans the weekend.",
        "Standard · without opening gap · squared opening gap instead of gap variance",
        "Yang & Zhang, Drift-Independent Volatility Estimation Based on High, Low, Open and Close Prices",
        "https://doi.org/10.1086/209650",
    ),
}
INK, MUTED, TEAL, NAVY = "#192b41", "#536578", "#157a78", "#203c69"
PALE, GRID = "#f3f7fa", "#dbe6ed"
COLORS = (TEAL, "#7053a1", "#c16b39", "#3286b9", "#ad4562", "#586d40", "#45334e")


def label(arm, compact=False):
    parts = arm.removeprefix("tree/").split("/")
    if len(parts) == 6:
        calendar, method, window, side, view, mode = parts
        name = f"{calendar} · {method} · {window} · {side} · {view}"
    else:
        name = arm
    return name.replace("_", " ").replace("history percentile", "TS pct").replace(
        "history bins", "TS bins") if compact else name.replace("_", " ")


def section(fig, number, title, subtitle=""):
    fig.text(.065, .943, "NUMERAI CRYPTO   /   VOLATILITY PROXY RESEARCH", color=TEAL,
             fontsize=10, weight="bold")
    fig.text(.065, .873, title, color=INK, fontsize=26, weight="bold")
    if subtitle:
        paragraph(fig, .825, subtitle, size=10.5)
    fig.add_artist(plt.Line2D([.065, .935], [.06, .06], color=GRID, linewidth=.9))
    fig.text(.065, .034, fig.study_title + "    /    Development study", color=MUTED, fontsize=8)
    fig.text(.935, .034, f"{number:02d}", color=MUTED, fontsize=8, ha="right")


def paragraph(fig, y, content, size=11, width=140, color=MUTED, x=.065):
    fig.text(x, y, textwrap.fill(content, width=width), color=color,
             fontsize=size, linespacing=1.46, va="top")


def subhead(fig, y, content):
    fig.text(.065, y, content, fontsize=14, weight="bold", va="top", color=INK)


def new_page(family):
    fig = plt.figure(figsize=(15, 8.5), facecolor="white")
    fig.study_title = TITLES[family]
    return fig


def table(fig, frame, rect, widths=None, font=9):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tb = ax.table(cellText=frame.to_numpy(), colLabels=frame.columns.tolist(),
                  cellLoc="right", colLoc="right", bbox=(0, 0, 1, 1),
                  colWidths=widths)
    tb.auto_set_font_size(False)
    tb.set_fontsize(font)
    for (row, col), cell in tb.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = .055
        cell.get_text().set_color("white" if row == 0 else INK)
        cell.set_facecolor(NAVY if row == 0 else PALE if row % 2 else "white")
        if row == 0:
            cell.get_text().set_weight("bold")
        if col == 0:
            cell.get_text().set_ha("left")


def save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def max_drawdown(series):
    cum = np.r_[0, np.cumsum(series)]
    return (np.maximum.accumulate(cum) - cum).max()


def summarize(scores):
    base = scores.groupby(["sample", "arm"], sort=False)
    summary = base.agg(corr_all=("corr_all", "mean"), corr=("corr", "mean"),
                       mmc=("mmc", "mean"), payout=("payout", "mean"),
                       dates=("mmc", "count"), all_dates=("corr_all", "count"))
    summary["drawdown"] = base.payout.agg(lambda s: max_drawdown(s.dropna()))
    summary["calmar"] = summary.payout / summary.drawdown.replace(0, np.nan)
    return summary.reset_index()


def stats(values):
    values = np.asarray(values.dropna())
    n = len(values)
    if n < 2:
        return (np.nan,) * 6
    mean = values.mean()
    x = values - mean
    lags = min(20, n - 1)
    omega = x @ x / n + sum(
        2 * (1 - lag / (lags + 1)) * (x[:-lag] @ x[lag:]) / n
        for lag in range(1, lags + 1)
    )
    se = np.sqrt(max(0, omega) / n)
    rng = np.random.default_rng(42)
    starts = rng.integers(n, size=(2000, (n + 19) // 20, 1))
    indices = ((starts + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return mean, mean / se if se else np.nan, lo, hi, (values > 0).mean(), n


def pair_rows(pivot, comparisons):
    rows = []
    for description, lhs, rhs in comparisons:
        for metric in ("corr", "mmc", "payout"):
            result = stats(pivot[metric][lhs] - pivot[metric][rhs])
            rows.append((description, metric, *result))
    return pd.DataFrame(rows, columns=[
        "Comparison", "Metric", "Delta", "HAC t", "CI low", "CI high", "Win", "Dates"])


def display_pairs(pairs):
    return pairs.assign(**{
        "Delta": pairs.Delta.map("{:+.5f}".format),
        "HAC t": pairs["HAC t"].map("{:+.2f}".format),
        "95% block interval": [f"[{lo:+.5f}, {hi:+.5f}]" for lo, hi in
                               zip(pairs["CI low"], pairs["CI high"], strict=True)],
        "Win": pairs.Win.map("{:.1%}".format),
        "Dates": pairs.Dates.map("{:.0f}".format),
    }).drop(columns=["CI low", "CI high"])[[
        "Comparison", "Metric", "Delta", "HAC t", "95% block interval", "Win", "Dates"]]


def metrics_table(rows):
    out = pd.DataFrame({
        "Arm": [label(arm, True) for arm in rows.arm], "CORR": rows["corr"].map("{:.5f}".format),
        "Proxy MMC": rows.mmc.map("{:.5f}".format), "Payout": rows.payout.map("{:.5f}".format),
        "Max DD": rows.drawdown.map("{:.3f}".format), "Calmar": rows.calmar.map("{:.4f}".format),
        "CORR all": rows.corr_all.map("{:.5f}".format), "Dates": rows.dates.astype(int),
    })
    return out


def styled_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def make_report(root, family, output):
    path = root / family
    manifest = json.loads((path / "complete.json").read_text())
    scores = pd.read_parquet(path / "scores.parquet").sort_values("date", kind="stable")
    scores["payout"] = .1 * scores["corr"] + scores["mmc"]
    summary = summarize(scores)
    settings = pd.read_csv(path / "settings.csv")
    picks = pd.read_csv(path / "picks.csv")
    coverage = pd.read_parquet(path / "coverage.parquet")
    per_date = coverage.groupby("date").agg(coverage=("covered", "mean"),
                                           any_raw=("raw_any", "mean"),
                                           all_raw=("raw_all", "mean"))
    daily = scores[scores["sample"] == "routed"].pivot(index="date", columns="arm",
                                                     values=["corr", "mmc", "payout"])
    routed = summary[summary["sample"] == "routed"].set_index("arm", drop=False)
    covered = summary[summary["sample"] == "covered"].set_index("arm", drop=False)
    exploratory = settings.merge(routed.reset_index(drop=True), on="arm")
    cover_grid = settings.merge(covered.reset_index(drop=True), on="arm")
    source = pd.read_json(root / "price_requests.json")
    success = int(source.error.isna().sum()) if "error" in source else len(source)
    source_ids = int(source.loc[source.n.gt(0), "ucid"].nunique())
    bars = int(source.n.sum())
    availability = (
        "Successful requests cover the planned windows. Vendor gaps and listing history still limit coverage."
        if success == len(source) else
        "Some windows remain uncached after rate limiting or failed requests. Download order favored lower IDs."
    )
    method, side = ANCHORS[family]

    def anchor(window, view, mode):
        return f"weekday/{method}/{window}/{side}/{view}/{mode}"

    selected_comparisons = [
        ("Selected + v2 vs matched v2", "selected_add_v2", "v2_covered"),
        ("Selected + asset vs asset only", "selected_add_asset", "asset_only"),
        ("Selected proxy vs matched v2", "selected_standalone", "v2_covered"),
    ]
    tree_comparisons = [
        (f"LightGBM · {window} · {view.replace('history_', 'TS ')}",
         "tree/" + anchor(window, view, "add_v2"), "tree_v2_covered")
        for window in (20, 60) for view in ("raw", "history_percentile")
    ]
    transform_comparisons = []
    for window in (20, 60):
        transform_comparisons += [
            (f"{window} · stand-alone TS percentile vs raw",
             anchor(window, "history_percentile", "standalone"), anchor(window, "raw", "standalone")),
            (f"{window} · stand-alone TS bins vs percentile",
             anchor(window, "history_bins", "standalone"), anchor(window, "history_percentile", "standalone")),
            (f"{window} · raw + asset vs asset only", anchor(window, "raw", "add_asset"), "asset_only"),
            (f"{window} · TS bin + asset vs asset only", anchor(window, "history_bins", "add_asset"), "asset_only"),
        ]
    diagnostic_comparisons = [
        ("Causal panel age", "causal_panel_age", "v2_all"),
        ("Retrospective never observed", "placebo_never_observed", "v2_all"),
        ("Retrospective vendor bar count", "placebo_vendor_history", "v2_all"),
        ("Covered v2 vs untouched v2", "v2_covered", "v2_all"),
    ]
    comp = pair_rows(daily, selected_comparisons)
    paired = pair_rows(daily, transform_comparisons)
    trees = pair_rows(daily, tree_comparisons)
    diagnostics = pair_rows(daily, diagnostic_comparisons)
    pd.concat([comp, paired, trees, diagnostics], ignore_index=True).to_csv(
        output / f"{family}_paired.csv", index=False)
    summary.merge(settings, on="arm", how="left").sort_values(
        ["sample", "mmc"], ascending=[True, False]).to_csv(output / f"{family}_full_sweep.csv", index=False)

    mmc_delta = comp.loc[(comp.Comparison == selected_comparisons[0][0])
                         & comp.Metric.eq("mmc")].iloc[0]
    corr_delta = comp.loc[(comp.Comparison == selected_comparisons[0][0])
                          & comp.Metric.eq("corr")].iloc[0]
    coverage_mean = coverage.covered.mean()
    headline_dates = daily["mmc"]["v2_all"].dropna().index
    best = cover_grid[cover_grid.model.eq("standalone")].nlargest(1, "mmc").iloc[0]
    chosen = picks[picks.arm.eq("selected_add_v2")].selection.value_counts()
    n_folds = picks.fold.nunique()
    destination = output / f"{family}_research_report.pdf"
    with PdfPages(destination, metadata={"Title": TITLES[family] + " · Numerai Crypto"}) as pdf:
        fig = new_page(family)
        section(fig, 1, TITLES[family],
                "Development results from a purged walk-forward experiment on Numerai Crypto v2.0.")
        subhead(fig, .743, "What the experiment found")
        paragraph(fig, .696,
                  f"The fold-selected proxy added to the 22 shipped features changed routed CORR by "
                  f"{corr_delta.Delta:+.5f} and proxy MMC by {mmc_delta.Delta:+.5f} versus the same-row "
                  f"shipped-feature control. The 95% block interval for the MMC difference is "
                  f"[{mmc_delta['CI low']:+.5f}, {mmc_delta['CI high']:+.5f}]. "
                  f"Common candidate coverage was {coverage_mean:.1%} of development panel rows.")
        paragraph(fig, .589,
                  f"The descriptive covered-only stand-alone MMC leader was {label(best.arm)} "
                  f"({best.mmc:.5f}). The inner search selected {len(chosen)} distinct add-on configurations "
                  f"in {n_folds} folds. Fixed-grid outer rankings reflect repeated development comparisons.")
        table(fig, metrics_table(routed.loc[
            ["v2_all", "v2_covered", "selected_add_v2", "asset_only", "selected_add_asset",
             "selected_standalone"]]),
              [.065, .213, .87, .284], [.32, .10, .11, .10, .09, .09, .10, .09])
        paragraph(fig, .172,
                  f"Headline metrics use {len(headline_dates)} dates with observed meta predictions "
                  f"({headline_dates.min():%Y-%m-%d} to {headline_dates.max():%Y-%m-%d}). "
                  "CORR all uses the complete scored development calendar. Payout score is 0.1 × CORR + proxy MMC.")
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 2, "Estimator, sweep and evaluation")
        formula, variants, reference, url = DEFINITIONS[family]
        for y, heading, body in (
            (.805, "Estimator and implementation variants", formula + "  " + variants + "."),
            (.639, "Calendar and history", "Daily uses every UTC bar. Weekday uses Monday–Friday bars. "
             "For a prediction date D, candidate inputs stop at the last completed selected bar dated D−1 or earlier. "
             "Missing selected bars interrupt contiguous windows. The continuous own-history percentile compares "
             "against the prior 252 selected bars, requires 126 valid values and gives ties half weight. "
             "The discretized version applies the .05/.25/.75/.95 cutpoints before cross-sectional binning."),
            (.470, "Models and selection", f"{len(settings):,} ridge arms test stand-alone, add-to-shipped "
             "and add-to-asset effects for each configuration and transform. Cross-sectional five-bin ranks "
             "preserve ties on the common covered cohort. Inner CORR selects from 13 ridge alphas (0.0001 to 100); "
             "the same criterion selects one configuration per fold in each selected arm. Four LightGBM settings "
             "cross depths 3/5 with 100/250 trees at learning rate .05."),
            (.298, "Purged chronology and full-universe scoring", f"{n_folds} folds train on 130 dates, "
             "purge 22 dates and predict 130. Inner tuning uses the last 40 training dates for validation, with "
             "a 22-date purge before validation. Both covered fits see exactly the same rows. Their forecasts "
             "occupy the baseline rank slots assigned to covered assets. Uncovered assets retain their baseline "
             "ranks. Equal learning budgets apply within model type."),
        ):
            subhead(fig, y, heading)
            paragraph(fig, y - .039, body)
        fig.text(.065, .118, reference, color=TEAL, fontsize=9, url=url)
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 3, "The horizon sweep",
                "Stand-alone covered-only proxy MMC by calendar and transformation. All curves use the common cohort. "
                "The fixed .94 RiskMetrics setting, where applicable, sits at 20 bars of warm-up.")
        ax_grid = {}
        for row, calendar in enumerate(("daily", "weekday")):
            for col, view in enumerate(VIEWS):
                ax = fig.add_axes([.075 + col * .30, .46 - row * .30, .255, .245])
                ax_grid[calendar, view] = ax
                subset = cover_grid[(cover_grid.model == "standalone")
                                    & (cover_grid.calendar == calendar) & (cover_grid.view == view)]
                for idx, (name, rows) in enumerate(subset.groupby("method", sort=False)):
                    for j, (side_name, line) in enumerate(rows.groupby("side", sort=False)):
                        line = line.sort_values("window")
                        ax.plot(line.window, line.mmc, color=COLORS[idx],
                                linestyle=(":", "--", "-")[j] if len(rows.side.unique()) == 3 else "-",
                                marker="o", markersize=2.6, linewidth=1.45,
                                label=name.replace("_", " ") if j == 0 else None)
                ax.axhline(0, lw=.6, color=MUTED)
                ax.set_xscale("log")
                ax.set_xticks([5, 20, 60, 240], labels=["5", "20", "60", "240"])
                ax.set_title(f"{calendar.title()} · {view.replace('history_', 'TS ').title()}",
                             loc="left", fontsize=11, color=INK)
                styled_axis(ax)
                if col == 0:
                    ax.set_ylabel("Proxy MMC")
        limits = [ax.get_ylim() for ax in ax_grid.values()]
        for ax in ax_grid.values():
            ax.set_ylim(min(a for a, b in limits), max(b for a, b in limits))
        handles, names = ax_grid["weekday", "raw"].get_legend_handles_labels()
        if handles:
            fig.legend(handles, names, ncol=min(5, len(names)), loc="lower center",
                       bbox_to_anchor=(.5, .082), frameon=False, fontsize=9)
        if family == "semivariance":
            fig.text(.065, .116, "Dotted = downside    Dashed = upside    Solid = joint", color=MUTED, fontsize=9)
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 4, "Matched comparisons and controls",
                "Per-date differences in routed scores. Newey–West lag 20 and 2,000 circular block-20 draws. "
                "Intervals are descriptive across the development sample.")
        table(fig, display_pairs(comp), [.065, .479, .87, .283],
              [.31, .10, .10, .08, .18, .08, .08], font=8.6)
        subhead(fig, .438, "Own-history transformation and asset effects")
        chosen_pairs = paired[paired.Metric.eq("mmc")]
        table(fig, display_pairs(chosen_pairs), [.065, .126, .87, .268],
              [.37, .08, .09, .07, .16, .07, .08], font=8.6)
        paragraph(fig, .101, "20 and 60 use the pre-specified weekday implementation. TS bin and "
                  "TS percentile comparisons use the same features, keys, folds and cross-sectional bins.", size=9)
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 5, "Cumulative score and nonlinear sensitivity",
                "Cumulative payout score on the same observed-meta rows. Drawdown measures the fall "
                "from the running peak in cumulative score, with the series starting at zero.")
        ax = fig.add_axes([.072, .416, .86, .32])
        for color, (name, curve) in zip(COLORS, [
            ("Full v2", daily.payout.v2_all),
            ("Matched v2", daily.payout.v2_covered),
            ("Selected proxy + v2", daily.payout.selected_add_v2),
            ("Asset only", daily.payout.asset_only),
            ("Selected proxy + asset", daily.payout.selected_add_asset),
        ], strict=False):
            ax.plot(curve.dropna().cumsum(), lw=1.7, color=color, label=name)
        styled_axis(ax)
        ax.set_ylabel("Cumulative payout score")
        ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
        subhead(fig, .363, "LightGBM with the same covered-feature contrast")
        table(fig, display_pairs(trees[trees.Metric.isin(["corr", "mmc"])]),
              [.065, .094, .87, .229], [.32, .09, .10, .08, .17, .08, .08], font=8.6)
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 6, "Coverage and audit",
                "Coverage is measured on the full base panel. The study uses a single within-family cohort "
                "for candidate fitting, defined by the intersection of finite values across the full grid and all history transforms.")
        ax = fig.add_axes([.074, .53, .85, .228])
        ax.plot(per_date.index, per_date.any_raw.rolling(20, min_periods=1).mean(),
                color="#7c8eac", label="Any raw setting")
        ax.plot(per_date.index, per_date.all_raw.rolling(20, min_periods=1).mean(),
                color="#c16b39", label="All raw settings")
        ax.plot(per_date.index, per_date.coverage.rolling(20, min_periods=1).mean(),
                color=TEAL, label="Common covered")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=8)
        styled_axis(ax)
        meta_coverage = per_date.loc[headline_dates, "coverage"]
        coverage_order = meta_coverage.rank(method="first")
        bands = pd.qcut(coverage_order, 3, labels=["Low", "Middle", "High"])
        cohort_frame = pd.DataFrame({
            "coverage": meta_coverage,
            "mmc_delta": (daily.mmc.selected_add_v2 - daily.mmc.v2_covered).loc[headline_dates],
            "corr_delta": (daily["corr"].selected_add_v2 - daily["corr"].v2_covered).loc[headline_dates],
        }).groupby(bands, observed=True).agg(coverage=("coverage", "mean"),
                                         mmc=("mmc_delta", "mean"), corr=("corr_delta", "mean"),
                                         dates=("mmc_delta", "count"))
        table(fig, cohort_frame.reset_index(names="Coverage tertile").assign(
            coverage=lambda x: x.coverage.map("{:.1%}".format),
            mmc=lambda x: x.mmc.map("{:+.5f}".format),
            corr=lambda x: x["corr"].map("{:+.5f}".format),
        ).rename(columns={"coverage": "Mean coverage", "mmc": "Selected Δ MMC",
                          "corr": "Selected Δ CORR", "dates": "Dates"}),
              [.065, .327, .87, .145], [.23, .22, .20, .20, .15])
        paragraph(fig, .282,
                  f"PASS WITH CONDITIONS. The stored panel retains {manifest['panel_rows']:,} rows. "
                  f"{manifest['covered_rows']:,} rows across {manifest['covered_ids']:,} IDs form the "
                  f"matched cohort. {manifest['tested_cutoffs']} cut-date replays passed. Every routed fold "
                  "asserted finite outputs, exact uncovered ranks and preservation of every covered slot.")
        paragraph(fig, .182,
                  f"Material source limitation. The retrospective API snapshot returned {bars:,} bars for "
                  f"{source_ids:,} IDs. {success:,}/{len(source):,} planned request windows were cached. "
                  f"{availability} The margins apply to this cached-data routing policy. Historical vendor "
                  "availability and an independent "
                  "live replay remain unverified.")
        save(pdf, fig)

        fig = new_page(family)
        section(fig, 7, "Development leaders and provenance",
                "Rows are sorted by outer proxy MMC or improvement. The complete sweep, selection records and "
                "per-date scores provide the underlying comparisons.")
        top = cover_grid[cover_grid.model.eq("standalone")].nlargest(6, "mmc")
        table(fig, metrics_table(top), [.065, .546, .87, .219],
              [.46, .085, .09, .085, .07, .07, .08, .06], font=8.3)
        subhead(fig, .507, "Largest routed v2 add-on improvements in the sweep")
        addon = exploratory[exploratory.model.eq("add_v2")].copy()
        addon["delta"] = addon.mmc - routed.loc["v2_covered", "mmc"]
        best_addon = addon.nlargest(6, "delta")
        table(fig, metrics_table(best_addon).assign(
            **{"Δ MMC": best_addon.delta.map("{:+.5f}".format)}),
              [.065, .23, .87, .223], [.38, .08, .08, .08, .07, .07, .08, .06, .10], font=8.3)
        paragraph(fig, .182,
                  "Development evidence uses the February 2026 cutoff from the reference synthesis. "
                  "The two retrospective availability placebos and causal age scores are in the paired CSV. "
                  "The study manifest records hashes for the price snapshot, request manifest and Numerai inputs. "
                  "Calmar is mean payout score divided by maximum cumulative drawdown.", size=9.5)
        fig.text(.065, .088, f"Generated {datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%d %H:%M %Z}",
                 color=MUTED, fontsize=8)
        save(pdf, fig)
    print(f"{family}: {destination.name}, ΔCORR {corr_delta.Delta:+.5f}, "
          f"ΔMMC {mmc_delta.Delta:+.5f}, coverage {coverage_mean:.1%}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("results/volatility_proxies"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    output = args.output or args.study_root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    for family in (args.family,) if args.family else FAMILIES:
        make_report(args.study_root, family, output)


if __name__ == "__main__":
    main()
