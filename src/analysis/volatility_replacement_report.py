"""Audit and report the matched volatility replacement experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

from src.analysis.volatility_experiment import TARGET, cross_section_bins, digest, groups, load
from src.analysis.volatility_proxy_report import (
    INK, MUTED, max_drawdown, paragraph, save, section, stats, styled_axis, subhead, table,
)
from src.analysis.volatility_replacement import ARMS, MODELS
from src.data_mani.build_volatility_replacement import NEW, REMOVED

LABELS = dict(zip(ARMS, ("Original · full", "Original · matched", "Replacement · matched"), strict=True))
COLORS = dict(zip(MODELS, ("#147a78", "#5d4a98", "#bc6033"), strict=True))


def audit(root, run, dataset, scores):
    expected = dataset["inputs"] | run["inputs"]
    for path, checksum in expected.items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Input hash differs for {path}")
    panel_path = next(Path(path) for path in run["inputs"] if path in dataset["inputs"])
    meta_path = next(Path(path) for path in run["inputs"]
                     if Path(path).suffix == ".parquet" and Path(path) != panel_path
                     and Path(path).name != "v2.0_volatility_replacement.parquet")
    base = pd.read_parquet(panel_path)
    replacement = pd.read_parquet(root / "v2.0_volatility_replacement.parquet")
    pd.testing.assert_frame_equal(replacement.drop(columns=list(NEW)), base.drop(columns=list(REMOVED)))
    availability = pd.read_parquet(root / "candidate_availability.parquet")
    estimates = availability[list(NEW)].to_numpy()
    mask = np.isfinite(estimates).all(axis=1)
    np.testing.assert_array_equal(mask, availability.covered)
    np.testing.assert_array_equal(mask, replacement[list(NEW)].notna().all(axis=1))
    pd.testing.assert_frame_equal(availability[["ucid", "date"]],
                                  base[["ucid", "date"]].reset_index(drop=True))
    order = availability.assign(ucid=availability.ucid.astype(int)).sort_values(["date", "ucid"]).index
    indices = order[mask[order]]
    np.testing.assert_array_equal(
        replacement[list(NEW)].to_numpy()[indices],
        cross_section_bins(estimates[indices], availability.date.to_numpy()[indices]),
    )
    panel, _ = load(panel_path, meta_path)
    covered_keys = availability.loc[mask, ["ucid", "date"]].assign(ucid=lambda x: x.ucid.astype(int))
    meta_days = scores[scores["sample"].eq("routed") & scores.mmc.notna()].date.unique()
    for fold, file in enumerate(sorted(root.glob("routed_fold_*.parquet"))):
        predictions = pd.read_parquet(file)
        frame = predictions.merge(panel[["date", "ucid", TARGET, "meta"]], on=["date", "ucid"],
                                  validate="one_to_one", how="left")
        panel_fold = panel[panel.date.isin(frame.date)]
        pd.testing.assert_frame_equal(frame[["date", "ucid"]], panel_fold[["date", "ucid"]].reset_index(drop=True))
        mask = frame.set_index(["ucid", "date"]).index.isin(covered_keys.set_index(["ucid", "date"]).index)
        np.testing.assert_array_equal(mask, frame.covered)
        for model in MODELS:
            names = [f"{model}/{arm}" for arm in ARMS]
            values = frame[names].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Saved forecast has non-finite values")
            np.testing.assert_array_equal(values[~mask, 1:], np.repeat(values[~mask, :1], 2, axis=1))
            for ix in groups(frame.date.to_numpy()):
                slots = np.sort(values[ix[mask[ix]], :], axis=0)
                np.testing.assert_array_equal(slots[:, 1:], np.repeat(slots[:, :1], 2, axis=1))
            observed = frame[frame.date.isin(meta_days) & frame.meta.notna()]
            if observed.empty:
                continue
            sample = observed[observed.date.eq(observed.date.iloc[-1])]
            reference = scores[scores["sample"].eq("routed") & scores.date.eq(sample.date.iloc[0])
                               & scores.model.eq(model)].set_index("arm").loc[list(ARMS)]
            np.testing.assert_allclose(numerai_corr(sample[names], sample[TARGET].astype(float)),
                                       reference["corr"], atol=1e-7, rtol=0)
            np.testing.assert_allclose(correlation_contribution(
                sample[names], sample.meta, sample[TARGET].astype(float)), reference.mmc, atol=1e-7, rtol=0)
    if scores.fold.nunique() != fold + 1 or len(meta_days) != 468:
        raise ValueError("Unexpected number of saved folds or scored meta dates")


def summarize(scores):
    summary = scores.groupby(["sample", "model", "arm"], sort=False).agg(
        corr=("corr", "mean"), mmc=("mmc", "mean"), payout=("payout", "mean"),
        maxdd=("payout", lambda x: max_drawdown(x.dropna())), corr_all=("corr_all", "mean"),
        dates=("mmc", "count"))
    summary["calmar"] = summary.payout / summary.maxdd.replace(0, np.nan)
    return summary.reset_index()


def matched_deltas(scores):
    rows = []
    for (sample, model), data in scores.groupby(["sample", "model"], sort=False):
        pivot = data.pivot(index="date", columns="arm", values=["corr", "mmc", "payout"])
        for metric in ("corr", "mmc", "payout"):
            delta = pivot[metric].replacement - pivot[metric].original_matched
            rows.append((sample, model, metric, *stats(delta)))
    return pd.DataFrame(rows, columns=["sample", "model", "metric", "delta", "hac_t",
                                       "ci_low", "ci_high", "positive_days", "dates"])


def report(root):
    dataset = json.loads((root / "dataset_manifest.json").read_text())
    run = json.loads((root / "run_manifest.json").read_text())
    scores = pd.read_parquet(root / "scores.parquet").sort_values("date", kind="stable")
    audit(root, run, dataset, scores)
    summary = summarize(scores)
    paired = matched_deltas(scores)
    summary.to_csv(root / "performance.csv", index=False)
    paired.to_csv(root / "paired_deltas.csv", index=False)
    coverage = pd.read_csv(root / "coverage_by_date.csv", parse_dates=["date"])
    routed = summary[summary["sample"].eq("routed")].set_index(["model", "arm"])
    meta = scores[scores["sample"].eq("routed") & scores.mmc.notna()]
    dates = pd.DatetimeIndex(meta.date.unique())
    tested = coverage.date.between(run["development_start"], run["development_end"])
    dev_coverage = coverage.loc[tested, "covered_rows"].sum() / coverage.loc[tested, "panel_rows"].sum()
    reuse = run.get("reused_non_mlp")
    recipe = run.get("mlp_recipe_multiplier")
    epoch_text = (f"runs {recipe}× the size-dependent recipe per fit"
                  if recipe else f"fits {run['mlp_epochs']:,} epochs")

    def page(number, title, subtitle=""):
        fig = plt.figure(figsize=(15, 8.5), facecolor="white")
        fig.study_title = "Volatility replacement"
        section(fig, number, title, subtitle)
        return fig

    with PdfPages(root / "volatility_replacement_report.pdf",
                  metadata={"Title": "Numerai Crypto v2.0 volatility replacement"}) as pdf:
        fig = page(1, "Replacing volatility in Numerai Crypto v2.0",
                   "New MLP fits with the original Ridge and LightGBM forecasts."
                   if reuse else "Scratch fits of Ridge, LightGBM and MLP on the shipped and replacement feature sets.")
        rows = []
        for model in MODELS:
            for arm in ARMS:
                row = routed.loc[(model, arm)]
                rows.append([model, LABELS[arm], f"{row['corr']:.5f}", f"{row.mmc:.5f}",
                             f"{row.payout:.5f}", f"{row.maxdd:.4f}", f"{row.calmar:.4f}",
                             f"{row.corr_all:.5f}"])
        comparison = pd.DataFrame(rows, columns=["Model", "Feature set", "CORR", "Proxy MMC", "Payout",
                                                 "Max DD", "Calmar", "CORR all"])
        table(fig, comparison, [.065, .438, .87, .322],
              [.12, .23, .106, .112, .111, .107, .102, .112])
        subhead(fig, .402, "Replacement minus original on matched rows")
        payout = paired[paired["sample"].eq("routed") & paired.metric.eq("payout")].set_index("model")
        rows = []
        for model in MODELS:
            a, b = routed.loc[(model, "original_matched")], routed.loc[(model, "replacement")]
            d = payout.loc[model]
            rows.append([model, f"{b['corr'] - a['corr']:+.5f}", f"{b.mmc - a.mmc:+.5f}",
                         f"{d.delta:+.5f}", f"[{d.ci_low:+.5f}, {d.ci_high:+.5f}]",
                         f"{b.maxdd - a.maxdd:+.4f}"])
        table(fig, pd.DataFrame(rows, columns=["Model", "Δ CORR", "Δ proxy MMC", "Δ payout",
                                              "95% block interval, payout", "Δ Max DD"]),
              [.065, .191, .87, .16], [.15, .14, .15, .14, .26, .16])
        paragraph(fig, .152, f"Headline metrics share {len(dates)} dates from {dates.min():%Y-%m-%d} through "
                  f"{dates.max():%Y-%m-%d} and the same meta-observed assets per date. CORR all uses the full "
                  "scored development calendar. Calmar is mean payout divided by maximum cumulative payout drawdown.",
                  size=9.5)
        save(pdf, fig)

        fig = page(2, "Cumulative payout",
                   "Payout score is 0.1 × CORR + proxy MMC. Curves start at zero and use the same dates and y-scale.")
        axes = []
        for i, model in enumerate(MODELS):
            ax = fig.add_axes([.076 + i * .298, .245, .256, .45])
            axes.append(ax)
            ax.set_title(model, loc="left", color=INK, fontsize=15, weight="bold")
            for arm, color, style in zip(ARMS, ("#8f99a6", "#243f70", "#168580"),
                                         ("--", "-", "-"), strict=True):
                values = meta[meta.model.eq(model) & meta.arm.eq(arm)].set_index("date").payout
                start = pd.Series([0.], index=[values.index[0] - pd.Timedelta(days=1)])
                ax.plot(pd.concat([start, values.cumsum()]), color=color, linestyle=style,
                        label=LABELS[arm], linewidth=1.8)
            styled_axis(ax)
            ax.tick_params(axis="x", labelrotation=32)
        low, high = min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes)
        for ax in axes:
            ax.set_ylim(low, high)
        axes[0].set_ylabel("Cumulative payout score")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .113),
                   frameon=False, ncol=3)
        save(pdf, fig)

        fig = page(3, "Original versus replacement",
                   "One point per model. The x-axis uses shipped v2.0 and the y-axis uses the replacement. "
                   "Both forecasts are trained on the same covered rows and routed on the same full universe.")
        handles = []
        for i, (metric, title) in enumerate((("mmc", "Mean proxy MMC"),
                                             ("payout", "Mean payout"), ("maxdd", "Max payout drawdown"))):
            ax = fig.add_axes([.083 + i * .297, .267, .241, .425])
            old = np.array([routed.loc[(m, "original_matched"), metric] for m in MODELS])
            new = np.array([routed.loc[(m, "replacement"), metric] for m in MODELS])
            lo, hi = min(old.min(), new.min()), max(old.max(), new.max())
            pad = max((hi - lo) * .12, .00001)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "--", color=MUTED, linewidth=1)
            for model, x, y in zip(MODELS, old, new, strict=True):
                point = ax.scatter(x, y, s=62, color=COLORS[model], zorder=3, label=model)
                if i == 0:
                    handles.append(point)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Original v2.0", ylabel="Replacement v2.0")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, loc="left", color=INK, fontsize=13, weight="bold")
            styled_axis(ax)
        fig.legend(handles, MODELS, loc="lower center", bbox_to_anchor=(.5, .14), frameon=False, ncol=3)
        paragraph(fig, .106, "Above the 45° line means higher MMC or payout. Below the line means lower drawdown.",
                  size=9.5)
        save(pdf, fig)

        fig = page(4, "Coverage and validation",
                   f"The dataset preserves all {dataset['panel_rows']:,} original asset-date rows. The jointly "
                   f"covered share over scored development dates is {dev_coverage:.1%}.")
        ax = fig.add_axes([.083, .505, .844, .25])
        ax.plot(coverage.date, coverage.covered_share.rolling(20, min_periods=1).mean(),
                color="#168580", linewidth=1.6)
        ax.axvline(pd.Timestamp(run["development_end"]), linestyle="--", color=MUTED, linewidth=1)
        ax.set_ylim(0, 1)
        styled_axis(ax)
        ax.set_ylabel("Covered share")
        subhead(fig, .461, "Payout changes by development-date coverage")
        day = coverage[coverage.date.isin(dates)].set_index("date").covered_share
        buckets = pd.qcut(day.rank(method="first"), 3, labels=["Low", "Middle", "High"])
        rows = []
        for label in buckets.cat.categories:
            ix = buckets[buckets.eq(label)].index
            selected = meta[meta.date.isin(ix)].groupby(["model", "arm"]).payout.mean()
            deltas = [selected[m, "replacement"] - selected[m, "original_matched"] for m in MODELS]
            rows.append([label, f"{day.loc[ix].mean():.1%}", len(ix), *[f"{d:+.5f}" for d in deltas]])
        table(fig, pd.DataFrame(rows, columns=["Coverage tertile", "Coverage", "Dates", *MODELS]),
              [.065, .257, .87, .15], [.235, .16, .125, .16, .16, .16])
        subhead(fig, .222, "PASS WITH CONDITIONS")
        paragraph(fig, .176, f"{len(dataset['cutoff_replays'])} truncated-price replays passed. "
                  "Stored forecasts reproduce Numerai scores on sampled dates and preserve baseline ranks and covered "
                  "slots on every date. The price history is retrospective and inherits the original snapshot's "
                  "source gaps. Historical vendor availability and independent live parity remain unverified. "
                  "Coverage tertiles are descriptive.", size=9.5)
        save(pdf, fig)

        fig = page(5, "Dataset and training specification")
        specs = pd.DataFrame(dataset["columns"])[["column", "calendar", "estimator", "window", "view"]]
        specs.columns = ["Column", "Calendar", "Estimator", "Bars", "Processing"]
        table(fig, specs, [.065, .489, .87, .307], [.52, .11, .16, .10, .11])
        for y, heading, body in (
            (.445, "Dataset construction", "The replacement drops the shipped 20- and 60-day volatility columns and "
             "adds nine fixed proxy columns, retaining the other 20 shipped features. Fixed variants use the most "
             "frequent earlier fold nominations with standard-estimator and daily tie breaks. The downside and "
             "upside pair uses the most frequent joint semivariance variant. Candidate bars close by D−1. "
             "Only jointly covered rows enter candidate cross-sectional binning and fitting."),
            (.309, "Fresh model fits", "Eleven folds use 130 training dates, a 22-date purge and 130 prediction dates. "
             "The last 40 training dates select parameters by inner CORR after an internal 22-date purge. Ridge "
             "tests 13 alphas. LightGBM tests depths 3/5 and 100/250 trees. The ReLU MLP tests two architectures "
             f"and two penalties, {epoch_text} and averages three fresh refits with paired seeds."),
            (.177, "Interpretation", "The nine columns were selected from earlier development studies on overlapping "
             "dates. Results are exploratory. The packaged dataset contains later rows for future use. Training "
             "and scoring here stop at the original February 19, 2026 development cutoff."),
        ):
            subhead(fig, y, heading)
            paragraph(fig, y - .035, body, size=9.4)
        save(pdf, fig)

        if reuse:
            prior = pd.read_parquet(Path(reuse) / "scores.parquet").sort_values("date", kind="stable")
            prior_run = json.loads((Path(reuse) / "run_manifest.json").read_text())
            pd.testing.assert_frame_equal(
                scores[scores.model.ne("MLP")].reset_index(drop=True),
                prior[prior.model.ne("MLP")].reset_index(drop=True),
            )
            original_mlp = summarize(prior).query("sample == 'routed' and model == 'MLP'").set_index("arm")
            updated_mlp = routed.loc["MLP"]
            comparison = pd.concat({"Previous": original_mlp, "Rerun": updated_mlp},
                                   names=["run"]).reset_index()
            comparison.to_csv(root / "mlp_epoch_comparison.csv", index=False)
            fig = page(6, "MLP epoch-budget comparison",
                       f"The previous run used {prior_run['mlp_epochs']} epochs per fit. "
                       f"The rerun {epoch_text} on {run['mlp_device'].upper()}.")
            rows = [[row.run, LABELS[row.arm], f"{row.corr:.5f}", f"{row.mmc:.5f}",
                     f"{row.payout:.5f}", f"{row.maxdd:.4f}"]
                    for row in comparison.itertuples(index=False)]
            table(fig, pd.DataFrame(rows, columns=["Run", "Feature set", "CORR", "Proxy MMC",
                                                  "Payout", "Max DD"]),
                  [.065, .447, .87, .29], [.16, .28, .14, .14, .14, .14])
            subhead(fig, .397, "Fixed epochs per fit")
            schedule = [[width, units, epochs] for width, settings in run["mlp_epoch_schedule"].items()
                        for units, epochs in settings.items()]
            if not schedule:
                schedule = [["All", "All", run["mlp_epochs"]]]
            table(fig, pd.DataFrame(schedule, columns=["Input columns", "Hidden layers", "Epochs"]),
                  [.065, .176, .87, .18], [.35, .35, .30])
            paragraph(fig, .132, "Each training loop completes its full epoch count. Inner validation CORR selects "
                      "the architecture and regularization after training. Final forecasts average three fresh refits.",
                      size=9.5)
            save(pdf, fig)
    print(routed[["corr", "mmc", "payout", "maxdd"]].to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/volatility_replacement"))
    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
