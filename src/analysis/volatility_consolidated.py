"""Replay four volatility settings per proxy and report forecast correlations."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from src.analysis.volatility_experiment import (
    TARGET, check_scoring, cross_section_bins, digest, groups, load, ridge_grid,
    route, score_date, select_model,
)
from src.analysis.volatility_proxy_report import (
    INK, MUTED, TEAL, max_drawdown, paragraph, save, section, stats, subhead, table,
)
from src.data_mani.volatility_proxies import FAMILIES

SETTINGS = (
    ("Raw 120", "raw", 120), ("Percentile 40", "history_percentile", 40),
    ("Bins 40", "history_bins", 40), ("Bins 120", "history_bins", 120),
)
MODELS = {
    "standalone": "Ridge proxy only",
    "add_asset": "Ridge proxy + asset effects",
    "add_v2": "Ridge proxy + v2",
    "tree_add_v2": "LightGBM proxy + v2",
}
PROXIES = dict(zip(FAMILIES, (
    "EWMA", "Semivariance", "Idiosyncratic", "Parkinson", "Garman–Klass",
    "Rogers–Satchell", "Yang–Zhang",
), strict=True))


def forecast_name(model, family, processing):
    return f"{MODELS[model]} | {PROXIES[family]} | {processing}"


def prepare_features(study, panel, covered):
    dates = panel.date.to_numpy()[covered]
    result = {}
    for view in dict.fromkeys(view for _, view, _ in SETTINGS):
        path = study / f"features_{view}.parquet"
        windows = {str(window) for _, name, window in SETTINGS if name == view}
        names = [col for col in pq.read_schema(path).names if "/" in col
                 and col.split("/")[2] in windows]
        frame = pd.read_parquet(path, columns=["date", "ucid", *names])
        pd.testing.assert_frame_equal(frame[["date", "ucid"]], panel[["date", "ucid"]])
        values = frame.loc[covered, names].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid covered {view} input in {study}")
        result[view] = (cross_section_bins(values, dates),
                        {name: i for i, name in enumerate(names)})
    return result


def choose_config(grid, picks, model, view, window):
    keys = grid.loc[grid.model.eq(model) & grid.view.eq(view) & grid.window.eq(window), "arm"]
    eligible = picks.loc[keys]
    return eligible.inner_corr.round(12).idxmax()


def score_forecast_dates(test_panel, forecasts):
    names = forecasts.columns.tolist()
    target = test_panel[TARGET].to_numpy()
    meta = test_panel.meta.to_numpy()
    result = []
    for ix in groups(test_panel.date.to_numpy()):
        values = forecasts.iloc[ix].to_numpy()
        corr_all, corr, mmc = score_date(values, target[ix], meta[ix])
        result.extend(zip([test_panel.date.iloc[ix[0]]] * len(names), names, corr_all, corr, mmc,
                          strict=True))
    frame = pd.DataFrame(result, columns=["date", "series", "corr_all", "corr", "mmc"])
    frame["payout"] = .1 * frame["corr"] + frame.mmc
    return frame


def replay_family(task):
    root, output, panel_path, meta_path, family = task
    study = root / family
    manifest = json.loads((study / "complete.json").read_text())
    paths = (panel_path, meta_path, root / "ohlc.parquet", root / "price_requests.json")
    for path in paths:
        expected = {Path(name).name: value for name, value in manifest["inputs"].items()}
        if digest(path) != expected[path.name]:
            raise ValueError(f"Frozen input differs from the original study: {path}")
    check_scoring()
    panel, shipped = load(panel_path, meta_path)
    coverage = pd.read_parquet(study / "coverage.parquet")
    pd.testing.assert_frame_equal(coverage[["date", "ucid"]], panel[["date", "ucid"]])
    covered = coverage.covered.to_numpy()
    features = prepare_features(study, panel, covered)
    eligible = panel.loc[covered].reset_index(drop=True)
    x = shipped[covered]
    y = eligible[TARGET].to_numpy(dtype=float)
    day = eligible.date.to_numpy()
    asset = pd.factorize(panel.ucid, sort=True)[0][covered]
    folds = pd.read_csv(study / "folds.csv", parse_dates=list(range(1, 7)))
    grid = pd.read_csv(study / "settings.csv")
    picks = pd.read_csv(study / "picks.csv")
    names = [forecast_name(model, family, processing) for model in MODELS
             for processing, _, _ in SETTINGS]
    controls = ["v2_all", "v2_covered", "asset_only", "tree_v2_all", "tree_v2_covered"]
    routed_parts, covered_parts, selections, score_parts = [], [], [], []
    for split in folds.itertuples(index=False):
        fold_picks = picks[picks.fold.eq(split.fold)].set_index("arm")
        dates = eligible.date
        before = np.flatnonzero(dates.between(split.train_start, split.inner_train_end))
        validation = np.flatnonzero(dates.between(split.validation_start, split.train_end))
        train = np.flatnonzero(dates.between(split.train_start, split.train_end))
        test = np.flatnonzero(dates.between(split.test_start, split.test_end))
        test_panel = panel[panel.date.between(split.test_start, split.test_end)].reset_index(drop=True)
        stored = pd.read_parquet(study / f"routed_fold_{split.fold:02d}.parquet")
        pd.testing.assert_frame_equal(stored[["date", "ucid"]], test_panel[["date", "ucid"]])
        mask = stored.covered.to_numpy()
        np.testing.assert_array_equal(mask, covered[panel.date.between(split.test_start, split.test_end)])
        pd.testing.assert_frame_equal(test_panel.loc[mask, ["date", "ucid"]].reset_index(drop=True),
                                      eligible.iloc[test][["date", "ucid"]].reset_index(drop=True))
        predictions = {}
        nominated = {}
        for model in MODELS:
            for processing, view, window in SETTINGS:
                name = forecast_name(model, family, processing)
                tree = model == "tree_add_v2"
                if tree:
                    arm = nominated[processing]
                else:
                    arm = choose_config(grid, fold_picks, model, view, window)
                    if model == "add_v2":
                        nominated[processing] = arm
                calendar, method, _, side, _, _ = arm.split("/")
                sides = ("down", "up") if side == "joint" else (side,)
                matrix, columns = features[view]
                indices = [columns[f"{calendar}/{method}/{window}/{s}"] for s in sides]
                candidate = matrix[:, indices]
                if tree:
                    prediction, parameter, validation_corr = select_model(
                        candidate, y, before, validation, test, day, tree=True, base=x)
                else:
                    parameter = float(fold_picks.at[arm, "selection"])
                    validation_corr = float(fold_picks.at[arm, "inner_corr"])
                    xt, xe = candidate[train], candidate[test]
                    if model == "add_v2":
                        xt, xe = np.column_stack((x[train], xt)), np.column_stack((x[test], xe))
                    ids, eval_ids = (asset[train], asset[test]) if model == "add_asset" else (None, None)
                    prediction = ridge_grid(xt, y[train], xe, ids, eval_ids, np.array([parameter]))[:, 0]
                predictions[name] = prediction
                selections.append((split.fold, name, arm, parameter, validation_corr))
        local = np.column_stack([predictions[name] for name in names])
        baselines = ["tree_v2_all" if name.startswith("LightGBM") else "v2_all" for name in names]
        routed = np.empty((len(stored), len(names)))
        for ix in groups(stored.date.to_numpy()):
            base = stored.iloc[ix][baselines].to_numpy()
            keep = mask[ix]
            coords = np.searchsorted(np.flatnonzero(mask), ix[keep])
            values = route(base, keep, local[coords])
            np.testing.assert_array_equal(values[~keep], base[~keep])
            np.testing.assert_array_equal(np.sort(values[keep], axis=0), np.sort(base[keep], axis=0))
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite routed forecast in {family}")
            routed[ix] = values
        routed_frame = pd.DataFrame(routed, columns=names)
        keys = stored[["date", "ucid"]]
        routed_parts.append(pd.concat([keys, routed_frame], axis=1))
        covered_parts.append(pd.concat([keys[mask].reset_index(drop=True),
                                        pd.DataFrame(local, columns=names)], axis=1))
        scores = score_forecast_dates(test_panel, pd.concat([routed_frame, stored[controls]], axis=1))
        score_parts.append(scores.assign(fold=split.fold))
    choices = pd.DataFrame(selections, columns=["fold", "series", "original_arm", "parameter", "inner_corr"])
    scores = pd.concat(score_parts, ignore_index=True)
    original_names = choices.loc[~choices.series.str.startswith("LightGBM"), "original_arm"].unique()
    compare = pd.read_parquet(study / "scores.parquet",
                              filters=[("sample", "=", "routed"), ("arm", "in", list(original_names))])
    replay = scores.merge(choices[~choices.series.str.startswith("LightGBM")], on=["fold", "series"])
    validated = replay.merge(compare, left_on=["fold", "date", "original_arm"],
                             right_on=["fold", "date", "arm"], validate="many_to_one",
                             suffixes=("", "_original"))
    if len(validated) != len(replay):
        raise ValueError(f"Missing original scores for {family}")
    for metric in ("corr_all", "corr", "mmc"):
        np.testing.assert_allclose(validated[metric], validated[f"{metric}_original"],
                                   atol=1e-7, rtol=0, equal_nan=True)
    choices.to_csv(output / f"{family}_choices.csv", index=False)
    scores.to_parquet(output / f"{family}_scores.parquet", index=False)
    pd.concat(routed_parts).to_parquet(output / f"{family}_routed.parquet", index=False)
    pd.concat(covered_parts).to_parquet(output / f"{family}_covered.parquet", index=False)
    print(f"{family}: 16 configurations, {len(validated):,} original Ridge date scores reproduced",
          flush=True)


def aligned_forecasts(output, sample):
    frames = [pd.read_parquet(output / f"{family}_{sample}.parquet").set_index(["date", "ucid"])
              for family in FAMILIES]
    if any(not frame.index.is_unique for frame in frames):
        raise ValueError(f"Duplicate forecast keys on {sample}")
    if sample == "routed" and any(not frame.index.equals(frames[0].index) for frame in frames[1:]):
        raise ValueError("Routed forecast keys unexpectedly differ between proxies")
    combined = pd.concat(frames, axis=1, join="inner").sort_index()
    columns = [forecast_name(model, family, processing) for model in MODELS
               for family in FAMILIES for processing, _, _ in SETTINGS]
    return combined[columns]


def forecast_correlation(frame):
    width = frame.shape[1]
    totals = np.zeros((width, width))
    counts = np.zeros((width, width), dtype=int)
    for ix in groups(frame.index.get_level_values("date").to_numpy()):
        if len(ix) < 4:
            continue
        ranks = rankdata(frame.iloc[ix].to_numpy(), axis=0)
        ranks -= ranks.mean(axis=0)
        norms = np.linalg.norm(ranks, axis=0)
        present = norms > 0
        ranks[:, present] /= norms[present]
        totals += ranks.T @ ranks
        counts += np.outer(present, present)
    mean = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    np.testing.assert_allclose(mean, mean.T, atol=1e-12)
    return (pd.DataFrame(mean.clip(-1, 1), index=frame.columns, columns=frame.columns),
            pd.DataFrame(counts, index=frame.columns, columns=frame.columns))


def summarize_scores(output):
    rows = []
    for family in FAMILIES:
        scores = pd.read_parquet(output / f"{family}_scores.parquet")
        dates = scores.pivot(index="date", columns="series", values=["corr", "mmc", "payout", "corr_all"])
        for model in MODELS:
            baseline = {"standalone": None, "add_asset": "asset_only",
                        "add_v2": "v2_covered", "tree_add_v2": "tree_v2_covered"}[model]
            for processing, _, window in SETTINGS:
                name = forecast_name(model, family, processing)
                row = dict(series=name, proxy=PROXIES[family], model=MODELS[model],
                           processing=processing, window=window, baseline=baseline)
                for metric in ("corr", "mmc", "payout", "corr_all"):
                    row[metric] = dates[metric][name].mean()
                payout = dates["payout"][name].dropna()
                row["max_drawdown"] = max_drawdown(payout) if len(payout) else np.nan
                row["calmar"] = row["payout"] / row["max_drawdown"] if row["max_drawdown"] > 0 else np.nan
                row["meta_dates"] = dates["mmc"][name].count()
                if baseline:
                    for metric in ("corr", "mmc"):
                        delta, _, lo, hi, _, _ = stats(dates[metric][name] - dates[metric][baseline])
                        row.update({f"delta_{metric}": delta, f"{metric}_lo": lo, f"{metric}_hi": hi})
                rows.append(row)
    summary = pd.DataFrame(rows).sort_values("payout", ascending=False, kind="stable").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(output / "consolidated_scores.csv", index=False)
    return summary


def payout_leaderboard(summary):
    return pd.DataFrame({
        "Rank": summary["rank"],
        "Model": summary.model,
        "Proxy": summary.proxy,
        "Processing": summary.processing,
        "Payout": summary.payout.map("{:.5f}".format),
        "CORR": summary["corr"].map("{:.5f}".format),
        "Proxy MMC": summary.mmc.map("{:.5f}".format),
        "Max DD": summary.max_drawdown.map("{:.3f}".format),
        "Calmar": summary.calmar.map("{:.4f}".format),
        "CORR all": summary.corr_all.map("{:.5f}".format),
        "Δ MMC": summary.delta_mmc.map(lambda x: f"{x:+.5f}" if pd.notna(x) else "—"),
    })


def matrix_figure(matrix, title, subtitle):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fig = plt.figure(figsize=(15, 13), facecolor="white")
    fig.text(.065, .948, "NUMERAI CRYPTO   /   CONSOLIDATED VOLATILITY STUDY",
             color=TEAL, fontsize=10, weight="bold")
    fig.text(.065, .908, title, color=INK, fontsize=24, weight="bold")
    paragraph(fig, .876, subtitle, size=10, width=150)
    ax = fig.add_axes([.22, .18, .68, .63])
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu", interpolation="nearest")
    blocks = len(FAMILIES)
    size = len(SETTINGS)
    centers = np.arange(blocks * len(MODELS)) * size + (size - 1) / 2
    labels = [f"{short} · {proxy}" for short in ("R", "R+A", "R+V", "L+V")
              for proxy in PROXIES.values()]
    ax.xaxis.set_major_locator(FixedLocator(centers))
    ax.yaxis.set_major_locator(FixedLocator(centers))
    ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=7.4)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0, colors=MUTED)
    for position in range(size, len(matrix), size):
        major = position % (blocks * size) == 0
        color = "#263744" if major else "#ffffff"
        ax.axhline(position - .5, linewidth=1.2 if major else .35, color=color)
        ax.axvline(position - .5, linewidth=1.2 if major else .35, color=color)
    cbar = fig.colorbar(image, cax=fig.add_axes([.852, .19, .015, .61]))
    cbar.set_label("Mean daily Spearman correlation", fontsize=10, color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    paragraph(fig, .073, "Within each labeled block, from left to right and top to bottom: "
              "raw 120, own-history percentile 40, own-history bins 40, own-history bins 120.",
              size=9, width=166)
    paragraph(fig, .044, "R = proxy-only Ridge · R+A = Ridge with asset effects · "
              "R+V = Ridge with shipped v2.0 · L+V = LightGBM with shipped v2.0.",
              size=9, width=166)
    return fig


def within_model_medians(matrix):
    rows = []
    for label in MODELS.values():
        block = matrix.loc[matrix.index.str.startswith(label + " | "),
                           matrix.columns.str.startswith(label + " | ")]
        rows.append(float(np.nanmedian(block.to_numpy()[np.triu_indices(len(block), k=1)])))
    return rows


def write_report(output):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    summary = summarize_scores(output)
    correlations, samples = {}, {}
    for sample in ("routed", "covered"):
        frame = aligned_forecasts(output, sample)
        matrix, counts = forecast_correlation(frame)
        matrix.to_csv(output / f"prediction_correlations_{sample}.csv")
        counts.to_csv(output / f"correlation_date_counts_{sample}.csv")
        correlations[sample] = matrix
        samples[sample] = (len(frame), frame.index.get_level_values("date").nunique())
        del frame
    covers = pd.DataFrame({
        "Model": list(MODELS.values()),
        "Full universe": [f"{v:.3f}" for v in within_model_medians(correlations["routed"])],
        "Common covered": [f"{v:.3f}" for v in within_model_medians(correlations["covered"])],
    })
    add_ons = summary[summary.model.isin([MODELS["add_v2"], MODELS["tree_add_v2"]])]
    leader = add_ons.loc[add_ons.delta_mmc.idxmax()]

    def page(number, title, subtitle=""):
        fig = plt.figure(figsize=(15, 8.5), facecolor="white")
        fig.study_title = "Consolidated volatility study"
        section(fig, number, title, subtitle)
        return fig

    destination = output / "consolidated_volatility_report.pdf"
    with PdfPages(destination, metadata={"Title": "Consolidated Numerai Crypto volatility study"}) as pdf:
        fig = page(1, "Volatility proxies at four settings",
                   "Seven proxy families, four model configurations and 112 forecast series on the original development folds.")
        subhead(fig, .753, "Which settings are included")
        paragraph(fig, .706, "Raw → cross-sectional bins at 120 bars. Own-history percentile → cross-sectional bins "
                  "at 40 bars. Own-history bins → cross-sectional bins at 40 and 120 bars. EWMA windows are half-lives. "
                  "The calendar and estimator variant may differ by fold, including the downside, upside or joint "
                  "semivariance choice.")
        subhead(fig, .574, "Largest matched proxy MMC change among shipped-feature add-ons")
        paragraph(fig, .527,
                  f"{leader['model']} · {leader.proxy} · {leader.processing} had a mean proxy MMC change of "
                  f"{leader.delta_mmc:+.5f} with a 95% block interval of [{leader.mmc_lo:+.5f}, "
                  f"{leader.mmc_hi:+.5f}]. The windows were chosen after the original sweep on these dates, "
                  "so this is a descriptive maximum.")
        subhead(fig, .423, "Median pairwise prediction correlation within each model")
        table(fig, covers, [.065, .201, .87, .177], [.48, .26, .26])
        paragraph(fig, .151,
                  f"The routed sample contains {samples['routed'][0]:,} asset-date rows. "
                  f"The common covered sample contains {samples['covered'][0]:,} rows. "
                  "Routing and shipped predictors can both contribute to similarity on the full universe.")
        save(pdf, fig)

        for i, (model, model_label) in enumerate(MODELS.items(), start=2):
            data = summary[summary.model.eq(model_label)]
            suff = "Each cell gives absolute CORR / proxy MMC."
            if model != "standalone":
                suff += " The second line gives the matched Δ proxy MMC."
            fig = page(i, model_label, suff + " All scores are from routed forecasts on the same 468 meta-covered dates.")
            entries = []
            for proxy in PROXIES.values():
                group = data[data.proxy.eq(proxy)].set_index("processing").loc[
                    [kind for kind, _, _ in SETTINGS]
                ].reset_index()
                values = [proxy]
                for row in group.itertuples(index=False):
                    value = f"{row.corr:.5f} / {row.mmc:.5f}"
                    if model != "standalone":
                        value += f"\n{row.delta_mmc:+.5f}"
                    values.append(value)
                entries.append(values)
            frame = pd.DataFrame(entries, columns=["Proxy", *[kind for kind, _, _ in SETTINGS]])
            table(fig, frame, [.065, .253, .87, .50], [.24, .19, .19, .19, .19], font=10)
            if model == "standalone":
                footer = "Stand-alone Ridge fits use the proxy alone on covered rows. Baseline ranks are retained on uncovered rows."
            elif model == "add_asset":
                footer = "Matched comparator uses permanent-asset effects alone, with the same covered training and routing."
            else:
                footer = "Matched comparator uses the 22 shipped features, with the same covered training and routing."
            paragraph(fig, .199, footer + " See the score CSV for the paired CORR and proxy MMC intervals.")
            save(pdf, fig)

        for sample, title, description in (
            ("routed", "Prediction correlations on the full universe",
             "Uses each model’s routed out-of-fold forecasts. Uncovered assets retain their full-universe baseline ranks."),
            ("covered", "Prediction correlations on common covered assets",
             "Uses covered-model forecasts before routing. All pairs share the intersection of covered asset-date rows across proxy families."),
        ):
            nrows, ndates = samples[sample]
            fig = matrix_figure(correlations[sample], title,
                                f"Mean within-date Spearman correlations across {ndates:,} development dates "
                                f"and {nrows:,} matched asset-date rows. {description}")
            fig.savefig(output / f"prediction_heatmap_{sample}.png", dpi=180)
            save(pdf, fig)

        fig = page(8, "Evaluation and provenance")
        for y, heading, body in (
            (.797, "Fold selection and replay", "For each Ridge model and each fixed processing/window pair, the original "
             "inner-validation CORR selects the calendar, estimator implementation and side. Ridge reuses the recorded "
             "alpha. LightGBM uses the variant nominated by Ridge + v2 for that fold and tunes depth and tree count over "
             "the original four-point grid. Each model predicts the original 130-date outer block."),
            (.634, "Matched evaluation", "Forecasts inherit the original proxy-specific covered cohorts and D−1 feature cutoff. "
             "Every covered comparator shares its cohort and training calendar. The 130-date refit has a 22-date purge "
             "before prediction. Inner selection uses 40 validation dates with its own 22-date purge. Headline CORR, "
             "proxy MMC and payout share the 468 dates with historical meta predictions."),
            (.466, "Prediction correlations", "The matrices average within-date Spearman correlations with equal date weight. "
             "Each date uses matched permanent asset IDs. Average ranks preserve ties before routing. A pair with a "
             "constant forecast on a date excludes that date, and the companion count matrices give the exact denominators."),
            (.303, "PASS WITH CONDITIONS", "Ridge date scores match the original artifacts within 1e−7. Replay checks verify "
             "identical key and mask alignment, finite routed forecasts, exact uncovered ranks and preservation of "
             "covered rank slots. The underlying OHLC snapshot was retrospective and 224 of 3,982 windows remained "
             "uncached. Historical source availability and independent live parity remain unverified."),
        ):
            subhead(fig, y, heading)
            paragraph(fig, y - .045, body)
        save(pdf, fig)

        leaderboard = payout_leaderboard(summary)
        rows_per_page = 28
        for start in range(0, len(leaderboard), rows_per_page):
            rows = leaderboard.iloc[start:start + rows_per_page]
            fig = page(9 + start // rows_per_page, "Performance ranked by payout",
                       f"Ranks {rows.Rank.iloc[0]}–{rows.Rank.iloc[-1]} of {len(leaderboard)}. "
                       "Mean payout = 0.1 × CORR + proxy MMC on the 468 matched meta dates.")
            table(fig, rows, [.05, .148, .90, .61],
                  [.047, .184, .11, .103, .081, .078, .089, .071, .074, .08, .083], font=8.5)
            paragraph(fig, .122, "Max DD is maximum cumulative payout drawdown. Calmar is mean payout divided by "
                      "Max DD. CORR all uses all 1,430 test dates. Δ MMC is relative to the matched control.",
                      size=9, width=160)
            save(pdf, fig)
    print(f"{destination}: {len(summary)} forecast series", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("results/volatility_proxies"))
    parser.add_argument("--panel", type=Path, default=Path("data/crypto/v2.0/train.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("data/crypto/v2.0/historical_meta_models.parquet"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=len(FAMILIES))
    parser.add_argument("--report-only", action="store_true", help="Reuse completed replay forecasts")
    args = parser.parse_args()
    output = args.output or args.study_root / "consolidated"
    output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        tasks = [(args.study_root, output, args.panel, args.meta, family) for family in FAMILIES]
        with ProcessPoolExecutor(args.workers, mp_context=get_context("fork")) as pool:
            list(pool.map(replay_family, tasks))
    write_report(output)


if __name__ == "__main__":
    main()
