"""Run one development-only, cross-fitted Numerai Crypto study per volatility family."""

import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ndtri
from scipy.stats import rankdata
from lightgbm import LGBMRegressor
from numerai_tools.scoring import correlation_contribution, numerai_corr

from src.data_mani.fetch_volatility_prices import CUTOFF
from src.data_mani.utils import fill_missing_features
from src.data_mani.volatility_proxies import (
    FAMILIES, VIEWS, bins, build_asset, catalog, clean_bars,
)
from src.live.submit import rank_normalise

TARGET = "target_binned_return_20"
ALPHAS = np.logspace(-4, 2, 13)
TRAIN, PURGE, VALID, TEST = 130, 22, 40, 130
TREE_DEPTHS = (3, 5)
TREE_ITERATIONS = (100, 250)
ANCHORS = {
    "ewma": ("recursive_log", "ewma"),
    "semivariance": ("zero_log", "joint"),
    "idiosyncratic": ("btc", "residual"),
    "parkinson": ("standard", "range"),
    "garman_klass": ("standard", "range"),
    "rogers_satchell": ("standard", "range"),
    "yang_zhang": ("standard", "range"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def groups(date_codes):
    return np.split(np.arange(len(date_codes)), np.flatnonzero(np.diff(date_codes)) + 1)


def route(base, covered, prediction):
    result = base.copy()
    slots = np.sort(base[covered], axis=0)
    # Inputs have a permanent-ID secondary sort. Stable sorting gives the same
    # tie treatment as src.live.submit.rank_normalise.
    order = np.argsort(prediction, axis=0, kind="stable")
    np.put_along_axis(result, np.flatnonzero(covered)[order], slots, axis=0)
    return result


def rank_columns(prediction):
    return (np.argsort(np.argsort(prediction, axis=0, kind="stable"),
                       axis=0, kind="stable") + .5) / len(prediction)


def cc_fast(predictions, target):
    """Vectorized Numerai CORR for the inner grid, checked against official scoring."""
    normal = ndtri((rankdata(predictions, axis=0) - .5) / len(target))
    x = np.sign(normal) * np.abs(normal) ** 1.5
    centered = target - target.mean()
    y = np.sign(centered) * np.abs(centered) ** 1.5
    x -= x.mean(axis=0)
    y -= y.mean()
    scale = np.linalg.norm(x, axis=0) * np.linalg.norm(y)
    return np.divide(y @ x, scale, out=np.zeros_like(scale), where=scale > 0)


def ridge_grid(x, y, xe, ids, eval_ids, alphas):
    """Ridge with an intercept, optionally adding penalized asset dummies."""
    mean_x, mean_y = x.mean(axis=0), y.mean()
    if ids is None:
        xc = x - mean_x
        eig, basis = np.linalg.eigh(xc.T @ xc)
        weights = basis @ ((basis.T @ (xc.T @ (y - mean_y)))[:, None]
                           / (np.maximum(eig, 0)[:, None] + alphas))
        return (xe - mean_x) @ weights + mean_y

    # Eliminate the diagonal asset dummy block analytically, leaving only
    # the small feature/intercept block. Unseen assets have zero offsets.
    x = np.column_stack((np.ones(len(x)), x))
    xe = np.column_stack((np.ones(len(xe)), xe))
    size = max(ids.max(), eval_ids.max()) + 1
    counts = np.bincount(ids, minlength=size)
    asset_x = np.column_stack([
        np.bincount(ids, weights=column, minlength=size) for column in x.T
    ])
    asset_y = np.bincount(ids, weights=y, minlength=size)
    xy, xx = x.T @ y, x.T @ x
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    predictions = []
    for alpha in alphas:
        shrink = 1 / (counts + alpha)
        cross = asset_x.T * shrink
        coef = np.linalg.solve(xx + alpha * penalty - cross @ asset_x,
                               xy - cross @ asset_y)
        offsets = (asset_y - asset_x @ coef) * shrink
        predictions.append(xe @ coef + offsets[eval_ids])
    return np.column_stack(predictions)


def tree_grid(x, y, xe, parameters):
    predictions = []
    for depth, iterations in parameters:
        model = LGBMRegressor(max_depth=depth, num_leaves=2 ** depth,
                             n_estimators=iterations, learning_rate=.05,
                             min_child_samples=100, verbosity=-1, n_jobs=1,
                             random_state=0)
        model.fit(x, y)
        predictions.append(model.booster_.predict(xe, num_threads=1))
    return np.column_stack(predictions)


def cross_section_bins(matrix, dates):
    reduced = np.ascontiguousarray(matrix)
    for ix in groups(dates):
        rank = (rankdata(reduced[ix], axis=0) - .5) / len(ix)
        reduced[ix] = bins(rank)
    return reduced.astype(np.float32)


def feature_matrix(panel, candles, family, output):
    configs = catalog(family)
    price_groups = {ucid: clean_bars(group) for ucid, group in candles.groupby("ucid")}
    benchmark = ({ucid: price_groups[ucid].close for ucid in (1, 1027)}
                 if family == "idiosyncratic" else {})
    unique = list(dict.fromkeys((*c[:3], side) for c in configs
                                for side in (("down", "up") if c[3] == "joint" else (c[3],))))
    positions = {key: i for i, key in enumerate(unique)}
    raw = np.full((len(panel), len(unique)), np.nan, dtype=np.float64)
    hist = np.full(raw.shape, np.nan, dtype=np.float32)
    checked = 0
    for ucid, rows in panel.groupby("ucid", sort=False):
        if ucid not in price_groups or (family == "idiosyncratic" and ucid in benchmark):
            continue
        source = price_groups[ucid]
        dates = pd.DatetimeIndex(rows.date)
        asset = build_asset(dates, source, benchmark, family, configs)
        if checked < 12 and len(dates) >= 260:
            cut = dates[len(dates) * 3 // 4]
            prior = dates <= cut
            replay = build_asset(dates[prior], source.loc[source.index < cut],
                                 {i: s.loc[s.index < cut] for i, s in benchmark.items()},
                                 family, configs)
            for key, (values, histories) in replay.items():
                np.testing.assert_allclose(values, asset[key][0][prior], rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(histories, asset[key][1][prior])
            checked += 1
        for key, (values, histories) in asset.items():
            sides = ("down", "up") if family == "semivariance" else (
                next(c[3] for c in configs if c[:3] == key),)
            for j, side in enumerate(sides):
                col = positions[(*key, side)]
                raw[rows.index, col] = values[:, j]
                hist[rows.index, col] = histories[:, j]
    covered = np.isfinite(hist).all(axis=1) & np.isfinite(raw).all(axis=1)
    coverage = panel[["date", "ucid"]].copy()
    coverage["covered"] = covered
    coverage["raw_any"] = np.isfinite(raw).any(axis=1)
    coverage["raw_all"] = np.isfinite(raw).all(axis=1)
    coverage.to_parquet(output / "coverage.parquet", index=False)
    data = {}
    for view, matrix in zip(VIEWS, (raw, hist, bins(hist)), strict=True):
        # Keep candidate values/missingness on the original panel in storage.
        values = pd.DataFrame(matrix, columns=["/".join(map(str, key)) for key in unique])
        pd.concat([panel[["date", "ucid"]], values], axis=1).to_parquet(
            output / f"features_{view}.parquet", index=False)
        data[view] = cross_section_bins(matrix[covered], panel.date.to_numpy()[covered])
    counts = pd.DataFrame([
        {"calendar": c[0], "method": c[1], "window": c[2], "side": c[3],
         "raw_rows": int(np.isfinite(raw[:, ix]).all(axis=1).sum()),
         "history_rows": int(np.isfinite(hist[:, ix]).all(axis=1).sum())}
        for c in configs for ix in [config_columns(positions, c)]
    ])
    counts.to_csv(output / "coverage_by_setting.csv", index=False)
    return covered, data, positions, checked


def config_columns(positions, config):
    sides = ("down", "up") if config[3] == "joint" else (config[3],)
    return [positions[(*config[:3], side)] for side in sides]


def load(panel_path, meta_path):
    panel = pd.read_parquet(panel_path)
    if "symbol" in panel.index.names:
        panel = panel.reset_index()
    panel = panel.loc[panel.date.le(CUTOFF)].copy()
    panel["ucid"] = panel.ucid.astype(int)
    meta = pd.read_parquet(meta_path, columns=["id", "date", "prediction"])
    meta["id"] = meta.id.astype(int)
    meta = meta[meta.date.le(CUTOFF)].rename(columns={"id": "ucid", "prediction": "meta"})
    panel = panel.merge(meta, on=["date", "ucid"], how="left", validate="one_to_one")
    panel = panel.sort_values(["date", "ucid"], ignore_index=True)
    features = [col for col in panel if col.startswith("feature_")]
    if len(features) != 22 or panel[TARGET].isna().any():
        raise ValueError("Unexpected shipped feature or target schema")
    shipped = fill_missing_features(panel[features], features, .5, verbose=False)
    return panel, shipped.to_numpy(dtype=float)


def controls(panel, candles, shipped):
    counts = candles.groupby("ucid").size()
    never = panel.ucid.map(counts).isna()
    # This intentionally leaks look-ahead. Counts come from the downloaded
    # retrospective snapshot, including candles after many prediction dates.
    vendor = np.log1p(panel.ucid.map(counts).fillna(0))
    age = (panel.date - panel.groupby("ucid").date.transform("min")).dt.days
    return {
        "placebo_never_observed": np.column_stack([shipped, never]),
        "placebo_vendor_history": np.column_stack([shipped, vendor]),
        "causal_panel_age": np.column_stack([shipped, np.log1p(age)]),
    }


def arms_for(family, transforms, positions):
    arms = {}
    for config in catalog(family):
        columns = config_columns(positions, config)
        for view, matrix in transforms.items():
            for mode in ("standalone", "add_v2", "add_asset"):
                arms[(*config, view, mode)] = (matrix, columns)
    return arms


def score_date(matrix, target, meta):
    corr = cc_fast(matrix, target)
    mask = np.isfinite(meta)
    missing = np.full(matrix.shape[1], np.nan)
    if mask.sum() < 4:
        return corr, missing, missing.copy()
    # CORR and proxy MMC use the same observed-meta asset sample.
    y, predictions = target[mask], matrix[mask]
    matched_corr = cc_fast(predictions, y)
    n = len(y)
    p = ndtri((rankdata(predictions, axis=0) - .5) / n)
    m = ndtri((rankdata(meta[mask]) - .5) / n)
    meta_energy = m @ m
    neutral = p - m[:, None] * ((m @ p) / meta_energy) if meta_energy else p
    scaled_y = y * 4 if (y >= 0).all() and (y <= 1).all() else y
    return corr, matched_corr, ((scaled_y - scaled_y.mean()) @ neutral) / n


def check_scoring():
    rng = np.random.default_rng(0)
    predictions = rng.integers(0, 5, (59, 8))
    target = rng.integers(0, 5, 59) / 4
    meta = rng.random(59)
    meta[::4] = np.nan
    corr, matched_corr, mmc = score_date(predictions, target, meta)
    preds = pd.DataFrame(predictions)
    y = pd.Series(target, name="target")
    present = np.isfinite(meta)
    np.testing.assert_allclose(corr, numerai_corr(preds, y), atol=1e-12)
    np.testing.assert_allclose(matched_corr, numerai_corr(preds[present], y[present]), atol=1e-12)
    np.testing.assert_allclose(mmc, correlation_contribution(
        preds[present], pd.Series(meta, name="meta")[present], y[present].copy()), atol=1e-12)


def select_model(x, y, fit, val, test, date_code, ids=None, tree=False,
                 columns=None, base=None):
    grid = [(depth, trees) for depth in TREE_DEPTHS for trees in TREE_ITERATIONS] if tree else ALPHAS

    def take(rows):
        values = x[rows] if columns is None else x[np.ix_(rows, columns)]
        return values if base is None else np.column_stack((base[rows], values))

    def predict(train, evaluate, choices):
        xt, xe = take(train), take(evaluate)
        if tree:
            return tree_grid(xt, y[train], xe, choices)
        train_ids = ids[train] if ids is not None else None
        eval_ids = ids[evaluate] if ids is not None else None
        return ridge_grid(xt, y[train], xe, train_ids, eval_ids, choices)

    pred = predict(fit, val, grid)
    date_groups = groups(date_code[val])
    scores = np.mean([cc_fast(pred[ix], y[val[ix]]) for ix in date_groups], axis=0)
    pick = int(np.argmax(scores.round(12)))
    train = np.flatnonzero((date_code >= date_code[fit[0]]) & (date_code <= date_code[val[-1]]))
    selection = str(grid[pick]) if tree else float(grid[pick])
    return predict(train, test, grid[pick:pick + 1])[:, 0], selection, float(scores[pick])


def record_scores(fold, dates, matrix, cov_matrix, covered, panel, names, score_rows):
    for i, ix in enumerate(dates):
        ix, date = ix
        y = panel[TARGET].to_numpy()[ix]
        meta = panel.meta.to_numpy()[ix]
        all_corr, matched_corr, mmc = score_date(matrix[i], y, meta)
        for j, name in enumerate(names):
            score_rows.append((fold, date, name, "routed", all_corr[j], matched_corr[j], mmc[j]))
        mask = covered[ix]
        if mask.sum() >= 4:
            cov_names, cov_predictions = cov_matrix[i]
            all_corr, matched_corr, mmc = score_date(cov_predictions, y[mask], meta[mask])
            for j, name in enumerate(cov_names):
                score_rows.append((fold, date, name, "covered", all_corr[j], matched_corr[j], mmc[j]))


_fold_job = None


def initialize_fold_worker(job):
    global _fold_job
    _fold_job = job


def execute_fold(item):
    return _fold_job(*item)


def run_folds(panel, shipped, candles, family, covered, transforms, positions, output, workers=1):
    y = panel[TARGET].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(panel.date.unique())
    date_code = dates.get_indexer(panel.date)
    ids = pd.factorize(panel.ucid, sort=True)[0]
    covered_rows = np.flatnonzero(covered)
    cy, cd, ci = y[covered], date_code[covered], ids[covered]
    ctrl = controls(panel, candles, shipped)
    covered_shipped = shipped[covered]
    arms = arms_for(family, transforms, positions)
    arm_ids = list(arms)
    labels = {key: "/".join(map(str, key)) for key in arms}
    settings = pd.DataFrame(arm_ids, columns=["calendar", "method", "window", "side", "view", "model"])
    settings["arm"] = [labels[key] for key in arm_ids]
    settings.to_csv(output / "settings.csv", index=False)
    anchor_method, anchor_side = ANCHORS[family]
    anchor_keys = [
        ("weekday", anchor_method, window, anchor_side, view, "add_v2")
        for window in (20, 60) for view in ("raw", "history_percentile")
    ]

    def run_fold(fold, start):
        pick_rows = []
        split, last_train = start + TRAIN - VALID, start + TRAIN
        first_test = last_train + PURGE
        last_test = first_test + TEST
        fold_dates = (fold, dates[start], dates[split - PURGE - 1], dates[split],
                      dates[last_train - 1], dates[first_test], dates[last_test - 1])

        def indices(codes):
            return (
                np.flatnonzero((codes >= start) & (codes < split - PURGE)),
                np.flatnonzero((codes >= split) & (codes < last_train)),
                np.flatnonzero((codes >= first_test) & (codes < last_test)),
            )

        f, v, t = indices(date_code)
        cf, cv, ct = indices(cd)
        if min(len(cf), len(cv), len(ct)) == 0:
            raise ValueError(f"No matching covered observations in fold {fold}")
        full_predictions = {}
        candidate_predictions = {}

        def run(name, x, cohort=False, asset=False, tree=False, columns=None, add_v2=False):
            if cohort:
                pr, param, score = select_model(x, cy, cf, cv, ct, cd,
                                                ci if asset else None, tree, columns,
                                                covered_shipped if add_v2 else None)
                candidate_predictions[name] = pr
            else:
                pr, param, score = select_model(x, y, f, v, t, date_code, tree=tree)
                full_predictions[name] = pr
            pick_rows.append((fold, name, param, score))
            return score

        run("v2_all", shipped)
        run("v2_covered", covered_shipped, cohort=True)
        run("asset_only", np.empty((len(cy), 0)), cohort=True, asset=True)
        run("tree_v2_covered", covered_shipped, cohort=True, tree=True)
        run("tree_v2_all", shipped, tree=True)
        for name, x in ctrl.items():
            run(name, x)
        inner = {}
        for key, (matrix, columns) in arms.items():
            mode = key[-1]
            inner[key] = run(labels[key], matrix, cohort=True, asset=mode == "add_asset",
                             columns=columns, add_v2=mode == "add_v2")
        for key in anchor_keys:
            matrix, columns = arms[key]
            run("tree/" + labels[key], matrix, cohort=True, tree=True, columns=columns, add_v2=True)
        chosen = {
            "selected_" + mode: max((key for key in arm_ids if key[-1] == mode),
                                    key=lambda key: round(inner[key], 12))
            for mode in ("add_v2", "standalone", "add_asset")
        }
        for name, key in chosen.items():
            candidate_predictions[name] = candidate_predictions[labels[key]]
            pick_rows.append((fold, name, labels[key], inner[key]))

        names = list(full_predictions) + list(candidate_predictions)
        full = np.column_stack(list(full_predictions.values()))
        candidates = np.column_stack(list(candidate_predictions.values()))
        tree = np.array([name.startswith("tree") for name in candidate_predictions])
        routed_dates, covered_dates, dates_ix, audit_frames = [], [], [], []
        audit_columns = ["v2_all", "v2_covered", "asset_only", "selected_add_v2",
                         "selected_standalone", "selected_add_asset",
                         "tree_v2_all", "tree_v2_covered"]
        candidate_rows = covered_rows[ct]
        for ix in groups(date_code[t]):
            universe = t[ix]
            mask = covered[universe]
            candidate_ix = np.searchsorted(candidate_rows, universe[mask])
            np.testing.assert_array_equal(candidate_rows[candidate_ix], universe[mask])
            local = candidates[candidate_ix]
            ranked_full = rank_columns(full[ix])
            base = np.where(tree[None, :], ranked_full[:, [list(full_predictions).index("tree_v2_all")]],
                            ranked_full[:, [0]])
            routed = route(base, mask, local)
            np.testing.assert_array_equal(ranked_full[:, 0], rank_normalise(full[ix, 0]))
            np.testing.assert_array_equal(base[~mask], routed[~mask])
            np.testing.assert_array_equal(np.sort(base[mask], axis=0),
                                          np.sort(routed[mask], axis=0))
            joined = np.column_stack([ranked_full, routed])
            if not np.isfinite(joined).all():
                raise ValueError("Non-finite routed prediction")
            routed_dates.append(joined)
            covered_dates.append((list(candidate_predictions), local))
            date = panel.date.iloc[universe[0]]
            dates_ix.append((universe, date))
            frame = panel.iloc[universe][["date", "ucid"]].assign(covered=mask, fold=fold)
            for name in audit_columns:
                frame[name] = joined[:, names.index(name)]
            audit_frames.append(frame)
        score_rows = []
        record_scores(fold, dates_ix, routed_dates, covered_dates, covered,
                      panel, names, score_rows)
        score_part = output / f"scores_fold_{fold:02d}.parquet"
        pd.DataFrame(score_rows, columns=[
            "fold", "date", "arm", "sample", "corr_all", "corr", "mmc",
        ]).to_parquet(score_part, index=False)
        pd.concat(audit_frames).to_parquet(output / f"routed_fold_{fold:02d}.parquet", index=False)
        print(f"{family}: fold {fold + 1} through {dates[last_test - 1]:%Y-%m-%d}", flush=True)
        return fold_dates, pick_rows, score_part

    offset = (len(dates) - TRAIN - PURGE) % TEST
    schedule = list(enumerate(range(offset, len(dates) - TRAIN - PURGE - TEST + 1, TEST)))
    if workers > 1:
        # Fork shares the read-only feature matrix across independent fold fits.
        with get_context("fork").Pool(workers, initialize_fold_worker, (run_fold,)) as pool:
            results = pool.map(execute_fold, schedule)
    else:
        results = [run_fold(*item) for item in schedule]
    fold_rows, picks_by_fold, score_parts = zip(*results, strict=True)
    pick_rows = [row for picks in picks_by_fold for row in picks]
    pd.DataFrame(fold_rows, columns=[
        "fold", "train_start", "inner_train_end", "validation_start", "train_end",
        "test_start", "test_end",
    ]).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(pick_rows, columns=["fold", "arm", "selection", "inner_corr"]).to_csv(
        output / "picks.csv", index=False)
    temporary = output / "scores.tmp"
    with pq.ParquetWriter(temporary, pq.read_schema(score_parts[0])) as writer:
        for path in score_parts:
            writer.write_table(pq.read_table(path))
    temporary.replace(output / "scores.parquet")
    for path in score_parts:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--panel", type=Path, default=Path("data/crypto/v2.0/train.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("data/crypto/v2.0/historical_meta_models.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/volatility_proxies"))
    parser.add_argument("--fold-workers", type=int, default=1,
                        help="Fit independent folds in parallel on Unix")
    args = parser.parse_args()
    check_scoring()
    output = args.output / args.family
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    panel, shipped = load(args.panel, args.meta)
    source = args.output / "ohlc.parquet"
    candles = pd.read_parquet(source)
    if candles.duplicated(["ucid", "bar_date"]).any():
        raise ValueError("Source candles must have unique asset and date")
    # Check ticker changes without ever using them to join prices or the meta model.
    ticker_overlap = panel[["ucid", "symbol"]].drop_duplicates().merge(
        candles[["ucid", "source_symbol"]].drop_duplicates(), on="ucid")
    mismatches = ticker_overlap[ticker_overlap.symbol.str.upper() != ticker_overlap.source_symbol.str.upper()]
    mismatches.to_csv(output / "ticker_diagnostics.csv", index=False)
    ticker_keys = panel[["ucid", "symbol"]].drop_duplicates()
    ticker_reuse = ticker_keys.groupby(ticker_keys.symbol.str.upper()).ucid.nunique()
    ticker_reuse[ticker_reuse > 1].to_csv(output / "ticker_reuse.csv", header=["distinct_ucid"])
    print(f"{args.family}: {len(panel):,} panel rows. Building features.", flush=True)
    covered, transformed, positions, checked = feature_matrix(panel, candles, args.family, output)
    print(f"{args.family}: {covered.sum():,} common covered rows across {checked} cutoff replays", flush=True)
    run_folds(panel, shipped, candles, args.family, covered, transformed, positions,
              output, workers=args.fold_workers)
    write_manifest(output, args.family, (args.panel, args.meta, source, args.output / "price_requests.json"),
                   panel, covered, checked, len(mismatches), int(ticker_reuse.gt(1).sum()))


def write_manifest(output, family, inputs, panel, covered, checked, mismatches, reuse):
    manifest = {
        "study": family, "max_panel_date": str(CUTOFF.date()),
        "inputs": {str(path): digest(path) for path in inputs},
        "panel_rows": len(panel), "covered_rows": int(covered.sum()),
        "covered_ids": int(panel.loc[covered, "ucid"].nunique()),
        "tested_cutoffs": checked, "ticker_diagnostic_rows": mismatches,
        "reused_ticker_count": reuse,
        "ridge_alphas": ALPHAS.tolist(), "tree_grid": [TREE_DEPTHS, TREE_ITERATIONS],
        "target": TARGET,
    }
    (output / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
