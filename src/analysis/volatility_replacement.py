"""Run matched scratch fits on original and volatility-replaced Numerai v2.0."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.analysis.volatility_experiment import (
    ALPHAS, TARGET, TREE_DEPTHS, TREE_ITERATIONS, cc_fast, check_scoring, digest,
    groups, load, rank_columns, route, score_date, select_model,
)
from src.data_mani.build_volatility_replacement import NEW, REMOVED

MODELS = ("Ridge", "LightGBM", "MLP")
ARMS = ("original_full", "original_matched", "replacement")
SEEDS = (0, 1, 2)
MLP_EPOCHS = 25
MLP_GRID = [(units, alpha) for units in ((32,), (64, 32)) for alpha in (.0001, .01)]


def cuda_mlp(x, y, xe, units, alpha, seed, epochs):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x, y, xe = (torch.as_tensor(v, dtype=torch.float32, device="cuda") for v in (x, y, xe))
    layers = []
    previous = x.shape[1]
    for width in units:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, 1))
    model = nn.Sequential(*layers).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    weights = [layer.weight for layer in model if isinstance(layer, nn.Linear)]
    for _ in range(epochs):
        order = torch.randperm(len(x), device=x.device)
        for batch in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            residual = model(x[batch]).flatten() - y[batch]
            penalty = sum(weight.square().sum() for weight in weights)
            loss = .5 * residual.square().mean() + .5 * alpha * penalty / len(batch)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        return model(xe).flatten().cpu().numpy()


def mlp_predict(x, y, xe, parameters, seeds, epochs, device):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    xe = scaler.transform(xe)
    predictions = []
    for units, alpha in parameters:
        fitted = []
        for seed in seeds:
            if device == "cuda":
                fitted.append(cuda_mlp(x, y, xe, units, alpha, seed, epochs))
                continue
            model = MLPRegressor(hidden_layer_sizes=units, activation="relu", solver="adam",
                                 alpha=alpha, batch_size=256, learning_rate_init=.001,
                                 random_state=np.random.RandomState(seed))
            for _ in range(epochs):
                model.partial_fit(x, y)
            fitted.append(model.predict(xe))
        predictions.append(np.mean(fitted, axis=0))
    return np.column_stack(predictions)


def fit_model(model, x, y, dates, fold, epochs, device):
    train = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
    inner = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.inner_train_end))
    validate = np.flatnonzero((dates >= fold.validation_start) & (dates <= fold.train_end))
    test = np.flatnonzero((dates >= fold.test_start) & (dates <= fold.test_end))
    if not all(len(ix) for ix in (train, inner, validate, test)):
        raise ValueError(f"Empty split {fold.fold}")
    if model != "MLP":
        pred, param, score = select_model(x, y, inner, validate, test, dates, tree=model == "LightGBM")
        return pred, param, score

    original = x.astype(np.float32, copy=False)
    inner_preds = mlp_predict(original[inner], y[inner], original[validate], MLP_GRID, SEEDS[:1], epochs, device)
    scores = np.mean([cc_fast(inner_preds[ix], y[validate[ix]])
                      for ix in groups(dates[validate])], axis=0)
    pick = int(np.argmax(scores.round(12)))
    preds = mlp_predict(original[train], y[train], original[test], MLP_GRID[pick:pick + 1], SEEDS, epochs, device)
    return preds[:, 0], repr(MLP_GRID[pick]), float(scores[pick])


def model_inputs(panel_path, meta_path, replacement_path):
    base = pd.read_parquet(panel_path)
    replacement = pd.read_parquet(replacement_path)
    pd.testing.assert_frame_equal(replacement.drop(columns=list(NEW)),
                                  base.drop(columns=list(REMOVED)))
    candidate = replacement[["ucid", "date", *NEW]].reset_index(drop=True)
    candidate["ucid"] = candidate.ucid.astype(int)
    panel, original = load(panel_path, meta_path)
    aligned = panel[["ucid", "date"]].merge(candidate, on=["ucid", "date"], validate="one_to_one",
                                          how="left", sort=False, indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Missing replacement asset-date keys")
    pd.testing.assert_frame_equal(panel[["ucid", "date"]], aligned[["ucid", "date"]])
    values = aligned[list(NEW)].to_numpy(dtype=float)
    covered = np.isfinite(values).all(axis=1)
    if np.isfinite(values[~covered]).any() or not np.isin(values[covered], [0, .25, .5, .75, 1]).all():
        raise ValueError("Invalid candidate bins or coverage block")
    retained = [j for j, col in enumerate(c for c in panel if c.startswith("feature_")) if col not in REMOVED]
    replacement_inputs = np.column_stack((original[covered][:, retained], values[covered]))
    return panel, original, covered, replacement_inputs


def write_fold(fold, panel, original, covered, replacement, output, epochs, device):
    dates = panel.date.to_numpy()
    y = panel[TARGET].to_numpy(dtype=float)
    dates_covered = dates[covered]
    test = (dates >= fold.test_start) & (dates <= fold.test_end)
    is_covered = covered[test]
    evaluated = panel.loc[test].reset_index(drop=True)
    pd.testing.assert_index_equal(pd.Index(evaluated.date.unique()),
                                  pd.Index(evaluated.loc[is_covered, "date"].unique()))
    scores, picks = [], []
    routed_frame = evaluated[["date", "ucid"]].assign(covered=is_covered)
    coverage_keys = np.flatnonzero(is_covered)
    original_covered = original[covered]
    for model in MODELS:
        predictions = {}
        for arm, cols in zip(ARMS, (original, original_covered, replacement), strict=True):
            yc, dc = (y, dates) if arm == "original_full" else (y[covered], dates_covered)
            predicted, parameter, score = fit_model(model, cols, yc, dc, fold, epochs, device)
            predictions[arm] = predicted
            picks.append((fold.fold, model, arm, parameter, score))
        candidates = np.column_stack([predictions[arm] for arm in ARMS[1:]])
        full = np.empty((len(evaluated), len(ARMS)))
        for ix in groups(evaluated.date.to_numpy()):
            mask = is_covered[ix]
            base = rank_columns(predictions["original_full"][ix])
            repeated = np.repeat(base[:, None], 2, axis=1)
            local = candidates[np.searchsorted(coverage_keys, ix[mask])]
            reranked = route(repeated, mask, local)
            np.testing.assert_array_equal(reranked[~mask], repeated[~mask])
            np.testing.assert_array_equal(np.sort(reranked[mask], axis=0), np.sort(repeated[mask], axis=0))
            full[ix] = np.column_stack((base, reranked))
            if not np.isfinite(full[ix]).all():
                raise ValueError("Routed forecasts must be finite")
            targets = evaluated[TARGET].to_numpy()[ix]
            meta = evaluated.meta.to_numpy()[ix]
            date = evaluated.date.iloc[ix[0]]
            samples = [(ARMS, "routed", full[ix], targets, meta)]
            if mask.sum() >= 4:
                samples.append((ARMS[1:], "covered", local, targets[mask], meta[mask]))
            for arms, sample, preds, ys, ref in samples:
                values = score_date(preds, ys, ref)
                scores.extend((fold.fold, date, model, arm, sample, ca, corr, mmc)
                              for arm, ca, corr, mmc in zip(arms, *values, strict=True))
        for j, arm in enumerate(ARMS):
            routed_frame[f"{model}/{arm}"] = full[:, j]
    routed_frame.to_parquet(output / f"routed_fold_{fold.fold:02d}.parquet", index=False)
    print(f"Fold {fold.fold + 1} through {fold.test_end:%Y-%m-%d}: all three models complete", flush=True)
    return scores, picks


_WORK = None


def init_workers(panel, original, covered, replacement, output, epochs, device, folds):
    global _WORK
    _WORK = (panel, original, covered, replacement, output, epochs, device, folds)


def run_worker(index):
    *args, folds = _WORK
    return write_fold(folds[index], *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/crypto/v2.0/train.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("data/crypto/v2.0/historical_meta_models.parquet"))
    parser.add_argument("--folds", type=Path, default=Path("results/volatility_proxies/ewma/folds.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/volatility_replacement"))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS)
    parser.add_argument("--mlp-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mlp_epochs < 1:
        parser.error("--mlp-epochs must be positive")
    if args.mlp_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        args.workers = 1
    path = args.output / "v2.0_volatility_replacement.parquet"
    panel, original, covered, replacement = model_inputs(args.panel, args.meta, path)
    check_scoring()
    folds = list(pd.read_csv(args.folds, parse_dates=list(range(1, 7))).itertuples(index=False))
    dates = pd.DatetimeIndex(panel.date.unique())
    for fold in folds:
        starts = dates.get_indexer([fold.train_start, fold.inner_train_end, fold.validation_start,
                                    fold.train_end, fold.test_start, fold.test_end])
        if tuple(np.diff(starts)) != (67, 23, 39, 23, 129):
            raise ValueError(f"Unexpected purges or split sizes in fold {fold.fold}")
    worker_args = (panel, original, covered, replacement, args.output, args.mlp_epochs, args.mlp_device, folds)
    if args.workers == 1:
        records = [write_fold(fold, *worker_args[:-1]) for fold in folds]
    else:
        with ProcessPoolExecutor(min(args.workers, len(folds)), mp_context=get_context("fork"),
                                 initializer=init_workers, initargs=worker_args) as pool:
            records = list(pool.map(run_worker, range(len(folds))))
    scores = pd.DataFrame([row for part, _ in records for row in part],
                          columns=["fold", "date", "model", "arm", "sample", "corr_all", "corr", "mmc"])
    scores["payout"] = .1 * scores["corr"] + scores.mmc
    scores.to_parquet(args.output / "scores.parquet", index=False)
    pd.DataFrame([row for _, part in records for row in part],
                 columns=["fold", "model", "arm", "parameters", "inner_corr"]).to_csv(
                     args.output / "picks.csv", index=False)
    execution = dict(development_start=str(folds[0].test_start.date()),
                     development_end=str(folds[-1].test_end.date()),
                     inputs={str(p): digest(p) for p in (args.panel, args.meta, args.folds, path)},
                     ridge_alphas=list(ALPHAS), tree_depths=TREE_DEPTHS, tree_iterations=TREE_ITERATIONS,
                     mlp_seeds=SEEDS, mlp_grid=MLP_GRID, mlp_epochs=args.mlp_epochs, mlp_device=args.mlp_device)
    (args.output / "run_manifest.json").write_text(json.dumps(execution, indent=2) + "\n")


if __name__ == "__main__":
    main()
