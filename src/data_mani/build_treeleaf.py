"""
Causal leaf-value features from two of the book's tree models (study E130).

    python3 -m src.data_mani.build_treeleaf --source cat     # fit one source, cache its leaf panel
    python3 -m src.data_mani.build_treeleaf --source rf
    python3 -m src.data_mani.build_treeleaf --merge          # both caches -> the two -treeleaf variants
    python3 -m src.data_mani.build_treeleaf --source cat --smoke

The two sources
---------------
  cat  catboost_ktz on crypto/v2.0-fenccausal-fmx with only E118's x29 block (the
       LightGBM stage profile) kept: E118's arm e118_catboost_x29, 46 features,
       50-draw purged search.
  rf   rf_ktz on crypto/v2.0-tenc-onehot-spread-interact (allfeat, 788 features):
       MA_ALLFEAT's run e52_allfeat, 40-draw purged search.

Both are fit 20, rolling 130 / refit 130 / embargo 20, seed 42, searched from
scratch at every refit -- no parameter is carried over from the stored runs.

What a "leaf" feature is
------------------------
The same design as E129 (build_rfleaf.py). Leaf IDS are useless downstream: each
source is refit every 130 periods, so leaf 7 of tree 12 means something different in
every refit. A leaf's VALUE does carry over, so for every row we take its value in
every tree and summarise the distribution: mean, std, and the 10/25/50/75/90th
percentiles.
  - rf: a tree's leaf value is that tree's prediction (the mean target of the
    training rows in the leaf). The mean over 100 trees IS the forest's prediction.
  - cat: a tree's leaf value is that tree's additive contribution, with the learning
    rate already applied. scale * sum + bias is CatBoost's prediction, so the mean
    is the prediction up to a positive affine map -- rank-identical within a date.
    The spread and quantiles describe how the boosting path built it.
No meta-residualised column: E129 carried one because its source was a resid
model, and neither source here is. Every column is rank-binned within the PANEL's
date onto Numerai's {0,.25,.5,.75,1} after the merge, with AVERAGE ranks so a tie
group shares one bin (the RF quantiles tie on up to 74% of rows): feature_lfcat_*
and feature_lfrf_*, 7 each.

Missing leaves are written as the neutral bin 0.5 and PAIRED with one 0/1 column
per source, feature_lfcat_missing / feature_lfrf_missing, so 0.5 from "no leaf"
stays distinguishable from 0.5 from "middle of the field". A row has no leaf when
it predates the source's first walk-forward block, or sits in the 20 embargo dates
make_validation_split drops before the source's validation start.

Why it is causal
----------------
Each source is re-run through the pipeline's own forecast() with the validation
start moved back BLOCKS_BACK = 2 blocks of 130 periods. A row at date d therefore
gets the leaves of a model trained only on rows dated <= (start of d's block) - 20,
whose 20-day targets had resolved before d, and no row ever gets leaves from a
model that trained on it. Two blocks is the minimum that covers a downstream
130-period training window plus its 20-period embargo (150 < 260). It is also as far
back as the cat source can go: E118's x29 block starts at foundation_repr.START
(2023-05-15). Even so, the cat source's FIRST training window is ~75% before START,
so x29 is the neutral 0.5 there. That block's leaves feed only the first 20 periods
of the downstream first training window. Dates before the first leaf block get the
neutral 0.5 and missing = 1 (see below); no downstream window reads them.

Verified at fit time:
  - asserted, per block: the per-tree values reproduce the source's own predict();
  - reported: on the stored run's blocks 2+ (wholly after the embargo gap that
    make_validation_split opens before the validation start, as E129 found), the
    re-run's prediction against the stored forecast. Max |diff| and the within-date
    rank correlation go into the cache's json.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.data_mani import binning  # noqa: E402
from src.data_mani.utils import local_path  # noqa: E402

SOURCES = {
    "cat": dict(model="catboost_ktz", data="crypto/v2.0-fenccausal-fmx",
                exclude=r"^feature_x(?!29_)\d\d_", n_iter=50, n_splits=5,
                stored=("catboost_ktz", "e118_catboost_x29")),
    "rf": dict(model="rf_ktz", data="crypto/v2.0-tenc-onehot-spread-interact",
               exclude=None, n_iter=40, n_splits=5,
               stored=("rf_ktz", "e52_allfeat")),
}
SUMMARIES = ("mean", "std", "q10", "q25", "q50", "q75", "q90")
BASES = ("v2.0-fenccausal", "v2.0-tenc-onehot-rollsel-spread-interact-fmx")
SUFFIX = "-treeleaf"
CACHE_DIR = os.path.join("data", "crypto", "treeleaf_cache")
BLOCK, BLOCKS_BACK = 130, 2
TIES = "average"          # tie groups share a bin; see binning.rank_to_bins
MAX_MASS_DRIFT = 0.02     # average ties move a little mass across boundaries


def prefix(source):
    return "feature_lf%s_" % source


def _fitted(estimator):
    """The fitted tree model inside whatever forecast() hands predict_signal."""
    if hasattr(estimator, "best_estimator_"):
        estimator = estimator.best_estimator_
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    return estimator


def tree_values(estimator, X, prediction):
    """Per-row, per-tree leaf values; asserts they rebuild the model's prediction."""
    model = _fitted(estimator)
    if hasattr(model, "estimators_"):                    # sklearn forest
        Xa = np.asarray(X, dtype=np.float32)
        values = np.column_stack([t.predict(Xa) for t in model.estimators_]).astype(np.float32)
        rebuilt = values.mean(axis=1, dtype=np.float64)
    else:                                                # CatBoost
        # CatBoost returns uint32 for both; int64 before the arithmetic, or the
        # concatenate with a Python 0 promotes the offsets to float.
        index = np.asarray(model.calc_leaf_indexes(X), dtype=np.int64)
        leaves = np.asarray(model.get_leaf_values(), dtype=np.float64)
        counts = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        values = leaves[offsets[None, :] + index].astype(np.float32)
        scale, bias = model.get_scale_and_bias()
        bias = float(np.ravel(bias)[0])
        rebuilt = scale * leaves[offsets[None, :] + index].sum(axis=1) + bias
    gap = float(np.max(np.abs(rebuilt - np.asarray(prediction, dtype=np.float64))))
    assert gap < 1e-5, "per-tree leaf values do not rebuild predict(): max |diff| %.2e" % gap
    return values, gap


def summarise(values):
    out = {"mean": values.mean(axis=1, dtype=np.float64), "std": values.std(axis=1, dtype=np.float64)}
    for q, col in zip((10, 25, 50, 75, 90), np.percentile(values, (10, 25, 50, 75, 90), axis=1)):
        out["q%d" % q] = col
    return out


def fit_leaves(source, n_jobs, search_jobs, verbose=True, smoke=False):
    """Re-run one source over the extended window; return its per-row leaf summaries."""
    from src.prediction import functions

    spec = SOURCES[source]
    captured, gaps = [], []
    original = functions.predict_signal

    def capture(estimator, X, task):
        out = original(estimator, X, task)
        caller = sys._getframe(1)
        if caller.f_code.co_name == "walk_forward_fit_and_predict":
            # Key the leaves by the walk-forward's own rows, not by call order.
            loc = caller.f_locals
            keys = loc["df"].loc[loc["block_mask"], [loc["id_col"], loc["period_col"]]]
            assert len(keys) == len(X)
            values, gap = tree_values(estimator, X, out)
            gaps.append(gap)
            block = keys.reset_index(drop=True).rename(
                columns={loc["id_col"]: "symbol", loc["period_col"]: "date"})
            block["prediction"], block["refit"] = np.asarray(out, dtype="float64"), len(captured)
            block["n_trees"] = values.shape[1]
            for name, col in summarise(values).items():
                block[prefix(source) + name] = col
            captured.append(block)
            if verbose:
                print("  leaves: block %d, %d rows x %d trees, rebuild gap %.1e"
                      % (len(captured) - 1, len(block), values.shape[1], gap), flush=True)
            del values
        return out

    functions.predict_signal = capture
    try:
        dates = pd.read_parquet(local_path("crypto", spec["data"].split("/")[1], "train.parquet"),
                                columns=["date"])["date"]
        n_periods = pd.to_datetime(dates).nunique()
        n_val = BLOCK if smoke else int(round(n_periods * 0.3)) + BLOCKS_BACK * BLOCK
        val_frac = n_val / n_periods
        assert int(round(n_periods * val_frac)) == n_val
        options = {"iterations": 200} if (smoke and source == "cat") else None
        predictions, _, _ = functions.forecast(
            tournament="crypto", model_name=spec["model"], task="regression", metric="corr",
            data_version=spec["data"], feature_exclude=spec["exclude"], era_step=1,
            val_frac=val_frac, refit_every=BLOCK, embargo=20, scheme="rolling",
            train_window=BLOCK, n_iter=2 if smoke else spec["n_iter"],
            n_splits=2 if smoke else spec["n_splits"], n_jobs=n_jobs, seed=42,
            verbose=verbose, search_jobs=search_jobs, min_train=0,
            target="target_binned_return_20", inner_cv="purged", model_options=options)
    finally:
        functions.predict_signal = original

    frame = pd.concat(captured, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    assert len(frame) == len(predictions) and not frame.duplicated(["symbol", "date"]).any()
    return frame, dict(n_periods=int(n_periods), val_frac=val_frac, max_rebuild_gap=max(gaps))


def check_stored(source, frame):
    """Prediction vs the stored run's forecast, per stored block; blocks 2+ must match."""
    model, run = SOURCES[source]["stored"]
    path = os.path.join("results", "forecast", "crypto", model, run + ".csv")
    if not os.path.exists(path):
        print("stored forecast %s/%s reclaimed; no reproduction check" % (model, run))
        return {}
    stored = pd.read_csv(path, usecols=["symbol", "date", "prediction", "refit"], parse_dates=["date"])
    joined = stored.merge(frame[["symbol", "date", "prediction"]], on=["symbol", "date"],
                          suffixes=("_stored", "_mine"))
    report = {}
    for refit, block in joined.groupby("refit"):
        rho = float(block.groupby("date").apply(
            lambda g: g.prediction_stored.rank().corr(g.prediction_mine.rank())).mean())
        report[int(refit)] = {"rows": int(len(block)), "rank_corr": rho,
                              "max_abs_diff": float(np.abs(block.prediction_stored - block.prediction_mine).max())}
        print("%s block %d: %d rows, max |diff| %.2e, mean within-date rank corr %.6f"
              % (run, refit, len(block), report[int(refit)]["max_abs_diff"], rho))
    later = [r for k, r in report.items() if k >= 2]
    if not later or min(r["rank_corr"] for r in later) < 0.999:
        print("WARNING: the re-run does not reproduce %s on its blocks 2+ -- read the numbers above" % run)
    return report


def merge(leaf, base, write=True):
    """Bin the leaf columns over the base PANEL's own cross-section and write base-treeleaf."""
    cols = [c for c in leaf.columns if c.startswith("feature_lf")]
    src = pd.read_parquet(local_path("crypto", base, "train.parquet"))
    idx_name = src.index.name or "symbol"
    panel = src.reset_index() if "symbol" not in src.columns else src.reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    clash = [c for c in cols if c in panel.columns]
    assert not clash, clash
    merged = panel.merge(leaf, on=["symbol", "date"], how="left")
    assert len(merged) == len(panel)
    # Average ranks: the RF quantile columns tie on up to 74% of rows, and a tie
    # group must share a bin rather than be split across two by row order.
    merged = binning.bin_cross_sectional(merged, cols, "date", ties=TIES)
    for c in cols:
        mass = binning.bin_mass(merged[c])
        drift = float(np.max(np.abs(np.asarray(mass) - np.array([.05, .20, .50, .20, .05]))))
        assert drift < MAX_MASS_DRIFT, (base, c, mass)
    print("  bin mass: max drift from 5/20/50/20/5 over the %d leaf columns %.4f"
          % (len(cols), max(float(np.max(np.abs(np.asarray(binning.bin_mass(merged[c]))
                                                  - np.array([.05, .20, .50, .20, .05])))) for c in cols)))
    # Missing leaves -> the neutral bin 0.5, PAIRED with a 0/1 indicator per source
    # (feature_lf<src>_missing), so a model can tell "no leaf for this row" from
    # "a leaf in the middle bin". A source's 7 columns are missing together.
    for source in SOURCES:
        block = [c for c in cols if c.startswith(prefix(source))]
        if not block:
            continue
        gone = merged[block].isna()
        assert gone.all(axis=1).equals(gone.any(axis=1)), "partial leaf rows for %s" % source
        merged[prefix(source) + "missing"] = gone.any(axis=1).astype("float32")
        merged[block] = merged[block].fillna(0.5)
    name = base + SUFFIX
    flags = [c for c in merged.columns if c.startswith("feature_lf") and c.endswith("_missing")]
    print("crypto/%s: %d rows, %d columns (+%d leaf, +%d indicators), missing share %s"
          % (name, len(merged), merged.shape[1], len(cols), len(flags),
             {f: round(float(merged[f].mean()), 4) for f in flags}))
    if not write:
        return name
    path = local_path("crypto", name, "train.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged.set_index(idx_name).to_parquet(path)
    meta_src = local_path("crypto", base, "historical_meta_models.parquet")
    if not os.path.exists(meta_src):
        meta_src = local_path("crypto", "v2.0", "historical_meta_models.parquet")
    shutil.copy(meta_src, os.path.join(os.path.dirname(path), "historical_meta_models.parquet"))
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--merge", action="store_true", help="write the -treeleaf variants from both caches")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--search-jobs", type=int, default=4)
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--smoke", action="store_true",
                    help="one block, 2-draw search (200 CatBoost trees); merges in memory, writes nothing")
    args = ap.parse_args()
    if not (args.source or args.merge):
        ap.error("give --source and/or --merge")

    if args.source:
        frame, info = fit_leaves(args.source, args.n_jobs, args.search_jobs, smoke=args.smoke)
        cols = [prefix(args.source) + s for s in SUMMARIES]
        print("%s: %d leaf rows over %d dates, %d blocks, trees per block %s, first leaf date %s"
              % (args.source, len(frame), frame.date.nunique(), frame.refit.nunique(),
                 sorted(frame.n_trees.unique()), frame.date.min().date()))
        print(frame[cols].describe().T.round(5).to_string())
        if args.smoke:
            for base in [b for b in args.bases.split(",") if b]:
                merge(frame[["symbol", "date"] + cols], base, write=False)
            return
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache = os.path.join(CACHE_DIR, args.source + ".parquet")
        frame.to_parquet(cache, index=False)      # before the check: a failed check must not cost the fit
        info.update(first_leaf_date=str(frame.date.min().date()), blocks=int(frame.refit.nunique()),
                    stored_check=check_stored(args.source, frame))
        json.dump(info, open(cache.replace(".parquet", ".json"), "w"), indent=1)

    if args.merge:
        leaf = None
        for source in sorted(SOURCES):
            part = pd.read_parquet(os.path.join(CACHE_DIR, source + ".parquet"))
            part = part[["symbol", "date"] + [prefix(source) + s for s in SUMMARIES]]
            leaf = part if leaf is None else leaf.merge(part, on=["symbol", "date"], how="outer")
        for base in [b for b in args.bases.split(",") if b]:
            merge(leaf, base)


if __name__ == "__main__":
    main()
