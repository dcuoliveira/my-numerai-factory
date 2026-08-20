"""
Refit a live model with fidelity to the pseudo-out-of-sample protocol.

    python3 -m src.live.refit --model CURUPIRA_01
    python3 -m src.live.refit --all --force

A live refit has to reproduce what the backtest measured, or the backtest is not
evidence about the live model. Concretely, every refit here uses:

    rolling window   the last 130 periods of TARGET-RESOLVED data
    inner CV         purged_period_splits, 5 folds, 20-period embargo, rolling
    search           RandomizedSearchCV, 50 draws, objective = per-period CORR
    seed             42

which is the `config` block of the winning run for each slot. The one thing that
is NOT reproduced is the walk-forward loop itself -- live there is only ever one
block, the current one. That is the point of the exercise: the backtest ran this
same fit 4 times over history to estimate what this single fit will do next.

------------------------------------------------------------------------------
Refit cadence and why models are cached
------------------------------------------------------------------------------

The study swept refit frequency at a fixed 6-month window and found a clear
inverted U: monthly 0.127, quarterly 0.145, SIX-MONTHLY 0.153, nine-monthly
0.150, yearly 0.144. Refitting more often is actively worse -- the model spends
its time chasing a regime that has not changed, on a 26-effective-observation
sample.

So models are fit on a 130-period cadence and cached in between. A round that
does not need a refit costs a parquet read and a matrix multiply, which is what
makes the tight submission deadline comfortable. `fit_seconds` is recorded in
every model's metadata so the orchestrator can decide, before starting, whether
a refit fits in the time remaining.

------------------------------------------------------------------------------
The embargo, live
------------------------------------------------------------------------------

In the backtest the embargo was enforced explicitly: training ended 20 periods
before the block being predicted. Live it is enforced by the data itself --
train.parquet only ships rows whose 20-day target has resolved, so it lags the
live date by 20+ days. The assertion below makes that structural rather than
assumed, because if Numerai ever changes that policy the model would silently
start training on rows that overlap the prediction date.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from src.data_mani.utils import REFERENCE_COL
from src.live import registry
from src.live.features import (PERIOD, SYMBOL, TARGET, build_features, load_panel)
from src.prediction.functions import hyper_params_search
from src.prediction.models import build_wrapper

# Ridge fits are single-threaded, so the parallelism belongs to the search.
# Crypto's matrix is small enough (~100MB) that forking per candidate is cheap;
# the module note in hyper_params_search explains why both must never be > 1.
SEARCH_JOBS = 8
ESTIMATOR_JOBS = 1


def training_block(panel, columns, train_window, embargo, live_date,
                   target_col=TARGET):
    """
    The last `train_window` target-resolved periods, as (X, y, meta).

    `target_col` is a parameter because crypto ships two labels and a slot may
    be fitted on either. It has to move together with `embargo`: the 60-day
    label resolves 60 periods out and is NaN over the panel's last 40 dates, so
    fitting it under the 20-day embargo would both leak and hand NaN targets to
    the estimator.

    X is deliberately left as a DataFrame: the CV scorer recovers each row's
    period through meta.loc[X.index], which only works because sklearn's iloc
    slicing preserves the index. Passing .values here would break the scorer in
    a way that still runs and silently scores garbage.
    """
    if target_col not in panel.columns:
        raise ValueError("target %r is not in the panel; available: %s"
                         % (target_col,
                            sorted(c for c in panel.columns if c.startswith("target"))))
    resolved = panel[~panel["is_live"] & panel[target_col].notna()].copy()
    if resolved.empty:
        raise ValueError("no target-resolved rows in the panel")

    periods = np.sort(resolved[PERIOD].unique())
    kept = periods[-train_window:]
    block = resolved[resolved[PERIOD].isin(set(kept))]

    # Structural embargo check -- see the module docstring.
    gap = (live_date - pd.Timestamp(kept[-1])).days
    if gap < embargo:
        raise ValueError(
            "only %d days between the last training period (%s) and the live "
            "date (%s); the %d-day target horizon requires at least %d. "
            "Training rows would overlap the prediction window."
            % (gap, pd.Timestamp(kept[-1]).date(), live_date.date(), embargo, embargo))

    X = block[columns]
    y = block[target_col]
    meta = block[[PERIOD]].copy()
    return X, y, meta, kept, gap


def feature_signature(columns):
    """
    Cheap fingerprint of the feature set.

    A cached model is only reusable if its columns still mean what they meant at
    fit time. The one-hot vocabulary changes as tokens list and delist, so this
    catches "the cache is stale for a reason other than the calendar".
    """
    import hashlib
    return hashlib.sha1("|".join(columns).encode()).hexdigest()[:12]


def load_slot_blend(name, panel):
    """
    Assemble a slot that rank-averages OTHER slots.

    Each member slot is loaded on its own feature set, then the whole thing is
    predicted from one frame: this slot's feature_set is declared as a superset
    of every member's columns, so a single build serves them all.

    RankBlend nests naturally -- a member that is itself a blend receives the
    frame and does its own per-member selection inside.
    """
    from src.live.blend import RankBlend
    from src.live.features import build_features

    spec = registry.get(name)
    members, missing = [], []
    for entry in spec["slot_members"]:
        member_spec = registry.get(entry["name"])
        try:
            _, member_columns, _ = build_features(
                panel.copy(), member_spec["feature_set"], verbose=False)
        except Exception as error:
            missing.append("%s (feature build: %s)" % (entry["name"], error))
            continue
        estimator, member_meta = load_cached(entry["name"], member_columns)
        if estimator is None:
            missing.append("%s (no cached fit)" % entry["name"])
            continue
        members.append((entry["name"], estimator, member_columns, entry["weight"]))

    if missing:
        print("   %s cannot be assembled: %s" % (name, "; ".join(missing)))
        return None, None

    metadata = {
        "model": name, "wrapper": "slot_blend",
        "feature_set": spec["feature_set"],
        "members": [{"name": n, "weight": w, "n_features": len(c)}
                    for n, _, c, w in members],
        "fitted_at": None,   # it fits nothing; freshness is its members'
    }
    return RankBlend(members), metadata


def load_cached(name, columns=None):
    """
    Return (estimator, metadata) for a slot, or (None, None) if not cached.

    Blend slots dispatch to load_blend, which needs the current column list to
    re-derive each member's subset. Slot blends are assembled by the caller
    (they need the panel, not just a column list) and never reach here.
    """
    if registry.is_blend(registry.get(name)):
        if columns is None:
            return None, None
        return load_blend(name, columns)

    model_file, meta_file = registry.model_path(name), registry.meta_path(name)
    if not (os.path.exists(model_file) and os.path.exists(meta_file)):
        return None, None
    try:
        with open(meta_file) as handle:
            metadata = json.load(handle)
        return joblib.load(model_file), metadata
    except Exception as error:                      # corrupt cache is not fatal
        print("   cache for %s is unreadable (%s) -- treating as absent" % (name, error))
        return None, None


def refit_due(metadata, live_date, refit_every, signature):
    """
    (due, reason). A model refits when it has never been fit, when the feature
    set changed under it, or when `refit_every` periods have passed.
    """
    if metadata is None:
        return True, "no cached fit"
    if metadata.get("feature_signature") != signature:
        return True, "feature set changed (vocabulary or columns)"
    fitted_through = pd.Timestamp(metadata["fitted_through"])
    elapsed = (live_date - fitted_through).days
    if elapsed >= refit_every:
        return True, "%d days since last fit (cadence %d)" % (elapsed, refit_every)
    return False, "fit is %d of %d days old" % (elapsed, refit_every)


def fit_one(wrapper_name, X, y, meta, fit, verbose=False):
    """Run the protocol search on one design matrix and return the search."""
    wrapper = build_wrapper(wrapper_name, fit["task"], seed=fit["seed"],
                            n_jobs=ESTIMATOR_JOBS)
    return hyper_params_search(
        X=X, y=y, wrapper=wrapper, meta=meta,
        metric=fit["metric"], period_col=PERIOD, reference_col=None,
        task=fit["task"], n_iter=fit["n_iter"], n_splits=fit["n_splits"],
        # A slot may pin its estimator instead of searching. Without this the
        # live refit would run the 50-draw search and fit a DIFFERENT model than
        # the arm that justified the slot -- the exact drift registry.py's
        # "fidelity to the backtest" note exists to prevent.
        fixed_params=fit.get("fixed_params"),
        embargo=fit["embargo"], scheme=fit["cv_scheme"],
        train_window=fit["train_window"], n_jobs=ESTIMATOR_JOBS,
        seed=fit["seed"], verbose=verbose, search_jobs=SEARCH_JOBS)


def refit_blend(name, panel, columns, live_date, verbose=True):
    """
    Fit every member of a blend slot and cache them individually.

    Each member gets the full protocol on its own subset of columns -- its own
    alpha path, its own search. That independence is the point: one shrinkage
    parameter cannot serve a 740-column identity block and a 4-column price
    block, and fitting them separately is what the backtested blend did.
    """
    from src.prediction.functions import filter_features

    spec = registry.get(name)
    fit = spec["fit"]
    started = time.time()

    X_all, y, meta, periods, gap = training_block(
        panel, columns, fit["train_window"], fit["embargo"], live_date,
        target_col=fit["target"])
    if verbose:
        print("   training block: %d rows, %d periods %s..%s (%d days before live)"
              % (len(X_all), len(periods), pd.Timestamp(periods[0]).date(),
                 pd.Timestamp(periods[-1]).date(), gap))

    os.makedirs(registry.MODEL_DIR, exist_ok=True)
    records, union = [], []
    for member in spec["members"]:
        member_columns = filter_features(columns, member["include"], None, verbose=False)
        # The backtest fitted these exact widths; a drift here means the live
        # model is not the model that was measured.
        if member["n_features"] and len(member_columns) != member["n_features"]:
            raise ValueError(
                "blend member '%s' selected %d columns, the backtest used %d "
                "(pattern %r)" % (member["name"], len(member_columns),
                                  member["n_features"], member["include"]))

        search = fit_one(member["wrapper"], X_all[member_columns], y, meta, fit)
        joblib.dump(search.best_estimator_,
                    registry.member_path(name, member["name"]), compress=3)
        union += [c for c in member_columns if c not in union]
        records.append({
            "name": member["name"], "wrapper": member["wrapper"],
            "weight": member["weight"], "n_features": len(member_columns),
            "include": member["include"],
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
            "cv_objective": float(search.best_score_),
        })
        if verbose:
            print("      %-9s %-13s %4d feat  CV corr %.6f  %s"
                  % (member["name"], member["wrapper"], len(member_columns),
                     search.best_score_, records[-1]["best_params"]))

    elapsed = time.time() - started
    metadata = {
        "model": name, "wrapper": "blend", "feature_set": spec["feature_set"],
        "n_features": len(union), "feature_signature": feature_signature(columns),
        "members": records,
        "fitted_at": pd.Timestamp.now("UTC").isoformat(),
        "fitted_through": str(pd.Timestamp(periods[-1]).date()),
        "live_date": str(live_date.date()),
        "train_rows": int(len(X_all)), "train_periods": int(len(periods)),
        "train_start": str(pd.Timestamp(periods[0]).date()),
        "embargo_days_actual": int(gap),
        "fit_seconds": round(elapsed, 1), "protocol": fit,
    }
    with open(registry.meta_path(name), "w") as handle:
        json.dump(metadata, handle, indent=2)
    if verbose:
        print("   blend fit in %.1fs across %d members, %d union columns"
              % (elapsed, len(records), len(union)))
    return metadata


def load_blend(name, columns):
    """
    Rebuild a cached blend, or (None, None) if any member is missing.

    Member column lists are re-derived from their include patterns against the
    CURRENT feature set rather than stored, so a vocabulary change is caught
    here as a width mismatch instead of surfacing as a silent shape error
    inside predict().
    """
    from src.live.blend import RankBlend
    from src.prediction.functions import filter_features

    meta_file = registry.meta_path(name)
    if not os.path.exists(meta_file):
        return None, None
    try:
        with open(meta_file) as handle:
            metadata = json.load(handle)
        members = []
        for member in metadata["members"]:
            path = registry.member_path(name, member["name"])
            if not os.path.exists(path):
                return None, None
            member_columns = filter_features(columns, member["include"], None,
                                             verbose=False)
            if len(member_columns) != member["n_features"]:
                raise ValueError(
                    "member '%s' now selects %d columns, was fitted on %d"
                    % (member["name"], len(member_columns), member["n_features"]))
            members.append((member["name"], joblib.load(path),
                            member_columns, member["weight"]))
        return RankBlend(members), metadata
    except Exception as error:
        print("   blend cache for %s is unreadable (%s) -- treating as absent"
              % (name, error))
        return None, None


def refit(name, panel, columns, live_date, verbose=True):
    """
    Fit one slot on the current rolling window and cache it.

    Returns the metadata dict that was written alongside the estimator.
    """
    spec = registry.get(name)
    fit = spec["fit"]
    started = time.time()

    X, y, meta, periods, gap = training_block(
        panel, columns, fit["train_window"], fit["embargo"], live_date,
        target_col=fit["target"])

    if verbose:
        print("   training block: %d rows x %d features, %d periods %s..%s "
              "(%d days before live)" % (len(X), len(columns), len(periods),
                                         pd.Timestamp(periods[0]).date(),
                                         pd.Timestamp(periods[-1]).date(), gap))

    wrapper = build_wrapper(spec["wrapper"], fit["task"], seed=fit["seed"],
                            n_jobs=ESTIMATOR_JOBS)

    search = hyper_params_search(
        X=X, y=y, wrapper=wrapper, meta=meta,
        metric=fit["metric"], period_col=PERIOD,
        reference_col=None,               # no benchmark series in the live block
        task=fit["task"],
        n_iter=fit["n_iter"], n_splits=fit["n_splits"], embargo=fit["embargo"],
        fixed_params=fit.get("fixed_params"),
        scheme=fit["cv_scheme"], train_window=fit["train_window"],
        n_jobs=ESTIMATOR_JOBS, seed=fit["seed"], verbose=False,
        search_jobs=SEARCH_JOBS,
    )

    elapsed = time.time() - started
    metadata = {
        "model": name,
        "wrapper": spec["wrapper"],
        "feature_set": spec["feature_set"],
        "n_features": len(columns),
        "feature_signature": feature_signature(columns),
        "features": columns if len(columns) <= 64 else columns[:64] + ["..."],
        "fitted_at": pd.Timestamp.now("UTC").isoformat(),
        "fitted_through": str(pd.Timestamp(periods[-1]).date()),
        "live_date": str(live_date.date()),
        "train_rows": int(len(X)),
        "train_periods": int(len(periods)),
        "train_start": str(pd.Timestamp(periods[0]).date()),
        "embargo_days_actual": int(gap),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "cv_objective": float(search.best_score_),
        "fit_seconds": round(elapsed, 1),
        "protocol": fit,
    }

    os.makedirs(registry.MODEL_DIR, exist_ok=True)
    joblib.dump(search.best_estimator_, registry.model_path(name), compress=3)
    with open(registry.meta_path(name), "w") as handle:
        json.dump(metadata, handle, indent=2)

    if verbose:
        print("   fit in %.1fs | CV corr %.6f | params %s" % (
            elapsed, search.best_score_, metadata["best_params"]))
    return metadata


def _jsonable(value):
    """numpy scalars are not JSON-serialisable; everything here is a scalar."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Refit live models on the current rolling window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=None, help="one slot; default is all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="refit even if the cadence does not require it")
    args = parser.parse_args()

    names = [args.model] if args.model else registry.ORDER
    panel, live_date, gap_days = load_panel()

    for name in names:
        spec = registry.get(name)
        print("\n=== %s (%s, %s) ===" % (name, spec["wrapper"], spec["feature_set"]))
        built, columns, diagnostics = build_features(panel.copy(), spec["feature_set"])
        _, metadata = load_cached(name, columns)
        due, reason = refit_due(metadata, live_date, spec["fit"]["refit_every"],
                                feature_signature(columns))
        print("   %s: %s" % ("REFIT" if (due or args.force) else "skip", reason))
        if due or args.force:
            if registry.is_blend(spec):
                refit_blend(name, built, columns, live_date)
            else:
                refit(name, built, columns, live_date)


if __name__ == "__main__":
    main()
