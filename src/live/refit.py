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

    # Structural embargo check -- see the module docstring. Counted in PERIODS,
    # because `embargo` is a period count: 20 periods span 28 calendar days on
    # this weekday grid, so comparing it against a day count admitted any gap of
    # 20 days = 14 periods and left the guard ~30% under-strength. It has never
    # actually let anything through -- real gaps are far larger -- but the one
    # assertion the design calls structural was sized for the wrong unit.
    gap = periods_between(kept[-1], live_date)
    if gap < embargo:
        raise ValueError(
            "only %d periods between the last training period (%s) and the live "
            "date (%s); the %d-period target horizon requires at least %d. "
            "Training rows would overlap the prediction window."
            % (gap, pd.Timestamp(kept[-1]).date(), live_date.date(), embargo, embargo))

    X = block[columns]
    y = block[target_col]
    meta = block[[PERIOD]].copy()
    return X, y, meta, kept, gap


def periods_between(start, end):
    """
    How many trading periods separate two dates.

    The crypto grid is weekday-only -- the panel holds 1719 dates across a span
    of 1733 business days, 99.2% of them -- so a calendar-day count is not a
    period count and the two diverge by ~40%: 130 periods is 182 calendar days,
    and 130 calendar days is 93 periods.

    Counted on the business-day CALENDAR rather than on the panel's own dates,
    because the panel is not contiguous where it matters most. train.parquet
    ends ~30 days before the live date and live_history fills that stretch one
    date per round, so today the panel holds 8 rows across a 22-period gap.
    Counting rows would have called that gap 8 and made the embargo guard reject
    every refit. What the embargo means is trading days elapsed, which is what
    this measures.
    """
    return len(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))) - 1


def feature_signature(columns):
    """
    Cheap fingerprint of the feature set.

    A cached model is only reusable if its columns still mean what they meant at
    fit time. The one-hot vocabulary changes as tokens list and delist, so this
    catches "the cache is stale for a reason other than the calendar".
    """
    import hashlib
    return hashlib.sha1("|".join(columns).encode()).hexdigest()[:12]


def fitted_columns(estimator):
    """The column names an estimator was fitted on, in order, or None if it does not say."""
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        names = getattr(estimator, "feature_names_", None)    # CatBoost
    if names is None:
        return None
    return [str(n) for n in names]


def usable_columns(estimator, columns):
    """
    The columns a CACHED estimator can actually be predicted on.

    A cache whose feature signature no longer matches -- almost always because
    the one-hot vocabulary grew when a token crossed ONEHOT_MIN_DATES -- used
    to be handed the NEW, wider column list, so a skipped refit's "stale cache"
    rung could never run: LightGBM refused 777 columns for a 776-column fit
    (SACI_06, 2026-10-21) and the slot fell to the sibling. The vocabulary only
    grows, so every column the cache was fitted on is still in the frame; the
    stale model predicts on exactly those, which is what it would have done the
    round before -- the new token is simply cold to it.

    :return: (columns, note). columns is the fitted list when it differs, the
             given list when it matches or the estimator does not name its
             inputs, and None when a fitted column is missing from the build.
    """
    fitted = fitted_columns(estimator)
    columns = list(columns)
    if fitted is None or fitted == columns:
        return columns, None
    present = set(columns)
    if not any(c in present for c in fitted):
        # Positional names ("0", "1", ...) say nothing about identity.
        return columns, None
    missing = [c for c in fitted if c not in present]
    if missing:
        return None, ("cached fit needs %d column(s) the current build lacks (%s)"
                      % (len(missing), ", ".join(missing[:3])))
    return fitted, ("cached fit predicts on its own %d fitted columns (the build "
                    "now has %d)" % (len(fitted), len(columns)))


def load_slot_blend(name, panel, seen=None):
    """
    Assemble a slot that rank-averages OTHER slots.

    Each member slot is loaded on its own feature set, then the whole thing is
    predicted from one frame: this slot's feature_set is declared as a superset
    of every member's columns, so a single build serves them all.

    RankBlend nests naturally -- a member that is itself a blend receives the
    frame and does its own per-member selection inside.

    A member that is itself a SLOT blend has to be assembled here rather than
    loaded, because slot blends fit nothing and so persist no joblib. Sending one
    to load_cached returns (None, None) and the whole blend reports it as having
    "no cached fit" -- which is precisely how CURUPIRA_SACI_03 came to upload
    Numerai's example predictions every round from 1338 onward, its member
    CURUPIRA_08 being the only nested slot blend in ORDER. load_cached's own
    docstring asserts slot blends "never reach here"; that invariant was true
    only until a blend was made a member of another blend.
    """
    from src.live.blend import JaciForecast, MetaResidual, RankBlend, RoutedBlend
    from src.live.features import build_features

    spec = registry.get(name)
    # A blend cannot be its own member, directly or through a chain. Nothing in
    # ORDER does that today, but the recursion below would not survive it.
    seen = set(seen or ())
    if name in seen:
        return None, {"missing": ["%s (circular membership)" % name]}
    seen = seen | {name}

    members, missing, partial = [], [], []
    for entry in spec["slot_members"]:
        member_spec = registry.get(entry["name"])
        if member_spec.get("wrapper") == "precomputed_jaci":
            # E216: produced by the src.live.jaci pre-stage, not fitted here. It
            # covers only part of the universe, so the blend is routed.
            estimator = JaciForecast(entry["name"])
            members.append((entry["name"], estimator, estimator.columns, entry["weight"]))
            partial.append(entry["name"])
            continue
        try:
            _, member_columns, _ = build_features(
                panel.copy(), member_spec["feature_set"], verbose=False)
        except Exception as error:
            missing.append("%s (feature build: %s)" % (entry["name"], error))
            continue

        if registry.is_slot_blend(member_spec):
            estimator, member_meta = load_slot_blend(entry["name"], panel, seen)
            if estimator is None:
                missing.append("%s (%s)" % (entry["name"], "; ".join(
                    (member_meta or {}).get("missing") or ["unavailable"])))
                continue
            # Its own union, so the outer blend hands it every column it needs.
            member_columns = estimator.columns
        else:
            estimator, member_meta = load_cached(entry["name"], member_columns)
            if estimator is None:
                missing.append("%s (no cached fit)" % entry["name"])
                continue
            # A member whose own refit was skipped this round still carries its
            # last fit; predict it on the columns it was fitted on.
            usable, why = usable_columns(estimator, member_columns)
            if usable is None:
                missing.append("%s (%s)" % (entry["name"], why))
                continue
            member_columns = usable

        if entry.get("resid"):
            # E107: this leg enters the blend residualised on the lagged meta
            # model, not as it was fitted. The wrapper adds META_COL to the
            # columns it asks for, so the frame must carry it -- prepare_slot
            # sees to that.
            estimator = MetaResidual(entry["name"], estimator, member_columns,
                                     strength=entry["resid"])
            member_columns = estimator.columns

        members.append((entry["name"], estimator, member_columns, entry["weight"]))

    if missing:
        print("   %s cannot be assembled: %s" % (name, "; ".join(missing)))
        # Hand the real list back. The caller used to report EVERY declared
        # member as unavailable, so round 1338 blamed SACI_04 and CURUPIRA_05 --
        # both of which had fitted fine in the same pass -- for a fault that was
        # CURUPIRA_08's alone, and that misdiagnosis reached CLAUDE.md.
        return None, {"missing": missing}

    metadata = {
        "model": name, "wrapper": "slot_blend",
        "feature_set": spec["feature_set"],
        "members": [{"name": n, "weight": w, "n_features": len(c),
                     "resid": e.strength if isinstance(e, MetaResidual) else None}
                    for n, e, c, w in members],
        "fitted_at": None,   # it fits nothing; freshness is its members'
    }
    if partial:
        metadata["routed"] = partial
        return RoutedBlend(members, partial), metadata
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

    TWO THINGS THAT WERE WRONG HERE, AND THEY COMPOUNDED.

    The UNIT. registry.FIT declares `"refit_every": 130,  # periods between
    refits` and this compared it against `.days`. The panel is weekday-only, so
    130 calendar days is 93 periods -- a 29% short cadence, landing between the
    r6m and r3m arms of the repo's own cadence sweep, whose stated conclusion is
    that refitting more often is actively worse. `train_window`, from the same
    dict, was consumed correctly as periods eight lines away.

    The ANCHOR. It measured from `fitted_through`, the last TRAINING period,
    rather than from when the fit happened. train.parquet lags live, so a model
    refit today reported itself 44 days old the same afternoon: each fit's
    counter started at the embargo instead of zero, making the effective cadence
    `refit_every - gap` and shrinking it as the gap grew. The 60-day slots
    started at 93 and 99, so their service life was set by their target horizon
    rather than by the protocol.

    """
    if metadata is None:
        return True, "no cached fit"
    if metadata.get("feature_signature") != signature:
        return True, "feature set changed (vocabulary or columns)"
    # Anchor at the live date the fit was made FOR, which is a panel date on the
    # same grid as live_date -- unlike fitted_at, which is wall clock.
    anchor = metadata.get("live_date") or metadata.get("fitted_through")
    if anchor:
        elapsed = periods_between(anchor, live_date)
        if elapsed >= refit_every:
            return True, "%d periods since last fit (cadence %d)" % (
                elapsed, refit_every)
        return False, "fit is %d of %d periods old" % (elapsed, refit_every)

    if not metadata.get("fitted_through"):
        # A slot blend fits nothing and records no training edge, so there is no
        # cadence to be due against. Anything else with no edge has never been
        # fitted properly and should be.
        return (False, "assembled from members, never fitted") \
            if metadata.get("wrapper") == "slot_blend" \
            else (True, "cached fit records no training edge")
    fitted_through = pd.Timestamp(metadata["fitted_through"])
    elapsed = (live_date - fitted_through).days
    if elapsed >= refit_every:
        return True, ("%d days since last fit (cadence %d periods, counted in "
                      "DAYS -- no panel grid available)" % (elapsed, refit_every))
    return False, ("fit is %d days old of a %d-period cadence, counted in DAYS "
                   "-- no panel grid available" % (elapsed, refit_every))


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
        print("   training block: %d rows, %d periods %s..%s (%d periods before live)"
              % (len(X_all), len(periods), pd.Timestamp(periods[0]).date(),
                 pd.Timestamp(periods[-1]).date(), gap))

    os.makedirs(registry.MODEL_DIR, exist_ok=True)
    records, union = [], []
    for member in spec["members"]:
        member_columns = filter_features(columns, member["include"], None, verbose=False)
        # The backtest fitted these exact widths; a drift here means the live
        # model is not the model that was measured. Unless the block's width is
        # a property of the live universe rather than of the model -- the one-hot
        # identity columns are one per qualifying symbol, so they legitimately
        # grow, and pinning them meant the blend could never be refit again.
        if member.get("n_features_varies"):
            if not member_columns:
                raise ValueError(
                    "blend member '%s' selected no columns (pattern %r)"
                    % (member["name"], member["include"]))
            if verbose and len(member_columns) != member["n_features"]:
                print("   member '%s': %d columns, %+d against the backtest's %d"
                      % (member["name"], len(member_columns),
                         len(member_columns) - member["n_features"],
                         member["n_features"]))
        elif member["n_features"] and len(member_columns) != member["n_features"]:
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
        # PERIODS, as the protocol's embargo is. The old
        # "embargo_days_actual" held a calendar-day count beside a
        # period-denominated protocol.embargo, which made a 32/20
        # divergence read as 44/20.
        "embargo_periods_actual": int(gap),
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
            estimator = joblib.load(path)
            # Predict each member on the columns IT was fitted on. When the
            # vocabulary grew under the cache those are all still selected, so
            # the cache stays usable -- refit_due still sees the signature change
            # and refits when the budget allows. A width check alone used to be
            # the only guard, which also let an equal-width but different set
            # through silently.
            usable, _ = usable_columns(estimator, member_columns)
            if usable is None or len(usable) != member["n_features"]:
                raise ValueError(
                    "member '%s' now selects %d columns, was fitted on %d"
                    % (member["name"], len(member_columns), member["n_features"]))
            member_columns = usable
            members.append((member["name"], estimator,
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
              "(%d periods before live)" % (len(X), len(columns), len(periods),
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
        # PERIODS, as the protocol's embargo is. The old
        # "embargo_days_actual" held a calendar-day count beside a
        # period-denominated protocol.embargo, which made a 32/20
        # divergence read as 44/20.
        "embargo_periods_actual": int(gap),
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
    # Accepted because the README and this module's docstring both show
    # `--all --force`, and because saying it is clearer than omitting --model.
    # It has always been the default: names falls back to the whole of ORDER.
    parser.add_argument("--all", action="store_true",
                        help="refit every slot in ORDER (the default)")
    parser.add_argument("--force", action="store_true",
                        help="refit even if the cadence does not require it")
    args = parser.parse_args()

    names = [args.model] if args.model else registry.ORDER
    panel, live_date, gap_days = load_panel()

    failed = []
    for name in names:
        spec = registry.get(name)
        print("\n=== %s (%s, %s) ===" % (name, spec["wrapper"], spec["feature_set"]))

        # A slot blend rank-averages OTHER slots and fits nothing of its own, so
        # it has no wrapper to build. Sending it down the fitting path called
        # build_wrapper(None, ...) and raised "unknown model 'None'" -- and with
        # no try around this loop that killed the whole run on CURUPIRA_06,
        # leaving CURUPIRA_07, _08 and all four CURUPIRA_SACI_* untouched and the
        # cache half refreshed. That made `refit` useless as the operator remedy
        # for exactly the blend faults it was needed for.
        if registry.is_slot_blend(spec):
            print("   skip: assembled from %s at predict time, fits nothing"
                  % "+".join(m["name"] for m in spec["slot_members"]))
            continue

        try:
            built, columns, diagnostics = build_features(panel.copy(),
                                                        spec["feature_set"])
            _, metadata = load_cached(name, columns)
            due, reason = refit_due(metadata, live_date, spec["fit"]["refit_every"],
                                    feature_signature(columns))
            print("   %s: %s" % ("REFIT" if (due or args.force) else "skip", reason))
            if due or args.force:
                if registry.is_blend(spec):
                    refit_blend(name, built, columns, live_date)
                else:
                    refit(name, built, columns, live_date)
        except Exception as error:
            # One bad slot must not cost the other seventeen their refit.
            failed.append(name)
            print("   FAILED: %s: %s" % (type(error).__name__, error))

    if failed:
        print("\n%d slot(s) failed to refit: %s" % (len(failed), ", ".join(failed)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
