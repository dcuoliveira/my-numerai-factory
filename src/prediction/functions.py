"""
The evaluation pipeline: cross-validation, hyperparameter search, walk-forward.

The shape mirrors the causal-feature pipeline it is adapted from:

    forecast()                     load data, then hand off to the walk-forward
      walk_forward_fit_and_predict()   refit periodically, predict forward
        hyper_params_search()            RandomizedSearchCV on the training block
          purged_period_splits()           era-aware, embargoed inner CV folds
          build_scorer()                   corr / bmc / sharpe / drawdown

Three things differ from the original, and all three are about not lying to
yourself with a leaky score:

1. TimeSeriesSplit is replaced by purged_period_splits(). Splitting on raw rows
   would cut an era in half, and it has no embargo at all -- which matters
   enormously here, because a 20-day target overlaps 75% between consecutive
   Classic eras and 95% between consecutive Crypto dates.

2. The scorer optimises the metric the tournament actually pays on, rather than
   ROC AUC computed on hard labels.

3. There is no StandardScaler. Numerai ships features already rank-binned to a
   common scale -- every Crypto feature has min 0.0, mean 0.500, std 0.224,
   max 1.0 -- so scaling cannot change their relative magnitudes and only adds
   a fit-on-train surface to get wrong.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from src.data_mani.utils import (
    META_PROXY_COL,
    attach_meta_model,
    attach_reference,
    binarize_target,
    build_meta_proxy,
    default_embargo,
    get_spec,
    load_meta_model,
    load_reference,
    load_tournament_data,
)
from src.prediction.metrics import (
    per_period_bmc,
    per_period_corr,
    per_period_table,
    summarize,
    summarize_predictions,
)
from src.prediction.models import build_wrapper
from src.prediction.search_spaces import (
    build_bayes_search,
    describe_budget,
    initial_points,
    split_budget,
)


# The six objectives the search can optimise.
METRICS = ("corr", "bmc", "corr_sharpe", "bmc_sharpe",
           "corr_drawdown", "bmc_drawdown")

# Returned when a fold cannot be scored at all (too few periods, constant
# predictions, no reference coverage). A plain NaN would poison the mean across
# folds and make argmax pick that candidate, so we use a very negative finite
# number instead -- it ranks last without breaking the comparison.
UNSCORABLE = -1e9


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

def purged_period_splits(periods, n_splits=5, embargo=4,
                         scheme="expanding", train_window=None,
                         min_train_size=0):
    """
    Sequential CV folds that split on whole periods and keep an embargo gap.

    Folds are laid out chronologically. Each validation block is preceded by a
    gap of `embargo` periods that belong to neither side, which is what stops a
    training row's forward-looking target window from overlapping the block it
    is being validated against.

    The two schemes differ in exactly one thing -- where training starts:

        expanding:  train_start = 0                              (grows forever)
        rolling:    train_start = train_end - train_window       (fixed length)

    The first fold has no history before it and is skipped.

    :param periods: period label for every row, in row order
    :type periods: array-like
    :param n_splits: number of validation blocks
    :type n_splits: int
    :param embargo: periods discarded between train and validation
    :type embargo: int
    :param scheme: "expanding" or "rolling"
    :type scheme: str
    :param train_window: periods of history to keep, required when rolling
    :type train_window: int or None
    :return: [(train_row_positions, val_row_positions)]
    :rtype: [(np.ndarray, np.ndarray)]
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if scheme not in ("expanding", "rolling"):
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if scheme == "rolling" and not train_window:
        raise ValueError("train_window is required when scheme='rolling'")

    periods = np.asarray(periods)
    unique = sorted_unique(periods)
    n_periods = len(unique)

    if n_splits > n_periods:
        raise ValueError("n_splits ({}) exceeds the number of periods ({})".format(
            n_splits, n_periods))

    fold_size = n_periods // n_splits
    remainder = n_periods % n_splits

    splits = []
    for fold in range(n_splits):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        if fold == n_splits - 1:
            end += remainder                      # last fold absorbs the remainder

        val_periods = unique[start:end]

        # Training always stops `embargo` periods short of the validation block.
        train_end = max(0, start - embargo)
        if scheme == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - train_window)
        train_periods = unique[train_start:train_end]

        # Early folds can have nothing to train on. Skip rather than fail, so a
        # small n_splits still yields usable folds.
        #
        # min_train_size is the burn-in floor: a fold that trains on a handful
        # of periods still produces a score, and that score still votes on which
        # hyperparameter wins. Without a floor the earliest folds are pure noise
        # given equal weight to the later, well-trained ones.
        if len(train_periods) < max(1, min_train_size) or len(val_periods) == 0:
            continue

        # Hash lookup, not np.isin. Classic eras are zero-padded strings, so the
        # array is object dtype, and np.isin falls back to a sort-based path that
        # compares Python strings: ~3.4s per call on 600k rows versus ~0.02s for
        # a set membership test. This runs once per fold per refit, so the
        # difference is minutes, not milliseconds.
        train_set, val_set = set(train_periods), set(val_periods)
        period_series = pd.Series(periods)
        train_idx = np.where(period_series.isin(train_set).to_numpy())[0]
        val_idx = np.where(period_series.isin(val_set).to_numpy())[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError(
            "no usable CV folds: {} periods with n_splits={}, embargo={}, "
            "min_train_size={}".format(
                n_periods, n_splits, embargo, min_train_size)
        )
    return splits


def sorted_unique(periods):
    """
    Unique periods in chronological order.

    Classic eras are zero-padded strings, Crypto periods are timestamps. Sort
    numerically when the labels look like integers so unpadded eras cannot
    mis-order, and fall back to natural ordering otherwise.
    """
    unique = pd.unique(pd.Series(periods))
    try:
        return np.array(sorted(unique, key=lambda x: int(x)))
    except (TypeError, ValueError):
        return np.array(sorted(unique))


# --------------------------------------------------------------------------- #
# Scoring for the hyperparameter search
# --------------------------------------------------------------------------- #

def predict_signal(estimator, X, task):
    """
    Get a continuous signal out of a fitted estimator.

    Numerai scores by rank correlation, so what matters is the ordering. For
    classification we take the positive-class probability rather than the hard
    label -- collapsing to 0/1 first is exactly the bug that made the original
    pipeline's ROC AUC degenerate into accuracy.
    """
    if task == "classification":
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict(X)


def score_frame(frame, metric, period_col, reference_col):
    """
    Turn a frame of predictions into a single number the search can maximise.

    :param frame: columns "prediction", "target", the period, maybe the reference
    :type frame: pd.DataFrame
    :param metric: one of METRICS
    :type metric: str
    :param period_col: name of the period column
    :type period_col: str
    :param reference_col: name of the reference column, or None
    :type reference_col: str or None
    :return: score, higher is always better
    :rtype: float
    """
    family, _, statistic = metric.partition("_")

    if family == "corr":
        scores = per_period_corr(frame, "prediction", "target", period_col)
    elif family == "bmc":
        # BMC is meaningless without a reference to contribute over.
        if reference_col is None or reference_col not in frame.columns:
            return UNSCORABLE
        scores = per_period_bmc(
            frame, "prediction", reference_col, "target", period_col
        )
    else:
        raise ValueError("unknown metric family '{}'".format(family))

    summary = summarize(scores)

    if statistic == "":
        value = summary["mean"]
    elif statistic == "sharpe":
        value = summary["sharpe"]
    elif statistic == "drawdown":
        # sklearn always maximises, and a smaller drawdown is better, so negate
        # it -- the same convention as sklearn's own neg_mean_squared_error.
        # Get this backwards and the search silently picks the worst candidate.
        value = -summary["max_drawdown"]
    else:
        raise ValueError("unknown metric statistic '{}'".format(statistic))

    return float(value) if np.isfinite(value) else UNSCORABLE


def build_scorer(metric, meta, period_col, reference_col, task,
                 metric2=None, metric_weight=1.0, score_target_col=None):
    """
    Build an sklearn scorer that can see the period and reference columns.

    Optionally blends two objectives:

        score = metric_weight * metric + (1 - metric_weight) * metric2

    A caution on that blend: the components are NOT on a common scale, and the
    weight does not rescale them. Rough magnitudes from real runs --

        corr           0.015 (Classic)   0.14 (Crypto)
        bmc            0.0004            0.05
        *_sharpe       1.0               1.3
        *_drawdown    -0.03             -0.6

    -- so a 50/50 blend of `corr` and `corr_sharpe` on Classic is effectively
    99% sharpe, because 1.0 dwarfs 0.015. Choose the weight against the numbers
    your own runs produce, not against the nominal split. Blending two metrics
    from the same family (corr + bmc, or their two sharpes) is far better
    behaved than blending across families.

    sklearn hands a scorer only (estimator, X, y) -- no era, no benchmark. The
    trick is that sklearn slices DataFrames with .iloc, which preserves the
    original index labels, so the scorer can look the extra columns up in a side
    table keyed by that index.

    This is why X must stay a DataFrame and never become X.values: with a numpy
    array the row labels are gone and there is no way to recover which eras the
    fold covered. It also keeps era and benchmark out of the model's feature
    matrix entirely, so there is nothing for the estimator to accidentally learn
    from.

    :param metric: one of METRICS
    :type metric: str
    :param meta: frame indexed like X, holding the period and reference columns
    :type meta: pd.DataFrame
    :param period_col: name of the period column
    :type period_col: str
    :param reference_col: name of the reference column, or None
    :type reference_col: str or None
    :param task: "regression" or "classification"
    :type task: str
    :return: callable(estimator, X, y) -> float
    """
    for name in (metric, metric2):
        if name is not None and name not in METRICS:
            raise ValueError("unknown metric '{}', expected one of {}".format(
                name, list(METRICS)))
    if not 0.0 <= metric_weight <= 1.0:
        raise ValueError("metric_weight must be in [0, 1]")

    def _scorer(estimator, X, y):
        predictions = predict_signal(estimator, X, task)

        # Recover this fold's periods (and reference) by index label.
        fold_meta = meta.loc[X.index]

        # Score against the ORIGINAL target, not the y the model was fit on.
        # Under --task classification the fit target is binarised to 0/1, but the
        # tournament pays on the 5-bin target -- scoring on the binarised one
        # silently measures a different quantity and makes classification runs
        # incomparable with regression runs. For regression the two are identical.
        if score_target_col is not None and score_target_col in fold_meta.columns:
            target_values = fold_meta[score_target_col].to_numpy()
        else:
            target_values = np.asarray(y, dtype="float64")

        frame = pd.DataFrame(
            {
                "prediction": np.asarray(predictions, dtype="float64"),
                "target": np.asarray(target_values, dtype="float64"),
                period_col: fold_meta[period_col].to_numpy(),
            },
            index=X.index,
        )
        if reference_col is not None and reference_col in fold_meta.columns:
            frame[reference_col] = fold_meta[reference_col].to_numpy()

        primary = score_frame(frame, metric, period_col, reference_col)
        if metric2 is None:
            return primary

        secondary = score_frame(frame, metric2, period_col, reference_col)

        # If either component is unscorable the blend is meaningless, so the
        # candidate loses outright rather than being ranked on half a score.
        if primary <= UNSCORABLE or secondary <= UNSCORABLE:
            return UNSCORABLE

        return metric_weight * primary + (1.0 - metric_weight) * secondary

    return _scorer


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #

def grid_bounds(param_grid):
    """
    Record the span each numeric parameter was actually searched over.

    This matters most for ridge_glmnet, whose alpha path is rebuilt from every
    refit's own training block -- knowing alpha=612 means little without knowing
    it was picked from [70, 700000]. The Bayesian path produces the same triples
    from its own dimensions, see search_spaces.space_bounds().

    param_grid may be a list of sub-grids (the boosters pair max_depth with
    num_leaves that way), so flatten before reading bounds.

    :param param_grid: a dict, or a list of sub-grids
    :return: name -> (min, max, size)
    :rtype: dict
    """
    bounds = {}
    flat = {}
    for sub in (param_grid if isinstance(param_grid, list) else [param_grid]):
        for k, v in sub.items():
            flat.setdefault(k, []).append(v)
    for name, group in flat.items():
        values = np.concatenate([np.asarray(v, dtype="float64")
                                 for v in group if not hasattr(v, "rvs")]) \
                 if any(not hasattr(v, "rvs") for v in group) else group[0]
        try:
            numeric = np.asarray(values, dtype="float64")
        except (TypeError, ValueError):
            continue                      # a distribution, not an explicit list
        if numeric.ndim == 0:
            continue
        if numeric.ndim == 1 and numeric.size:
            bounds[name] = (
                float(numeric.min()), float(numeric.max()), int(numeric.size)
            )
    return bounds


def hyper_params_search(X, y, wrapper, meta, metric, period_col, reference_col,
                        task, n_iter, n_splits, embargo, scheme, train_window,
                        n_jobs, seed, verbose, search_jobs=1,
                        metric2=None, metric_weight=1.0, min_train_size=0,
                        score_target_col=None, search_type="random",
                        bayes_init_points=0):
    """
    Search hyperparameters on one training block, scoring with the chosen metric.

    :param X: training features, kept as a DataFrame (see build_scorer)
    :type X: pd.DataFrame
    :param y: training target
    :type y: pd.Series
    :param wrapper: a model wrapper from prediction.models
    :param meta: period/reference lookup indexed like X
    :type meta: pd.DataFrame
    :param metric: one of METRICS
    :type metric: str
    :param period_col: name of the period column
    :type period_col: str
    :param reference_col: name of the reference column, or None
    :type reference_col: str or None
    :param task: "regression" or "classification"
    :type task: str
    :param n_iter: number of hyperparameter draws
    :type n_iter: int
    :param n_splits: inner CV folds
    :type n_splits: int
    :param embargo: periods between train and validation within the inner CV
    :type embargo: int
    :param scheme: "expanding" or "rolling"
    :type scheme: str
    :param train_window: history length when rolling
    :type train_window: int or None
    :param n_jobs: threads given to the estimator itself
    :type n_jobs: int
    :param seed: random seed
    :type seed: int or None
    :param verbose: search verbosity
    :type verbose: bool or int
    :param search_jobs: processes used to evaluate candidates in parallel
    :type search_jobs: int
    :param search_type: which optimiser consumes the grid -- "random" for
                        RandomizedSearchCV, "bayes" for scikit-optimize's
                        BayesSearchCV. Note this is a different thing from
                        `wrapper.search_type`, which declares the DIALECT the
                        wrapper's grid is written in; both optimisers read that
                        one dialect, see prediction.search_spaces
    :type search_type: str
    :param bayes_init_points: random warm-up draws per sub-grid before the
                              surrogate is first fitted, 0 for auto
    :type bayes_init_points: int
    :return: the fitted search object
    """
    if wrapper.search_type != "random":
        raise ValueError(
            "wrapper grids must be written in the 'random' dialect (scipy "
            "distributions and explicit lists), got '{}'".format(
                wrapper.search_type)
        )
    if search_type not in ("random", "bayes"):
        raise ValueError(
            "unknown search_type '{}', expected 'random' or 'bayes'".format(
                search_type)
        )

    # A wrapper may derive its grid from the training data -- currently only
    # ridge_glmnet, which needs y to locate lambda_max. Doing it here, per refit,
    # is what keeps that computation off the validation rows.
    if hasattr(wrapper, "fit_grid"):
        param_grid = wrapper.fit_grid(X, y)
    else:
        param_grid = wrapper.param_grid

    folds = purged_period_splits(
        meta.loc[X.index, period_col].to_numpy(),
        n_splits=n_splits,
        embargo=embargo,
        scheme=scheme,
        train_window=train_window,
    )

    # Multi-metric scoring. "objective" is what the search refits on; corr and
    # bmc ride along so the report can show CV train *and* validation scores for
    # both families, whatever the objective happened to be. sklearn evaluates
    # each scorer separately, so this costs extra predict calls -- cheap for
    # ridge, and worth it to avoid re-running to learn what a run scored.
    scoring = {
        "objective": build_scorer(metric, meta, period_col, reference_col, task,
                                  metric2=metric2, metric_weight=metric_weight,
                                  score_target_col=score_target_col),
        "corr": build_scorer("corr", meta, period_col, reference_col, task,
                             score_target_col=score_target_col),
    }
    if reference_col is not None:
        scoring["bmc"] = build_scorer("bmc", meta, period_col, reference_col, task,
                                      score_target_col=score_target_col)

    # search_jobs defaults to 1 on purpose. n_jobs is already handed to the
    # estimator, which threads *within* one fit and shares the feature matrix.
    # Setting both would oversubscribe the machine (cores x cores threads) and
    # fork a full copy of the data per worker -- on Classic's "all" feature set
    # that is ~5.8GB each.
    #
    # Raise it when the data is small enough that forking is cheap: on Crypto
    # (22 features, ~100MB) parallelising the search is roughly 4x faster than
    # relying on model-level threading alone.
    if search_type == "bayes":
        # Both optimisers read the same wrapper grid; search_spaces translates
        # it -- including the two grids rebuilt from this block's own data just
        # above -- and sets searched_bounds_ in the same form. Everything the
        # walk-forward reads off the object is identical either way, which is
        # why nothing downstream branches on the search type.
        search = build_bayes_search(
            estimator=clone(wrapper.ModelClass),
            param_grid=param_grid,
            n_iter=n_iter,
            cv=folds,
            scoring=scoring,
            refit="objective",
            search_jobs=search_jobs,
            seed=seed,
            error_score=UNSCORABLE,
            init_points=bayes_init_points,
        )
    else:
        search = RandomizedSearchCV(
            estimator=clone(wrapper.ModelClass),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=folds,
            scoring=scoring,
            n_jobs=search_jobs,
            random_state=seed,
            verbose=1 if verbose else 0,
            refit="objective",        # rank and refit on it, report the rest
            return_train_score=True,  # so the report shows CV train vs validation
            error_score=UNSCORABLE,   # a crashed candidate loses, does not abort
        )
        search.searched_bounds_ = grid_bounds(param_grid)

    search.fit(X, y)

    # If every candidate came back unscorable the search learned nothing and
    # "best" is just whichever one happened to be first. That is far worse than
    # an error, because the run completes and reports plausible metrics.
    #
    # The usual cause is a BMC objective whose reference does not exist in the
    # training window: on Crypto the published meta model starts in 2024, so a
    # benchmark-referenced BMC is undefined across nearly all of the training
    # history the inner folds live in.
    if search.best_score_ <= UNSCORABLE:
        raise ValueError(
            "hyperparameter search could not score any candidate. If the metric "
            "is BMC-based, its reference series has no coverage in the training "
            "window -- pass --bmc-reference meta_proxy, which is defined over "
            "every period, or optimise a corr-based metric instead."
        )

    return search


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #

def walk_forward_fit_and_predict(df, wrapper, features, target_col, period_col,
                                 id_col, meta, reference_col, task, metric,
                                 refit_every, embargo, scheme, train_window,
                                 n_iter, n_splits, n_jobs, seed, verbose,
                                 search_jobs=1, metric2=None, metric_weight=1.0,
                                 extra_cols=(), min_train=0, train_target_col=None,
                                 search_type="random", bayes_init_points=0):
    """
    Refit periodically and predict forward over the validation block.

    The analogue of the original annualy_fit_and_predict(), generalised in two
    ways: the refit cadence is a parameter rather than "once per calendar year",
    and training can either expand from the start of history or roll over a fixed
    window.

    Layout of one iteration:

        [-------- training --------][ embargo ][ predict this block ]

    Training always stops `embargo` periods short of the block being predicted,
    so no training row's forward target window reaches into it.

    Only periods marked "validation" are predicted. The training block is drawn
    from everything chronologically before them, which is what makes the reported
    metrics an honest out-of-sample estimate rather than a mix of in-sample and
    out-of-sample eras.

    :param df: full frame with a "split" column
    :type df: pd.DataFrame
    :param wrapper: model wrapper from prediction.models
    :param features: feature column names
    :type features: [str]
    :param target_col: target column name
    :type target_col: str
    :param period_col: period column name
    :type period_col: str
    :param id_col: row id column name
    :type id_col: str
    :param meta: period/reference lookup indexed like df
    :type meta: pd.DataFrame
    :param reference_col: reference column name, or None
    :type reference_col: str or None
    :param task: "regression" or "classification"
    :type task: str
    :param metric: one of METRICS
    :type metric: str
    :param refit_every: periods predicted per refit
    :type refit_every: int
    :param embargo: periods between training and the predicted block
    :type embargo: int
    :param scheme: "expanding" or "rolling"
    :type scheme: str
    :param train_window: history length when rolling
    :type train_window: int or None
    :param n_iter: hyperparameter draws per refit
    :type n_iter: int
    :param n_splits: inner CV folds
    :type n_splits: int
    :param n_jobs: concurrent workers
    :type n_jobs: int
    :param seed: random seed
    :type seed: int or None
    :param verbose: print progress
    :type verbose: bool
    :return: (predictions frame, list of per-refit records)
    :rtype: (pd.DataFrame, [dict])
    """
    # Fail loudly rather than producing empty training blocks. With
    # scheme="rolling" the training slice is all_periods[train_end - window :
    # train_end], so a falsy window silently yields ZERO training periods for
    # every block -- every block is then skipped and the run dies much later
    # with a confusing "produced no predictions". Catch it at the source.
    if scheme == "rolling" and not train_window:
        raise ValueError(
            "scheme='rolling' needs a positive train_window, got {!r}. Every "
            "training block would be empty.".format(train_window)
        )

    all_periods = sorted_unique(df[period_col].to_numpy())
    position = {period: i for i, period in enumerate(all_periods)}

    val_periods = sorted_unique(
        df.loc[df["split"] == "validation", period_col].to_numpy()
    )
    if len(val_periods) == 0:
        raise ValueError("no validation periods to predict")

    predictions = []
    refits = []
    skipped = []

    # Say how the Bayesian budget divides BEFORE the first search runs. Reported
    # afterwards it would arrive an hour into a booster run, too late to act on;
    # and it is a property of the run, not of one refit, so it is said once.
    #
    # The sub-grid count is read off the static grid rather than a fitted one
    # because fit_grid() only rescales entries -- it never changes how many
    # sub-grids there are -- so this is exact, not an estimate.
    if search_type == "bayes" and verbose:
        grids = wrapper.param_grid
        budgets = split_budget(len(grids) if isinstance(grids, list) else 1, n_iter)
        for note in describe_budget(
                budgets, initial_points(min(budgets), bayes_init_points)):
            print(note)

    for refit_index, block_start in enumerate(range(0, len(val_periods), refit_every)):
        block = val_periods[block_start:block_start + refit_every]

        # Training ends `embargo` periods before the first period we predict.
        train_end = max(0, position[block[0]] - embargo)
        if scheme == "expanding":
            train_start = 0
        else:
            train_start = max(0, train_end - train_window)
        train_periods = all_periods[train_start:train_end]

        # Burn-in floor. Without it, how much history the first predicted block
        # gets is an accident of val_frac: at val_frac=0.95 the walk-forward
        # would train on 45 periods, predict, and have those predictions scored
        # alongside blocks trained on thousands. Skipped blocks are not
        # predicted and therefore never enter the metrics.
        if len(train_periods) < max(1, min_train):
            skipped.append({
                "refit": refit_index,
                "train_periods": int(len(train_periods)),
                "predict_start": str(block[0]),
                "predict_end": str(block[-1]),
            })
            if verbose:
                print("refit {}: only {} training periods (< min_train={}), "
                      "block {} to {} not predicted".format(
                          refit_index, len(train_periods), min_train,
                          block[0], block[-1]))
            continue

        train_mask = df[period_col].isin(train_periods)
        block_mask = df[period_col].isin(block)

        X_train = df.loc[train_mask, features]
        # Fit on the (possibly binarised) training target; everything downstream
        # still reports and scores the original one.
        y_train = df.loc[train_mask, train_target_col or target_col]
        X_block = df.loc[block_mask, features]

        if len(X_train) == 0 or len(X_block) == 0:
            continue

        if verbose:
            print("refit {}: train {:,} rows / {} periods -> predict {} periods".format(
                refit_index, len(X_train), len(train_periods), len(block)))

        search = hyper_params_search(
            X=X_train, y=y_train, wrapper=wrapper, meta=meta, metric=metric,
            period_col=period_col, reference_col=reference_col, task=task,
            n_iter=n_iter, n_splits=n_splits, embargo=embargo, scheme=scheme,
            train_window=train_window, n_jobs=n_jobs, seed=seed, verbose=False,
            search_jobs=search_jobs, metric2=metric2,
            metric_weight=metric_weight, min_train_size=min_train,
            score_target_col=target_col, search_type=search_type,
            bayes_init_points=bayes_init_points,
        )

        block_predictions = predict_signal(search.best_estimator_, X_block, task)

        out = pd.DataFrame({
            id_col: df.loc[block_mask, id_col].to_numpy(),
            period_col: df.loc[block_mask, period_col].to_numpy(),
            target_col: df.loc[block_mask, target_col].to_numpy(),
            "prediction": np.asarray(block_predictions, dtype="float64"),
            "refit": refit_index,
        })
        if reference_col is not None and reference_col in df.columns:
            # The reference is renamed to REFERENCE_COL on load precisely so it
            # can never land on top of our own "prediction" column here.
            if reference_col == "prediction":
                raise ValueError(
                    "the reference column must not be named 'prediction'; it "
                    "would overwrite the model's own output"
                )
            out[reference_col] = df.loc[block_mask, reference_col].to_numpy()
        # Carry any other reference series through so the final report can score
        # MMC as well as BMC without re-joining anything.
        for column in extra_cols:
            if column and column in df.columns and column not in out.columns:
                out[column] = df.loc[block_mask, column].to_numpy()

        predictions.append(out)

        # Pull the winning candidate's train and validation fold scores for
        # every metric, not just the objective. "cv_train_*" is measured on the
        # inner training folds and "cv_val_*" on the held-out inner folds; the
        # gap between them is the overfitting signal, and both differ again from
        # the hold-out numbers computed on the walk-forward predictions.
        cv = search.cv_results_
        best = search.best_index_
        record = {
            "refit": refit_index,
            # The time axis. Without these the hyperparameter history is just a
            # list of numbers with no way to say when each one applied.
            "train_start": str(train_periods[0]),
            "train_end": str(train_periods[-1]),
            "predict_start": str(block[0]),
            "predict_end": str(block[-1]),
            "train_periods": int(len(train_periods)),
            "train_rows": int(len(X_train)),
            "predicted_periods": int(len(block)),
            "cv_score": float(search.best_score_),
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
            "searched_bounds": {
                name: {"min": lo, "max": hi, "size": size}
                for name, (lo, hi, size) in
                getattr(search, "searched_bounds_", {}).items()
            },
        }
        for name in ("objective", "corr", "bmc"):
            for split in ("train", "test"):
                key = "mean_{}_{}".format(split, name)
                if key in cv:
                    label = "cv_{}_{}".format(
                        "train" if split == "train" else "val", name)
                    record[label] = float(cv[key][best])
        refits.append(record)

    if not predictions:
        raise ValueError("walk-forward produced no predictions; check refit_every "
                         "and embargo against the number of periods")

    return pd.concat(predictions, ignore_index=True), refits, skipped


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def filter_features(features, include=None, exclude=None, verbose=True):
    """
    Subset a feature list by regex, include first then exclude.

    Both patterns are full `re.search` matches against the column name, so
    "_rm\\d+$" drops the rolling block and "^feature_sym_(?!tenc$)" isolates the
    ticker one-hot without catching the target encoding that shares its prefix.

    :param features: every feature column found in the data
    :type features: [str]
    :param include: keep only columns matching this, or None for all
    :type include: str or None
    :param exclude: drop columns matching this, applied after include
    :type exclude: str or None
    :return: the surviving feature names, order preserved
    :rtype: [str]
    :raises ValueError: if the filters leave nothing to fit on
    """
    import re

    kept = list(features)
    if include:
        pattern = re.compile(include)
        kept = [c for c in kept if pattern.search(c)]
    if exclude:
        pattern = re.compile(exclude)
        kept = [c for c in kept if not pattern.search(c)]

    if not kept:
        raise ValueError(
            "feature filters removed every column (include={!r}, exclude={!r}) "
            "from {} available".format(include, exclude, len(features)))

    if verbose and len(kept) != len(features):
        print("features: {} -> {} after include={!r} exclude={!r}".format(
            len(features), len(kept), include, exclude))
    return kept


def forecast(tournament, model_name, task="regression", metric="corr",
             data_version=None, feature_set=None, era_step=1, val_frac=0.3,
             refit_every=52, embargo=None, scheme="expanding", train_window=None,
             n_iter=50, n_splits=5, n_jobs=-1, seed=None, verbose=True,
             search_jobs=1, metric2=None, metric_weight=1.0,
             bmc_reference="benchmark", proxy_alpha=1.0, min_train=0,
             feature_include=None, feature_exclude=None,
             search_type="random", bayes_init_points=0, target=None):
    """
    Run one evaluation: load, walk forward, score.

    One call is one configuration -- a single tournament, model, task, metric and
    CV setup. Comparing setups means calling it again with different arguments.

    :return: (predictions frame, per-period table, results dict)
    :rtype: (pd.DataFrame, pd.DataFrame, dict)
    """
    spec = get_spec(tournament)
    period_col = spec["period_col"]
    id_col = spec["id_col"]
    # The target is normally the tournament default, but crypto also ships a
    # 60-day label. Everything downstream keys off this: the embargo is derived
    # from its horizon suffix, the loader drops rows where it has not resolved,
    # and the scorer reads it.
    target_col = target or spec["target"]

    # Embargo is derived from the target horizon unless overridden, because a
    # raw period count means different things in the two tournaments.
    if embargo is None:
        embargo = default_embargo(tournament, target_col)
    if verbose:
        print("embargo = {} periods ({} market days per period)".format(
            embargo, spec["period_days"]))

    df, features = load_tournament_data(
        tournament=tournament, version=data_version, feature_set=feature_set,
        era_step=era_step, val_frac=val_frac, embargo=embargo, verbose=verbose,
        target=target_col,
    )

    # Regex subsetting of the feature list. This is what makes ablation studies
    # cheap: one 812-column dataset on disk, and every leave-one-block-out or
    # block-alone arm is a different --feature-exclude / --feature-include
    # against it. Building a parquet per arm would be ~30 files of duplicated
    # data whose only difference is which columns were dropped, and each one is
    # a chance for the wrong file to be paired with the right run name.
    #
    # Applied AFTER loading so the filter sees the real column names, and before
    # anything is fit, so a dropped feature is invisible to the search, the
    # scorer and the meta proxy alike.
    features = filter_features(features, feature_include, feature_exclude,
                               verbose=verbose)

    # The reference series powers BMC. We always attach it when available, so
    # BMC is reported even when the search optimised CORR.
    reference_col = None
    coverage = 0.0
    try:
        reference, reference_col = load_reference(tournament, data_version, verbose)
        df, coverage = attach_reference(df, reference, tournament, reference_col)
        if verbose:
            print("reference '{}' covers {:.1%} of rows".format(reference_col, coverage))
    except Exception as error:
        # BMC is optional; a missing reference should not sink a CORR run.
        if verbose:
            print("could not load reference ({}); BMC will be skipped".format(error))
        reference_col = None

    # The real meta model, for MMC. Separate from the benchmark above: MMC is
    # what actually gets paid, but it only exists where the meta model has been
    # published, so it is reported on that subset alone.
    # These two are deliberately independent. The meta model can be perfectly
    # usable for MMC while the proxy is impossible to build from it -- on
    # Classic the meta model covers only eras 1133-1218, which sit entirely
    # inside the validation block, so there is nothing in training to fit a
    # proxy on. Failing to build the proxy must not take MMC down with it.
    meta_model_col = None
    meta_coverage = 0.0
    try:
        meta_frame, meta_model_col = load_meta_model(tournament, data_version, verbose)
        df, meta_coverage = attach_meta_model(df, meta_frame, tournament)
        if verbose:
            print("meta model '{}' covers {:.1%} of rows".format(
                meta_model_col, meta_coverage))
    except Exception as error:
        if verbose:
            print("could not load the meta model ({}); MMC will be skipped".format(error))
        meta_model_col = None

    # A Ridge stand-in for the meta model, fit on training rows only and
    # predicted everywhere. This is what gives BMC full-history coverage when
    # the real meta model spans only part of it.
    proxy_periods = 0
    proxy_error = "no meta model available"
    if meta_model_col is not None:
        try:
            df, proxy_periods = build_meta_proxy(
                df, features=features, period_col=period_col,
                train_mask=(df["split"] == "train"), meta_col=meta_model_col,
                alpha=proxy_alpha, verbose=verbose,
            )
            proxy_error = None
        except Exception as error:
            proxy_error = str(error)
            if verbose:
                print("could not build the meta proxy ({}); "
                      "--bmc-reference meta_proxy is unavailable".format(error))

    # Which series BMC scores against.
    if bmc_reference == "meta_proxy":
        if META_PROXY_COL not in df.columns:
            raise ValueError(
                "--bmc-reference meta_proxy is unavailable for '{}': {}. Use "
                "--bmc-reference benchmark instead; MMC is still reported on "
                "whatever subset the real meta model covers.".format(
                    tournament, proxy_error)
            )
        reference_col = META_PROXY_COL
        coverage = float(df[META_PROXY_COL].notna().mean())
        if verbose:
            print("BMC reference set to the meta proxy ({:.1%} coverage)".format(
                coverage))
    elif bmc_reference != "benchmark":
        raise ValueError("bmc_reference must be 'benchmark' or 'meta_proxy'")

    uses_bmc = metric.startswith("bmc") or (metric2 or "").startswith("bmc")
    if uses_bmc and reference_col is None:
        raise ValueError("metric '{}' needs a reference series, which failed to "
                         "load for tournament '{}'".format(metric, tournament))

    # Classification reframes the graded target as a binary up/down label -- but
    # only for fitting. The original target is kept intact so every metric is
    # computed on the quantity the tournament actually pays for, which is what
    # makes a classification run comparable with a regression one.
    train_target_col = target_col
    if task == "classification":
        train_target_col = "_fit_target"
        df[train_target_col] = binarize_target(df[target_col])

    # The side table the scorer reads periods and the reference out of. Indexed
    # exactly like df, which is what makes meta.loc[X.index] work.
    meta_cols = [period_col, target_col] + ([reference_col] if reference_col else [])
    meta = df[meta_cols].copy()

    wrapper = build_wrapper(model_name, task=task, seed=seed, n_jobs=n_jobs)

    predictions, refits, skipped = walk_forward_fit_and_predict(
        df=df, wrapper=wrapper, features=features, target_col=target_col,
        period_col=period_col, id_col=id_col, meta=meta,
        reference_col=reference_col, task=task, metric=metric,
        refit_every=refit_every, embargo=embargo, scheme=scheme,
        train_window=train_window, n_iter=n_iter, n_splits=n_splits,
        n_jobs=n_jobs, seed=seed, verbose=verbose, search_jobs=search_jobs,
        metric2=metric2, metric_weight=metric_weight,
        extra_cols=(meta_model_col,), min_train=min_train,
        train_target_col=train_target_col,
        search_type=search_type, bayes_init_points=bayes_init_points,
    )

    results = evaluate(
        predictions, target_col=target_col, period_col=period_col,
        reference_col=reference_col, meta_model_col=meta_model_col,
    )

    # Per-period corr / bmc / mmc / pnl -- everything a report needs to draw
    # cumulative curves without recomputing anything.
    periods_table = per_period_table(
        predictions, pred_col="prediction", target_col=target_col,
        period_col=period_col, reference_col=reference_col,
        meta_model_col=meta_model_col,
    )
    results["pnl"] = summarize(periods_table["pnl"])
    results["reference"] = {
        "column": reference_col, "row_coverage": coverage,
        "kind": bmc_reference,
        "proxy_fit_periods": proxy_periods if bmc_reference == "meta_proxy" else None,
    }
    results["meta_model"] = {
        "column": meta_model_col, "row_coverage": meta_coverage,
    }
    results["refits"] = refits
    results["skipped_blocks"] = skipped
    results["config"] = {
        "tournament": tournament, "model": model_name, "task": task,
        "target": target_col,
        "metric": metric, "data_version": data_version or spec["version"],
        "feature_set": feature_set or spec.get("feature_set"),
        "n_features": len(features), "era_step": era_step, "val_frac": val_frac,
        # Recorded so an ablation result can never be read without knowing which
        # columns were actually fitted on -- the dataset name alone would say
        # 812 features for every arm in the study.
        "feature_include": feature_include, "feature_exclude": feature_exclude,
        "refit_every": refit_every, "embargo": embargo, "cv_scheme": scheme,
        "train_window": train_window, "n_iter": n_iter, "n_splits": n_splits,
        "seed": seed, "search_jobs": search_jobs, "metric2": metric2,
        # Which optimiser produced these hyperparameters. Without it a bayes run
        # and a random run are indistinguishable in the results tree, and they
        # searched different spaces -- see prediction.search_spaces.
        "search_type": search_type,
        "bayes_init_points": bayes_init_points if search_type == "bayes" else None,
        "metric_weight": metric_weight if metric2 else None,
        "bmc_reference": bmc_reference, "proxy_alpha": proxy_alpha,
        "min_train": min_train,
    }
    return predictions, periods_table, results


def evaluate(predictions, target_col, period_col, reference_col=None,
             meta_model_col=None):
    """
    Summarise a predictions frame into the reported metrics.

    Both CORR and BMC are always reported, with all four statistics each, no
    matter which one drove the hyperparameter search -- so two runs optimised for
    different objectives stay directly comparable.
    """
    return summarize_predictions(
        predictions, pred_col="prediction", target_col=target_col,
        period_col=period_col, reference_col=reference_col,
        meta_model_col=meta_model_col,
    )


def _jsonable(value):
    """Coerce numpy scalars so the results dict survives json.dump()."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value
