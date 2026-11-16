"""
Choosing ridge's penalty two ways on one walk-forward: prequential loss against
time-series cross-validation (E115).

The pipeline picks every hyperparameter the same way. At each refit it cuts
purged chronological folds INSIDE the training window, averages each candidate's
held-out score over the folds, and keeps the best (functions.hyper_params_search).
That is the `tscv` rule here. The `prequential` rule is Dawid's: score each
candidate on the forecasts it actually made out of sample, refit after refit, and
keep the one whose accumulated loss is lowest so far.

    rule          evidence for lambda at refit b
    prequential   every forecast that lambda has made since the warm-up whose
                  target has resolved by refit b -- genuinely out of sample, at
                  the deployed cadence and training size, over all of history
    tscv          n_splits purged folds inside refit b's own training window --
                  pseudo out of sample, on smaller training sets, over the last
                  train_window periods only

The two rules never need different fits, and that is what makes the comparison
clean. At every refit the ridge is fitted for EVERY lambda in one fixed grid on
the same training window, and every lambda predicts the block. Those forecasts do
not depend on which lambda gets picked, so both arms read their block out of the
same matrix and differ ONLY in the column they take.

Two things are easy to get wrong:

* Resolution. A 20-day target is not known until 20 periods later. A prequential
  loss may count a period only if the TRAINING data of the same refit could have
  used it -- position < train_end, the bound the walk-forward already enforces on
  its training rows. Counting the previous block as soon as it was predicted
  would leak up to 20 days of target into the choice.

* A FIXED grid. ridge_glmnet rebuilds its alpha path from each refit's own y, so
  "the 37th alpha" is a different penalty at every refit, and a running loss per
  position on that path would add up unlike things. The grid here is set once,
  on sklearn's alpha scale.
"""

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

from src.prediction.functions import purged_period_splits, sorted_unique


LOSSES = ("corr", "mse", "payout")
RULES = ("prequential", "tscv")

# The CORR weight in Numerai Crypto's payout, 0.1 * corr + mmc. The same number
# as results_table.CORR_WEIGHT, which a test holds this to: the selection
# utility and the reported payout must be one formula.
PAYOUT_CORR_WEIGHT = 0.1


# --------------------------------------------------------------------------- #
# Ridge over a whole grid
# --------------------------------------------------------------------------- #

def ridge_path(X, y, alphas):
    """
    Ridge coefficients for every alpha at once.

    The same estimator as sklearn's Ridge(alpha=a, fit_intercept=True) for each
    a, held to that by the tests, but from ONE SVD of the centred design rather
    than a solve per alpha:

        w(a) = V diag(s / (s^2 + a)) U'(y - ybar),    b(a) = ybar - xbar'w(a)

    :param X: training features, n x p
    :type X: np.ndarray or pd.DataFrame
    :param y: training target, n
    :type y: np.ndarray or pd.Series
    :param alphas: penalties on sklearn's scale, K
    :type alphas: np.ndarray
    :return: (coef p x K, intercept K)
    :rtype: (np.ndarray, np.ndarray)
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    alphas = np.asarray(alphas, dtype="float64")
    x_mean, y_mean = X.mean(axis=0), y.mean()
    U, s, Vt = np.linalg.svd(X - x_mean, full_matrices=False)
    shrink = s[:, None] / (s[:, None] ** 2 + alphas[None, :])
    coef = Vt.T @ (shrink * (U.T @ (y - y_mean))[:, None])
    intercept = y_mean - x_mean @ coef
    return coef, intercept


def design_spectrum(X):
    """
    Smallest and largest eigenvalue of the centred X'X.

    An alpha far below the smallest is ordinary least squares and one far above
    the largest only rescales X'y, which a rank score cannot see -- so this is the
    span a useful grid has to cover, recorded so a run can show its grid did.

    :param X: features, n x p
    :type X: np.ndarray or pd.DataFrame
    :return: (smallest, largest)
    :rtype: (float, float)
    """
    X = np.asarray(X, dtype="float64")
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    return float(s.min() ** 2), float(s.max() ** 2)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def period_losses(predictions, target, period, loss, meta=None):
    """
    One loss per (period, alpha). Lower is always better.

        corr    minus numerai_corr, the per-period score the pipeline's scorer
                averages (metrics.per_period_corr), all K columns in one call. A
                period with no defined correlation is charged 0, as
                results_table.charged_corr charges it.
        mse     mean squared error over the period's rows.
        payout  minus the per-period payout, 0.1 * corr + mmc, built the way
                results_table.payout_series builds it: corr over every row
                (charged 0 when degenerate), mmc = correlation_contribution
                against the meta model over the rows that HAVE one
                (metrics.per_period_bmc drops the rest). A period with no meta
                model has no payout and is NaN -- it does not vote, rather than
                voting 0.

    Averaging within the period first gives every date one vote whatever its
    symbol count, which is how CORR is scored and how both rules average.

    :param predictions: n x K, one column per alpha
    :type predictions: np.ndarray
    :param target: n
    :type target: np.ndarray
    :param period: n period labels
    :type period: np.ndarray
    :param loss: one of LOSSES
    :type loss: str
    :param meta: n meta-model values, NaN where there is none; payout only
    :type meta: np.ndarray or None
    :return: indexed by sorted period, one column per alpha
    :rtype: pd.DataFrame
    """
    if loss not in LOSSES:
        raise ValueError("unknown loss '{}', expected one of {}".format(loss, LOSSES))
    predictions = np.asarray(predictions, dtype="float64")
    target = np.asarray(target, dtype="float64")
    period = np.asarray(period)
    if loss == "payout":
        if meta is None:
            raise ValueError("loss 'payout' needs the meta model")
        meta = np.asarray(meta, dtype="float64")

    if loss == "mse":
        frame = pd.DataFrame((predictions - target[:, None]) ** 2)
        frame["__period__"] = period
        return frame.groupby("__period__", sort=True).mean()

    order = np.argsort(period, kind="stable")
    labels, starts = np.unique(period[order], return_index=True)
    stops = np.append(starts[1:], len(order))
    rows = []
    for start, stop in zip(starts, stops):
        idx = order[start:stop]
        corr = np.nan_to_num(_corr_columns(predictions[idx], target[idx]), nan=0.0)
        if loss == "corr":
            rows.append(-corr)
            continue
        covered = idx[np.isfinite(meta[idx])]
        mmc = _mmc_columns(predictions[covered], meta[covered], target[covered])
        rows.append(-(PAYOUT_CORR_WEIGHT * corr + mmc))
    return pd.DataFrame(rows, index=pd.Index(labels))


def _corr_columns(predictions, target):
    """numerai_corr of every column against one target, NaN where undefined."""
    frame = pd.DataFrame(predictions)
    # Named: numerai_corr looks the target up by its name after to_frame(), so
    # an unnamed Series raises inside it and every column would read as NaN.
    series = pd.Series(target, name="target")
    try:
        return numerai_corr(frame, series).to_numpy(dtype="float64")
    except Exception:
        # One column numerai_corr cannot score must not cost the others theirs.
        out = np.full(frame.shape[1], np.nan)
        for j in range(frame.shape[1]):
            try:
                out[j] = float(numerai_corr(frame[[j]], series).iloc[0])
            except Exception:
                pass
        return out


def _mmc_columns(predictions, meta, target):
    """correlation_contribution of every column over one meta model, NaN where undefined."""
    k = predictions.shape[1]
    if len(meta) < 2:
        return np.full(k, np.nan)
    frame = pd.DataFrame(predictions)
    meta_series = pd.Series(meta, name="meta")
    target_series = pd.Series(target, name="target")
    try:
        return correlation_contribution(frame, meta_series, target_series).to_numpy(
            dtype="float64")
    except Exception:
        out = np.full(k, np.nan)
        for j in range(k):
            try:
                out[j] = float(correlation_contribution(
                    frame[[j]], meta_series, target_series).iloc[0])
            except Exception:
                pass
        return out


# --------------------------------------------------------------------------- #
# The two rules
# --------------------------------------------------------------------------- #

def prequential_choice(losses, cutoff):
    """
    The alpha whose out-of-sample forecasts have the lowest mean loss so far.

    `losses` may hold every out-of-sample period the run will ever have, future
    ones included -- the cutoff is the only thing standing between the choice
    and the future, which is why it is an argument rather than a convention.

    Ties go to the first alpha in grid order. A period whose loss is undefined
    (payout before the meta model exists) is skipped, and is not counted.

    :param losses: period_losses() of the out-of-sample forecasts, periods x alphas
    :type losses: pd.DataFrame
    :param cutoff: the first period NOT yet resolved, all_periods[train_end]
    :return: (column position, mean loss per alpha, periods counted)
    :rtype: (int, np.ndarray, int)
    """
    seen = losses.loc[losses.index < cutoff]
    seen = seen.loc[seen.notna().any(axis=1)]
    if seen.empty:
        raise ValueError(
            "no resolved out-of-sample period with a defined loss before {}: the "
            "prequential rule has nothing to choose on.".format(cutoff))
    mean = seen.mean(axis=0).to_numpy()
    return _argmin(mean), mean, int(len(seen))


def _argmin(mean):
    """argmin that skips an alpha with no defined loss but never picks one."""
    if not np.isfinite(mean).any():
        raise ValueError("no alpha has a defined loss")
    return int(np.nanargmin(mean))


def tscv_losses(X, y, period, alphas, losses, n_splits, embargo, scheme,
                train_window, meta=None):
    """
    Mean held-out loss of every alpha over purged chronological folds inside one
    training window -- the pipeline's own selection rule.

    The folds are functions.purged_period_splits with the arguments
    hyper_params_search hands it, and the average is the unweighted mean over
    folds of each fold's mean per-period loss, which is what RandomizedSearchCV
    reports as mean_test_score. Under loss "corr" the argmin is the alpha the
    pipeline itself would pick from the same grid; test_lambda_selection holds
    it to that.

    A fold with no defined loss at all (payout, before the meta model exists)
    drops out of the average rather than voting; the pipeline has no payout
    objective, so for that loss this is the same folds and the same averaging,
    not a reproduction of anything it runs. When NO fold has one, that loss
    comes back as (None, 0) and the caller decides what to do.

    :param X: training features, n x p
    :type X: np.ndarray
    :param y: training target, n
    :type y: np.ndarray
    :param period: n period labels
    :type period: np.ndarray
    :param alphas: the grid, K
    :type alphas: np.ndarray
    :param losses: which of LOSSES to compute
    :type losses: [str]
    :param meta: n meta-model values, NaN where there is none; payout only
    :type meta: np.ndarray or None
    :return: loss -> (mean loss per alpha, validation periods counted)
    :rtype: dict
    """
    folds = purged_period_splits(period, n_splits=n_splits, embargo=embargo,
                                 scheme=scheme, train_window=train_window)
    per_fold = {loss: [] for loss in losses}
    counted = {loss: 0 for loss in losses}
    for train_idx, val_idx in folds:
        coef, intercept = ridge_path(X[train_idx], y[train_idx], alphas)
        held_out = X[val_idx] @ coef + intercept
        for loss in losses:
            fold = period_losses(held_out, y[val_idx], period[val_idx], loss,
                                 meta=None if meta is None else meta[val_idx])
            defined = fold.notna().any(axis=1)
            if not defined.any():
                continue
            per_fold[loss].append(fold.loc[defined].mean(axis=0).to_numpy())
            counted[loss] += int(defined.sum())
    return {loss: ((np.nanmean(per_fold[loss], axis=0), counted[loss])
                   if per_fold[loss] else (None, 0))
            for loss in losses}


# --------------------------------------------------------------------------- #
# The walk-forward
# --------------------------------------------------------------------------- #

def refit_blocks(all_periods, val_periods, refit_every, embargo, scheme,
                 train_window, min_train=0):
    """
    Every refit block, in time order.

    The scored blocks are the validation blocks exactly as
    functions.walk_forward_fit_and_predict lays them out, so a scored block here
    trains on the same periods as the pipeline's refit of the same index. The
    stride is then carried BACK through the training history, block by block,
    for as long as a block still gets a full training window. Those warm-up
    blocks are predicted and never scored: they are what the prequential rule
    reads before the first scored block.

    :param all_periods: every period, sorted
    :type all_periods: np.ndarray
    :param val_periods: the validation periods, sorted -- the tail of all_periods
    :type val_periods: np.ndarray
    :return: (blocks, skipped) where each block is a dict of positions into
             all_periods plus "scored" and, when scored, "refit"
    :rtype: ([dict], [dict])
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if scheme == "rolling" and not train_window:
        raise ValueError("scheme='rolling' needs a positive train_window")

    n_periods = len(all_periods)
    first_val = n_periods - len(val_periods)
    if not np.array_equal(np.asarray(all_periods[first_val:]), np.asarray(val_periods)):
        raise ValueError("the validation periods must be the tail of all periods")

    floor = max(1, min_train)
    if scheme == "rolling":
        floor = max(floor, train_window)

    def block(start, stop):
        train_end = max(0, start - embargo)
        train_start = 0 if scheme == "expanding" else max(0, train_end - train_window)
        return {"start": start, "stop": stop,
                "train_start": train_start, "train_end": train_end}

    warmup = []
    start = first_val - refit_every
    while start >= 0:
        candidate = block(start, start + refit_every)
        if candidate["train_end"] - candidate["train_start"] < floor:
            break
        candidate["scored"] = False
        warmup.append(candidate)
        start -= refit_every
    warmup.reverse()

    scored, skipped = [], []
    for refit, start in enumerate(range(first_val, n_periods, refit_every)):
        candidate = block(start, min(start + refit_every, n_periods))
        candidate["scored"] = True
        candidate["refit"] = refit
        # The pipeline's burn-in floor (walk_forward_fit_and_predict's min_train):
        # a block below it is not predicted and never enters the metrics.
        if candidate["train_end"] - candidate["train_start"] < max(1, min_train):
            skipped.append({
                "refit": refit,
                "train_periods": candidate["train_end"] - candidate["train_start"],
                "predict_start": str(all_periods[start]),
                "predict_end": str(all_periods[candidate["stop"] - 1]),
            })
            continue
        scored.append(candidate)
    return warmup + scored, skipped


def lambda_walk_forward(df, features, target_col, period_col, id_col, alphas,
                        refit_every, embargo, scheme, train_window, n_splits,
                        losses=("corr", "mse"), min_train=0, extra_cols=(),
                        meta_col=None, arms=None, verbose=True):
    """
    Fit every alpha at every refit, then let each (rule, loss) pick per block.

    `arms` narrows the run to the given (rule, loss) pairs; by default it is
    every rule with every one of `losses`. Only what those arms read is
    computed, which matters at a short cadence: a refit every period is 510
    inner-CV blocks, and a tscv loss nobody asked for would be most of the bill.

    :param df: the loaded frame, with its "split" column
    :type df: pd.DataFrame
    :param alphas: the fixed grid, ascending
    :type alphas: np.ndarray
    :param extra_cols: columns carried into every arm's predictions (the
                       reference and the meta model, for BMC and MMC)
    :type extra_cols: tuple
    :param meta_col: the meta-model column, required for loss "payout". Its
                     values are published with each date, not after it, so a
                     resolved period's meta model is as known as its target.
                     A block where a rule has NO payout evidence (no resolved
                     period, or no inner fold, with a meta model) takes that
                     rule's corr choice instead -- corr is the other half of
                     payout -- and its record says so in "selection_loss".
    :type meta_col: str or None
    :return: dict with
        "arms":     (rule, loss) -> (predictions frame, per-refit records)
        "losses":   loss -> period_losses() of every out-of-sample forecast
        "blocks":   every block, warm-up included
        "skipped":  scored blocks below min_train, as the pipeline records them
        "oos":      row-level frame of every out-of-sample row (id, period,
                    target, block) aligned with "matrix"
        "matrix":   the out-of-sample predictions, rows x alphas
    :rtype: dict
    """
    alphas = np.asarray(alphas, dtype="float64")
    if np.any(np.diff(alphas) <= 0):
        raise ValueError("alphas must be strictly ascending")
    arms = [tuple(a) for a in arms] if arms is not None else \
        [(rule, loss) for rule in RULES for loss in losses]
    for rule, loss in arms:
        if rule not in RULES:
            raise ValueError("unknown rule '{}'".format(rule))
        if loss not in LOSSES:
            raise ValueError("unknown loss '{}'".format(loss))
    losses = [loss for loss in LOSSES if any(loss == l for _, l in arms)]

    def needed(rule):
        """The losses one rule must compute; payout brings corr, its fallback."""
        own = [loss for r, loss in arms if r == rule]
        if "payout" in own and "corr" not in own:
            own.append("corr")
        return [loss for loss in LOSSES if loss in own]

    tscv_needed, preq_needed = needed("tscv"), needed("prequential")
    meta_all = None
    if "payout" in losses:
        if not meta_col or meta_col not in df.columns:
            raise ValueError("loss 'payout' needs meta_col, the meta-model column")
        meta_all = df[meta_col].to_numpy(dtype="float64")

    all_periods = sorted_unique(df[period_col].to_numpy())
    val_periods = sorted_unique(
        df.loc[df["split"] == "validation", period_col].to_numpy())
    blocks, skipped = refit_blocks(all_periods, val_periods, refit_every, embargo,
                                   scheme, train_window, min_train)
    if not any(b["scored"] for b in blocks):
        raise ValueError("no scored block survived; check refit_every, embargo "
                         "and min_train against the number of periods")

    period_values = df[period_col].to_numpy()
    X_all = df[features].to_numpy(dtype="float64")
    y_all = df[target_col].to_numpy(dtype="float64")
    if not np.isfinite(X_all).all():
        raise ValueError("features hold NaN or inf; the closed-form ridge needs a "
                         "finite design")

    # all_periods is sorted and unique, so a row's position is a binary search.
    row_position = np.searchsorted(all_periods, period_values)

    matrices, frames, tscv = [], [], {}
    for index, b in enumerate(blocks):
        train = (row_position >= b["train_start"]) & (row_position < b["train_end"])
        predict = (row_position >= b["start"]) & (row_position < b["stop"])
        coef, intercept = ridge_path(X_all[train], y_all[train], alphas)
        matrices.append(X_all[predict] @ coef + intercept)
        frames.append(pd.DataFrame({"row": np.where(predict)[0], "block": index}))
        if b["scored"] and tscv_needed:
            tscv[index] = tscv_losses(
                X_all[train], y_all[train], period_values[train], alphas, tscv_needed,
                n_splits=n_splits, embargo=embargo, scheme=scheme,
                train_window=train_window,
                meta=None if meta_all is None else meta_all[train])
        if b["scored"] and verbose:
            print("refit {:>3}: train {} -> {} ({} periods) | predict {} -> {}".format(
                b["refit"], all_periods[b["train_start"]],
                all_periods[b["train_end"] - 1], b["train_end"] - b["train_start"],
                all_periods[b["start"]], all_periods[b["stop"] - 1]))

    oos = pd.concat(frames, ignore_index=True)
    matrix = np.vstack(matrices)
    rows = oos["row"].to_numpy()
    oos[id_col] = df[id_col].to_numpy()[rows]
    oos[period_col] = period_values[rows]
    oos[target_col] = y_all[rows]

    period_loss = {loss: period_losses(
        matrix, y_all[rows], period_values[rows], loss,
        meta=None if meta_all is None else meta_all[rows]) for loss in preq_needed}

    bounds = {"alpha": {"min": float(alphas[0]), "max": float(alphas[-1]),
                        "size": int(len(alphas))}}
    built = {arm: ([], []) for arm in arms}
    for index, b in enumerate(blocks):
        if not b["scored"]:
            continue
        in_block = (oos["block"] == index).to_numpy()
        block_rows = rows[in_block]
        cutoff = all_periods[b["train_end"]]

        def choose(rule, loss):
            """(pick, mean loss per alpha, evidence, loss actually used)."""
            if rule == "prequential":
                resolved = period_loss[loss].loc[period_loss[loss].index < cutoff]
                if loss == "payout" and not resolved.notna().any(axis=1).any():
                    return choose(rule, "corr")
                pick, mean, counted = prequential_choice(period_loss[loss], cutoff)
                return pick, mean, counted, loss
            mean, counted = tscv[index][loss]
            if mean is None:
                if loss != "payout":
                    raise ValueError("no inner fold has a defined '{}' loss".format(loss))
                return choose(rule, "corr")
            return _argmin(mean), mean, counted, loss

        for rule, loss in arms:
            pick, mean, counted, used = choose(rule, loss)
            out = pd.DataFrame({
                id_col: oos[id_col].to_numpy()[in_block],
                period_col: oos[period_col].to_numpy()[in_block],
                target_col: oos[target_col].to_numpy()[in_block],
                "prediction": matrix[in_block, pick],
                "refit": b["refit"],
            })
            for column in extra_cols:
                if column and column in df.columns and column not in out.columns:
                    out[column] = df[column].to_numpy()[block_rows]
            predictions, records = built[(rule, loss)]
            predictions.append(out)
            records.append({
                "refit": b["refit"],
                "train_start": str(all_periods[b["train_start"]]),
                "train_end": str(all_periods[b["train_end"] - 1]),
                "predict_start": str(all_periods[b["start"]]),
                "predict_end": str(all_periods[b["stop"] - 1]),
                "train_periods": int(b["train_end"] - b["train_start"]),
                "train_rows": int(((row_position >= b["train_start"])
                                   & (row_position < b["train_end"])).sum()),
                "predicted_periods": int(b["stop"] - b["start"]),
                # Higher is better, as in the pipeline's cv_score.
                "cv_score": float(-mean[pick]),
                "best_params": {"alpha": float(alphas[pick])},
                "searched_bounds": bounds,
                "alpha_index": pick,
                "at_grid_edge": bool(pick in (0, len(alphas) - 1)),
                # How much evidence the choice stood on: resolved
                # out-of-sample periods (prequential) or inner validation
                # periods summed over folds (tscv), with a defined loss.
                "selection_periods": counted,
                # The loss the pick was made on: the arm's own, or "corr"
                # for a payout block with no meta model to score on.
                "selection_loss": used,
            })

    return {
        "arms": {key: (pd.concat(p, ignore_index=True), r)
                 for key, (p, r) in built.items()},
        "losses": period_loss,
        "blocks": [{**b, "start_period": str(all_periods[b["start"]])} for b in blocks],
        "skipped": skipped,
        "oos": oos,
        "matrix": matrix,
    }
