"""
Tests for the FARM forecast-combination layer.

FARM is easier to get subtly wrong than the weighting schemes, because it is not
constrained to a simplex: it regresses the simple average's own error on the
members' deviations, so a sign slip or a leaked period produces a combination that
looks plausible and is worthless. These tests pin the five things that would
silently invalidate an E27 arm:

  * the `equal` arm must BE the simple average, exactly -- it is the baseline
    every other arm is paired against, so any drift there moves all 24 deltas;
  * the estimation window must never reach past the resolution lag;
  * the Gram-matrix L2-Boosting shortcut must equal textbook componentwise
    boosting, since the whole loop is written in terms of D^T D;
  * DS must regress on the FORECASTS, not the deviations, or it silently becomes
    FARM1-with-a-lasso and the paper's comparison is degenerate;
  * the combination must be invariant to shifting beta along the all-ones
    direction, because sum_i d_i = 0 makes that direction unidentified.
"""

import numpy as np
import pandas as pd
import pytest

from src.prediction import farm


def make_panel(n_periods=300, n_symbols=25, seed=0, flip=False):
    """A small panel: two informative members plus one noise member."""
    rng = np.random.default_rng(seed)
    ranks, targets, periods = [], [], []
    for i in range(n_periods):
        signal = rng.normal(0, 1, n_symbols)
        if flip:
            # member 0 informative in the first half, member 1 in the second
            good, bad = (0, 1) if i < n_periods // 2 else (1, 0)
            cols = {}
            cols[good] = signal + rng.normal(0, 0.3, n_symbols)
            cols[bad] = rng.normal(0, 1, n_symbols)
            raw = np.column_stack([cols[0], cols[1], rng.normal(0, 1, n_symbols)])
        else:
            raw = np.column_stack([
                signal + rng.normal(0, 0.4, n_symbols),
                signal + rng.normal(0, 0.8, n_symbols),
                rng.normal(0, 1, n_symbols),
            ])
        block = pd.DataFrame(raw).rank(pct=True).to_numpy()
        ranks.append(block)
        targets.append(signal)
        periods.append(np.full(n_symbols, i))
    return (np.vstack(ranks), np.concatenate(targets), np.concatenate(periods))


# --------------------------------------------------------------------------
# the baseline every arm is measured against
# --------------------------------------------------------------------------

def test_equal_is_exactly_the_simple_average():
    ranks, target, periods = make_panel()
    pred, records = farm.walk_forward_combination(ranks, target, periods, "equal")
    assert np.allclose(pred, ranks.mean(axis=1), atol=0), (
        "the equal arm must BE the simple average, not approximate it")
    assert records == [], "the equal arm estimates nothing and has no updates"


def test_cold_start_returns_the_simple_average_and_says_so():
    ranks, target, periods = make_panel(n_periods=120)
    pred, records = farm.walk_forward_combination(
        ranks, target, periods, "farm1", min_history=10_000)   # never satisfied
    assert np.allclose(pred, ranks.mean(axis=1)), (
        "with no resolved history every arm must fall back to 1/N")
    assert records and all(r["scheme"].startswith("equal") for r in records), (
        "a run that never left the simple average must not be reported as FARM")


@pytest.mark.parametrize("method", ["farm1", "farm2", "farm3", "gr", "ds"])
def test_every_method_returns_finite_predictions(method):
    ranks, target, periods = make_panel()
    pred, _ = farm.walk_forward_combination(ranks, target, periods, method)
    assert np.isfinite(pred).all()
    assert len(pred) == len(ranks)


# --------------------------------------------------------------------------
# leakage -- where a combination study silently becomes worthless
# --------------------------------------------------------------------------

def test_fits_never_read_beyond_the_resolution_lag():
    """Each update must be estimable from strictly earlier, RESOLVED periods."""
    ranks, target, periods = make_panel(n_periods=400)
    lag, update_every, min_history = 20, 22, 60
    _, records = farm.walk_forward_combination(
        ranks, target, periods, "farm1", lag=lag,
        update_every=update_every, min_history=min_history)
    order = sorted(set(periods))
    for record in records:
        start = order.index(int(record["from_period"]))
        assert record["history_periods"] <= max(0, start - lag), (
            "update at period %d used %d periods of history, more than the %d "
            "that had resolved" % (start, record["history_periods"],
                                   max(0, start - lag)))


def test_future_targets_cannot_change_past_predictions():
    """
    The direct test: corrupt every target from the midpoint on. Predictions dated
    before the midpoint minus the lag must not move by a single bit.
    """
    ranks, target, periods = make_panel(n_periods=400)
    cut = 200
    corrupted = target.copy()
    rng = np.random.default_rng(99)
    corrupted[periods >= cut] = rng.normal(0, 50, (periods >= cut).sum())

    clean_pred, _ = farm.walk_forward_combination(ranks, target, periods, "farm1")
    dirty_pred, _ = farm.walk_forward_combination(ranks, corrupted, periods, "farm1")

    past = periods < cut
    assert np.array_equal(clean_pred[past], dirty_pred[past]), (
        "corrupting the future changed a past prediction -- the loop is peeking")
    assert not np.allclose(clean_pred[~past], dirty_pred[~past]), (
        "corrupting the future changed nothing at all -- the fit is not binding, "
        "so this test would not detect a leak either")


# --------------------------------------------------------------------------
# the estimators
# --------------------------------------------------------------------------

def test_gram_l2boosting_equals_textbook_componentwise_boosting():
    rng = np.random.default_rng(3)
    design = rng.normal(0, 1, (500, 4))
    design -= design.mean(axis=0)
    u = design @ np.array([0.5, -0.2, 0.0, 0.1]) + rng.normal(0, 0.5, 500)

    gram, cross = design.T @ design, design.T @ u
    fast = farm.fit_l2boost(design, u, gram, cross, nu=0.01, iterations=200)

    # textbook: recompute the residual over the ROWS every step
    beta = np.zeros(4)
    for _ in range(200):
        resid = u - design @ beta
        proj = design.T @ resid
        gnorm = (design ** 2).sum(axis=0)
        j = int(np.argmax(np.abs(proj) / np.sqrt(gnorm)))
        beta[j] += 0.01 * proj[j] / gnorm[j]

    assert np.allclose(fast, beta, atol=1e-12), (
        "the Gram shortcut must reproduce componentwise boosting exactly")


def test_boosting_starts_from_the_simple_average():
    """Zero iterations must leave beta at 0, i.e. the combination at f_bar."""
    rng = np.random.default_rng(4)
    design = rng.normal(0, 1, (100, 3))
    u = rng.normal(0, 1, 100)
    beta = farm.fit_l2boost(design, u, design.T @ design, design.T @ u, iterations=0)
    assert np.allclose(beta, 0.0)


def test_gr_weights_sum_to_one():
    rng = np.random.default_rng(5)
    forecasts = rng.uniform(0, 1, (300, 4))
    target = forecasts @ np.array([0.4, 0.3, 0.2, 0.1]) + rng.normal(0, 0.1, 300)
    weights = farm.fit_gr(forecasts, target)
    assert abs(weights.sum() - 1.0) < 1e-8, (
        "Granger-Ramanathan constrains the combining weights to sum to one")


def test_ds_regresses_on_forecasts_not_deviations():
    """
    DS (Egalitarian Lasso) must fit on the f_i. Fitting on the d_i instead makes
    it identical to FARM1-with-a-lasso, because sum_i d_i = 0 removes the f_bar
    direction that distinguishes the two designs.
    """
    ranks, target, periods = make_panel(n_periods=300)
    fbar = ranks.mean(axis=1)
    deviations = ranks - fbar[:, None]
    target_rank = np.concatenate(
        [farm._rank_pct(target[periods == p]) for p in sorted(set(periods))])
    ubar = target_rank - fbar

    on_forecasts = farm.fit_ds(ubar, ranks)
    from sklearn.linear_model import LassoLarsIC
    on_deviations = LassoLarsIC(criterion="bic", fit_intercept=False).fit(
        deviations, ubar).coef_

    assert not np.allclose(on_forecasts, on_deviations), (
        "DS fitted on the deviations is not the paper's Egalitarian Lasso")


# --------------------------------------------------------------------------
# the identification the decomposition rests on
# --------------------------------------------------------------------------

def test_deviations_sum_to_zero_so_beta_is_shift_invariant():
    """
    sum_i d_i = 0 by construction, so adding a constant to every beta leaves the
    combined forecast untouched. Only the CONTRASTS between members are
    identified -- which is why the paper's sum-to-one constraint is free here.
    """
    ranks, _, periods = make_panel(n_periods=50)
    fbar = ranks.mean(axis=1)
    deviations = ranks - fbar[:, None]
    assert np.allclose(deviations.sum(axis=1), 0.0, atol=1e-12)

    beta = np.array([0.3, -0.1, 0.5])
    shifted = beta + 7.0
    assert np.allclose(deviations @ beta, deviations @ shifted, atol=1e-10)


def test_rank_pct_matches_pandas():
    rng = np.random.default_rng(6)
    values = rng.integers(0, 5, 40).astype(float)   # deliberate ties
    assert np.allclose(farm._rank_pct(values),
                       pd.Series(values).rank(pct=True).to_numpy())


def test_unknown_method_is_rejected():
    ranks, target, periods = make_panel(n_periods=80)
    with pytest.raises(ValueError):
        farm.walk_forward_combination(ranks, target, periods, "magic")
