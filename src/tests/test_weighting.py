"""
Tests for the ensemble weighting schemes.

Weight estimation is the easiest place in this pipeline to leak: the quantity
being estimated (which member scores well) is a function of the target, and the
target does not resolve for 20 periods. So these tests concentrate on the two
things that would silently invalidate a weighting study -- a scheme that ignores
its constraints, and an estimation window that reaches into the future.
"""

import numpy as np
import pandas as pd
import pytest

from src.prediction import weighting


def make_scores(n=200, seed=0):
    """Three members: one clearly good, one middling, one useless."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        0.05 + rng.normal(0, 0.02, n),      # best
        0.02 + rng.normal(0, 0.02, n),
        0.00 + rng.normal(0, 0.08, n),      # noise, high variance
    ])


@pytest.mark.parametrize("scheme", [s for s in weighting.SCHEMES
                                    if s not in weighting.PREDICTION_BASED])
def test_score_based_schemes_return_a_valid_simplex_point(scheme):
    weights = weighting.estimate(scheme, scores=make_scores())
    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-9, "weights must sum to 1"
    assert (weights >= -1e-12).all(), "a Numerai slot cannot hold a short member"
    assert np.isfinite(weights).all()


@pytest.mark.parametrize("scheme", weighting.PREDICTION_BASED)
def test_prediction_based_schemes_return_a_valid_simplex_point(scheme):
    rng = np.random.default_rng(1)
    target = rng.normal(0, 1, 500)
    predictions = np.column_stack([
        target + rng.normal(0, 0.5, 500),   # informative
        target + rng.normal(0, 2.0, 500),
        rng.normal(0, 1.0, 500),            # pure noise
    ])
    weights = weighting.estimate(scheme, predictions=predictions, target=target)
    assert abs(weights.sum() - 1.0) < 1e-9
    assert (weights >= -1e-12).all()
    # The informative member must outweigh the noise one, or the estimator is
    # not doing the job its name claims.
    assert weights[0] > weights[2]


def test_equal_is_exactly_one_over_n():
    assert np.allclose(weighting.equal(make_scores()), np.ones(3) / 3)


def test_inverse_variance_penalises_the_noisy_member():
    weights = weighting.inverse_variance(make_scores())
    assert weights[2] < weights[0], "the high-variance member must get less"
    assert weights[2] < weights[1]


def test_shrinkage_zero_is_equal_weight_and_one_is_raw_tangency():
    scores = make_scores()
    assert np.allclose(weighting.shrunk_tangency(scores, shrinkage=0.0),
                       np.ones(3) / 3)
    shrunk = weighting.shrunk_tangency(scores, shrinkage=0.2)
    raw = weighting.shrunk_tangency(scores, shrinkage=1.0)
    # Shrinking must move every weight toward 1/N, never past it.
    for s, r in zip(shrunk, raw):
        assert abs(s - 1 / 3) <= abs(r - 1 / 3) + 1e-9


def test_caruana_weights_are_quantised_by_the_iteration_count():
    # Coarse weights are the point: the method cannot express the knife-edge
    # solutions that make tangency weights flip sign between subsamples.
    weights = weighting.caruana(make_scores(), iterations=10)
    scaled = weights * 10
    assert np.allclose(scaled, np.round(scaled), atol=1e-9)


def test_caruana_concentrates_on_the_better_member():
    weights = weighting.caruana(make_scores(), iterations=50)
    assert weights[0] > weights[2]


def test_bagging_spreads_weight_relative_to_plain_greedy():
    scores = make_scores()
    greedy = weighting.caruana(scores, iterations=25)
    bagged = weighting.bagged_caruana(scores, iterations=25, bags=20)
    # Bagging exists to reduce selection variance, so it must not be MORE
    # concentrated than the thing it bags.
    assert (bagged > 1e-6).sum() >= (greedy > 1e-6).sum()


def test_complete_subset_reduces_to_equal_when_nothing_is_dropped():
    scores = make_scores()
    weights = weighting.complete_subset(scores, subset_size=2, keep_fraction=1.0)
    assert np.allclose(weights, np.ones(3) / 3), (
        "keeping every subset must reproduce 1/N")


def test_degenerate_input_falls_back_to_equal_rather_than_nan():
    # Identical members: the covariance is singular and the objective is flat.
    scores = np.tile(np.linspace(0, 1, 100)[:, None], (1, 3))
    for scheme in weighting.SCHEMES:
        if scheme in weighting.PREDICTION_BASED:
            continue
        weights = weighting.estimate(scheme, scores=scores)
        assert np.isfinite(weights).all(), scheme
        assert abs(weights.sum() - 1.0) < 1e-9, scheme


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown weighting scheme"):
        weighting.estimate("magic", scores=make_scores())


# --------------------------------------------------------------------------
# The walk-forward loop -- where leakage would actually happen
# --------------------------------------------------------------------------

def test_weights_never_read_beyond_the_resolution_lag():
    """
    A period's weights must be estimable from strictly earlier, RESOLVED data.

    Constructed so the answer flips halfway: member 0 is better in the first
    half, member 1 in the second. If the loop peeked at the future, the weights
    in force during the first half would already favour member 1.
    """
    from types import SimpleNamespace
    from src.run_weighted_ensemble import walk_forward_weights

    n_periods, n_symbols = 400, 20
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
    rng = np.random.default_rng(0)

    rows = []
    for i, date in enumerate(dates):
        target = rng.normal(0, 1, n_symbols)
        good, bad = (0, 1) if i < n_periods // 2 else (1, 0)
        preds = {}
        preds["m%d" % good] = target + rng.normal(0, 0.3, n_symbols)
        preds["m%d" % bad] = rng.normal(0, 1, n_symbols)
        for s in range(n_symbols):
            rows.append({"symbol": "S%d" % s, "date": date,
                         "target_binned_return_20": target[s],
                         "m0": preds["m0"][s], "m1": preds["m1"][s]})
    frame = pd.DataFrame(rows).set_index(["symbol", "date"])
    ranked = frame[["m0", "m1"]].groupby(level=1).rank(pct=True)
    extras = frame[["target_binned_return_20"]]

    args = SimpleNamespace(scheme="caruana", update_every=20, lag=20,
                           min_history=40, window=0, objective="sharpe",
                           shrinkage=0.2, subset_size=2, seed=0)
    weights, records = walk_forward_weights(
        ranked, extras, "target_binned_return_20", "date", args)

    # Every update must be estimated on strictly fewer periods than have elapsed,
    # by at least the lag.
    periods = list(weights.index)
    for record in records:
        start = periods.index(pd.Timestamp(record["from_period"]))
        assert record["history_periods"] <= max(0, start - args.lag), (
            "update at period %d used %d periods of history, more than the "
            "%d that had resolved" % (start, record["history_periods"],
                                      max(0, start - args.lag)))

    # And the regime flip must not be anticipated: while m0 is the better member,
    # the weights in force must not already favour m1.
    first_half = weights.iloc[args.min_history + args.lag:n_periods // 2]
    assert (first_half["m0"] >= first_half["m1"]).mean() > 0.8, (
        "weights anticipated the regime change -- the loop is peeking")


def test_cold_start_uses_equal_weights_and_says_so():
    from types import SimpleNamespace
    from src.run_weighted_ensemble import walk_forward_weights

    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(2)
    rows = []
    for date in dates:
        for s in range(10):
            rows.append({"symbol": "S%d" % s, "date": date,
                         "target_binned_return_20": rng.normal(),
                         "m0": rng.normal(), "m1": rng.normal()})
    frame = pd.DataFrame(rows).set_index(["symbol", "date"])
    ranked = frame[["m0", "m1"]].groupby(level=1).rank(pct=True)

    args = SimpleNamespace(scheme="caruana", update_every=20, lag=20,
                           min_history=1000,   # never satisfied
                           window=0, objective="sharpe", shrinkage=0.2,
                           subset_size=2, seed=0)
    weights, records = walk_forward_weights(
        ranked, frame[["target_binned_return_20"]],
        "target_binned_return_20", "date", args)

    assert np.allclose(weights.to_numpy(), 0.5), "must stay at 1/N"
    assert all("equal" in r["scheme"] for r in records), (
        "a run that never left 1/N must not be reported as its scheme")
