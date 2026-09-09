"""
Tests for the cross-sectional dynamic stake backtest.

    python3 -m pytest src/tests/test_xs_stake.py -q

Three invariants carry the whole result. If any breaks, every arm-vs-arm number
is meaningless:

  * every allocator returns weights that sum to 1 (equal capital across arms),
  * no allocator can look ahead -- perturbing a future round's payout must not
    move any weight decided `lag` rounds or more earlier (20-day resolution lag),
  * a flat/equal allocator is deployed once enough history exists.
"""

import numpy as np
import pandas as pd

from src.analysis import xs_stake as X


def _panels(n=300, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-25", periods=n, freq="D")
    cols = ["M%d" % i for i in range(k)]
    payout = pd.DataFrame(rng.normal(0.05, 0.07, (n, k)), index=idx, columns=cols)
    mmc = pd.DataFrame(rng.normal(0.03, 0.05, (n, k)), index=idx, columns=cols)
    return payout, mmc


ALLOCATORS = [
    (X.alloc_equal, {}),
    (X.alloc_single_best, {"recent": 60}),
    (X.alloc_rolling_mv, {"cov_window": 120}),
    (X.alloc_rank, {"k": 3, "recent": 60}),
    (X.alloc_rank_future, {"k": 3, "recent": 60}),
    (X.alloc_hybrid, {"k": 3, "recent": 60, "cov_window": 120}),
]


def test_weights_sum_to_one_and_are_nonnegative():
    payout, mmc = _panels()
    for fn, params in ALLOCATORS:
        _, _, W = X.simulate_xs(payout, mmc, fn, params)
        deployed = W[W.abs().sum(axis=1) > 1e-9]        # rows past the warm-up
        assert len(deployed) > 0, "%s never deployed" % fn.__name__
        assert (deployed.to_numpy() >= -1e-9).all(), "%s went short" % fn.__name__
        assert np.allclose(deployed.sum(axis=1).to_numpy(), 1.0), (
            "%s weights do not sum to 1" % fn.__name__)


def test_no_lookahead_a_future_payout_cannot_move_an_earlier_weight():
    """The core guard: a huge shock to the LAST round must leave every weight
    decided at least `lag` rounds earlier untouched."""
    payout, mmc = _panels()
    head = slice(0, len(payout) - X.LAG)
    for fn, params in ALLOCATORS:
        _, _, base = X.simulate_xs(payout, mmc, fn, params)
        bumped = payout.copy()
        bumped.iloc[-1] += 10.0
        _, _, after = X.simulate_xs(bumped, mmc, fn, params)
        a = np.nan_to_num(base.iloc[head].to_numpy())
        b = np.nan_to_num(after.iloc[head].to_numpy())
        assert np.allclose(a, b), (
            "%s let a future payout move an earlier weight" % fn.__name__)


def test_equal_allocator_is_uniform_when_deployed():
    payout, mmc = _panels(k=5)
    _, _, W = X.simulate_xs(payout, mmc, X.alloc_equal, {})
    deployed = W[W.abs().sum(axis=1) > 1e-9]
    assert np.allclose(deployed.to_numpy(), 1.0 / payout.shape[1])


def test_static_book_holds_only_its_slots():
    payout, mmc = _panels()
    payout = payout.rename(columns={payout.columns[0]: "CURUPIRA_01",
                                    payout.columns[1]: "CURUPIRA_05",
                                    payout.columns[2]: "CURUPIRA_08"})
    mmc = mmc.rename(columns=dict(zip(mmc.columns[:3],
                                      ["CURUPIRA_01", "CURUPIRA_05", "CURUPIRA_08"])))
    _, _, W = X.simulate_xs(payout, mmc, X.alloc_static_book, {})
    deployed = W[W.abs().sum(axis=1) > 1e-9]
    off_book = [c for c in payout.columns if c not in X.STATIC_BOOK]
    assert np.allclose(deployed[off_book].to_numpy(), 0.0), "static book leaked weight off-book"
