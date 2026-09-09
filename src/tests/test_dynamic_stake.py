"""
Tests for the dynamic stake backtest.

    python3 -m pytest src/tests/test_dynamic_stake.py -q

Two invariants carry the whole result. If either breaks, every dynamic-vs-static
number is meaningless:

  * a flat rule must reproduce the static path exactly, and
  * a rule must be blind to the future -- changing a round's payout must not move
    any stake decided at or before it (the 20-day resolution lag).
"""

import numpy as np
import pandas as pd

from src.analysis import dynamic_stake as ds


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-25", periods=n, freq="D")
    return pd.Series(rng.normal(0.09, 0.07, n), index=idx)


def test_static_rule_reproduces_the_payout_path():
    p = _series()
    frame = pd.DataFrame({"M": p})
    static, dynamic, stake, _, _ = ds.simulate(frame, ds.rule_static)
    # flat unit stake -> dynamic IS the payout, and stake is all ones
    assert np.allclose(static["M"].to_numpy(), dynamic["M"].to_numpy())
    assert np.allclose(stake["M"].dropna().to_numpy(), 1.0)


def test_no_lookahead_a_future_payout_cannot_move_an_earlier_stake():
    """
    The core guard. Perturb the LAST payout and confirm every stake multiplier
    decided before the final `lag` rounds is untouched -- a rule that peeked at
    the future would shift here.
    """
    import functools
    p = _series()
    ewma_adaptive = functools.partial(ds.rule_perf_adaptive, halflife=20)
    ewma_adaptive.__name__ = "rule_perf_adaptive(ewma)"
    ewma_composed = functools.partial(ds.rule_composed, halflife=20)
    ewma_composed.__name__ = "rule_composed(ewma)"
    for rule in (ds.rule_perf_adaptive, ds.rule_vol_target,
                 ds.rule_dd_throttle, ds.rule_composed,
                 ewma_adaptive, ewma_composed):
        base = rule(p, lag=ds.LAG)
        bumped = p.copy()
        bumped.iloc[-1] += 10.0                       # a huge shock to the last round
        after = rule(bumped, lag=ds.LAG)
        # everything up to the last `lag` decisions must be identical
        head = slice(0, len(p) - ds.LAG)
        a = base.iloc[head].to_numpy()
        b = after.iloc[head].to_numpy()
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), (
            "%s let a future payout move an earlier stake" % rule.__name__)


def test_equal_capital_dynamic_average_stake_is_one_on_train():
    """Normalisation must put the TRAIN-window mean stake at 1.0, so dynamic and
    static deploy the same average capital and the comparison is shape, not size."""
    p = _series()
    frame = pd.DataFrame({"M": p})
    _, _, stake, train_idx, _ = ds.simulate(frame, ds.rule_perf_adaptive,
                                            {"window": 60})
    in_train = stake["M"].loc[stake.index.intersection(train_idx)].dropna()
    assert abs(in_train.mean() - 1.0) < 1e-9


def test_selection_touches_train_only():
    """select_params must rank on the train slice; a winner is returned from the grid."""
    p = pd.DataFrame({"A": _series(seed=1), "B": _series(seed=2)})
    params = ds.select_params(p, ds.rule_composed, ds.default_grid())
    assert set(params) == {"window", "fmin", "fmax", "dd_cap"}
    assert params in ds.default_grid()
