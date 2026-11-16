"""
Tests for the factor-momentum stake timing backtest (E55).

    python3 -m pytest src/tests/test_factor_momentum_stake.py -q

Two invariants carry the result: a rule must be blind to the future (the 20-day
lag), and the withdraw-delay ladder must actually smooth the target.
"""

import numpy as np
import pandas as pd

from src.analysis import factor_momentum_stake as fm


def _panel(n=260, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-25", periods=n, freq="D")
    return pd.DataFrame(rng.normal(0.09, 0.07, (n, k)), index=idx,
                        columns=["M%d" % i for i in range(k)])


def test_equal_weight_is_the_panel_mean():
    p = _panel()
    path, w, _ = fm.simulate(p, "equal")
    assert np.allclose(w.to_numpy(), 1.0 / p.shape[1])
    assert np.allclose(path.to_numpy(), p.mean(axis=1).to_numpy())


def test_weights_sum_to_one_every_scheme():
    p = _panel()
    for kw in ({"rule": "ztilt", "lookback": 60}, {"rule": "topk", "lookback": 60, "topk": 2},
               {"rule": "ewma_placeholder"} if False else {"rule": "ztilt", "halflife": 20}):
        _, w, _ = fm.simulate(p, **kw)
        s = w.sum(axis=1).to_numpy()
        assert np.allclose(s, 1.0, atol=1e-9), kw


def test_no_lookahead_future_payout_cannot_move_an_earlier_weight():
    """Perturb the LAST payout; every REALISED weight decided before the last
    lag+ladder rounds must be untouched -- a rule peeking at the future shifts."""
    p = _panel()
    for kw in ({"rule": "ztilt", "lookback": 60}, {"rule": "ztilt", "halflife": 20},
               {"rule": "topk", "lookback": 60, "topk": 1}):
        _, base_w, _ = fm.simulate(p, **kw)
        bumped = p.copy(); bumped.iloc[-1] += 10.0
        _, after_w, _ = fm.simulate(bumped, **kw)
        head = slice(0, len(p) - fm.LAG - fm.LADDER)
        a = np.nan_to_num(base_w.iloc[head].to_numpy())
        b = np.nan_to_num(after_w.iloc[head].to_numpy())
        assert np.allclose(a, b), "%s let a future payout move an earlier stake" % kw


def test_ladder_smooths_the_target():
    """Realised weights must move LESS than the target -- the withdraw-delay
    glide reduces turnover; a target that jumps must arrive gradually."""
    p = _panel()
    _, realised, target = fm.simulate(p, "topk", lookback=40, topk=1)
    assert fm.turnover(realised) < fm.turnover(target)
    # and the realised weight is exactly the trailing-ladder mean of the target
    manual = target.rolling(fm.LADDER, min_periods=1).mean()
    manual = manual.div(manual.sum(axis=1), axis=0)
    assert np.allclose(realised.to_numpy(), manual.to_numpy(), atol=1e-9)


def test_floor_zeros_dust_weights():
    p = _panel()
    _, w, _ = fm.simulate(p, "ztilt", lookback=60, floor=0.05)
    nz = w.to_numpy()
    assert ((nz > 0) & (nz < 0.05)).sum() == 0        # nothing between 0 and floor
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
