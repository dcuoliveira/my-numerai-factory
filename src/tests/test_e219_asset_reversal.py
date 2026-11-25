"""E219 reversal definitions: causality, hand-checked formulas, runner wiring."""
import numpy as np
import pandas as pd
import pytest
from experiments import e219_asset_reversal as E
from src.analysis import asset_momentum as M
from src.data_mani import reversal_metrics as R


def synthetic_prices(n_days=900, symbols=('BTC', 'ETH', 'AAA', 'BBB'), seed=219):
    rng = np.random.default_rng(seed)
    days = pd.date_range('2020-01-01', periods=n_days)
    rows = []
    for s in symbols:
        c = np.exp(np.cumsum(rng.normal(0, .03, n_days)))
        o = c*np.exp(rng.normal(0, .01, n_days))
        rows.append(pd.DataFrame(dict(symbol=s, date=days, open=o, close=c,
                                      high=np.maximum(o, c)*1.02, low=np.minimum(o, c)*.98,
                                      volume=rng.lognormal(10, 1, n_days))))
    p = pd.concat(rows, ignore_index=True)
    p.loc[p.sample(frac=.001, random_state=1).index, 'close'] = np.nan   # missing bars stay missing
    return p


@pytest.mark.parametrize('family', list(R.FAMILIES))
def test_every_family_is_causal_under_source_truncation(family):
    p = synthetic_prices()
    bars = R.wide_bars(p)
    f = {'btc': bars['close']['BTC'], 'eth': bars['close']['ETH']}
    cut = bars['close'].index[400]
    short = R.wide_bars(p[p.date <= cut])
    sf = {'btc': short['close']['BTC'], 'eth': short['close']['ETH']}
    for slot in (0, 4, 8):
        full = R.signal(bars, family, slot, f)
        prefix = R.signal(short, family, slot, sf)
        np.testing.assert_allclose(full.loc[:cut].to_numpy(), prefix.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
        assert np.isfinite(full.to_numpy()).mean() > .3, family


def test_hand_formulas():
    p = synthetic_prices()
    b = R.wide_bars(p)
    f = {'btc': b['close']['BTC'], 'eth': b['close']['ETH']}
    c, h, l = b['close']['AAA'], b['high']['AAA'], b['low']['AAA']
    t = 300
    slot = R.WINDOWS.index(5)
    np.testing.assert_allclose(R.signal(b, 'short_return', slot, f)['AAA'].iloc[t], c.iloc[t]/c.iloc[t-5]-1)
    stoch = (c.iloc[t]-l.iloc[t-4:t+1].min())/(h.iloc[t-4:t+1].max()-l.iloc[t-4:t+1].min())
    np.testing.assert_allclose(R.signal(b, 'stochastic_k', slot, f)['AAA'].iloc[t], stoch)
    # Residual on BTC with a beta from 60 returns ending t equals manual OLS.
    r, m = c.pct_change(fill_method=None), f['btc'].pct_change(fill_method=None)
    win = pd.concat([r, m], axis=1).iloc[t-59:t+1]
    if win.notna().all().all():
        beta = np.polyfit(win.iloc[:, 1], win.iloc[:, 0], 1)[0]
        manual = (r.iloc[t-4:t+1] - beta*m.iloc[t-4:t+1]).sum()
        np.testing.assert_allclose(R.signal(b, 'residual_btc', slot, f)['AAA'].iloc[t], manual, rtol=1e-8)
    rsi = R.signal(b, 'rsi_sma', slot, f)['AAA']
    assert ((rsi.dropna() >= 0) & (rsi.dropna() <= 100)).all()


def test_swing_pivot_enters_only_after_confirmation():
    idx = pd.bdate_range('2021-01-01', periods=30)
    high = pd.DataFrame({'X': np.r_[np.arange(10.), 20., np.arange(10.)[::-1], np.ones(9)]}, index=idx)
    close = high*.99
    k = 3
    d = R._swing_distance(close, high, k, True)
    assert d['X'].iloc[:10+k].isna().all()              # peak at bar 10 unknown before bar 13
    np.testing.assert_allclose(d['X'].iloc[10+k], np.log(close['X'].iloc[13]/20.))


def test_runner_wiring():
    assert len(E.FORMULAS) == 17 and len(E.TASKS) == 36
    assert E.GROUPS['momentum132'] == 132
    assert len(R.WINDOWS) == len(R.SWING_K) == 11
    v = M.prior_bins(np.random.default_rng(0).normal(size=(400, 2)))
    assert set(np.unique(v[np.isfinite(v)])) <= {0., .25, .5, .75, 1.}
