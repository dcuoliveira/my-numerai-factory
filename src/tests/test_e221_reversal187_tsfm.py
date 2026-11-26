"""E221 wiring: portfolio grid, matched masks, dual ridge with design-C kernel plus own X."""
import numpy as np
import pandas as pd
from experiments import e221_reversal187_tsfm as S
from src.analysis import tsfm_momentum_study as N


def panel(tmp_path, monkeypatch, n_assets=3, n_dates=1600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2021-01-01', periods=n_dates)
    p = pd.DataFrame([(a, d) for d in dates for a in range(n_assets)], columns=['ucid', 'date'])
    p['step'] = p.date.factorize()[0]
    for name, mask in zip(['train', 'validation', 'final_fit', 'test'], N.split_masks(p.date)):
        p[name] = mask
    p[S.TARGET] = rng.normal(.5, .1, len(p))
    for j in range(3):
        p[f'feature_rev_{j}'] = rng.choice([0, .25, .5, .75, 1], len(p))
    common = np.ones(len(p), bool); common[::7] = False
    monkeypatch.setattr(S, 'SCR', tmp_path); np.save(tmp_path/'common_rows.npy', common)
    return p, common


def test_28_portfolios():
    t = S.portfolios()
    assert len(t) == 28 == len(set(t)) and sum(x.startswith('X_') for x in t) == 4
    assert S.MODELS == ['chronos2', 't0', 'timestral'] and {m: S.NORMS[m] for m in S.MODELS} == {'chronos2': 'rolling', 't0': 'rolling', 'timestral': 'robust'}


def test_masks_respect_common_and_splits(tmp_path, monkeypatch):
    p, common = panel(tmp_path, monkeypatch)
    sets = S.masks(p)
    assert len(sets) == 3
    for tr, va, ft, te in sets.values():
        assert common[np.r_[tr, va, ft, te]].all()
        assert p.train.to_numpy()[tr].all() and p.test.to_numpy()[te].all()
        assert p.date.iloc[tr].max() < p.date.iloc[va].min() - pd.Timedelta(days=30)


def test_ridge_kernel_plus_x_equals_primal_concatenation(tmp_path, monkeypatch):
    p, _ = panel(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(p.step.max()+1, 5))          # stand-in design-C rows, one per date
    kern = phi @ phi.T; valid = np.ones(len(p), bool)
    X = p[[c for c in p if c.startswith('feature_')]].to_numpy(float)
    preds, records = S.ridge_portfolio(p, {'selection': (kern, valid), 'final': (kern, valid)}, X)
    asset, (tr, va, ft, te) = next(iter(S.masks(p).items()))
    alpha = records[str(asset)]['alpha']
    z = S.standardized(X, ft, np.r_[ft, te])
    A = np.column_stack([phi[p.step.to_numpy()[np.r_[ft, te]]], z])
    a, q = A[:len(ft)], A[len(ft):]
    y = p.loc[ft, S.TARGET].to_numpy()
    am, ym = a.mean(0), y.mean()
    beta = np.linalg.solve((a-am).T@(a-am)+alpha*np.eye(a.shape[1]), (a-am).T@(y-ym))
    np.testing.assert_allclose(preds['final'][te], (q-am)@beta+ym, rtol=1e-6, atol=1e-8)


def test_tabnet_wrapper_is_wide_and_others_unchanged():
    from src.analysis.wide_tabnet import WideTabNetSkRegressor
    w = S.wrapper('tabnet_ktz')
    assert isinstance(w.ModelClass.steps[-1][1], WideTabNetSkRegressor)
    assert S.wrapper('lgbm_ktz').ModelClass is not None
