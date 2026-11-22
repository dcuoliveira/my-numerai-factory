"""E205 causality, model independence and scalar numerical references."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.analysis import asset_momentum as M


def test_ridge_matches_independent_sklearn_with_missing_rows():
    rng = np.random.default_rng(2)
    x, y, q = rng.normal(size=(3, 30)), rng.normal(size=(3, 30)), rng.normal(size=(3, 7))
    x[1, :5] = np.nan
    y[2, :4] = np.nan
    alphas = [.001, .5, 100]
    result = M.ridge_candidates(x, y, q, alphas)
    for g in range(3):
        ok = np.isfinite(x[g]) & np.isfinite(y[g])
        scale = StandardScaler().fit(x[g, ok, None])
        for j, a in enumerate(alphas):
            est = Ridge(alpha=a).fit(scale.transform(x[g, ok, None]), y[g, ok])
            np.testing.assert_allclose(result[g, :, j], est.predict(scale.transform(q[g, :, None])), atol=1e-12)


def test_initialisation_exactly_matches_sklearn():
    from sklearn.neural_network import MLPRegressor
    arch = (8, 3)
    est = MLPRegressor(hidden_layer_sizes=arch, activation='identity', random_state=7)
    est._random_state = np.random.RandomState(7)
    est._initialize(np.zeros((4, 1)), [1, 8, 3, 1], np.float64)
    for (w, b), sw, sb in zip(M.sklearn_weights(arch, 7), est.coefs_, est.intercepts_):
        np.testing.assert_array_equal(w, sw)
        np.testing.assert_array_equal(b, sb)


def scalar_torch(x, y, q, params, seed, epochs, tol):
    import torch
    from src.prediction.torch_mlp import TorchMLPRegressor

    class SklearnInit(TorchMLPRegressor):
        def _build(self, n_features, torch, nn, generator):
            net = super()._build(n_features, torch, nn, generator)
            with torch.no_grad():
                for layer, (w, b) in zip([z for z in net if isinstance(z, nn.Linear)],
                                         M.sklearn_weights(self.hidden_layer_sizes, self.random_state, n_features)):
                    layer.weight.copy_(torch.tensor(w.T))
                    layer.bias.copy_(torch.tensor(b))
            return net

    x = x[:, None] if x.ndim == 1 else x
    q = q[:, None] if q.ndim == 1 else q
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    scaler = StandardScaler().fit(x[mask])
    est = SklearnInit(**params, random_state=seed, max_iter=epochs, tol=tol,
                      n_iter_no_change=10, shuffle=False)
    est.fit(scaler.transform(x[mask]), y[mask])
    return est.predict(scaler.transform(q)), est.n_iter_


@pytest.mark.parametrize('tol', [0., 1e-4])
@pytest.mark.parametrize('n_features', [1, 11])
def test_batched_torch_matches_scalar_models_and_stopping(tol, n_features):
    pytest.importorskip('torch')
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 25, n_features))
    y, q = rng.normal(size=(3, 25)), rng.normal(size=(3, 6, n_features))
    x[1, :4] = np.nan
    y[2, :7] = np.nan
    params = [dict(hidden_layer_sizes=(8,), alpha=.01, learning_rate_init=.005,
                   activation='identity', batch_size=1024),
              dict(hidden_layer_sizes=(8,), alpha=.1, learning_rate_init=.02,
                   activation='identity', batch_size=1024),
              dict(hidden_layer_sizes=(8, 3), alpha=.1, learning_rate_init=.001,
                   activation='identity', batch_size=1024)]
    pred, epochs = M.torch_candidates(x, y, q, params, seed=2, max_iter=65, tol=tol)
    for g in range(len(x)):
        for j, p in enumerate(params):
            expected, n = scalar_torch(x[g], y[g], q[g], p, 2, 65, tol)
            np.testing.assert_allclose(pred[g, :, j], expected, rtol=1e-9, atol=1e-10)
            assert epochs[g, j] == n


def test_other_assets_cannot_affect_tuning_predictions():
    pytest.importorskip('torch')
    rng = np.random.default_rng(8)
    x, y, q = rng.normal(size=(2, 15)), rng.normal(size=(2, 15)), rng.normal(size=(2, 4))
    params = [dict(hidden_layer_sizes=(3,), alpha=.01, learning_rate_init=.01)]
    first, _ = M.torch_candidates(x, y, q, params, max_iter=20)
    x[1] *= 70
    y[1] += 80
    q[1] -= 10
    second, _ = M.torch_candidates(x, y, q, params, max_iter=20)
    np.testing.assert_array_equal(first[0], second[0])


def test_all_signal_formulas_and_prior_processing_are_causal():
    rng = np.random.default_rng(4)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .02, (310, 2)), axis=0)),
                         index=pd.bdate_range('2020-01-01', periods=310))
    for family in M.FAMILIES:
        full = M.signal(close, family, 19)
        short = M.signal(close.iloc[:240], family, 19)
        np.testing.assert_allclose(full.iloc[:240], short, rtol=1e-10, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(M.prior_bins(full.to_numpy())[:240], M.prior_bins(short.to_numpy()))
    values = np.arange(20, dtype=float)[:, None]
    bins = M.prior_bins(values, history=8, minimum=4)
    assert np.isnan(bins[:4]).all()
    assert (bins[4:] == 1).all()


def test_rolling_slope_matches_direct_definition():
    close = pd.DataFrame(np.random.default_rng(5).uniform(1, 5000, (80, 2)))
    close.iloc[30, 1] = np.nan
    w = 19
    weights = np.arange(w)-(w-1)/2
    for family in ['absolute_price_slope', 'log_price_slope', 'relative_price_slope']:
        values = np.log(close) if family == 'log_price_slope' else close
        expected = values.rolling(w, min_periods=w).apply(lambda a: a@weights/(weights@weights), raw=True)
        if family == 'relative_price_slope':
            expected /= close.rolling(w, min_periods=w).mean()
        np.testing.assert_allclose(M.signal(close, family, w), expected, rtol=1e-9, atol=1e-10, equal_nan=True)


def test_attachment_cutoff_staleness_and_missing_ticker():
    dates = pd.bdate_range('2024-01-01', periods=4)
    raw = pd.DataFrame({'A': [1., 2., 3., 4.]}, index=dates)
    panel = pd.DataFrame({'date': pd.to_datetime(['2024-01-04', '2024-01-08', '2024-01-10', '2024-01-04'])})
    attached = M.attach(panel, raw, np.array([0, 0, 0, -1]))
    np.testing.assert_allclose(attached, [3., 4., np.nan, np.nan], equal_nan=True)


@pytest.mark.parametrize('candidate', [np.array([np.nan]*5), np.array([3., 1., 1., 2., 4.]),
                                       np.array([np.nan, 8., 8., np.nan, 2.])])
def test_rank_slots_cover_empty_full_and_ties(candidate):
    baseline = np.array([.1, .3, .5, .7, .9])
    routed = M.route_slots(baseline, candidate)
    mask = np.isfinite(candidate)
    np.testing.assert_array_equal(routed[~mask], baseline[~mask])
    np.testing.assert_array_equal(np.sort(routed[mask]), np.sort(baseline[mask]))
    assert np.isfinite(routed).all()


def test_joint_lookback_ridge_matches_sklearn():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(2, 60, 11))
    y, q = rng.normal(size=(2, 60)), rng.normal(size=(2, 9, 11))
    x[1, :8, 4] = np.nan
    # Include exactly collinear lookbacks to exercise small-eigenvalue handling.
    x[:, :, 10] = x[:, :, 9]
    q[:, :, 10] = q[:, :, 9]
    alphas = [.0001, 1., 100.]
    pred = M.ridge_candidates(x, y, q, alphas)
    for g in range(2):
        ok = np.isfinite(x[g]).all(axis=1)
        scaler = StandardScaler().fit(x[g, ok])
        for j, a in enumerate(alphas):
            est = Ridge(alpha=a).fit(scaler.transform(x[g, ok]), y[g, ok])
            np.testing.assert_allclose(pred[g, :, j], est.predict(scaler.transform(q[g])), atol=1e-9)


@pytest.mark.parametrize('model', ['ridge', 'mlp_torch'])
def test_fold_joint_inputs_never_reads_outer_labels(tmp_path, monkeypatch, model):
    from experiments import e205_asset_momentum as E
    if model == 'mlp_torch':
        pytest.importorskip('torch')
    rng = np.random.default_rng(15)
    dates = pd.bdate_range('2020-01-01', periods=282).to_numpy()
    frame = pd.DataFrame([(a, d) for a in [1, 2, 3] for d in dates], columns=['ucid', 'date'])
    frame['row'] = np.arange(len(frame))
    frame['covered'] = True
    frame[E.TARGET] = rng.uniform(0, 1, len(frame))
    for w in M.WINDOWS:
        frame[f'feature_ts_{w}'] = rng.integers(0, 5, len(frame))*.25
    params = [.01, 1., 100.] if model == 'ridge' else [
        dict(hidden_layer_sizes=[4], alpha=.01, learning_rate_init=.005, activation='identity', batch_size=1024)]
    tr, te = dates[:130], dates[152:]
    monkeypatch.setattr(E, 'ROOT', tmp_path/'first')
    first = pd.read_parquet(E.fit_fold(model, 'test', 0, tr, te, frame, params))
    # No future target contributes to choices, fitting, preprocessing or query eligibility.
    frame.loc[frame.date.isin(te), E.TARGET] += 1000
    monkeypatch.setattr(E, 'ROOT', tmp_path/'second')
    second = pd.read_parquet(E.fit_fold(model, 'test', 0, tr, te, frame, params))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3*130*(1 if model == 'ridge' else 5)


def test_complete_portfolio_scoring_and_fixed_24_model_ensemble(tmp_path, monkeypatch):
    """Exercise actual aggregation/scoring, including uncovered-asset routing."""
    from experiments import e205_asset_momentum as E
    monkeypatch.setattr(E, 'ROOT', tmp_path)
    monkeypatch.setattr(E, 'verify', lambda: {})
    reported = []
    monkeypatch.setattr(E, 'make_report', lambda *args: reported.append(args))
    rng = np.random.default_rng(205)
    keys = pd.DataFrame([(d, a) for d in pd.bdate_range('2024-01-01', periods=4)
                         for a in range(12)], columns=['date', 'ucid'])
    keys['row'] = np.arange(len(keys))
    keys['symbol'] = keys.ucid.map(lambda a: f'A{a:02d}')
    keys[E.TARGET] = rng.uniform(0, 1, len(keys)).astype(np.float32)
    keys['meta_model'] = rng.uniform(0, 1, len(keys))
    E.parquet(tmp_path/'keys.parquet', keys)
    baseline = keys[['row', 'date']].copy()
    baseline['baseline'] = rng.uniform(0, 1, len(keys))
    baseline['baseline_rank'] = baseline.groupby('date').baseline.transform(lambda s: E.rank_normalise(s.to_numpy()))
    baseline['block'] = 0
    E.parquet(tmp_path/'baseline.parquet', baseline)
    tasks = [(m, f) for m in ['ridge', 'mlp_torch'] for f in M.FAMILIES]
    for model, family in tasks:
        parts = []
        for seed in ([0] if model == 'ridge' else E.SEEDS):
            p = keys.loc[keys.ucid < 10, ['row']].copy()
            p['seed'], p['block'] = seed, 0
            p['prediction'] = rng.normal(size=len(p))
            p['asset_mean'] = (p.row.to_numpy() % 12)/12
            parts.append(p)
        E.parquet(tmp_path/'fits'/f'{model}__{family}'/'block0.parquet', pd.concat(parts, ignore_index=True))
        E.save(tmp_path/'fits'/f'{model}__{family}'/'COMPLETE.json',
               dict(task=dict(tag=f'{model}__{family}', model=model)))
    E.finish(tasks)
    table = pd.read_csv(tmp_path/'table.csv')
    assert table[table.seed == 'ensemble'].task.nunique() == 27
    assert len(reported) == 1
    from src.analysis.experiment_db import scan_checkpoint_arms
    indexed = scan_checkpoint_arms('E205', dict(artifact_root=str(tmp_path),
              models=['ridge', 'mlp_torch'], prefix='e205_'))
    assert len(indexed) == 24 and all(row[7] is not None for row in indexed)
    for model in ['ridge', 'mlp_torch', 'ALL24']:
        p = pd.read_parquet(tmp_path/'ensembles'/f'{model}__ALL12__fixed_equal_rank.parquet')
        p = p.merge(baseline[['row', 'baseline_rank']], on='row').merge(keys[['row', 'ucid']], on='row')
        uncovered = p.ucid >= 10
        np.testing.assert_array_equal(p.loc[uncovered, 'routed'], p.loc[uncovered, 'baseline_rank'])
        for _, g in p.groupby('date'):
            np.testing.assert_array_equal(np.sort(g.routed), np.sort(g.baseline_rank))


@pytest.mark.parametrize('model', ['ridge', 'mlp_torch'])
def test_early_fold_with_no_eligible_training_assets(tmp_path, monkeypatch, model):
    from experiments import e205_asset_momentum as E
    monkeypatch.setattr(E, 'ROOT', tmp_path)
    dates = pd.bdate_range('2020-01-01', periods=282).to_numpy()
    frame = pd.DataFrame(dict(row=np.arange(len(dates)), date=dates, ucid=1,
                              covered=False, **{E.TARGET: .5}))
    for w in M.WINDOWS:
        frame[f'feature_ts_{w}'] = np.nan
    path = E.fit_fold(model, 'empty', 0, dates[:130], dates[152:], frame, [])
    out = pd.read_parquet(path)
    assert out.empty and {'row', 'prediction', 'seed', 'block'}.issubset(out)
