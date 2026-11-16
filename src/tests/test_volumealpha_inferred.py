"""Tests of the explicitly inferred data-source workaround (no vintage claims)."""
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_mani import build_volumealpha_inferred as B
from src.analysis.volumealpha_evaluation import arm_families


def test_inferred_identity_excludes_reused_ticker_and_keeps_permanent_ids(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'onchain').mkdir(parents=True)
    (source / 'onchain/llama_protocols.json').write_text(json.dumps([
        {'cmcId': '99', 'address': 'ethereum:0x99'}, {'cmcId': '99', 'address': 'ethereum:0x99'}]))
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(B, 'SOURCE', source)
    monkeypatch.setattr(B, 'WORK', work)
    panel = pd.DataFrame({'symbol': ['BTC', 'X', 'REUSE', 'REUSE'], 'ucid': ['1', '99', '4', '5']})
    out, excluded = B.identities(panel)
    assert excluded == ['REUSE']
    assert set(out.ucid) == {'1', '99'}
    assert out.set_index('ucid').loc['99', 'asset_id'] == 'ethereum:0x99'
    assert not out.historical_mapping_verified.any()


def test_sparse_source_arms_do_not_require_unavailable_families():
    out = arm_families({'technical': ['feature_va_x']})
    assert 'TECHNICAL' in out
    assert 'EVENTS' not in out and 'COSTS' not in out and 'MODEL_COMMON' not in out
    assert all(set(groups) <= {'technical'} for groups in out.values())


def test_vectorized_features_are_causal_with_calendar_gaps():
    rng = np.random.default_rng(3)
    grid = pd.date_range('2020-01-01', periods=300)
    pieces = []
    for a in ['1', '1027', '2', '74']:
        price = np.exp(np.cumsum(rng.normal(0, 0.02, len(grid)))) * 100
        pieces.append(pd.DataFrame({'ucid': a, 'date': grid, 'close': price, 'high': price * 1.02,
                                   'low': price * 0.98, 'dollar_volume': np.exp(rng.normal(10, 1, len(grid))),
                                   'venue_count': 2, 'venue_hhi': 0.5}))
    daily = pd.concat(pieces, ignore_index=True)
    panel = daily[['ucid', 'date']].copy()
    panel['symbol'] = panel.ucid
    for w in [20, 60]:
        # Give the gate its matching causal date-specific reference (no labels).
        wide = daily.pivot(index='date', columns='ucid', values='close')
        vol = wide.pct_change(fill_method=None).rolling(w, min_periods=w).std().shift(2)
        v = vol.rank(axis=1, pct=True).stack(future_stack=True)
        panel['feature_volatility_' + str(w) + 'd'] = v.reindex(pd.MultiIndex.from_frame(panel[['date', 'ucid']])).to_numpy()
    ids = pd.DataFrame({'ucid': ['1', '1027', '2', '74'], 'asset_id': ['bitcoin:native', 'ethereum:native', 'litecoin:native', 'dogecoin:native']})
    a, _, _ = B.compute_raw(daily, panel, ids, metrics=False)
    cutoff = pd.Timestamp('2020-08-01', tz='UTC')
    changed = daily.copy()
    changed.loc[changed.date >= cutoff.tz_localize(None) - pd.Timedelta(days=1), ['close', 'dollar_volume']] *= 100
    b, _, _ = B.compute_raw(changed, panel, ids, metrics=False)
    cols = [c for c in a if c.startswith('feature_va_')]
    first = a[a.date <= cutoff].set_index(['ucid', 'date'])[cols].sort_index()
    second = b[b.date <= cutoff].set_index(['ucid', 'date'])[cols].sort_index()
    assert first.notna().any().any()
    assert_frame_equal(first, second)


def test_capped_auxiliary_fits_remain_earlier_and_direct_level_forecasts_exist():
    from src.tests.test_volumealpha_causal import auxiliary_fixture
    from src.prediction.volumealpha import walk_forward_volume
    raw, targets = auxiliary_fixture()
    out, predictions, logs = walk_forward_volume(raw, targets, ['feature_va_logdv_ma5'],
        refit_every=15, min_dates=10, min_rows=15, iterations=2, max_train_rows=25)
    assert predictions['target'].eq('logdv_1d').any()
    assert out.feature_va_hat_logdv_1d.notna().any()
    assert all(x['rows'] <= 25 for x in logs)
    assert (predictions.max_train_date < predictions.fit_at).all()
    assert (predictions.max_label_available_at <= predictions.fit_at).all()


def test_auxiliary_stability_bounds_use_only_training_labels():
    from src.prediction.volumealpha import bound_to_training_range
    got = bound_to_training_range(np.array([-1e9, 3., 1e9]), [1., 2., 4.])
    np.testing.assert_array_equal(got, [1., 3., 4.])


def test_constant_interaction_omitted_without_changing_base_or_index(tmp_path):
    from src.data_mani.prune_volumealpha_constants import prune_dataset
    base = pd.DataFrame({'symbol': ['A', 'B', 'A', 'B'], 'date': pd.to_datetime(['2020-01-01'] * 2 + ['2020-01-02'] * 2),
                         'target': [0., 1., .25, .75]}).set_index('symbol')
    d = base.assign(feature_va_weekend=0.5, feature_va_volume=[0., 1., .25, .75])
    d.to_parquet(tmp_path / 'train.parquet')
    pd.DataFrame({'date': d.date.to_numpy(), 'feature_va_weekend': 0., 'feature_va_volume': [1., 2., 3., 4.]}).to_parquet(tmp_path / 'continuous_panel.parquet')
    pd.DataFrame({'date': d.date.to_numpy(), 'ucid': ['1', '2'] * 2, 'cov_events': True, 'cov_technical': True}).to_parquet(tmp_path / 'coverage.parquet')
    (tmp_path / 'manifest.json').write_text(json.dumps({'families': {'events': ['feature_va_weekend'], 'technical': ['feature_va_volume']}}))
    result = prune_dataset(tmp_path)
    out = pd.read_parquet(tmp_path / 'train.parquet')
    assert_frame_equal(out[base.columns], base)
    assert 'events' not in result['families']
    assert 'feature_va_weekend' not in out
    assert result['feature_count'] == 1
