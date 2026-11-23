"""Causal horizon switches, native residualization and cross-target scoring."""
import itertools
from src.cluster.budget import Host

import numpy as np
import pandas as pd

from experiments import e211_cross_target_tsfm as S
from src.data_mani import e206_tsfm_embeddings as X
from src.prediction import meta_resid as MR


def test_gpu_worker_reservations_enforce_one_card_per_host():
    host = Host(name='gpu-box', cores=32, ram_gb=251, gpus=2, reachable=True)
    owned = {'nmr-e211_worker': dict(host='gpu-box', gpus=[1])}
    restricted = S.extraction_hosts([host], owned, {'nmr-e211_worker'})[0]
    assert restricted.busy_gpus == (0, 1)
    assert S.extraction_hosts([host], owned, set())[0] == host


def test_both_horizons_resolve_before_validation_and_forecasts():
    p = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=800)})
    for horizon, wall in [(20, 35), (60, 91)]:
        q, protocol = S.fold_panel(p, 650, horizon)
        assert q.loc[q.train, 'date'].max()+pd.Timedelta(days=wall) <= q.loc[q.validation, 'date'].min()
        assert q.loc[q.final_fit, 'date'].max()+pd.Timedelta(days=wall) <= q.loc[q.test, 'date'].min()
        assert q.loc[q.validation, 'date'].max()+pd.Timedelta(days=wall) <= q.loc[q.test, 'date'].min()
        assert q.test.sum() == 130 and q.validation.sum() == 63
        assert not (q.final_fit & q.test).any()


def test_target60_extractor_cannot_read_newer_labels(monkeypatch):
    monkeypatch.setattr(X, 'WALL', 91)
    dates = pd.bdate_range('2020-01-01', periods=400).to_numpy().astype('datetime64[D]')
    y = np.arange(400, dtype=np.float32)
    x = np.tile(y[:, None], (1, 33))
    origin = 350
    original = X.inputs(y, x, dates, origin)
    wall = np.searchsorted(dates, dates[origin]-np.timedelta64(91, 'D'), side='right')-1
    assert original[0][0, -1] == wall
    y[wall+1:] = -999
    x[origin+1:] = 999
    changed = X.inputs(y, x, dates, origin)
    for a, b in zip(original, changed):
        np.testing.assert_array_equal(a, b)


def test_cached_lagged_meta_matches_native_and_future_mutations():
    meta = pd.DataFrame({'date': pd.to_datetime(['2024-01-01']*3+['2024-01-08']*3),
                         'symbol': ['A', 'B', 'C']*2, 'prediction': [.2, .8, .5, .9, .1, .3]})
    symbols = ['C', 'A', 'MISSING', 'B']
    date = pd.Timestamp('2024-01-10')
    values, snapshot = S.lagged_snapshots(meta, date, symbols, {})
    native, expected = MR.lagged_meta(meta, symbols, date)
    np.testing.assert_array_equal(values, native)
    assert snapshot == expected == pd.Timestamp('2024-01-01')
    changed = meta.copy()
    changed.loc[changed.date > date-pd.Timedelta(days=5), 'prediction'] = 1000
    again, _ = S.lagged_snapshots(changed, date, symbols, {})
    np.testing.assert_array_equal(values, again)
    pred = np.array([.8, .2, .6, .4])
    np.testing.assert_array_equal(MR.residualise(pred, values)[0], MR.residualise(pred, native)[0])


def test_all_member_combinations_unique_and_batched_scoring_matches_scalar():
    combos = S.definitions()
    assert len(combos) == 6884 and len(set(combos)) == 6884
    assert {n: sum(len(c) == n for c in combos) for n in range(1, 6)} == {1:16, 2:120, 3:560, 4:1820, 5:4368}
    rng = np.random.default_rng(211)
    members = rng.uniform(size=(31, 16))
    choices = [(0,), (0, 8), (1, 2, 3), (4, 5, 6, 7), (0, 1, 2, 3, 4)]
    predictions = np.column_stack([members[:, c].mean(axis=1) for c in choices])
    y20 = rng.choice([0., .25, .5, .75, 1.], size=31).astype(np.float32)
    y60 = rng.choice([0., .25, .5, .75, 1.], size=31).astype(np.float32)
    meta = rng.uniform(size=31)
    batched = S.score_matrix(predictions, y20, y60, meta)
    for horizon, y in [(20, y20), (60, y60)]:
        for column in range(len(choices)):
            corr, mmc = S.E.matrix_scores(predictions[:, column, None], y, meta)
            np.testing.assert_allclose([batched[horizon][0][column], batched[horizon][1][column]], [corr[0], mmc[0]])
