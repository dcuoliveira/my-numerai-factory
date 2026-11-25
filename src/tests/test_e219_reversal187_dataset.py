"""reversal187 schema, missingness and generic training guard."""
import json
import numpy as np
import pandas as pd
import pytest
from src.data_mani import build_e219_reversal187 as D
from src.data_mani import reversal_metrics as R
from src.data_mani import utils


def test_187_names_unique_and_grouped_by_family():
    names = D.feature_names()
    assert len(names) == 187 == len(set(names))
    assert all(n.startswith('feature_rev_') for n in names)
    g = D.groups()
    assert list(g) == list(R.FAMILIES) and all(len(v) == 11 for v in g.values())
    assert 'feature_rev_rsi_wilder_ts_2d' in names and 'feature_rev_swing_low_distance_ts_k40' in names


def test_partial_coverage_keeps_observed_zero_and_ignores_momentum():
    keys = pd.DataFrame(dict(row=[0, 1, 2], symbol=['a', 'b', 'c'], ucid=[1, 2, 3],
                             date=pd.to_datetime(['2024-01-01']*3)))
    values = np.zeros((3, 187), np.float32)
    values[1, 12] = np.nan          # residual_btc, window 3
    values[2] = np.nan
    coverage, availability = D.coverage_tables(keys, values, [True, True, False])
    assert coverage.covered.tolist() == [True, False, False]
    assert coverage.short_return.tolist() == [True, True, False]
    assert coverage.residual_btc.tolist() == [True, False, False]
    assert availability.iloc[1].sum() == 186 and values[1, 0] == 0
    assert not any('momentum' in c for c in coverage)


def test_generic_loader_rejects_coverage_required(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, 'DATA_DIR', str(tmp_path))
    root = tmp_path/'crypto'/'rev187'
    root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(dict(missing_value_policy='coverage_required')))
    monkeypatch.setattr(utils, '_load_crypto', lambda *a: pytest.fail('must reject before filling'))
    with pytest.raises(ValueError, match='coverage-aware training'):
        utils.load_tournament_data('crypto', version='crypto/rev187', verbose=False)


@pytest.mark.skipif(not (D.OUTPUT/'manifest.json').exists(), reason='dataset not built')
def test_built_dataset_contract():
    m = json.loads((D.OUTPUT/'manifest.json').read_text())
    assert m['missing_value_policy'] == 'coverage_required' and m['features'] == D.feature_names()
    train = pd.read_parquet(D.OUTPUT/'train.parquet')
    cov = pd.read_parquet(D.OUTPUT/'coverage.parquet')
    assert train.index.equals(cov.index) and len(train) == m['rows']
    v = train[m['features']].to_numpy()
    assert set(np.unique(v[np.isfinite(v)])) <= {0., .25, .5, .75, 1.}
    assert np.isfinite(v[cov.covered.to_numpy()]).all()
    assert not np.isfinite(v[~cov.source_eligible.to_numpy()]).any()
    assert not any(c in train for c in ['covered', 'source_eligible', 'row'])
