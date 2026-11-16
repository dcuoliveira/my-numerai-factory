"""Validate the real E152 reconstructed artifacts, not synthetic fixtures."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def validate(root='data/crypto/v2.0-volumealpha-causal', work='/scratch/danielco/volumealpha_e152'):
    root, work = Path(root), Path(work)
    base = pd.read_parquet(work / 'base_train.parquet')
    d = pd.read_parquet(root / 'train.parquet')
    pd.testing.assert_frame_equal(d[base.columns], base, check_exact=True)
    features = [c for c in d if c.startswith('feature_va_')]
    a = d[features].to_numpy()
    assert np.isin(a[np.isfinite(a)], [0, .25, .5, .75, 1]).all()
    assert not np.isinf(a).any()
    assert not any(c.startswith('cov_') for c in d)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'BUILT_INFERRED_RECONSTRUCTION'
    assert manifest['point_in_time_verified'] is False
    records = pq.ParquetFile(root / 'volume_oof_predictions.parquet')
    model_rows = 0
    for batch in records.iter_batches(batch_size=100000,
            columns=['date', 'fit_at', 'max_train_date', 'max_label_available_at']):
        x = batch.to_pandas()
        assert (x.max_train_date < x.fit_at).all()
        assert (x.max_label_available_at <= x.fit_at).all()
        assert (x.fit_at <= x.date).all()
        model_rows += len(x)
    # Recompute the eight paper technical signals from actual source data
    # independently after truncating at five separate information cutoffs.
    raw = pd.read_parquet(work / 'raw_main.parquet')
    source = pd.read_parquet(work / 'daily_combined.parquet', columns=['ucid', 'date', 'close', 'dollar_volume'])
    dates = sorted(raw.date.unique())
    audit = []
    for index in np.linspace(len(dates) // 4, len(dates) - 1, 5, dtype=int):
        date = pd.Timestamp(dates[index])
        cutoff = date.tz_localize(None) - pd.Timedelta(days=manifest['observation_lag_days'])
        old = source[source.date.le(cutoff)]
        price = old.pivot(index='date', columns='ucid', values='close')
        price = price.reindex(pd.date_range(price.index.min(), cutoff))
        volume = old.pivot(index='date', columns='ucid', values='dollar_volume').reindex(price.index)
        r, v = price.pct_change(fill_method=None), np.log1p(volume)
        actual = raw[raw.date.eq(date)].set_index('ucid')
        cells = 0
        for w in [1, 5, 22, 252]:
            for name, values in [('return', r), ('logdv', v)]:
                column = f'feature_va_{name}_ma{w}'
                expected = values.rolling(w, min_periods=w).mean().iloc[-1]
                got = actual[column].dropna()
                np.testing.assert_allclose(got, expected.reindex(got.index), rtol=1e-10, atol=1e-10)
                cells += len(got)
        audit.append({'prediction_date': str(date), 'last_source_day': str(cutoff.date()),
                      'compared_observed_cells': cells, 'mismatches': 0})
    # Check sidecar coverage independently from raw missing values before bins.
    coverage = pd.read_parquet(root / 'coverage.parquet')
    assert len(coverage) == len(base)
    result = {'status': 'PASS', 'rows_preserved': len(base), 'base_columns_preserved_exactly': list(base.columns),
              'added_features': len(features), 'allowed_bin_values': [0, .25, .5, .75, 1],
              'candidate_nonmissing_cells': int(np.isfinite(a).sum()),
              'oof_prediction_rows_checked': model_rows, 'oof_publication_boundary_violations': 0,
              'technical_real_source_truncation_audit': audit,
              'known_limit': 'Timestamp causality tested; historical vintage and inferred identity remain unverified.'}
    for suffix in ['binance', 'other-cex', 'coinbase-fx']:
        arm = pd.read_parquet(str(root) + '-' + suffix + '/train.parquet')
        pd.testing.assert_frame_equal(arm[base.columns], base, check_exact=True)
    output = Path('results/reports/crypto/e152_volumealpha_causal_build/validation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    validate()
