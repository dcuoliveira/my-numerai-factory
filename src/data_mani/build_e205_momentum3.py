"""Combine the three requested, frozen E205 TS formula datasets verbatim.

This is the development-only E205 panel, not a live dataset rebuild. Source
coverage is deliberately unchanged, including E205's common 12-formula mask.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'results/reports/crypto/e205_asset_momentum'
OUTPUT = REPO/'data/crypto/v2.0-e205-ts-momentum3'
FAMILIES = ['price_sma_ratio', 'ewma_return', 'price_ema_ratio']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    prepared = json.loads((SOURCE/'PREPARED.json').read_text())
    plan = json.loads((SOURCE/'registration.json').read_text())
    frames, hashes = [], {}
    for family in FAMILIES:
        path = SOURCE/'datasets'/f'{family}.parquet'
        hashes[str(path)] = sha(path)
        assert hashes[str(path)] == prepared['dataset_hashes'][family], family
        frames.append(pd.read_parquet(path))
    keys = ['row', 'symbol', 'ucid', 'date', 'target_binned_return_20', 'covered']
    base = frames[0][keys].copy()
    features = []
    for family, frame in zip(FAMILIES, frames):
        pd.testing.assert_frame_equal(base[keys], frame[keys])
        cols = [f'feature_ts_{w}' for w in plan['windows']]
        assert [c for c in frame if c.startswith('feature_')] == cols
        for w, col in zip(plan['windows'], cols):
            name = f'feature_{family}_ts_{w}d'
            base[name] = frame[col]
            features.append(name)
            np.testing.assert_array_equal(base[name], frame[col])
    coverage = base[['row', 'symbol', 'ucid', 'date', 'covered']].set_index(['symbol', 'date'])
    train = base.drop(columns=['row', 'covered']).set_index(['symbol', 'date'])
    assert len(features) == 33 and len(train) == prepared['rows']
    assert train.loc[~coverage.covered, features].isna().all().all()
    assert train.loc[coverage.covered, features].notna().all().all()
    assert train.index.is_unique and train.index.get_level_values('date').max() < pd.Timestamp('2026-02-20')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [('train.parquet', train), ('coverage.parquet', coverage)]:
        tmp = OUTPUT/(name+'.partial')
        frame.to_parquet(tmp)
        tmp.replace(OUTPUT/name)
    meta = pd.read_parquet(SOURCE/'keys.parquet', columns=['symbol', 'date', 'meta_model'])
    meta = meta.rename(columns={'meta_model': 'prediction'}).set_index(['symbol', 'date'])
    pd.testing.assert_index_equal(train.index, meta.index)
    meta.to_parquet(OUTPUT/'historical_meta_models.parquet')
    manifest = dict(formulas=FAMILIES, features=features, n_features=33, rows=len(train),
        covered_rows=int(coverage.covered.sum()), assets=int(train.ucid.nunique()),
        source='E205 frozen development datasets; all 33 feature columns copied exactly',
        processing=dict(calendar='weekdays', effective_window='nominal - 1',
                        nominal_windows=plan['windows'], history=252, minimum=126,
                        current_excluded=True, bins=[.05, .25, .75, .95],
                        source_lag_days=1, max_stale_calendar_days=4,
                        cross_sectional_processing=False, one_hot=False),
        coverage='Exact E205 common 12-formula coverage retained; eligibility metadata in coverage.parquet only',
        cutoff='Development rows before 2026-02-20 only; holdout not opened',
        input_hashes=hashes, upstream_provenance=plan['input_hashes'],
        builder_sha256=sha(__file__), output_hashes={p.name:sha(p) for p in OUTPUT.glob('*.parquet')},
        audit=dict(outcome='PASS WITH CONDITIONS',
                   evidence='Source hashes verified; exact source features/keys/masks; NaNs preserved; no fitted transforms or new models',
                   conditions='Inherits E205 vendor-vintage and common-coverage restrictions; development-only, no live or marginal-value claim'))
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(path=str(OUTPUT), features=33, rows=len(train), covered_rows=manifest['covered_rows']), indent=2))


if __name__ == '__main__':
    build()
