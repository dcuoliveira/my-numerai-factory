"""Omit features with no observed cross-sectional variation on development data."""
import json
from pathlib import Path

import pandas as pd


def prune_dataset(root):
    root = Path(root)
    d = pd.read_parquet(root / 'train.parquet')
    cols = [c for c in d if c.startswith('feature_va_')]
    constant = d.groupby('date')[cols].nunique(dropna=True).max().le(1)
    removed = constant[constant].index.tolist()
    manifest = json.loads((root / 'manifest.json').read_text())
    if not removed:
        return manifest
    # The frozen dataset predates the registered forward test; no held-out
    # outcome or forward-feature distribution enters this omission rule.
    if pd.to_datetime(d.date).max() >= pd.Timestamp('2026-11-13'):
        raise ValueError('constant-feature rule must be frozen using development dates only')
    d = d.drop(columns=removed)
    continuous = pd.read_parquet(root / 'continuous_panel.parquet').drop(columns=removed)
    coverage = pd.read_parquet(root / 'coverage.parquet')
    families = {k: [c for c in cc if c not in removed] for k, cc in manifest['families'].items()}
    empty = [k for k, cc in families.items() if not cc]
    families = {k: cc for k, cc in families.items() if cc}
    coverage = coverage[[c for c in coverage if not c.startswith('cov_')]]
    for k, cc in families.items():
        coverage['cov_' + k] = continuous[cc].notna().all(axis=1).to_numpy()
    manifest['families'] = families
    manifest['coverage'] = coverage.filter(like='cov_').mean().to_dict()
    manifest['feature_count'] = sum(c.startswith('feature_va_') for c in d)
    manifest['omitted_constant_cross_section_features'] = removed
    manifest['unavailable_families_after_constant_filter'] = empty
    for frame, name in [(d, 'train.parquet'), (continuous, 'continuous_panel.parquet'), (coverage, 'coverage.parquet')]:
        temp = root / (name + '.tmp')
        frame.to_parquet(temp, index=None if name == 'train.parquet' else False)
        temp.replace(root / name)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
