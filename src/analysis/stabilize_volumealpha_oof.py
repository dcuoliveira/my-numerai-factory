"""Apply the causal training-range stability guard to existing OOF fits.

No model selection or return score is used. Reproduces the original per-fit
sample exactly, bounds predictions by that sample's label extrema, and rebuilds
realized-error/cost fields. Unbounded predictions are retained for audit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.volumealpha import auxiliary_metrics, trading_costs


def main():
    w = Path('/scratch/danielco/volumealpha_e152')
    raw = pd.read_parquet(w / 'raw_main.parquet')
    targets = pd.read_parquet(w / 'volume_targets.parquet')
    d = raw.merge(targets, on=['asset_id', 'date'], how='left', validate='1:1').sort_values(['date', 'asset_id']).reset_index(drop=True)
    xcols = [f'feature_va_{kind}_ma{win}' for kind in ['return', 'logdv'] for win in [1, 5, 22, 252]]
    complete = np.isfinite(d[xcols].to_numpy(float)).all(axis=1)
    p = pd.read_parquet(w / 'volume_oof_predictions.parquet')
    original = w / 'volume_oof_predictions_unbounded.parquet'
    if not original.exists():
        p.to_parquet(original, index=False)
    else:
        p = pd.read_parquet(original)
    dates = sorted(d.date.unique())
    fixes = []
    for (fit, target), ix in p.groupby(['fit_at', 'target']).groups.items():
        suffix = {'eta': '1d', 'logdv_1d': '1d', 'logdv_5d': '5d', 'logdv_challenge': 'challenge'}[target]
        tr = (d.date < fit) & (d['ready_' + suffix] <= fit) & complete & d[target].notna()
        positions = np.flatnonzero(tr)
        start = dates.index(fit)
        if len(positions) > 50000:
            positions = np.random.default_rng(152 + start).choice(positions, 50000, replace=False)
        values = d.loc[positions, target].astype(float)
        lo, hi = values.min(), values.max()
        before = p.loc[ix, 'prediction']
        after = before.clip(lo, hi)
        fixes.append({'fit_at': str(fit), 'target': target, 'min_training_target': lo, 'max_training_target': hi,
                      'rows_changed': int((before != after).sum()), 'training_rows': len(positions)})
        p.loc[ix, 'prediction'] = after
    p.to_parquet(w / 'volume_oof_predictions.parquet', index=False)
    f = pd.read_parquet(w / 'forecasts.parquet').set_index(['asset_id', 'date'])
    means = p.groupby(['asset_id', 'date', 'target']).prediction.mean().unstack('target')
    for target, name in {'eta': 'hat_eta_1d', 'logdv_1d': 'hat_logdv_1d', 'logdv_5d': 'hat_logdv_5d', 'logdv_challenge': 'hat_logdv_challenge'}.items():
        f['feature_va_' + name] = means[target].reindex(f.index)
    spread = p[p.target.eq('logdv_1d')].groupby(['asset_id', 'date']).prediction.std(ddof=0)
    f['feature_va_disagreement'] = spread.reindex(f.index)
    f['feature_va_downside_logdv_1d'] = np.minimum(f.feature_va_hat_logdv_1d, f.feature_va_lower_logdv_1d)
    f = f.reset_index()
    aligned = targets.set_index(['asset_id', 'date']).reindex(pd.MultiIndex.from_frame(f[['asset_id', 'date']]))
    f['feature_va_previous_error'] = np.nan
    f['feature_va_uncertainty'] = np.nan
    for _, ix in f.groupby('asset_id').groups.items():
        ix = np.asarray(ix)
        errors = aligned.logdv_1d.iloc[ix].to_numpy(float) - f.feature_va_hat_logdv_1d.iloc[ix].to_numpy(float)
        ready = pd.to_datetime(aligned.ready_1d.iloc[ix], utc=True).astype('int64').to_numpy()
        days = f.date.iloc[ix].astype('int64').to_numpy()
        good = np.isfinite(errors) & aligned.ready_1d.iloc[ix].notna().to_numpy()
        previous, uncertainty = np.full(len(ix), np.nan), np.full(len(ix), np.nan)
        for i, day in enumerate(days):
            resolved = np.flatnonzero(good[:i] & (ready[:i] <= day))
            if len(resolved): previous[i] = errors[resolved[-1]]
            if len(resolved) >= 20: uncertainty[i] = errors[resolved[-120:]].std()
        f.loc[ix, 'feature_va_previous_error'], f.loc[ix, 'feature_va_uncertainty'] = previous, uncertainty
    f.to_parquet(w / 'forecasts.parquet', index=False)
    auxiliary_metrics(f, targets, raw, 20, p).to_csv(w / 'volume_oof_metrics.csv', index=False)
    costs = trading_costs(f, raw)
    full = raw.merge(f, on=['asset_id', 'date'], how='left', validate='1:1').merge(costs, on=['asset_id', 'date'], how='left', validate='1:1')
    full.to_parquet(w / 'raw_with_forecasts.parquet', index=False)
    # Write a fresh staged main dataset; preserve the first materialization.
    from src.data_mani.build_volumealpha_inferred import write_dataset
    root = Path('data/crypto/v2.0-volumealpha-causal')
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['auxiliary_stability_rule'] = 'Each model forecast clipped to min/max target in its own exact past training sample; no future/return score used'
    manifest['stability_rows_changed'] = sum(x['rows_changed'] for x in fixes)
    families = manifest['families']
    columns = sum(families.values(), [])
    full = full[['ucid', 'asset_id', 'date'] + columns]
    staged = Path(str(root) + '-staged')
    base = pd.read_parquet(w / 'base_train.parquet')
    write_dataset(base, full, staged, families, manifest)
    for name in ['volume_targets.parquet', 'volume_oof_predictions.parquet', 'volume_oof_metrics.csv', 'volume_fits.json', 'identities_inferred.parquet']:
        import shutil
        shutil.copyfile(w / name, staged / name)
    manifest = json.loads((staged / 'manifest.json').read_text())
    manifest['version'] = 'crypto/v2.0-volumealpha-causal'
    (staged / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    root.rename(root.parent / '.volumealpha_before_stability_guard')
    staged.rename(root)
    (w / 'stability_audit.json').write_text(json.dumps(fixes, indent=2))
    print('Stability adjustment complete. Prediction rows changed:', sum(x['rows_changed'] for x in fixes))


if __name__ == '__main__':
    main()
