"""Package E219's 17 reversal formulas x 11 windows as one reusable dataset.

python3 -m src.data_mani.build_e219_reversal187 [--output <new-directory>]

No model fitting or label-based selection. Values are E219's pre-mask TS-binned
inputs (strictly prior own-history five bins, D-1 attach, at most four days
stale), rebuilt from the frozen price snapshot and checked bit for bit. Missing
values follow the repo standard (numerai-crypto-missing-values): unavailable
values stay NaN per feature, no neutral fill, no coverage flags among the
features, every panel key kept, and coverage masks shipped as routing metadata
only. Reversal eligibility does not depend on momentum availability (E219's
common mask additionally required momentum132; this dataset does not).
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.analysis import asset_momentum as M
from src.data_mani import reversal_metrics as R

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'results/reports/crypto/e219_asset_reversal'
OUTPUT = REPO/'data/crypto/v2.0-e219-ts-reversal187'
TARGET = 'target_binned_return_20'
CUTOFF = pd.Timestamp('2026-02-20')
FACTOR_IDS = ('1', '1027')
FAMILIES = list(R.FAMILIES)
REBUILD_CUTS = ('2022-12-30', '2024-12-31')
SWING = ('swing_high_distance', 'swing_low_distance')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def feature_names():
    """Window suffix is the lookback in weekday bars; swing pivots carry their half-width k."""
    return [f'feature_rev_{family}_ts_k{R.SWING_K[s]}' if family in SWING else f'feature_rev_{family}_ts_{w}d'
            for family in FAMILIES for s, w in enumerate(R.WINDOWS)]


def groups():
    names = feature_names()
    return {family: names[i*11:(i+1)*11] for i, family in enumerate(FAMILIES)}


def factors(bars):
    return {'btc': bars['close']['BTC'], 'eth': bars['close']['ETH']}


def compute(keys, bars, cols):
    """All 187 TS-binned inputs on the panel keys, before any coverage mask."""
    out = np.full((len(keys), 187), np.nan, dtype=np.float32)
    f = factors(bars)
    for i, family in enumerate(FAMILIES):
        for slot in range(11):
            raw = R.signal(bars, family, slot, f)
            ts = pd.DataFrame(M.prior_bins(raw.to_numpy()), index=raw.index, columns=raw.columns)
            out[:, i*11+slot] = M.attach(keys, ts, cols)
    return out


def coverage_tables(keys, values, eligible):
    """Metadata only: per-feature availability, complete-family masks and complete187."""
    names = feature_names()
    availability = pd.DataFrame(np.isfinite(values), columns=names, index=keys.index)
    coverage = keys[['row', 'symbol', 'ucid', 'date']].copy()
    coverage['source_eligible'] = np.asarray(eligible, dtype=bool)
    for family, cols in groups().items():
        coverage[family] = availability[cols].all(axis=1)
    coverage['covered'] = availability.all(axis=1)
    return coverage, availability


def store_parquet(path, frame):
    tmp = path.with_suffix('.partial')
    frame.to_parquet(tmp)
    tmp.replace(path)


def build(source=SOURCE, output=OUTPUT):
    source, output = Path(source), Path(output)
    if (output/'manifest.json').exists():
        raise FileExistsError(f'Dataset already exists: {output}')
    plan = json.loads((source/'registration.json').read_text())
    prepared = json.loads((source/'PREPARED.json').read_text())
    assert plan['windows'] == list(R.WINDOWS) and plan['swing_k'] == list(R.SWING_K)
    assert list(plan['families']) == FAMILIES
    for relative in ['src/analysis/asset_momentum.py', 'src/data_mani/reversal_metrics.py']:
        assert sha(REPO/relative) == plan['code_hashes'][relative], relative
    assert sha(source/'keys.parquet') == prepared['keys_sha256']
    price_relative = next(s for s in plan['input_hashes'] if s.endswith('prices.parquet'))
    prices_path = REPO/price_relative
    assert sha(prices_path) == plan['input_hashes'][price_relative]
    output.mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(source/'keys.parquet')
    assert len(keys) == prepared['rows'] and keys.row.eq(np.arange(len(keys))).all()
    assert not keys.duplicated(['ucid', 'date']).any() and keys.date.max() < CUTOFF
    prices = pd.read_parquet(prices_path, filters=[('date', '<', CUTOFF)])
    close, cols, cohort = M.price_context(keys, prices)
    bars = R.wide_bars(prices)
    if not bars['close'].equals(close):
        raise RuntimeError('Wide bars differ from the E163 price context')
    eligible = cohort & ~keys.ucid.astype(str).isin(FACTOR_IDS).to_numpy()

    # 1) Full rebuild from prices must equal E219's cached pre-mask inputs exactly.
    values = compute(keys, bars, cols)
    hashes = {str(p): sha(p) for p in [source/'keys.parquet', prices_path, source/'registration.json',
                                       source/'PREPARED.json']}
    for i, family in enumerate(FAMILIES):
        path = source/'inputs'/f'{family}_ts.parquet'
        cached = pd.read_parquet(path).to_numpy()
        np.testing.assert_array_equal(values[:, i*11:(i+1)*11], cached)
        hashes[str(path)] = sha(path)
        print('REBUILT', family, flush=True)
    finite = values[np.isfinite(values)]
    if not np.isin(finite, [0, .25, .5, .75, 1]).all():
        raise ValueError('Invalid TS-bin values')
    values[~eligible] = np.nan

    # 2) Every value E219 trained on is reproduced after the dataset's own mask.
    for i, family in enumerate(FAMILIES):
        path = source/'datasets'/f'{family}.parquet'
        assert sha(path) == prepared['dataset_hashes'][family]
        old = pd.read_parquet(path)
        hashes[str(path)] = sha(path)
        pd.testing.assert_frame_equal(keys[['row', 'symbol', 'ucid', 'date', TARGET, 'covered']],
                                      old[['row', 'symbol', 'ucid', 'date', TARGET, 'covered']])
        np.testing.assert_array_equal(values[keys.covered, i*11:(i+1)*11],
                                      old.loc[keys.covered, [f'feature_ts_{j}' for j in range(11)]].to_numpy())

    # 3) Cut-date rebuild: prices truncated at the cut reproduce every value dated <= cut.
    checks = []
    for cut in REBUILD_CUTS:
        cut = pd.Timestamp(cut)
        p = prices[prices.date <= cut]
        sub = keys[keys.date <= cut + pd.Timedelta(days=1)]
        sbars = R.wide_bars(p)
        scols = sbars['close'].columns.get_indexer(sub.symbol)
        replay = compute(sub, sbars, scols)
        replay[~eligible[sub.index]] = np.nan
        diff = ~((replay == values[sub.index]) | (np.isnan(replay) & np.isnan(values[sub.index])))
        checks.append(dict(cutoff=str(cut.date()), rows=len(sub), features=187, cells=int(diff.size),
                           changed=int(diff.sum())))
        if diff.any():
            raise AssertionError(f'Cut-date rebuild differs at {cut.date()}: {int(diff.sum())} cells')
        print('CUT-DATE', cut.date(), 'cells', diff.size, 'changed 0', flush=True)

    coverage, availability = coverage_tables(keys, values, eligible)
    names = feature_names()
    train = pd.concat([keys[['symbol', 'date', 'ucid', TARGET]], pd.DataFrame(values, columns=names)],
                      axis=1).set_index(['symbol', 'date'])
    coverage = coverage.set_index(['symbol', 'date'])
    availability.index = train.index
    meta = keys[['symbol', 'date', 'meta_model']].rename(columns={'meta_model': 'prediction'}).set_index(['symbol', 'date'])
    assert train.index.is_unique and len(names) == 187 == len(set(names))
    pd.testing.assert_index_equal(train.index, coverage.index)
    pd.testing.assert_index_equal(train.index, meta.index)
    assert train.loc[coverage.covered, names].notna().all().all()
    assert not train.loc[~coverage.source_eligible, names].notna().any().any()
    assert not any(c in train for c in ['row', 'covered', 'source_eligible'])
    for name, frame in [('train.parquet', train), ('coverage.parquet', coverage),
                        ('feature_availability.parquet', availability), ('historical_meta_models.parquet', meta)]:
        store_parquet(output/name, frame)
    restored = pd.read_parquet(output/'train.parquet')
    pd.testing.assert_frame_equal(restored, train)

    coverage.groupby(level='date').covered.agg(panel_rows='size', covered_rows='sum',
                                               covered_share='mean').to_csv(output/'coverage_by_date.csv')
    total_assets = keys.ucid.nunique()
    summaries = []
    for name, mask in ([('ALL187', coverage.covered.to_numpy())] + [(n, coverage[n].to_numpy()) for n in FAMILIES]
                       + [(n, availability[n].to_numpy()) for n in names]):
        assets = keys.loc[mask, 'ucid'].nunique()
        summaries.append(dict(information_set=name, covered_rows=int(mask.sum()), total_rows=len(keys),
                              covered_row_pct=float(mask.mean()*100), covered_assets=int(assets),
                              total_assets=int(total_assets), covered_asset_pct=float(assets/total_assets*100)))
    pd.DataFrame(summaries).to_csv(output/'coverage_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(output/'rebuild_checks.csv', index=False)
    identity = prices.groupby(['ucid', 'symbol', 'source', 'provider_ticker'], dropna=False).agg(
        first_bar=('date', 'min'), last_bar=('date', 'max'), bars=('date', 'size')).reset_index()
    identity.to_csv(output/'source_identity.csv', index=False)
    covered_rows = int(coverage.covered.sum())
    covered_assets = int(keys.loc[coverage.covered.to_numpy(), 'ucid'].nunique())
    audit = dict(outcome='PASS WITH CONDITIONS', scope='Dataset construction/storage only; no training or performance claim.',
        evidence=['All E219 panel keys and targets retained; exact serialized roundtrip.',
                  '187 uniquely named float32 prior-history-binned features; finite values exactly in {0,.25,.5,.75,1}.',
                  'Full rebuild of all 187 columns from the hashed price snapshot equals E219 pre-mask inputs bit for bit.',
                  f'Cut-date rebuilds at {", ".join(REBUILD_CUTS)} change 0 cells across all 187 columns.',
                  'Every value E219 trained on is reproduced exactly; per-feature missingness kept, no neutral fill.',
                  'E163 identity/OHLC gate on permanent IDs; BTC/ETH (residual factors) excluded from eligibility.',
                  'Coverage metadata kept out of train inputs; reversal eligibility independent of momentum.'],
        conditions=['Retrospective vendor snapshot and identity filters cannot establish historical source availability.',
                    'Development sample only (targets before 2026-02-20); no live feed, replay, fitting or routing here.',
                    'Train candidates on covered rows only with matched B_covered controls and full-universe rank-slot routing; '
                    'generic neutral-fill training is disallowed (loader rejects coverage_required).'])
    manifest = dict(dataset='crypto/v2.0-e219-ts-reversal187', n_features=187, features=names, feature_groups=groups(),
        families=R.FAMILIES, windows=list(R.WINDOWS), swing_k=list(R.SWING_K),
        lookbacks=dict(beta=R.BETA_WINDOW, volume_base=R.VOLUME_BASE, trend=R.TREND_WINDOW, swing_search=R.SWING_SEARCH),
        rows=len(train), assets=int(total_assets), covered_rows=covered_rows, covered_assets=covered_assets,
        date_min=str(keys.date.min().date()), date_max=str(keys.date.max().date()), target=TARGET,
        index=['symbol', 'date'], feature_dtype='float32', missing_value_policy='coverage_required',
        processing=dict(bars='weekday, E163 bar-validity rule', history=252, min_history=126, current_excluded=True,
                        bin_cuts=[.05, .25, .75, .95], source_cutoff='D-1 completed bar', max_staleness_days=4,
                        fill_missing_bars=False, cross_sectional_processing=False, asset_one_hot=False,
                        shared_inputs='BTC and ETH factor returns for the two residual families only'),
        coverage='Per-feature finite values under the E163 identity/OHLC gate; per-family and complete187 masks in '
                 'coverage.parquet are routing metadata, never features.',
        source=dict(experiment='E219', original_code_hashes=plan['code_hashes'],
                    original_input_hashes=plan['input_hashes'], cut_date_checks=checks),
        input_hashes=hashes, builder_sha256=sha(__file__),
        output_hashes={p.name: sha(p) for p in output.iterdir() if p.suffix in ['.parquet', '.csv']}, audit=audit)
    (output/'features.json').write_text(json.dumps(dict(feature_sets=dict(all=names, **groups())), indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'README.md').write_text(f'''# E219 reversal187

187 inputs = 17 reversal formulas x 11 windows (2-120 weekday bars; swing pivots use half-widths k 1-40).
Definitions: src/data_mani/reversal_metrics.py.

- Index: symbol/date; permanent identity: ucid; target: {TARGET}.
- Historical sample: {manifest['date_min']} to {manifest['date_max']}; {len(train):,} rows.
- Complete187 coverage: {covered_rows:,} rows and {covered_assets} assets.
- Features: strictly prior own-history five bins, D-1 attach; no cross-sectional processing or asset dummies.
- Missing values stay NaN per feature; 0 is a genuine observed bin. Partial rows keep their observed values.
- coverage.parquet (per-family and complete187 masks) and feature_availability.parquet are metadata only.
- historical_meta_models.parquet is the aligned reference series, never a feature.

Use a coverage-aware consumer: fit candidates only on rows covering the features they need, match
B_covered controls on the same rows, and route into the full-universe baseline rank slots. The generic
neutral-fill loader rejects this manifest. Development dataset; no live-parity or historical-vintage claim.

Rebuild: python3 -m src.data_mani.build_e219_reversal187 --output <new-directory>
''')
    print(json.dumps(dict(path=str(output), features=187, rows=len(train), covered_rows=covered_rows,
                          assets=int(total_assets), covered_assets=covered_assets, audit=audit['outcome']), indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)
