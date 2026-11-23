"""Package E209's nine volatility formulas x eleven windows as a reusable dataset.

python3 -m src.data_mani.build_e209_volatility99

No new model fitting or return-label selection. This is the frozen E209
historical sample. Per-feature availability is restored from pre-mask source
artifacts; volatility eligibility never depends on momentum availability.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.analysis import asset_momentum as M
from src.data_mani import volatility_proxies as V
from src.data_mani.build_volatility_replacement import SPECS

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'results/reports/crypto/e209_asset_volatility'
OUTPUT=REPO/'data/crypto/v2.0-e209-ts-volatility99'
TARGET='target_binned_return_20'
WINDOWS=tuple(V.WINDOWS)
GROUPS={f'vol_{i+1}_{s[1]}_{s[2]}_{s[3]}_{s[5]}':s for i,s in enumerate(SPECS)}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def feature_names():
    return [f'feature_vol_{family}_{calendar}_{method}_{side}_ts_{w}d'
            for _,family,calendar,method,_,side,_ in GROUPS.values() for w in WINDOWS]


def verify_cache(frame,rows):
    expected=['row']+[str(i) for i in range(108)]
    if list(frame.columns)!=expected:raise ValueError('Unexpected E209 source-cache schema')
    np.testing.assert_array_equal(frame.row.to_numpy(),np.asarray(rows))
    values=frame[[str(i) for i in range(99)]].to_numpy(dtype=np.float32)
    finite=values[np.isfinite(values)]
    if not np.isin(finite,[0,.25,.5,.75,1]).all():raise ValueError('Invalid TS-bin values')
    if np.isinf(values).any():raise ValueError('Infinite candidate values')
    return values


def coverage_tables(keys,values,source_eligible):
    """Metadata only: feature availability and complete group/all-feature masks."""
    names=feature_names()
    availability=pd.DataFrame(np.isfinite(values),columns=names,index=keys.index)
    coverage=keys[['row','symbol','ucid','date']].copy()
    coverage['source_eligible']=np.asarray(source_eligible,dtype=bool)
    for j,name in enumerate(GROUPS):
        coverage[name]=availability.iloc[:,j*11:(j+1)*11].all(axis=1)
    coverage['covered']=availability.all(axis=1)
    return coverage,availability


def replay_values(dates,bars,benchmark):
    result=[]
    for _,family,calendar,method,_,side,_ in GROUPS.values():
        settings=[(calendar,method,w,side) for w in WINDOWS]
        source=V.build_asset(dates,bars,benchmark if family=='idiosyncratic' else {},family,settings)
        col=int(side=='up')
        result.extend(V.bins(source[calendar,method,w][1][:,col]) for w in WINDOWS)
    return np.column_stack(result)


def source_metadata(prices,prices_path):
    """Record observed mapping/date ranges without inventing retrieval vintages."""
    identity=prices.groupby(['ucid','symbol','source','provider_ticker'],dropna=False).agg(
        first_bar=('date','min'),last_bar=('date','max'),bars=('date','size')).reset_index()
    registration=Path(prices_path).parent.parent/'registration.json'
    snapshot=json.loads(registration.read_text()) if registration.exists() else None
    metadata=dict(provider_identity_columns=['ucid','symbol','source','provider_ticker'],
        source_bar_timestamp='date; completed daily candle',retrieved_at=None,provider_dataset_version=None,
        retrieval_note='No original provider retrieval timestamp/version is present in this snapshot; '
                       'registration time below is not a historical-availability timestamp.',
        snapshot_registration=snapshot,
        snapshot_registration_sha256=sha(registration) if registration.exists() else None)
    return identity,metadata


def store_parquet(path,frame):
    tmp=path.with_suffix('.partial');frame.to_parquet(tmp);tmp.replace(path)


def build(source=SOURCE,output=OUTPUT):
    source,output=Path(source),Path(output)
    if (output/'manifest.json').exists():raise FileExistsError(f'Dataset already exists: {output}')
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((source/'registration.json').read_text())
    prepared=json.loads((source/'PREPARED.json').read_text())
    assert plan['selected_specs']==[list(s) for s in SPECS] and plan['windows']==list(WINDOWS)
    assert sha(source/'keys.parquet')==prepared['keys_sha256']
    # Verify the reusable calculation code and the original price snapshot.
    for relative in ['src/analysis/asset_momentum.py','src/data_mani/volatility_proxies.py',
                     'src/data_mani/build_volatility_replacement.py']:
        assert sha(REPO/relative)==plan['code_hashes'][relative],relative
    price_relative=next(s for s in plan['input_hashes'] if s.endswith('/inputs/prices.parquet'))
    prices_path=REPO/price_relative
    assert sha(prices_path)==plan['input_hashes'][price_relative]
    keys=pd.read_parquet(source/'keys.parquet')
    assert len(keys)==prepared['rows'] and keys.index.equals(pd.RangeIndex(len(keys)))
    assert keys.row.eq(np.arange(len(keys))).all() and not keys.duplicated(['ucid','date']).any()
    assert keys.date.max()<pd.Timestamp('2026-02-20')
    prices=pd.read_parquet(prices_path,filters=[('date','<',pd.Timestamp('2026-02-20'))])
    _,_,cohort=M.price_context(keys,prices)
    eligible=cohort & ~keys.ucid.astype(str).isin(['1','1027']).to_numpy()
    # Require one source symbol for each permanent asset ID as in E209.
    unique_symbols=prices.groupby('symbol').ucid.nunique().eq(1)
    clean=prices[prices.symbol.map(unique_symbols)]
    unique_ids=clean.groupby('ucid').symbol.nunique().eq(1)
    bars={str(i):V.clean_bars(g.rename(columns={'date':'bar_date'}))
          for i,g in clean.groupby('ucid') if unique_ids.loc[i]}
    benchmark={i:bars[str(i)].close for i in (1,1027)}
    eligible &= keys.ucid.astype(str).isin(bars).to_numpy()
    values=np.full((len(keys),99),np.nan,dtype=np.float32)
    hashes={str(source/'keys.parquet'):sha(source/'keys.parquet'),str(prices_path):sha(prices_path),
            str(source/'registration.json'):sha(source/'registration.json'),
            str(source/'PREPARED.json'):sha(source/'PREPARED.json')}
    checked=[];prefix_records=[]
    upstream_prefix=pd.read_csv(source/'source_prefix_audits.csv')
    assert len(upstream_prefix)==prepared['prefix_checks'] and upstream_prefix.changed.eq(0).all()
    hashes[str(source/'source_prefix_audits.csv')]=sha(source/'source_prefix_audits.csv')
    # Cache values precede E209's joint momentum/volatility coverage restriction.
    for i,path in enumerate(sorted((source/'asset_inputs').glob('*.parquet')),1):
        ucid=path.stem;rows=keys.index[keys.ucid.astype(str).eq(ucid)]
        if not len(rows):raise ValueError(f'Unknown source-cache permanent ID {ucid}')
        frame=pd.read_parquet(path);v=verify_cache(frame,rows)
        values[rows]=v;hashes[str(path)]=sha(path)
        if len(checked)<12 and eligible[rows].any() and len(rows)>=400:
            dates=pd.DatetimeIndex(keys.loc[rows,'date'])
            full=replay_values(dates,bars[ucid],benchmark)
            np.testing.assert_array_equal(v,full.astype(np.float32))
            cut=dates[len(dates)*3//4];keep=dates<=cut
            replay=replay_values(dates[keep],bars[ucid].loc[bars[ucid].index<cut],
                                 {k:s.loc[s.index<cut] for k,s in benchmark.items()})
            np.testing.assert_array_equal(v[keep],replay.astype(np.float32))
            checked.append(ucid);prefix_records.append(dict(ucid=ucid,cutoff=str(cut.date()),features=99,changed=0))
        if i%100==0:print('SOURCE CACHES',i,flush=True)
    values[~eligible]=np.nan
    # Every value used by E209 must be reproduced exactly after the new mask.
    for j,name in enumerate(GROUPS):
        path=source/'datasets'/f'{name}.parquet'
        assert sha(path)==prepared['dataset_hashes'][name]
        old=pd.read_parquet(path);hashes[str(path)]=sha(path)
        pd.testing.assert_frame_equal(keys[['row','symbol','ucid','date',TARGET,'covered']],
                                      old[['row','symbol','ucid','date',TARGET,'covered']])
        np.testing.assert_array_equal(values[keys.covered,j*11:(j+1)*11],
                  old.loc[keys.covered,[f'feature_ts_{i}' for i in range(11)]].to_numpy())
    coverage,availability=coverage_tables(keys,values,eligible)
    names=feature_names()
    train=pd.concat([keys[['symbol','date','ucid',TARGET]],pd.DataFrame(values,columns=names)],axis=1).set_index(['symbol','date'])
    coverage=coverage.set_index(['symbol','date'])
    availability.index=train.index
    meta=keys[['symbol','date','meta_model']].rename(columns={'meta_model':'prediction'}).set_index(['symbol','date'])
    assert train.index.is_unique and len(names)==99 and len(set(names))==99
    pd.testing.assert_index_equal(train.index,coverage.index)
    pd.testing.assert_index_equal(train.index,meta.index)
    assert train.loc[coverage.covered,names].notna().all().all()
    assert not any(c in train for c in ['row','covered','source_eligible'])
    for name,frame in [('train.parquet',train),('coverage.parquet',coverage),
                       ('feature_availability.parquet',availability),('historical_meta_models.parquet',meta)]:
        store_parquet(output/name,frame)
    # Read back actual serialized output, including keys, NaNs and target dtypes.
    restored=pd.read_parquet(output/'train.parquet')
    pd.testing.assert_frame_equal(restored,train)
    pd.testing.assert_series_equal(restored[TARGET],keys.set_index(['symbol','date'])[TARGET])
    coverage.groupby(level='date').covered.agg(panel_rows='size',covered_rows='sum',covered_share='mean').to_csv(output/'coverage_by_date.csv')
    total_assets=keys.ucid.nunique();summaries=[]
    for name,mask in [('ALL99',coverage.covered.to_numpy())]+[(n,coverage[n].to_numpy()) for n in GROUPS]+[(n,availability[n].to_numpy()) for n in names]:
        assets=keys.loc[mask,'ucid'].nunique()
        summaries.append(dict(information_set=name,covered_rows=int(mask.sum()),total_rows=len(keys),
            covered_row_pct=float(mask.mean()*100),covered_assets=int(assets),total_assets=int(total_assets),
            covered_asset_pct=float(assets/total_assets*100)))
    pd.DataFrame(summaries).to_csv(output/'coverage_summary.csv',index=False)
    pd.DataFrame(prefix_records).to_csv(output/'rebuild_checks.csv',index=False)
    identity,provenance=source_metadata(prices,prices_path)
    identity.to_csv(output/'source_identity.csv',index=False)
    (output/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    audit=dict(outcome='PASS WITH CONDITIONS',scope='Dataset construction/storage only; no training or performance claim.',
        evidence=['All original E209 panel keys/targets retained with exact serialized roundtrip.',
                  '99 uniquely named float32 prior-history-binned features; finite values exactly in {0,.25,.5,.75,1}.',
                  'Input price/calculation hashes verified;12 full asset reconstructions and12 cutoff-prefix rebuilds match all99 columns.',
                  'All E209 covered input values reproduced exactly; per-feature missingness restored before joint masking.',
                  'Permanent IDs checked against source symbols; ambiguous IDs excluded; BTC/ETH excluded from candidate eligibility.',
                  'Coverage metadata excluded from train inputs; incomplete rows kept with their actual per-feature NaNs.'],
        conditions=['Retrospective vendor snapshots/identity filters cannot establish historical source availability.',
                    'Development sample only; final target holdout remains closed. No live feed/replay, fitting or rank-routing is implemented by this dataset builder.',
                    'Future training must use covered-only candidate fits and matched controls/full-universe routing; generic neutral-filling training is disallowed.'])
    groups={name:names[i*11:(i+1)*11] for i,name in enumerate(GROUPS)}
    manifest=dict(dataset='crypto/v2.0-e209-ts-volatility99',n_features=99,features=names,feature_groups=groups,
        specifications=[dict(group=name,family=s[1],calendar=s[2],method=s[3],side=s[5]) for name,s in GROUPS.items()],
        windows=WINDOWS,rows=len(train),assets=int(total_assets),covered_rows=int(coverage.covered.sum()),
        covered_assets=int(keys.loc[coverage.covered.to_numpy(),'ucid'].nunique()),
        date_min=str(keys.date.min().date()),date_max=str(keys.date.max().date()),target=TARGET,
        index=['symbol','date'],feature_dtype='float32',missing_value_policy='coverage_required',
        processing=dict(actual_window='nominal window',history=252,min_history=126,current_excluded=True,
            bin_cuts=[.05,.25,.75,.95],source_cutoff='D-1 completed bar',weekday_rollback_max_age_days=3,
            fill_missing_bars=False,cross_sectional_processing=False,asset_one_hot=False),
        coverage='Per-feature finite values with E209 independent identity/OHLC gate; complete99 mask independent of momentum. '
                 'No all-features wipeout on partially covered rows. Masks are routing metadata, never features.',
        source=dict(experiment='E209',pre_mask_caches=True,original_code_hashes=plan['code_hashes'],
                    original_input_hashes=plan['input_hashes'],source_manifest=provenance,
                    upstream_prefix_checks=len(upstream_prefix),additional_asset_prefix_checks=len(checked)),
        input_hashes=hashes,builder_sha256=sha(__file__),
        output_hashes={p.name:sha(p) for p in output.iterdir() if p.suffix in ['.parquet','.csv']},audit=audit)
    (output/'features.json').write_text(json.dumps(dict(feature_sets=dict(all=names,**groups)),indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text(f'''# E209 volatility99\n\n99 inputs = nine frozen volatility formula/calendar/side configurations x11 windows.\n\n- Index: symbol/date; permanent identity: ucid; target: {TARGET}.\n- Historical sample: {manifest['date_min']} to {manifest['date_max']}; {len(train):,} rows.\n- Complete99 coverage: {manifest['covered_rows']:,} rows and {manifest['covered_assets']} assets.\n- Features: prior-history five bins, no cross-sectional processing or asset dummies.\n- Missing values remain NaN; zero is a genuine observed bin. Partial feature availability is retained.\n- coverage.parquet and feature_availability.parquet contain eligibility metadata only.\n- historical_meta_models.parquet is the exactly aligned reference series, never a feature.\n- coverage_summary.csv reports per-feature, per-group and complete99 coverage.\n\nUse a coverage-aware consumer: fit candidate models only on required-feature-covered rows, match controls and preserve the full-universe baseline rank slots. Do not neutral-fill unavailable candidate blocks into learning. The generic neutral-filling training loader rejects this manifest. This is a development dataset, with no live-parity or historical-vintage claim.\n\nRebuild: python3 -m src.data_mani.build_e209_volatility99 --output <new-directory>\n\nNo new models were fitted. See manifest.json, audit.json and rebuild_checks.csv for provenance and verification.\n''')
    print(json.dumps(dict(path=str(output),features=99,rows=len(train),covered_rows=manifest['covered_rows'],
        assets=manifest['assets'],covered_assets=manifest['covered_assets'],audit=audit['outcome']),indent=2),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();build(args.source,args.output)
