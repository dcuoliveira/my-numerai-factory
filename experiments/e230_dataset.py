"""E230 immutable concatenation of three frozen input-feature sources."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import e212_volatility99_tsfm as S
from src.analysis import tsfm_momentum_study as N

ROOT=S.REPO/'results/reports/crypto/e230_momentum_volatility134'
DATA=S.REPO/'data/crypto/v2.0-e230-momentum-volatility134'
MOM=S.REPO/'data/crypto/v2.0-e205-ts-momentum3'
VOL=S.REPO/'data/crypto/v2.0-e209-ts-volatility99'
NATIVE=S.REPO/'data/crypto/v2.0-e156-frozen/train.parquet'
E228=S.REPO/'results/reports/crypto/e228_momentum_volatility_correlation'
TARGET='target_binned_return_20'
ARMS=['asset_ridge','asset_mlp','panel_ridge','panel_mlp']

def source(path,hashes):
    path=Path(path);h=S.sha(path);hashes[str(path)]=h
    value=pd.read_parquet(path);assert S.sha(path)==h
    return value

def keyed(frame):
    q=frame.reset_index() if 'symbol' not in frame else frame.copy()
    q.ucid=q.ucid.astype(str)
    assert q.ucid.str.fullmatch(r'\d+').all()
    assert not q.duplicated(['ucid','date']).any()
    assert not q.duplicated(['symbol','date']).any()
    return q.set_index(['ucid','date'])

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'PREPARED.json').exists():
        m=json.loads((ROOT/'PREPARED.json').read_text());assert m['builder_sha256']==S.sha(__file__)
        for p,h in m['input_hashes'].items():assert S.sha(p)==h,p
        for p,h in m['output_hashes'].items():assert S.sha(p)==h,p
        return m
    hashes={};manifests={}
    for root in (MOM,VOL):
        path=root/'manifest.json';hashes[str(path)]=S.sha(path);m=json.loads(path.read_text());manifests[root.name]=m
        for name,h in m['output_hashes'].items():assert S.sha(root/name)==h,(root,name)
    momentum=keyed(source(MOM/'train.parquet',hashes));volatility=keyed(source(VOL/'train.parquet',hashes));native=keyed(source(NATIVE,hashes))
    assert momentum.index.difference(volatility.index).empty and volatility.index.difference(momentum.index).empty
    momentum=momentum.reindex(volatility.index);native=native.reindex(volatility.index)
    for label,q in [('momentum',momentum),('native',native)]:
        np.testing.assert_array_equal(q.symbol,volatility.symbol,err_msg=label)
        assert TARGET in q
        for name in (TARGET,'target_binned_return_60'):
            if name in q:np.testing.assert_array_equal(q[name],volatility[name],err_msg=label+' '+name)
            else:assert label=='momentum' and name=='target_binned_return_60'
    mf=[c for c in momentum if c.startswith('feature_')];vf=[c for c in volatility if c.startswith('feature_')];nf=['feature_volatility_20d','feature_volatility_60d'];features=mf+vf+nf
    assert [len(mf),len(vf),len(nf)]==[33,99,2] and len(set(features))==134
    assert volatility.index.get_level_values('date').max()<pd.Timestamp('2026-02-20')
    out=volatility.join(momentum[mf]).join(native[nf]);out=out.reset_index().set_index(['symbol','date'])
    for cols,q in [(mf,momentum),(vf,volatility),(nf,native)]:np.testing.assert_array_equal(out[cols],q[cols])
    out[features]=out[features].astype(np.float32)
    cm=keyed(source(MOM/'coverage.parquet',hashes)).reindex(volatility.index)
    cv=keyed(source(VOL/'coverage.parquet',hashes)).reindex(volatility.index)
    source_eligible=cm.covered.fillna(False).to_numpy(bool)&cv.source_eligible.fillna(False).to_numpy(bool)
    finite=np.isfinite(out[features]).all(axis=1).to_numpy();covered=finite&source_eligible
    values=out[features].to_numpy();assert np.isin(values[np.isfinite(values)],[0,.25,.5,.75,1]).all()
    DATA.mkdir(parents=True,exist_ok=True)
    def store(path,frame):
        tmp=path.with_suffix('.partial');frame.to_parquet(tmp);tmp.replace(path)
    store(DATA/'train.parquet',out)
    cov=out[['ucid']].copy();cov['source_eligible']=source_eligible;cov['covered']=covered;cov['finite134']=finite
    for name,cols in [('momentum33',mf),('volatility99',vf),('native2',nf)]:cov[name]=np.isfinite(out[cols]).all(axis=1)
    store(DATA/'coverage.parquet',cov);store(DATA/'feature_availability.parquet',out[features].notna())
    meta=source(VOL/'historical_meta_models.parquet',hashes);shutil.copy2(VOL/'historical_meta_models.parquet',DATA/'historical_meta_models.parquet')
    S.save(DATA/'features.json',dict(feature_sets={'all':features,'momentum33':mf,'volatility99':vf,'native2':nf}))
    summary=[]
    for name in ['momentum33','volatility99','native2','covered']:
        mask=cov[name].to_numpy();n=out.loc[mask,'ucid'].nunique();summary.append(dict(feature_group=name,covered_rows=int(mask.sum()),total_rows=len(out),covered_row_pct=100*mask.mean(),covered_assets=int(n),total_assets=int(out.ucid.nunique()),covered_asset_pct=100*n/out.ucid.nunique()))
    pd.DataFrame(summary).to_csv(DATA/'coverage_summary.csv',index=False);cov.groupby(level='date').covered.agg(['sum','size','mean']).to_csv(DATA/'coverage_by_date.csv')
    pd.testing.assert_frame_equal(pd.read_parquet(DATA/'train.parquet'),out)
    p=out.reset_index().sort_values(['date','ucid'],kind='stable').reset_index(drop=True)
    cc=cov.reset_index()[['symbol','date','source_eligible','covered']];p=p.merge(cc,on=['symbol','date'],how='left',validate='1:1',sort=False)
    mm=meta.reset_index().rename(columns={'prediction':'meta_model'})
    p=p.merge(mm[['symbol','date','meta_model']],on=['symbol','date'],how='left',validate='1:1',sort=False)
    p['row']=np.arange(len(p));p['ucid']=p.ucid.astype(str)
    for name,mask in zip(['train','validation','final_fit','test'],N.split_masks(p.date)):p[name]=mask
    usable=p.covered.to_numpy()&np.isfinite(p[TARGET]).to_numpy()
    counts=p.assign(train_n=usable&p.train,validation_n=usable&p.validation).groupby('ucid')[['train_n','validation_n']].sum()
    assets=counts.index[(counts.train_n>=60)&(counts.validation_n>=20)].tolist();p['asset_eligible']=p.ucid.isin(assets)
    indices={phase:np.flatnonzero(usable&p.asset_eligible.to_numpy()&p[phase].to_numpy()) for phase in ['train','validation','final_fit']}
    indices['query']=np.flatnonzero(p.covered.to_numpy()&p.asset_eligible.to_numpy()&p.test.to_numpy())
    old=source(E228/'common_keys.parquet',hashes);old.ucid=old.ucid.astype(str)
    index=pd.MultiIndex.from_frame(p[['ucid','date']]);desired=pd.MultiIndex.from_frame(old[['ucid','date']]);loc=index.get_indexer(desired);assert (loc>=0).all()
    np.testing.assert_array_equal(p.loc[loc,'symbol'].to_numpy(),old.symbol.to_numpy())
    np.testing.assert_array_equal(p.loc[loc,TARGET].to_numpy(),old[TARGET].to_numpy())
    np.testing.assert_array_equal(p.loc[loc,'meta_model'].to_numpy(),old.meta_model.to_numpy())
    scored=loc[np.isin(loc,indices['query'])];assert len(scored)>0
    indices['score']=scored
    for phase in ['train','validation','final_fit','query']:assert np.isfinite(p.loc[indices[phase],features]).all().all()
    assert p.loc[indices['train'],'date'].max()+pd.Timedelta(days=35)<=pd.Timestamp('2024-01-01')
    assert p.loc[indices['validation'],'date'].max()+pd.Timedelta(days=35)<=pd.Timestamp('2025-01-01')
    assert p.loc[indices['final_fit'],'date'].max()+pd.Timedelta(days=35)<=pd.Timestamp('2025-01-01')
    assert len(set(indices['final_fit'])&set(indices['query']))==0
    S.parquet(ROOT/'panel.parquet',p);np.savez_compressed(ROOT/'indices.npz',**indices)
    counts.to_csv(ROOT/'asset_split_counts.csv');S.save(ROOT/'eligible_assets.json',assets)
    conditions=['Frozen retrospective vendor snapshots and source identity/coverage restrictions inherited from E205/E209; historical availability is not certified.','Source feature selection and E228/E229 comparison history were previously inspected; development comparison, not new holdout evidence.','Covered-sample learner comparison only; no native22 marginal feature claim, full-universe routing or live model.']
    outputs={str(path):S.sha(path) for path in DATA.iterdir() if path.is_file()}
    manifest=dict(experiment='E230',dataset=str(DATA.relative_to(S.REPO/'data')),rows=len(out),assets=int(out.ucid.nunique()),n_features=134,features=features,feature_sets={'all':features,'momentum33':mf,'volatility99':vf,'native2':nf},index=['symbol','date'],feature_dtype='float32',missing_value_policy='coverage_required',processing='Unchanged source bins concatenated by permanent ID/date; no rebinning or filling',input_hashes=hashes,builder_sha256=S.sha(__file__),output_hashes={Path(k).name:h for k,h in outputs.items()},conditions=conditions)
    S.save(DATA/'manifest.json',manifest)
    (DATA/'README.md').write_text('# E230 momentum and volatility134\n\n33 E205 momentum TS bins +99 E209 volatility TS bins +2 native v2.0 volatility bins. Original panel/targets/partial NaNs retained. Complete/source-eligible rows only for modeling; coverage metadata never enters predictors. Research dataset with inherited vendor-vintage restrictions.\n')
    outputs.update({str(ROOT/'panel.parquet'):S.sha(ROOT/'panel.parquet'),str(ROOT/'indices.npz'):S.sha(ROOT/'indices.npz'),str(DATA/'manifest.json'):S.sha(DATA/'manifest.json')})
    prepared=dict(experiment='E230',features=features,assets=assets,rows=len(p),input_hashes=hashes,output_hashes=outputs,builder_sha256=S.sha(__file__),counts={k:len(v) for k,v in indices.items()},e228_requested_rows=len(old),e228_retained_rows=len(scored),e228_excluded_rows=len(old)-len(scored),test_dates=int(p.loc[scored,'date'].nunique()),conditions=conditions)
    S.save(ROOT/'PREPARED.json',prepared);S.save(ROOT/'status.json',dict(stage='prepared',counts=prepared['counts']))
    print(json.dumps({k:v for k,v in prepared.items() if k in ['counts','e228_requested_rows','e228_retained_rows','test_dates']},indent=2),flush=True)
    return prepared
