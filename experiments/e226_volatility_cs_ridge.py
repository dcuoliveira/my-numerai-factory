"""E226: CS-only / concatenated TS+CS volatility ridge add-on to E212.

No embedding extraction, original forecast mutation or ensemble search.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import json
import shutil
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from experiments import e212_volatility99_tsfm as S
from src.data_mani import build_e209_volatility99 as B
from src.data_mani import volatility_proxies as V
from src.analysis import tsfm_momentum_study as N
from src.analysis.asset_momentum import route_slots
from src.live.submit import rank_normalise
from src.prediction import meta_resid as MR
from experiments.e211_cross_target_tsfm import fold_panel,lagged_snapshots
from numerai_tools.scoring import numerai_corr,correlation_contribution

REPO=S.REPO
ROOT=REPO/'results/reports/crypto/e226_volatility_cs_ridge'
DATA=REPO/'data/crypto/v2.0-e226-ts-cs-volatility198'
BASE=REPO/'data'/S.DATASET
SOURCE=S.ROOT
save,parquet,sha=S.save,S.parquet,S.sha

def raw_asset(dates,bars,benchmark):
    """Exact build_asset raw branch, deliberately no own-history transformation."""
    out=[]
    for _,family,calendar,method,_,side,_ in B.GROUPS.values():
        cutoff=dates-pd.Timedelta(days=1)
        if calendar=='weekday':cutoff=pd.DatetimeIndex([pd.offsets.BDay().rollback(d) for d in cutoff])
        grid=pd.date_range(dates.min()-pd.Timedelta(days=800),cutoff.max(),freq='D' if calendar=='daily' else 'B')
        source=bars.reindex(grid)
        market={k:np.log(s.reindex(grid)).diff() for k,s in benchmark.items()} if family=='idiosyncratic' else {}
        ix=grid.get_indexer(cutoff)
        for w in B.WINDOWS:
            values=np.maximum(np.column_stack(V.estimate(source,market,family,method,w)),0)
            out.append(values[ix,int(side=='up')])
    return np.column_stack(out)

def cs_bins(raw,dates):
    """Available-only per-feature CS midranks; NaNs never affect ranks."""
    out=np.full(raw.shape,np.nan,dtype=np.float32)
    for ix in pd.Series(np.arange(len(dates))).groupby(np.asarray(dates),sort=True).indices.values():
        for j in range(raw.shape[1]):
            rows=ix[np.isfinite(raw[ix,j])]
            if len(rows)>=2:out[rows,j]=V.bins((rankdata(raw[rows,j],method='average')-.5)/len(rows))
    return out

def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists():
        m=json.loads((DATA/'manifest.json').read_text())
        assert m['builder_sha256']==sha(__file__),'Builder changed: use a new output directory'
        for name,h in m['output_hashes'].items():assert sha(DATA/name)==h,name
        return m
    plan=json.loads((B.SOURCE/'registration.json').read_text())
    pricepath=REPO/next(p for p in plan['input_hashes'] if p.endswith('/inputs/prices.parquet'))
    assert sha(pricepath)==plan['input_hashes'][str(pricepath.relative_to(REPO))]
    original_manifest=json.loads((BASE/'manifest.json').read_text())
    for name,h in original_manifest['output_hashes'].items():assert sha(BASE/name)==h,name
    base=pd.read_parquet(BASE/'train.parquet');keys=base.reset_index()
    cov=pd.read_parquet(BASE/'coverage.parquet');pd.testing.assert_index_equal(base.index,cov.index)
    eligible=cov.source_eligible.to_numpy()
    prices=pd.read_parquet(pricepath,filters=[('date','<',pd.Timestamp('2026-02-20'))])
    unique_symbols=prices.groupby('symbol').ucid.nunique().eq(1)
    clean=prices[prices.symbol.map(unique_symbols)]
    ids=clean.groupby('ucid').symbol.nunique().eq(1)
    bars={str(i):V.clean_bars(g.rename(columns={'date':'bar_date'})) for i,g in clean.groupby('ucid') if ids.loc[i]}
    benchmark={i:bars[str(i)].close for i in (1,1027)}
    raw=np.full((len(base),99),np.nan);checks=[]
    cache=ROOT/'raw_assets';cache.mkdir(exist_ok=True)
    identity=dict(price_sha256=sha(pricepath),base_manifest_sha256=sha(BASE/'manifest.json'),builder_sha256=sha(__file__),calculation_sha256=sha(V.__file__))
    if (cache/'identity.json').exists():assert json.loads((cache/'identity.json').read_text())==identity
    else:save(cache/'identity.json',identity)
    groups=list(keys.groupby(keys.ucid.astype(str),sort=True))
    for n,(asset,g) in enumerate(groups,1):
        rows=g.index.to_numpy()
        if asset not in bars or not eligible[rows].any():continue
        dates=pd.DatetimeIndex(g.date);path=cache/f'{asset}.npy'
        if path.exists():v=np.load(path)
        else:
            v=raw_asset(dates,bars[asset],benchmark);np.save(path,v)
        assert v.shape==(len(rows),99)
        raw[rows]=v
        if len(checks)<4 and len(rows)>400:
            reference=[]
            for _,family,calendar,method,_,side,_ in B.GROUPS.values():
                values=V.build_asset(dates,bars[asset],benchmark if family=='idiosyncratic' else {},family,[(calendar,method,w,side) for w in B.WINDOWS])
                reference.extend(values[calendar,method,w][0][:,int(side=='up')] for w in B.WINDOWS)
            np.testing.assert_array_equal(v,np.column_stack(reference))
            cut=dates[len(dates)*3//4];keep=dates<=cut
            replay=raw_asset(dates[keep],bars[asset].loc[lambda f:f.index<cut],{k:s.loc[s.index<cut] for k,s in benchmark.items()})
            np.testing.assert_array_equal(v[keep],replay)
            checks.append(dict(ucid=asset,cutoff=str(cut),raw_columns=99,raw_reference_exact=True,prefix_exact=True))
        if n%25==0:print('RAW',n,'/',len(groups),flush=True)
    raw[~eligible]=np.nan
    cs=cs_bins(raw,keys.date.to_numpy());ts=base[B.feature_names()].to_numpy()
    csnames=[n.replace('_ts_','_cs_') for n in B.feature_names()];names=B.feature_names()+csnames
    result=base.join(pd.DataFrame(cs,index=base.index,columns=csnames))
    pd.testing.assert_frame_equal(result[base.columns],base)
    assert result.index.is_unique and len(names)==198
    masks={'TS99':np.isfinite(ts).all(1),'CS99':np.isfinite(cs).all(1),'TS_CS198':np.isfinite(ts).all(1)&np.isfinite(cs).all(1)}
    assert np.array_equal(masks['TS99'],cov.covered.to_numpy())
    assert not (masks['TS99']&~masks['CS99']).any(),'E212 covered keys must retain valid CS values'
    DATA.mkdir(parents=True,exist_ok=True)
    parquet(DATA/'train.parquet',result)
    metadata=keys[['symbol','date','ucid']].set_index(['symbol','date'])
    for arm,mask in masks.items():metadata[arm]=mask
    metadata['covered']=masks['TS_CS198'];metadata['source_eligible']=eligible
    parquet(DATA/'coverage.parquet',metadata)
    parquet(DATA/'feature_availability.parquet',result[names].notna())
    parquet(DATA/'historical_meta_models.parquet',pd.read_parquet(BASE/'historical_meta_models.parquet'))
    parquet(ROOT/'raw_volatility.parquet',pd.DataFrame(raw,index=base.index,columns=csnames))
    summary=[]
    for arm,mask in list(masks.items())+[(n,np.isfinite(cs[:,j])) for j,n in enumerate(csnames)]:
        count=keys.loc[mask,'ucid'].nunique()
        summary.append(dict(information_set=arm,covered_rows=int(mask.sum()),total_rows=len(keys),covered_row_pct=float(mask.mean()*100),covered_assets=int(count),total_assets=int(keys.ucid.nunique()),covered_asset_pct=100*count/keys.ucid.nunique()))
    pd.DataFrame(summary).to_csv(DATA/'coverage_summary.csv',index=False)
    metadata.groupby(level='date')[list(masks)].mean().to_csv(DATA/'coverage_by_date.csv')
    save(DATA/'features.json',dict(feature_sets={'all':names,'TS99':B.feature_names(),'CS99':csnames,'TS_CS198':names}))
    shutil.copy2(BASE/'source_identity.csv',DATA/'source_identity.csv');shutil.copy2(BASE/'source_provenance.json',DATA/'source_provenance.json')
    pd.DataFrame(checks).to_csv(DATA/'rebuild_checks.csv',index=False)
    # Date truncation reproduces CS values with the same current-date universe.
    cutoff=pd.Timestamp('2025-01-01');keep=keys.date.le(cutoff).to_numpy()
    np.testing.assert_array_equal(cs[keep],cs_bins(raw[keep],keys.date.to_numpy()[keep]))
    # Hand-calculated available-only/tie/NaN test; absent assets never enter ranks.
    tiny=np.array([[1.,np.nan],[1.,1.],[3.,2.],[np.nan,3.]])
    actual=cs_bins(tiny,np.repeat(np.datetime64('2020-01-01'),4))
    np.testing.assert_array_equal(actual,np.array([[.5,np.nan],[.5,.25],[.75,.5],[np.nan,.75]],np.float32))
    pd.testing.assert_frame_equal(pd.read_parquet(DATA/'train.parquet'),result)
    manifest=dict(experiment='E226',dataset=str(DATA.relative_to(REPO/'data')),n_features=198,features=names,feature_sets={'TS99':B.feature_names(),'CS99':csnames,'TS_CS198':names},rows=len(base),assets=int(keys.ucid.nunique()),target=original_manifest['target'],date_min=original_manifest['date_min'],date_max=original_manifest['date_max'],missing_value_policy='coverage_required',processing={'TS99':'Unchanged E209 prior-history five bins','CS99':'Raw source-eligible D-1 volatility, average finite per-feature/date midranks, (rank-.5)/n, cuts .05/.25/.75/.95; minimum2 assets; no TS preprocessing','TS_CS198':'Column concatenation; no sequential TS-then-CS processing','source_cutoff':'D-1 completed daily / weekday bar, no filling'},source_manifest_sha256=sha(BASE/'manifest.json'),price_sha256=sha(pricepath),price_path=str(pricepath),builder_sha256=sha(__file__),source_calculation_hashes={str(p.relative_to(REPO)):sha(p) for p in [Path(V.__file__),Path(B.__file__)]},output_hashes={p.name:sha(p) for p in DATA.iterdir() if p.suffix in ['.parquet','.csv','.json']},conditions=original_manifest['audit']['conditions'])
    save(DATA/'manifest.json',manifest)
    save(DATA/'audit.json',dict(outcome='PASS WITH CONDITIONS',scope='Dataset construction',exact_original_keys_targets_TS99=True,raw_reference_and_prefix_replays=checks,cs_cut_date_replay=True,missing_and_tie_fixture=True,conditions=manifest['conditions']))
    (DATA/'README.md').write_text('# E226 TS+CS volatility198\n\n198 inputs = unchanged TS99 + independent CS99 of the same nine selected formula groups x11 windows. CS is applied to raw volatility, not to TS bins. Feature sets TS99, CS99 and TS_CS198 are in features.json. All base keys, targets and partial NaNs retained; coverage metadata is excluded from predictors. Use covered-only fitting and matched causal controls. This is retrospective research, not a certified historical-vintage or live dataset.\n')
    print('DATASET COMPLETE',len(result),len(names),flush=True)
    return manifest

def fit(arm,horizon):
    destination=ROOT/'fits'/f'{arm}_t{horizon}'
    if (destination/'COMPLETE.json').exists():return
    manifest=json.loads((DATA/'manifest.json').read_text());names=manifest['feature_sets'][arm]
    panel=pd.read_parquet(SOURCE/'panel.parquet');features=pd.read_parquet(DATA/'train.parquet',columns=names).reset_index()
    panel=panel.drop(columns=[n for n in names if n in panel]).merge(features[['symbol','date']+names],on=['symbol','date'],validate='1:1',sort=False)
    np.testing.assert_array_equal(panel.row,np.arange(len(panel)))
    common=np.load(SOURCE/'cache/common_rows.npy');assert np.isfinite(panel.loc[common,names]).all().all()
    # Reuse the exact E212 baseline function and fold generator; only feature names differ.
    old_settings,old_scr=S.settings,S.SCR
    (ROOT/'cache').mkdir(exist_ok=True);np.save(ROOT/'cache/common_rows.npy',common)
    S.settings=lambda:dict(features=names);S.SCR=ROOT/'cache'
    outputs=[]
    try:
        for block,start in enumerate(range(200,panel.step.max()+1,130)):
            part=fold_panel(panel,start,horizon)
            if part is None:continue
            q,protocol=part;directory=ROOT/'blocks'/f'{arm}_t{horizon}'/f'f{block:02d}'
            if (directory/'COMPLETE.json').exists():outputs.append(pd.read_parquet(directory/'predictions.parquet'));continue
            original=json.loads((SOURCE/'blocks'/f'feature_ridge_t{horizon}'/f'f{block:02d}'/'protocol.json').read_text())
            assert json.loads(json.dumps(protocol,default=str))==original
            assert np.isfinite(q.loc[common,names]).all().all()
            save(directory/'protocol.json',protocol)
            values,_=S.feature_baseline(q,horizon,directory)
            forecast=q.loc[q.test,['row','ucid','symbol','date']].copy();forecast['prediction']=values[q.test]
            parquet(directory/'predictions.parquet',forecast)
            save(directory/'COMPLETE.json',dict(predictions=int(forecast.prediction.notna().sum())))
            outputs.append(forecast);print('FIT',arm,horizon,block,int(forecast.prediction.notna().sum()),flush=True)
    finally:S.settings,S.SCR=old_settings,old_scr
    result=pd.concat(outputs,ignore_index=True);assert result.row.is_unique
    original=pd.read_parquet(SOURCE/'fits'/f'feature_ridge_t{horizon}'/'predictions.parquet')
    np.testing.assert_array_equal(result.row,original.row)
    np.testing.assert_array_equal(result.prediction.notna(),original.prediction.notna())
    parquet(destination/'predictions.parquet',result)
    save(destination/'COMPLETE.json',dict(experiment='E226',arm=arm,horizon=horizon,features=len(names),blocks=len(outputs),predictions=int(result.prediction.notna().sum()),prediction_sha256=sha(destination/'predictions.parquet')))

def score():
    p=pd.read_parquet(SOURCE/'panel.parquet');coverage=pd.read_parquet(SOURCE/'coverage.parquet')
    np.testing.assert_array_equal(p.row,coverage.row)
    causal=coverage.causal_common.to_numpy();scored=coverage.evaluation_available.to_numpy()
    dates=pd.read_parquet(SOURCE/'score_dates.parquet')
    names=[f'ridge_{arm}_t{h}' for arm in ('CS99','TS_CS198') for h in (20,60)]
    raw=np.column_stack([pd.read_parquet(ROOT/'fits'/f'{arm}_t{h}'/'predictions.parquet').set_index('row').prediction.reindex(p.row).to_numpy() for arm in ('CS99','TS_CS198') for h in (20,60)])
    assert np.isfinite(raw[causal]).all()
    allnames=names+[n+'_residual' for n in names];candidate=np.full((len(p),8),np.nan);candidate[:,:4]=raw
    meta=pd.read_parquet(BASE/'historical_meta_models.parquet').reset_index();cache={};groups=[];routed=np.full_like(candidate,np.nan);native_checks=[];snapshots=[]
    for date,idx in p.groupby('date',sort=True).indices.items():
        rows=idx[causal[idx]]
        if len(rows):
            lagged,snapshot=lagged_snapshots(meta,pd.Timestamp(date),p.loc[rows,'symbol'],cache)
            value,beta=S.residual_matrix(raw[rows],lagged);candidate[rows,4:]=value
            snapshots.append(dict(date=date,snapshot=snapshot,beta=beta.tolist()))
        br=rank_normalise(coverage.baseline.to_numpy()[idx])
        for j in range(8):
            routed[idx,j]=route_slots(br,np.where(causal[idx],candidate[idx,j],np.nan))
            assert np.array_equal(routed[idx[~causal[idx]],j],br[~causal[idx]])
            np.testing.assert_array_equal(np.sort(routed[idx[causal[idx]],j]),np.sort(br[causal[idx]]))
        pos=np.flatnonzero(scored[rows])
        if len(pos)>=10:groups.append((pd.Timestamp(date),rows,pos))
    assert np.isfinite(routed).all()
    np.testing.assert_array_equal(dates.date.to_numpy(),pd.to_datetime([d for d,_,_ in groups]).to_numpy())
    np.testing.assert_array_equal(dates.assets.to_numpy(),[len(pos) for _,_,pos in groups])
    configs=[]
    for j,name in enumerate(allnames):
        original=name.removesuffix('_residual');h=int(original.rsplit('_t',1)[1]);arm=original[len('ridge_'):].rsplit('_t',1)[0]
        configs.append(dict(id=f'e226_{name}',members=[name],member_indices=[j],size=1,kind='standalone',training_targets=str(h),model='feature_ridge',view='residual' if j>=4 else 'raw',standardization='asset',design='raw signals',information_set=arm,post_residualized=False))
    arrays={h:np.empty((len(dates),8,2)) for h in (20,60)}
    for d,(date,rows,pos) in enumerate(groups):
        for h in arrays:
            y=p.loc[rows[pos],f'target_binned_return_{h}'].to_numpy();m=p.loc[rows[pos],'meta_model'].to_numpy()
            corr,mmc=S.matrix_scores(candidate[rows[pos]],y,m);corr=np.nan_to_num(corr,nan=0.)
            arrays[h][d,:,0]=corr;arrays[h][d,:,1]=mmc
            if d in (0,len(groups)//2,len(groups)-1):
                for j in range(8):
                    z=pd.DataFrame({'prediction':candidate[rows[pos],j]});yy=pd.Series(y,name='target');mm=pd.DataFrame({'meta_model':m})
                    nc=numerai_corr(z,yy).iloc[0];nm=correlation_contribution(z,mm,yy).iloc[0]
                    np.testing.assert_allclose([corr[j],mmc[j]],[nc,nm],atol=3e-8,rtol=1e-6)
                native_checks.append(dict(date=str(date),target=h,columns=8))
    summary=[]
    assets=p.loc[np.concatenate([rows[pos] for _,rows,pos in groups]),'ucid'].nunique()
    for h,a in arrays.items():
        np.save(ROOT/f'daily_scores_t{h}.npy',a)
        for j,config in enumerate(configs):
            summary.append(dict({**config,'members':config['members'][0],'member_indices':str(j)},evaluation_target=h,status='success',assets=assets,assets_mean=float(dates.assets.mean()),assets_min=int(dates.assets.min()),dates=len(dates),**S.payout_metrics(a[:,j,0],a[:,j,1])))
    table=pd.DataFrame(summary).sort_values(['evaluation_target','payout'],ascending=[True,False])
    parquet(ROOT/'summary.parquet',table);table.to_csv(ROOT/'summary.csv',index=False);save(ROOT/'trials.json',configs)
    parquet(ROOT/'standalone_forecasts.parquet',p[['row','ucid','symbol','date']].join(pd.DataFrame(candidate,columns=allnames)))
    parquet(ROOT/'routed_forecasts.parquet',p[['row','date']].join(pd.DataFrame(routed,columns=allnames)))
    save(ROOT/'residual_snapshots.json',snapshots)
    # Compare matched TS controls per date, not independently filtered samples.
    old_configs=json.loads((SOURCE/'trials.json').read_text());old_index={c['members'][0]:j for j,c in enumerate(old_configs) if c['kind']=='standalone'}
    paired=[]
    for h,a in arrays.items():
        old=np.load(SOURCE/'cache/daily_scores'/f't{h}'/'corr_mmc.npy',mmap_mode='r')
        for j,c in enumerate(configs):
            ref=f'feature_ridge_t{c["training_targets"]}'+('_residual' if c['view']=='residual' else '')
            diff=.1*(a[:,j,0]-old[:,old_index[ref],0])+a[:,j,1]-old[:,old_index[ref],1]
            paired.append(dict(id=c['id'],evaluation_target=h,reference=ref,mean_payout_delta=float(diff.mean()),positive_date_share=float((diff>0).mean()),dates=len(diff)))
    pd.DataFrame(paired).to_csv(ROOT/'paired_vs_TS99.csv',index=False)
    audit=dict(outcome='PASS WITH CONDITIONS',experiment='E226',scope='Matched feature-representation research comparison, not marginal native-feature attribution',original_keys_TS_and_targets_unchanged=True,missing_values_retained=True,fit_and_prediction_masks_exactly_E212=True,folds_and_search_same_E212=True,raw_residual_native_score_checks=native_checks,full_universe_routing='All dates: finite full output; uncovered ranks exact; covered rank-slot set exact',evaluated_dates=len(dates),scored_asset_dates=int(dates.assets.sum()),distinct_assets=int(assets),new_fit_count=4,conditions=['CS cross-sections are available-only retrospective source-eligible assets; vendor vintages/identity history uncertified.','No native v2 B_all/B_covered/F_covered trio; these are feature-only matched representation controls, no marginal native-feature or live claim.','Existing selected formulas and E212 ensemble ranking used development history; new streams are standalone comparisons only, original ensembles unchanged.'])
    save(ROOT/'audit.json',audit)
    return table,configs,dates,arrays

def integrate():
    # Preserve the previously delivered report and all original numerical artifacts.
    history=SOURCE/'history'/f'before_E226_{int(time.time())}';history.mkdir(parents=True)
    for name in ('E212_volatility99_tsfm.pdf','E212_volatility99_tsfm.html','report_qa.json','COMPLETE.json','status.json','independent_completion_audit.json'):
        if (SOURCE/name).exists():shutil.copy2(SOURCE/name,history/name)
    frozen={str(p):sha(p) for p in [SOURCE/'all_trials.parquet',SOURCE/'all_trials.csv',SOURCE/'trials.json']+list((SOURCE/'fits').glob('*/predictions.parquet'))}
    supplement=dict(experiment='E226',root=str(ROOT),dataset=str(DATA.relative_to(REPO/'data')),summary_sha256=sha(ROOT/'summary.parquet'),trial_sha256=sha(ROOT/'trials.json'),array_sha256={str(h):sha(ROOT/f'daily_scores_t{h}.npy') for h in (20,60)},note='E226 adds CS-only99 and concatenated TS+CS198 ridge controls; original E212 ensembles are not re-searched.')
    save(SOURCE/'report_supplement.json',supplement)
    table=pd.read_parquet(SOURCE/'all_trials.parquet');configs=json.loads((SOURCE/'trials.json').read_text());dates=pd.read_parquet(SOURCE/'score_dates.parquet')
    arrays={h:np.load(SOURCE/'cache/daily_scores'/f't{h}'/'corr_mmc.npy',mmap_mode='r') for h in (20,60)}
    S.report(table,configs,dates,arrays)
    assert all(sha(p)==h for p,h in frozen.items())
    pdfhash=sha(SOURCE/'E212_volatility99_tsfm.pdf')
    c=json.loads((SOURCE/'COMPLETE.json').read_text());c.update(report_sha256=pdfhash,report_supplement='E226',report_revision_time=time.time());save(SOURCE/'COMPLETE.json',c)
    status=json.loads((SOURCE/'status.json').read_text());status.update(report_sha256=pdfhash,current_report_emailed=False,report_revision='E226 CS-only and TS+CS ridge controls');save(SOURCE/'status.json',status)
    cfg=json.loads((SOURCE/'config.json').read_text());cfg['code_hashes'][str(Path(S.__file__).relative_to(REPO))]=sha(S.__file__);cfg['report_revision']=dict(experiment='E226',original_numerical_results_unchanged=True,prior_report=str(history));save(SOURCE/'config.json',cfg)
    save(ROOT/'COMPLETE.json',dict(experiment='E226',status='success',heads=4,summary_rows=16,report_sha256=pdfhash,report=str(SOURCE/'E212_volatility99_tsfm.pdf'),original_numerical_hashes=frozen,audit='PASS WITH CONDITIONS',time=time.time()))

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    save(ROOT/'status.json',dict(stage='building_dataset',host=socket.gethostname(),time=time.time()))
    build()
    for arm in ('CS99','TS_CS198'):
        for horizon in (20,60):
            save(ROOT/'status.json',dict(stage='fitting',arm=arm,target=horizon,time=time.time()));fit(arm,horizon)
    save(ROOT/'status.json',dict(stage='scoring',time=time.time()));score()
    save(ROOT/'status.json',dict(stage='reporting',time=time.time()));integrate()
    save(ROOT/'status.json',dict(stage='complete',time=time.time()));print('E226 COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','build','score','integrate']);a=ap.parse_args()
    globals()[a.action]()
