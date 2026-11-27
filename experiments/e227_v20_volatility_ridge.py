"""E227: original v2.0 two-feature volatility ridge controls, matched to E212."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import e212_volatility99_tsfm as S
from src.analysis import tsfm_momentum_study as N
from experiments.e211_cross_target_tsfm import fold_panel,lagged_snapshots
from src.prediction import meta_resid as MR
from src.live.submit import rank_normalise
from src.analysis.asset_momentum import route_slots
from numerai_tools.scoring import numerai_corr,correlation_contribution

REPO=S.REPO
SOURCE=S.ROOT
ROOT=REPO/'results/reports/crypto/e227_v20_volatility_ridge'
NATIVE=REPO/'data/crypto/v2.0-e156-frozen/train.parquet'
FEATURES=['feature_volatility_20d','feature_volatility_60d']
save,parquet,sha=S.save,S.parquet,S.sha

def prepare():
    if (ROOT/'PREPARED.json').exists():
        c=json.loads((ROOT/'PREPARED.json').read_text())
        assert c['runner_sha256']==sha(__file__)
        assert c['panel_sha256']==sha(ROOT/'panel.parquet')
        assert c['native_sha256']==sha(NATIVE)
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    p=pd.read_parquet(SOURCE/'panel.parquet')
    names=['ucid','date','target_binned_return_20','target_binned_return_60']+FEATURES
    v=pd.read_parquet(NATIVE,columns=names,filters=[('date','<',pd.Timestamp('2026-02-20'))]).reset_index()
    assert not v.duplicated(['ucid','date']).any()
    q=p.merge(v,on=['ucid','date'],how='left',validate='1:1',suffixes=('','_native'),sort=False)
    np.testing.assert_array_equal(q.row,p.row)
    np.testing.assert_array_equal(q.symbol,q.symbol_native)
    for h in (20,60):np.testing.assert_array_equal(q[f'target_binned_return_{h}'],q[f'target_binned_return_{h}_native'])
    pd.testing.assert_frame_equal(q[p.columns],p)
    q=q.drop(columns=['symbol_native','target_binned_return_20_native','target_binned_return_60_native'])
    common=np.load(SOURCE/'cache/common_rows.npy');assert np.isfinite(q.loc[common,FEATURES]).all().all()
    assert np.isin(q[FEATURES].to_numpy()[np.isfinite(q[FEATURES].to_numpy())],[0,.25,.5,.75,1]).all()
    parquet(ROOT/'panel.parquet',q)
    save(ROOT/'PREPARED.json',dict(experiment='E227',native_source=str(NATIVE),native_sha256=sha(NATIVE),runner_sha256=sha(__file__),features=FEATURES,feature_processing='As supplied; no further binning or filling',panel_sha256=sha(ROOT/'panel.parquet'),common_sha256=sha(SOURCE/'cache/common_rows.npy'),rows=len(q),alphas=N.ALPHAS.tolist(),solver='Primal ridge with unpenalized intercept; objective identical to E212 dual ridge'))

def primal(x,v,y,alphas):
    y=np.asarray(y,float)
    shift=x.mean(0);x=x-shift;v=v-shift;intercept=float(y.mean())
    eig,vec=np.linalg.eigh(x.T@x)
    weights=vec@((vec.T@(x.T@(y-intercept)))[:,None]/(np.maximum(eig,0)[:,None]+np.asarray(alphas)[None,:]))
    return v@weights+intercept,dict(weights=weights,shift=shift,intercept=intercept)

def fit(h):
    destination=ROOT/'fits'/f'v20_vol_t{h}'
    if (destination/'COMPLETE.json').exists():return
    p=pd.read_parquet(ROOT/'panel.parquet');common=np.load(SOURCE/'cache/common_rows.npy');target=f'target_binned_return_{h}';outputs=[]
    for block,start in enumerate(range(200,p.step.max()+1,130)):
        part=fold_panel(p,start,h)
        if part is None:continue
        q,protocol=part;directory=ROOT/'blocks'/f'v20_vol_t{h}'/f'f{block:02d}'
        if (directory/'COMPLETE.json').exists():outputs.append(pd.read_parquet(directory/'predictions.parquet'));continue
        reference=json.loads((SOURCE/'blocks'/f'feature_ridge_t{h}'/f'f{block:02d}'/'protocol.json').read_text())
        assert json.loads(json.dumps(protocol,default=str))==reference
        save(directory/'protocol.json',protocol)
        pred=np.full(len(q),np.nan);records=[]
        for asset,g in q.groupby('ucid'):
            masks={k:g.index[common[g.index]&q[k].to_numpy()[g.index]&np.isfinite(q[target].to_numpy()[g.index])].to_numpy() for k in ['train','validation','final_fit']}
            tr,va,ft=masks['train'],masks['validation'],masks['final_fit'];te=g.index[common[g.index]&q.test.to_numpy()[g.index]].to_numpy()
            if len(tr)<60 or len(va)<20 or not len(te):continue
            x=q.loc[tr,FEATURES].to_numpy(float);v=q.loc[va,FEATURES].to_numpy(float);center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1
            trial,_=primal((x-center)/scale,(v-center)/scale,q.loc[tr,target].to_numpy(),N.ALPHAS)
            loss=((trial-q.loc[va,target].to_numpy()[:,None])**2).mean(0);alpha=float(N.ALPHAS[loss.argmin()])
            x=q.loc[ft,FEATURES].to_numpy(float);v=q.loc[te,FEATURES].to_numpy(float);center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1
            values,parameters=primal((x-center)/scale,(v-center)/scale,q.loc[ft,target].to_numpy(),[alpha]);pred[te]=values[:,0]
            dest=directory/'ridge';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'asset_{asset}.npz',training_rows=ft,selection_training_rows=tr,validation_rows=va,query_rows=te,alpha=alpha,center=center,scale=scale,validation_mse=float(loss.min()),**parameters)
            records.append(dict(asset=str(asset),alpha=alpha,validation_mse=float(loss.min()),training_rows=len(ft)))
        old=json.loads((SOURCE/'cache/blocks'/f'feature_ridge_t{h}'/f'f{block:02d}'/'parameters.json').read_text())
        assert {r['asset']:r['training_rows'] for r in records}=={r['asset']:r['training_rows'] for r in old}
        save(directory/'parameters.json',records)
        forecast=q.loc[q.test,['row','ucid','symbol','date']].copy();forecast['prediction']=pred[q.test]
        parquet(directory/'predictions.parquet',forecast);save(directory/'COMPLETE.json',dict(predictions=int(forecast.prediction.notna().sum())))
        outputs.append(forecast);print('FIT',h,block,int(forecast.prediction.notna().sum()),flush=True)
    result=pd.concat(outputs,ignore_index=True);reference=pd.read_parquet(SOURCE/'fits'/f'feature_ridge_t{h}'/'predictions.parquet')
    np.testing.assert_array_equal(result.row,reference.row);np.testing.assert_array_equal(result.prediction.notna(),reference.prediction.notna())
    parquet(destination/'predictions.parquet',result)
    save(destination/'COMPLETE.json',dict(experiment='E227',task=dict(tag=f'v20_vol_t{h}',model='ridge'),blocks=len(outputs),features=FEATURES,predictions=int(result.prediction.notna().sum()),prediction_sha256=sha(destination/'predictions.parquet')))

def score():
    p=pd.read_parquet(ROOT/'panel.parquet');cov=pd.read_parquet(SOURCE/'coverage.parquet');causal=cov.causal_common.to_numpy();scored=cov.evaluation_available.to_numpy()
    names=[f'ridge_V20_VOL2_t{h}' for h in (20,60)];allnames=names+[n+'_residual' for n in names]
    raw=np.column_stack([pd.read_parquet(ROOT/'fits'/f'v20_vol_t{h}'/'predictions.parquet').set_index('row').prediction.reindex(p.row).to_numpy() for h in (20,60)])
    assert np.isfinite(raw[causal]).all();candidate=np.full((len(p),4),np.nan);candidate[:,:2]=raw
    routed=np.empty_like(candidate);groups=[];snapshots=[];cache={}
    meta=pd.read_parquet(REPO/'data'/S.DATASET/'historical_meta_models.parquet').reset_index()
    for date,idx in p.groupby('date',sort=True).indices.items():
        rows=idx[causal[idx]]
        if len(rows):
            lagged,snapshot=lagged_snapshots(meta,pd.Timestamp(date),p.loc[rows,'symbol'],cache)
            value,beta=S.residual_matrix(raw[rows],lagged);candidate[rows,2:]=value
            native,ns=MR.lagged_meta(meta,p.loc[rows,'symbol'],date);np.testing.assert_array_equal(native,lagged);assert ns==snapshot
            snapshots.append(dict(date=date,snapshot=snapshot,beta=beta.tolist()))
        br=rank_normalise(cov.baseline.to_numpy()[idx])
        for j in range(4):
            routed[idx,j]=route_slots(br,np.where(causal[idx],candidate[idx,j],np.nan))
            np.testing.assert_array_equal(routed[idx[~causal[idx]],j],br[~causal[idx]])
            np.testing.assert_array_equal(np.sort(routed[idx[causal[idx]],j]),np.sort(br[causal[idx]]))
        pos=np.flatnonzero(scored[rows])
        if len(pos)>=10:groups.append((pd.Timestamp(date),rows,pos))
    assert np.isfinite(routed).all()
    dates=pd.read_parquet(SOURCE/'score_dates.parquet');np.testing.assert_array_equal(dates.date.to_numpy(),pd.to_datetime([d for d,_,_ in groups]).to_numpy());np.testing.assert_array_equal(dates.assets.to_numpy(),[len(pos) for _,_,pos in groups])
    configs=[dict(id=f'e227_{name}',members=[name],member_indices=[j],size=1,kind='standalone',training_targets=str(20 if '_t20' in name else 60),model='feature_ridge',view='raw' if j<2 else 'residual',standardization='asset',design='own native bins',information_set='V20_VOL2',post_residualized=False) for j,name in enumerate(allnames)]
    arrays={h:np.empty((len(groups),4,2)) for h in (20,60)};native_checks=[]
    for d,(date,rows,pos) in enumerate(groups):
        for h,a in arrays.items():
            y=p.loc[rows[pos],f'target_binned_return_{h}'].to_numpy();m=p.loc[rows[pos],'meta_model'].to_numpy();values=candidate[rows[pos]]
            corr,mmc=S.matrix_scores(values,y,m);a[d,:,0]=np.nan_to_num(corr,nan=0.);a[d,:,1]=mmc
            if d in (0,len(groups)//2,len(groups)-1):
                z=pd.DataFrame(values,columns=allnames);yy=pd.Series(y,name='target');mm=pd.Series(m,name='meta_model')
                nc=numerai_corr(z,yy).fillna(0).to_numpy();nm=correlation_contribution(z,mm,yy).to_numpy()
                np.testing.assert_allclose(a[d,:,0],nc,atol=3e-8,rtol=1e-6);np.testing.assert_allclose(a[d,:,1],nm,atol=1e-12,rtol=1e-10)
                native_checks.append(dict(date=date,target=h,columns=4))
    assets=p.loc[np.concatenate([rows[pos] for _,rows,pos in groups]),'ucid'].nunique();summary=[]
    for h,a in arrays.items():
        assert np.isfinite(a).all();np.save(ROOT/f'daily_scores_t{h}.npy',a)
        for j,c in enumerate(configs):summary.append(dict({**c,'members':c['members'][0],'member_indices':str(j)},evaluation_target=h,status='success',assets=int(assets),assets_mean=float(dates.assets.mean()),assets_min=int(dates.assets.min()),dates=len(dates),**S.payout_metrics(a[:,j,0],a[:,j,1])))
    table=pd.DataFrame(summary).sort_values(['evaluation_target','payout'],ascending=[True,False]);parquet(ROOT/'summary.parquet',table);table.to_csv(ROOT/'summary.csv',index=False);save(ROOT/'trials.json',configs)
    parquet(ROOT/'standalone_forecasts.parquet',p[['row','ucid','symbol','date']].join(pd.DataFrame(candidate,columns=allnames)));parquet(ROOT/'routed_forecasts.parquet',p[['row','date']].join(pd.DataFrame(routed,columns=allnames)));save(ROOT/'residual_snapshots.json',snapshots)
    save(ROOT/'scoring_audit.json',dict(outcome='PASS WITH CONDITIONS',dates=len(dates),assets=int(assets),scored_asset_dates=int(dates.assets.sum()),native_score_checks=native_checks,all_date_routing_invariants=True,limitation='Matched feature-only research comparison, not native22 marginal or live performance.'))

def audit():
    p=pd.read_parquet(ROOT/'panel.parquet');prepared=json.loads((ROOT/'PREPARED.json').read_text());assert prepared['runner_sha256']==sha(__file__);assert prepared['panel_sha256']==sha(ROOT/'panel.parquet');assert prepared['native_sha256']==sha(NATIVE)
    common=np.load(SOURCE/'cache/common_rows.npy');assert prepared['common_sha256']==sha(SOURCE/'cache/common_rows.npy')
    original=pd.read_parquet(NATIVE,columns=['ucid','date']+FEATURES,filters=[('date','<',pd.Timestamp('2026-02-20'))]).reset_index().set_index(['ucid','date'])
    np.testing.assert_array_equal(p[FEATURES],original.reindex(pd.MultiIndex.from_frame(p[['ucid','date']]))[FEATURES])
    replays=[];maximum=0.
    for h in (20,60):
        qpred=pd.read_parquet(ROOT/'fits'/f'v20_vol_t{h}'/'predictions.parquet').set_index('row');target=f'target_binned_return_{h}'
        paths=sorted((ROOT/'blocks'/f'v20_vol_t{h}').glob('*/protocol.json'));assert len(paths)==11
        for path in paths:
            protocol=json.loads(path.read_text());reference=json.loads((SOURCE/'blocks'/f'feature_ridge_t{h}'/path.parent.name/'protocol.json').read_text());assert protocol==reference
            for item in sorted((path.parent/'ridge').glob('*.npz'))[:1] if path.parent.name in ('f04','f05','f10') else []:
                with np.load(item) as z:
                    asset=item.stem.removeprefix('asset_');g=p[p.ucid.astype(str).eq(asset)];start=int(p.loc[p.date.eq(pd.Timestamp(protocol['origin'])),'step'].iloc[0]);q,_=fold_panel(p,start,h)
                    tr=g.index[common[g.index]&q.train.to_numpy()[g.index]&np.isfinite(q[target].to_numpy()[g.index])].to_numpy();va=g.index[common[g.index]&q.validation.to_numpy()[g.index]&np.isfinite(q[target].to_numpy()[g.index])].to_numpy();ft=g.index[common[g.index]&q.final_fit.to_numpy()[g.index]&np.isfinite(q[target].to_numpy()[g.index])].to_numpy();te=g.index[common[g.index]&q.test.to_numpy()[g.index]].to_numpy()
                    for key,rows in [('training_rows',ft),('selection_training_rows',tr),('validation_rows',va),('query_rows',te)]:np.testing.assert_array_equal(z[key],rows)
                    x=p.loc[tr,FEATURES].to_numpy(float);v=p.loc[va,FEATURES].to_numpy(float);center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1;x=(x-center)/scale;v=(v-center)/scale
                    fast,_=primal(x,v,p.loc[tr,target].to_numpy(),N.ALPHAS);dual,_=N.dual_ridge(x@x.T,v@x.T,p.loc[tr,target].to_numpy())
                    err=float(np.max(np.abs(fast-dual)));maximum=max(maximum,err);np.testing.assert_allclose(fast,dual,atol=2e-8,rtol=2e-8)
                    loss=((dual-p.loc[va,target].to_numpy()[:,None])**2).mean(0);assert float(N.ALPHAS[loss.argmin()])==float(z['alpha'])
                    x=p.loc[ft,FEATURES].to_numpy(float);v=p.loc[te,FEATURES].to_numpy(float);center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1
                    np.testing.assert_array_equal(center,z['center']);np.testing.assert_array_equal(scale,z['scale']);x=(x-center)/scale;v=(v-center)/scale
                    dual,_=N.dual_ridge(x@x.T,v@x.T,p.loc[ft,target].to_numpy(),[float(z['alpha'])]);actual=qpred.prediction.reindex(p.loc[te,'row']).to_numpy();err=float(np.max(np.abs(dual[:,0]-actual)));maximum=max(maximum,err);np.testing.assert_allclose(actual,dual[:,0],atol=2e-8,rtol=2e-8)
                    replay=((v-z['shift'])@z['weights']+z['intercept']).ravel();np.testing.assert_array_equal(replay,actual)
                    replays.append(dict(target=h,fold=path.parent.name,asset=asset,alpha=float(z['alpha']),dual_prediction_error=err))
    assert len(replays)>=4
    forecasts=pd.read_parquet(ROOT/'standalone_forecasts.parquet');cov=pd.read_parquet(SOURCE/'coverage.parquet');meta=pd.read_parquet(REPO/'data'/S.DATASET/'historical_meta_models.parquet').reset_index()
    for date in pd.read_parquet(SOURCE/'score_dates.parquet').date.iloc[[0,234,-1]]:
        rows=p.index[p.date.eq(date)&cov.causal_common].to_numpy();lagged,snapshot=MR.lagged_meta(meta,p.loc[rows,'symbol'],date)
        for h in (20,60):
            name=f'ridge_V20_VOL2_t{h}';native,_=MR.residualise(forecasts.loc[rows,name].to_numpy(),lagged);np.testing.assert_allclose(native,forecasts.loc[rows,name+'_residual'],atol=1e-12,rtol=1e-12)
    result=dict(outcome='PASS WITH CONDITIONS',features=FEATURES,source_values_keys_labels_verified=True,all22_fold_protocols_matched=True,all_asset_training_counts_matched=True,common_coverage_unchanged=True,dual_solver_sweep_and_forecast_replays=replays,max_primal_dual_error=maximum,independent_native_residual_replays=True,conditions=['Native features are used exactly as supplied in the frozen v2.0 snapshot; no claim to reconstruct proprietary source formulas.','Same covered E212 development history; not a native22 marginal, live or investment-return claim.'])
    save(ROOT/'independent_completion_audit.json',result);print('E227 AUDIT',json.dumps(result,default=str),flush=True)

def integrate():
    history=SOURCE/'history'/f'before_E227_{int(time.time())}';history.mkdir(parents=True)
    for name in ['E212_volatility99_tsfm.pdf','E212_volatility99_tsfm.html','report_supplement.json','report_qa.json','COMPLETE.json','status.json','config.json','independent_completion_audit.json']:
        if (SOURCE/name).exists():shutil.copy2(SOURCE/name,history/name)
    frozen={str(p):sha(p) for p in [SOURCE/'all_trials.parquet',SOURCE/'all_trials.csv',SOURCE/'trials.json']+list((SOURCE/'fits').glob('*/predictions.parquet'))+list((REPO/'results/reports/crypto/e226_volatility_cs_ridge/fits').glob('*/predictions.parquet'))}
    extra=dict(experiment='E227',root=str(ROOT),summary_sha256=sha(ROOT/'summary.parquet'),trial_sha256=sha(ROOT/'trials.json'),array_sha256={str(h):sha(ROOT/f'daily_scores_t{h}.npy') for h in (20,60)})
    supplement=json.loads((SOURCE/'report_supplement.json').read_text());supplement['additional_supplements']=[extra];save(SOURCE/'report_supplement.json',supplement)
    table=pd.read_parquet(SOURCE/'all_trials.parquet');configs=json.loads((SOURCE/'trials.json').read_text());dates=pd.read_parquet(SOURCE/'score_dates.parquet');arrays={h:np.load(SOURCE/'cache/daily_scores'/f't{h}'/'corr_mmc.npy',mmap_mode='r') for h in (20,60)}
    S.report(table,configs,dates,arrays);assert all(sha(p)==expected for p,expected in frozen.items())
    pdfhash=sha(SOURCE/'E212_volatility99_tsfm.pdf')
    for root in (SOURCE,REPO/'results/reports/crypto/e226_volatility_cs_ridge'):
        file=root/'COMPLETE.json';c=json.loads(file.read_text());c.update(report_sha256=pdfhash,report_revision='E227 native v2.0 volatility ridge controls');save(file,c)
    c=json.loads((SOURCE/'config.json').read_text());c['code_hashes'][str(Path(S.__file__).relative_to(REPO))]=sha(S.__file__);c['report_revision']=dict(experiment='E227',original_numerical_results_unchanged=True,prior_report=str(history));save(SOURCE/'config.json',c)
    status=json.loads((SOURCE/'status.json').read_text());status.update(report_sha256=pdfhash,current_report_emailed=False,report_revision='E227 v2.0 volatility ridge controls');save(SOURCE/'status.json',status)
    save(ROOT/'COMPLETE.json',dict(experiment='E227',status='success',heads=2,summary_rows=8,report_sha256=pdfhash,original_numerical_hashes=frozen,audit='PASS WITH CONDITIONS',time=time.time()))

def run():
    save(ROOT/'status.json',dict(stage='preparing',time=time.time()));prepare()
    for h in (20,60):save(ROOT/'status.json',dict(stage='fitting',target=h,time=time.time()));fit(h)
    save(ROOT/'status.json',dict(stage='scoring',time=time.time()));score();audit()
    save(ROOT/'status.json',dict(stage='reporting',time=time.time()));integrate()
    from experiments.e226_validate_completion import run as check226
    from experiments.e212_validate_completion import run as check212
    check226();check212()
    save(ROOT/'status.json',dict(stage='complete',time=time.time()));print('E227 COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','audit','integrate']);a=ap.parse_args();globals()[a.action]()
