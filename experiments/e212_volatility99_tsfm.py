"""E212: complete, GPU-queued TSFM signal workflow on volatility99.

Frozen horizon-specific H; independent per-asset C-only ridge heads; matched
feature-only controls; both evaluation targets; exhaustive raw/post-residual
ensembles. No momentum data, old embedding caches or live changes.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import fcntl
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from experiments.e211_cross_target_tsfm import fold_panel,lagged_snapshots
from src.analysis import tsfm_momentum_study as N
from src.prediction import meta_resid as MR
from src.cluster import budget,hosts,manager
from src.analysis.asset_momentum import route_slots
from src.live.submit import rank_normalise
from e150_finish_scoring import matrix_scores

ROOT=REPO/'results/reports/crypto/e212_volatility99_tsfm'
SCR=ROOT/'cache'
DATASET='crypto/v2.0-e209-ts-volatility99'
MODELS=['chronos2','t0','tabpfnts','timestral']
NORMS={'chronos2':'rolling','t0':'rolling','tabpfnts':'asset','timestral':'robust','feature_ridge':'asset'}
WALLS={20:35,60:91}
PYTHON=E.PYTHON
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{socket.gethostname()}_{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


parquet=E.parquet
sha=E.sha
CODE=[Path(__file__),REPO/'src/data_mani/e212_signal_embeddings.py',
      REPO/'src/analysis/tsfm_momentum_study.py',REPO/'experiments/e206_momentum3_tsfm.py',
      REPO/'experiments/e211_cross_target_tsfm.py',REPO/'src/prediction/meta_resid.py',
      REPO/'experiments/e150_finish_scoring.py',REPO/'src/live/submit.py',
      REPO/'src/analysis/asset_momentum.py']+list((REPO/'src/data_mani/tsfm_state_adapters').glob('*.py'))


def counts(n_models):
    k=4*(n_models+1);e=sum(math.comb(k,s) for s in range(2,min(5,k)+1))
    return dict(heads=2*(n_models+1),standalone_streams=k,ensemble_subsets=e,
                scored_rows_per_target=k+2*e,total_scored_rows=2*(k+2*e))


def settings():
    return json.loads((ROOT/'config.json').read_text())


def verify():
    c=settings()
    for path,expected in c['code_hashes'].items():
        if sha(REPO/path)!=expected:raise RuntimeError('Frozen study code changed: '+path)
    for path,expected in c['input_hashes'].items():
        if sha(path)!=expected:raise RuntimeError('Frozen dataset input changed: '+path)
    assert sha(ROOT/'panel.parquet')==json.loads((ROOT/'PREPARED.json').read_text())['panel_hash']
    return c


def prepare(dataset=DATASET,models=None):
    if (ROOT/'PREPARED.json').exists():return verify()
    models=MODELS if models is None else models
    assert set(models)<=set(MODELS) and len(models)==len(set(models))
    data=REPO/'data'/dataset
    manifest=json.loads((data/'manifest.json').read_text())
    names=manifest['features']
    for name,expected in manifest['output_hashes'].items():assert sha(data/name)==expected,name
    frame=pd.read_parquet(data/'train.parquet').reset_index().sort_values(['date','ucid']).reset_index(drop=True)
    assert [c for c in frame if c.startswith('feature_')]==names and len(names)==manifest['n_features']
    assert not frame.duplicated(['ucid','date']).any() and not frame.duplicated(['symbol','date']).any()
    cutoff=pd.Timestamp(manifest['date_max'])+pd.Timedelta(days=1)
    assert frame.date.max()<cutoff<=pd.Timestamp('2026-02-20')
    cover=pd.read_parquet(data/'coverage.parquet').reset_index()[['ucid','date','covered']]
    frame=frame.merge(cover,on=['ucid','date'],how='left',validate='1:1',sort=False)
    assert frame.covered.notna().all() and np.isfinite(frame.loc[frame.covered,names]).all().all()
    # Match permanent identity, symbols and existing target20 before attaching target60.
    label_path=REPO/'data/crypto/v2.0-e156-frozen/train.parquet'
    labels=pd.read_parquet(label_path,columns=['ucid','date','target_binned_return_20','target_binned_return_60'],filters=[('date','<',cutoff)]).reset_index()
    assert not labels.duplicated(['ucid','date']).any()
    merged=frame.merge(labels,on=['ucid','date'],how='left',suffixes=('','_official'),validate='1:1',sort=False)
    assert (merged.symbol==merged.symbol_official).all()
    np.testing.assert_array_equal(merged.target_binned_return_20,merged.target_binned_return_20_official)
    frame=merged.drop(columns=['symbol_official','target_binned_return_20_official'])
    meta=pd.read_parquet(data/'historical_meta_models.parquet').reset_index().rename(columns={'prediction':'meta_model'})
    frame=frame.merge(meta,on=['symbol','date'],how='left',validate='1:1',sort=False)
    frame['row']=np.arange(len(frame));dates=np.sort(frame.date.unique());assets=np.sort(frame.ucid.unique())
    frame['step']=np.searchsorted(dates,frame.date);frame['asset_position']=pd.Index(assets).get_indexer(frame.ucid)
    ROOT.mkdir(parents=True,exist_ok=True);SCR.mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(REPO)):sha(p) for p in CODE}
    config=dict(experiment='E212',dataset=dataset,models=models,features=names,feature_count=len(names),
        training_targets=[20,60],evaluation_targets=[20,60],walls=WALLS,context=256,min_resolved=32,
        head_history=504,validation_dates=63,forecast_dates=130,first_origin_step=200,
        alpha_grid=N.ALPHAS.tolist(),rolling_windows=N.ROLLING,minimum_rolling=32,
        normalizations={m:NORMS[m] for m in models},design='C only; each own-asset included once; no raw-X append',
        covered_context_labels_only=True,standalone_baseline='feature_ridge; same common training/validation keys',
        residualization=dict(strength=MR.STRENGTH,lag_calendar_days=MR.LAG_DAYS,ensemble_post_pass=True),
        trial_counts=counts(len(models)),gpu_policy='GPU only extraction; one E212 card per host; wait/reprobe if occupied; no old share exception',
        cpu_ram_share=budget.SHARE,checkpoint_models='Pinned revisions in extractor; manifests/fingerprints saved per horizon/model',
        metric_definitions='p=0.1*corr+historical-meta proxyMMC; mean(p),mean(p)/std(p,ddof=0), max drawdown of [0,cumsum(p)]',
        code_hashes=code,input_hashes={str(p):sha(p) for p in [data/'train.parquet',data/'coverage.parquet',data/'historical_meta_models.parquet',data/'manifest.json',label_path]},
        limitations=['Development signal/formula selection already used this history; ensemble ranking exploratory.',
                    'Historical vendor availability and pretraining overlap uncertified; no native22 marginal/live/investment-return claim.'])
    save(ROOT/'config.json',config);parquet(ROOT/'panel.parquet',frame)
    estimate={}
    for horizon in WALLS:
        n=0
        for _,g in frame.groupby('ucid'):
            yy=np.full(len(dates),np.nan);yy[g.step]=np.where(g.covered,g[f'target_binned_return_{horizon}'],np.nan)
            prefix=np.r_[0,np.cumsum(np.isfinite(yy))]
            wall=np.searchsorted(dates,g.date.to_numpy()-np.timedelta64(WALLS[horizon],'D'),side='right')-1
            hi=np.maximum(wall+1,0);lo=np.maximum(0,hi-256)
            n+=int(((prefix[hi]-prefix[lo]>=32)&g.covered.to_numpy()).sum())
        estimate[str(horizon)]=dict(eligible_origins_per_model=n,total_model_origins=n*len(models))
    prepared=dict(experiment='E212',dataset=dataset,rows=len(frame),features=names,assets=assets.tolist(),dates=list(map(str,dates)),
        panel_hash=sha(ROOT/'panel.parquet'),dataset_manifest=manifest,date_cutoff=str(cutoff),
        source_covered_rows=int(frame.covered.sum()),origin_counts=estimate,
        estimate='GPU elapsed ETA updated from smoke and actual per-asset timings. Approx10GB shared-home embeddings/caches plus score arrays<1GB; normalized H stays in RAM, scaler references and head parameters retained. Exhaustive report table exported, PDF summarizes all standalone/top3 ensembles.',
        labels='Verified official frozen source by permanent ID/date, symbol and target20 equality; unavailable target60 retained NaN.')
    save(ROOT/'PREPARED.json',prepared)
    save(ROOT/'status.json',dict(stage='prepared_queued_gpu_smokes',trial_counts=config['trial_counts'],origin_counts=estimate,time=time.time()))
    print('PREPARED',json.dumps(config['trial_counts']),json.dumps(estimate),flush=True)


def extraction_dir(h,model):return SCR/f't{h}'/'embeddings'/model


def extraction_complete(panel,h,model):
    directory=extraction_dir(h,model)
    wanted=panel.groupby('ucid').covered.any()
    return all((directory/f'asset_{a}.npz').exists() and (directory/f'asset_{a}.json').exists() for a in wanted.index[wanted])


def collect():
    c=verify();p=pd.read_parquet(ROOT/'panel.parquet');common=p.covered.to_numpy().copy()
    for h in WALLS:
        for model in c['models']:
            assert extraction_complete(p,h,model)
            directory=extraction_dir(h,model);manifest=json.loads((directory/'extractor.json').read_text());width=manifest['width']
            assert manifest['target']==f'target_binned_return_{h}' and manifest['features']==c['features']
            path=directory/'H_rows.npy'
            if not path.exists():
                tmp=directory/'H_rows.partial.npy'
                x=np.lib.format.open_memmap(tmp,mode='w+',dtype='float16',shape=(len(p),width));x[:]=np.nan
                for asset in directory.glob('asset_*.npz'):
                    info=json.loads(asset.with_suffix('.json').read_text());assert info['fingerprint']==manifest['fingerprint']
                    with np.load(asset) as z:
                        assert p.loc[z['rows'],'ucid'].astype(str).eq(asset.stem.removeprefix('asset_')).all()
                        x[z['rows']]=z['states']
                x.flush();del x;tmp.replace(path)
            valid=np.isfinite(np.load(path,mmap_mode='r')).all(axis=1)
            np.save(directory/'valid_rows.npy',valid);common &= valid
    common=N.common_history(p.ucid.to_numpy(),p.date.to_numpy(),common,32)
    common &= pd.Series(common).groupby(p.date).transform('sum').ge(2).to_numpy()
    path=SCR/'common_rows.npy';tmp=SCR/'common_rows.partial.npy'
    np.save(tmp,common);tmp.replace(path)
    parquet(ROOT/'embedding_coverage.parquet',p[['row','ucid','symbol','date','covered']].assign(common_causal=common))
    save(ROOT/'EMBEDDINGS_READY.json',dict(models=c['models'],targets=[20,60],common_rows=int(common.sum()),mask_sha256=sha(path),time=time.time()))


def solve_head(panel,horizon,model,norm,window,phase,blockroot,blockscr,selected=None,representation_cache=None):
    target=f'target_binned_return_{horizon}';common=np.load(SCR/'common_rows.npy')
    train=common & panel.train.to_numpy() & np.isfinite(panel[target])
    fit=common & panel['train' if phase=='selection' else 'final_fit'].to_numpy() & np.isfinite(panel[target])
    h=np.load(extraction_dir(horizon,model)/'H_rows.npy',mmap_mode='r')
    valid=np.load(extraction_dir(horizon,model)/'valid_rows.npy') & panel.covered.to_numpy()
    directory=blockscr/'normalizations'/f'{phase}_w{window}';directory.mkdir(parents=True,exist_ok=True)
    # Derived normalized H is used once per block/phase/window. Recompute it
    # from frozen H and retain exact scaler references instead of hundreds of
    # GB of duplicate matrices. This does not alter any numerical operation.
    # Rolling statistics depend only on immutable H/history, not fit labels or
    # forecast-block masks. Reuse precisely those arrays within this one head
    # process; pooled/asset/robust scalers always refit for the current block.
    memo=representation_cache if norm=='rolling' else None
    key=(model,horizon,window)
    if memo is not None:
        source=extraction_dir(horizon,model)/'H_rows.npy'
        signature=(str(source.resolve()),source.stat().st_size,source.stat().st_mtime_ns)
        identity=panel[['row','ucid','date','step','covered']]
        if 'identity' not in memo:
            memo.update(identity=identity.copy(),source=signature,valid=valid.copy(),normalizations={},kernels={})
        if memo['source']!=signature or not memo['identity'].equals(identity) or not np.array_equal(memo['valid'],valid):
            raise RuntimeError('Rolling representation cache source/row/coverage identity changed')
        if key not in memo['normalizations']:
            z,refs=N.normalize(h,panel.ucid.to_numpy(),panel.date.to_numpy(),valid,fit,norm,window)
            z.flags.writeable=False
            memo['normalizations'][key]=(z,refs,np.isfinite(z).all(axis=1))
        z,refs,valid=memo['normalizations'][key]
    else:
        z,refs=N.normalize(h,panel.ucid.to_numpy(),panel.date.to_numpy(),valid,fit,norm,window)
        valid=np.isfinite(z).all(axis=1)
    np.savez_compressed(directory/'scaler.npz',**refs)
    vocabulary=np.sort(panel.loc[train].groupby('ucid').size().loc[lambda s:s>=60].index.to_numpy())
    kernel_key=(key,tuple(vocabulary))
    if memo is not None and kernel_key in memo['kernels']:
        kernel=memo['kernels'][kernel_key]
    else:
        kernel=N.concat_kernel(z,panel.ucid.to_numpy(),panel.step.to_numpy(),valid,vocabulary,panel.step.max()+1)
        if memo is not None:
            kernel.flags.writeable=False
            memo['kernels'][kernel_key]=kernel
    info=dict(vocabulary=vocabulary.tolist(),columns=len(vocabulary)*h.shape[1],own_once=True,fill='zero absent context blocks; no masks as predictors',constants='exact implicit removal by centered dual kernel')
    save(directory/'design.json',info)
    query=common & valid & panel['validation' if phase=='selection' else 'test'].to_numpy()
    if phase=='selection':query &= np.isfinite(panel[target])
    result=np.full(len(panel),np.nan);records=[];parameters={}
    for asset,g in panel.groupby('ucid'):
        fi=g.index[fit[g.index]&valid[g.index]].to_numpy();vi=g.index[query[g.index]].to_numpy()
        if len(fi)<60 or not len(vi) or (phase=='selection' and len(vi)<20):continue
        if phase=='final' and str(asset) not in selected:continue
        ft,qt=panel.loc[fi,'step'].to_numpy(),panel.loc[vi,'step'].to_numpy()
        alphas=N.ALPHAS if phase=='selection' else [selected[str(asset)]['alpha']]
        pred,params=N.dual_ridge(kernel[np.ix_(ft,ft)],kernel[np.ix_(qt,ft)],panel.loc[fi,target].to_numpy(),alphas)
        if phase=='selection':
            loss=np.mean((pred-panel.loc[vi,target].to_numpy()[:,None])**2,axis=0);j=int(loss.argmin());alpha=float(alphas[j]);mse=float(loss[j])
        else:j=0;alpha=float(alphas[0]);mse=selected[str(asset)]['mse']
        result[vi]=pred[:,j];parameters[str(asset)]=dict(alpha=alpha,mse=mse)
        records.append(dict(asset=str(asset),alpha=alpha,mse=mse,n_train=len(fi),n_query=len(vi)))
        dest=directory/'ridge';dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/f'asset_{asset}.npz',training_rows=fi,dual_weights=params['dual_weights'][:,j],kernel_train_mean=params['kernel_train_mean'],kernel_grand_mean=params['kernel_grand_mean'],intercept=params['intercept'],alpha=alpha)
    return result,records,parameters,info


def feature_baseline(panel,horizon,directory):
    target=f'target_binned_return_{horizon}';features=settings()['features'];common=np.load(SCR/'common_rows.npy')
    values=np.full(len(panel),np.nan);records=[]
    for asset,g in panel.groupby('ucid'):
        masks={k:g.index[common[g.index]&panel[k].to_numpy()[g.index]&np.isfinite(panel[target].to_numpy()[g.index])] for k in ['train','validation','final_fit']}
        tr,va,ft=masks['train'],masks['validation'],masks['final_fit']
        te=g.index[common[g.index]&panel.test.to_numpy()[g.index]]
        if len(tr)<60 or len(va)<20 or not len(te):continue
        x=panel.loc[tr,features].to_numpy(float);v=panel.loc[va,features].to_numpy(float)
        center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1
        x=(x-center)/scale;v=(v-center)/scale
        pred,_=N.dual_ridge(x@x.T,v@x.T,panel.loc[tr,target].to_numpy())
        error=((pred-panel.loc[va,target].to_numpy()[:,None])**2).mean(0);alpha=float(N.ALPHAS[error.argmin()])
        x=panel.loc[ft,features].to_numpy(float);v=panel.loc[te,features].to_numpy(float)
        center=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1;x=(x-center)/scale;v=(v-center)/scale
        pred,params=N.dual_ridge(x@x.T,v@x.T,panel.loc[ft,target].to_numpy(),[alpha]);values[te]=pred[:,0]
        dest=directory/'ridge';dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/f'asset_{asset}.npz',training_rows=ft,center=center,scale=scale,alpha=alpha,**params)
        records.append(dict(asset=str(asset),alpha=alpha,validation_mse=float(error.min()),training_rows=len(ft)))
    # Output fallback uses resolved past labels for every original asset, not candidate imputation.
    means=panel.loc[panel.final_fit & np.isfinite(panel[target])].groupby('ucid')[target].mean()
    fallback=panel.ucid.map(means).fillna(.5).to_numpy()
    save(directory/'parameters.json',records)
    return values,np.where(np.isfinite(values),values,fallback)


def fit_stream(model,horizon):
    c=verify();assert model in c['models']+['feature_ridge']
    name=f'{model}_t{horizon}';destination=ROOT/'fits'/name
    if (destination/'COMPLETE.json').exists():return
    p=pd.read_parquet(ROOT/'panel.parquet');outputs=[];representation_cache={}
    for block,start in enumerate(range(200,p.step.max()+1,130)):
        part=fold_panel(p,start,horizon)
        if part is None:continue
        q,protocol=part;root=ROOT/'blocks'/name/f'f{block:02d}';scratch=SCR/'blocks'/name/f'f{block:02d}'
        if (root/'COMPLETE.json').exists():outputs.append(pd.read_parquet(root/'predictions.parquet'));continue
        root.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True);save(root/'protocol.json',protocol)
        forecast=q.loc[q.test,['row','ucid','symbol','date']].copy();values=np.full(len(q),np.nan)
        if model=='feature_ridge':
            values,fallback=feature_baseline(q,horizon,scratch);forecast['baseline']=fallback[q.test]
        else:
            common=np.load(SCR/'common_rows.npy');finite=np.isfinite(q[f'target_binned_return_{horizon}'])
            training=q.loc[common&q.train.to_numpy()&finite].groupby('ucid').size()
            validation=q.loc[common&q.validation.to_numpy()&finite].groupby('ucid').size().reindex(training.index,fill_value=0)
            if ((training>=60)&(validation>=20)).any():
                choices=[]
                for window in (N.ROLLING if NORMS[model]=='rolling' else [126]):
                    pred,records,params,info=solve_head(q,horizon,model,NORMS[model],window,'selection',root,scratch,representation_cache=representation_cache)
                    if records:choices.append((np.mean([r['mse'] for r in records]),window,records,params,info))
                if choices:
                    loss,window,records,params,info=min(choices,key=lambda x:(x[0],x[1]))
                    save(root/'selection.json',dict(window=window,loss=loss,assets=records,candidates=[dict(window=x[1],loss=x[0]) for x in choices]))
                    values,records,_,info=solve_head(q,horizon,model,NORMS[model],window,'final',root,scratch,params,representation_cache=representation_cache)
                    save(root/'parameters.json',dict(assets=records,design=info))
        forecast['prediction']=values[q.test.to_numpy()]
        parquet(root/'predictions.parquet',forecast);save(root/'COMPLETE.json',dict(predictions=int(forecast.prediction.notna().sum())))
        outputs.append(forecast);print('BLOCK',name,block,int(forecast.prediction.notna().sum()),flush=True)
    result=pd.concat(outputs,ignore_index=True);assert result.row.is_unique
    parquet(destination/'predictions.parquet',result)
    save(destination/'COMPLETE.json',dict(task=dict(tag=name,model=model),blocks=len(outputs),predictions=int(result.prediction.notna().sum()),time=time.time()))


def residual_matrix(predictions,meta):
    """Columnwise exact shared residualise algebra, finite candidate matrix."""
    x=np.asarray(predictions,float);assert np.isfinite(x).all()
    gp=ndtri((rankdata(x,axis=0,method='average')-.5)/len(x))
    gm=MR.gauss_rank(meta) if meta is not None else np.zeros(len(x))
    denominator=float(gm@gm)
    if denominator==0:return gp,np.full(x.shape[1],np.nan)
    beta=gp.T@gm/denominator
    return gp-MR.STRENGTH*np.outer(gm,beta),beta


def definitions(k):
    return [members for size in range(2,min(5,k)+1) for members in itertools.combinations(range(k),size)]


def payout_metrics(corr,mmc):
    p=.1*np.asarray(corr)+np.asarray(mmc);cum=np.r_[0,p.cumsum()]
    return dict(corr=float(np.mean(corr)),mmc=float(np.mean(mmc)),payout=float(p.mean()),
                **{'sharpe.p':float(p.mean()/p.std(ddof=0)) if p.std(ddof=0)>0 else np.nan,
                   'maxdd.p':float(np.max(np.maximum.accumulate(cum)-cum))})


def forecast_pool():
    c=verify();p=pd.read_parquet(ROOT/'panel.parquet');names=[f'{m}_t{h}' for h in WALLS for m in c['models']+['feature_ridge']]
    raw=[]
    for name in names:
        f=pd.read_parquet(ROOT/'fits'/name/'predictions.parquet');assert f.row.is_unique
        raw.append(f.set_index('row').prediction.reindex(p.row).to_numpy())
    raw=np.column_stack(raw);causal=np.isfinite(raw).all(axis=1)&np.load(SCR/'common_rows.npy')
    candidate=np.full((len(p),2*len(names)),np.nan)
    candidate[:,:len(names)]=raw;allnames=names+[n+'_residual' for n in names]
    baseline=np.full(len(p),.5)
    base=pd.read_parquet(ROOT/'fits/feature_ridge_t20/predictions.parquet').set_index('row')
    mask=p.row.isin(base.index)
    baseline[mask]=base.baseline.reindex(p.loc[mask,'row']).to_numpy()
    meta=pd.read_parquet(REPO/'data'/c['dataset']/'historical_meta_models.parquet').reset_index()
    assert meta.date.max()<pd.Timestamp(json.loads((ROOT/'PREPARED.json').read_text())['date_cutoff'])
    cache={};snapshots=[];groups=[];routed=np.empty_like(candidate)
    scored=causal & np.isfinite(p.target_binned_return_20)&np.isfinite(p.target_binned_return_60)&np.isfinite(p.meta_model)
    for date,idx in p.groupby('date',sort=True).indices.items():
        eligible=idx[causal[idx]]
        lagged,snapshot=lagged_snapshots(meta,pd.Timestamp(date),p.loc[eligible,'symbol'],cache)
        if len(eligible):
            value,beta=residual_matrix(raw[eligible],lagged);candidate[eligible,len(names):]=value
            native,nativesnapshot=MR.lagged_meta(meta,p.loc[eligible,'symbol'],date)
            np.testing.assert_array_equal(lagged,native);assert snapshot==nativesnapshot
            snapshots.extend(dict(date=date,snapshot=snapshot,stream=names[j],beta=float(b),meta_rows=int(np.isfinite(lagged).sum()),pass_through=bool(np.isnan(b))) for j,b in enumerate(beta))
        br=rank_normalise(baseline[idx])
        for j in range(len(allnames)):routed[idx,j]=route_slots(br,np.where(causal[idx],candidate[idx,j],np.nan))
        evalpos=np.flatnonzero(np.asarray(scored)[eligible])
        if len(evalpos)>=10:groups.append((pd.Timestamp(date),eligible,evalpos,lagged))
    assert groups and np.isfinite(routed).all()
    parquet(ROOT/'standalone_forecasts.parquet',p[['row','ucid','symbol','date']].join(pd.DataFrame(candidate,columns=allnames)))
    parquet(SCR/'routed_standalone.parquet',p[['row','date']].join(pd.DataFrame(routed,columns=allnames)))
    parquet(ROOT/'coverage.parquet',p[['row','ucid','symbol','date']].assign(causal_common=causal,evaluation_available=scored,baseline=baseline))
    parquet(ROOT/'residualization_snapshots.parquet',pd.DataFrame(snapshots))
    save(ROOT/'forecast_reconstruction.json',dict(standalone_candidates='standalone_forecasts.parquet',
        routing_baseline='coverage.parquet:baseline',causal_mask='coverage.parquet:causal_common',
        ensembles='Equal means of causal-universe within-date standalone percentile ranks; trials.json membership. '
                  'Post view applies shared residualise once against recorded D-5 snapshots, then exact baseline slots. '
                  'Saved candidates, masks, snapshots and definitions reproduce all ensemble forecasts without refitting.',
        ensemble_beta='cache/ensemble_residual_beta.npy: date x combination index; NaN denotes shared-helper pass-through; source snapshot in residualization_snapshots.parquet',
        ranking='scipy rankdata average; (rank-.5)/n for blending; deterministic project rank_normalise/route_slots for output'))
    return p,allnames,candidate,groups


def trials(names):
    c=settings();configs=[]
    for j,name in enumerate(names):
        original=name.removesuffix('_residual');model=original.rsplit('_t',1)[0];h=int(original.rsplit('_t',1)[1])
        configs.append(dict(id=f'e212_volatility99_r01_standalone_{j:02d}',members=[name],member_indices=[j],size=1,kind='standalone',
            training_targets=str(h),model=model,view='residual' if name.endswith('_residual') else 'raw',
            standardization=NORMS[model],design='raw signals' if model=='feature_ridge' else 'C',post_residualized=False))
    for i,members in enumerate(definitions(len(names))):
        for post in [False,True]:
            configs.append(dict(id=f'e212_volatility99_r01_blend_{i:05d}_{"post" if post else "raw"}',
                members=[names[j] for j in members],member_indices=list(members),size=len(members),kind='ensemble',
                training_targets=','.join(sorted({str(20 if '_t20' in names[j] else 60) for j in members})),
                model='equal-rank ensemble',view='post-residualized blend' if post else 'rank-average blend',
                standardization='member defaults',design='member C / raw feature control',post_residualized=post))
    assert len(configs)==c['trial_counts']['scored_rows_per_target']
    return configs


def score():
    if (ROOT/'COMPLETE.json').exists():return
    p,names,candidate,groups=forecast_pool();configs=trials(names);k=len(names);combos=definitions(k)
    save(ROOT/'trials.json',configs)
    dates=pd.DataFrame([dict(index=i,date=g[0],assets=len(g[2]),causal_assets=len(g[1])) for i,g in enumerate(groups)])
    parquet(ROOT/'score_dates.parquet',dates)
    weights=np.zeros((k,len(combos)))
    for i,members in enumerate(combos):weights[list(members),i]=1/len(members)
    beta_path=SCR/'ensemble_residual_beta.npy'
    ensemble_beta=np.lib.format.open_memmap(beta_path,mode='r+' if beta_path.exists() else 'w+',dtype='float64',shape=(len(groups),len(combos)))
    arrays={}
    for h in WALLS:
        directory=SCR/'daily_scores'/f't{h}';directory.mkdir(parents=True,exist_ok=True)
        path=directory/'corr_mmc.npy'
        arrays[h]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float64',shape=(len(groups),len(configs),2))
    parity=json.loads((ROOT/'scoring_parity.json').read_text()) if (ROOT/'scoring_parity.json').exists() else []
    for d,(date,causal_rows,evalpos,lagged) in enumerate(groups):
        done=ROOT/'scored_dates'/f'd{d:04d}.json'
        if done.exists():continue
        values=candidate[causal_rows];ranks=(rankdata(values,axis=0,method='average')-.5)/len(causal_rows)
        scored_rows=causal_rows[evalpos]
        def assign(matrix,columns):
            for h in WALLS:
                corr,mmc=matrix_scores(matrix[evalpos],p[f'target_binned_return_{h}'].to_numpy()[scored_rows],p.meta_model.to_numpy()[scored_rows])
                corr=np.nan_to_num(corr,nan=0.)
                assert np.isfinite(corr).all() and np.isfinite(mmc).all()
                arrays[h][d,columns,0]=corr;arrays[h][d,columns,1]=mmc
        assign(values,np.arange(k))
        for first in range(0,len(combos),256):
            last=min(first+256,len(combos));blend=ranks@weights[:,first:last]
            post,beta=residual_matrix(blend,lagged)
            ensemble_beta[d,first:last]=beta
            columns=np.arange(k+2*first,k+2*last)
            both=np.empty((len(blend),2*(last-first)));both[:,0::2]=blend;both[:,1::2]=post
            assign(both,columns)
            if d in [0,len(groups)//2,len(groups)-1] and first==0:
                for j in range(min(3,blend.shape[1])):
                    native,b=MR.residualise(blend[:,j],lagged)
                    np.testing.assert_allclose(native,post[:,j],rtol=1e-12,atol=1e-12)
                for h in WALLS:
                    for j in [0,1,min(5,both.shape[1]-1)]:
                        corr,mmc=matrix_scores(both[evalpos,j,None],p[f'target_binned_return_{h}'].to_numpy()[scored_rows],p.meta_model.to_numpy()[scored_rows])
                        np.testing.assert_allclose([corr[0],mmc[0]],arrays[h][d,columns[j]],rtol=1e-11,atol=1e-12,equal_nan=True)
                parity.append(dict(date=date,scalar_batch=True,native_residual=True))
        ensemble_beta.flush()
        for array in arrays.values():array.flush()
        save(ROOT/'scoring_parity.json',parity)
        save(done,dict(date=date,assets=len(evalpos),complete=True));save(ROOT/'status.json',dict(stage='scoring',completed_dates=d+1,total_dates=len(groups),time=time.time()))
    summaries=[]
    unique_assets=int(p.loc[np.concatenate([rows[pos] for _,rows,pos,_ in groups]),'ucid'].nunique())
    for h,array in arrays.items():
        for i,config in enumerate(configs):
            summaries.append(dict(**config,evaluation_target=h,status='success',assets=unique_assets,assets_mean=float(dates.assets.mean()),assets_min=int(dates.assets.min()),dates=len(groups),**payout_metrics(array[:,i,0],array[:,i,1])))
    table=pd.DataFrame(summaries).sort_values(['evaluation_target','payout','id'],ascending=[True,False,True],kind='stable')
    table['members']=table.members.map(' + '.join);table['member_indices']=table.member_indices.map(lambda x:','.join(map(str,x)))
    table.to_csv(ROOT/'all_trials.csv',index=False);parquet(ROOT/'all_trials.parquet',table)
    standalone=table[table.kind=='standalone'];standalone.to_csv(ROOT/'standalone_summary.csv',index=False)
    controls=standalone[(standalone.evaluation_target==20)&(standalone.view=='raw')].copy()
    controls['tag']=controls.members;controls.rename(columns={'dates':'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv',index=False)
    save(ROOT/'scoring_parity.json',parity)
    audit=dict(outcome='PASS WITH CONDITIONS',checks=['Frozen dataset/key/target identity, no missing labels imputed; all input/cache fingerprints.',
        'Each model/horizon/device source prefix, future-label and batch-origin audit before extraction.',
        'Common causal H coverage atomically saved before fitting; own-asset missing rows excluded.',
        'Per-asset MSE heads, causal scalers and fixed training vocabulary; original targets unchanged.',
        'D-5 snapshot parity with shared helper; raw and once-residualized standalones and ensembles.',
        'Full finite routed standalone output with exact uncovered baseline ranks and covered slot sets.',
        'All declared trials, shared assets/dates/both labels, fresh prediction-blend scoring and scalar parity.'],
        conditions=settings()['limitations']+['Historical meta is a proxy benchmark for target60; all curves are additive payout scores.'])
    save(ROOT/'audit.json',audit);report(table,configs,dates,arrays)
    save(ROOT/'COMPLETE.json',dict(experiment='E212',status='success',**settings()['trial_counts'],
        actual_rows=len(table),dates=len(groups),report=str(ROOT/'E212_volatility99_tsfm.pdf'),report_sha256=sha(ROOT/'E212_volatility99_tsfm.pdf'),audit=audit['outcome'],time=time.time()))


def report(table,configs,dates,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index={c['id']:i for i,c in enumerate(configs)};body='<h1>E212 — Volatility99 TSFM signal experiment</h1>'
    prepared=json.loads((ROOT/'PREPARED.json').read_text())
    labels=pd.read_parquet(ROOT/'panel.parquet',columns=['date','target_binned_return_20','target_binned_return_60','meta_model'])
    base_keys=int((labels.date.isin(dates.date)&np.isfinite(labels.target_binned_return_20)&np.isfinite(labels.target_binned_return_60)&np.isfinite(labels.meta_model)).sum())
    body+=f'<p>Dataset: {html.escape(settings()["dataset"])}. Evaluation covers {len(dates):,} dates, {int(table.assets.iloc[0]):,} distinct assets and {int(dates.assets.sum()):,} scored asset-date observations from {prepared["rows"]:,} original rows. All comparisons use the same common covered keys for both targets.</p>'
    body+=f'<p>On the evaluated dates, the common predictor cohort covers {int(dates.assets.sum()):,} / {base_keys:,} rows with both labels and the historical meta available ({100*dates.assets.sum()/base_keys:.1f}%). Results describe this covered cohort.</p>'
    body+='<p>Head inputs: design C concatenates standardized H from each asset in the fixed training universe once, with a separate supervised ridge head for each predicted asset. The feature-only control uses the 99 signals of the predicted asset. Comparisons reflect both the representation and the information set.</p>'
    body+=f'<p>{len(settings()["models"])} frozen foundation models; separate target20/60 contexts and independently fitted per-asset ridge heads. '
    body+='Recommended H standardization and design C only; 99-feature ridge controls on matched covered training keys. '
    body+='Raw/residual views use strength0.5 and meta snapshots at least5 calendar days old. Every 2–5-member rank blend '
    body+='also receives a separate, single post-residualization pass, including blends with residualized members. '
    body+='Common causal ranks/transforms precede the joint evaluation-label/meta mask. '
    body+='Payout=0.1 CORR+historical-meta proxy MMC. sharpe.p uses population standard deviation, unannualized; '
    body+='maxdd.p is drawdown of additive cumulative payout. Target60 MMC uses the available historical-meta proxy. '
    body+='Top THREE ensembles are chosen on displayed development history and are exploratory. '
    body+='All trials (including equivalent configurations) are in all_trials.csv/parquet; complete daily CORR/MMC arrays '
    body+='are stored in the shared cache daily_scores directory, indexed by score_dates.parquet and trials.json.</p>'
    raw_names=[v['members'][0] for v in configs if v['kind']=='standalone' and v['view']=='raw']
    colors={name:plt.get_cmap('tab10')(i) for i,name in enumerate(raw_names)}
    cols=['id','members','training_targets','evaluation_target','view','standardization','design','assets','assets_mean','dates','corr','mmc','payout','sharpe.p','maxdd.p']
    for h,array in arrays.items():
        section=table[table.evaluation_target==h];stand=section[section.kind=='standalone'];top=section[section.kind=='ensemble'].head(3)
        pay=.1*array[:,:,0]+array[:,:,1]
        # Record the precise plot selections and curves for independent reconstruction.
        picks=stand.id.tolist()+top.id.tolist()
        plotdata=pd.DataFrame({'date':dates.date})
        for identity in picks:plotdata[identity]=pay[:,index[identity]]
        parquet(ROOT/f'plot_daily_payout_t{h}.parquet',plotdata)
        stand_indices=[index[x] for x in stand.id];corr=np.corrcoef(pay[:,stand_indices],rowvar=False)
        pd.DataFrame(corr,index=stand.members,columns=stand.members).to_csv(ROOT/f'daily_payout_correlation_t{h}.csv')
        fig,ax=plt.subplots(figsize=(12,10));image=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(len(stand)),stand.members,rotation=90,fontsize=9);ax.set_yticks(range(len(stand)),stand.members,fontsize=9)
        ax.set_title(f'Target{h}: Pearson correlation of DAILY payout scores');fig.colorbar(image,ax=ax);fig.tight_layout()
        fig.savefig(ROOT/f'payout_heatmap_t{h}.png',dpi=150);plt.close(fig)
        for add_top in [False,True]:
            fig,ax=plt.subplots(figsize=(15,9),layout='constrained')
            rows=pd.concat([top,stand]) if add_top else stand
            for row in rows.itertuples():
                y=pay[:,index[row.id]].cumsum();np.testing.assert_allclose(y,plotdata[row.id].cumsum())
                if row.kind=='ensemble':
                    place=top.id.tolist().index(row.id)
                    color=['#111111','#8e44ad','#d99000'][place];style=['-','--',':'][place]
                else:
                    color=colors[row.members.removesuffix('_residual')];style='--' if row.view=='residual' else '-'
                ax.plot(dates.date,y,label=(row.id if row.kind=='ensemble' else row.members)+(' [POST]' if row.post_residualized else ''),
                        color=color,ls=style,lw=2.4 if row.kind=='ensemble' else .9,alpha=1 if row.kind=='ensemble' else .7)
            ax.set_title(f'Target{h}: '+('top THREE ensembles + all standalone variants' if add_top else 'all standalone variants'))
            ax.set_ylabel('Cumulative payout score');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
            name=f'{"top3_plus" if add_top else "all"}_standalones_t{h}.png';fig.savefig(ROOT/name,dpi=150);plt.close(fig)
        body+=f'<h2>Evaluation target{h}: all standalone variants</h2>'+stand[cols].to_html(index=False,float_format=lambda v:f'{v:.6f}')
        body+=f'<img style="width:100%" src="all_standalones_t{h}.png"><img style="width:100%" src="payout_heatmap_t{h}.png">'
        body+='<h2>Top THREE ensembles (exploratory)</h2>'+top[cols].to_html(index=False,float_format=lambda v:f'{v:.6f}')
        body+=f'<img style="width:100%" src="top3_plus_standalones_t{h}.png">'
    body+='<h2>Interpretation</h2><p>This is a retrospective development comparison. Volatility features were selected in earlier studies, and the displayed history also selects the top ensembles. Historical vendor vintages and pretrained-model overlap are not independently certified. Payout scores are Numerai metrics, not investment returns.</p>'
    body+='<h2>Audit</h2><p>'+html.escape(json.loads((ROOT/'audit.json').read_text())['outcome'])+'. Full audit, provenance and configuration are retained in separate machine-readable artifacts.</p>'
    page=ROOT/'E212_volatility99_tsfm.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:10px Arial}table{border-collapse:collapse;font-size:8px}td,th{border:1px solid #ccc;padding:3px;overflow-wrap:anywhere}pre{white-space:pre-wrap}img{display:block;margin:8px auto;object-fit:contain}@media print{table{break-inside:avoid!important;page-break-inside:avoid!important}td:nth-child(n+8),th:nth-child(n+8){white-space:nowrap;overflow-wrap:normal}img{max-width:100%;max-height:165mm;width:auto!important;break-inside:avoid}}</style></head><body>'+body+'</body></html>')
    command='cd '+shlex.quote(str(REPO))+'\nexec '+shlex.join([PYTHON,'-m','src.analysis.html_pdf',str(page)])
    subprocess.run(['ssh','-o','BatchMode=yes','lovelace','bash','-s'],input=command,text=True,check=True,timeout=300)
    text=subprocess.check_output(['pdftotext',str(page.with_suffix('.pdf')),'-'],text=True)
    assert 'E212' in text and 'top three' in text.lower()


def email_report():
    # Reuse the verified attachment/receipt implementation without its dataset paths.
    receipt=ROOT/'email_delivery.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('smtp_accepted'):return
    import smtplib
    from email.message import EmailMessage
    from src.live.env import load_env
    from src.live.notify import _config
    load_env(required=());config=_config();assert config
    pdf=ROOT/'E212_volatility99_tsfm.pdf'
    msg=EmailMessage();msg['From']=config['user'];msg['To']=config['to'];msg['Subject']='[numerai-crypto] E212 volatility99 TSFM signal experiment — completed PDF'
    msg.set_content(f'E212 is complete. Attached: {len(settings()["models"])} foundation models, horizon-specific H and per-asset ridge heads, feature-only baselines, both evaluation targets, all standalone variants and top-three exhaustive ensemble plots. Full machine-readable trial/score artifacts accompany the local report. Development research only.')
    msg.add_attachment(pdf.read_bytes(),maintype='application',subtype='pdf',filename=pdf.name)
    assert len(msg.as_bytes())<24_000_000
    with smtplib.SMTP(config['host'],config['port'],timeout=120) as server:
        server.starttls();server.login(config['user'],config['password']);refused=server.send_message(msg);assert not refused
        save(receipt,dict(smtp_accepted=True,recipient=config['to'],pdf_sha256=sha(pdf),time=time.time()))


def recover_locks():
    for path in SCR.glob('t*/embeddings/*/*.lock'):
        try:owner=json.loads(path.read_text());pid=int(owner['pid'])
        except (OSError,ValueError,KeyError):continue
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',owner['host'],f'ps -p {pid} -o args='],capture_output=True,text=True,timeout=8)
            if result.returncode in [0,1] and 'src.data_mani.e212_signal_embeddings' not in result.stdout:path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:continue


def gpu_snapshot(host):
    script="""nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
export XDG_RUNTIME_DIR=/run/user/$(id -u)
for unit in $(systemctl --user list-units 'nmr-*' --state=active --no-legend | awk '{print $1}'); do
 for entry in $(systemctl --user show "$unit" -p Environment --value); do
  case "$entry" in CUDA_VISIBLE_DEVICES=*) printf 'RESERVED:%s\\n' "${entry#CUDA_VISIBLE_DEVICES=}";; esac
 done
done
"""
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',host,'bash','-s'],input=script,capture_output=True,text=True,timeout=12)
        blocked=set();readings=[];seen=set()
        for line in result.stdout.splitlines():
            if line.startswith('RESERVED:'):
                blocked.update(int(x) for x in line.split(':',1)[1].split(',') if x.isdigit());continue
            fields=[x.strip() for x in line.split(',')]
            if len(fields)==3 and all(x.isdigit() for x in fields):
                device,free,util=map(int,fields);seen.add(device);readings.append(dict(gpu=device,free_mib=free,utilization=util))
                if free<6000 or util>10:blocked.add(device)
        return host,blocked,readings,seen
    except subprocess.TimeoutExpired:return host,set(),[],set()


def admission_hosts(probes,owned,active,snapshots):
    """Honor active study allocations and all managed GPU reservations."""
    occupied={r['host'] for unit,r in owned.items() if unit in active and r.get('gpus')}
    result=[]
    for host in probes:
        blocked,_,seen=snapshots.get(host.name,(set(),[],set()))
        busy=set(host.busy_gpus)|blocked|(set(range(host.gpus))-seen)
        if host.name in occupied:busy=set(range(host.gpus))
        result.append(replace(host,busy_gpus=tuple(sorted(busy))))
    return result


def worker_command(model,horizon,smoke):
    env=[f'TSFM_EMBED_ROOT={ROOT}',f'TSFM_EMBED_SCR={SCR}/t{horizon}',
         f'TSFM_EMBED_TARGET=target_binned_return_{horizon}',f'TSFM_EMBED_WALL={WALLS[horizon]}','E206_BATCH=4']
    args=['env',*env,E.VENVS[model],'-u','-m','src.data_mani.e212_signal_embeddings','--model',model,'--device','cuda:0']
    if smoke:args.append('--smoke')
    return shlex.join(args)


def extraction_progress(config):
    prepared=json.loads((ROOT/'PREPARED.json').read_text());progress=[]
    for h in WALLS:
        expected=prepared['origin_counts'][str(h)]['eligible_origins_per_model']
        for model in config['models']:
            directory=extraction_dir(h,model);completed=0;seconds=0.;assets=0
            for path in directory.glob('asset_*.json'):
                if not path.stem.removeprefix('asset_').isdigit():continue
                record=json.loads(path.read_text());completed+=record.get('rows',0);seconds+=record.get('seconds',0);assets+=1
            partial=0;partial_seconds=0.
            for path in directory.glob('asset_*.progress.json'):
                if path.with_name(path.name.replace('.progress.json','.npz')).exists():continue
                record=json.loads(path.read_text());partial+=record.get('completed',0);partial_seconds+=record.get('elapsed_seconds',0)
            sample_rows=completed+partial
            rate=(seconds+partial_seconds)/sample_rows if sample_rows else None
            smoke=directory/'SMOKE_COMPLETE.json'
            if rate is None and smoke.exists():rate=json.loads(smoke.read_text())['seconds']/12
            progress.append(dict(model=model,training_target=h,completed_assets=assets,completed_origins=completed,
                active_cached_origins=partial,total_origins=expected,seconds_per_origin=rate,
                remaining_gpu_hours=(max(0,expected-sample_rows)*rate/3600) if rate is not None else None,
                estimate='provisional observed-device timing; queue wait/availability may change'))
    save(ROOT/'extraction_progress.json',progress)
    return progress


def confirm_unit(host,unit):
    """None means unknown, never permission to launch a duplicate worker."""
    command='export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user show '+shlex.quote(unit)+' -p ActiveState -p LoadState'
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',host,command],capture_output=True,text=True,timeout=8)
    except subprocess.TimeoutExpired:return None
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if fields.get('ActiveState') in ['active','activating','reloading']:return True
    if fields.get('ActiveState') in ['inactive','failed'] or fields.get('LoadState')=='not-found':return False
    return None


def supervise(dataset=DATASET,models=None):
    prepare(dataset,models);config=verify();p=pd.read_parquet(ROOT/'panel.parquet')
    path=ROOT/'supervisor.json';state=json.loads(path.read_text()) if path.exists() else dict(owned={},attempts={},launches=[])
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        while not (ROOT/'COMPLETE.json').exists():
            probes=[h for h in hosts.reachable(hosts.probe_all(timeout=4)) if h.name not in {'westeros','imeusp'}]
            statuses=manager.status(names=[h.name for h in probes],timeout=4);active,failed=E.unit_states(statuses)
            recover_locks()
            for unit in list(state['owned']):
                if unit not in active:
                    confirmed=confirm_unit(state['owned'][unit]['host'],unit)
                    if confirmed is False:del state['owned'][unit]
                    else:active.add(unit)
            gpu_hosts=[h for h in probes if h.gpus]
            with ThreadPoolExecutor(max_workers=max(1,len(gpu_hosts))) as pool:
                snapshots={name:(blocked,readings,seen) for name,blocked,readings,seen in pool.map(gpu_snapshot,[h.name for h in gpu_hosts])}
            jobs=[];stage='gpu_smokes';ready=(ROOT/'EMBEDDINGS_READY.json').exists()
            if not ready:
                pending_smokes=[(h,m) for h in WALLS for m in config['models'] if not (extraction_dir(h,m)/'SMOKE_COMPLETE.json').exists()]
                if pending_smokes:
                    for h,m in pending_smokes:jobs.append(budget.Job(f'e212_gpu_smoke_{m}_t{h}',worker_command(m,h,True),cores=2,ram_gb=16,gpus=1))
                else:
                    stage='gpu_extracting';completed=[]
                    for h in WALLS:
                        for m in config['models']:
                            if extraction_complete(p,h,m):completed.append(f'{m}_t{h}');continue
                            jobs.append(budget.Job(f'e212_gpu_extract_{m}_t{h}',worker_command(m,h,False),cores=2,ram_gb=24,gpus=1))
                    state['completed_extractions']=completed
                    if len(completed)==len(config['models'])*2:collect();ready=True
            if ready:
                stage='fitting_heads'
                for h in WALLS:
                    for model in config['models']+['feature_ridge']:
                        if (ROOT/'fits'/f'{model}_t{h}'/'COMPLETE.json').exists():continue
                        jobs.append(budget.Job(f'e212_head_{model}_t{h}',shlex.join([PYTHON,'-u','experiments/e212_volatility99_tsfm.py','fit','--model',model,'--target',str(h)]),cores=4,ram_gb=32))
                if not jobs:
                    # Scoring is CPU work and receives a separately pinned allocation.
                    stage='scoring'
                    jobs=[budget.Job('e212_score',shlex.join([PYTHON,'-u','experiments/e212_volatility99_tsfm.py','score']),cores=4,ram_gb=24)]
            claim_path=ROOT/'external_head_claims.json'
            external=set(json.loads(claim_path.read_text())['jobs']) if claim_path.exists() else set()
            pending=[]
            for job in jobs:
                if job.name in external:continue
                unit='nmr-'+job.name
                if unit in active or unit in state['owned']:continue
                if state['attempts'].get(job.name,0)>=3:
                    save(ROOT/'terminal_failures.json',dict(job=job.name,attempts=3,log=f'logs/cluster/{job.name}.out',pool_unchanged=True))
                    raise RuntimeError('Three attempts exhausted: '+job.name+'; inspect its log; no stream silently removed')
                pending.append(job)
            placements,unplaced=budget.distribute(pending,admission_hosts(probes,state['owned'],active,snapshots),share=budget.SHARE,gpu_slots=1)
            waiting=[dict(job=job.name,reason=reason) for job,reason in unplaced]
            for placement in placements:
                subprocess.run(['ssh','-o','BatchMode=yes',placement.host,'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],capture_output=True,timeout=15)
                ok,message=manager.launch(placement)
                if ok:
                    record=dict(host=placement.host,unit=placement.unit,cpus=list(placement.cpus),gpus=list(placement.gpus),time=time.time())
                    state['owned'][placement.unit]=record;state['launches'].append(record)
                    state['attempts'][placement.job.name]=state['attempts'].get(placement.job.name,0)+1
                print('LAUNCH',ok,message,flush=True)
            progress=extraction_progress(config)
            state.update(extraction_progress=progress,waiting=waiting,stage=stage,updated=time.time(),gpu_snapshots={h:readings for h,(_,readings,_) in snapshots.items()})
            save(path,state);save(ROOT/'status.json',dict(stage=stage,active_study_units=state['owned'],waiting=waiting,completed_extractions=state.get('completed_extractions',[]),extraction_progress=progress,time=time.time()))
            time.sleep(30)
        # Completion covers validated reports; delivery is separately retryable.
        if not (ROOT/'email_delivery.json').exists():email_report()
    except Exception as exc:
        save(ROOT/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        for host in {r['host'] for r in state['owned'].values()}:
            manager.stop(host,[unit for unit,r in state['owned'].items() if r['host']==host and unit.startswith('nmr-e212_')],timeout=20)
        save(ROOT/'resource_lease.json',dict(default_share=budget.SHARE,status='restored',global_defaults_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','supervise','fit','score','email'])
    parser.add_argument('--dataset',default=DATASET);parser.add_argument('--models',nargs='+',choices=MODELS)
    parser.add_argument('--model',choices=MODELS+['feature_ridge']);parser.add_argument('--target',type=int,choices=[20,60])
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if args.command=='fit':fit_stream(args.model,args.target)
    elif args.command=='score':score()
    elif args.command=='email':email_report()
    elif args.command=='prepare':prepare(args.dataset,args.models)
    else:
        with open(ROOT/'supervisor.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            supervise(args.dataset,args.models)
