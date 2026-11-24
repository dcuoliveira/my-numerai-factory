"""E206 durable supervisor and forecasting/portfolio pipeline. No notifications."""
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
import argparse
import hashlib
import json
import signal
import socket
import subprocess
import time
import traceback
import numpy as np
import pandas as pd

from src.analysis import tsfm_momentum_study as N
from src.live.submit import rank_normalise
from src.analysis.asset_momentum import route_slots
from e150_finish_scoring import matrix_scores

ROOT = Path(os.environ.get('E206_SCOPE_ROOT',str(REPO/'results/reports/crypto/e206_momentum3_tsfm')))
SCR = Path(os.environ.get('E206_SCOPE_SCR','/scratch/danielco/e206_momentum3_tsfm'))
DATA = REPO/'data/crypto/v2.0-e205-ts-momentum3'
TARGET = 'target_binned_return_20'
RUN = os.environ.get('E206_SCOPE_RUN','r01')
PYTHON = '/home/danielco/miniconda3/bin/python3'
VENVS = {'timestral':'/scratch/danielco/tsfm2/venv_tt/bin/python',
         't0':'/scratch/danielco/tsfm2/venv_tt/bin/python',
         'chronos2':'/scratch/danielco/chronos2/venv/bin/python',
         'tabpfnts':'/scratch/danielco/tsfm/envs/tabpfnts/bin/python'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2, default=str)+'\n'); tmp.replace(path)


def parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial'); frame.to_parquet(tmp, index=False); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda:f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def config_id(model, norm, design):
    study=os.environ.get('E206_SCOPE_ID','e206')
    return f'{study}_crypto-v2_0-e205-ts-momentum3_{model}_{norm}_{design}_{RUN}'


def prepare():
    if (ROOT/'PREPARED.json').exists():
        coverage_reports(pd.read_parquet(ROOT/'panel.parquet'))
        return
    ROOT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA/'manifest.json').read_text())
    for name, digest in source['output_hashes'].items():
        assert sha(DATA/name) == digest, name
    frame = pd.read_parquet(DATA/'train.parquet').reset_index().sort_values(['date','ucid']).reset_index(drop=True)
    coverage = pd.read_parquet(DATA/'coverage.parquet').reset_index()
    frame = frame.merge(coverage[['symbol','date','covered']],on=['symbol','date'],how='left',validate='1:1')
    meta = pd.read_parquet(DATA/'historical_meta_models.parquet').reset_index()
    frame = frame.merge(meta.rename(columns={'prediction':'meta_model'}),on=['symbol','date'],how='left',validate='1:1')
    assert frame.date.max() < pd.Timestamp('2026-02-20') and frame.covered.notna().all()
    assert not frame.duplicated(['ucid','date']).any()
    features = [c for c in frame if c.startswith('feature_')]; assert len(features)==33
    frame['row'] = np.arange(len(frame))
    dates = np.sort(frame.date.unique()); assets = np.sort(frame.ucid.unique())
    frame['step'] = np.searchsorted(dates,frame.date)
    frame['asset_position'] = pd.Index(assets).get_indexer(frame.ucid)
    for name, mask in zip(['train','validation','final_fit','test'],N.split_masks(frame.date)):
        frame[name] = mask
    parquet(ROOT/'panel.parquet',frame)
    coverage_reports(frame)
    prices_path = REPO/'results/reports/crypto/e157_volatility_reverse_engineering/inputs/prices.parquet'
    prices = pd.read_parquet(prices_path)
    assert not prices.duplicated(['symbol','date']).any()
    # Prices are evaluation outcomes only; never supplied to ridge selection.
    price = prices.set_index(['symbol','date']).open
    execution = frame.date+pd.Timedelta(days=1)
    next_origins = np.r_[dates[1:],dates[-1]+np.timedelta64(1,'D')]
    exit_dates = pd.to_datetime(next_origins[frame.step])+pd.Timedelta(days=1)
    entry = price.reindex(pd.MultiIndex.from_arrays([frame.symbol,execution])).to_numpy()
    exit_price = price.reindex(pd.MultiIndex.from_arrays([frame.symbol,exit_dates])).to_numpy()
    returns = np.divide(exit_price,entry,out=np.full_like(entry,np.nan),where=entry>0)-1
    returns[(entry<=0)|(exit_price<=0)] = np.nan
    parquet(ROOT/'execution_returns.parquet',pd.DataFrame(dict(row=frame.row,origin=frame.date,
        execution=execution,exit=exit_dates,entry_open=entry,exit_open=exit_price,return_value=returns)))
    configs = [dict(id=config_id(m,n,d),model=m,standardization=n,design=d,run_id=RUN)
               for m in N.MODELS for n in N.NORMS for d in N.DESIGNS]
    save(ROOT/'grid.json',configs)
    save(ROOT/'PREPARED.json',dict(rows=len(frame),features=features,assets=assets.tolist(),
        dates=list(map(str,dates)),dataset_hash=sha(DATA/'train.parquet'),price_source_hash=sha(prices_path),
        panel_hash=sha(ROOT/'panel.parquet'),source_manifest=source,
        temporal_protocol='train2020..2023-11-27; val2024-01-01..2024-11-27; final-fit<=2024-11-27; test2025-01-01..2026-02-19',
        portfolio='next-calendar-day source open to next-origin+1 open; last origin holds one calendar day;10bps absolute traded weight; decile long/gross1 neutral; zero financing/borrow fee assumption',
        historical_limit='The test dates and signal selection were already inspected in E205; reserved for this grid only. Pretrained checkpoints postdate parts of history; pretraining contamination not ruled out.',
        code_hashes={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'src/analysis/tsfm_momentum_study.py',REPO/'src/data_mani/e206_tsfm_embeddings.py']}))
    print('PREPARED',len(frame), 'rows,72 configurations',flush=True)


def coverage_reports(frame):
    frame.groupby('ucid').agg(first_panel_date=('date','min'),last_panel_date=('date','max'),
          panel_rows=('row','size'),signal_covered_rows=('covered','sum')).to_csv(ROOT/'coverage_by_asset.csv')
    frame.groupby('date').agg(panel_assets=('ucid','size'),signal_covered_assets=('covered','sum')).to_csv(ROOT/'coverage_by_date.csv')


def extraction_complete(model,panel):
    wanted = panel.groupby('ucid').covered.any()
    directory=SCR/'embeddings'/model
    return all((directory/f'asset_{a}.npz').exists() for a in wanted.index[wanted])


def terminal(model,norm,design):
    dest=ROOT/'fits'/config_id(model,norm,design)
    return any((dest/name).exists() for name in ['COMPLETE.json','FAILED.json','UNSUPPORTED.json'])


def unit_states(snapshots):
    """systemctl adds .service; Placement.unit deliberately does not."""
    active={r['unit'].removesuffix('.service') for rows in snapshots.values() for r in rows if r['state'].startswith('active')}
    failed={r['unit'].removesuffix('.service') for rows in snapshots.values() for r in rows if r['state'].startswith('failed')}
    return active,failed


def placement_hosts(probes,owned,active):
    """Enforce the scoped GPU share across already-running study placements too."""
    from dataclasses import replace
    occupied={}
    for unit,r in owned.items():
        if unit in active:occupied[r['host']]=occupied.get(r['host'],0)+len(r.get('gpus',[]))
    return [replace(h,busy_gpus=tuple(range(h.gpus))) if occupied.get(h.name,0)>=int(.60*h.gpus) and h.gpus else h for h in probes]


def recover_locks():
    """Verify the owning process on its host before reclaiming an interrupted unit."""
    grouped={}
    for path in (SCR/'embeddings').glob('*/*.lock'):
        try:
            owner=json.loads(path.read_text());grouped.setdefault(owner['host'],[]).append((path,int(owner['pid'])))
        except (ValueError,KeyError,OSError):continue
    for host,entries in grouped.items():
        ids=','.join(str(pid) for _,pid in entries)
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',host,
                 'ps -p '+ids+' -o pid=,args='],capture_output=True,text=True,timeout=10)
            if result.returncode not in [0,1]:continue
            active={int(line.strip().split(None,1)[0]) for line in result.stdout.splitlines()
                    if 'src.data_mani.e206_tsfm_embeddings' in line}
            for path,pid in entries:
                if pid not in active:path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:continue


def collect():
    """Logical [N,T,q] sparse-indexed tensor plus materialized memmap and mask."""
    p=pd.read_parquet(ROOT/'panel.parquet'); shape=json.loads((ROOT/'PREPARED.json').read_text())
    models=[]
    for model in N.MODELS:
        directory=SCR/'embeddings'/model
        if not extraction_complete(model,p):
            continue
        manifest=json.loads((directory/'extractor.json').read_text());q=manifest['width']
        flatpath=directory/'H_rows.npy'
        if not flatpath.exists():
            flat=np.lib.format.open_memmap(directory/'H_rows.tmp.npy',mode='w+',dtype='float16',shape=(len(p),q));flat[:]=np.nan
            for path in directory.glob('asset_*.npz'):
                z=np.load(path); flat[z['rows']]=z['states']
            flat.flush();del flat;(directory/'H_rows.tmp.npy').replace(flatpath)
            flat=np.load(flatpath,mmap_mode='r');mask=np.isfinite(flat).all(axis=1)
            np.save(directory/'valid_rows.npy',mask)
            tensor=np.lib.format.open_memmap(directory/'H.tmp.npy',mode='w+',dtype='float16',shape=(len(shape['assets']),len(shape['dates']),q));tensor[:]=np.nan
            for a,g in p.groupby('asset_position'):
                tensor[a,g.step]=flat[g.row]
            tensor.flush();del tensor;(directory/'H.tmp.npy').replace(directory/'H.npy')
            np.save(directory/'valid.npy',np.isfinite(np.load(directory/'H.npy',mmap_mode='r')).all(axis=2))
            save(directory/'COLLECTED.json',dict(shape=[len(shape['assets']),len(shape['dates']),q],valid_rows=int(mask.sum()),checkpoint=manifest))
        models.append(model)
    if not models:
        raise RuntimeError('No valid model embeddings; cannot fit representations')
    common=p.covered.to_numpy().copy()
    for model in models:
        common &= np.load(SCR/'embeddings'/model/'valid_rows.npy')
    common=N.common_history(p.ucid.to_numpy(),p.date.to_numpy(),common)
    # Snapshot designs require at least two cutoff-available model embeddings.
    counts=pd.Series(common).groupby(p.date).transform('sum').to_numpy()
    common &= counts>=2
    np.save(SCR/'common_rows.npy',common)
    diagnostics=p[['row','ucid','date','covered']].copy()
    for model in models:diagnostics[model+'_embedding_valid']=np.load(SCR/'embeddings'/model/'valid_rows.npy')
    diagnostics['common_training_eligible']=common
    parquet(ROOT/'embedding_coverage.parquet',diagnostics)
    save(ROOT/'EMBEDDINGS_READY.json',dict(models=models,common_rows=int(common.sum())))


def normalized(model,norm,phase,window):
    directory=SCR/'normalizations'/model/norm/f'{phase}_{window}'
    path=directory/'H.npy'
    if path.exists():
        return np.load(path,mmap_mode='r')
    p=pd.read_parquet(ROOT/'panel.parquet'); common=np.load(SCR/'common_rows.npy')
    h=np.load(SCR/'embeddings'/model/'H_rows.npy',mmap_mode='r')
    valid=np.load(SCR/'embeddings'/model/'valid_rows.npy') & p.covered.to_numpy()
    fit=common & p['train' if phase=='selection' else 'final_fit'].to_numpy() & np.isfinite(p[TARGET])
    z,refs=N.normalize(h,p.ucid.to_numpy(),p.date.to_numpy(),valid,fit,norm,window)
    directory.mkdir(parents=True,exist_ok=True)
    # One norm/config job per model/norm guarantees a single writer.
    with open(directory/'H.tmp.npy','wb') as f:np.save(f,z)
    (directory/'H.tmp.npy').replace(path)
    np.savez_compressed(directory/'scaler.npz',**refs)
    save(directory/'manifest.json',dict(model=model,norm=norm,phase=phase,window=window,
         fit_rows=int(fit.sum()),additional_design_scaling='none, consistently for A/B/C',
         immutable_for_test=phase=='final'))
    return np.load(path,mmap_mode='r')


def build_design(model,norm,phase,window,design,p):
    h=normalized(model,norm,phase,window)
    valid=np.isfinite(h).all(axis=1)
    if design=='B':
        x,info=N.distribution_design(h,p.date.to_numpy(),valid,norm)
        valid &= np.isfinite(x).all(axis=1)
        return x,valid,info
    if design=='C':
        train=np.load(SCR/'common_rows.npy') & p.train.to_numpy() & np.isfinite(p[TARGET])
        counts=p.loc[train].groupby('ucid').size(); vocab=np.sort(counts.index[counts>=60].to_numpy())
        kernel=N.concat_kernel(h,p.ucid.to_numpy(),p.step.to_numpy(),valid,vocab,p.step.max()+1)
        return kernel,valid,dict(vocabulary=vocab.tolist(),fill='zero absent blocks; no indicators',
            later_entrants='no context slot; no predicted-asset fit without training history',
            constant_columns='removed exactly by per-asset centered kernel: zero centered training columns have zero coefficient',
            columns=len(vocab)*h.shape[1],own_included_exactly_once=True)
    return h,valid,dict(columns=h.shape[1])


def solve_assets(model,norm,window,design,phase,p,selected=None):
    x,valid,info=build_design(model,norm,phase,window,design,p)
    common=np.load(SCR/'common_rows.npy')
    fit=common & p['train' if phase=='selection' else 'final_fit'].to_numpy() & np.isfinite(p[TARGET])
    query=p['validation' if phase=='selection' else 'test'].to_numpy() & valid
    result=np.full(len(p),np.nan); rows=[]; parameters={}
    for asset,g in p.groupby('ucid',sort=True):
        fi=g.index[fit[g.index] & valid[g.index]].to_numpy()
        vi=g.index[query[g.index]].to_numpy()
        if len(fi)<60 or not len(vi) or (phase=='selection' and np.isfinite(p.loc[vi,TARGET]).sum()<20):
            continue
        if phase=='final' and str(asset) not in selected:
            continue
        if design=='C':
            ft,qt=p.loc[fi,'step'].to_numpy(),p.loc[vi,'step'].to_numpy()
            ktt,kqt=x[np.ix_(ft,ft)],x[np.ix_(qt,ft)]
            dropped=dict(constant_columns='implicit exact centered-kernel removal')
        else:
            keep,dropped=N.drop_constants(x[fi],np.ones(len(fi),bool))
            train=np.asarray(x[fi][:,keep],float);test=np.asarray(x[vi][:,keep],float)
            ktt,kqt=train@train.T,test@train.T
        alphas=N.ALPHAS if phase=='selection' else [selected[str(asset)]['alpha']]
        pred,params=N.dual_ridge(ktt,kqt,p.loc[fi,TARGET].to_numpy(),alphas)
        if phase=='selection':
            target=p.loc[vi,TARGET].to_numpy();good=np.isfinite(target)
            loss=np.mean((pred[good]-target[good,None])**2,axis=0)
            j=int(np.argmin(loss));alpha=float(alphas[j]);mse=float(loss[j]); chosen=pred[:,j]
        else:
            j=0;alpha=float(alphas[0]);mse=float(selected[str(asset)]['mse']);chosen=pred[:,0]
        result[vi]=chosen
        record=dict(asset=int(asset),alpha=alpha,mse=mse,n_train=len(fi),n_query=len(vi),constant_columns=dropped)
        rows.append(record)
        parameters[str(asset)]=dict(alpha=alpha,mse=mse)
        destination=SCR/'ridge_parameters'/config_id(model,norm,design)/f'w{window}_{phase}'/f'asset_{asset}.npz'
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,training_rows=fi,dual_weights=params['dual_weights'][:,j],
            kernel_train_mean=params['kernel_train_mean'],kernel_grand_mean=params['kernel_grand_mean'],intercept=params['intercept'],alpha=alpha)
    return result,rows,parameters,info


def fit_group(model,norm):
    p=pd.read_parquet(ROOT/'panel.parquet')
    for design in N.DESIGNS:
        tag=config_id(model,norm,design);dest=ROOT/'fits'/tag
        if (dest/'COMPLETE.json').exists():continue
        choices=[]
        for window in (N.ROLLING if norm=='rolling' else [126]):
            pred,records,params,info=solve_assets(model,norm,window,design,'selection',p)
            if not records:raise RuntimeError('No eligible validation assets: '+tag)
            # Equal weight per asset: predeclared validation selection only.
            score=float(np.mean([r['mse'] for r in records]))
            choices.append((score,window,records,params,info))
            parquet(dest/f'validation_w{window}.parquet',p.loc[np.isfinite(pred),['row','ucid','date',TARGET]].assign(prediction=pred[np.isfinite(pred)]))
        score,window,records,params,info=min(choices,key=lambda c:(c[0],c[1]))
        save(dest/'validation_selection.json',dict(window=window,mean_asset_mse=score,
             candidate_windows=[dict(window=c[1],mse=c[0]) for c in choices],assets=records,design=info))
        pred,records,_,info=solve_assets(model,norm,window,design,'final',p,params)
        out=p.loc[p.test,['row','symbol','ucid','date',TARGET,'meta_model']].copy()
        out['prediction']=pred[p.test.to_numpy()]
        parquet(dest/'predictions.parquet',out)
        save(dest/'ridge_parameters.json',dict(window=window,assets=records,design=info))
        save(dest/'COMPLETE.json',dict(task=dict(tag=tag,model=model),status='success',
             predictions=int(np.isfinite(out.prediction).sum()),fit_assets=len(records),time=time.time()))
        print('FIT COMPLETE',tag,flush=True)


def momentum_baseline():
    destination=ROOT/'momentum_baseline.parquet'
    if destination.exists():return
    p=pd.read_parquet(ROOT/'panel.parquet');features=[c for c in p if c.startswith('feature_')]
    common=np.load(SCR/'common_rows.npy'); all_pred=np.full(len(p),np.nan); choices=[]
    for asset,g in p.groupby('ucid'):
        tr=g.index[common[g.index]&p.train.to_numpy()[g.index]&np.isfinite(p[TARGET].to_numpy()[g.index])]
        va=g.index[common[g.index]&p.validation.to_numpy()[g.index]&np.isfinite(p[TARGET].to_numpy()[g.index])]
        ft=g.index[common[g.index]&p.final_fit.to_numpy()[g.index]&np.isfinite(p[TARGET].to_numpy()[g.index])]
        te=g.index[p.test.to_numpy()[g.index]&p.covered.to_numpy()[g.index]]
        if len(tr)<60 or len(va)<20 or not len(te):continue
        x=p.loc[tr,features].to_numpy(float);v=p.loc[va,features].to_numpy(float)
        mean=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1
        x=(x-mean)/scale;v=(v-mean)/scale
        pred,_=N.dual_ridge(x@x.T,v@x.T,p.loc[tr,TARGET].to_numpy())
        error=((pred-p.loc[va,TARGET].to_numpy()[:,None])**2).mean(0);alpha=float(N.ALPHAS[error.argmin()])
        x=p.loc[ft,features].to_numpy(float);v=p.loc[te,features].to_numpy(float)
        mean=x.mean(0);scale=x.std(0);scale[scale<N.FLOOR]=1;x=(x-mean)/scale;v=(v-mean)/scale
        pred,params=N.dual_ridge(x@x.T,v@x.T,p.loc[ft,TARGET].to_numpy(),[alpha]);all_pred[te]=pred[:,0]
        dest=SCR/'momentum_parameters'/f'asset_{asset}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,center=mean,scale=scale,training_rows=ft,alpha=alpha,**params)
        choices.append(dict(asset=int(asset),alpha=alpha,validation_mse=float(error.min())))
    mean=p.loc[p.final_fit].groupby('ucid')[TARGET].mean()
    out=p.loc[p.test,['row','symbol','ucid','date',TARGET,'meta_model']].copy()
    out['prediction']=all_pred[p.test.to_numpy()]
    fallback=out.ucid.map(mean).fillna(.5).to_numpy()
    out['baseline']=np.where(np.isfinite(out.prediction),out.prediction,fallback)
    out['baseline_rank']=out.groupby('date').baseline.transform(lambda s:rank_normalise(s.to_numpy()))
    parquet(destination,out);save(ROOT/'momentum_baseline_parameters.json',choices)


def score():
    if os.environ.get('E206_SCOPE_FOLD') and not os.environ.get('E206_SCOPE_AGG'):
        save(ROOT/'COMPLETE.json',dict(status='fold_predictions_complete',time=time.time()))
        return
    from src.analysis.cross_target import newey_west
    (SCR/'holdings').mkdir(parents=True,exist_ok=True)
    momentum_baseline()
    baseline=pd.read_parquet(ROOT/'momentum_baseline.parquet')
    execution=pd.read_parquet(ROOT/'execution_returns.parquet')
    configs=json.loads((ROOT/'grid.json').read_text())
    successful=[c for c in configs if (ROOT/'fits'/c['id']/'COMPLETE.json').exists()]
    predictions={c['id']:pd.read_parquet(ROOT/'fits'/c['id']/'predictions.parquet') for c in successful}
    common=set(baseline.loc[np.isfinite(baseline.prediction),'row'])
    for frame in predictions.values():common &= set(frame.loc[np.isfinite(frame.prediction),'row'])
    rows=[]; summaries=[]
    test_dates=np.sort(baseline.date.unique());assets=np.sort(baseline.ucid.unique())
    durations={d:float((execution[execution.origin==d].exit.iloc[0]-execution[execution.origin==d].execution.iloc[0])/pd.Timedelta(days=1)) for d in test_dates}
    # Fixed test row-set is evaluation metadata, never a learning/selection input.
    save(ROOT/'common_test_coverage.json',dict(rows=len(common),assets=len(assets),n_dates=len(test_dates)))
    for c in successful+[dict(id='momentum_baseline',model='momentum',standardization='asset',design='raw')]:
        name=c['id']; frame=baseline if name=='momentum_baseline' else predictions[name]
        g=baseline.drop(columns=['prediction']).merge(frame[['row','prediction']],on='row',how='left',validate='1:1')
        g=g.merge(execution[['row','return_value']],on='row',how='left',validate='1:1')
        for universe in ['available','common']:
            daily=[];holds={s:[] for s in ['long','neutral']};ret=[];eligible_counts=[]
            for d,day in g.groupby('date',sort=True):
                day=day.sort_values('ucid');valid=np.isfinite(day.prediction)
                if universe=='common':valid &= day.row.isin(common)
                candidate=np.where(valid,day.prediction,np.nan)
                routed=route_slots(day.baseline_rank.to_numpy(),candidate)
                piece=day[['row','ucid','date']].copy();piece['prediction']=candidate;piece['routed']=routed;piece['eligible']=valid.to_numpy()
                parquet(SCR/'portfolios'/name/universe/(str(pd.Timestamp(d).date())+'.parquet'),piece)
                finite=valid & np.isfinite(day[TARGET]) & np.isfinite(day.meta_model)
                if finite.sum()>=10:
                    corr,mmc=matrix_scores(candidate[finite,None],day.loc[finite,TARGET].to_numpy(),day.loc[finite,'meta_model'].to_numpy())
                    base_c,base_m=matrix_scores(day.loc[finite,'baseline_rank'].to_numpy()[:,None],day.loc[finite,TARGET].to_numpy(),day.loc[finite,'meta_model'].to_numpy())
                    daily.append(dict(date=d,n=int(valid.sum()),corr=float(corr[0]) if np.isfinite(corr[0]) else 0.,mmc=float(mmc[0]),
                         rank_ic=N.rank_ic(candidate[finite],day.loc[finite,TARGET].to_numpy()),mse=float(np.mean((candidate[finite]-day.loc[finite,TARGET].to_numpy())**2)),
                         mae=float(np.mean(np.abs(candidate[finite]-day.loc[finite,TARGET].to_numpy()))),baseline_corr=float(base_c[0]),baseline_mmc=float(base_m[0])))
                pos=pd.Index(assets).get_indexer(day.ucid);rr=np.full(len(assets),np.nan);rr[pos]=day.return_value;ret.append(rr)
                eligible_counts.append(int(valid.sum()))
                for side in holds:
                    w=np.zeros(len(assets));w[pos[np.asarray(valid)]]=N.decile_weights(candidate[valid],side)
                    holds[side].append(w)
            scores=pd.DataFrame(daily);parquet(ROOT/'scores'/f'{name}_{universe}.parquet',scores)
            for side,weights in holds.items():
                weights=np.array(weights);returns=N.account_portfolio(weights,np.array(ret));returns['date']=test_dates
                returns['days']=[durations[d] for d in test_dates];returns['eligible_assets']=eligible_counts
                returns['segment_wealth']=(1+returns.net).groupby(returns.segment).cumprod()
                parquet(ROOT/'returns'/f'{name}_{universe}_{side}.parquet',returns)
                np.savez_compressed(SCR/'holdings'/f'{name}_{universe}_{side}.npz',weights=weights,assets=assets,dates=test_dates)
                accountable=returns.accountable.to_numpy() & ((returns.eligible_assets.to_numpy()>=10) | bool(os.environ.get('E206_SCOPE_FULLSAMPLE')))
                r=returns.loc[accountable,'net'].to_numpy();days=returns.loc[accountable,'days'].to_numpy()
                # Calendar-time rate and risk, explicit non-daily intervals.
                rate=float(r.sum()/days.sum()) if len(r) else np.nan
                variance=float(np.square(r-rate*days).sum()/days.sum()) if len(r) else np.nan
                sharpe=rate*np.sqrt(365/max(variance,1e-16)) if len(r) else np.nan
                # Drawdowns computed inside accountable segments only.
                drawdowns=[]
                for _,segment in returns.groupby('segment'):
                    q=segment[segment.accountable & ((segment.eligible_assets>=10) | bool(os.environ.get('E206_SCOPE_FULLSAMPLE')))].net.to_numpy()
                    if len(q):
                        wealth=np.cumprod(1+q);peak=np.maximum.accumulate(np.r_[1,wealth])[1:];drawdowns.append(float((1-wealth/peak).max()))
                summaries.append(dict(tag=name,model=c['model'],standardization=c['standardization'],design=c['design'],universe=universe,side=side,
                    n_dates=len(r),unaccountable_dates=int((~returns.accountable).sum()),rank_ic=float(scores.rank_ic.mean()) if len(scores) else np.nan,
                    corr=float(scores['corr'].mean()) if len(scores) else np.nan,mmc=float(scores.mmc.mean()) if len(scores) else np.nan,
                    mse=float(scores.mse.mean()) if len(scores) else np.nan,mae=float(scores.mae.mean()) if len(scores) else np.nan,
                    mean_net=float(np.mean(r)) if len(r) else np.nan,sharpe=sharpe,max_drawdown=max(drawdowns,default=np.nan),
                    turnover=float(returns.turnover.mean()),cost=float(returns.cost.sum()),mean_coverage=float(np.mean(eligible_counts)),
                    accounting='segment-only if missing held prices; not continuous wealth across gaps'))
            print('SCORE',name,universe,flush=True)
    table=pd.DataFrame(summaries).assign(payout=lambda x:.1*x['corr']+x.mmc)
    table=table.sort_values('payout',ascending=False,kind='stable',na_position='last')
    table['status']='success'
    for c in configs:
        if c['id'] not in predictions:
            directory=ROOT/'fits'/c['id']
            paths=[directory/n for n in ['FAILED.json','UNSUPPORTED.json'] if (directory/n).exists()]
            record=json.loads(paths[0].read_text()) if paths else {'status':'failed','error':'Missing artifact'}
            table=pd.concat([table,pd.DataFrame([dict(tag=c['id'],model=c['model'],standardization=c['standardization'],design=c['design'],status=record.get('status','failed'),error=record.get('error'))])],ignore_index=True)
    table.to_csv(ROOT/'ranked_results.csv',index=False)
    table[(table.universe=='common')&(table.side=='neutral')&(table.model!='momentum')].to_csv(ROOT/'all_model_window_rows.csv',index=False)
    validation=[]
    for c in successful:
        choice=json.loads((ROOT/'fits'/c['id']/'validation_selection.json').read_text())
        validation.append(dict(configuration=c['id'],mean_asset_mse=choice['mean_asset_mse'],window=choice['window']))
    validation=pd.DataFrame(validation).sort_values('mean_asset_mse')
    validation.to_csv(ROOT/'validation_rankings.csv',index=False)
    save(ROOT/'validation_selected_configuration.json',validation.iloc[0].to_dict() if len(validation) else {})
    # Exact equivalence detection preserves every named configuration/result.
    fingerprints={};equiv=[]
    for name,frame in predictions.items():
        key=hashlib.sha256(frame[['row','prediction']].to_numpy().tobytes()).hexdigest()
        if key in fingerprints:equiv.append(dict(configuration=name,equivalent_to=fingerprints[key]))
        else:fingerprints[key]=name
    save(ROOT/'equivalent_configurations.json',equiv)
    pairs=[]
    for c in successful:
        for universe in ['available','common']:
            a=pd.read_parquet(ROOT/'scores'/f'{c["id"]}_{universe}.parquet')
            own=config_id(c['model'],c['standardization'],'A')
            for reference in ['momentum_baseline',own]:
                if reference not in predictions and reference!='momentum_baseline':continue
                b=pd.read_parquet(ROOT/'scores'/f'{reference}_{universe}.parquet')
                pair=a.merge(b,on='date',suffixes=('_a','_b'))
                for metric in ['corr','mmc','rank_ic']:
                    # Momentum control is additionally scored on each candidate's exact
                    # available asset set, rather than compared on a different universe.
                    if reference=='momentum_baseline' and metric in ['corr','mmc']:
                        diff=(a[metric]-a['baseline_'+metric]).to_numpy()
                    else:diff=(pair[metric+'_a']-pair[metric+'_b']).to_numpy()
                    mean,t=newey_west(diff,lag=35)
                    rng=np.random.default_rng(206);n=len(diff)
                    if n:
                        starts=rng.integers(n,size=(2000,int(np.ceil(n/35))));ind=(starts[:,:,None]+np.arange(35))%n
                        ci=np.quantile(diff[ind.reshape(2000,-1)[:,:n]].mean(1),[.025,.975])
                    else:ci=[np.nan,np.nan]
                    pairs.append(dict(configuration=c['id'],universe=universe,reference=reference,metric=metric,
                                      delta=mean,hac35_t=t,ci_low=ci[0],ci_high=ci[1],n=n))
    pd.DataFrame(pairs).to_csv(ROOT/'paired_comparisons.csv',index=False)
    portfolio_pairs=[];shared=[]
    for universe in ['available','common']:
        for side in ['long','neutral']:
            data={name:pd.read_parquet(ROOT/'returns'/f'{name}_{universe}_{side}.parquet') for name in list(predictions)+['momentum_baseline']}
            common_dates=None
            for frame in data.values():
                eligible=set(frame.loc[frame.accountable & ((frame.eligible_assets>=10) | bool(os.environ.get('E206_SCOPE_FULLSAMPLE'))),'date'])
                common_dates=eligible if common_dates is None else common_dates&eligible
            save(ROOT/f'common_accountable_dates_{universe}_{side}.json',list(map(str,sorted(common_dates))))
            for c in successful:
                name=c['id'];a=data[name]
                own=config_id(c['model'],c['standardization'],'A')
                for reference in ['momentum_baseline',own]:
                    if reference not in data:continue
                    b=data[reference]
                    pair=a.merge(b,on='date',suffixes=('_a','_b'))
                    pair=pair[pair.accountable_a & pair.accountable_b & (((pair.eligible_assets_a>=10)&(pair.eligible_assets_b>=10)) | bool(os.environ.get('E206_SCOPE_FULLSAMPLE')))].sort_values('date')
                    if len(pair)<2:continue
                    for metric in ['gross','net']:
                        diff=(pair[metric+'_a']-pair[metric+'_b']).to_numpy();delta,t=newey_west(diff,lag=35)
                        rng=np.random.default_rng(206);n=len(diff);starts=rng.integers(n,size=(2000,int(np.ceil(n/35))));idx=(starts[:,:,None]+np.arange(35))%n
                        ci=np.quantile(diff[idx.reshape(2000,-1)[:,:n]].mean(1),[.025,.975])
                        portfolio_pairs.append(dict(configuration=name,reference=reference,universe=universe,side=side,metric=metric,delta=delta,hac35_t=t,ci_low=ci[0],ci_high=ci[1],n=n))
            for name,frame in data.items():
                q=frame[frame.date.isin(common_dates)]
                shared.append(dict(configuration=name,universe=universe,side=side,n=len(q),mean_net=q.net.mean(),
                                   mean_gross=q.gross.mean(),turnover=q.turnover.mean(),cost=q.cost.sum()))
    pd.DataFrame(portfolio_pairs).to_csv(ROOT/'paired_portfolio_returns.csv',index=False)
    pd.DataFrame(shared).to_csv(ROOT/'common_accountable_portfolio_summary.csv',index=False)
    doc='<html><head><meta charset="utf-8"></head><body><h1>E206 — Momentum3 frozen representations</h1><p>72 pretrained-model/scaler/design configurations; independent per-asset ridge. Validation selects parameters; the entire test grid is reported. Signal selection and historical checkpoint pretraining limit independent confirmation.</p>'
    doc+='<p>Actual source-open return portfolios: decile long and gross-one neutral; execution lag1 calendar day;10bps per absolute traded weight; no borrow/funding charges. Missing held prices produce unaccountable segments, never zero returns. Numerai CORR/MMC are separate binned-target metrics.</p><p>Payout = 0.1 CORR + historical-meta proxy MMC. The recommended summary is sorted by payout descending; payout is a Numerai score, not portfolio return.</p>'
    from src.analysis.tsfm_recommended_report import render
    doc+=render(ROOT,table)
    doc+='</body></html>'
    (ROOT/'E206_momentum3_tsfm.html').write_text(doc)
    try:
        from src.analysis.html_pdf import to_pdf
        to_pdf(str(ROOT/'E206_momentum3_tsfm.html'),'lovelace')
    except Exception as e:save(ROOT/'pdf_error.json',dict(error=str(e)))
    save(ROOT/'audit.json',dict(outcome='PASS WITH CONDITIONS',conditions=[
        'Vendor history and checkpoint pretraining overlap are not certified point-in-time snapshots.',
        'Dataset signals selected in E205 on dates overlapping this reserved-grid test.',
        'Actual-return summaries exclude explicitly flagged missing valuations and restart accounting segments.',
        'Standalone representation comparisons, not marginal value alongside native features; no live deployment.']))
    save(ROOT/'COMPLETE.json',dict(status='success' if len(predictions)==72 else 'completed_with_failures',configurations=72,successful=len(predictions),failed_or_unsupported=72-len(predictions),time=time.time(),
         report=str(ROOT/'E206_momentum3_tsfm.html')))


def supervise():
    global PYTHON
    if os.environ.get('E206_SCOPE_ROOT'):
        import shlex
        PYTHON='env '+' '.join(shlex.quote(k+'='+v) for k,v in os.environ.items() if k.startswith('E206_SCOPE_'))+' '+PYTHON
    """Scheduling loop, bounded retries, physical capacity checks and scoped lease."""
    from src.cluster import hosts,manager,budget
    prepare()
    prior=json.loads((ROOT/'supervisor.json').read_text()) if (ROOT/'supervisor.json').exists() else {}
    stop=[False];owned=prior.get('owned',{});attempts=prior.get('attempts',{});failures=prior.get('failures',{})
    launched=prior.get('all_launches',[dict(unit=u,**r) for u,r in owned.items()])
    def halt(*_):stop[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    save(ROOT/'resource_lease.json',dict(default_share=budget.SHARE,active_share=min(1.,budget.SHARE*2),
         scope='E206 only; global default unmodified',status='active',pid=os.getpid(),host=socket.gethostname(),start=time.time()))
    p=pd.read_parquet(ROOT/'panel.parquet')
    try:
        while not stop[0] and not (ROOT/'COMPLETE.json').exists():
            probes=hosts.reachable(hosts.probe_all(timeout=3))
            snapshots=manager.status(names=[h.name for h in probes],timeout=3)
            live,failed=unit_states(snapshots)
            recover_locks()
            # Recover locks only after their owning study worker is no longer alive.
            for name,record in list(owned.items()):
                if name in live:continue
                if name in failed:
                    failures[name.removeprefix('nmr-')]=failures.get(name.removeprefix('nmr-'),0)+1
                model=record.get('model')
                if record['kind']=='extract' and model:
                    directory=SCR/'embeddings'/model
                    for lock in directory.glob('*.lock'):
                        try:
                            owner=json.loads(lock.read_text())
                            if owner['host']==record['host']:
                                probe=subprocess.run(['ssh','-o','BatchMode=yes',record['host'],'kill -0 '+str(owner['pid'])],capture_output=True,timeout=10)
                                if probe.returncode:lock.unlink(missing_ok=True)
                        except Exception:pass
                del owned[name]
            complete=[m for m in N.MODELS if extraction_complete(m,p)]
            excluded=[m for m in N.MODELS if (ROOT/'model_failures'/f'{m}.json').exists()]
            if len(set(complete+excluded))==4 and not (ROOT/'EMBEDDINGS_READY.json').exists():collect()
            jobs=[]
            if not (ROOT/'EMBEDDINGS_READY.json').exists():
                # Distinct GPU models use separate idle cards. TabPFN also has
                # independent CPU claim workers while waiting for a free GPU.
                for model in N.MODELS:
                    if model in complete or model in excluded:continue
                    if model=='tabpfnts' and any(
                        m not in complete+excluded and not any(r.get('model')==m and r['kind']=='extract' and r.get('gpu') for r in owned.values())
                        for m in ['timestral','t0','chronos2']):
                        continue
                    directory=SCR/'embeddings'/model
                    wanted=p.groupby('ucid').covered.any()
                    if not any(not (directory/f'asset_{a}.npz').exists() and not (directory/f'asset_{a}.lock').exists() for a in wanted.index[wanted]):continue
                    if any(r.get('model')==model and r['kind']=='extract' and r.get('gpu') for r in owned.values()):continue
                    base=f'e206_{RUN}_tsm3_{model}_extract_gpu'
                    if failures.get(base,0)>=3:continue
                    command=f'{VENVS[model]} -u -m src.data_mani.e206_tsfm_embeddings --model {model}'
                    jobs.append((budget.Job(base,command,cores=2,ram_gb=16,gpus=1),dict(kind='extract',model=model,gpu=True)))
                directory=SCR/'embeddings'/'tabpfnts';wanted=p.groupby('ucid').covered.any()
                unclaimed=any(not (directory/f'asset_{a}.npz').exists() and not (directory/f'asset_{a}.lock').exists() for a in wanted.index[wanted])
                if 'tabpfnts' not in complete+excluded and unclaimed:
                    for i in range(8):
                        base=f'e206_{RUN}_tsm3_tabpfnts_extract_cpu{i}'
                        if any(n.endswith(base) for n in live) or failures.get(base,0)>=3:continue
                        command=f'{VENVS["tabpfnts"]} -u -m src.data_mani.e206_tsfm_embeddings --model tabpfnts --device cpu'
                        jobs.append((budget.Job(base,command,cores=2,ram_gb=8),dict(kind='extract',model='tabpfnts',gpu=False)))
            else:
                for model in N.MODELS:
                    for norm in N.NORMS:
                        if all(terminal(model,norm,d) for d in N.DESIGNS):continue
                        base=f'e206_{RUN}_tsm3_{model}_{norm}_ABC'
                        if any(n.endswith(base) for n in live) or failures.get(base,0)>=3:continue
                        jobs.append((budget.Job(base,f'{PYTHON} -u experiments/e206_momentum3_tsfm.py fit --model {model} --norm {norm}',cores=4,ram_gb=32),dict(kind='fit',model=model,norm=norm)))
                if all(terminal(m,n,d) for m in N.MODELS for n in N.NORMS for d in N.DESIGNS):
                    score();break
            if jobs:
                placements,unplaced=budget.distribute([j for j,_ in jobs],placement_hosts(probes,owned,live),share=.60,gpu_slots=None)
                descriptions={j.name:r for j,r in jobs}
                for placement in placements:
                    if placement.unit in live:continue
                    # A failed transient unit needs resetting before reusing its name.
                    subprocess.run(['ssh','-o','BatchMode=yes',placement.host,
                         'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],capture_output=True,timeout=15)
                    ok,message=manager.launch(placement)
                    if ok:
                        attempts[placement.job.name]=attempts.get(placement.job.name,0)+1
                        owned[placement.unit]=dict(descriptions[placement.job.name],host=placement.host,
                            cpus=list(placement.cpus),gpus=list(placement.gpus),command=placement.job.command)
                        if not any(r['host']==placement.host and r['unit']==placement.unit for r in launched):
                            launched.append(dict(unit=placement.unit,**owned[placement.unit]))
                    print('LAUNCH',ok,message,flush=True)
            save(ROOT/'supervisor.json',dict(updated=time.time(),completed_embeddings=complete,owned=owned,attempts=attempts,failures=failures,all_launches=launched,
                 completed_configurations=len(list((ROOT/'fits').glob('*/COMPLETE.json')))))
            save(ROOT/'resources'/f'{int(time.time())}.json',dict(hosts=[h.__dict__ for h in probes],units=snapshots,active_share=.60,default_share=budget.SHARE))
            # Fail explicitly after bounded recovery instead of waiting forever.
            exhausted=[name for name,n in failures.items() if n>=3 and not any(unit.endswith(name) for unit in live|set(owned))]
            if exhausted:
                # Successful finished jobs can legitimately have used retries.
                unresolved=[n for n in exhausted if ('extract' in n and not any(n.startswith(f'e206_{RUN}_tsm3_{m}_') for m in complete)) or ('_ABC' in n and not any(all((ROOT/'fits'/config_id(m,z,d)/'COMPLETE.json').exists() for d in N.DESIGNS) and n==f'e206_{RUN}_tsm3_{m}_{z}_ABC' for m in N.MODELS for z in N.NORMS))]
                for name in unresolved:
                    record=dict(status='failed',error='Three failed attempts exhausted; see logs/cluster/'+name+'.out',time=time.time())
                    if 'extract' in name:
                        model=next(m for m in N.MODELS if name.startswith(f'e206_{RUN}_tsm3_{m}_'))
                        save(ROOT/'model_failures'/f'{model}.json',record)
                        for norm in N.NORMS:
                            for design in N.DESIGNS:save(ROOT/'fits'/config_id(model,norm,design)/'FAILED.json',record)
                        for unit,r in list(owned.items()):
                            if r.get('model')==model:manager.stop(r['host'],[unit],timeout=20);del owned[unit]
                    else:
                        for m in N.MODELS:
                            for n in N.NORMS:
                                if name==f'e206_{RUN}_tsm3_{m}_{n}_ABC':
                                    for d in N.DESIGNS:
                                        if not terminal(m,n,d):save(ROOT/'fits'/config_id(m,n,d)/'FAILED.json',record)
            for _ in range(30):
                if stop[0]:break
                time.sleep(1)
    except Exception as e:
        save(ROOT/'FAILED.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        for host in {r['host'] for r in launched}:
            manager.stop(host,[r['unit'] for r in launched if r['host']==host],timeout=20)
        recover_locks()
        save(ROOT/'resource_lease.json',dict(default_share=budget.SHARE,active_share=budget.SHARE,
             scope='E206 only; global default was never modified',status='restored',time=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','supervise','collect','fit','score']);ap.add_argument('--model',choices=N.MODELS);ap.add_argument('--norm',choices=N.NORMS)
    a=ap.parse_args()
    if a.command=='fit':fit_group(a.model,a.norm)
    else:globals()[a.command]()
