"""E231 TabNet reconstruction audits and exhaustive matched six-model rank blending."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import itertools
import json
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e231_tabnet_ensembles as T
from experiments import e230_models as M
from experiments import e230_dataset as D
from experiments import e212_volatility99_tsfm as S

ROOT=T.ROOT

def aggregate_and_audit(meta,p,indices):
    x=p[meta['features']].to_numpy(float);y=p[D.TARGET].to_numpy(float);new={};checks=[];choices=[]
    plan=T.prepare()
    for path,h in plan['identity']['hashes'].items():assert S.sha(path)==h,path
    for mode in ['asset','panel']:
        units=meta['assets'] if mode=='asset' else ['panel'];pred=np.full(len(p),np.nan);unions={phase:[] for phase in ['train','validation','final_fit','query']}
        for asset in units:
            directory=ROOT/'models'/('asset_'+asset if mode=='asset' else 'panel');record=json.loads((directory/'record.json').read_text());rows=dict(np.load(directory/'rows.npz'));model=joblib.load(directory/'model.joblib');assert S.sha(directory/'model.joblib')==record['model_sha256']
            expected={k:idx if mode=='panel' else idx[p.ucid.to_numpy()[idx]==asset] for k,idx in indices.items() if k in unions}
            for k,v in expected.items():np.testing.assert_array_equal(rows[k],v);unions[k].append(v)
            tr,va,ft,te=[rows[k] for k in ['train','validation','final_fit','query']]
            assert len(tr)>=60 and len(va)>=20 and not set(ft)&set(te)
            assert p.loc[tr,'date'].max()+pd.Timedelta(days=35)<=p.loc[va,'date'].min()
            if len(te):assert p.loc[ft,'date'].max()+pd.Timedelta(days=35)<=p.loc[te,'date'].min()
            assert all(np.isfinite(x[v]).all() for v in rows.values())
            np.testing.assert_allclose(model.named_steps['scale'].mean_,x[ft].mean(0),atol=1e-12)
            np.testing.assert_allclose(model.named_steps['scale'].var_,x[ft].var(0),atol=1e-12)
            assert record['selected']==min(record['choices'],key=lambda v:v['mse']) and len(record['choices'])==4
            for trial in record['choices']+[dict(coverage=record['final_coverage'])]:
                c=trial['coverage'];n=len(ft) if trial.get('parameters') is None else len(tr)
                assert c['epoch_examples']==[n]*40 and c['total_examples']==n*40
                assert c['epochs']==40 and c['minimum_batch']>=2 and not c['drop_last']
                assert c['epoch_batches']==[int(np.ceil(n/c['effective_batch']))]*40
            assert model.named_steps['net']._model.updated_weights==0
            values=np.load(directory/'predictions.npy');assert len(values)==len(te) and np.isfinite(values).all()
            replay=model.predict(x[te[:min(len(te),12)]]) if len(te) else np.empty(0)
            np.testing.assert_allclose(replay,values[:len(replay)],atol=2e-6,rtol=2e-5)
            pred[te]=values;choices.append(dict(arm=mode+'_tabnet',unit=asset,**record['selected']['parameters'],validation_mse=record['selected']['mse'],parameters=record['final_coverage']['parameter_count'],seconds=record['seconds']))
        for phase,parts in unions.items():np.testing.assert_array_equal(np.sort(np.concatenate(parts)),np.sort(indices[phase]))
        assert np.isfinite(pred[indices['query']]).all();assert np.isnan(pred[np.setdiff1d(np.arange(len(p)),indices['query'])]).all()
        arm=mode+'_tabnet';root=ROOT/'fits'/arm;S.parquet(root/'predictions.parquet',p[['row','ucid','symbol','date']].assign(prediction=pred));S.save(root/'COMPLETE.json',dict(task=dict(tag=arm,model='tabnet'),units=len(units),prediction_sha256=S.sha(root/'predictions.parquet')));new[arm]=pred
        checks.append(dict(arm=arm,units=len(units),all_saved_models_replayed=True,all_scalers_final_only=True,all_row_unions_match_e230=True,all_eligible_rows_processed_every_epoch=True))
    pd.DataFrame(choices).to_csv(ROOT/'tabnet_selected_parameters.csv',index=False)
    return new,checks

def finish():
    meta,p,indices=M.load();S.save(ROOT/'status.json',dict(stage='auditing',time=time.time()));new,checks=aggregate_and_audit(meta,p,indices)
    names=list(D.ARMS)+['asset_tabnet','panel_tabnet'];keys=['AR','AM','PR','PM','AT','PT'];values=[]
    for arm in D.ARMS:
        f=pd.read_parquet(D.ROOT/'fits'/arm/'predictions.parquet');pd.testing.assert_frame_equal(p[['row','ucid','symbol','date']],f[['row','ucid','symbol','date']]);values.append(f.prediction.to_numpy())
        S.save(ROOT/'fits'/arm/'COMPLETE.json',dict(task=dict(tag=arm,model=arm.split('_')[1]),frozen_source=str(D.ROOT/'fits'/arm),source_sha256=S.sha(D.ROOT/'fits'/arm/'predictions.parquet')))
    values.extend([new['asset_tabnet'],new['panel_tabnet']]);rows=indices['score'];base=p.loc[rows].reset_index(drop=True);x=np.column_stack(values)[rows];assert np.isfinite(x).all()
    subsets=[(j,) for j in range(6)]+[s for n in range(2,7) for s in itertools.combinations(range(6),n)];assert len(subsets)==63
    membership=np.zeros((6,len(subsets)),np.int64);records=[]
    for j,subset in enumerate(subsets):
        membership[list(subset),j]=1;records.append(dict(id=keys[j] if j<6 else f'B{j-5:03d}',kind='Standalone' if j<6 else 'Ensemble',members='+'.join(keys[k] for k in subset),models='+'.join(names[k] for k in subset),size=len(subset),column=j,tabnet_members=sum(k>=4 for k in subset)))
    sizes=membership.sum(0);groups=list(base.groupby('date',sort=True).indices.items());dates=pd.DatetimeIndex([d for d,_ in groups]);scores=np.empty((len(groups),len(subsets),2));native=[];parity=[]
    for d,(date,idx) in enumerate(groups):
        nums=(2*rankdata(x[idx],axis=0,method='average')-1).astype(np.int64);blends=S.rank_average_blends(nums,membership,sizes)
        y=base[D.TARGET].iloc[idx].to_numpy();m=base.meta_model.iloc[idx].to_numpy();corr,mmc=S.matrix_scores(blends,y,m);scores[d,:,0]=np.nan_to_num(corr,nan=0.);scores[d,:,1]=mmc
        if d in [0,len(groups)//2,len(groups)-1]:
            direct=np.column_stack([nums[:,list(s)].sum(1)/(2.*len(idx)*len(s)) for s in subsets]);np.testing.assert_array_equal(blends,direct)
            nc=numerai_corr(pd.DataFrame(direct),pd.Series(y,name='target')).fillna(0).to_numpy();nm=correlation_contribution(pd.DataFrame(direct),pd.Series(m,name='meta'),pd.Series(y,name='target')).to_numpy();np.testing.assert_allclose(scores[d,:,0],nc,atol=2e-7,rtol=2e-6);np.testing.assert_allclose(mmc,nm,atol=1e-12,rtol=1e-10);native.append(dict(date=str(date),max_corr_error=float(abs(scores[d,:,0]-nc).max()),max_mmc_error=float(abs(mmc-nm).max())))
    assert np.isfinite(scores).all();pay=.1*scores[:,:,0]+scores[:,:,1];np.save(ROOT/'daily_corr_mmc.npy',scores)
    prior=pd.read_csv(D.ROOT/'daily_payout.csv',index_col=0,parse_dates=True);np.testing.assert_array_equal(dates.to_numpy(),prior.index.to_numpy());np.testing.assert_allclose(pay[:,:4],prior[D.ARMS],atol=1e-12,rtol=1e-10)
    table=pd.DataFrame(records)
    for metric in ['corr','mmc','payout','sharpe.p','maxdd.p']:table[metric]=[S.payout_metrics(scores[:,j,0],scores[:,j,1])[metric] for j in range(len(subsets))]
    table['cumulative_payout']=pay.sum(0);table=table.sort_values(['payout','id'],ascending=[False,True],kind='stable');table.to_csv(ROOT/'all_results.csv',index=False);stand=table[table.kind.eq('Standalone')];stand.to_csv(ROOT/'standalone_results.csv',index=False)
    daily=pd.DataFrame(pay,index=dates,columns=[r['id'] for r in records]);daily.index.name='date';daily.to_csv(ROOT/'daily_payout.csv');daily.cumsum().to_csv(ROOT/'cumulative_payout.csv')
    S.parquet(ROOT/'standalone_forecasts.parquet',base[['row','ucid','symbol','date',D.TARGET,'meta_model']].join(pd.DataFrame(x,columns=keys)));S.save(ROOT/'trials.json',records);S.save(ROOT/'model_mapping.json',dict(zip(keys,names)))
    for row in table[table.kind.eq('Ensemble')].itertuples():S.save(ROOT/'fits'/row.id/'COMPLETE.json',dict(task=dict(tag=row.id,model='rank_ensemble'),members=row.members))
    ensemble=table[table.kind.eq('Ensemble')];top=ensemble.head(3);all6=ensemble[ensemble['size'].eq(6)];shown=pd.concat([top,all6,stand]).drop_duplicates('id');shown.to_csv(ROOT/'comparison_table.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'AR':'Ridge per asset','AM':'MLP per asset','PR':'Pooled ridge','PM':'Pooled MLP','AT':'TabNet per asset','PT':'Pooled TabNet'}
    for add in [False,True]:
        fig,ax=plt.subplots(figsize=(13,7),layout='constrained');selected=shown if add else stand
        for row in selected.itertuples():ax.plot(dates,daily[row.id].cumsum(),label=labels[row.id] if row.kind=='Standalone' else row.id+': '+row.members,lw=2.3 if row.kind=='Ensemble' else 1.5,ls='--' if add and row.kind=='Standalone' else '-')
        if add:ax.plot(dates,prior.E229_best_mixed.cumsum(),label='E229 mixed reference',color='black',ls=':',lw=2)
        ax.set_title('E231: '+('top ensembles and six models' if add else 'six standalone models on feature134'));ax.set_ylabel('Additive cumulative target20 payout score');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.savefig(ROOT/('ensemble_comparison.png' if add else 'standalone_comparison.png'),dpi=160);plt.close(fig)
    for path,h in T.prepare()['identity']['hashes'].items():assert S.sha(path)==h,path
    audit=dict(outcome='PASS WITH CONDITIONS',tabnet_checks=checks,exact_e230_keys_and_masks=True,original_four_scores_match_e230=True,native_score_checks=native,all_subsets_integer_reconstruction=True,source_hashes_unchanged=True,dates=len(groups),asset_dates=len(base),assets=int(base.ucid.nunique()),conditions=meta['conditions']+['TabNet40-epoch four-candidate search is bounded and does not establish optimal compute.','Ensemble selection uses displayed development history; no independent confirmation.'])
    S.save(ROOT/'audit.json',audit)
    def display(df):
        show=df[['id','members','corr','mmc','payout','sharpe.p','maxdd.p','cumulative_payout']].rename(columns={'id':'ID','members':'Members','corr':'CORR','mmc':'MMC','payout':'Mean payout','sharpe.p':'Sharpe','maxdd.p':'Max DD','cumulative_payout':'Cumulative'})
        return show.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body=f'<h1>E231 — TabNet and six-model ensembles on feature134</h1><p>Added independent per-asset TabNet and pooled-panel TabNet to the four frozen E230 ridge/MLP models. All use the same134 original features,226 pretest-eligible assets and identical train/validation/final/query masks. Selection training ends2023-11-27; validation2024-01-01..2024-11-27; fresh final fitting ends2024-11-27;35-day label walls. Pooled models use asset-date rows with no asset-ID input.</p><p>Scoring:{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d};{len(groups)} dates;{len(base):,} asset-dates;{base.ucid.nunique()} distinct assets, exactly E230. Payout=0.1CORR+historical-meta proxy MMC. Cumulative scores are additive; Sharpe uses population daily standard deviation, unannualized. Scores are not investment returns.</p>'
    body+='<h2>Six standalone models</h2><p>'+ '; '.join(k+' = '+v for k,v in labels.items())+'</p>'+display(stand)+'<img src="standalone_comparison.png">'
    body+='<h2>Top ensembles and all-six blend</h2><p>Every equal-weight within-date average prediction-rank subset of size2–6:57 ensembles, plus6 standalones. Exact integer midrank sums preserve ties. Blended predictions are scored directly; member payouts are not averaged. Frozen E229 reference is context only and is excluded from this ensemble pool. No new residualization. Winners are selected on the displayed history and remain exploratory.</p>'+display(pd.concat([top,all6]).drop_duplicates('id'))+'<img src="ensemble_comparison.png"><h2>Top ten ensembles</h2>'+display(ensemble.head(10))
    body+='<h2>TabNet tuning and checks</h2><p>Four candidates: tied decision/attention widths8 or16 x sparse penalties0.001 or0.1;3 decision steps,Adam lr0.01,40 fixed epochs,seed231. No eval_set or early stopping. Each asset chooses by earlier validation MSE independently; pooled selection occurs once. Final weights and scaler are fitted fresh on final history. Batch capped256 per asset/1024 panel and available rows; singleton remainders avoided by adjusting batch; virtual batch capped128. No final rows dropped. Actual optimizer batch counts confirm every eligible example is processed in every epoch. Forty epochs is a bounded search, not a claim of optimal training.</p><p>Every saved TabNet and final scaler replayed; row unions, label walls, hyperparameter selection, source hashes, original E230 score parity, independent subset sums and native scoring checks pass. PASS WITH CONDITIONS. Full configurations, source hashes and detailed audits are separate artifacts. Original E230 forecasts, dataset and report remain unchanged.</p><p>Historical vendor-vintage/coverage restrictions and prior feature selection remain inherited. No native22 marginal-feature, full-universe or live prediction claim. E230 ridge grid-boundary limits remain unchanged. E229 reference retains its own fitting schedules and representations.</p>'
    page=ROOT/'E231_tabnet_six_model_ensembles.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:9px;break-inside:avoid!important;page-break-inside:avoid!important}td,th{border:1px solid #ccc;padding:4px}img{display:block;width:100%;height:auto;margin:auto;break-inside:avoid}td:nth-child(n+3){white-space:nowrap}</style></head><body>'+body+'</body></html>');subprocess.run([S.PYTHON,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=300)
    S.save(ROOT/'COMPLETE.json',dict(experiment='E231',status='success',standalone_models=6,ensembles=57,report=str(page.with_suffix('.pdf')),pdf_sha256=S.sha(page.with_suffix('.pdf')),audit='PASS WITH CONDITIONS',time=time.time()));S.save(ROOT/'status.json',dict(stage='complete',time=time.time()));print(shown.to_string(index=False),flush=True)
