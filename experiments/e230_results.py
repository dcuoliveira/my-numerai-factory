"""E230 matched covered-cohort scoring, audits and result report."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e230_dataset as D
from experiments import e230_models as M
from experiments import e212_volatility99_tsfm as S


def audit_models(meta,p,indices):
    x=p[meta['features']].to_numpy(float);y=p[D.TARGET].to_numpy(float);checks=[];counts={};parameter_choices=[];ridge_checks=[]
    for arm in D.ARMS:
        root=D.ROOT/'fits'/arm;complete=json.loads((root/'COMPLETE.json').read_text());assert complete['identity']['models_sha256']==S.sha(M.__file__)
        assert S.sha(root/'predictions.parquet')==complete['prediction_sha256']
        files=sorted((root/'models').glob('*/record.json'));assert len(files)==complete['models'];counts[arm]=len(files)
        unions={phase:[] for phase in ['train','validation','final_fit','query']}
        model_replays=0
        for n,file in enumerate(files):
            directory=file.parent;record=json.loads(file.read_text());rows=dict(np.load(directory/'rows.npz'));modelpath=directory/('model.npz' if arm.endswith('ridge') else 'model.joblib');assert S.sha(modelpath)==record['model_sha256']
            for phase,ids in rows.items():
                assert len(ids)==len(np.unique(ids))
                assert set(ids)<=set(indices[phase]);unions[phase].append(ids)
                assert p.loc[ids,'covered'].all() and p.loc[ids,'asset_eligible'].all()
                assert np.isfinite(x[ids]).all()
            tr,va,ft,te=[rows[k] for k in ['train','validation','final_fit','query']]
            assert len(tr)>=60 and len(va)>=20
            assert p.loc[tr,'date'].max()+pd.Timedelta(days=35)<=p.loc[va,'date'].min()
            if len(te):assert p.loc[ft,'date'].max()+pd.Timedelta(days=35)<=p.loc[te,'date'].min()
            assert not set(ft)&set(te)
            if arm.startswith('asset'):
                assert len(p.loc[ft,'ucid'].unique())==1
                assert set(p.loc[np.r_[tr,va,te],'ucid'])<=set(p.loc[ft,'ucid'])
            chosen=min(record['choices'],key=lambda r:r['mse']);assert chosen==record['selected']
            # Check scaler fit references and a serialized model replay for every fitted unit.
            query=te[:min(5,len(te))]
            if arm.endswith('ridge'):
                model=np.load(modelpath);center=x[ft].mean(0);scale=x[ft].std(0);scale[scale<1e-8]=1
                np.testing.assert_allclose(model['center'],center,atol=1e-13);np.testing.assert_allclose(model['scale'],scale,atol=1e-13)
                prediction=(((x[query]-model['center'])/model['scale']-model['shift'])@model['weights']+model['intercept'])[:,0]
                if n in {0,len(files)//2,len(files)-1}:
                    scaler_mean=x[tr].mean(0);scaler_std=x[tr].std(0);scaler_std[scaler_std<1e-8]=1
                    tx=(x[tr]-scaler_mean)/scaler_std;vx=(x[va]-scaler_mean)/scaler_std
                    losses=[]
                    for choice in record['choices']:
                        z=Ridge(alpha=choice['alpha'],solver='cholesky').fit(tx,y[tr]).predict(vx)
                        losses.append(float(np.mean((z-y[va])**2)))
                    np.testing.assert_allclose(losses,[v['mse'] for v in record['choices']],atol=1e-9,rtol=1e-6)
                    assert int(np.argmin(losses))==int(np.argmin([v['mse'] for v in record['choices']]))
                    z=Ridge(alpha=record['selected']['alpha'],solver='cholesky').fit((x[ft]-center)/scale,y[ft]).predict((x[query]-center)/scale) if len(query) else np.empty(0)
                    np.testing.assert_allclose(z,prediction,atol=1e-9,rtol=1e-6)
                    ridge_checks.append(dict(arm=arm,unit=directory.name,max_validation_mse_error=float(np.max(np.abs(np.array(losses)-np.array([v['mse'] for v in record['choices']])))),final_prediction_error=float(np.max(np.abs(z-prediction))) if len(query) else 0.))
            else:
                model=joblib.load(modelpath);net=model.named_steps['net'];scaler=model.named_steps['scale']
                assert net.activation=='relu' and net.n_iter_==100 and not net.early_stopping
                assert net.n_iter_no_change==101 and net.random_state==230
                np.testing.assert_allclose(scaler.mean_,x[ft].mean(0),atol=1e-12)
                np.testing.assert_allclose(scaler.var_,x[ft].var(0),atol=1e-12)
                prediction=model.predict(x[query]) if len(query) else np.empty(0)
                params=record['selected']['parameters'];size=M.dense_parameters(params['hidden_layer_sizes'])
                assert record['final_parameters_count']==size
                assert record['final_examples_seen']==100*len(ft)
                assert record['final_estimated_flops']==6*size*100*len(ft)
                assert len(record['choices'])==4 and all(c['epochs']==100 for c in record['choices'])
                parameter_choices.append(dict(arm=arm,unit=directory.name,width='x'.join(map(str,params['hidden_layer_sizes'])),alpha=params['alpha'],parameters=size,final_examples=record['final_examples_seen'],estimated_flops=record['final_estimated_flops'],search_seconds=sum(v['seconds'] for v in record['choices']),final_seconds=record['final_seconds']))
            np.testing.assert_allclose(prediction,np.load(directory/'predictions.npy')[:len(query)],atol=1e-12,rtol=1e-10);model_replays+=1
        for phase,parts in unions.items():np.testing.assert_array_equal(np.sort(np.concatenate(parts)),np.sort(indices[phase]))
        checks.append(dict(arm=arm,units=counts[arm],all_serialized_units_replayed=model_replays,exact_union_rows_all_phases=True,all_selection_and_final_scalers_causal=True))
    for path,h in meta['input_hashes'].items():assert S.sha(path)==h,path
    for path,h in meta['output_hashes'].items():assert S.sha(path)==h,path
    pd.DataFrame(parameter_choices).to_csv(D.ROOT/'mlp_selected_parameters.csv',index=False)
    return dict(outcome='PASS WITH CONDITIONS',model_checks=checks,ridge_sklearn_checks=ridge_checks,source_and_dataset_hashes_unchanged=True,no_imputation=True,all_finite_training_and_query_inputs=True,identical_training_validation_final_query_unions=True,score_sample_fixed_before_fitting=True,conditions=meta['conditions'])


def finish():
    meta,p,indices=M.load();S.save(D.ROOT/'status.json',dict(stage='auditing',time=time.time()))
    assert all((D.ROOT/'fits'/a/'COMPLETE.json').exists() for a in D.ARMS)
    audit=audit_models(meta,p,indices)
    rows=indices['score'];base=p.loc[rows].reset_index(drop=True);names=list(D.ARMS);values=[];hashes={}
    for arm in D.ARMS:
        path=D.ROOT/'fits'/arm/'predictions.parquet';hashes[str(path)]=S.sha(path);f=pd.read_parquet(path)
        pd.testing.assert_frame_equal(p[['row','ucid','symbol','date']],f[['row','ucid','symbol','date']]);values.append(f.prediction.to_numpy()[rows])
    source=D.E228/'selected_forecasts.parquet';hashes[str(source)]=S.sha(source);reference=pd.read_parquet(source);reference.ucid=reference.ucid.astype(str)
    idx=pd.MultiIndex.from_frame(reference[['ucid','date']]).get_indexer(pd.MultiIndex.from_frame(base[['ucid','date']]));assert (idx>=0).all();reference=reference.iloc[idx].reset_index(drop=True)
    np.testing.assert_array_equal(base.symbol,reference.symbol)
    members=[f'{f}{i}' for f in ['M','V'] for i in range(1,6)]
    # Freeze original E229 member choices; rebuild ranks only on this exact matched scoring cohort.
    pool=reference[members].to_numpy(float);assert np.isfinite(pool).all()
    values.extend([reference.M1.to_numpy(),reference.V1.to_numpy()]);names.extend(['E228_M1','E228_V1'])
    x=np.column_stack(values);blend=np.full((len(base),2),np.nan);groups=list(base.groupby('date',sort=True).indices.items())
    for date,ix in groups:
        numerators=(2*rankdata(pool[ix],axis=0,method='average')-1).astype(np.int64)
        blend[ix,0]=numerators[:,[0,1,2,5,7,9]].sum(1)/(2.*len(ix)*6)
        blend[ix,1]=numerators.sum(1)/(2.*len(ix)*10)
    x=np.column_stack([x,blend]);names.extend(['E229_best_mixed','E229_all10']);assert np.isfinite(x).all()
    scores=np.empty((len(groups),len(names),2));native_checks=[]
    for d,(date,ix) in enumerate(groups):
        y=base[D.TARGET].iloc[ix].to_numpy();m=base.meta_model.iloc[ix].to_numpy();corr,mmc=S.matrix_scores(x[ix],y,m);scores[d,:,0]=np.nan_to_num(corr,nan=0.);scores[d,:,1]=mmc
        if d in (0,len(groups)//2,len(groups)-1):
            nc=numerai_corr(pd.DataFrame(x[ix],columns=names),pd.Series(y)).fillna(0).to_numpy();nm=correlation_contribution(pd.DataFrame(x[ix],columns=names),pd.Series(m),pd.Series(y)).to_numpy();np.testing.assert_allclose(scores[d,:,0],nc,atol=2e-7,rtol=2e-6);np.testing.assert_allclose(mmc,nm,atol=1e-12,rtol=1e-10)
            native_checks.append(dict(date=str(date),max_corr_error=float(abs(scores[d,:,0]-nc).max()),max_mmc_error=float(abs(mmc-nm).max())))
    assert np.isfinite(scores).all();np.save(D.ROOT/'daily_corr_mmc.npy',scores)
    payout=.1*scores[:,:,0]+scores[:,:,1];dates=pd.DatetimeIndex([d for d,_ in groups]);daily=pd.DataFrame(payout,index=dates,columns=names);daily.index.name='date';daily.to_csv(D.ROOT/'daily_payout.csv');daily.cumsum().to_csv(D.ROOT/'cumulative_payout.csv')
    records=[]
    for j,name in enumerate(names):records.append(dict(model=name,kind='New fit' if name in D.ARMS else 'Frozen reference',features=134 if name in D.ARMS else ('33→H' if name=='E228_M1' else ('99→H' if name=='E228_V1' else 'mixed inputs')),**S.payout_metrics(scores[:,j,0],scores[:,j,1]),cumulative_payout=float(payout[:,j].sum()),dates=len(groups),asset_dates=len(base),assets=int(base.ucid.nunique())))
    table=pd.DataFrame(records).sort_values('payout',ascending=False,kind='stable');table.to_csv(D.ROOT/'results.csv',index=False)
    S.parquet(D.ROOT/'scored_forecasts.parquet',base[['row','ucid','symbol','date',D.TARGET,'meta_model']].join(pd.DataFrame(x,columns=names)))
    audit.update(native_scoring_checks=native_checks,score_dates=len(groups),score_rows=len(base),score_assets=int(base.ucid.nunique()),e228_requested_rows=meta['e228_requested_rows'],e228_retained_rows=meta['e228_retained_rows']);S.save(D.ROOT/'audit.json',audit)
    S.save(D.ROOT/'scoring_manifest.json',dict(source_hashes=hashes,member_choices_frozen=True,new_residualization=False,runner_sha256=S.sha(__file__),model_order=names))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'asset_ridge':'Ridge per asset','asset_mlp':'MLP per asset','panel_ridge':'Pooled panel ridge','panel_mlp':'Pooled panel MLP','E228_M1':'Best momentum standalone (M1)','E228_V1':'Best volatility standalone (V1)','E229_best_mixed':'E229 best mixed ensemble','E229_all10':'E229 all-ten ensemble'}
    for refs in [False,True]:
        fig,ax=plt.subplots(figsize=(13,7),layout='constrained')
        for name in names if refs else D.ARMS:
            ax.plot(dates,daily[name].cumsum(),label=labels[name],lw=2.1 if name in D.ARMS else 1.6,ls='-' if name in D.ARMS else '--')
        ax.set_title('E230: concatenated134 features — '+('four learners and frozen references' if refs else 'four fresh learners'));ax.set_ylabel('Additive cumulative target20 payout score');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2);fig.savefig(D.ROOT/('cumulative_with_references.png' if refs else 'cumulative_four_models.png'),dpi=160);plt.close(fig)
    frame=table.copy();frame['model']=frame.model.map(labels)
    show=frame[['model','kind','features','corr','mmc','payout','sharpe.p','maxdd.p','cumulative_payout']].rename(columns={'model':'Model','kind':'Type','features':'Inputs','corr':'CORR','mmc':'MMC','payout':'Mean payout','sharpe.p':'Sharpe','maxdd.p':'Max DD','cumulative_payout':'Cumulative'})
    cov=pd.read_csv(D.DATA/'coverage_summary.csv');choices=pd.read_csv(D.ROOT/'mlp_selected_parameters.csv');choicecounts=choices.groupby(['arm','width','alpha']).size().reset_index(name='Selected units')
    body=f'<h1>E230 — Concatenated momentum and volatility134</h1><p>134 unique original input columns:33 E205 momentum TS bins,99 E209 volatility TS bins and2 native v2.0 volatility bins. No hidden states, new binning or filling. Ridge and nonlinear ReLU MLP are fitted per asset and across pooled asset-date rows. Pooled models have134 inputs and no asset-ID input or cross-asset slot expansion.</p><p>Target20. Selection training ends2023-11-27; validation2024-01-01 through2024-11-27; fresh final training ends2024-11-27. Both label walls are35 calendar days. Scaling uses training rows only and is refitted on final training. Eligible assets are fixed using at least60 training and20 validation complete/source-eligible rows. All four arms use the exact same row union in every phase.</p><p>Evaluation:{dates.min():%Y-%m-%d} through{dates.max():%Y-%m-%d};{len(groups)} dates;{len(base):,} asset-dates;{base.ucid.nunique()} assets. Retained{meta["e228_retained_rows"]:,}/{meta["e228_requested_rows"]:,} E228 comparison keys; eligibility fixed before fitting. Frozen E229 memberships and E228 raw/residual forecasts are unchanged; prediction ranks are rebuilt on these same evaluation assets. No new residualization.</p>'
    body+='<h2>Results on identical keys</h2>'+show.to_html(index=False,float_format=lambda z:f'{z:.6f}')+'<p>Payout=0.1CORR+historical-meta proxy MMC. Cumulative values are additive score sums; Sharpe uses population daily standard deviation and is unannualized. Max drawdown includes an initial zero. These scores are not investment returns.</p><img src="cumulative_with_references.png"><h2>Four fresh feature-only learners</h2><img src="cumulative_four_models.png">'
    body+='<h2>Dataset coverage</h2><p>Stored dataset availability; model eligibility and the matched scoring cohort are stricter. All original keys and partial NaNs are retained in storage.</p>'+cov.to_html(index=False,float_format=lambda z:f'{z:.2f}')
    body+='<h2>Hyperparameter selection</h2><p>Ridge searches33 regularization strengths,10⁻⁴ to10⁴. MLP compares widths32 and64→32, each with alpha0.01 or1; Adam learning rate0.001,batch256 capped to sample size,100 full epochs with early stopping disabled,seed230. Selection uses chronological validation MSE. Each per-asset model selects independently; each pooled model selects once. The fixed100-epoch search is not a scaling-law replication or a claim of optimal training compute.</p>'+choicecounts.to_html(index=False)
    body+='<h2>Checks and limits</h2><p>PASS WITH CONDITIONS. Exact source keys/symbols/target20 and unchanged134 columns, source hashes, serialized dataset roundtrip, preserved NaNs, source-eligible fitting, every serialized model/scaler replay, same training/validation/final/query row unions, label walls, native scoring parity and selected-ridge parity with an independent sklearn solver passed. Full audit and provenance are saved separately.</p><p>This is a retrospective learner comparison on a covered cohort. Source feature selection and the reference ensemble were chosen using previously inspected history. The displayed period is not new independent holdout evidence. Vendor-vintage restrictions remain inherited; no native-feature marginal-value, full-universe or live prediction claim.</p>'
    page=D.ROOT/'E230_momentum_volatility134.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:9px;break-inside:avoid!important;page-break-inside:avoid!important}td,th{border:1px solid #ccc;padding:4px}img{display:block;width:100%;height:auto;margin:auto;break-inside:avoid}td:nth-child(n+4){white-space:nowrap}</style></head><body>'+body+'</body></html>')
    subprocess.run([S.PYTHON,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=300)
    assert all(S.sha(path)==h for path,h in hashes.items())
    S.save(D.ROOT/'COMPLETE.json',dict(experiment='E230',status='success',arms=4,dataset=str(D.DATA),features=134,report=str(page.with_suffix('.pdf')),report_sha256=S.sha(page.with_suffix('.pdf')),audit='PASS WITH CONDITIONS',time=time.time()))
    S.save(D.ROOT/'status.json',dict(stage='complete',time=time.time()));print(table.to_string(index=False),flush=True)
