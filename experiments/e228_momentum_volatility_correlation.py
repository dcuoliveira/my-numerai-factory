"""E228: shared-cohort top-five standalone momentum/volatility correlations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e212_volatility99_tsfm as S

REPO=S.REPO
ROOT=REPO/'results/reports/crypto/e228_momentum_volatility_correlation'
R=REPO/'results/reports/crypto'
FAMILY={'chronos2':'Chronos-2','t0':'t0','tabpfnts':'TabPFN-TS','timestral':'Timestral'}
save,parquet,sha=S.save,S.parquet,S.sha

def indexed(frame):
    frame=frame.copy();frame['ucid']=frame.ucid.astype(str)
    assert frame.ucid.str.fullmatch(r'\d+').all()
    assert not frame.duplicated(['ucid','date']).any()
    return frame.set_index(['ucid','date'])

def load():
    hashes={};checks=[];catalog=[];values=[]
    def read(path,columns=None):
        path=Path(path)
        files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        before={str(p):sha(p) for p in files};result=pd.read_parquet(path,columns=columns)
        assert all(sha(p)==h for p,h in before.items())
        if path.is_dir():assert [str(p) for p in sorted(p for p in path.rglob('*') if p.is_file())]==list(before)
        hashes.update(before);return result
    p=indexed(read(S.ROOT/'panel.parquet',columns=['row','ucid','symbol','date','target_binned_return_20','meta_model']))
    cover=indexed(read(S.ROOT/'coverage.parquet'))
    base=p.loc[cover.evaluation_available].copy();keys=base.index
    joint=np.ones(len(base),bool)
    def aligned(frame,label):
        nonlocal joint
        q=indexed(frame).reindex(keys)
        valid=q.symbol.notna() if 'symbol' in q else q.row.notna()
        if 'symbol' in q:assert q.loc[valid,'symbol'].eq(base.loc[valid,'symbol']).all(),label
        if 'target_binned_return_20' in q:
            observed=valid&q.target_binned_return_20.notna()
            np.testing.assert_array_equal(q.loc[observed,'target_binned_return_20'],base.loc[observed,'target_binned_return_20'])
        if 'meta_model' in q:
            observed=valid&q.meta_model.notna();difference=np.abs(q.loc[observed,'meta_model'].to_numpy()-base.loc[observed,'meta_model'].to_numpy())
            checks.append(dict(source=label,overlap_rows=int(observed.sum()),different_meta_rows=int((difference>1e-12).sum()),max_meta_difference=float(difference.max()) if len(difference) else 0.))
        return q
    def add(identity,family,label,study,model,norm,training,view,prediction,source_payout):
        catalog.append(dict(id=identity,family=family,label=label,study=study,model=model,standardization=norm,training_target=training,view=view,source_payout=float(source_payout)))
        values.append(np.asarray(prediction,float))
    # All completed E206 design-C H variants, one common eligibility set.
    folder=R/'e206_momentum3_tsfm';path=folder/'ranked_results.csv';hashes[str(path)]=sha(path);table=pd.read_csv(path)
    selected=table[table.design.eq('C')&table.universe.eq('common')&table.side.eq('neutral')&table.status.eq('success')&table.model.isin(FAMILY)].sort_values('tag')
    assert len(selected)==24 and selected.tag.is_unique
    first=selected.iloc[0].tag
    mask=aligned(read(Path('/scratch/danielco/e206_momentum3_tsfm/portfolios')/first/'common'), 'E206 common')
    joint &= mask.eligible.fillna(False).to_numpy(bool)
    for row in selected.itertuples():
        q=aligned(read(folder/'fits'/row.tag/'predictions.parquet'),row.tag)
        label=f'E206 {FAMILY[row.model]} / {row.standardization} / 20d'
        add('E206::'+row.tag,'Momentum',label,'E206',row.model,row.standardization,20,'raw',q.prediction,row.payout)
    # All completed E211 H-only standalone raw/residual streams.
    folder=R/'e211_cross_target_tsfm';path=folder/'all_trials.csv';hashes[str(path)]=sha(path);table=pd.read_csv(path)
    q=aligned(read(folder/'standalone_forecasts.parquet'),'E211 forecasts')
    eligibility=indexed(read(folder/'coverage.parquet')).reindex(keys).scored.fillna(False).to_numpy(bool);joint &= eligibility
    original=aligned(read(folder/'panel.parquet',columns=['row','symbol','ucid','date','target_binned_return_20','meta_model']),'E211 panel')
    count=0
    for row in table[table.kind.eq('standalone')&table.evaluation_target.eq(20)].sort_values('id').itertuples():
        name=row.members;model=name.split('_t')[0]
        if model not in ('chronos2','t0','timestral'):continue
        h=int(name.split('_t')[1].split('_')[0]);view='residual' if name.endswith('_r') else 'raw';norm=S.NORMS[model]
        label=f'E211 {FAMILY[model]} / {norm} / {h}d'+(' / residual' if view=='residual' else '')
        add('E211::'+name,'Momentum',label,'E211',model,norm,h,view,q[name],row.payout);count+=1
    assert count==12
    # Every standalone in the updated E212 report, including E226/E227 controls.
    path=S.ROOT/'report_comparison_trials.parquet';table=read(path)
    table=table[table.evaluation_target.eq(20)&table.kind.eq('standalone')].sort_values('id');assert len(table)==28
    folders=[S.ROOT,R/'e226_volatility_cs_ridge',R/'e227_v20_volatility_ridge'];streams={}
    for folder in folders:
        frame=aligned(read(folder/'standalone_forecasts.parquet'),folder.name)
        for name in table.members:
            if name in frame:
                assert name not in streams;streams[name]=frame[name]
    for row in table.itertuples():
        name=row.members;h=int(row.training_targets);view=row.view
        if row.model!='feature_ridge':label=f'E212 {FAMILY[row.model]} / {row.standardization} / {h}d'
        else:
            mode={'TS99':'TS-only99','CS99':'CS-only99','TS_CS198':'TS+CS198','V20_VOL2':'v2.0 vol2'}[row.information_set]
            label=f'{mode} ridge / {h}d'
        if view=='residual':label+=' / residual'
        add('Volatility::'+row.id,'Volatility',label,row.id.split('_')[0].upper(),row.model,row.standardization,h,view,streams[name],row.payout)
    catalog=pd.DataFrame(catalog);x=np.column_stack(values)
    assert len(catalog)==64 and catalog.id.is_unique
    joint &= np.isfinite(x).all(axis=1)
    joint &= np.isfinite(base.target_binned_return_20)&np.isfinite(base.meta_model)
    counts=pd.Series(joint,index=base.index).groupby(level='date').sum();joint &= base.index.get_level_values('date').map(counts).to_numpy()>=10
    base=base.loc[joint];x=x[joint];assert np.isfinite(x).all()
    assert not base.empty and not base.index.duplicated().any()
    parquet(ROOT/'common_keys.parquet',base.reset_index())
    parquet(ROOT/'all_candidate_forecasts.parquet',base.reset_index()[['row','ucid','symbol','date']].join(pd.DataFrame(x,columns=catalog.id)))
    save(ROOT/'source_manifest.json',dict(experiment='E228',runner_sha256=sha(__file__),candidate_count=len(catalog),input_hashes=hashes,meta_comparisons=checks,scoring_benchmark='E212 frozen historical meta, target20; all candidates rescored on identical joint eligible keys',candidate_pool='24 completed E206 design-C H variants +12 E211 H-only raw/residual variants;28 updated E212 standalone volatility streams. No ensembles, momentum raw-feature controls or unfinished studies.',transform_policy='Use frozen delivered raw/residual forecasts; do not refit or residualize again on the comparison cohort.'))
    return base,x,catalog

def heatmap(matrix,title,path,split=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,11) if len(matrix)==10 else (12,8.5),layout='constrained')
    im=ax.imshow(matrix.to_numpy(),cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=55,ha='right',fontsize=10)
    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=10)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=11,color='white' if abs(v)>.58 else 'black')
    if split:ax.axhline(4.5,color='#222',lw=2);ax.axvline(4.5,color='#222',lw=2)
    ax.set_title(title,fontsize=13,pad=15);fig.colorbar(im,ax=ax,shrink=.75,label='Correlation');fig.savefig(path,dpi=160);plt.close(fig)

def run():
    ROOT.mkdir(parents=True,exist_ok=True);save(ROOT/'status.json',dict(stage='aligning',time=time.time()))
    base,x,catalog=load();groups=list(base.groupby(level='date',sort=True).indices.items());scores=np.empty((len(groups),len(catalog),2));native_checks=[]
    for d,(date,idx) in enumerate(groups):
        y=base.target_binned_return_20.iloc[idx].to_numpy();m=base.meta_model.iloc[idx].to_numpy();corr,mmc=S.matrix_scores(x[idx],y,m)
        scores[d,:,0]=np.nan_to_num(corr,nan=0.);scores[d,:,1]=mmc
        if d in (0,len(groups)//2,len(groups)-1):
            pred=pd.DataFrame(x[idx],columns=catalog.id);target=pd.Series(y,name='target');benchmark=pd.Series(m,name='meta_model')
            nc=numerai_corr(pred,target).fillna(0).to_numpy();nm=correlation_contribution(pred,benchmark,target).to_numpy()
            np.testing.assert_allclose(scores[d,:,0],nc,rtol=2e-6,atol=2e-7);np.testing.assert_allclose(scores[d,:,1],nm,rtol=1e-10,atol=1e-12)
            native_checks.append(dict(date=str(date),assets=len(idx),candidates=len(catalog),max_corr_error=float(np.max(np.abs(scores[d,:,0]-nc))),max_mmc_error=float(np.max(np.abs(scores[d,:,1]-nm)))))
    assert np.isfinite(scores).all();np.save(ROOT/'daily_corr_mmc.npy',scores)
    payout=.1*scores[:,:,0]+scores[:,:,1];dates=pd.DatetimeIndex([d for d,_ in groups])
    ranked=catalog.copy()
    for name in ('corr','mmc','payout','sharpe.p','maxdd.p'):ranked[name]=[S.payout_metrics(scores[:,j,0],scores[:,j,1])[name] for j in range(len(catalog))]
    ranked['candidate_column']=np.arange(len(catalog));ranked['dates']=len(groups);ranked['asset_dates']=len(base)
    ranked=ranked.sort_values(['family','payout','id'],ascending=[True,False,True],kind='stable');ranked['family_rank']=ranked.groupby('family').cumcount()+1;ranked.to_csv(ROOT/'all_candidate_ranking.csv',index=False)
    picked=pd.concat([ranked[ranked.family.eq(f)].head(5) for f in ['Momentum','Volatility']],ignore_index=True)
    picked['key']=[f'{f}{i}' for f in ('M','V') for i in range(1,6)];picked['axis_label']=picked.key+' '+picked.label
    picked.to_csv(ROOT/'selected_models.csv',index=False);indices=picked.candidate_column.to_numpy(int);selected=x[:,indices]
    daily=pd.DataFrame(payout[:,indices],index=dates,columns=picked.key);daily.index.name='date';daily.to_csv(ROOT/'daily_payout_selected.csv')
    daily_all=pd.DataFrame(payout,index=dates,columns=catalog.id);daily_all.index.name='date';parquet(ROOT/'daily_payout_all.parquet',daily_all.reset_index())
    correlation=daily.corr();correlation.to_csv(ROOT/'daily_payout_correlation.csv');correlation.loc[[f'M{i}' for i in range(1,6)],[f'V{i}' for i in range(1,6)]].to_csv(ROOT/'daily_payout_cross_correlation.csv')
    rank_correlations=[];rank_dates=[];skipped=[]
    for date,idx in groups:
        ranks=rankdata(selected[idx],axis=0,method='average')
        if (np.std(ranks,axis=0)==0).any():skipped.append(str(date));continue
        a=np.corrcoef(ranks,rowvar=False);reference=spearmanr(selected[idx],axis=0).statistic
        np.testing.assert_allclose(a,reference,atol=1e-12,rtol=1e-12);rank_correlations.append(a);rank_dates.append(date)
    assert rank_correlations
    prediction=pd.DataFrame(np.mean(rank_correlations,axis=0),index=picked.key,columns=picked.key);prediction.to_csv(ROOT/'prediction_rank_correlation.csv');prediction.iloc[:5,5:].to_csv(ROOT/'prediction_rank_cross_correlation.csv')
    np.save(ROOT/'daily_prediction_rank_correlations.npy',rank_correlations);save(ROOT/'prediction_rank_dates.json',list(map(str,rank_dates)))
    for a in (correlation.to_numpy(),prediction.to_numpy()):
        assert np.isfinite(a).all();np.testing.assert_allclose(a,a.T,atol=1e-12);np.testing.assert_allclose(np.diag(a),1,atol=1e-12);assert np.linalg.eigvalsh(a).min()>-1e-9
    # Independent direct Pearson reconstruction from centered daily series.
    centered=daily.to_numpy()-daily.to_numpy().mean(0);norm=np.sqrt((centered**2).sum(0));manual=(centered.T@centered)/np.outer(norm,norm);np.testing.assert_allclose(manual,correlation.to_numpy(),atol=1e-12,rtol=1e-12)
    for family in ('Momentum','Volatility'):
        expected=ranked[ranked.family.eq(family)].head(5).id.tolist();assert picked[picked.family.eq(family)].id.tolist()==expected
    forecasts=base.reset_index()[['row','ucid','symbol','date']].join(pd.DataFrame(selected,columns=picked.key));parquet(ROOT/'selected_forecasts.parquet',forecasts)
    coverage=base.reset_index().groupby('date').agg(assets=('ucid','nunique'),rows=('ucid','size'));coverage.to_csv(ROOT/'coverage_by_date.csv')
    labels=picked.axis_label.to_list();span=f'{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}; {len(dates)} shared dates'
    full=correlation.copy();full.index=full.columns=labels
    heatmap(full,'E228: daily payout Pearson correlation\n'+span,ROOT/'daily_payout_correlation.png',True)
    cross=full.iloc[:5,5:];heatmap(cross,'E228: momentum versus volatility — daily payout Pearson\n'+span,ROOT/'daily_payout_cross_correlation.png')
    full=prediction.copy();full.index=full.columns=labels
    heatmap(full,'E228: equal-date mean prediction Spearman correlation\n'+f'{len(rank_dates)} shared dates',ROOT/'prediction_rank_correlation.png',True)
    heatmap(full.iloc[:5,5:],'E228: momentum versus volatility — mean daily prediction Spearman',ROOT/'prediction_rank_cross_correlation.png')
    manifest=json.loads((ROOT/'source_manifest.json').read_text());assert all(sha(path)==h for path,h in manifest['input_hashes'].items())
    audit=dict(outcome='PASS WITH CONDITIONS',native_score_checks=native_checks,permanent_keys_symbols_targets_verified=True,all_candidates_same_scored_rows=True,candidate_count=len(catalog),momentum_candidates=int(catalog.family.eq('Momentum').sum()),volatility_candidates=int(catalog.family.eq('Volatility').sum()),dates=len(dates),asset_dates=len(base),distinct_assets=int(base.reset_index().ucid.nunique()),prediction_correlation_dates=len(rank_dates),prediction_constant_dates_skipped=skipped,pearson_direct_reconstruction=True,spearman_daily_scipy_parity=True,top5_selection_replayed=True,source_hashes_unchanged=True,conditions=['Ranking and correlation share retrospective development data; selected top models are exploratory, not independent confirmation.','Source-vintage/pretraining limitations are inherited; no new native-feature marginal, live or investment-return claim.','Raw/residual variants of the same fitted model may both be selected. Frozen residual transforms retain their original causal universes; only scoring is recomputed on the joint cohort.'])
    save(ROOT/'audit.json',audit)
    show=picked[['key','family','label','view','corr','mmc','payout','sharpe.p']].rename(columns={'key':'Key','family':'Family','label':'Model','view':'View','corr':'CORR','mmc':'MMC','payout':'Mean payout','sharpe.p':'Sharpe'})
    body=f'<h1>E228 — Top-five momentum versus volatility correlations</h1><p>Standalone models only; ensembles excluded. Selection: top five per family by mean target20 payout after rescoring every candidate on identical permanent asset/date keys and one historical-meta benchmark.</p><p>{span}; {len(base):,} asset-dates; {audit["distinct_assets"]} distinct assets; daily cohort {int(coverage.assets.min())}–{int(coverage.assets.max())} assets. Payout = 0.1 CORR + historical-meta proxy MMC. Values are Numerai scores, not investment returns.</p>'
    body+='<p>Momentum pool: 24 completed E206 design-C TSFM/scaler configurations plus 12 E211 hidden-state raw/residual variants; momentum feature-only controls and unfinished studies excluded. Volatility pool: all 28 standalone streams in the updated E212 report, including E226/E227 feature-only controls. Both training horizons are eligible; all ranking uses target20. Variants from the same fitted head may appear separately. Forecasts and original residual transforms are frozen; no fitting or new ensembles.</p>'
    body+='<h2>Selected models — common-cohort ranking</h2>'+show.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body+='<p>Source-PDF rankings can differ because these candidates are freshly scored on the smaller joint cohort. Full candidate ranking, original source means and exact daily scores are exported separately.</p>'
    body+='<h2>Daily payout correlation</h2><p>Pearson correlation across shared dates between the models’ daily payout scores. The cross-family matrix isolates momentum rows and volatility columns.</p><img src="daily_payout_correlation.png"><img src="daily_payout_cross_correlation.png">'
    body+='<h2>Prediction-rank correlation</h2><p>Spearman correlation across shared assets separately on each date, then an equal-weight average across dates. This compares signal directions rather than daily scoring outcomes. '+f'{len(rank_dates)} dates used; {len(skipped)} dates excluded because at least one selected forecast was constant.</p><img src="prediction_rank_correlation.png"><img src="prediction_rank_cross_correlation.png">'
    body+='<h2>Interpretation and checks</h2><p>Same-history top-five selection is exploratory. Correlation does not establish additional predictive value or ensemble improvement. Historical provider vintages and pretraining overlap remain uncertified. Native scoring parity, direct Pearson reconstruction, per-date SciPy Spearman parity, coverage/identity checks and unchanged source hashes passed. Full audit and provenance remain in separate machine-readable artifacts.</p>'
    page=ROOT/'E228_momentum_volatility_correlation.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:9px}td,th{border:1px solid #ccc;padding:4px}img{display:block;max-width:100%;max-height:165mm;object-fit:contain;margin:auto;break-inside:avoid}table{break-inside:avoid}</style></head><body>'+body+'</body></html>')
    subprocess.run([S.PYTHON,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=300)
    for item in picked.itertuples():save(ROOT/'fits'/item.key/'COMPLETE.json',dict(task=dict(tag=item.key,model='forecast_correlation'),source_model=item.id))
    save(ROOT/'COMPLETE.json',dict(experiment='E228',status='success',selected_models=10,candidate_models=len(catalog),dates=len(dates),asset_dates=len(base),report=str(page.with_suffix('.pdf')),pdf_sha256=sha(page.with_suffix('.pdf')),audit='PASS WITH CONDITIONS',time=time.time()))
    save(ROOT/'status.json',dict(stage='complete',time=time.time()));print('E228 COMPLETE',picked[['key','label','payout']].to_string(index=False),flush=True)

if __name__=='__main__':run()
