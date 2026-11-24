"""Matched per-asset KTZ portfolios and exhaustive, batched equal-rank ensembles."""
import base64
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.analysis.asset_momentum import route_slots
from src.analysis.asset_ktz_states import ALIASES


def definitions(count, maximum=5):
    return [(i,) for i in range(count)]+[v for k in range(2,min(maximum,count)+1)
                                        for v in itertools.combinations(range(count),k)]


def score_batch(raw, target, meta, members, scorer):
    # Integer first-tie ranks match the live helper. Sum integers before any
    # division, so equivalent rank averages cannot acquire floating-point ties.
    ranks=rankdata(raw,axis=0,method='ordinal').astype(float)
    weights=np.zeros((raw.shape[1],len(members)))
    for j,part in enumerate(members):weights[list(part),j]=1
    values=ranks@weights
    for j,part in enumerate(members):
        if len(part)==1:values[:,j]=raw[:,part[0]]
    return np.column_stack(scorer(values,target,meta))


def metrics(scores):
    payout=.1*scores[:,:,0]+scores[:,:,1]
    average=payout.mean(axis=0);std=payout.std(axis=0)
    cumulative=np.vstack([np.zeros((1,payout.shape[1])),payout.cumsum(axis=0)])
    return dict(corr=scores[:,:,0].mean(axis=0),mmc=scores[:,:,1].mean(axis=0),payout=average,
         **{'sharpe.p':np.divide(average,std,out=np.full_like(average,np.nan),where=std>0),
            'maxdd.p':(np.maximum.accumulate(cumulative,axis=0)-cumulative).max(axis=0)})


def analyze(root,scratch,source,configs):
    from experiments import e206_momentum3_tsfm as E
    from src.analysis.html_pdf import to_pdf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root,scratch,source=map(Path,[root,scratch,source])
    source_scratch=Path('/scratch/danielco/e206_momentum3_tsfm')
    base=pd.read_parquet(source/'momentum_baseline.parquet').sort_values(['date','ucid']).reset_index(drop=True)
    assert base.row.is_unique and not base[['ucid','date']].duplicated().any()
    assert np.isfinite(base.baseline_rank).all()
    streams=[];columns=[];provenance=[]
    for c in configs:
        for family in [*ALIASES,'ridge']:
            tag=f'{c["id"]}_{family}'
            directory=source/'fits'/c['source'] if family=='ridge' else root/'fits'/tag
            if family!='ridge':assert (directory/'COMPLETE.json').exists()
            path=directory/'predictions.parquet';p=pd.read_parquet(path)
            assert p.row.is_unique
            columns.append(p.set_index('row').prediction.reindex(base.row).to_numpy(float))
            streams.append(dict(id=tag,config=c['id'],model=c['model'],standardization=c['norm'],design='C',learner=family))
            provenance.append(dict(id=tag,path=str(path),sha256=E.sha(path)))
    raw=np.column_stack(columns);assert raw.shape[1]==28
    np.savez_compressed(scratch/'standalone_raw_predictions.npz',row=base.row.to_numpy(),predictions=raw)
    pd.DataFrame(streams).to_csv(root/'standalone_key.csv',index=False)
    # Reuse the original report's all-config common sample, rather than widening
    # its sample to just these seven models.
    mask_path=source_scratch/'portfolios'/configs[0]['source']/'common.parquet'
    if not mask_path.exists():
        candidates=list((source_scratch/'portfolios'/configs[0]['source']/'common').glob('*.parquet'))
        if not candidates:raise FileNotFoundError('Original common score mask: '+str(mask_path))
        mask_path=candidates[0].parent
    old=pd.read_parquet(mask_path,columns=['row','eligible'])
    assert old.row.is_unique
    original=old.set_index('row').eligible.reindex(base.row).fillna(False).to_numpy(bool)
    covered=np.isfinite(raw).all(axis=1)&np.isfinite(base[E.TARGET])&np.isfinite(base.meta_model)
    selected=original&covered
    sample=base.loc[selected].copy();x=raw[selected]
    sample.to_parquet(root/'common_keys.parquet',index=False)
    coverage=base[['row','date','ucid']].copy();coverage['original_common']=original
    coverage['all28_covered']=covered;coverage['scored_common']=selected
    for i,s in enumerate(streams):coverage[s['id']]=np.isfinite(raw[:,i])
    coverage.to_parquet(root/'coverage.parquet',index=False)
    cohorts=coverage.groupby('date')[['original_common','all28_covered','scored_common']].sum()
    cohorts.to_csv(root/'coverage_by_date.csv')
    dates=sample.date.drop_duplicates().sort_values().to_numpy()
    assert len(dates)==297, f'Expected original 297 score dates; got {len(dates)}'
    assert sample.groupby('date').size().min()>=10
    boundaries=np.r_[0,np.flatnonzero(sample.date.to_numpy()[1:]!=sample.date.to_numpy()[:-1])+1,len(sample)]
    # Explicit full-panel fallback portfolios: unavailable assets keep their
    # baseline rank positions exactly. This is research routing, not a native
    # v2.0-feature B_all marginal comparison or a production artifact.
    routed=np.empty_like(raw)
    full_scores=[]
    for date,idx in base.groupby('date',sort=True).indices.items():
        b=base.iloc[idx];outputs=[]
        for j in range(len(streams)):
            routed[idx,j]=route_slots(b.baseline_rank.to_numpy(),raw[idx,j])
        eligible=np.isfinite(b[E.TARGET].to_numpy())&np.isfinite(b.meta_model.to_numpy())
        if eligible.sum()>=10:
            c,m=E.matrix_scores(routed[idx][eligible],b.loc[eligible,E.TARGET].to_numpy(),b.loc[eligible,'meta_model'].to_numpy())
            for j,s in enumerate(streams):full_scores.append(dict(date=date,id=s['id'],corr=c[j],mmc=m[j],payout=.1*c[j]+m[j]))
    assert np.isfinite(routed).all()
    np.savez_compressed(scratch/'standalone_routed_predictions.npz',row=base.row.to_numpy(),predictions=routed)
    pd.DataFrame(full_scores).to_parquet(root/'routed_daily_scores.parquet',index=False)
    members=definitions(len(streams));assert len(members)==122437
    names=[s['id'] for s in streams]
    labels=[' + '.join(names[i] for i in v) for v in members]
    E.save(root/'ensemble_definitions.json',[dict(id=i,members=[names[j] for j in v]) for i,v in enumerate(members)])
    scores=np.lib.format.open_memmap(scratch/'daily_scores.npy',mode='w+',dtype=np.float64,shape=(len(dates),len(members),2))
    for di,(start,stop) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        part=sample.iloc[start:stop];xx=x[start:stop]
        for offset in range(0,len(members),512):
            block=members[offset:offset+512]
            scores[di,offset:offset+len(block)]=score_batch(xx,part[E.TARGET].to_numpy(),part.meta_model.to_numpy(),block,E.matrix_scores)
        if di in [0,len(dates)//2,len(dates)-1]:
            for j in [0,28,len(members)-1]:
                ids=members[j]
                v=xx[:,ids[0]] if len(ids)==1 else np.sum(np.column_stack([rankdata(xx[:,i],method='ordinal') for i in ids]),axis=1)
                reference=np.r_[E.matrix_scores(v[:,None],part[E.TARGET].to_numpy(),part.meta_model.to_numpy())]
                np.testing.assert_allclose(scores[di,j],reference,rtol=1e-10,atol=1e-12)
        if di%10==0:
            scores.flush();E.save(root/'analysis_progress.json',dict(scored_dates=di+1,total_dates=len(dates),trials=len(members),time=time.time()))
    scores.flush();assert np.isfinite(scores).all()
    replay_error=0.
    if int(original.sum())==int(selected.sum()):
        for j,stream in enumerate(streams):
            if stream['learner']!='ridge':continue
            c=next(v for v in configs if v['id']==stream['config'])
            reference=pd.read_parquet(source/'scores'/f'{c["source"]}_common.parquet').set_index('date').reindex(dates)
            expected=reference[['corr','mmc']].to_numpy(float)
            np.testing.assert_allclose(scores[:,j],expected,rtol=1e-6,atol=1e-8)
            replay_error=max(replay_error,float(np.max(np.abs(scores[:,j]-expected))))
    table=pd.DataFrame(dict(id=np.arange(len(members)),kind=['standalone' if len(v)==1 else 'ensemble' for v in members],
           components=list(map(len,members)),members=labels,score_dates=len(dates),mean_coverage=len(sample)/len(dates),**metrics(scores)))
    table=table.sort_values('payout',ascending=False)
    table.to_csv(root/'all_trials.csv',index=False)
    standalone=table[table.kind.eq('standalone')].merge(pd.DataFrame(streams).rename(columns={'id':'members'}),on='members',validate='1:1')
    standalone.to_csv(root/'standalone_results.csv',index=False)
    standalone.rename(columns={'members':'tag','score_dates':'n_dates'}).to_csv(root/'all_model_window_rows.csv',index=False)
    ensembles=table[table.kind.eq('ensemble')];ensembles.to_csv(root/'ensemble_results.csv',index=False)
    payout=.1*scores[:,:,0]+scores[:,:,1];np.save(scratch/'daily_payout.npy',payout);np.save(scratch/'score_dates.npy',dates)
    pd.DataFrame(payout[:,:28],index=dates,columns=names).to_csv(root/'standalone_daily_payout.csv',index_label='date')
    def embedded(fig,name):
        path=root/(name+'.png');fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)
        return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
    fig,ax=plt.subplots(figsize=(13,7))
    for j in range(28):ax.plot(dates,payout[:,j].cumsum(),lw=1,label=names[j])
    ax.set(ylabel='Cumulative payout',title='All 28 standalone portfolios');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=7)
    standalone_plot=embedded(fig,'standalone_cumulative_payout')
    correlation=np.corrcoef(payout[:,:28].T)
    pd.DataFrame(correlation,index=names,columns=names).to_csv(root/'standalone_payout_correlation.csv')
    fig,ax=plt.subplots(figsize=(14,11));im=ax.imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(28),names,rotation=90,fontsize=7);ax.set_yticks(range(28),names,fontsize=7)
    ax.set_title('Correlation of daily payouts');fig.colorbar(im,ax=ax,shrink=.75)
    heatmap=embedded(fig,'payout_correlation')
    top=ensembles.head(3)
    fig,ax=plt.subplots(figsize=(13,7))
    for j in range(28):ax.plot(dates,payout[:,j].cumsum(),lw=.8,alpha=.5,label=names[j])
    for n,row in enumerate(top.itertuples(),1):ax.plot(dates,payout[:,row.id].cumsum(),lw=2.7,label=f'Top {n} ensemble (trial {row.id})')
    ax.set(ylabel='Cumulative payout',title='Top three ensembles and all standalone portfolios');ax.grid(alpha=.2)
    ax.legend(ncol=4,fontsize=7);top_plot=embedded(fig,'top_ensembles_cumulative_payout')
    original_rows=int(original.sum());common_rows=int(selected.sum())
    E.save(root/'audit.json',dict(outcome='PASS WITH CONDITIONS',original_common_rows=original_rows,
        scored_common_rows=common_rows,removed_rows=original_rows-common_rows,dates=len(dates),
        source_mask=str(mask_path),prediction_sources=provenance,common_keys_hash=E.sha(root/'common_keys.parquet'),
        source_ridge_replay_max_error=replay_error if int(original.sum())==int(selected.sum()) else None,
        verified=['Exact C block-sum identity during preparation','Train-only constants and native MLP scalers; fresh candidate/refit weights',
           'No test labels used for tuning','All 28 streams scored on identical keys','Uncovered baseline ranks and covered slot sets preserved',
           'Batched/scalar scoring parity, finite full-panel routing'],
        conditions=['Matched historical research; original source dataset vintage and pretrained checkpoint chronology limitations remain',
           'No native-feature marginal attribution, no deployed-model claim, no verified actual-price returns',
           'All ensembles selected and ranked on the same historical sample; results are exploratory, not untouched holdout evidence']))
    display_cols=['members','components','score_dates','mean_coverage','corr','mmc','payout','sharpe.p','maxdd.p']
    def html_table(frame,cols):return frame[cols].to_html(index=False,float_format=lambda v:f'{v:.6f}',escape=True)
    key=pd.DataFrame(configs)[['id','model','norm','window']].rename(columns={'norm':'standardization'})
    html='''<!doctype html><html><head><meta charset="utf-8"><title>E218 per-asset KTZ portfolios</title>
    <style>body{font-family:Arial,sans-serif;color:#17212b;margin:24px;font-size:12px}table{border-collapse:collapse;width:100%;font-size:10px}
    th,td{padding:5px;border:1px solid #bbb;text-align:right}td:first-child{text-align:left;overflow-wrap:anywhere;max-width:400px}
    th{background:#e9edf1}img{width:100%}figure{margin:10px 0;break-inside:avoid}h2{break-before:page}</style></head><body>'''
    html+='<h1>E218 — independent per-asset KTZ portfolios</h1>'
    html+=f'<p>Seven E206 design-C input groups × three fresh learners, with seven source ridge references. Four frozen KTZ candidates per learner, selected independently for each asset on past validation MSE, then refitted. Native sklearn MLP initialization. All 122,409 equal-rank ensembles of two to five streams.</p><p>Common sample: {common_rows:,} asset/date rows across {len(dates)} score dates, {len(sample)/len(dates):.3f} assets/date. Original E206 common sample: {original_rows:,} rows; removed for joint coverage: {original_rows-common_rows:,}. Every table is sorted by mean payout. Payout = 0.1 × CORR + historical-meta proxy MMC. sharpe.p is unannualized mean/std payout; maxdd.p is peak-to-trough cumulative payout. These plots show payout, not investable price returns.</p>'
    html+='<p>Same-history ensemble selection is exploratory. Source data vintage and pretrained checkpoint chronology limits carry over. No production or marginal native-feature claim. Completion audit: PASS WITH CONDITIONS.</p>'
    html+='<h3>Input group key</h3>'+key.to_html(index=False)+'<h2>Standalone portfolios</h2>'+html_table(standalone,['members','model','standardization','learner','score_dates','corr','mmc','payout','sharpe.p','maxdd.p'])
    html+='<h2>All standalone cumulative payouts</h2>'+standalone_plot+'<h2>Standalone daily payout correlation</h2>'+heatmap
    html+='<h2>Top 50 ensembles</h2>'+html_table(ensembles.head(50),display_cols)
    html+='<h2>Top three ensembles and standalone portfolios</h2>'+html_table(top,display_cols)+top_plot
    html+='<p>The accompanying all_trials.csv contains every standalone and ensemble trial. Raw missingness, full-panel routed predictions, daily scores, coverage and source hashes are retained separately.</p></body></html>'
    page=root/'E218_asset_ktz_states.html';page.write_text(html)
    pdf=to_pdf(str(page),'lovelace');assert Path(pdf).stat().st_size>1000
    E.save(root/'ANALYSIS_COMPLETE.json',dict(trials=len(members),ensembles=len(members)-28,dates=len(dates),
         common_rows=common_rows,pdf_sha256=E.sha(pdf),time=time.time()))
