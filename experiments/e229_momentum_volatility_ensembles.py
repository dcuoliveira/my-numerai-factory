"""E229: exhaustive equal-weight rank blends of the ten frozen E228 selections."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import itertools
import json
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numerai_tools.scoring import numerai_corr,correlation_contribution
from experiments import e212_volatility99_tsfm as S

ROOT=S.REPO/'results/reports/crypto/e229_momentum_volatility_ensembles'
SOURCE=S.REPO/'results/reports/crypto/e228_momentum_volatility_correlation'

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    S.save(ROOT/'status.json',dict(stage='loading',time=time.time()))
    names=['common_keys.parquet','selected_forecasts.parquet','selected_models.csv','daily_payout_selected.csv','audit.json','source_manifest.json','COMPLETE.json']
    hashes={str(SOURCE/n):S.sha(SOURCE/n) for n in names}
    hashes.update({str(p):S.sha(p) for p in [__file__,S.__file__,S.REPO/'experiments/e150_finish_scoring.py']})
    base=pd.read_parquet(SOURCE/'common_keys.parquet');forecasts=pd.read_parquet(SOURCE/'selected_forecasts.parquet');models=pd.read_csv(SOURCE/'selected_models.csv')
    keys=[f'{f}{i}' for f in ('M','V') for i in range(1,6)]
    assert models.key.tolist()==keys
    assert not base.duplicated(['ucid','date']).any()
    pd.testing.assert_frame_equal(base[['row','ucid','symbol','date']],forecasts[['row','ucid','symbol','date']])
    x=forecasts[keys].to_numpy(float);assert np.isfinite(x).all()
    assert np.isfinite(base[['target_binned_return_20','meta_model']]).all().all()
    subsets=[(j,) for j in range(10)]+[s for size in range(2,11) for s in itertools.combinations(range(10),size)]
    membership=np.zeros((10,len(subsets)),dtype=np.int64)
    records=[]
    for j,subset in enumerate(subsets):
        membership[list(subset),j]=1
        m=sum(i<5 for i in subset);v=len(subset)-m
        records.append(dict(id=keys[j] if j<10 else f'E{j-9:04d}',kind='Standalone' if len(subset)==1 else 'Ensemble',family='Mixed' if m and v else ('Momentum' if m else 'Volatility'),members='+'.join(keys[i] for i in subset),size=len(subset),momentum_members=m,volatility_members=v,column=j))
    catalog=pd.DataFrame(records);sizes=membership.sum(0)
    assert len(catalog)==1023 and catalog.id.is_unique
    assert (catalog.kind.eq('Ensemble')&catalog.family.eq('Mixed')).sum()==961
    groups=list(base.groupby('date',sort=True).indices.items());dates=pd.DatetimeIndex([d for d,_ in groups])
    scores=np.empty((len(groups),len(catalog),2));checks=[];tie_checks=[]
    for d,(date,idx) in enumerate(groups):
        numerators=(2*rankdata(x[idx],axis=0,method='average')-1).astype(np.int64)
        blended=S.rank_average_blends(numerators,membership,sizes)
        y=base.target_binned_return_20.iloc[idx].to_numpy();meta=base.meta_model.iloc[idx].to_numpy()
        corr,mmc=S.matrix_scores(blended,y,meta)
        scores[d,:,0]=np.nan_to_num(corr,nan=0.);scores[d,:,1]=mmc
        if d in (0,len(groups)//2,len(groups)-1):
            # Independent numerator summation verifies every subset and exact ties.
            reference=np.column_stack([numerators[:,list(s)].sum(axis=1)/(2.0*len(idx)*len(s)) for s in subsets])
            np.testing.assert_array_equal(blended,reference)
            frame=pd.DataFrame(reference,columns=catalog.id)
            nc=numerai_corr(frame,pd.Series(y,name='target')).fillna(0).to_numpy()
            nm=correlation_contribution(frame,pd.Series(meta,name='meta'),pd.Series(y,name='target')).to_numpy()
            np.testing.assert_allclose(corr,nc,atol=2e-7,rtol=2e-6);np.testing.assert_allclose(mmc,nm,atol=1e-12,rtol=1e-10)
            checks.append(dict(date=str(date),arms=len(catalog),assets=len(idx),max_corr_error=float(abs(corr-nc).max()),max_mmc_error=float(abs(mmc-nm).max())))
        if d%25==0:S.save(ROOT/'status.json',dict(stage='scoring',date=str(date),completed_dates=d+1,total_dates=len(groups),arms=len(catalog),time=time.time()))
    assert np.isfinite(scores).all()
    # Synthetic half-rank ties, singleton ranks and member-order permutations.
    toy=np.array([[0,2,1],[1,1,1],[2,0,1],[1,1,1]],float)
    nums=(2*rankdata(toy,axis=0)-1).astype(np.int64)
    a=S.rank_average_blends(nums,np.ones((3,1),dtype=np.int64),np.array([3]))
    b=S.rank_average_blends(nums[:,::-1],np.ones((3,1),dtype=np.int64),np.array([3]))
    np.testing.assert_array_equal(a,b);assert a[0,0]==a[1,0]==a[2,0]==a[3,0]
    payout=.1*scores[:,:,0]+scores[:,:,1]
    inherited=pd.read_csv(SOURCE/'daily_payout_selected.csv',index_col=0,parse_dates=True)
    pd.testing.assert_index_equal(dates,pd.DatetimeIndex(inherited.index),check_names=False)
    np.testing.assert_allclose(payout[:,:10],inherited[keys].to_numpy(),atol=1e-12,rtol=1e-10)
    for name in ('corr','mmc','payout','sharpe.p','maxdd.p'):
        catalog[name]=[S.payout_metrics(scores[:,j,0],scores[:,j,1])[name] for j in range(len(catalog))]
    catalog['cumulative_payout']=payout.sum(0);catalog['dates']=len(groups);catalog['asset_dates']=len(base)
    catalog=catalog.sort_values(['payout','id'],ascending=[False,True],kind='stable').reset_index(drop=True);catalog['rank']=np.arange(1,len(catalog)+1)
    ensemble=catalog[catalog.kind.eq('Ensemble')];mixed=ensemble[ensemble.family.eq('Mixed')]
    best_standalone=catalog[catalog.kind.eq('Standalone')].iloc[0]
    catalog['delta_mean_vs_best_standalone']=catalog.payout-best_standalone.payout
    catalog.to_csv(ROOT/'all_results.csv',index=False);models.to_csv(ROOT/'member_mapping.csv',index=False)
    np.save(ROOT/'daily_corr_mmc.npy',scores)
    daily=pd.DataFrame(payout,index=dates,columns=[r['id'] for r in records]);daily.index.name='date';S.parquet(ROOT/'daily_payout.parquet',daily.reset_index())
    coverage=base.groupby('date').agg(assets=('ucid','nunique'),rows=('ucid','size'));coverage.to_csv(ROOT/'coverage_by_date.csv')
    top3=mixed.head(3)
    pure=pd.concat([ensemble[ensemble.family.eq(f)].head(1) for f in ('Momentum','Volatility')])
    best_singles=pd.concat([catalog[catalog.kind.eq('Standalone')&catalog.family.eq(f)].head(1) for f in ('Momentum','Volatility')])
    allten=ensemble[ensemble['size'].eq(10)]
    shown=pd.concat([top3,pure,allten,best_singles]).drop_duplicates('id');shown.to_csv(ROOT/'comparison_table.csv',index=False)
    daily[shown.id].to_csv(ROOT/'plot_daily_payout.csv');daily[shown.id].cumsum().to_csv(ROOT/'plot_cumulative_payout.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def plot(rows,filename,title):
        fig,ax=plt.subplots(figsize=(13,7),layout='constrained')
        for row in rows.itertuples():
            label=f'{row.id}: {row.members}'
            ax.plot(dates,daily[row.id].cumsum(),label=label,lw=2.3 if row.family=='Mixed' else 1.7,ls='--' if row.kind=='Standalone' else '-')
        ax.axhline(0,color='gray',lw=.6);ax.set_title(title+'\nTarget20 · identical shared cohort · exploratory ranking');ax.set_ylabel('Additive cumulative payout score');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left',ncol=2);fig.savefig(ROOT/filename,dpi=160);plt.close(fig)
    plot(shown,'ensemble_comparison.png','E229: top three mixed ensembles, family controls and all-ten blend')
    plot(catalog[catalog.kind.eq('Standalone')],'standalone_comparison.png','E229: all ten frozen standalone variants')
    bysize=ensemble.groupby(['size','family']).agg(ensembles=('id','size'),best_mean_payout=('payout','max'),median_mean_payout=('payout','median')).reset_index();bysize.to_csv(ROOT/'results_by_size.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for family,g in bysize.groupby('family'):
        ax.plot(g['size'],g.best_mean_payout,marker='o',label=family+' best');ax.plot(g['size'],g.median_mean_payout,ls='--',label=family+' median')
    ax.axhline(best_standalone.payout,color='black',ls=':',label='Best standalone');ax.set_xlabel('Ensemble members');ax.set_ylabel('Mean daily payout score');ax.set_title('E229: exhaustive ensemble sweep by family and size');ax.grid(alpha=.2);ax.legend(ncol=2);fig.savefig(ROOT/'ensemble_size_sweep.png',dpi=160);plt.close(fig)
    assert all(S.sha(p)==h for p,h in hashes.items())
    audit=dict(outcome='PASS WITH CONDITIONS',input_hashes=hashes,exact_keys_match=True,finite_common_predictions=True,imputation='none',source_cohort='E228 unchanged',asset_dates=len(base),dates=len(dates),distinct_assets=int(base.ucid.nunique()),native_checks=checks,all_subset_reconstruction=True,synthetic_tie_and_order_checks=True,standalone_e228_payout_parity=True,conditions=['All models were selected retrospectively in E228; this sweep and displayed winners use the same development history, not independent confirmation.','Covered-cohort research comparison; no full-universe routing, native-feature marginal or live claim.','Original source-vintage/pretraining limitations persist. Frozen residual variants may share a fitted head.'],ensembles=len(ensemble),mixed_ensembles=len(mixed))
    S.save(ROOT/'audit.json',audit)
    S.save(ROOT/'source_manifest.json',dict(experiment='E229',input_hashes=hashes,member_keys=keys,subsets=subsets,rank_policy='equal mean of within-date average ranks; exact integer midrank sums',scoring='target20;0.1 CORR + historical-meta proxy MMC',new_residualization=False))
    def table(frame):
        cols=['id','family','members','size','corr','mmc','payout','sharpe.p','maxdd.p','cumulative_payout']
        return frame[cols].rename(columns={'id':'ID','family':'Family','members':'Members','size':'N','corr':'CORR','mmc':'MMC','payout':'Mean payout','sharpe.p':'Sharpe','maxdd.p':'Max DD','cumulative_payout':'Cumulative'}).to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body=f'<h1>E229 — Momentum and volatility ensembles</h1><p>Every equal-weight prediction-rank blend of 2–10 E228 selected standalone variants:1,013 ensembles (961 mixed;26 momentum-only;26 volatility-only), plus10 standalone controls. No model refits or new residualization. Members retain their original raw/residual transformations. Ensembles are scored from blended predictions, rather than averaged member payouts.</p><p>{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}; {len(dates)} dates; {len(base):,} asset-dates; {base.ucid.nunique()} distinct assets; {coverage.assets.min()}–{coverage.assets.max()} assets/date. All arms use exactly the E228 matched cohort and target20/historical meta. Payout=0.1 CORR+proxy MMC. Cumulative curves are additive score sums; Sharpe uses population daily standard deviation and is unannualized. Max DD includes an initial zero.</p>'
    body+='<p>Top-five source selection and ensemble ranking use this same retrospective history. The top results are exploratory and do not establish an out-of-sample improvement. These scores are not investment returns.</p><h2>Top mixed ensembles and comparison controls</h2>'+table(shown)+'<img src="ensemble_comparison.png">'
    body+='<h2>Top ten mixed ensembles</h2>'+table(mixed.head(10))+'<h2>All standalone controls</h2>'+table(catalog[catalog.kind.eq('Standalone')])+'<img src="standalone_comparison.png">'
    body+='<h2>Member mapping</h2>'+models[['key','family','label','view']].to_html(index=False)+'<h2>Ensemble size sweep</h2><img src="ensemble_size_sweep.png">'+bysize.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body+='<h2>Audit and interpretation</h2><p>PASS WITH CONDITIONS. Exact permanent-ID/date and symbol equality, finite predictions with no filling, unchanged source hashes, native scoring parity on first/middle/last dates for all1,023 arms, independent subset reconstruction, exact-tie/member-order tests and E228 standalone-score parity passed. Shared coverage is inherited from E228; no full-universe routing or marginal feature attribution is claimed. Historical vendor vintages and pretraining overlap remain uncertified. Complete results and daily score artifacts accompany this report.</p>'
    page=ROOT/'E229_momentum_volatility_ensembles.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:9px;break-inside:avoid}td,th{border:1px solid #ccc;padding:4px}img{display:block;max-width:100%;max-height:160mm;object-fit:contain;margin:auto;break-inside:avoid}td:nth-child(n+5){white-space:nowrap}</style></head><body>'+body+'</body></html>')
    subprocess.run([S.PYTHON,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=300)
    for row in catalog.itertuples():S.save(ROOT/'fits'/row.id/'COMPLETE.json',dict(task=dict(tag=row.id,model='rank_ensemble'),members=row.members))
    S.save(ROOT/'COMPLETE.json',dict(experiment='E229',status='success',arms=len(catalog),ensemble_count=len(ensemble),report=str(page.with_suffix('.pdf')),pdf_sha256=S.sha(page.with_suffix('.pdf')),audit='PASS WITH CONDITIONS',time=time.time()))
    S.save(ROOT/'status.json',dict(stage='complete',time=time.time()))
    print(shown[['id','members','payout','sharpe.p','maxdd.p']].to_string(index=False),flush=True)

if __name__=='__main__':run()
