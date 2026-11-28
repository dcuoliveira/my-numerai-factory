"""E233: Chronos-2 / t0 / Timestral hidden states on volatility297 (TS + TS->CS + CS bins),
per-asset design-C ridge under all six H standardizations, compared with E212 (TS99 states) and E226.

    python3 -m experiments.e233_volatility297_tsfm build       # dataset
    python3 -m experiments.e233_volatility297_tsfm supervise   # smokes -> GPU extraction -> heads -> score
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse
import json
import shlex
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import e212_volatility99_tsfm as S
from experiments import e226_volatility_cs_ridge as C
from experiments.e211_cross_target_tsfm import fold_panel
from src.analysis import tsfm_momentum_study as N
from src.data_mani import build_e209_volatility99 as B
from src.cluster import budget,hosts,manager

REPO=S.REPO
ROOT=REPO/'results/reports/crypto/e233_volatility297_tsfm'
SCR=ROOT/'cache'
DATA=REPO/'data/crypto/v2.0-e233-ts-tscs-cs-volatility297'
DATASET=str(DATA.relative_to(REPO/'data'))
TSFMS=['chronos2','t0','timestral']
NORMS=['pooled','asset','cross_section','rolling','robust','l2']
HORIZON=20
E212=S.ROOT;E226=C.ROOT
PY='/home/danielco/miniconda3/bin/python3'
save,parquet,sha=S.save,S.parquet,S.sha

# Point the frozen E212 components at this study. They read these module globals at call time.
S.ROOT=ROOT;S.SCR=SCR;S.WALLS={HORIZON:35};S.MODELS=list(TSFMS)


def build():
    """E226 dataset (TS99 + CS99) plus TS->CS99: the E209 TS bins ranked across assets per date."""
    if (DATA/'manifest.json').exists():
        m=json.loads((DATA/'manifest.json').read_text())
        for name,h in m['output_hashes'].items():assert sha(DATA/name)==h,name
        return m
    src=json.loads((C.DATA/'manifest.json').read_text())
    for name,h in src['output_hashes'].items():assert sha(C.DATA/name)==h,name
    base=pd.read_parquet(C.DATA/'train.parquet');cov=pd.read_parquet(C.DATA/'coverage.parquet');pd.testing.assert_index_equal(base.index,cov.index)
    keys=base.reset_index();ts=B.feature_names();cs=[n.replace('_ts_','_cs_') for n in ts];tscs=[n.replace('_ts_','_tscs_') for n in ts]
    values=base[ts].to_numpy(float);values[~cov.source_eligible.to_numpy()]=np.nan
    x=C.cs_bins(values,keys.date.to_numpy())
    out=base.join(pd.DataFrame(x,index=base.index,columns=tscs))
    names=ts+tscs+cs;out=out[[c for c in base.columns if not c.startswith('feature_')]+names]
    masks={'TS99':np.isfinite(out[ts]).all(1).to_numpy(),'TSCS99':np.isfinite(out[tscs]).all(1).to_numpy(),'CS99':np.isfinite(out[cs]).all(1).to_numpy()}
    covered=masks['TS99']&masks['TSCS99']&masks['CS99']
    assert np.array_equal(covered,cov.covered.to_numpy()),'TS->CS must be finite wherever E226 TS+CS is'
    assert np.isin(x[np.isfinite(x)],[0,.25,.5,.75,1]).all()
    # TS->CS is a pure per-date function of the TS bins: truncating the panel leaves earlier values unchanged.
    keep=keys.date.le(pd.Timestamp('2024-06-30')).to_numpy()
    np.testing.assert_array_equal(x[keep],C.cs_bins(values[keep],keys.date.to_numpy()[keep]))
    DATA.mkdir(parents=True,exist_ok=True)
    B.store_parquet(DATA/'train.parquet',out)
    meta=cov.copy();meta['TSCS99']=masks['TSCS99'];meta['covered']=covered
    B.store_parquet(DATA/'coverage.parquet',meta)
    B.store_parquet(DATA/'feature_availability.parquet',out[names].notna())
    shutil.copy2(C.DATA/'historical_meta_models.parquet',DATA/'historical_meta_models.parquet')
    for f in ['source_identity.csv','source_provenance.json']:shutil.copy2(C.DATA/f,DATA/f)
    save(DATA/'features.json',dict(feature_sets={'all':names,'TS99':ts,'TSCS99':tscs,'CS99':cs}))
    summary=[dict(information_set=k,covered_rows=int(v.sum()),covered_assets=int(keys.loc[v,'ucid'].nunique())) for k,v in {**masks,'all297':covered}.items()]
    pd.DataFrame(summary).to_csv(DATA/'coverage_summary.csv',index=False)
    manifest=dict(experiment='E233',dataset=DATASET,n_features=297,features=names,feature_sets={'TS99':ts,'TSCS99':tscs,'CS99':cs},index=['symbol','date'],feature_dtype='float32',
        rows=len(out),assets=int(keys.ucid.nunique()),target=src['target'],date_min=src['date_min'],date_max=src['date_max'],missing_value_policy='coverage_required',
        processing={'TS99':'unchanged E209 own-history five bins','TSCS99':'E209 TS bins of source-eligible rows, per-date per-feature average midranks, (rank-.5)/n, cuts .05/.25/.75/.95, min 2 assets','CS99':'unchanged E226 CS bins of raw D-1 volatility'},
        source_manifest_sha256=sha(C.DATA/'manifest.json'),builder_sha256=sha(__file__),
        output_hashes={p.name:sha(p) for p in DATA.iterdir() if p.suffix in ['.parquet','.csv','.json']},conditions=src.get('conditions',[]))
    save(DATA/'manifest.json',manifest)
    (DATA/'README.md').write_text('# E233 volatility297\n\nE209 TS99 + TS->CS99 (TS bins ranked across assets per date) + E226 CS99. Covered = all 297 finite and source-eligible.\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    return manifest


def prepare():
    build()
    if not (ROOT/'PREPARED.json').exists():
        S.prepare(DATASET,TSFMS)
        c=json.loads((ROOT/'config.json').read_text());c.update(experiment='E233',runner_sha256=sha(__file__),normalizations={m:NORMS for m in TSFMS},training_targets=[HORIZON],evaluation_targets=[HORIZON])
        save(ROOT/'config.json',c)
    return S.verify()


def fit_stream(model,norm):
    c=prepare();name=f'{model}_{norm}_t{HORIZON}' if model!='feature_ridge' else f'feature_ridge_t{HORIZON}';destination=ROOT/'fits'/name
    if (destination/'COMPLETE.json').exists():return
    p=pd.read_parquet(ROOT/'panel.parquet');outputs=[];cache={}
    for block,start in enumerate(range(200,p.step.max()+1,130)):
        part=fold_panel(p,start,HORIZON)
        if part is None:continue
        q,protocol=part;root=ROOT/'blocks'/name/f'f{block:02d}';scratch=SCR/'blocks'/name/f'f{block:02d}'
        if (root/'COMPLETE.json').exists():outputs.append(pd.read_parquet(root/'predictions.parquet'));continue
        root.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True);save(root/'protocol.json',protocol)
        forecast=q.loc[q.test,['row','ucid','symbol','date']].copy();values=np.full(len(q),np.nan)
        if model=='feature_ridge':values,_=S.feature_baseline(q,HORIZON,scratch)
        else:
            choices=[]
            for window in (N.ROLLING if norm=='rolling' else [126]):
                pred,records,params,info=S.solve_head(q,HORIZON,model,norm,window,'selection',root,scratch,representation_cache=cache)
                if records:choices.append((np.mean([r['mse'] for r in records]),window,records,params,info))
            if choices:
                loss,window,records,params,info=min(choices,key=lambda x:(x[0],x[1]))
                save(root/'selection.json',dict(window=window,loss=loss,assets=records,candidates=[dict(window=x[1],loss=x[0]) for x in choices]))
                values,records,_,info=S.solve_head(q,HORIZON,model,norm,window,'final',root,scratch,params,representation_cache=cache)
                save(root/'parameters.json',dict(assets=records,design=info))
        forecast['prediction']=values[q.test.to_numpy()]
        parquet(root/'predictions.parquet',forecast);save(root/'COMPLETE.json',dict(predictions=int(forecast.prediction.notna().sum())))
        outputs.append(forecast);print('BLOCK',name,block,int(forecast.prediction.notna().sum()),flush=True)
    result=pd.concat(outputs,ignore_index=True);assert result.row.is_unique
    parquet(destination/'predictions.parquet',result)
    save(destination/'COMPLETE.json',dict(task=dict(tag=name,model=model,norm=norm),blocks=len(outputs),predictions=int(result.prediction.notna().sum()),time=time.time()))


def streams():
    return [(m,n) for m in TSFMS for n in NORMS]+[('feature_ridge','asset')]


def score():
    from scipy.stats import rankdata
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=prepare();p=pd.read_parquet(ROOT/'panel.parquet');save(ROOT/'status.json',dict(stage='scoring',time=time.time()))
    key=lambda f:pd.MultiIndex.from_arrays([f.ucid.astype(str),pd.to_datetime(f.date)])
    pidx=key(p);cols={};info={}
    for m,n in streams():
        name=f'{m}_{n}_t{HORIZON}' if m!='feature_ridge' else f'feature_ridge_t{HORIZON}'
        f=pd.read_parquet(ROOT/'fits'/name/'predictions.parquet');label=f'E233 {m} {n}' if m!='feature_ridge' else 'E233 feature ridge 297'
        cols[label]=pd.Series(f.prediction.to_numpy(),index=key(f)).reindex(pidx).to_numpy();info[label]=dict(study='E233',tsfm=m,norm=n if m!='feature_ridge' else 'asset (features)',inputs='volatility297')
    for m in TSFMS+['feature_ridge']:
        f=pd.read_parquet(E212/'fits'/f'{m}_t{HORIZON}'/'predictions.parquet');label=f'E212 {m} {S.NORMS[m]}' if m!='feature_ridge' else 'E212 feature ridge TS99'
        cols[label]=pd.Series(f.prediction.to_numpy(),index=key(f)).reindex(pidx).to_numpy();info[label]=dict(study='E212',tsfm=m,norm=S.NORMS[m] if m!='feature_ridge' else 'asset (features)',inputs='volatility99 TS')
    for arm in ['CS99','TS_CS198']:
        f=pd.read_parquet(E226/'fits'/f'{arm}_t{HORIZON}'/'predictions.parquet');label=f'E226 feature ridge {arm}'
        cols[label]=pd.Series(f.prediction.to_numpy(),index=key(f)).reindex(pidx).to_numpy();info[label]=dict(study='E226',tsfm='feature_ridge',norm='asset (features)',inputs=arm)
    names=list(cols);x=np.column_stack([cols[k] for k in names]);y=p[f'target_binned_return_{HORIZON}'].to_numpy();meta=p.meta_model.to_numpy()
    ok=np.isfinite(x).all(1)&np.isfinite(y)&np.isfinite(meta)
    own=pd.DataFrame([dict(stream=k,rows=int(np.isfinite(cols[k]).sum()),dates=int(p.loc[np.isfinite(cols[k]),'date'].nunique())) for k in names])
    q=p.loc[ok].reset_index(drop=True);xx=x[ok];groups=list(q.groupby('date',sort=True).indices.items());dates=pd.DatetimeIndex([d for d,_ in groups])
    corr=np.zeros((len(groups),len(names)));mmc=corr.copy()
    for d,(_,idx) in enumerate(groups):
        r=(rankdata(xx[idx],axis=0,method='average')-.5)/len(idx)
        cc,mm=S.matrix_scores(r,y[ok][idx],meta[ok][idx]);corr[d]=np.nan_to_num(cc,nan=0.);mmc[d]=mm
    pay=.1*corr+mmc;rows=[]
    for j,k in enumerate(names):
        m=S.payout_metrics(corr[:,j],mmc[:,j]);rows.append(dict(stream=k,**info[k],corr=m['corr'],mmc=m['mmc'],payout=m['payout'],**{'sharpe.p':m['sharpe.p'],'maxdd.p':m['maxdd.p']},cumulative=pay[:,j].sum(),dates=len(dates),assets=int(q.ucid.nunique())))
    table=pd.DataFrame(rows).sort_values('payout',ascending=False);table.to_csv(ROOT/'summary.csv',index=False);own.to_csv(ROOT/'own_coverage.csv',index=False)
    pd.DataFrame(pay,index=dates,columns=names).to_csv(ROOT/'daily_payout.csv');pd.DataFrame(corr,index=dates,columns=names).to_csv(ROOT/'daily_corr.csv')
    pivot=table[table.study.eq('E233')&table.tsfm.ne('feature_ridge')].pivot(index='norm',columns='tsfm',values='payout');pivot.to_csv(ROOT/'payout_by_norm.csv')
    colors={'chronos2':'tab:blue','t0':'tab:orange','timestral':'tab:green','feature_ridge':'black'}
    figs=[]
    for t in TSFMS:
        fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
        for k in names:
            i=info[k]
            if i['tsfm']!=t:continue
            ax.plot(dates,pay[:,names.index(k)].cumsum(),label=k,lw=2.6 if i['study']=='E212' else 1.3,ls='--' if i['study']=='E212' else '-')
        ax.set_title(f'{t}: E233 volatility297 states, six standardizations, vs E212 TS99 states');ax.set_ylabel('cumulative payout score');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fn=f'cum_{t}.png';fig.savefig(ROOT/fn,dpi=150);plt.close(fig);figs.append(fn)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    best=table[table.study.eq('E233')&table.tsfm.ne('feature_ridge')].groupby('tsfm').head(1)
    for k in list(best.stream)+[k for k in names if info[k]['study']!='E233' or info[k]['tsfm']=='feature_ridge']:
        i=info[k];ax.plot(dates,pay[:,names.index(k)].cumsum(),label=k,color=colors[i['tsfm']],ls={'E233':'-','E212':'--','E226':':'}[i['study']],lw=2 if i['study']=='E233' else 1.3)
    ax.set_title('Best E233 standardization per TSFM vs E212 states and feature ridges');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(ROOT/'cum_overview.png',dpi=150);plt.close(fig)
    show=lambda df:df[['stream','study','inputs','tsfm','norm','corr','mmc','payout','sharpe.p','maxdd.p','cumulative']].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body=(f'<h1>E233: TSFM hidden states on volatility297 (TS + TS→CS + CS bins) vs E212</h1>'
        f'<p>Chronos-2, t0 and Timestral states extracted one asset at a time on the 297 volatility features (99 E209 own-history TS bins, the same TS bins re-ranked across assets per date, and 99 E226 cross-sectional bins of raw volatility), training target20. Per-asset ridge on design-C concatenated states over a fixed training asset vocabulary, alpha tuned from scratch by validation MSE in every block (and the rolling window 63/126/252 for rolling), under all six standardizations. Same E212 block schedule (504-date history, 63 validation, 130-date forecast blocks, 35-day label wall), so E212 rows are directly comparable.</p>'
        f'<p>Scored on rows where every stream predicts: {len(dates)} dates ({dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}), {len(q):,} asset-dates, {q.ucid.nunique()} assets. Payout = 0.1 CORR + MMC against the historical meta model, target20, raw predictions (no residualization). E212 rows use their recommended standardization only; E226 rows are direct feature ridges, not hidden states.</p>'
        +'<h2>Payout by standardization (E233)</h2>'+pivot.to_html(float_format=lambda v:f'{v:.5f}')
        +'<h2>All streams</h2>'+show(table)+''.join(f'<img src="{f}">' for f in ['cum_overview.png']+figs)
        +'<p>Exploratory development comparison; the feature families and history were previously inspected. No live, full-universe or investment-return claim.</p>')
    page=ROOT/'E233_volatility297_tsfm.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:8px;page-break-inside:avoid}td,th{border:1px solid #ccc;padding:3px}img{width:100%}</style></head><body>'+body+'</body></html>')
    r=subprocess.run(['ssh','-o','BatchMode=yes','lovelace',f'cd {REPO} && {PY} -m src.analysis.html_pdf {page}'],capture_output=True,text=True,timeout=900)
    save(ROOT/'COMPLETE.json',dict(experiment='E233',status='success' if r.returncode==0 else 'pdf_failed',report=str(page.with_suffix('.pdf')),stderr=r.stderr[-2000:],time=time.time()))
    save(ROOT/'status.json',dict(stage='complete',time=time.time()));print(table.drop(columns=['inputs']).to_string(index=False),flush=True)


def supervise():
    prepare();p=pd.read_parquet(ROOT/'panel.parquet');owned={};attempts={}
    def head_job(m,n):
        return budget.Job(f'e233_head_{m}_{n}',shlex.join([PY,'-u','-m','experiments.e233_volatility297_tsfm','fit','--model',m,'--norm',n]),cores=4,ram_gb=40)
    try:
        while True:
            # Probe every host over ssh, including this one: a local child inherits this process's
            # CPU affinity and under-reports cores (same fix as e231_supervisor_cpu).
            with ThreadPoolExecutor(max_workers=16) as pool:
                probes=hosts.reachable(list(pool.map(lambda n:hosts.probe(n,8,local_name='__remote_probe__'),[n for n in hosts.roster() if n not in {'westeros','imeusp'}])))
            statuses=manager.status(names=[h.name for h in probes],timeout=4);active={r['unit'].removesuffix('.service') for st in statuses.values() for r in st if r['state'].startswith('active')}
            for u in list(owned):
                if 'nmr-'+u not in active and u not in active:owned.pop(u)
            snapshots={}
            for h in [h for h in probes if h.gpus]:
                name,blocked,readings,seen=S.gpu_snapshot(h.name);snapshots[name]=(blocked,readings,seen)
            jobs=[];stage='gpu_smokes'
            if not (ROOT/'EMBEDDINGS_READY.json').exists():
                smokes=[m for m in TSFMS if not (S.extraction_dir(HORIZON,m)/'SMOKE_COMPLETE.json').exists()]
                if smokes:jobs=[budget.Job(f'e233_gpu_smoke_{m}',S.worker_command(m,HORIZON,True),cores=1,ram_gb=16,gpus=1) for m in smokes]
                else:
                    stage='gpu_extracting';todo=[m for m in TSFMS if not S.extraction_complete(p,HORIZON,m)]
                    if todo:jobs=[budget.Job(f'e233_gpu_extract_{m}',S.worker_command(m,HORIZON,False),cores=1,ram_gb=24,gpus=1) for m in todo]
                    else:S.collect();stage='collected'
            else:
                stage='fitting_heads'
                for m,n in streams():
                    name=f'{m}_{n}_t{HORIZON}' if m!='feature_ridge' else f'feature_ridge_t{HORIZON}'
                    if not (ROOT/'fits'/name/'COMPLETE.json').exists():jobs.append(head_job(m,n))
                if not jobs:
                    save(ROOT/'status.json',dict(stage='scoring',time=time.time()));score();return
            pending=[]
            for j in jobs:
                if j.name in owned:continue
                if attempts.get(j.name,0)>=3:raise RuntimeError('three attempts exhausted: '+j.name+' (see logs/cluster/'+j.name+'.out)')
                pending.append(j)
            placed,unplaced=budget.distribute(pending,S.admission_hosts(probes,{},active,snapshots),share=budget.SHARE,gpu_slots=1)
            for pl in placed:
                ok,note=manager.launch(pl);print('LAUNCH',ok,note,flush=True)
                if ok:owned[pl.job.name]=dict(host=pl.host,unit=pl.unit,gpus=list(pl.gpus));attempts[pl.job.name]=attempts.get(pl.job.name,0)+1
            prog=S.extraction_progress(json.loads((ROOT/'config.json').read_text())) if stage.startswith('gpu') else None
            save(ROOT/'status.json',dict(stage=stage,running=owned,waiting=[dict(job=j.name,reason=r) for j,r in unplaced],extraction=prog,heads_done=len(list((ROOT/'fits').glob('*/COMPLETE.json'))) if (ROOT/'fits').exists() else 0,time=time.time()))
            time.sleep(60)
    except Exception as exc:
        save(ROOT/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','prepare','fit','score','supervise']);ap.add_argument('--model');ap.add_argument('--norm',default='asset');a=ap.parse_args()
    if a.action=='fit':fit_stream(a.model,a.norm)
    else:globals()[a.action]()
