"""E232 full-sample expanding walk-forward of the six E231 learners on E230 feature134, plus all 57 rank ensembles.

Blocks of 130 test dates from 2022-07-01 to 2026-02-19. For a block whose first test date is T:
final fit = dates <= T-35d; validation = the last 126 final-fit dates; selection training = dates <= first
validation date - 35d. Per block an asset is eligible with >=60 training and >=20 validation usable rows.
Same learners, grids, seeds and scaling rules as E230 (ridge/MLP) and E231 (TabNet).
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse
import itertools
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import e230_dataset as D
from experiments import e212_volatility99_tsfm as S

ROOT=S.REPO/'results/reports/crypto/e232_feature134_walkforward'
FIRST_TEST=pd.Timestamp('2022-07-01');END=pd.Timestamp('2026-02-20');BLOCK=130;VALIDATION=126;WALL=35
ARMS=['asset_ridge','asset_mlp','panel_ridge','panel_mlp','asset_tabnet','panel_tabnet']
KEYS=dict(zip(ARMS,['AR','AM','PR','PM','AT','PT']))
SHARDS=dict(asset_ridge=1,asset_mlp=2,asset_tabnet=4,panel_ridge=1,panel_mlp=1,panel_tabnet=1)
CORES=dict(panel_tabnet=2)
HOSTS=['tolstoi','hopper','dostoievski','deepfour','hamilton','lovelace']
PY='/home/danielco/miniconda3/bin/python3'

def panel():
    p=pd.read_parquet(D.ROOT/'panel.parquet');meta=json.loads((D.ROOT/'PREPARED.json').read_text())
    assert S.sha(D.ROOT/'panel.parquet')==meta['output_hashes'][str(D.ROOT/'panel.parquet')]
    return meta,p

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'PREPARED.json').exists():
        old=json.loads((ROOT/'PREPARED.json').read_text())
        assert old['panel_sha256']==S.sha(D.ROOT/'panel.parquet') and old['runner_sha256']==S.sha(__file__),'inputs or runner changed since prepare'
        return old,dict(np.load(ROOT/'blocks.npz'))
    meta,p=panel();y=p[D.TARGET].to_numpy(float);covered=p.covered.to_numpy(bool);usable=covered&np.isfinite(y)
    dates=np.sort(p.loc[covered,'date'].unique());test=dates[(dates>=FIRST_TEST)&(dates<END)]
    starts=[test[i] for i in range(0,len(test),BLOCK)];ends=[test[min(i+BLOCK,len(test))-1] for i in range(0,len(test),BLOCK)]
    d=p.date.to_numpy();u=p.ucid.to_numpy();arrays={};blocks=[]
    for b,(t0,t1) in enumerate(zip(starts,ends)):
        final=usable&(d<=t0-np.timedelta64(WALL,'D'))
        fdates=np.sort(np.unique(d[final]));v0=fdates[-VALIDATION]
        val=final&(d>=v0);train=final&(d<=v0-np.timedelta64(WALL,'D'))
        cnt=pd.DataFrame(dict(u=u,tr=train,va=val)).groupby('u')[['tr','va']].sum()
        assets=sorted(cnt.index[(cnt.tr>=60)&(cnt.va>=20)])
        el=np.isin(u,assets);query=covered&el&(d>=t0)&(d<=t1)
        for name,mask in [('train',train&el),('validation',val&el),('final_fit',final&el),('query',query)]:arrays[f'{b}_{name}']=np.flatnonzero(mask)
        assert d[arrays[f'{b}_train']].max()+np.timedelta64(WALL,'D')<=v0
        assert d[arrays[f'{b}_final_fit']].max()+np.timedelta64(WALL,'D')<=t0
        blocks.append(dict(block=b,test_start=str(pd.Timestamp(t0).date()),test_end=str(pd.Timestamp(t1).date()),validation_start=str(pd.Timestamp(v0).date()),train_end=str(pd.Timestamp(d[arrays[f'{b}_train']].max()).date()),final_end=str(pd.Timestamp(d[arrays[f'{b}_final_fit']].max()).date()),assets=assets,**{f'n_{k}':len(arrays[f'{b}_{k}']) for k in ['train','validation','final_fit','query']}))
    np.savez_compressed(ROOT/'blocks.npz',**arrays)
    shards={}
    for blk in blocks:
        b=blk['block'];sizes={a:int((u[arrays[f'{b}_train']]==a).sum()*4+(u[arrays[f'{b}_final_fit']]==a).sum()) for a in blk['assets']}
        for arm in ARMS:
            if not arm.startswith('asset'):continue
            n=SHARDS[arm];parts=[[] for _ in range(n)];cost=[0]*n
            for a in sorted(blk['assets'],key=lambda a:(-sizes[a],a)):j=int(np.argmin(cost));parts[j].append(a);cost[j]+=sizes[a]
            shards[f'{arm}|{b}']=parts
    prepared=dict(experiment='E232',panel_sha256=S.sha(D.ROOT/'panel.parquet'),runner_sha256=S.sha(__file__),features=meta['features'],blocks=blocks,shards=shards,protocol=__doc__)
    S.save(ROOT/'PREPARED.json',prepared)
    print(pd.DataFrame(blocks).drop(columns='assets').assign(assets=[len(b['assets']) for b in blocks]).to_string(index=False),flush=True)
    return prepared,arrays

def tasks(plan):
    out=[]
    for blk in plan['blocks']:
        b=blk['block']
        for arm in ARMS:
            for s in range(SHARDS[arm]):out.append((f'{arm}_b{b}_s{s}',f'fit --arm {arm} --block {b} --shard {s}',CORES.get(arm,1)))
    return out

def fit(arm,block,shard):
    from experiments import e230_models as M
    from experiments import e231_tabnet_ensembles as T
    plan,arrays=prepare();meta,p=panel();x=p[plan['features']].to_numpy(float);y=p[D.TARGET].to_numpy(float);u=p.ucid.to_numpy()
    name=f'{arm}_b{block}_s{shard}';done=ROOT/'workers'/name/'COMPLETE.json'
    if done.exists():return
    idx={k:arrays[f'{block}_{k}'] for k in ['train','validation','final_fit','query']}
    units=['panel'] if arm.startswith('panel') else plan['shards'][f'{arm}|{block}'][shard];family=arm.split('_')[1]
    for n,unit in enumerate(units,1):
        rows=idx if unit=='panel' else {k:v[u[v]==unit] for k,v in idx.items()}
        directory=ROOT/'models'/arm/f'b{block}'/unit
        if (directory/'record.json').exists():
            saved=dict(np.load(directory/'rows.npz'))
            for k in rows:np.testing.assert_array_equal(saved[k],rows[k])
            continue
        tick=time.time()
        if family=='tabnet':T.fit_one(x,y,rows['train'],rows['validation'],rows['final_fit'],rows['query'],directory,unit=='panel')
        else:M.train_one(family,x,y,rows['train'],rows['validation'],rows['final_fit'],rows['query'],directory)
        print('FIT',name,n,len(units),unit,round(time.time()-tick,1),flush=True)
    S.save(done,dict(worker=name,units=units,time=time.time()))

def probes():
    from src.cluster import hosts
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:return hosts.reachable(list(pool.map(hosts.probe,HOSTS)))

def supervise():
    from src.cluster import budget,manager
    plan,_=prepare();todo=tasks(plan);owned={};seen={}
    while True:
        finished=[n for n,_,_ in todo if (ROOT/'workers'/n/'COMPLETE.json').exists()]
        if len(finished)==len(todo):break
        snap=manager.status(HOSTS);active={r['unit'].removesuffix('.service') for st in snap.values() for r in st if r['state'].startswith('active')}
        for n,pl in owned.items():
            if pl.unit not in active and n not in finished:
                seen.setdefault(n,time.time())
                if time.time()-seen[n]>120:raise RuntimeError('worker stopped without completion: '+n)
            else:seen.pop(n,None)
        pending=[budget.Job('e232_'+n,PY+' -m experiments.e232_walkforward '+c,cores=k,ram_gb=6) for n,c,k in todo if n not in finished and n not in owned]
        if pending:
            placed,_=budget.distribute(pending,probes())
            for pl in placed:
                ok,note=manager.launch(pl)
                if not ok:raise RuntimeError(note)
                owned[pl.job.name.removeprefix('e232_')]=pl
        S.save(ROOT/'status.json',dict(stage='fitting',completed=len(finished),total=len(todo),running=sorted(n for n,v in owned.items() if v.unit in active and n not in finished),queued=len([1 for n,_,_ in todo if n not in finished and n not in owned]),time=time.time()))
        time.sleep(30)
    finish()

def finish():
    from scipy.stats import rankdata
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan,arrays=prepare();meta,p=panel();S.save(ROOT/'status.json',dict(stage='scoring',time=time.time()))
    preds={}
    for arm in ARMS:
        v=np.full(len(p),np.nan)
        for blk in plan['blocks']:
            b=blk['block'];q=arrays[f'{b}_query'];u=p.ucid.to_numpy()
            for unit in (['panel'] if arm.startswith('panel') else blk['assets']):
                directory=ROOT/'models'/arm/f'b{b}'/unit;rows=dict(np.load(directory/'rows.npz'))
                exp=q if unit=='panel' else q[u[q]==unit];np.testing.assert_array_equal(rows['query'],exp)
                vals=np.load(directory/'predictions.npy');assert np.isfinite(vals).all() and np.isnan(v[exp]).all();v[exp]=vals
        preds[arm]=v
    out=p[['row','ucid','symbol','date',D.TARGET,'meta_model']].copy()
    for arm in ARMS:out[KEYS[arm]]=preds[arm]
    S.parquet(ROOT/'walkforward_forecasts.parquet',out)
    keys=[KEYS[a] for a in ARMS];common=out[np.isfinite(out[keys]).all(1)&np.isfinite(out[D.TARGET])].reset_index(drop=True)
    subsets=[(j,) for j in range(6)]+[s for n in range(2,7) for s in itertools.combinations(range(6),n)]
    membership=np.zeros((6,len(subsets)),np.int64)
    for j,s in enumerate(subsets):membership[list(s),j]=1
    sizes=membership.sum(0);ids=[keys[j] if j<6 else f'B{j-5:03d}' for j in range(len(subsets))];members=['+'.join(keys[k] for k in s) for s in subsets]
    groups=list(common.groupby('date',sort=True).indices.items());dates=pd.DatetimeIndex([d for d,_ in groups]);corr=np.full((len(groups),len(subsets)),np.nan);mmc=corr.copy();x=common[keys].to_numpy()
    for d,(date,idx) in enumerate(groups):
        nums=(2*rankdata(x[idx],axis=0,method='average')-1).astype(np.int64);bl=S.rank_average_blends(nums,membership,sizes)
        yy=common[D.TARGET].iloc[idx].to_numpy();m=common.meta_model.iloc[idx].to_numpy();ok=np.isfinite(m).all()
        c,mm=S.matrix_scores(bl,yy,np.where(np.isfinite(m),m,.5));corr[d]=np.nan_to_num(c,nan=0.)
        if ok:mmc[d]=mm
    hasmeta=np.isfinite(mmc).all(1);pay=.1*corr+mmc
    def stats(s):
        s=s[np.isfinite(s)];cum=np.cumsum(s);return s.mean(),s.mean()/s.std(),float(np.max(np.maximum.accumulate(cum)-cum)),len(s)
    rows=[]
    for j in range(len(subsets)):
        cm,cs,cd,cn=stats(corr[:,j]);pm,ps,pd_,pn=stats(pay[:,j]);_,_,_,_=stats(corr[hasmeta,j])
        rows.append(dict(id=ids[j],kind='Standalone' if j<6 else 'Ensemble',members=members[j],size=len(subsets[j]),corr_full=cm,corr_sharpe_full=cs,corr_dd_full=cd,n_full=cn,corr_meta=np.mean(corr[hasmeta,j]),mmc_meta=np.nanmean(mmc[:,j]),payout_meta=pm,payout_sharpe_meta=ps,payout_dd_meta=pd_,n_meta=pn))
    table=pd.DataFrame(rows).sort_values(['payout_meta','id'],ascending=[False,True]);table.to_csv(ROOT/'all_results.csv',index=False)
    pd.DataFrame(corr,index=dates,columns=ids).to_csv(ROOT/'daily_corr.csv');pd.DataFrame(pay,index=dates,columns=ids).to_csv(ROOT/'daily_payout.csv')
    own=[]
    for k in keys:
        q=out[np.isfinite(out[k])&np.isfinite(out[D.TARGET])];cc=q.groupby('date').apply(lambda g:np.corrcoef(rankdata(g[k]),g[D.TARGET])[0,1] if len(g)>2 else np.nan)
        own.append(dict(model=k,rows=len(q),dates=q.date.nunique(),first=str(q.date.min().date()),assets=q.ucid.nunique()))
    pd.DataFrame(own).to_csv(ROOT/'own_coverage.csv',index=False)
    ens=table[table.kind.eq('Ensemble')];stand=table[table.kind.eq('Standalone')]
    e231=['B002','B022','B021','B057'];shown=pd.concat([ens.head(5),ens[ens.id.isin(e231)],stand]).drop_duplicates('id');shown.to_csv(ROOT/'comparison_table.csv',index=False)
    labels={'AR':'Ridge per asset','AM':'MLP per asset','PR':'Pooled ridge','PM':'Pooled MLP','AT':'TabNet per asset','PT':'Pooled TabNet'}
    for fname,series,mask,title in [('cumulative_corr.png',corr,np.ones(len(dates),bool),'cumulative CORR, full walk-forward window'),('cumulative_payout.png',pay,hasmeta,'cumulative payout (0.1 CORR + MMC), meta-model window')]:
        fig,ax=plt.subplots(figsize=(13,7),layout='constrained')
        for r in shown.itertuples():
            j=ids.index(r.id);ax.plot(dates[mask],np.cumsum(series[mask,j]),label=labels.get(r.id,r.id+': '+r.members),lw=2.2 if r.kind=='Ensemble' else 1.4,ls='-' if r.kind=='Ensemble' else '--')
        for blk in plan['blocks']:ax.axvline(pd.Timestamp(blk['test_start']),color='grey',lw=.5,alpha=.4)
        ax.set_title('E232: '+title);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.savefig(ROOT/fname,dpi=150);plt.close(fig)
    def html(df):
        cols=['id','members','corr_full','corr_sharpe_full','corr_dd_full','n_full','corr_meta','mmc_meta','payout_meta','payout_sharpe_meta','payout_dd_meta','n_meta']
        return df[cols].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    bt=pd.DataFrame(plan['blocks']).drop(columns='assets').assign(assets=[len(b['assets']) for b in plan['blocks']])
    body=(f'<h1>E232: full-sample walk-forward of the six E231 learners on feature134</h1>'
        f'<p>Expanding-window walk-forward in {len(plan["blocks"])} blocks of up to {BLOCK} test dates, {plan["blocks"][0]["test_start"]} to {plan["blocks"][-1]["test_end"]}. Each block selects hyperparameters on the last {VALIDATION} final-fit dates, then refits the chosen model fresh on all history up to the {WALL}-day label wall. Same grids, seeds and train-only scaling as E230 (ridge, MLP) and E231 (TabNet). Assets are eligible per block with at least 60 training and 20 validation rows.</p>'
        f'<p>Scored on rows where all six models predict: {len(dates)} dates, {len(common):,} asset-dates, {common.ucid.nunique()} assets. CORR covers the whole window. The historical meta model starts 2024-05-07, so MMC and payout cover only the {int(hasmeta.sum())} dates from then on. Sharpe is per period and unannualized; drawdowns are additive and positive.</p>'
        +'<h2>Blocks</h2>'+bt.to_html(index=False)+'<h2>Top ensembles, E231 picks, six standalones</h2>'+html(shown)+'<img src="cumulative_payout.png"><img src="cumulative_corr.png"><h2>All 63 rows</h2>'+html(table)
        +'<p>Ensembles are ranked on the displayed history and remain exploratory. Covered-sample comparison only; no full-universe, native22-marginal or live claim. Inherited vendor-vintage and coverage conditions from E205/E209 apply.</p>')
    page=ROOT/'E232_feature134_walkforward.html'
    page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;font-size:8px;page-break-inside:avoid}td,th{border:1px solid #ccc;padding:3px}img{width:100%}</style></head><body>'+body+'</body></html>')
    subprocess.run([PY,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=600)
    S.save(ROOT/'COMPLETE.json',dict(experiment='E232',status='success',report=str(page.with_suffix('.pdf')),time=time.time()));S.save(ROOT/'status.json',dict(stage='complete',time=time.time()))
    print(shown.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fit','supervise','finish']);ap.add_argument('--arm');ap.add_argument('--block',type=int);ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
    if a.action=='fit':fit(a.arm,a.block,a.shard)
    elif a.action=='prepare':prepare()
    else:globals()[a.action]()
