"""E231 TabNet on immutable E230 feature134 and exhaustive six-model rank ensembles."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments import e230_dataset as D
from experiments import e230_models as M
from experiments import e212_volatility99_tsfm as S
from src.cluster import hosts,budget,manager

ROOT=S.REPO/'results/reports/crypto/e231_volatility134_tabnet_ensembles'
GRID=[dict(n_d=w,n_a=w,n_steps=3,lambda_sparse=penalty,learning_rate=.01,max_epochs=40,seed=231) for w in [8,16] for penalty in [.001,.1]]
SHARDS=9
HOSTS=['tolstoi','hopper','dostoievski','deepfour','hamilton','lovelace']
PY='/home/danielco/miniconda3/bin/python3'

def identity():
    files=[Path(__file__),S.REPO/'experiments/e231_tabnet_model.py',D.ROOT/'PREPARED.json',D.ROOT/'panel.parquet',D.ROOT/'indices.npz',D.ROOT/'COMPLETE.json',D.ROOT/'independent_completion_audit.json']+[D.ROOT/'fits'/a/'predictions.parquet' for a in D.ARMS]
    return dict(hashes={str(p.resolve()):S.sha(p) for p in files},grid=GRID,shards=SHARDS,protocol='Frozen E230 eligible rows and target20 chronological masks; complete epochs, train-only scalers, fresh selected refit')

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);current=identity()
    if (ROOT/'PREPARED.json').exists():
        old=json.loads((ROOT/'PREPARED.json').read_text());assert old['identity']==current;return old
    meta,p,indices=M.load();assets=meta['assets'];rows=p.ucid.to_numpy();sizes={a:sum((rows[indices[k]]==a).sum()*mult for k,mult in [('train',4),('final_fit',1)]) for a in assets}
    shards=[[] for _ in range(SHARDS)];costs=[0]*SHARDS
    for asset in sorted(assets,key=lambda a:(-sizes[a],a)):
        slot=int(np.argmin(costs));shards[slot].append(asset);costs[slot]+=int(sizes[asset])
    prepared=dict(experiment='E231',identity=current,features=meta['features'],asset_shards=shards,costs=costs,assets=assets,conditions=meta['conditions'])
    S.save(ROOT/'PREPARED.json',prepared);S.save(ROOT/'candidates.json',GRID);return prepared

def fit_one(x,y,tr,va,ft,te,directory,panel):
    from experiments.e231_tabnet_model import CompleteTabNet
    directory.mkdir(parents=True,exist_ok=True);started=time.time();choices=[]
    def estimator(params):
        return Pipeline([('scale',StandardScaler()),('net',CompleteTabNet(**params,batch_size=1024 if panel else 256,virtual_batch_size=128,patience=0,device='cpu'))])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='No early stopping will be performed')
        warnings.filterwarnings('ignore',message='Device used : cpu')
        for j,params in enumerate(GRID):
            tick=time.time();model=estimator(params);model.fit(x[tr],y[tr]);pred=model.predict(x[va]);assert np.isfinite(pred).all()
            choices.append(dict(parameters=params,mse=float(np.mean((pred-y[va])**2)),coverage=model.named_steps['net'].coverage_,seconds=time.time()-tick))
            S.save(directory/'progress.json',dict(stage='search',completed_candidates=j+1,total_candidates=len(GRID),last_seconds=choices[-1]['seconds'],time=time.time()))
            del model
        selected=min(choices,key=lambda v:v['mse']);S.save(directory/'progress.json',dict(stage='final_refit',selected=selected['parameters'],time=time.time()))
        tick=time.time();model=estimator(selected['parameters']);model.fit(x[ft],y[ft]);values=model.predict(x[te]) if len(te) else np.empty(0);assert np.isfinite(values).all()
    path=directory/'model.joblib';temp=path.with_suffix('.tmp');joblib.dump(model,temp,compress=3);temp.replace(path);restored=joblib.load(path)
    np.testing.assert_allclose(restored.predict(x[te]) if len(te) else np.empty(0),values,atol=1e-7,rtol=1e-6)
    record=dict(selected=selected,choices=choices,final_coverage=model.named_steps['net'].coverage_,final_seconds=time.time()-tick,seconds=time.time()-started,model_sha256=S.sha(path),n_train=len(tr),n_validation=len(va),n_final=len(ft),n_query=len(te))
    np.savez_compressed(directory/'rows.npz',train=tr,validation=va,final_fit=ft,query=te);np.save(directory/'predictions.npy',values);S.save(directory/'record.json',record)
    return values,record

def fit(shard=None):
    meta,p,indices=M.load();plan=prepare();x=p[meta['features']].to_numpy(float);y=p[D.TARGET].to_numpy(float);assets=p.ucid.to_numpy()
    name='panel' if shard is None else f'shard_{shard:02d}';root=ROOT/'workers'/name
    if (root/'COMPLETE.json').exists():return
    allrecords=[]
    for asset in ['panel'] if shard is None else plan['asset_shards'][shard]:
        rows={k:idx if shard is None else idx[assets[idx]==asset] for k,idx in indices.items() if k in ['train','validation','final_fit','query']}
        directory=ROOT/'models'/('panel' if shard is None else 'asset_'+asset)
        if (directory/'record.json').exists():
            record=json.loads((directory/'record.json').read_text());saved=dict(np.load(directory/'rows.npz'))
            for k in rows:np.testing.assert_array_equal(saved[k],rows[k])
            assert S.sha(directory/'model.joblib')==record['model_sha256']
        else:_,record=fit_one(x,y,rows['train'],rows['validation'],rows['final_fit'],rows['query'],directory,shard is None)
        allrecords.append(asset);S.save(root/'progress.json',dict(completed=len(allrecords),total=1 if shard is None else len(plan['asset_shards'][shard]),latest=asset,latest_seconds=record['seconds'],time=time.time()));print('FIT',name,asset,record['seconds'],flush=True)
    S.save(root/'COMPLETE.json',dict(worker=name,assets=allrecords,identity=plan['identity'],time=time.time()))

def probes():
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:return hosts.reachable(list(pool.map(hosts.probe,HOSTS)))

def supervise():
    prepare();owned={};stopped_seen={};plan=prepare();tasks=[(f'shard_{j:02d}',f'fit --shard {j}',1) for j in range(SHARDS)]+[('panel','fit',2)]
    previous=ROOT/'placements.json'
    if previous.exists():
        byname={name:(cmd,cores) for name,cmd,cores in tasks}
        for name,v in json.loads(previous.read_text()).items():
            cmd,cores=byname[name];owned[name]=budget.Placement(budget.Job('e231_'+name,PY+' -m experiments.e231_tabnet_ensembles '+cmd,cores=cores,ram_gb=5),v['host'],tuple(v['cpus']))
    while True:
        finished=[name for name,_,_ in tasks if (ROOT/'workers'/name/'COMPLETE.json').exists()]
        if len(finished)==len(tasks):break
        snapshots=manager.status(HOSTS);active={r['unit'].removesuffix('.service') for states in snapshots.values() for r in states if r['state'].startswith('active')}
        failed=[name for name,placement in owned.items() if placement.unit not in active and name not in finished]
        for name in failed:
            stopped_seen.setdefault(name,time.time())
            if time.time()-stopped_seen[name]>90:raise RuntimeError('Worker stopped without completion: '+name)
        for name in list(stopped_seen):
            if name not in failed:stopped_seen.pop(name)
        pending=[budget.Job('e231_'+name,PY+' -m experiments.e231_tabnet_ensembles '+cmd,cores=cores,ram_gb=5) for name,cmd,cores in tasks if name not in finished and name not in owned]
        if pending:
            placements,unplaced=budget.distribute(pending,probes())
            for place in placements:
                okay,note=manager.launch(place)
                if not okay:raise RuntimeError(note)
                owned[place.job.name.removeprefix('e231_')]=place
        S.save(ROOT/'status.json',dict(stage='fitting',completed_workers=finished,total_workers=len(tasks),running_workers=[n for n,v in owned.items() if v.unit in active and n not in finished],queued_workers=[name for name,_,_ in tasks if name not in finished and name not in owned],time=time.time()))
        S.save(ROOT/'placements.json',{name:dict(host=v.host,cpus=v.cpus,cores=v.job.cores,unit=v.unit) for name,v in owned.items()})
        time.sleep(20)
    from experiments.e231_results import finish
    finish()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fit','supervise','finish']);ap.add_argument('--shard',type=int);a=ap.parse_args()
    if a.action=='fit':fit(a.shard)
    elif a.action=='finish':
        from experiments.e231_results import finish
        finish()
    else:globals()[a.action]()
