"""Parallelize E231's unchanged independent TabNet candidates, then refit once."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json
import time
import warnings
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments import e231_tabnet_ensembles as T
from experiments import e230_models as M
from experiments import e212_volatility99_tsfm as S

def estimator(params):
    from experiments.e231_tabnet_model import CompleteTabNet
    return Pipeline([('scale',StandardScaler()),('net',CompleteTabNet(**params,batch_size=1024,virtual_batch_size=128,patience=0,device='cpu'))])

def case(j):
    plan=T.prepare();meta,p,i=M.load();root=T.ROOT/'panel_cases'/str(j);root.mkdir(parents=True,exist_ok=True)
    if (root/'choice.json').exists():return
    x=p[meta['features']].to_numpy(float);y=p['target_binned_return_20'].to_numpy(float);tick=time.time();model=estimator(T.GRID[j])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='No early stopping will be performed');warnings.filterwarnings('ignore',message='Device used : cpu');model.fit(x[i['train']],y[i['train']])
    pred=model.predict(x[i['validation']]);assert np.isfinite(pred).all()
    choice=dict(parameters=T.GRID[j],mse=float(np.mean((pred-y[i['validation']])**2)),coverage=model.named_steps['net'].coverage_,seconds=time.time()-tick)
    np.save(root/'validation_predictions.npy',pred);S.save(root/'choice.json',choice);print('CASE COMPLETE',j,choice['seconds'],flush=True)

def final():
    plan=T.prepare();meta,p,i=M.load();root=T.ROOT/'models'/'panel';root.mkdir(parents=True,exist_ok=True);choices=[json.loads((T.ROOT/'panel_cases'/str(j)/'choice.json').read_text()) for j in range(4)];assert [v['parameters'] for v in choices]==T.GRID
    selected=min(choices,key=lambda v:v['mse']);x=p[meta['features']].to_numpy(float);y=p['target_binned_return_20'].to_numpy(float);tick=time.time();model=estimator(selected['parameters'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='No early stopping will be performed');warnings.filterwarnings('ignore',message='Device used : cpu');model.fit(x[i['final_fit']],y[i['final_fit']])
    pred=model.predict(x[i['query']]);assert np.isfinite(pred).all();path=root/'model.joblib';tmp=path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(path)
    np.testing.assert_allclose(joblib.load(path).predict(x[i['query']]),pred,atol=1e-7,rtol=1e-6)
    np.savez_compressed(root/'rows.npz',**{k:i[k] for k in ['train','validation','final_fit','query']});np.save(root/'predictions.npy',pred)
    record=dict(selected=selected,choices=choices,final_coverage=model.named_steps['net'].coverage_,final_seconds=time.time()-tick,seconds=sum(v['seconds'] for v in choices)+time.time()-tick,model_sha256=S.sha(path),n_train=len(i['train']),n_validation=len(i['validation']),n_final=len(i['final_fit']),n_query=len(i['query']),parallel_candidate_search=True)
    S.save(root/'record.json',record);S.save(T.ROOT/'workers/panel/COMPLETE.json',dict(worker='panel',assets=['panel'],identity=plan['identity'],time=time.time()));print('PANEL COMPLETE',record['final_seconds'],flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['case','final']);ap.add_argument('--candidate',type=int);a=ap.parse_args()
    if a.action=='case':case(a.candidate)
    else:final()
