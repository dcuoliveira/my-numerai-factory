"""E230 fresh chronological feature-only ridge and nonlinear MLP fits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from experiments import e230_dataset as D
from experiments.e227_v20_volatility_ridge import primal
from experiments import e212_volatility99_tsfm as S
from src.analysis import tsfm_momentum_study as N

GRID=[dict(hidden_layer_sizes=list(width),alpha=alpha,learning_rate_init=.001,epochs=100) for width in [(32,),(64,32)] for alpha in [.01,1.]]

def load():
    m=json.loads((D.ROOT/'PREPARED.json').read_text())
    assert m['builder_sha256']==S.sha(D.__file__)
    for path,h in m['output_hashes'].items():assert S.sha(path)==h,path
    return m,pd.read_parquet(D.ROOT/'panel.parquet'),dict(np.load(D.ROOT/'indices.npz'))

def make_mlp(parameters):
    return Pipeline([('scale',StandardScaler()),('net',MLPRegressor(hidden_layer_sizes=tuple(parameters['hidden_layer_sizes']),activation='relu',solver='adam',alpha=parameters['alpha'],batch_size=256,learning_rate_init=parameters['learning_rate_init'],max_iter=parameters['epochs'],early_stopping=False,tol=0,n_iter_no_change=parameters['epochs']+1,random_state=230,shuffle=True))])

def dense_parameters(width):
    layers=[134]+list(width)+[1]
    return int(sum((a+1)*b for a,b in zip(layers[:-1],layers[1:])))

def train_one(family,x,y,tr,va,ft,te,directory):
    started=time.time();directory.mkdir(parents=True,exist_ok=True)
    if family=='ridge':
        center=x[tr].mean(0);scale=x[tr].std(0);scale[scale<N.FLOOR]=1
        trial,_=primal((x[tr]-center)/scale,(x[va]-center)/scale,y[tr],N.ALPHAS)
        loss=np.mean((trial-y[va,None])**2,axis=0);best=int(np.argmin(loss));alpha=float(N.ALPHAS[best])
        center=x[ft].mean(0);scale=x[ft].std(0);scale[scale<N.FLOOR]=1
        pred,parameters=primal((x[ft]-center)/scale,(x[te]-center)/scale,y[ft],[alpha])
        np.savez_compressed(directory/'model.npz',center=center,scale=scale,alpha=alpha,**parameters)
        record=dict(selected=dict(alpha=alpha,mse=float(loss[best])),choices=[dict(alpha=float(a),mse=float(z)) for a,z in zip(N.ALPHAS,loss)])
        restored=np.load(directory/'model.npz');replay=((x[te]-restored['center'])/restored['scale']-restored['shift'])@restored['weights']+restored['intercept']
        np.testing.assert_array_equal(pred,replay);pred=pred[:,0]
    else:
        choices=[]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning);warnings.filterwarnings('ignore',message='Got `batch_size` less than 1 or larger than sample size')
            for params in GRID:
                tick=time.time();estimator=make_mlp(params);estimator.fit(x[tr],y[tr]);values=estimator.predict(x[va]);net=estimator.named_steps['net']
                assert net.n_iter_==params['epochs'] and np.isfinite(values).all()
                size=sum(w.size for w in net.coefs_)+sum(b.size for b in net.intercepts_);assert size==dense_parameters(params['hidden_layer_sizes'])
                examples=len(tr)*net.n_iter_
                choices.append(dict(parameters=params,mse=float(np.mean((values-y[va])**2)),epochs=int(net.n_iter_),parameters_count=size,examples_seen=examples,estimated_flops=6*size*examples,seconds=time.time()-tick))
            selected=min(choices,key=lambda v:v['mse']);estimator=make_mlp(selected['parameters']);tick=time.time();estimator.fit(x[ft],y[ft]);pred=estimator.predict(x[te]) if len(te) else np.empty(0);net=estimator.named_steps['net'];assert net.n_iter_==100
        joblib.dump(estimator,directory/'model.joblib',compress=3);restored=joblib.load(directory/'model.joblib');np.testing.assert_array_equal(restored.predict(x[te]) if len(te) else np.empty(0),pred)
        size=selected['parameters_count'];record=dict(selected=selected,choices=choices,final_epochs=int(net.n_iter_),final_parameters_count=size,final_examples_seen=len(ft)*int(net.n_iter_),final_estimated_flops=6*size*len(ft)*int(net.n_iter_),final_seconds=time.time()-tick)
    assert np.isfinite(pred).all()
    np.savez_compressed(directory/'rows.npz',train=tr,validation=va,final_fit=ft,query=te)
    np.save(directory/'predictions.npy',pred)
    record.update(n_train=len(tr),n_validation=len(va),n_final=len(ft),n_query=len(te),seconds=time.time()-started,model_sha256=S.sha(directory/('model.npz' if family=='ridge' else 'model.joblib')))
    S.save(directory/'record.json',record)
    return pred,record

def fit(arm):
    assert arm in D.ARMS
    meta,p,indices=load();root=D.ROOT/'fits'/arm;root.mkdir(parents=True,exist_ok=True)
    identity=dict(arm=arm,models_sha256=S.sha(__file__),panel_sha256=S.sha(D.ROOT/'panel.parquet'),indices_sha256=S.sha(D.ROOT/'indices.npz'),features=meta['features'],grid=GRID if arm.endswith('mlp') else N.ALPHAS.tolist())
    identityfile=root/'identity.json'
    if identityfile.exists():assert json.loads(identityfile.read_text())==identity
    else:S.save(identityfile,identity)
    if (root/'COMPLETE.json').exists():return
    x=p[meta['features']].to_numpy(float);y=p[D.TARGET].to_numpy(float);family=arm.split('_')[1]
    assert x.shape[1]==134
    pred=np.full(len(p),np.nan);records=[]
    if arm.startswith('panel'):
        pred[indices['query']],record=train_one(family,x,y,indices['train'],indices['validation'],indices['final_fit'],indices['query'],root/'models'/'panel');records.append(record)
    else:
        allassets=meta['assets'];assets=p.ucid.to_numpy()
        for n,asset in enumerate(allassets,1):
            rows={k:idx[assets[idx]==asset] for k,idx in indices.items() if k in ['train','validation','final_fit','query']}
            directory=root/'models'/('asset_'+asset)
            if (directory/'record.json').exists():
                record=json.loads((directory/'record.json').read_text());stored=np.load(directory/'rows.npz')
                for k in rows:np.testing.assert_array_equal(stored[k],rows[k])
                values=np.load(directory/'predictions.npy')
            else:values,record=train_one(family,x,y,rows['train'],rows['validation'],rows['final_fit'],rows['query'],directory)
            pred[rows['query']]=values;records.append(record)
            S.save(root/'progress.json',dict(completed_assets=len(records),total_eligible_assets=len(allassets),latest_asset=asset,latest_seconds=record['seconds'],time=time.time()))
            print('ASSET',arm,n,asset,record['seconds'],flush=True)
    assert np.isfinite(pred[indices['query']]).all()
    assert np.isnan(pred[np.setdiff1d(np.arange(len(p)),indices['query'])]).all()
    out=p[['row','ucid','symbol','date']].copy();out['prediction']=pred;S.parquet(root/'predictions.parquet',out)
    S.save(root/'records.json',records)
    S.save(root/'COMPLETE.json',dict(experiment='E230',task=dict(tag=arm,model=family),models=len(records),predictions=int(np.isfinite(pred).sum()),prediction_sha256=S.sha(root/'predictions.parquet'),identity=identity,time=time.time()))
    print('FIT COMPLETE',arm,len(records),flush=True)
