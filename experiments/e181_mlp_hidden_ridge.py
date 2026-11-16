"""E181: nested causal sklearn-initialised Torch KTZ representations + FE glmnet ridge."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys, json, time, traceback, argparse, hashlib, base64, html
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, QuantileTransformer
REPO=Path(__file__).resolve().parents[1]
os.chdir(REPO);sys.path[:0]=[str(REPO),str(REPO/'experiments')]
from src.prediction.torch_mlp import TorchMLPRegressor
from src.prediction.models import MLPKtzWrapper, glmnet_alpha_path
from e150_finish_scoring import matrix_scores
from src.analysis.cross_target import newey_west
from src.analysis.html_pdf import to_pdf
import e170_union_ktz as E
ROOT=REPO/'results/reports/crypto/e181_mlp_hidden_ridge'
T='target_binned_return_20'
CUTOFF=pd.Timestamp('2026-02-20')
ARCHS=[(8,),(16,),(32,),(64,),(128,),(32,16),(32,16,8)]
METHODS=['raw','standard','cs_bins','quantile_bins','squared','hidden_only','correction']
SEEDS=[42,43,44]
save=E.save

def mail(event,body,attachments=()):
    old=E.ROOT;E.ROOT=ROOT
    # Existing project notification transport, with E181-specific subject.
    try:
        from src.live.env import load_env
        from src.live.notify import _config
        from email.message import EmailMessage
        import smtplib
        path=ROOT/'notifications'/f'{event}.json'
        if path.exists() and json.loads(path.read_text()).get('sent'):return True
        load_env();config=_config()
        if not config:raise RuntimeError('Project notification configuration missing')
        msg=EmailMessage();msg['Subject']='[numerai-crypto] E181 '+event.upper()+': MLP hidden states + FE ridge'
        msg['From']=config['user'];msg['To']=config['to'];msg.set_content(body)
        for p in attachments:msg.add_attachment(Path(p).read_bytes(),maintype='application',subtype='pdf',filename=Path(p).name)
        with smtplib.SMTP(config['host'],config['port'],timeout=60) as s:
            s.starttls();s.login(config['user'],config['password']);s.send_message(msg)
        save(path,dict(sent=True,time=time.time(),attachments=list(map(str,attachments))))
        return True
    except Exception as exc:
        save(ROOT/'notifications'/f'{event}.json',dict(sent=False,error=str(exc),time=time.time()));return False
    finally:E.ROOT=old

class SklearnInitTorchMLP(TorchMLPRegressor):
    def _build(self,n_features,torch,nn,generator):
        widths=[n_features]+list(self.hidden_layer_sizes)+[1]
        assert self.activation=='identity' and not self._n_embed
        net=nn.Sequential(*[nn.Linear(a,b) for a,b in zip(widths[:-1],widths[1:])]).to(dtype=torch.float64,device=self.device)
        rng=np.random.RandomState(self.random_state)
        with torch.no_grad():
            for layer in net:
                if isinstance(layer,nn.Linear):
                    bound=np.sqrt(6/(layer.in_features+layer.out_features))
                    layer.weight.copy_(torch.as_tensor(rng.uniform(-bound,bound,(layer.in_features,layer.out_features)).T,device=self.device))
                    layer.bias.copy_(torch.as_tensor(rng.uniform(-bound,bound,layer.out_features),device=self.device))
        return net

def hidden(est,x):
    torch=est._torch;est._net.eval()
    with torch.no_grad():
        return np.concatenate([est._net[:-1](torch.as_tensor(z,dtype=torch.float64,device=est.device)).cpu().numpy() for z in np.array_split(x,max(1,len(x)//8192+1))])

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'registration.json').exists():return
    source=REPO/'data/crypto/v2.0-tenc-onehot-rollsel-spread-interact-fmx'
    # Predicate pushdown: held-out target rows never enter this experiment.
    b=pd.read_parquet(source/'train.parquet',filters=[('date','<',CUTOFF)]).reset_index()
    encpath=REPO/'data/crypto/v2.0-fenccausal/train.parquet'
    enc=pd.read_parquet(encpath,filters=[('date','<',CUTOFF)]).reset_index()
    bc=[c for c in b if c.startswith('feature_') and c!='feature_sym_tenc' and (not c.startswith('feature_x') or c.startswith('feature_x29_'))]
    dc=[c for c in bc if c.startswith('feature_sym_')]
    cont=[c for c in bc if c not in dc]
    assert len(bc)==813 and len(dc)>700
    D=b[dc].to_numpy();assert np.isin(D,[0,1]).all() and (D.sum(axis=1)<=1).all()
    b['group']=np.where(D.sum(axis=1)==1,D.argmax(axis=1)+1,0)
    ec=[c for c in enc if c.startswith('feature_')];assert len(ec)==44
    enc=enc[['symbol','date']+ec].rename(columns={c:'enc_'+c for c in ec})
    frame=b[['symbol','ucid','date',T,'group']+cont].merge(enc,on=['symbol','date'],validate='1:1')
    meta=pd.read_parquet(source/'historical_meta_models.parquet').reset_index()
    frame=frame.merge(meta[['symbol','date','prediction']].rename(columns={'prediction':'meta_model'}),on=['symbol','date'],how='left',validate='1:1')
    frame=frame.dropna(subset=[T]).sort_values(['date','symbol']).reset_index(drop=True)
    # Fixed neutral imputation, never fitted on evaluation rows.
    frame[cont+['enc_'+c for c in ec]]=frame[cont+['enc_'+c for c in ec]].replace([np.inf,-np.inf],np.nan).fillna(.5)
    frame.to_parquet(ROOT/'panel.parquet',index=False)
    dates=np.sort(frame.date.unique());folds=[]
    for i,start in enumerate(range(172,len(dates)-129,130)):
        tr=dates[start-152:start-22];te=dates[start:start+130]
        assert len(tr)==130 and tr[-1]<te[0] and (te[-1]<np.datetime64(CUTOFF))
        folds.append(dict(block=i,train=list(map(str,tr)),test=list(map(str,te))))
    assert folds
    save(ROOT/'registration.json',dict(experiment='E181',architectures=ARCHS,methods=METHODS,seeds=SEEDS,folds=folds,continuous=cont,encoder=['enc_'+c for c in ec],
        ridge='glmnet 100-point path recomputed on inner training only; tune path index then rebuild on outer training; exact unpenalised asset one-hot via within-group demeaning',
        encoder_search='40 lr/alpha cells per architecture; chronological inner68/purge22/validate40; select on per-date CORR; outer130/purge22/predict130; all encoders and ridge refit from scratch',
        processing='raw; training standardisation; per-date tie-preserving CS five bins; training-only marginal quantile five bins; standard states plus their squares; hidden-only + asset FE; MLP forecast plus ridge residual correction',
        selection='development exploratory dimension/processing sweep, no holdout opened; identity encoder affine, transformed variants nonlinear; three seeds scores averaged per date',
        causal='Existing causal x29 and fenccausal source datasets reused as frozen upstream features, not fitted estimators. No claim to revalidate historical data vintages. Every learned scaler, encoder and readout is fitted inside each split.',
        unpenalised_columns=dc,source_hashes={str(p):E.sha(p) for p in [source/'train.parquet',encpath,source/'historical_meta_models.parquet']},runner_sha256=E.sha(__file__),panel_sha256=E.sha(ROOT/'panel.parquet')))

def score(pred,frame):
    return np.nanmean([matrix_scores(pred[idx,None],frame[T].to_numpy()[idx],np.full(len(idx),np.nan))[0][0] for idx in frame.groupby('date').indices.values() if len(idx)>=10])

def train_encoder(train,query,cols,arch,params,seed,device):
    scaler=StandardScaler().fit(train[cols]);xt=scaler.transform(train[cols]);xq=scaler.transform(query[cols])
    est=SklearnInitTorchMLP(hidden_layer_sizes=arch,activation='identity',alpha=params['alpha'],learning_rate_init=params['lr'],batch_size=1024,max_iter=200,random_state=seed,device=device)
    est.fit(xt,train[T].to_numpy())
    ht,hq=hidden(est,xt),hidden(est,xq)
    return ht,hq,est.predict(xt),est.predict(xq),est

def process(ht,hq,train,query,method):
    if method=='cs_bins':
        def bins(h,f):
            out=np.zeros_like(h)
            for ids in f.groupby('date').indices.values():out[ids]=np.minimum(4,np.floor((rankdata(h[ids],axis=0,method='average')-.5)/len(ids)*5))/4
            return out
        return bins(ht,train),bins(hq,query)
    if method=='quantile_bins':
        qt=QuantileTransformer(n_quantiles=min(1000,len(ht)),output_distribution='uniform',random_state=42).fit(ht)
        return np.minimum(4,np.floor(qt.transform(ht)*5))/4,np.minimum(4,np.floor(qt.transform(hq)*5))/4
    if method in ['standard','squared','hidden_only','correction']:
        sc=StandardScaler().fit(ht);a,b=sc.transform(ht),sc.transform(hq)
        if method=='squared':return np.c_[a,a*a],np.c_[b,b*b]
        return a,b
    return ht,hq

def demean(x,y,g):
    ids,inv=np.unique(g,return_inverse=True);counts=np.bincount(inv)
    mx=np.zeros((len(ids),x.shape[1]));np.add.at(mx,inv,x);mx/=counts[:,None]
    my=np.bincount(inv,weights=y)/counts
    return x-mx[inv],y-my[inv],ids,mx,my

def ridge_path(xt,xq,y,gt,gq,index=None):
    # Continuous columns only. Fixed effects absorb the original whole dummy block exactly.
    xc,yc,ids,mx,my=demean(xt,y,gt)
    alphas=glmnet_alpha_path(xc,yc)
    if index is not None:alphas=alphas[[index]]
    eig,vec=np.linalg.eigh(xc.T@xc);eig=np.maximum(eig,0)
    coef=vec@((vec.T@(xc.T@yc))[:,None]/(eig[:,None]+alphas[None,:]))
    pos=np.searchsorted(ids,gq);known=(pos<len(ids))&(ids[np.minimum(pos,len(ids)-1)]==gq)
    means=np.tile(xt.mean(axis=0),(len(xq),1));ym=np.full(len(xq),y.mean())
    means[known]=mx[pos[known]];ym[known]=my[pos[known]]
    return (xq-means)@coef+ym[:,None],alphas,int((~known).sum())

def fit_readout(train,query,cont,ht,hq,pt,pq,method,index=None):
    base=StandardScaler().fit(train[cont]);a,b=base.transform(train[cont]),base.transform(query[cont])
    if ht is not None:
        ht,hq=process(ht,hq,train,query,method)
        a,b=(ht,hq) if method=='hidden_only' else (np.c_[a,ht],np.c_[b,hq])
    y=train[T].to_numpy().astype(float)
    if method=='correction':y=y-pt
    pred,alphas,unseen=ridge_path(a,b,y,train.group.to_numpy(),query.group.to_numpy(),index)
    if method=='correction':pred+=pq[:,None]
    return pred,alphas,unseen

def run_arch(arch,device):
    plan=json.loads((ROOT/'registration.json').read_text());assert plan['runner_sha256']==E.sha(__file__)
    assert plan['panel_sha256']==E.sha(ROOT/'panel.parquet')
    frame=pd.read_parquet(ROOT/'panel.parquet');name='x'.join(map(str,arch));out=ROOT/'fits'/name;out.mkdir(parents=True,exist_ok=True)
    for fold in plan['folds']:
        dest=out/f'block{fold["block"]}.parquet'
        if dest.exists():continue
        dates=pd.to_datetime(fold['train']);train=frame[frame.date.isin(dates)].reset_index(drop=True);query=frame[frame.date.isin(pd.to_datetime(fold['test']))].reset_index(drop=True)
        inner=train[train.date.isin(dates[:68])].reset_index(drop=True);val=train[train.date.isin(dates[-40:])].reset_index(drop=True)
        assert inner.date.max()<val.date.min() and train.date.max()<query.date.min()
        candidates=[dict(lr=float(lr),alpha=float(a)) for lr in MLPKtzWrapper.LEARNING_RATES for a in MLPKtzWrapper.ALPHAS]
        best=None;bestscore=-np.inf
        for candidate_index,params in enumerate(candidates):
            ht,hv,pt,pv,est=train_encoder(inner,val,plan['encoder'],arch,params,42,device)
            s=score(pv,val)
            if np.isfinite(s) and s>bestscore:bestscore=s;best=(params,ht,hv,pt,pv)
            del est
            print(name,'block',fold['block'],'encoder candidate',candidate_index+1,'/40',flush=True)
        assert best is not None
        params,ht,hv,pt,pv=best;indices={};picks=[]
        for method in ['baseline']+METHODS:
            pred,alphas,_=fit_readout(inner,val,plan['continuous'],None if method=='baseline' else ht,hv,pt,pv,method)
            scores=np.nanmean([matrix_scores(pred[ids],val[T].to_numpy()[ids],np.full(len(ids),np.nan))[0] for ids in val.groupby('date').indices.values() if len(ids)>=10],axis=0);idx=int(np.nanargmax(scores));indices[method]=idx
            picks.append(dict(method=method,path_index=idx,inner_corr=float(scores[idx]),inner_alpha=float(alphas[idx])))
        rows=[]
        for seed in SEEDS:
            ht,hq,pt,pq,est=train_encoder(train,query,plan['encoder'],arch,params,seed,device)
            weights=est._net[-1].weight.detach().cpu().numpy().ravel();bias=est._net[-1].bias.detach().cpu().numpy()[0]
            assert np.allclose(hq@weights+bias,pq,rtol=1e-9,atol=1e-9)
            # Keep fitted weights and training metadata for representation provenance.
            est._torch.save(est._net.state_dict(),out/f'block{fold["block"]}_s{seed}.pt')
            for method in ['baseline','mlp']+METHODS:
                if method=='baseline' and seed!=42:continue
                if method=='mlp':pred=pq;alpha=None;unseen=0
                else:
                    pp,aa,unseen=fit_readout(train,query,plan['continuous'],None if method=='baseline' else ht,hq,pt,pq,method,indices[method]);pred=pp[:,0];alpha=float(aa[0])
                part=query[['symbol','date',T,'meta_model']].copy();part['prediction']=pred;part['method']=method;part['seed']=seed;part['block']=fold['block'];rows.append(part)
                picks.append(dict(method=method,seed=seed,outer_alpha=alpha,unseen_rows=unseen))
            save(out/f'block{fold["block"]}_s{seed}_audit.json',dict(hidden_dim=ht.shape[1],hidden_rank=int(np.linalg.matrix_rank(ht-ht.mean(axis=0))),head_reconstruction=True,epochs=est.n_iter_))
        save(out/f'block{fold["block"]}_picks.json',dict(encoder_params=params,inner_encoder_corr=bestscore,readout=picks))
        tmp=dest.with_suffix('.partial');pd.concat(rows).to_parquet(tmp,index=False);tmp.replace(dest)
        print(name,'block',fold['block'],'complete',flush=True)


def finish():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((ROOT/'registration.json').read_text());scores=[]
    for arch in ARCHS:
        name='x'.join(map(str,arch))
        for fold in plan['folds']:
            p=ROOT/'fits'/name/f'block{fold["block"]}.parquet';assert p.exists(),str(p)
            f=pd.read_parquet(p)
            for (method,seed,date),g in f.groupby(['method','seed','date']):
                valid=np.isfinite(g.meta_model)&np.isfinite(g[T]);g=g[valid]
                if len(g)<10:continue
                c,m=matrix_scores(g.prediction.to_numpy()[:,None],g[T].to_numpy(),g.meta_model.to_numpy())
                scores.append(dict(architecture=name,dim=arch[-1],method=method,seed=seed,date=date,block=fold['block'],corr=c[0],mmc=m[0],payout=.1*c[0]+m[0]))
    S=pd.DataFrame(scores);S.to_parquet(ROOT/'scores_by_date_seed.parquet',index=False)
    A=S.groupby(['architecture','dim','method','date','block'],as_index=False)[['corr','mmc','payout']].mean()
    # Require all arms on the same dates.
    sets=[set(g.date) for _,g in A.groupby(['architecture','method'])];common=set.intersection(*sets);A=A[A.date.isin(common)]
    summary=[];pairs=[]
    for (arch,dim,method),g in A.groupby(['architecture','dim','method']):
        g=g.sort_values('date');v=g.payout.to_numpy();cum=np.cumsum(v);dd=np.max(np.maximum.accumulate(np.r_[0,cum])[1:]-cum)
        summary.append(dict(architecture=arch,dim=dim,method=method,corr=g['corr'].mean(),mmc=g.mmc.mean(),payout=v.mean(),maxdd=dd,calmar=v.mean()/dd if dd else np.nan,payout_sharpe=v.mean()/v.std(),burn_days=int((v<0).sum()),n=len(g)))
        if method!='baseline':
            base=A[(A.architecture==arch)&(A.method=='baseline')][['date','payout']]
            delta=g.merge(base,on='date',suffixes=('','_base'));nw=newey_west(delta.payout-delta.payout_base)
            pairs.append(dict(architecture=arch,method=method,delta=float(nw[0]),hac20_t=float(nw[1])))
    table=pd.DataFrame(summary).sort_values('mmc',ascending=False);table.to_csv(ROOT/'summary.csv',index=False);save(ROOT/'paired_deltas.json',pairs)
    S.groupby(['architecture','method','seed'],as_index=False)[['corr','mmc','payout']].mean().to_csv(ROOT/'by_seed.csv',index=False)
    A.groupby(['architecture','method','block'],as_index=False)[['corr','mmc','payout']].mean().to_csv(ROOT/'by_block.csv',index=False)
    text='<h1>E181 · MLP hidden states + fixed-effect glmnet ridge</h1><p>Development research. All headline scores use identical meta-covered dates and coins. Payout = 0.1 CORR + historical-meta proxy MMC. Drawdown is decline in cumulative score, not investment loss. Three seed scores averaged per date. Dimensions and processing ranked exploratorily; no holdout opened.</p>'
    text+='<h2>Descriptive best architecture per processing method</h2>'+table.drop_duplicates('method').to_html(index=False,float_format=lambda x:f'{x:.6f}')
    text+='<h2>Design and causal audit</h2><p>Source encoder: fenccausal44, identity KTZ MLP trained with actual NumPy sklearn initialisation draws in Torch. Readout: allfeat_pp minus tenc plus x29, exact unpenalised original asset dummies. The baseline is freshly trained with the same fixed-effect rule. Inner68 / purge22 / validate40 and outer130 / purge22 / predict130. Encoder selection and ridge path-index selection see inner validation only. Standardisation and quantile thresholds use training rows only. CS bins use contemporaneously available hidden states, no labels. Hidden-only retains asset effects. Correction adds ridge predictions of the MLP training residual to the MLP forecast. No previously fitted estimator is reused.</p><p>Identity states are affine projections, so raw hidden additions alter regularisation rather than add nonlinear capacity. Quantile/CS bins and squared states can add nonlinear capacity. Upstream causal x29 dataset is reused, with file hashes, and historical-vintage limits remain. This is not a new independently held-out confirmation of previously selected datasets.</p>'
    for method in METHODS:
        fig,ax=plt.subplots(figsize=(11,4))
        for arch in ['x'.join(map(str,a)) for a in ARCHS]:
            g=A[(A.architecture==arch)&(A.method==method)].sort_values('date');ax.plot(g.date,g.payout.cumsum(),label=arch)
        g=A[(A.architecture=='32')&(A.method=='baseline')].sort_values('date');ax.plot(g.date,g.payout.cumsum(),color='black',linestyle='--',label='fresh FE baseline')
        ax.set_title(method+' · cumulative payout score');ax.legend(ncol=4);fig.tight_layout();path=ROOT/f'cumulative_{method}.png';fig.savefig(path,dpi=140);plt.close(fig)
        text+='<h2>'+method+'</h2>'+table[table.method==method].to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<img style="width:100%" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
    text+='<h2>Seed diagnostics</h2>'+pd.read_csv(ROOT/'by_seed.csv').to_html(index=False)+'<h2>Block diagnostics</h2>'+pd.read_csv(ROOT/'by_block.csv').to_html(index=False)
    text+='<h2>Paired payout deltas against fresh baseline</h2>'+pd.DataFrame(pairs).to_html(index=False)+'<h2>Complete registration</h2><pre>'+html.escape(json.dumps({**{k:v for k,v in plan.items() if k not in ['folds','unpenalised_columns','continuous','encoder']},'folds':[{ 'block':f['block'],'train_start':f['train'][0],'train_end':f['train'][-1],'test_start':f['test'][0],'test_end':f['test'][-1]} for f in plan['folds']],'n_unpenalised_asset_columns':len(plan['unpenalised_columns']),'n_continuous_base_columns':len(plan['continuous']),'n_encoder_input_columns':len(plan['encoder'])},indent=2))+'</pre>'
    page=ROOT/'E181_hidden_states_DEVELOPMENT_RESEARCH.html';page.write_text('<html><head><meta charset="utf-8"><style>body{font:11px Arial}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:4px}h2{break-before:page}pre{white-space:pre-wrap}</style></head><body>'+text+'</body></html>')
    pdf=to_pdf(str(page),'lovelace');sent=mail('finished',table.head(12).to_string(index=False),[pdf]);save(ROOT/'COMPLETE.json',dict(time=time.time(),pdf=pdf,notification_sent=sent))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','finish','check']);p.add_argument('--device',default='cpu');p.add_argument('--arch');a=p.parse_args()
    if a.stage=='check':check();return
    try:
        prepare()
        if a.stage=='prepare':return
        if a.stage=='finish':finish();return
        mail('started','E181: 7 architectures, 7 hidden-state processing/readout variants, fresh fixed-effect glmnet ridge baseline, 3 seeds. Causal nested development evaluation. One consolidated PDF will be attached at completion. Output: '+str(ROOT))
        for arch in ARCHS:
            if a.arch and a.arch!='x'.join(map(str,arch)):continue
            run_arch(arch,a.device)
        if not a.arch:finish()
    except Exception:
        error=traceback.format_exc();save(ROOT/'FAILED.json',dict(time=time.time(),error=error));mail('failed',error);raise


def check():
    import torch
    from sklearn.neural_network import MLPRegressor
    from src.prediction.fe_ridge import FixedEffectRidge
    rng=np.random.RandomState(7);x=rng.normal(size=(80,4));y=rng.normal(size=80)
    est=SklearnInitTorchMLP(hidden_layer_sizes=(8,3),activation='identity',random_state=42)
    est._n_embed=0;net=est._build(4,torch,torch.nn,torch.Generator())
    sk=MLPRegressor(hidden_layer_sizes=(8,3),activation='identity',random_state=42);sk._random_state=np.random.RandomState(42);sk._initialize(y[:,None],[4,8,3,1],np.float64)
    for layer,w,b in zip(net,sk.coefs_,sk.intercepts_):assert np.array_equal(layer.weight.detach().numpy(),w.T) and np.array_equal(layer.bias.detach().numpy(),b)
    g=np.repeat(np.arange(4),20);q=rng.normal(size=(10,4));gq=np.array([0,1,2,3,8]*2)
    pred,alpha,_=ridge_path(x,q,y,g,gq,12)
    X=np.c_[x,np.eye(4)[g]];Q=np.c_[q,np.array([np.eye(4)[z] if z<4 else np.zeros(4) for z in gq])]
    ref=FixedEffectRidge(alpha=float(alpha[0]),unpenalised=tuple(range(4,8))).fit(X,y).predict(Q)
    assert np.allclose(pred[:,0],ref,atol=1e-10)
    f=pd.DataFrame(x,columns=list('abcd'));f['date']=np.repeat(pd.date_range('2020-01-01',periods=4),20);f[T]=y;f['group']=g
    query=f.iloc[:20].copy().reset_index(drop=True);query['date']=pd.Timestamp('2020-02-01')
    ht,hq,pt,pq,model=train_encoder(f,query,list('abcd'),(8,),dict(alpha=.1,lr=.001),42,'cpu')
    assert np.allclose(hq@model._net[-1].weight.detach().numpy().ravel()+model._net[-1].bias.detach().numpy()[0],pq)
    changed=query.copy();changed[T]=999
    for method in METHODS:
        pred,_,_=fit_readout(f,query,list('abcd'),ht,hq,pt,pq,method,12)
        other,_,_=fit_readout(f,changed,list('abcd'),ht,hq,pt,pq,method,12)
        assert np.isfinite(pred).all() and np.array_equal(pred,other),method
    print('PASS: exact sklearn initialisation; FE reference incl unseen assets; hidden-head reconstruction; all seven readouts ignore future labels')

if __name__=='__main__':main()
