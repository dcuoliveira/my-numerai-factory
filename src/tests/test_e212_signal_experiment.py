"""E212 trial accounting, causal contexts, head independence and scoring parity."""
import numpy as np
import pandas as pd
import pytest
from src.cluster import budget
from src.prediction import meta_resid as MR
from experiments import e212_volatility99_tsfm as S
from src.data_mani import e212_signal_embeddings as X


def test_actual_four_model_trial_count_and_both_ensemble_views(monkeypatch):
    assert S.counts(4)==dict(heads=10,standalone_streams=20,ensemble_subsets=21679,scored_rows_per_target=43378,total_scored_rows=86756)
    monkeypatch.setattr(S,'settings',lambda:dict(trial_counts=S.counts(4)))
    names=[f'{m}_t{h}' for h in [20,60] for m in S.MODELS+['feature_ridge']]
    names+= [n+'_residual' for n in names.copy()]
    configs=S.trials(names)
    assert len(configs)==43378 and len({c['id'] for c in configs})==43378
    assert sum(c['kind']=='standalone' for c in configs)==20
    assert sum(c['post_residualized'] for c in configs)==21679
    assert {s:sum(len(c)==s for c in S.definitions(20)) for s in range(2,6)}=={2:190,3:1140,4:4845,5:15504}


@pytest.mark.parametrize('wall',[35,91])
def test_99_channel_both_horizon_contexts_exclude_future_labels_and_covariates(monkeypatch,wall):
    monkeypatch.setattr(X,'WALL',wall)
    grid=pd.bdate_range('2020-01-01',periods=500).to_numpy().astype('datetime64[D]')
    y=np.arange(500,dtype=np.float32);x=np.tile(y[:,None],(1,99));origin=400
    first=X.inputs(y,x,grid,origin)
    boundary=np.searchsorted(grid,grid[origin]-np.timedelta64(wall,'D'),side='right')-1
    assert first[0][0,-1]==boundary and first[1].shape==(1,99,256)
    y[boundary+1:]=99999;x[origin+1:]=-999
    changed=X.inputs(y,x,grid,origin)
    prefix=X.inputs(y[:origin+1],x[:origin+1],grid[:origin+1],origin)
    for a,b,c in zip(first,changed,prefix):
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)
    x[origin,5]=np.nan
    assert X.inputs(y,x,grid,origin) is None


@pytest.mark.parametrize('missing',[False,True])
def test_batched_residualization_exactly_matches_shared_helper(missing):
    rng=np.random.default_rng(212);x=np.round(rng.normal(size=(31,17)),1);m=rng.normal(size=31)
    m[:6]=np.nan
    if missing:m[:]=np.nan
    batch,beta=S.residual_matrix(x,m)
    for j in range(x.shape[1]):
        expected,b=MR.residualise(x[:,j],m)
        np.testing.assert_allclose(batch[:,j],expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(beta[j],b,rtol=1e-12,atol=1e-12,equal_nan=True)


def test_rank_blend_then_post_residualization_freshly_scored():
    rng=np.random.default_rng(99);x=rng.normal(size=(40,20));m=rng.normal(size=40)
    ranks=(S.rankdata(x,axis=0,method='average')-.5)/40
    subset=[(0,2),(1,3,8),(0,2,3,4,5)]
    blend=np.column_stack([ranks[:,c].mean(axis=1) for c in subset])
    post,_=S.residual_matrix(blend,m)
    y=rng.choice([0,.25,.5,.75,1],40).astype(np.float32)
    both=np.column_stack([blend,post]);corr,mmc=S.matrix_scores(both,y,m)
    for j in range(both.shape[1]):
        c,v=S.matrix_scores(both[:,j,None],y,m)
        np.testing.assert_allclose([corr[j],mmc[j]],[c[0],v[0]],rtol=1e-12,atol=1e-12)


def test_payout_statistics_use_additive_drawdown_and_population_sharpe():
    corr=np.array([.2,-.5,.1]);mmc=np.array([.03,-.1,.07]);p=.1*corr+mmc
    out=S.payout_metrics(corr,mmc)
    assert out['payout']==pytest.approx(p.mean())
    assert out['sharpe.p']==pytest.approx(p.mean()/p.std(ddof=0))
    cum=np.r_[0,np.cumsum(p)]
    assert out['maxdd.p']==pytest.approx((np.maximum.accumulate(cum)-cum).max())


def test_gpu_queue_accounts_for_running_and_foreign_reservations():
    host=budget.Host(name='gpu',cores=32,ram_gb=251,gpus=2)
    snapshot={'gpu':({0},[],{0,1})}
    available=S.admission_hosts([host],{},set(),snapshot)[0]
    assert available.busy_gpus==(0,)
    owned={'nmr-e212_test':dict(host='gpu',gpus=[1])}
    blocked=S.admission_hosts([host],owned,set(owned),snapshot)[0]
    placed,waiting=budget.distribute([budget.Job('wait','run',2,16,1)],[blocked],gpu_slots=1)
    assert not placed and len(waiting)==1
    assert '--device cuda:0' in S.worker_command('chronos2',60,False)
    assert 'TSFM_EMBED_WALL=91' in S.worker_command('chronos2',60,False)


@pytest.mark.parametrize('horizon',[20,60])
def test_new_independent_heads_ignore_future_and_other_assets_labels(tmp_path,monkeypatch,horizon):
    monkeypatch.setattr(S,'ROOT',tmp_path/'root');monkeypatch.setattr(S,'SCR',tmp_path/'scratch')
    S.ROOT.mkdir();S.SCR.mkdir()
    rng=np.random.default_rng(54)
    dates=pd.bdate_range('2020-01-01',periods=600)
    p=pd.DataFrame([(d,a) for d in dates for a in [1,2]],columns=['date','ucid'])
    p['row']=np.arange(len(p));p['step']=np.repeat(np.arange(600),2);p['covered']=True
    p[f'target_binned_return_{horizon}']=rng.uniform(size=len(p))
    q,protocol=S.fold_panel(p,500,horizon)
    assert q.loc[q.final_fit,'date'].max()+pd.Timedelta(days=S.WALLS[horizon])<=q.loc[q.test,'date'].min()
    h=rng.normal(size=(len(p),4)).astype(np.float16)
    directory=S.extraction_dir(horizon,'chronos2');directory.mkdir(parents=True)
    np.save(directory/'H_rows.npy',h);np.save(directory/'valid_rows.npy',np.ones(len(p),bool))
    np.save(S.SCR/'common_rows.npy',np.ones(len(p),bool))
    first,records,params,_=S.solve_head(q,horizon,'chronos2','rolling',63,'selection',S.ROOT,S.SCR/'a')
    final,*_=S.solve_head(q,horizon,'chronos2','rolling',63,'final',S.ROOT,S.SCR/'a',params)
    q.loc[q.test,f'target_binned_return_{horizon}']+=1000
    second,_,params2,_=S.solve_head(q,horizon,'chronos2','rolling',63,'selection',S.ROOT,S.SCR/'b')
    last,*_=S.solve_head(q,horizon,'chronos2','rolling',63,'final',S.ROOT,S.SCR/'b',params2)
    np.testing.assert_array_equal(first,second);np.testing.assert_array_equal(final,last)
    q.loc[q.ucid==2,f'target_binned_return_{horizon}']+=100
    other,*_=S.solve_head(q,horizon,'chronos2','rolling',63,'selection',S.ROOT,S.SCR/'c')
    np.testing.assert_array_equal(first[q.ucid==1],other[q.ucid==1])


def test_prepare_preserves_panel_missing_target60_and_actual_features(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(S,'REPO',tmp_path);monkeypatch.setattr(S,'ROOT',tmp_path/'study')
    monkeypatch.setattr(S,'SCR',tmp_path/'scratch');monkeypatch.setattr(S,'CODE',[])
    dates=pd.bdate_range('2020-01-01',periods=100)
    frame=pd.DataFrame([(d,str(a)) for d in dates for a in [7,8]],columns=['date','ucid'])
    frame['symbol']=frame.ucid.map(lambda a:'A'+a);frame['target_binned_return_20']=.5
    features=[f'feature_x{i}' for i in range(99)]
    frame=pd.concat([frame,pd.DataFrame(np.zeros((len(frame),99),np.float32),columns=features)],axis=1)
    official=frame[['symbol','date','ucid','target_binned_return_20']].copy()
    official['target_binned_return_60']=.25;official.loc[official.date.isin(dates[-10:]),'target_binned_return_60']=np.nan
    labelroot=tmp_path/'data/crypto/v2.0-e156-frozen';labelroot.mkdir(parents=True)
    official.set_index('symbol').to_parquet(labelroot/'train.parquet')
    data=tmp_path/'data/crypto/test99';data.mkdir()
    frame.set_index(['symbol','date']).to_parquet(data/'train.parquet')
    frame[['symbol','date','ucid']].assign(covered=True).set_index(['symbol','date']).to_parquet(data/'coverage.parquet')
    frame[['symbol','date']].assign(prediction=.5).set_index(['symbol','date']).to_parquet(data/'historical_meta_models.parquet')
    manifest=dict(features=features,n_features=99,date_max=str(dates.max().date()),output_hashes={p.name:S.sha(p) for p in data.glob('*.parquet')})
    (data/'manifest.json').write_text(json.dumps(manifest))
    S.prepare('crypto/test99',['chronos2'])
    stored=pd.read_parquet(S.ROOT/'panel.parquet')
    assert len(stored)==len(frame) and stored.target_binned_return_60.isna().sum()==20
    assert [c for c in stored if c.startswith('feature_')]==features
    assert S.settings()['dataset']=='crypto/test99'


def test_complete_scoring_uses_causal_ranks_before_evaluation_mask(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(S,'REPO',tmp_path);monkeypatch.setattr(S,'ROOT',tmp_path/'root');monkeypatch.setattr(S,'SCR',tmp_path/'scratch')
    S.ROOT.mkdir();S.SCR.mkdir()
    config=dict(dataset='crypto/test',models=['chronos2'],trial_counts=S.counts(1),limitations=['synthetic'])
    monkeypatch.setattr(S,'settings',lambda:config);monkeypatch.setattr(S,'verify',lambda:config)
    rng=np.random.default_rng(123);dates=pd.bdate_range('2024-01-10',periods=3)
    p=pd.DataFrame([(d,str(a)) for d in dates for a in range(15)],columns=['date','ucid'])
    p['symbol']=p.ucid.map(lambda a:'A'+a);p['row']=np.arange(len(p))
    for h in [20,60]:p[f'target_binned_return_{h}']=rng.choice([0,.25,.5,.75,1],len(p)).astype(np.float32)
    p['meta_model']=rng.uniform(size=len(p));p.loc[p.ucid=='0','target_binned_return_60']=np.nan
    p.loc[p.ucid=='1','meta_model']=np.nan
    S.parquet(S.ROOT/'panel.parquet',p);np.save(S.SCR/'common_rows.npy',p.ucid.astype(int).lt(13).to_numpy())
    S.save(S.ROOT/'PREPARED.json',dict(date_cutoff='2024-02-01'))
    meta=pd.DataFrame(dict(date=[pd.Timestamp('2024-01-01')]*15,symbol=['A'+str(i) for i in range(15)],prediction=rng.uniform(size=15)))
    metadir=tmp_path/'data/crypto/test';metadir.mkdir(parents=True);meta.set_index(['symbol','date']).to_parquet(metadir/'historical_meta_models.parquet')
    for h in [20,60]:
        for model in ['chronos2','feature_ridge']:
            forecast=p[['row']].copy();forecast['prediction']=rng.normal(size=len(p));forecast.loc[p.ucid.astype(int)>=13,'prediction']=np.nan
            if model=='feature_ridge':forecast['baseline']=rng.uniform(size=len(p))
            S.parquet(S.ROOT/'fits'/f'{model}_t{h}'/'predictions.parquet',forecast)
    monkeypatch.setattr(S,'report',lambda *args:(S.ROOT/'E212_volatility99_tsfm.pdf').write_bytes(b'%PDF synthetic'))
    S.score()
    table=pd.read_csv(S.ROOT/'all_trials.csv')
    assert len(table)==S.counts(1)['total_scored_rows']
    assert table.assets_mean.eq(11).all() and table.dates.eq(3).all()
    configs=json.loads((S.ROOT/'trials.json').read_text());ensemble=next((i,c) for i,c in enumerate(configs) if c['kind']=='ensemble' and not c['post_residualized'])
    idx,c=ensemble
    forecast=pd.read_parquet(S.ROOT/'standalone_forecasts.parquet')
    day=p.date.eq(dates[0])&p.ucid.astype(int).lt(13)
    values=forecast.loc[day,c['members']].to_numpy()
    ranked=(S.rankdata(values,axis=0,method='average')-.5)/13
    blended=ranked.mean(axis=1);evalmask=p.loc[day,'target_binned_return_60'].notna().to_numpy()&p.loc[day,'meta_model'].notna().to_numpy()
    expected=S.matrix_scores(blended[evalmask,None],p.loc[day,'target_binned_return_20'].to_numpy()[evalmask],p.loc[day,'meta_model'].to_numpy()[evalmask])
    actual=np.load(S.SCR/'daily_scores/t20/corr_mmc.npy')[0,idx]
    np.testing.assert_allclose(actual,[expected[0][0],expected[1][0]],rtol=1e-12,atol=1e-12)


def test_extractor_manifest_has_dataset_feature_and_package_fingerprint(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(X,'ROOT',tmp_path)
    (tmp_path/'PREPARED.json').write_text(json.dumps(dict(features=[f'feature_{i}' for i in range(99)],dataset=S.DATASET,panel_hash='frozen')))
    ext=SimpleNamespace(dim=512,reg=SimpleNamespace(get_params=lambda:{}))
    module=SimpleNamespace(Extractor=lambda device:ext,CKPT_DEFAULT='pinned-checkpoint',CKPT_SHA256='verified-hash')
    monkeypatch.setattr(X.importlib,'import_module',lambda name:module)
    _,manifest=X.adapter('tabpfnts','cuda:0')
    assert manifest['dataset']==S.DATASET and len(manifest['features'])==99
    assert 'plus99' in manifest['channels'] and len(manifest['fingerprint'])==64
    assert 'extractor_sha256' in manifest and 'package_versions' in manifest
    assert all(isinstance(key,str) for key in manifest)
    json.dumps(manifest)
