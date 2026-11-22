"""Meaningful scalar references and cutoff/leakage checks for E206."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.analysis import tsfm_momentum_study as N
from src.data_mani.e206_tsfm_embeddings import inputs


def test_resolved_label_and_covariate_cutoffs():
    grid=pd.date_range('2020-01-01',periods=400).to_numpy().astype('datetime64[D]')
    y=np.arange(400,dtype=np.float32);x=np.tile(y[:,None],(1,33))
    a=inputs(y,x,grid,350)
    assert a[0][0,-1]==315 and a[2][0,0,-1]==350 and a[3]==35
    y[316:]=9999;x[351:]=-9999
    b=inputs(y,x,grid,350)
    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
    c=inputs(y[:351],x[:351],grid[:351],350)
    for aa,bb in zip(b,c):np.testing.assert_array_equal(aa,bb)


def test_temporal_splits_respect_calendar_label_availability():
    d=pd.date_range('2020-01-01','2026-02-19').to_numpy()
    tr,va,ft,te=N.split_masks(d)
    assert d[tr].max()+np.timedelta64(35,'D')<=d[va].min()
    assert d[va].max()+np.timedelta64(35,'D')<=d[te].min()
    assert not (ft&te).any() and te.sum()>300


def test_scaler_never_reads_validation_or_test():
    rng=np.random.default_rng(206);h=rng.normal(size=(90,4));a=np.repeat([1,2,3],30)
    d=np.tile(np.arange(30),3);fit=d<15;valid=np.ones(90,bool)
    for norm in ['pooled','asset','robust']:
        z,refs=N.normalize(h,a,d,valid,fit,norm)
        changed=h.copy();changed[~fit]+=1000
        other,again=N.normalize(changed,a,d,valid,fit,norm)
        np.testing.assert_array_equal(z[fit],other[fit])
        for key in refs:np.testing.assert_array_equal(refs[key],again[key])


def test_rolling_is_prior_only_and_short_history_masked():
    h=np.arange(140,dtype=float)[:,None];a=np.ones(140);d=np.arange(140);valid=np.ones(140,bool)
    z,_=N.normalize(h,a,d,valid,np.zeros(140,bool),'rolling',63)
    assert np.isnan(z[:32]).all() and np.isfinite(z[32:]).all()
    changed=h.copy();changed[100:]+=10000
    other,_=N.normalize(changed,a,d,valid,np.zeros(140,bool),'rolling',63)
    np.testing.assert_array_equal(z[:100],other[:100])
    expected=(h[70,0]-h[7:70,0].mean())/h[7:70,0].std()
    np.testing.assert_allclose(z[70,0],expected,rtol=1e-6)


def test_leave_one_out_aggregates_exact_against_direct_removal():
    h=np.random.default_rng(3).normal(size=(7,5));dates=np.ones(7);valid=np.ones(7,bool)
    z,_=N.distribution_design(h,dates,valid,'pooled')
    for i in range(7):
        others=np.delete(h,i,axis=0)
        expected=np.r_[h[i],others.mean(0),others.std(0),*np.quantile(others,[.25,.5,.75],axis=0)]
        np.testing.assert_allclose(z[i],expected,rtol=1e-6,atol=1e-7)
    small=np.ones((2,3));z,_=N.distribution_design(small,np.ones(2),np.ones(2,bool),'pooled')
    assert np.isfinite(z).all()
    cs,info=N.distribution_design(h,dates,valid,'cross_section')
    assert cs.shape[1]==4*h.shape[1] and len(info['removed'])==2


def test_dual_solver_matches_sklearn_even_p_greater_than_n():
    rng=np.random.default_rng(2);x=rng.normal(size=(30,90));q=rng.normal(size=(9,90));y=rng.normal(size=30)
    x[:,3]=1;q[:,3]=2
    pred,_=N.dual_ridge(x@x.T,q@x.T,y,[.001,1,100])
    for i,alpha in enumerate([.001,1,100]):
        est=Ridge(alpha=alpha).fit(x,y)
        np.testing.assert_allclose(pred[:,i],est.predict(q),rtol=1e-7,atol=1e-7)


def test_concat_kernel_equals_explicit_train_vocabulary_slots():
    rng=np.random.default_rng(3);h=rng.normal(size=(8,4));assets=np.repeat([1,2],4);steps=np.tile(np.arange(4),2)
    valid=np.ones(8,bool);valid[2]=False
    k=N.concat_kernel(h,assets,steps,valid,[1,2],4)
    explicit=np.zeros((4,8))
    for i in range(8):
        if valid[i]:explicit[steps[i],(assets[i]-1)*4:assets[i]*4]=h[i]
    np.testing.assert_allclose(k,explicit@explicit.T,atol=1e-12)
    # Unknown entrant never adds an unregistered concatenation slot.
    again=N.concat_kernel(h,assets,steps,valid,[1],4)
    np.testing.assert_allclose(again,explicit[:,:4]@explicit[:,:4].T,atol=1e-12)


def test_decile_portfolio_cost_drift_and_missing_valuations():
    for side in ['long','neutral']:
        w=N.decile_weights(np.arange(20),side)
        assert np.isclose(np.abs(w).sum(),1)
        assert np.isclose(w.sum(),1 if side=='long' else 0)
    weights=np.array([[1.,0.],[1.,0.],[0.,1.]])
    returns=np.array([[.1,0.],[.1,0.],[0.,np.nan]])
    out=N.account_portfolio(weights,returns)
    assert np.isclose(out.gross.iloc[0],.1) and np.isclose(out.net.iloc[0],.099)
    assert out.turnover.iloc[1]<.001
    assert not out.accountable.iloc[2] and np.isnan(out.net.iloc[2])


def test_per_asset_selection_and_final_predictions_ignore_test_labels(tmp_path,monkeypatch):
    from experiments import e206_momentum3_tsfm as E
    monkeypatch.setattr(E,'SCR',tmp_path)
    rng=np.random.default_rng(1)
    dates=np.r_[pd.date_range('2022-01-01',periods=80).to_numpy(),pd.date_range('2024-01-01',periods=30).to_numpy(),pd.date_range('2025-01-01',periods=20).to_numpy()]
    p=pd.DataFrame([(a,d) for a in [1,2] for d in dates],columns=['ucid','date'])
    p['target_binned_return_20']=rng.normal(size=len(p))
    for name,mask in zip(['train','validation','final_fit','test'],N.split_masks(p.date)):p[name]=mask
    p['step']=np.tile(np.arange(len(dates)),2)
    x=rng.normal(size=(len(p),5));np.save(tmp_path/'common_rows.npy',np.ones(len(p),bool))
    monkeypatch.setattr(E,'build_design',lambda *args:(x,np.ones(len(p),bool),{}))
    pred,records,selected,_=E.solve_assets('fake','pooled',126,'A','selection',p)
    final,*_=E.solve_assets('fake','pooled',126,'A','final',p,selected)
    p.loc[p.test,'target_binned_return_20']+=1000
    other,_,again,_=E.solve_assets('fake','pooled',126,'A','selection',p)
    last,*_=E.solve_assets('fake','pooled',126,'A','final',p,again)
    np.testing.assert_array_equal(pred,other);np.testing.assert_array_equal(final,last)
    assert selected==again


def test_portfolio_scoring_report_and_configuration_accounting(tmp_path,monkeypatch):
    from experiments import e206_momentum3_tsfm as E
    from src.analysis import html_pdf
    monkeypatch.setattr(E,'ROOT',tmp_path/'report');monkeypatch.setattr(E,'SCR',tmp_path/'scratch')
    monkeypatch.setattr(html_pdf,'to_pdf',lambda *args:None)
    E.ROOT.mkdir();(E.SCR/'holdings').mkdir(parents=True)
    rng=np.random.default_rng(71)
    dates=pd.bdate_range('2025-01-01',periods=5)
    p=pd.DataFrame([(d,a) for d in dates for a in range(12)],columns=['date','ucid'])
    p['symbol']=p.ucid.astype(str);p['row']=np.arange(len(p))
    p[E.TARGET]=rng.uniform(0,1,len(p)).astype(np.float32);p['meta_model']=rng.uniform(0,1,len(p))
    p['prediction']=rng.uniform(0,1,len(p));p['baseline']=p.prediction
    p['baseline_rank']=p.groupby('date').prediction.transform(lambda s:E.rank_normalise(s.to_numpy()))
    E.parquet(E.ROOT/'momentum_baseline.parquet',p)
    configs=[dict(id=E.config_id('fake','pooled',d),model='fake',standardization='pooled',design=d) for d in N.DESIGNS]
    E.save(E.ROOT/'grid.json',configs);E.save(E.ROOT/'PREPARED.json',{})
    for c in configs:
        frame=p[['row','symbol','ucid','date',E.TARGET,'meta_model','prediction']].copy()
        frame['prediction']+=rng.normal(0,.01,len(frame))
        E.parquet(E.ROOT/'fits'/c['id']/'predictions.parquet',frame)
        E.save(E.ROOT/'fits'/c['id']/'COMPLETE.json',dict(status='success'))
        E.save(E.ROOT/'fits'/c['id']/'validation_selection.json',dict(mean_asset_mse=.1,window=126))
    E.parquet(E.ROOT/'execution_returns.parquet',pd.DataFrame(dict(row=p.row,origin=p.date,
          execution=p.date+pd.Timedelta(days=1),exit=p.date+pd.Timedelta(days=2),return_value=rng.normal(0,.02,len(p)))))
    E.score()
    table=pd.read_csv(E.ROOT/'ranked_results.csv')
    assert len(table)==16 and set(table.status)=={'success'}
    assert (E.ROOT/'paired_portfolio_returns.csv').exists()
    assert (E.ROOT/'validation_selected_configuration.json').exists()
    assert len(list((E.SCR/'holdings').glob('*.npz')))==16


def test_interrupted_claims_recovered_only_after_owner_verification(tmp_path,monkeypatch):
    from experiments import e206_momentum3_tsfm as E
    from types import SimpleNamespace
    monkeypatch.setattr(E,'SCR',tmp_path)
    d=tmp_path/'embeddings'/'fake';d.mkdir(parents=True)
    E.save(d/'dead.lock',dict(host='known-host',pid=11))
    E.save(d/'alive.lock',dict(host='known-host',pid=12))
    monkeypatch.setattr(E.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,
                stdout='12 python -m src.data_mani.e206_tsfm_embeddings --model fake\n'))
    E.recover_locks()
    assert not (d/'dead.lock').exists() and (d/'alive.lock').exists()
    monkeypatch.setattr(E.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=255,stdout=''))
    E.recover_locks()
    assert (d/'alive.lock').exists()
