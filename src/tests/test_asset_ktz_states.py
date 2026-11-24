"""Representation, independent native fits, serialization and scalar score checks."""
import math

import joblib
import numpy as np
from sklearn.base import clone
from scipy.stats import rankdata

from src.analysis import asset_ktz_states as K
from src.analysis.e218_ktz_report import definitions,metrics,score_batch
from src.analysis.tsfm_momentum_study import concat_kernel


def test_explicit_c_preserves_fixed_slots_and_matches_source_kernel(tmp_path):
    h=np.arange(24,dtype=np.float32).reshape(8,3)
    assets=np.array([1,1,1,2,2,2,3,3]);steps=np.array([0,1,2,0,1,2,1,2])
    valid=np.array([True,True,False,True,False,True,True,True])
    result=K.materialize(h,assets,steps,valid,[2,1],3,tmp_path/'c.npy')
    expected=np.array([[9,10,11,0,1,2],[0,0,0,3,4,5],[15,16,17,0,0,0]],np.float32)
    np.testing.assert_array_equal(result,expected)
    np.testing.assert_allclose(result@result.T,concat_kernel(h,assets,steps,valid,[2,1],3))


def test_native_identity_mlp_collapse_and_saved_prediction_parity(tmp_path):
    rng=np.random.default_rng(218);x=rng.normal(size=(65,12));y=rng.normal(size=65)
    original=clone(K.wrapper('mlp_ktz').ModelClass).set_params(mlp__hidden_layer_sizes=(8,4),mlp__max_iter=8,mlp__activation='identity')
    original.fit(x[:50],y[:50]);query=x[50:]
    frozen=K.snapshot(original,'mlp_ktz',query)
    path=tmp_path/'mlp.joblib';joblib.dump(frozen,path)
    np.testing.assert_allclose(joblib.load(path).predict(query),original.predict(query),rtol=1e-7,atol=1e-8)
    assert not hasattr(frozen.steps[-1][1],'coefs_')
    # The native initialization remains reproducible independently from any
    # previously trained instance and from held-out labels.
    fresh=clone(K.wrapper('mlp_ktz').ModelClass).set_params(mlp__hidden_layer_sizes=(8,4),mlp__max_iter=8,mlp__activation='identity')
    fresh.fit(x[:50],y[:50]);np.testing.assert_array_equal(fresh.predict(query),original.predict(query))


def test_native_families_fresh_validation_selection_and_final_refit(tmp_path):
    rng=np.random.default_rng(31);x=rng.normal(size=(120,9));x[:,-1]=1
    y=.3*x[:,0]-.2*x[:,3]+rng.normal(scale=.05,size=120)
    for family in K.ALIASES:
        point=K.candidates(family)[0]
        if family=='mlp_ktz':point.update(mlp__max_iter=8,mlp__hidden_layer_sizes=(4,))
        elif family=='lgbm_ktz':point.update(n_estimators=5,min_child_samples=5)
        else:point.update(iterations=5,min_data_in_leaf=5)
        destination=tmp_path/(family+'.joblib')
        prediction,record=K.fit_asset(family,[point],x[:70],y[:70],x[70:90],y[70:90],x[:90],y[:90],x[90:],1,destination)
        state=joblib.load(destination)
        np.testing.assert_allclose(state['estimator'].predict(x[90:,state['columns']]),prediction,rtol=1e-7,atol=1e-8)
        assert record['n_train']==70 and record['n_validation']==20 and record['n_final']==90
        assert record['columns']==8 and record['sklearn_native_initialization']==(family=='mlp_ktz')
        assert len(record['choices'])==1 and np.isfinite(record['selected']['mse'])


def test_ensemble_count_and_batched_scores_equal_scalar_references():
    from experiments.e206_momentum3_tsfm import matrix_scores
    assert len(definitions(28))-28==sum(math.comb(28,k) for k in range(2,6))==122409
    rng=np.random.default_rng(99);raw=rng.normal(size=(31,5));raw[:8,0]=0
    y=rng.choice([0.,.25,.5,.75,1.],31).astype(np.float32);meta=rng.random(31)
    members=definitions(5);score=score_batch(raw,y,meta,members,matrix_scores)
    for j,ids in enumerate(members):
        x=raw[:,ids[0]] if len(ids)==1 else np.sum([rankdata(raw[:,i],method='ordinal') for i in ids],axis=0)
        expected=np.r_[matrix_scores(x[:,None],y,meta)]
        np.testing.assert_allclose(score[j],expected,rtol=1e-10,atol=1e-12)
    # Include an initial zero so a loss on the first date counts as drawdown.
    daily=np.zeros((3,1,2));daily[:,0,1]=[-1,.5,-.2]
    assert metrics(daily)['maxdd.p'][0]==1
