import joblib
import numpy as np
import torch
from src.prediction.tabnet import TabNetSkRegressor
from src.analysis.wide_tabnet import WideTabNetSkRegressor,group_identity,explain_identity


def test_sparse_identity_training_matches_dense_and_saves(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(1);x=rng.normal(size=(40,9)).astype('float32');y=rng.normal(size=40).astype('float32')
    kw=dict(n_d=4,n_a=4,n_steps=2,max_epochs=2,batch_size=16,virtual_batch_size=8,seed=1)
    a=TabNetSkRegressor(**kw).fit(x,y);b=WideTabNetSkRegressor(**kw).fit(x,y)
    np.testing.assert_allclose(a.predict(x),b.predict(x),rtol=1e-4,atol=1e-5)
    c=WideTabNetSkRegressor(**dict(kw,batch_size=1024)).fit(x,y)
    assert c.effective_batch_size_==32
    assert len(c._model.history['loss'])==2 and np.isfinite(c.predict(x)).all()
    joblib.dump(b,tmp_path/'m');np.testing.assert_array_equal(b.predict(x),joblib.load(tmp_path/'m').predict(x))
    assert group_identity([],300000)._nnz()==300000
    assert explain_identity(300000,1,[],300000).nnz==300000


def test_sparse_identity_forward_and_gradient():
    torch.set_num_threads(1)
    a=torch.randn(7,23,requires_grad=True);b=a.detach().clone().requires_grad_()
    x=a@torch.eye(23);y=b@group_identity([],23)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    (x*x).sum().backward();(y*y).sum().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
