"""E231 TabNet with complete per-epoch coverage and safe small-sample batches."""
import math
import numpy as np
from src.prediction.tabnet import TabNetSkRegressor
from pytorch_tabnet.callbacks import Callback

class SampleCounter(Callback):
    def on_train_begin(self,logs=None):
        self.epoch_examples=[];self.epoch_batches=[];self.minimum_batch=None
    def on_epoch_begin(self,epoch,logs=None):self.seen=0;self.batches=0
    def on_batch_end(self,batch,logs=None):
        n=int(logs['batch_size']);self.seen+=n;self.batches+=1
        self.minimum_batch=n if self.minimum_batch is None else min(n,self.minimum_batch)
    def on_epoch_end(self,epoch,logs=None):
        self.epoch_examples.append(self.seen);self.epoch_batches.append(self.batches)

class CompleteTabNet(TabNetSkRegressor):
    def fit(self,X,y):
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
        torch.set_num_threads(max(1,int(__import__('os').getenv('NMR_SLOT_CORES','1'))))
        X=np.asarray(X,np.float32);y=np.asarray(y,np.float32).reshape(-1,1)
        if len(X)<2:raise ValueError('At least two training rows required')
        batch=min(int(self.batch_size),len(X))
        while len(X)%batch==1:batch-=1
        virtual=min(int(self.virtual_batch_size),batch)
        assert batch>=2 and virtual>=2 and len(X)%batch!=1
        counter=SampleCounter()
        model=TabNetRegressor(n_d=int(self.n_d),n_a=int(self.n_a),n_steps=int(self.n_steps),gamma=float(self.gamma),n_independent=int(self.n_independent),n_shared=int(self.n_shared),lambda_sparse=float(self.lambda_sparse),momentum=float(self.momentum),mask_type=self.mask_type,optimizer_fn=torch.optim.Adam,optimizer_params={'lr':float(self.learning_rate)},seed=int(self.seed),verbose=0,device_name='cpu')
        model.fit(X,y,max_epochs=int(self.max_epochs),patience=0,batch_size=batch,virtual_batch_size=virtual,eval_set=None,drop_last=False,num_workers=0,compute_importance=False,callbacks=[counter])
        assert counter.epoch_examples==[len(X)]*int(self.max_epochs)
        assert counter.epoch_batches==[math.ceil(len(X)/batch)]*int(self.max_epochs)
        assert counter.minimum_batch>=2 and len(model.history['loss'])==self.max_epochs
        self._model=model;self.n_features_in_=X.shape[1]
        self.coverage_=dict(effective_batch=batch,effective_virtual_batch=virtual,epoch_examples=counter.epoch_examples,epoch_batches=counter.epoch_batches,minimum_batch=counter.minimum_batch,total_examples=sum(counter.epoch_examples),parameter_count=sum(p.numel() for p in model.network.parameters() if p.requires_grad),drop_last=False,epochs=int(self.max_epochs))
        return self
