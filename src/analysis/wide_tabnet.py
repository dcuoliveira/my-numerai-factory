"""Exact ungrouped TabNet identities in sparse storage for wide H inputs."""
import math
from unittest.mock import patch
import numpy as np
from src.prediction.tabnet import TabNetSkRegressor


def group_identity(groups,n):
    import torch
    if groups:raise ValueError('Only ungrouped continuous H is supported')
    a=torch.arange(n)
    return torch.sparse_coo_tensor(torch.stack([a,a]),torch.ones(n),(n,n)).coalesce()


def explain_identity(input_dim,cat_emb_dim,cat_idxs,post_embed_dim):
    from scipy.sparse import eye
    if cat_idxs or input_dim!=post_embed_dim:raise ValueError('Only continuous H supported')
    return eye(input_dim,format='csc')


class WideTabNetSkRegressor(TabNetSkRegressor):
    def fit(self,X,y):
        import torch
        from pytorch_tabnet import abstract_model
        if len(X)<2:raise ValueError('At least two training rows required')
        original_batch,original_virtual=self.batch_size,self.virtual_batch_size
        self.effective_batch_size_=min(64,int(original_batch),2**int(math.log2(len(X))))
        self.effective_virtual_batch_size_=min(int(original_virtual),self.effective_batch_size_)
        self.batch_size=self.effective_batch_size_
        self.virtual_batch_size=self.effective_virtual_batch_size_
        torch.set_num_threads(max(1,int(__import__('os').getenv('NMR_SLOT_CORES','1'))))
        try:
            with patch.object(abstract_model,'create_group_matrix',group_identity),patch.object(abstract_model,'create_explain_matrix',explain_identity):
                return super().fit(X,y)
        finally:self.batch_size,self.virtual_batch_size=original_batch,original_virtual
