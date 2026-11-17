"""
TabNet trained for a Timmermann-Vulicevic (TV) epoch count derived from its own size.

TabNetSkRegressor (tabnet.py) takes `max_epochs` as a free hyperparameter, so a
random search can pair a large network with a handful of epochs. This module
derives the epoch count INSIDE fit(). The count comes from the trainable
parameter count P of the network the sampled hyperparameters actually build,
on the actual number of input columns:

    E = max(floor(0.2 * P**0.8), 1) + 10          (TV model-size schedule)
    E = max(1, round(E * epoch_mult))             (optional multiplier, after the +10)
    E = min(E, epoch_cap) if epoch_cap is set     (recorded as epoch_cap_bound_)

So every candidate a search draws trains for the TV count of ITS architecture
on ITS dataset (22 raw vs 44 fenccausal columns). No grid entry can break the
coupling, because max_epochs is no longer a parameter. epochs_ and n_params_
record what each fit used. See .claude/skills/timmermann-mlp-compute.

ISOLATION: this is a separate module and a separate class on purpose. Running
tabnet_ktz studies (E153/E158) unpickle TabNetSkRegressor inside loky workers
that import src/prediction/tabnet.py fresh. Adding constructor parameters to
that class would make get_params() on an object pickled by an older parent
raise AttributeError, which the search would silently score as UNSCORABLE.
Nothing in tabnet.py is modified.
"""

import math

import numpy as np

from src.prediction.tabnet import TabNetSkRegressor


def tv_epochs(n_params, mult=1.0, cap=None):
    """TV model-size epoch schedule. Multiplier applied after rounding and +10."""
    base = max(int(math.floor(0.2 * float(n_params) ** 0.8)), 1) + 10
    epochs = max(1, int(round(base * float(mult))))
    if cap is not None:
        epochs = min(epochs, int(cap))
    return epochs


def count_tabnet_params(n_features, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                        n_independent=2, n_shared=2, momentum=0.02,
                        mask_type="sparsemax", virtual_batch_size=128):
    """
    Trainable scalars of the pytorch-tabnet network these settings build on
    `n_features` inputs with one output, including biases and BatchNorm affine
    parameters. Built exactly as TabNetRegressor builds it (identity group
    matrix, no categorical embeddings), so the count matches the fitted model's
    network, which fit() asserts.
    """
    import torch
    from pytorch_tabnet.tab_network import TabNet

    net = TabNet(int(n_features), 1, n_d=int(n_d), n_a=int(n_a),
                 n_steps=int(n_steps), gamma=float(gamma),
                 n_independent=int(n_independent), n_shared=int(n_shared),
                 virtual_batch_size=int(virtual_batch_size),
                 momentum=float(momentum), mask_type=str(mask_type),
                 group_attention_matrix=torch.eye(int(n_features)))
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


class TabNetTVRegressor(TabNetSkRegressor):
    """
    TabNetSkRegressor whose epoch count is the TV schedule of its own size.

    Same constructor as the parent minus `max_epochs`, plus:
      epoch_rule  "tv" (the only rule; kept as a parameter so the hyperparams
                  table records it)
      epoch_mult  multiplier on the TV count, applied after the +10 (default 1)
      epoch_cap   optional hard ceiling; None = the guarantee holds exactly
    """

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2,
                 n_shared=2, lambda_sparse=1e-3, momentum=0.02,
                 mask_type="sparsemax", learning_rate=0.02, patience=10,
                 batch_size=1024, virtual_batch_size=128, seed=None,
                 device="cpu", verbose=0, epoch_rule="tv", epoch_mult=1.0,
                 epoch_cap=None):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.mask_type = mask_type
        self.learning_rate = learning_rate
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.epoch_rule = epoch_rule
        self.epoch_mult = epoch_mult
        self.epoch_cap = epoch_cap

    def planned_epochs(self, n_features):
        """(epochs, n_params) this estimator would train for on n_features columns."""
        if self.epoch_rule != "tv":
            raise ValueError("epoch_rule must be 'tv', got %r" % (self.epoch_rule,))
        n_params = count_tabnet_params(
            n_features, n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, n_independent=self.n_independent,
            n_shared=self.n_shared, momentum=self.momentum,
            mask_type=self.mask_type, virtual_batch_size=self.virtual_batch_size)
        return tv_epochs(n_params, self.epoch_mult, self.epoch_cap), n_params

    def fit(self, X, y):
        n_features = np.asarray(X).shape[1]
        epochs, n_params = self.planned_epochs(n_features)
        uncapped = tv_epochs(n_params, self.epoch_mult, None)
        # The parent's fit reads self.max_epochs. It is NOT a constructor
        # parameter here (absent from get_params, so clone() never carries it):
        # it is derived per fit, like a fitted attribute.
        self.max_epochs = epochs
        super().fit(X, y)
        fitted = int(sum(p.numel() for p in self._model.network.parameters()
                         if p.requires_grad))
        if fitted != n_params:
            raise RuntimeError("TV parameter count %d != fitted network's %d"
                               % (n_params, fitted))
        self.n_params_ = n_params
        self.epochs_ = epochs
        self.epoch_cap_bound_ = epochs < uncapped
        return self
