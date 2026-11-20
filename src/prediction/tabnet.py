"""
TabNet (Arik & Pfister, 2019) wrapped as an sklearn estimator for this pipeline.

    from src.prediction.tabnet import TabNetSkRegressor

TabNet is the one deep-tabular architecture that drops into prediction/models.py
with almost no adaptation, because dreamquark-ai's `pytorch-tabnet` already
subclasses sklearn.base.BaseEstimator. That buys the two properties the LSTM in
prediction/nn/ never had and which are the whole contract of this module:

  * get_params()/clone() work, so hyper_params_search() can hand the estimator to
    sklearn.base.clone and RandomizedSearchCV with no special case, and
  * a fitted estimator survives joblib.dump -> load with identical predictions,
    which is exactly what src/live/refit.py does with search.best_estimator_.

So a `tabnet` arm is deployable to a live slot for free, unlike the sequence
model, which needs a bespoke live fit/predict path and `seq_length` dates of
history it does not yet have.

------------------------------------------------------------------------------
Why a thin subclass rather than the raw TabNetRegressor
------------------------------------------------------------------------------

The raw class is sklearn-shaped but not sklearn-complete for THIS pipeline. Four
gaps, each closed here and nowhere else so there is one implementation:

  1. It rejects a DataFrame outright ("apply X.values"), and walk_forward hands
     it a DataFrame all the way from src/prediction/functions.py. -> np.asarray.
  2. It demands a 2-D target of shape (n, 1); the pipeline's y is a 1-D Series.
     -> reshape(-1, 1) on the way in.
  3. predict() returns (n, 1); predict_signal() in functions.py takes the return
     value verbatim, so an un-raveled column would propagate a 2-D prediction
     into the forecast frame. -> ravel() on the way out.
  4. max_epochs / patience / batch_size / virtual_batch_size are arguments to
     fit(), not to __init__, so RandomizedSearchCV cannot see them and
     --fixed-params cannot pin them. -> they are constructor arguments here,
     stored verbatim, and passed through in fit().

Point 4 is the subtle one. sklearn introspects the __init__ SIGNATURE to build
get_params(), so every argument must appear there explicitly -- a **kwargs
forwarding shim would hide the parent's parameters and set_params('n_steps=...')
would then die inside the search with "Invalid parameter". Hence the long
explicit signature below, the same "stored verbatim, nothing derived" rule
torch_mlp.py follows.

------------------------------------------------------------------------------
No eval_set, and why that is the leak-safe choice
------------------------------------------------------------------------------

pytorch-tabnet's early stopping (`patience`) watches an `eval_set`. Passing one
from inside an inner CV fold would draw that holdout from the same periods the
fold trains on, which is precisely the leak purged_period_splits() exists to
prevent -- and it would not announce itself, it would just make the CV score
optimistic. So fit() passes NO eval_set. With none, `patience` is inert and the
network trains for the full `max_epochs`; that is the same posture MLPWrapper
takes when it holds early_stopping=False, and for the same reason. The honest
cost is that a run can stop short of or past its best epoch; the dishonest
alternative leaks.

------------------------------------------------------------------------------
Two training-loop settings that are deliberate, not defaults-by-accident
------------------------------------------------------------------------------

  * drop_last=True (the library default) is KEPT. TabNet uses BatchNorm, which
    errors on a batch of one row, and a walk-forward's inner folds are of
    arbitrary size, so a size-1 final batch is a live possibility. Dropping the
    short remainder each epoch -- ~one batch in thirty on a 39,000-row block --
    is cheaper than the crash, and it is what the library does.
  * device defaults to "cpu". The walk-forward runs ~24 small fits per arm and
    host-to-device transfer can cost more than the kernels save at this size;
    "cuda" is accepted and untested on this panel. See setup.sh -- torch is
    OPTIONAL, so pytorch_tabnet is imported inside fit(), never at module scope,
    exactly as LGBWrapper imports lightgbm.

------------------------------------------------------------------------------
The panel is small and noisy for a model this expressive
------------------------------------------------------------------------------

Each refit sees ~39,000 rows over 22-to-812 columns, and the effective sample is
closer to the ~130 periods than to the row count, because a period's ~300 rows
are one cross-section. TabNet's paper datasets were 10^5-10^7 rows with clean
signal. So on THIS data the library-default configuration is a viability probe,
not a tuned model, and `lambda_sparse` (the mask sparsity penalty, the closest
analogue to MLPWrapper's `alpha`) is the one knob that would matter most if a
search were justified. See TabNetWrapper for the reasoned grid.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class TabNetSkRegressor(RegressorMixin, BaseEstimator):
    """
    TabNetRegressor with the four sklearn-contract gaps closed. See module head.

    Every constructor argument is a real pytorch-tabnet knob at its library
    default, so an unset value reproduces the paper's configuration and pinning a
    subset via --fixed-params leaves the rest at those defaults.

    :param n_d: width of the decision output per step (main capacity dial)
    :type n_d: int
    :param n_a: width of the attention embedding per step; usually tied to n_d
    :type n_a: int
    :param n_steps: number of sequential decision steps
    :type n_steps: int
    :param gamma: feature-reuse relaxation across steps (>1 permits re-selection)
    :type gamma: float
    :param n_independent: independent GLU layers per step
    :type n_independent: int
    :param n_shared: shared GLU layers per step
    :type n_shared: int
    :param lambda_sparse: penalty on the attention masks; the regulariser, the
                          analogue of MLPWrapper's alpha
    :type lambda_sparse: float
    :param momentum: BatchNorm momentum
    :type momentum: float
    :param mask_type: "sparsemax" or "entmax"; how hard the feature selection is
    :type mask_type: str
    :param learning_rate: Adam step size (goes to optimizer_params["lr"])
    :type learning_rate: float
    :param max_epochs: training epochs; with no eval_set this is exact, not a cap
    :type max_epochs: int
    :param patience: early-stop patience; INERT here because no eval_set is used,
                     kept only so the parameter is pinnable and get_params-visible
    :type patience: int
    :param batch_size: rows per optimiser step
    :type batch_size: int
    :param virtual_batch_size: ghost-batch-norm sub-batch size
    :type virtual_batch_size: int
    :param seed: seed passed to TabNetRegressor
    :type seed: int or None
    :param device: "cpu", "cuda" or "auto"
    :type device: str
    :param verbose: TabNet's own training verbosity (0 = silent)
    :type verbose: int
    """

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2,
                 n_shared=2, lambda_sparse=1e-3, momentum=0.02,
                 mask_type="sparsemax", learning_rate=0.02, max_epochs=100,
                 patience=10, batch_size=1024, virtual_batch_size=128,
                 seed=None, device="cpu", verbose=0):
        # Stored verbatim, nothing derived: clone() rebuilds from get_params()
        # and sklearn forbids mutating arguments in __init__.
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
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the network on one training block.

        :param X: (n_samples, n_features); a DataFrame is accepted and converted
        :param y: (n_samples,) or (n_samples, 1)
        :return: self
        """
        # torch/pytorch_tabnet are OPTIONAL (setup.sh); import here so this
        # module -- and therefore run_forecast and src/live/refit.py -- import
        # with no torch present. Same lazy pattern as LGBWrapper/lightgbm.
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2-D, got shape {}".format(X.shape))
        y = y.reshape(-1, 1)                             # TabNet wants (n, 1)
        if len(X) != len(y):
            raise ValueError("X has {} rows, y has {}".format(len(X), len(y)))

        self.n_features_in_ = X.shape[1]

        model = TabNetRegressor(
            n_d=int(self.n_d), n_a=int(self.n_a), n_steps=int(self.n_steps),
            gamma=float(self.gamma), n_independent=int(self.n_independent),
            n_shared=int(self.n_shared), lambda_sparse=float(self.lambda_sparse),
            momentum=float(self.momentum), mask_type=str(self.mask_type),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=float(self.learning_rate)),
            seed=(0 if self.seed is None else int(self.seed)),
            verbose=int(self.verbose),
            device_name=str(self.device),
        )
        if len(X) < int(self.batch_size):
            # drop_last below would drop the only (partial) batch: zero optimiser steps, and
            # the "fitted" model is its random initialisation (E198's per-coin arms, 2026-11-20).
            raise ValueError("TabNet: %d training rows < batch_size %d; nothing would be trained"
                             % (len(X), int(self.batch_size)))
        model.fit(
            X, y,
            max_epochs=int(self.max_epochs), patience=int(self.patience),
            batch_size=int(self.batch_size),
            virtual_batch_size=int(self.virtual_batch_size),
            # No eval_set: an inner-fold holdout would leak (see module head).
            eval_set=None,
            # BatchNorm errors on a size-1 batch; the short remainder is dropped.
            drop_last=True,
            num_workers=0,
            compute_importance=False,
        )
        self._model = model
        return self

    def predict(self, X):
        """
        :param X: (n_samples, n_features)
        :return: (n_samples,) predictions
        :rtype: np.ndarray
        """
        check_is_fitted(self, "_model")
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X).ravel()           # (n, 1) -> (n,)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
