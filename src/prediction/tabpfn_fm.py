"""
TabPFN (Hollmann et al.) wrapped as an sklearn estimator for this pipeline.

    from src.prediction.tabpfn_fm import TabPFNSkRegressor

TabPFN is a tabular FOUNDATION model: a transformer pre-trained once on millions
of synthetic tabular tasks, which then solves a new dataset by IN-CONTEXT
LEARNING -- "fit" loads the training block into the context and does no gradient
training; "predict" is a single forward pass of the frozen network with that
block as context. There is therefore essentially nothing to tune, which is why
this file exists at all: it is the cheapest possible non-linear, non-tree arm to
put on the board after TabNet, because it removes the search entirely.

It reuses the exact contract discipline of tabnet.py / torch_mlp.py: an explicit
constructor signature (so get_params/clone/--fixed-params see every knob), a
DataFrame-and-1-D-y fit, a 1-D predict, and a lazy torch/tabpfn import so
prediction/models.py stays importable with no torch (setup.sh: OPTIONAL).

------------------------------------------------------------------------------
Two hard constraints of the model, and how each is handled
------------------------------------------------------------------------------

  1. CONTEXT SIZE. TabPFN's cost is O(context x query) attention and its prior is
     trained for a few thousand in-context rows; a 39,000-row CURUPIRA training
     block is both far outside that prior and ruinously slow to attend to. So
     fit() SUBSAMPLES the block to `max_context` rows with a seeded RNG before
     handing it over. This is a real modelling choice, not a detail: the model
     sees a sample of each block, not all of it, and `max_context` trades
     fidelity against a cost that grows linearly in it. It is recorded in the
     run's config like any other parameter.

  2. NETWORK / WEIGHTS. The pip package downloads its checkpoint from a gated
     HuggingFace repo on first use, which a restricted-egress box cannot do. So
     `model_path` defaults to a checkpoint vendored into the repo
     (models/tabpfn/tabpfn-v2-regressor.ckpt, the ungated Prior-Labs/TabPFN-v2-reg
     regressor), and nothing here reaches the network. Point it elsewhere to use
     a different checkpoint.

------------------------------------------------------------------------------
It wants a GPU
------------------------------------------------------------------------------

Every other wrapper here is CPU-cheap. TabPFN is not: measured on this box (CPU),
a 1,000-context / 300-query predict at n_estimators=1 is ~11 s, and cost scales
with context x query x n_estimators. A full walk-forward predicts tens of
thousands of rows per refit, so on CPU this is an hours-to-days arm even at
degraded settings, and `device="cuda"` is the intended way to run it. `device`
is exposed for exactly that reason and defaults to "cpu" only so an import on a
GPU-less box does not fail.
"""

import os

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

#: Vendored ungated checkpoints, so no network / no gated-repo auth is needed.
DEFAULT_MODEL_PATH = os.path.join(
    "models", "tabpfn", "tabpfn-v2-regressor.ckpt")
DEFAULT_CLF_MODEL_PATH = os.path.join(
    "models", "tabpfn", "tabpfn-v2-classifier.ckpt")


class TabPFNSkRegressor(RegressorMixin, BaseEstimator):
    """
    TabPFN regressor with the pipeline's sklearn-contract gaps closed.

    :param model_path: path to a local TabPFN checkpoint (default: the vendored
                       ungated v2 regressor); avoids the gated-repo download
    :type model_path: str
    :param n_estimators: TabPFN inference-time ensemble members; cost is linear
                         in this. The library default is higher; 1 is fastest
                         and deterministic
    :type n_estimators: int
    :param max_context: subsample each training block to at most this many rows
                        before fitting (see module head). None disables it
    :type max_context: int or None
    :param softmax_temperature: TabPFN's output temperature
    :type softmax_temperature: float
    :param ignore_pretraining_limits: allow blocks/feature counts beyond the
                                      prior's trained range rather than erroring
    :type ignore_pretraining_limits: bool
    :param device: "cpu" or "cuda"; cuda is strongly preferred (see module head)
    :type device: str
    :param random_state: seeds the context subsample and TabPFN's own RNG
    :type random_state: int or None
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, n_estimators=4,
                 max_context=4096, softmax_temperature=0.9,
                 ignore_pretraining_limits=True, memory_saving_mode="auto",
                 device="cpu", random_state=None):
        # Stored verbatim, nothing derived: clone() rebuilds from get_params().
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_context = max_context
        self.softmax_temperature = softmax_temperature
        self.ignore_pretraining_limits = ignore_pretraining_limits
        # "auto" lets TabPFN chunk the forward pass to fit the card; it is what
        # makes a large context predictable on an 8 GB GPU where a 39k-row query
        # against a 4k context already fills 6.8 GB. See E19's cumlag3 OOM.
        self.memory_saving_mode = memory_saving_mode
        self.device = device
        self.random_state = random_state

    @staticmethod
    def _n_est(value):
        """n_estimators is either the string 'auto' or an int; never int('auto')."""
        return value if value == "auto" else int(value)

    def _subsample(self, X, y):
        """Seeded context subsample; a no-op when the block already fits."""
        if self.max_context is None or len(X) <= int(self.max_context):
            return X, y
        rng = np.random.default_rng(
            0 if self.random_state is None else int(self.random_state))
        keep = rng.choice(len(X), size=int(self.max_context), replace=False)
        keep.sort()                                      # keep chronological order
        return X[keep], y[keep]

    def fit(self, X, y):
        """
        Load the model and store the (subsampled) block as its context.

        :param X: (n_samples, n_features); a DataFrame is accepted and converted
        :param y: (n_samples,)
        :return: self
        """
        # torch/tabpfn are OPTIONAL (setup.sh): import here, never at module
        # scope, so models.py imports with no torch. Same as tabnet.py.
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D, got shape {}".format(X.shape))
        if len(X) != len(y):
            raise ValueError("X has {} rows, y has {}".format(len(X), len(y)))

        self.n_features_in_ = X.shape[1]
        Xc, yc = self._subsample(X, y)
        self.context_rows_ = len(Xc)

        model = TabPFNRegressor(
            model_path=str(self.model_path),
            n_estimators=self._n_est(self.n_estimators),
            softmax_temperature=float(self.softmax_temperature),
            ignore_pretraining_limits=bool(self.ignore_pretraining_limits),
            memory_saving_mode=self.memory_saving_mode,
            device=str(self.device),
            random_state=(None if self.random_state is None
                          else int(self.random_state)),
        )
        model.fit(Xc, yc)
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
        return np.asarray(self._model.predict(X)).ravel()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # n_estimators > 1 ensembles randomised preprocessings, so mark it
        # non-deterministic even though n_estimators=1 is reproducible.
        tags.non_deterministic = True
        return tags


class TabPFNSkClassifierRegressor(RegressorMixin, BaseEstimator):
    """
    TabPFN's CLASSIFIER head on the 5-binned target, exposed as a regressor.

    Numerai crypto's target is already five ordered bins {0, .25, .5, .75, 1},
    and TabPFN was born a classifier -- the classification head is the stronger,
    longer-trained one. So this fits TabPFNClassifier on those bins as classes
    and returns the EXPECTED bin value, E[y|x] = sum_c c * P(y=c|x), as a
    continuous signal. That is a proper regressor output for the pipeline
    (predict_signal calls .predict), and because Numerai scores by rank it is
    exactly the right summary of the class distribution: monotone in how much
    probability mass sits on the higher bins.

    The bin values are the class LABELS themselves (0.0 ... 1.0), so classes_
    already carries them and the expectation is a plain dot product -- no mapping
    table, no leakage. A block that happens to miss a bin simply has one fewer
    class; the dot product still holds because it sums over classes_.

    Everything else -- context subsampling, the vendored offline checkpoint (the
    classifier one), the memory-saving mode, the lazy import -- matches
    TabPFNSkRegressor. Only the head and the checkpoint differ, which is the one
    thing this arm is meant to vary.
    """

    def __init__(self, model_path=DEFAULT_CLF_MODEL_PATH, n_estimators=4,
                 max_context=4096, softmax_temperature=0.9,
                 ignore_pretraining_limits=True, memory_saving_mode="auto",
                 device="cpu", random_state=None):
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_context = max_context
        self.softmax_temperature = softmax_temperature
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.memory_saving_mode = memory_saving_mode
        self.device = device
        self.random_state = random_state

    def _subsample(self, X, y):
        if self.max_context is None or len(X) <= int(self.max_context):
            return X, y
        rng = np.random.default_rng(
            0 if self.random_state is None else int(self.random_state))
        keep = rng.choice(len(X), size=int(self.max_context), replace=False)
        keep.sort()
        return X[keep], y[keep]

    def fit(self, X, y):
        """
        Fit the classifier on the binned target as ordered classes.

        :param X: (n_samples, n_features); a DataFrame is accepted and converted
        :param y: (n_samples,) the binned target values
        :return: self
        """
        from tabpfn import TabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D, got shape {}".format(X.shape))
        if len(X) != len(y):
            raise ValueError("X has {} rows, y has {}".format(len(X), len(y)))

        self.n_features_in_ = X.shape[1]
        Xc, yc = self._subsample(X, y)
        self.context_rows_ = len(Xc)

        # TabPFNClassifier (like sklearn) rejects float labels as "continuous":
        # five distinct floats read as a regression target, not five classes. So
        # the bins are encoded to integer class indices for the fit, and their
        # float VALUES are kept for the expectation. self._levels[i] is the bin
        # value of class i; searchsorted maps each y to its bin index exactly
        # because the bins are the sorted unique values.
        self._levels = np.unique(yc).astype(np.float64)
        yc_idx = np.searchsorted(self._levels, yc).astype(np.int64)

        model = TabPFNClassifier(
            model_path=str(self.model_path),
            n_estimators=TabPFNSkRegressor._n_est(self.n_estimators),
            softmax_temperature=float(self.softmax_temperature),
            ignore_pretraining_limits=bool(self.ignore_pretraining_limits),
            memory_saving_mode=self.memory_saving_mode,
            device=str(self.device),
            random_state=(None if self.random_state is None
                          else int(self.random_state)),
        )
        model.fit(Xc, yc_idx)
        self._model = model
        return self

    def predict(self, X):
        """
        :param X: (n_samples, n_features)
        :return: (n_samples,) expected bin value E[y|x], a continuous signal
        :rtype: np.ndarray
        """
        check_is_fitted(self, "_model")
        X = np.asarray(X, dtype=np.float32)
        proba = np.asarray(self._model.predict_proba(X), dtype=np.float64)
        # Columns follow model.classes_ (the integer indices actually seen);
        # weight each by its bin value. A block missing a bin simply has one
        # fewer column, and the dot product still holds.
        level_of_col = self._levels[np.asarray(self._model.classes_, dtype=int)]
        return proba @ level_of_col                      # E[y|x], 1-D

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
