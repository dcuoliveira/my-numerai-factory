"""
Exact Gaussian Process estimators that subsample the training rows.

An exact GP is O(n^3) in the number of training rows, and a pooled Crypto
training block is 20k-500k rows (a 130-date rolling window is ~40k; the default
expanding scheme reaches ~500k). An exact GP over that is infeasible, and a
per-date cross-sectional GP (~300 rows) cannot forecast a FUTURE date, so it does
not fit the walk-forward contract. The tractable option that keeps the whole
framework -- walk_forward_fit_and_predict, hyper_params_search, --fixed-params,
the bayes path and live/refit.py -- unchanged is to wrap sklearn's exact GP and
cap the TRAINING set to `subsample` rows with a seeded draw taken INSIDE fit.
Prediction is not subsampled: each ~300-row daily cross-section is scored against
the subsampled support, O(n_test * subsample), which is cheap.

Two deliberate choices, both so the outer search stays in charge of the kernel:

* The kernel is chosen by a STRING (`kernel`), never a kernel OBJECT. A kernel
  object in a param_grid is a silent --search-type bayes trap: search_spaces
  .sequence_valued_params only flags list/tuple-of-sequences, so it would not
  catch a kernel object -- the bayes guard would stay green and BayesSearchCV
  would then blow up inside .fit(). A `kernel` string is a plain Categorical, so
  every grid built on these estimators searches under both random and bayes.

* The kernel is FROZEN by default (`length_scale_bounds="fixed"` -> the sklearn
  GP's `optimizer=None`). The outer RandomizedSearchCV (or a --fixed-params pin)
  sets the scalar kernel params, so each fit is a single Cholesky (deterministic,
  cheap) and the winning scale lands in best_params_ -> the hyperparams CSV and
  is pinnable for a live slot, exactly like ridge's alpha. Passing a numeric
  `length_scale_bounds` tuple instead re-enables marginal-likelihood tuning of
  the length-scale within those bounds (the gp_ktz / ARD self-tuning fork).

Features reach the model already rank-binned to [0, 1] (the pipeline is
scaler-free by design) with missing values filled to 0.5, so a length-scale near
1 spans the feature range and the GP's no-NaN requirement already holds.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_random_state


# Matern smoothness nu per kernel name; RBF is the nu -> infinity limit.
_MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


def _build_kernel(kernel, ard, length_scale, length_scale_bounds, n_features):
    """
    Assemble an sklearn kernel from flat, JSON-pinnable parameters.

    Built INSIDE fit so no kernel OBJECT ever enters a param_grid (which would
    silently break --search-type bayes; see the module docstring).

    :param kernel: one of "rbf", "matern12", "matern32", "matern52"
    :param ard: one length-scale per feature (True) or a single shared one
    :param length_scale: the (initial) length-scale
    :param length_scale_bounds: "fixed" (frozen) or a (low, high) pair
    :param n_features: feature count, for the ARD length-scale vector
    :return: ConstantKernel(1.0) * (RBF|Matern)
    """
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern

    scale = [length_scale] * n_features if ard else length_scale
    # Amplitude is carried by normalize_y (regression) or left at unit scale
    # (classification); it is fixed here so the tuned axis is the length-scale.
    const = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")
    if kernel == "rbf":
        base = RBF(length_scale=scale, length_scale_bounds=length_scale_bounds)
    elif kernel in _MATERN_NU:
        base = Matern(length_scale=scale, length_scale_bounds=length_scale_bounds,
                      nu=_MATERN_NU[kernel])
    else:
        raise ValueError(
            "unknown kernel %r, expected one of %s"
            % (kernel, ["rbf"] + sorted(_MATERN_NU)))
    return const * base


def _subsample_index(n, subsample, random_state):
    """
    Seeded uniform-without-replacement draw of row positions; identity when the
    block already fits under the cap.

    The panel is ~300 rows/date, so a uniform draw over a 20k-500k-row block is
    automatically spread across dates -- no per-date machinery needed. The draw
    is sorted so the fitted support keeps chronological row order.
    """
    if subsample is None or n <= int(subsample):
        return np.arange(n)
    rng = check_random_state(random_state)
    return np.sort(rng.choice(n, size=int(subsample), replace=False))


class SubsampledGPRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn GaussianProcessRegressor over a seeded subsample of the training rows.

    The posterior mean is the signal. `alpha` is the observation-noise variance
    added to the kernel diagonal -- the GP's regularisation knob, and what keeps
    K + alpha*I positive definite; it leans HIGH on this noisy target. On a
    singular kernel (small alpha + short length-scale) the Cholesky can still
    fail; the fit then falls back to a constant (the training mean) rather than
    crashing a sweep, so a degenerate cell scores an honest flat prediction.
    """

    def __init__(self, kernel="matern32", ard=False, length_scale=1.0,
                 length_scale_bounds="fixed", alpha=1e-2, normalize_y=True,
                 subsample=6000, random_state=None):
        # BaseEstimator rule: store every arg verbatim under the same name, no
        # transformation, so get_params/set_params/clone round-trip. All work
        # happens in fit().
        self.kernel = kernel
        self.ard = ard
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor

        values = np.asarray(X, dtype="float64")       # DataFrame -> ndarray; the
        target = np.asarray(y, dtype="float64")       # index is only needed at
        self.n_features_in_ = values.shape[1]         # predict, by position
        idx = _subsample_index(len(values), self.subsample, self.random_state)
        train_x, train_y = values[idx], target[idx]
        self.n_train_ = len(idx)

        # normalize_y divides by std(y); a degenerate subsample would divide by 0.
        norm = bool(self.normalize_y) and np.std(train_y) > 0
        optimizer = None if self.length_scale_bounds == "fixed" else "fmin_l_bfgs_b"
        kernel = _build_kernel(self.kernel, self.ard, self.length_scale,
                               self.length_scale_bounds, self.n_features_in_)
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel, alpha=float(self.alpha), optimizer=optimizer,
            normalize_y=norm, copy_X_train=True, random_state=self.random_state)
        try:
            self.gp_.fit(train_x, train_y)
            self.constant_ = None
        except np.linalg.LinAlgError:
            # Near-singular kernel: fall back to the training mean so a sweep
            # records a flat prediction for this cell rather than dying.
            self.gp_ = None
            self.constant_ = float(np.mean(train_y))
        return self

    def predict(self, X):
        values = np.asarray(X, dtype="float64")
        if self.gp_ is None:
            return np.full(len(values), self.constant_, dtype="float64")
        return self.gp_.predict(values)               # O(n_test * subsample)


class SubsampledGPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn GaussianProcessClassifier over a seeded subsample of the training
    rows. The positive-class probability is the signal (predict_signal takes
    predict_proba[:, 1]).

    GPC has no noise diagonal: it uses the Laplace approximation, whose Newton
    mode-finding is O(n^3) PER step, so the subsample cap is lower than the
    regressor's. A degenerate subsample (a single class after binarisation) or a
    failed fit falls back to a constant 0.5 probability.
    """

    def __init__(self, kernel="matern32", ard=False, length_scale=1.0,
                 length_scale_bounds="fixed", subsample=1500, random_state=None,
                 max_iter_predict=100):
        self.kernel = kernel
        self.ard = ard
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.subsample = subsample
        self.random_state = random_state
        self.max_iter_predict = max_iter_predict

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier

        values = np.asarray(X, dtype="float64")
        target = np.asarray(y)
        self.n_features_in_ = values.shape[1]
        idx = _subsample_index(len(values), self.subsample, self.random_state)
        train_x, train_y = values[idx], target[idx]
        self.n_train_ = len(idx)

        optimizer = None if self.length_scale_bounds == "fixed" else "fmin_l_bfgs_b"
        classes = np.unique(target)
        if len(np.unique(train_y)) < 2:
            # Only one class in the subsample (or the whole block): GPC cannot
            # fit. Predict the constant 0.5 the neutral middle bin implies.
            self.gp_ = None
            self.classes_ = classes if len(classes) == 2 else np.array([0, 1])
            return self
        try:
            self.gp_ = GaussianProcessClassifier(
                kernel=_build_kernel(self.kernel, self.ard, self.length_scale,
                                     self.length_scale_bounds, self.n_features_in_),
                optimizer=optimizer, max_iter_predict=int(self.max_iter_predict),
                random_state=self.random_state)
            self.gp_.fit(train_x, train_y)
            self.classes_ = self.gp_.classes_
        except (np.linalg.LinAlgError, ValueError):
            self.gp_ = None
            self.classes_ = np.unique(train_y)
        return self

    def predict(self, X):
        values = np.asarray(X, dtype="float64")
        if self.gp_ is None:
            return np.full(len(values), self.classes_[-1])
        return self.gp_.predict(values)

    def predict_proba(self, X):
        values = np.asarray(X, dtype="float64")
        if self.gp_ is None:
            return np.full((len(values), len(self.classes_)),
                           1.0 / len(self.classes_), dtype="float64")
        return self.gp_.predict_proba(values)
