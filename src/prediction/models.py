"""
Model wrappers for the Numerai pipeline.

Every wrapper exposes the same four attributes, so hyper_params_search() never
needs to know which model it is tuning:

    .model_name   -- short name, used in output paths
    .search_type  -- "random" (only RandomizedSearchCV is supported)
    .param_grid   -- distributions for the random search
    .ModelClass   -- an *instance* of the estimator, ready to be cloned

One wrapper may also define an optional hook:

    .fit_grid(X, y) -> dict

which builds a param grid from the training data. Only RidgeGlmnet uses it; see
its docstring for why. hyper_params_search() calls it when present and falls
back to .param_grid otherwise.

------------------------------------------------------------------------------
Where the hyperparameter ranges come from
------------------------------------------------------------------------------

Two sources, both Numerai-specific rather than generic ML defaults.

1. Numerai's own shipped configs, which bracket the useful region:

        "small"  n_estimators 2,000   lr 0.01   max_depth 5   num_leaves 31
        "deep"   n_estimators 30,000  lr 0.001  max_depth 10  num_leaves 1024
        both:    colsample_bytree 0.1, min_data_in_leaf 10,000

2. The forum's Optuna study on "small" LGBM models
   (forum.numer.ai/t/hyperparameters-optimization-for-small-lgbm-models/6693),
   which swept n_estimators 1k-10k, lr 0.001-0.9, max_depth 2-10 (narrowed to
   4-7 once runtime mattered) and colsample_bytree 0.05-1.0. Its best model
   scored CORRV2 0.0247 at n_estimators=1885, lr=0.019, max_depth=5,
   colsample_bytree=0.13 -- essentially rediscovering the shipped "small" config.

Three idioms fall out of that, and every grid below encodes them:

  * Very low column sampling (~0.1, not the usual 0.6-1.0). The ~2,400 Classic
    features are heavily collinear, so starving each tree of columns is what
    buys ensemble diversity. This is the most distinctive Numerai setting.
  * Low learning rate against many trees: 0.001-0.05 with 1,000-10,000 trees.
  * Heavy leaf regularisation: min_data_in_leaf in the thousands, which is
    extreme by general-ML standards and reflects how noisy the target is.

LightGBM is directly evidenced above. XGBoost and CatBoost translate the same
three idioms onto their own parameter names (colsample_bytree / rsm,
min_child_weight / min_data_in_leaf, reg_lambda / l2_leaf_reg) -- that mapping is
reasoned, not measured, so treat those two grids as a starting point.

Note that n_estimators and learning_rate are strongly coupled: sampled
independently, a random search wastes draws on the overfit (high/high) and
underfit (low/low) corners. Give --n-iter enough budget to cover the diagonal.
"""

import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


# Shared ranges, spanning the two configurations every Numerai example notebook
# ships (crypto/example_model, numerai/example_model, hello_numerai,
# target_ensemble) and both legacy baseline configs:
#
#     "small"  n_estimators  2,000   lr 0.01   max_depth  5   num_leaves 2^5-1
#     "deep"   n_estimators 30,000   lr 0.001  max_depth 10   num_leaves 2^10
#              both: colsample_bytree 0.1 ; deep adds min_data_in_leaf 10,000
#
# The notebooks call the deep config "much better, but it requires much more CPU
# and RAM", so the search has to be able to reach it -- an earlier version of
# this grid topped out at 10,000 trees / depth 7 / 255 leaves and could not.
N_ESTIMATORS = sp_randint(2000, 30001)                # spans small -> deep
LEARNING_RATE = np.logspace(-3, -2, 40)               # 0.001 .. 0.01, log-spaced
COLSAMPLE = sp_uniform(loc=0.05, scale=0.25)          # 0.05 .. 0.30, centred on 0.1
SUBSAMPLE = sp_uniform(loc=0.5, scale=0.5)            # 0.50 .. 1.00
MIN_LEAF = sp_randint(1000, 20000)
DEPTHS = list(range(5, 11))                           # 5 .. 10, as the notebooks use


def depth_leaf_grids(shared, leaf_param, leaves_key="num_leaves"):
    """
    Build a list of per-depth grids so num_leaves stays coherent with max_depth.

    The notebooks always tie the two: depth 5 goes with 2^5-1 leaves, depth 10
    with 2^10. Sampling them independently produces combinations that cannot
    exist -- a depth-5 tree has at most 32 leaves, so a draw of num_leaves=1024
    at depth 5 is silently clamped by LightGBM and wastes the draw. Worse, the
    wasted draws are not uniform: most of the num_leaves range is unreachable at
    shallow depths, so the search spends its budget re-testing the same tree.

    RandomizedSearchCV accepts a *list* of grids and samples one uniformly before
    drawing parameters from it, which is how the pairing is enforced.

    :param shared: parameters common to every depth
    :type shared: dict
    :param leaf_param: this library's name for min-samples-per-leaf
    :type leaf_param: str
    :param leaves_key: this library's name for the leaf-count cap, or None if it
                       has none (CatBoost grows symmetric trees, so depth alone
                       determines the leaf count)
    :type leaves_key: str or None
    :return: one grid per depth
    :rtype: [dict]
    """
    grids = []
    for depth in DEPTHS:
        grid = dict(shared)
        grid["max_depth"] = [depth]
        if leaves_key:
            grid[leaves_key] = [2 ** depth - 1]
        grids.append(grid)
    return grids


def scaled_leaf_range(n_rows):
    """
    Rescale the minimum-leaf-size range to the size of the training block.

    Numerai's min_data_in_leaf=10,000 is not a portable constant -- it is about
    0.4% of Classic's ~2.4M training rows. Copied verbatim onto a smaller block
    it becomes a hard constraint rather than a mild one: on an 80k-row Crypto
    block, a draw of 13,000 asks for 16% of the data in every leaf, no split is
    valid, and the model returns a single constant for every row. CORR then comes
    out NaN because a constant prediction has no correlation to compute.

    So express the intent -- "heavy leaf regularisation" -- as a fraction of the
    data, capped by the absolute value the forum and configs actually evidence:

        upper = min(20,000, 2% of rows)
        lower = max(20, 0.02% of rows)

    On full Classic (2.4M rows) that gives 480-20,000, which brackets the
    shipped 10,000. On downsampled Crypto (80k rows) it gives 20-1,600, which
    still regularises hard without silencing the model.

    :param n_rows: rows in the training block
    :type n_rows: int
    :return: a scipy discrete distribution to sample from
    """
    upper = int(min(20000, max(50, n_rows // 50)))
    lower = int(max(20, min(n_rows // 5000, upper // 2)))
    return sp_randint(lower, upper + 1)


class _BoosterWrapper:
    """
    Shared behaviour for the three gradient boosters.

    Each names its minimum-leaf-size parameter differently, so subclasses set
    `leaf_param` and this hook rescales that one entry per refit. Everything else
    in the grid is scale-free and can stay static.
    """

    leaf_param = None

    def fit_grid(self, X, y):
        """
        Rebuild the grid with a leaf range sized to this training block.

        param_grid is a list of per-depth grids, so the rescale is applied to
        each of them.
        """
        leaves = scaled_leaf_range(len(X))
        grids = self.param_grid if isinstance(self.param_grid, list) else [self.param_grid]
        out = []
        for grid in grids:
            grid = dict(grid)
            grid[self.leaf_param] = leaves
            out.append(grid)
        return out


# --------------------------------------------------------------------------- #
# Ridge -- fixed grid
# --------------------------------------------------------------------------- #

class RidgeWrapper:
    """
    Ridge with a fixed, data-independent penalty grid.

    Note logspace rather than linspace. A regularisation strength is a scale
    parameter, so linear spacing over e.g. 0.001-50 puts almost every draw in
    the barely-regularised region and never probes strong shrinkage at all.
    """

    def __init__(self, task="regression", seed=None):
        self.model_name = "ridge"
        self.search_type = "random"

        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            # C is an inverse penalty, so a small C means strong regularisation.
            self.param_grid = {"C": np.logspace(-4, 2, 100)}
            self.ModelClass = LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed
            )
        else:
            from sklearn.linear_model import Ridge
            self.param_grid = {"alpha": np.logspace(-4, 2, 100)}
            self.ModelClass = Ridge(random_state=seed)


# --------------------------------------------------------------------------- #
# Ridge -- glmnet lambda path
# --------------------------------------------------------------------------- #

class RidgeGlmnetWrapper:
    """
    Ridge whose penalty path is built from the data, the way glmnet does it.

    glmnet does not ask you for a grid. It computes lambda_max -- the smallest
    penalty that drives every coefficient to zero -- and lays out `nlambda`
    log-spaced values down to eps * lambda_max. That adapts the path to the
    scale of the problem instead of guessing at it.

    Ridge never drives coefficients to exactly zero, so lambda_max is formally
    infinite. glmnet handles this by substituting alpha = 0.001 into the elastic
    net formula, and we do the same.

    Two things that are easy to get wrong:

    * Scale. glmnet minimises  (1/2N)*RSS + (lambda/2)*||b||^2, while sklearn's
      Ridge minimises  RSS + alpha*||b||^2. Multiplying through by 2N gives
      alpha = lambda * N. Skip that conversion and the entire path sits in the
      wrong regime by a factor of N -- roughly 500k on Crypto, 2.4M on Classic.

    * Leakage. lambda_max depends on y, so it must only ever see training rows.
      That is why this is a fit_grid() hook rather than a static .param_grid:
      each refit builds its own path from data it is allowed to see.

    Because the path scales with N, its alphas land orders of magnitude above the
    fixed RidgeWrapper grid, whose ceiling of 100 is effectively unregularised
    least squares on millions of rows. Running both answers whether Numerai's
    collinear features actually need strong shrinkage.
    """

    def __init__(self, task="regression", seed=None, n_lambda=100):
        self.model_name = "ridge_glmnet"
        self.search_type = "random"
        self.task = task
        self.n_lambda = n_lambda

        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            self.ModelClass = LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed
            )
            self.param_grid = {"C": np.logspace(-4, 2, n_lambda)}
        else:
            from sklearn.linear_model import Ridge
            self.ModelClass = Ridge(random_state=seed)
            self.param_grid = {"alpha": np.logspace(-4, 2, n_lambda)}

    def fit_grid(self, X, y):
        """
        Build the glmnet lambda path from this refit's training block.

        :param X: training features
        :type X: pd.DataFrame
        :param y: training target
        :type y: pd.Series
        :return: param grid for the random search
        :rtype: dict
        """
        alphas = glmnet_alpha_path(X, y, n_lambda=self.n_lambda)

        if self.task == "classification":
            # LogisticRegression takes C = 1 / alpha.
            return {"C": 1.0 / alphas}
        return {"alpha": alphas}


def glmnet_alpha_path(X, y, n_lambda=100, glmnet_alpha=0.001):
    """
    glmnet's lambda path, converted to sklearn's `alpha` scale.

    :param X: training features
    :type X: pd.DataFrame or np.ndarray
    :param y: training target
    :type y: pd.Series or np.ndarray
    :param n_lambda: number of points on the path
    :type n_lambda: int
    :param glmnet_alpha: the elastic-net alpha glmnet substitutes for ridge
    :type glmnet_alpha: float
    :return: alphas, descending (strongest penalty first)
    :rtype: np.ndarray
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n, p = X.shape

    # glmnet standardises the columns and centres y before computing lambda_max.
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0            # a constant column contributes nothing
    x_standardised = (X - x_mean) / x_std
    y_centered = y - y.mean()

    # lambda_max: the smallest penalty at which every coefficient is zero.
    lambda_max = np.abs(x_standardised.T @ y_centered).max() / (n * glmnet_alpha)

    # Degenerate case: a target with no variance gives lambda_max == 0.
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        return np.logspace(-4, 2, n_lambda)[::-1]

    # glmnet's own rule for how far down the path to go.
    eps = 1e-4 if n > p else 1e-2
    lambdas = np.logspace(
        np.log10(lambda_max), np.log10(eps * lambda_max), n_lambda
    )

    # glmnet scale -> sklearn scale. See the class docstring.
    return lambdas * n


# --------------------------------------------------------------------------- #
# Gradient boosting
# --------------------------------------------------------------------------- #

class LGBWrapper(_BoosterWrapper):
    """LightGBM. The ranges here are the directly evidenced ones."""

    leaf_param = "min_child_samples"

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        self.model_name = "lgb"
        self.search_type = "random"
        self.param_grid = depth_leaf_grids({
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "colsample_bytree": COLSAMPLE,
            "subsample": SUBSAMPLE,
            "subsample_freq": [1],           # subsample is ignored without this
            "min_child_samples": MIN_LEAF,   # sklearn API name for min_data_in_leaf
            "reg_lambda": np.logspace(0, 3, 50),
        }, leaf_param="min_child_samples")

        if task == "classification":
            from lightgbm import LGBMClassifier
            self.ModelClass = LGBMClassifier(
                objective="binary", random_state=seed, n_jobs=n_jobs, verbose=-1
            )
        else:
            from lightgbm import LGBMRegressor
            self.ModelClass = LGBMRegressor(
                objective="regression", random_state=seed, n_jobs=n_jobs, verbose=-1
            )


class XGBWrapper(_BoosterWrapper):
    """
    XGBoost, translating the LightGBM idioms onto XGBoost's parameter names.

    min_child_weight is the closest analogue of min_data_in_leaf: for the squared
    error objective each row contributes a hessian of 1, so the threshold is
    effectively a minimum row count and the same scaled range applies.
    """

    leaf_param = "min_child_weight"

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        self.model_name = "xgb"
        self.search_type = "random"
        self.param_grid = depth_leaf_grids({
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "colsample_bytree": COLSAMPLE,
            "subsample": SUBSAMPLE,
            "min_child_weight": MIN_LEAF,
            "reg_lambda": np.logspace(0, 3, 50),
        }, leaf_param="min_child_weight", leaves_key=None)

        # hist is much faster than exact on data this size and costs nothing here,
        # since the features are already binned to 5 levels.
        common = {"tree_method": "hist", "random_state": seed,
                  "n_jobs": n_jobs, "verbosity": 0}

        if task == "classification":
            from xgboost import XGBClassifier
            self.ModelClass = XGBClassifier(objective="binary:logistic", **common)
        else:
            from xgboost import XGBRegressor
            self.ModelClass = XGBRegressor(objective="reg:squarederror", **common)


class CatBoostWrapper(_BoosterWrapper):
    """
    CatBoost. `rsm` is its column-sampling parameter and `l2_leaf_reg` its L2.

    Two CatBoost-specific notes:

    * allow_writing_files=False. CatBoost otherwise drops a catboost_info/
      directory into the working directory on every single fit, which under a
      random search means hundreds of them.
    * CatBoost grows symmetric (oblivious) trees, which are more constrained than
      LightGBM's leaf-wise growth. That is usually an advantage on data this
      noisy, but it means `depth` is not directly comparable to LightGBM's.
    """

    leaf_param = "min_data_in_leaf"

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        self.model_name = "catboost"
        self.search_type = "random"
        # CatBoost names depth "depth", grows symmetric trees (so the leaf count
        # is 2^depth by construction, nothing to pair), and its own cap is 16.
        self.param_grid = [
            {"iterations": N_ESTIMATORS, "learning_rate": LEARNING_RATE,
             "depth": [d], "rsm": COLSAMPLE, "min_data_in_leaf": MIN_LEAF,
             "l2_leaf_reg": np.logspace(0, 2, 50)}
            for d in DEPTHS
        ]

        common = {"random_seed": seed, "thread_count": n_jobs,
                  "allow_writing_files": False, "verbose": 0}

        if task == "classification":
            from catboost import CatBoostClassifier
            self.ModelClass = CatBoostClassifier(loss_function="Logloss", **common)
        else:
            from catboost import CatBoostRegressor
            self.ModelClass = CatBoostRegressor(loss_function="RMSE", **common)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#
# Same plain-dict dispatch the CLI uses for tournaments. Adding a model means
# writing a wrapper and adding one line here.
#
MODELS = {
    "ridge": RidgeWrapper,
    "ridge_glmnet": RidgeGlmnetWrapper,
    "lgb": LGBWrapper,
    "xgb": XGBWrapper,
    "catboost": CatBoostWrapper,
}


def build_wrapper(model_name, task, seed=None, n_jobs=-1):
    """
    Instantiate a model wrapper by name.

    :param model_name: key into MODELS
    :type model_name: str
    :param task: "regression" or "classification"
    :type task: str
    :param seed: random seed, passed to both the search and the estimator
    :type seed: int or None
    :param n_jobs: worker count for the estimators that accept one
    :type n_jobs: int
    :return: a configured wrapper
    """
    if model_name not in MODELS:
        raise ValueError(
            "unknown model '{}', expected one of {}".format(
                model_name, sorted(MODELS)
            )
        )

    Wrapper = MODELS[model_name]

    # The linear wrappers have no use for a worker count.
    if model_name in ("ridge", "ridge_glmnet"):
        return Wrapper(task=task, seed=seed)
    return Wrapper(task=task, seed=seed, n_jobs=n_jobs)
