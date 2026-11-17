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
reasoned, not measured, so treat those two grids as a starting point. The MLP
grid carries the same caveat and only two of the three idioms: heavy
regularisation and a low learning rate map onto `alpha` and
`learning_rate_init`, while column subsampling has no analogue in a dense layer.
See MLPWrapper's docstring.

Note that n_estimators and learning_rate are strongly coupled: sampled
independently, a random search wastes draws on the overfit (high/high) and
underfit (low/low) corners. Give --n-iter enough budget to cover the diagonal.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from src.prediction.generalised_ridge import GeneralisedRidge, parse_group_spec
from src.prediction.sur_ridge import SurRidge, SurAugment
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
# Ridge -- generalised (W, b0, Delta)
# --------------------------------------------------------------------------- #

class GenRidgeWrapper:
    """
    Generalised Ridge: a weighting matrix, a coefficient prior and a shaped
    penalty, over the same walk-forward the other linear wrappers use.

    See prediction.generalised_ridge for the maths. What belongs HERE is the
    search space, and two decisions about it.

    ONE DICT, never a list of sub-grids. ParameterSampler enumerates the union
    of a list of sub-grids and samples without replacement, so draws land in
    proportion to sub-grid CARDINALITY -- a small sub-grid beside a large one
    gets a couple of draws out of fifty and the arm is starved without anything
    saying so. With a single dict, `structure` is sampled evenly.

    NO fit_grid. Every knob here is dimensionless -- `alpha` is a ratio to the
    Gram's own scale, `group_penalty` is a ratio, the two mixing weights are on
    [0, 1] -- so one static grid is correct at any row count and any width.
    ridge_glmnet needs fit_grid because its path is in absolute units; this one
    does not, and that buys something ridge_glmnet cannot have: --fixed-params
    BYPASSES fit_grid by design, so a pinned glmnet slot silently gets a penalty
    from a different regime, while a pinned alpha here means what it meant in
    the backtest.

    Which FAMILIES the search may choose among is a constructor option, reached
    through --model-options; which family a given fit uses is an estimator
    parameter, so it lands in best_params_ and can be pinned. The W and b0
    families are off by default: on the rank-binned raw 22 every column shares a
    scale, and a marginal prior is then a positive rescaling of the coefficient
    vector -- rank-identical, and numerai_corr is a rank metric, so it would
    burn draws on an axis that cannot move the score.
    """

    TASKS = ("regression",)

    #: The walk-forward routes the DATE of each row to fit() for wrappers that
    #: declare this. It is a per-wrapper gate rather than an unconditional fit
    #: param because ridge, ridge_glmnet, lgb, xgb and catboost would all ACCEPT
    #: a sample_weight and silently change their fits, while mlp, torch_mlp and
    #: torch_mlp_emb are Pipelines and REJECT any fit param outright
    #: ("Pipeline.fit does not accept the sample_weight parameter"). Measured on
    #: all nine wrappers.
    WANTS_PERIOD = True

    def __init__(self, task="regression", seed=None, n_jobs=None,
                 structures=("identity", "group"),
                 w_schemes=("uniform",),
                 b0_schemes=("zero",),
                 group_specs=None,
                 half_lives=(32, 45, 65, 90, 130),
                 alpha_lo=-4.0, alpha_hi=2.0,
                 n_alpha=60):
        # Accepted and dropped: the solve is a single Cholesky and sklearn's
        # threading is BLAS-level. Staying OUT of build_wrapper's
        # ("ridge", "ridge_glmnet") tuple is what keeps --model-options legal,
        # and that tuple is what forces this signature.
        del n_jobs
        if task != "regression":
            raise ValueError(
                "gen_ridge does not implement task '{}'; there is no closed "
                "form for a penalised logistic fit".format(task))

        structures = tuple(structures)
        w_schemes = tuple(w_schemes)
        b0_schemes = tuple(b0_schemes)
        for chosen, allowed, flag in (
                (structures, GeneralisedRidge.STRUCTURES, "structures"),
                (w_schemes, GeneralisedRidge.W_SCHEMES, "w_schemes"),
                (b0_schemes, GeneralisedRidge.B0_SCHEMES, "b0_schemes")):
            if not chosen:
                raise ValueError("--model-options {} must not be empty".format(flag))
            unknown = [value for value in chosen if value not in allowed]
            if unknown:
                raise ValueError(
                    "--model-options {} got {}, expected values from {}".format(
                        flag, unknown, list(allowed)))

        self.model_name = "gen_ridge"
        self.search_type = "random"
        self.structures = structures
        self.w_schemes = w_schemes
        self.b0_schemes = b0_schemes
        self.half_lives = tuple(half_lives)
        # Validated at CONSTRUCTION so a typo fails at argument parsing rather
        # than at the first refit, after the data has loaded. The stems cannot
        # be checked here -- they depend on the dataset -- so fit() checks those.
        self.group_specs = tuple(group_specs) if group_specs else None
        if self.group_specs:
            for spec in self.group_specs:
                parse_group_spec(spec)

        # alpha_scale is pinned rather than searched: "absolute" exists only so
        # the equivalence test against sklearn's Ridge can be written, and
        # searching it would make `alpha` mean two different things in one grid.
        self.ModelClass = GeneralisedRidge(alpha_scale="gram", random_state=seed)

        # A dimensionless ratio to the Gram's mean live diagonal. The span
        # brackets near-OLS and near-total shrinkage at every width tried, and
        # the log spacing is what makes search_spaces infer a log-uniform prior
        # under --search-type bayes.
        # Exponents, base 10. The default span brackets near-OLS and
        # near-total shrinkage at every width tried. It is adjustable because
        # the bottom of it is a real regime and sometimes the wrong one: on
        # allfeat_pp the search put alpha on the 1e-4 floor at 16 of 44 refits,
        # which is an unregularised fit on 812 columns, and a sign-constrained
        # solve there does not reach optimality in 2000 sweeps. Raising the
        # floor keeps a constrained arm and its control searching the SAME
        # space, so the comparison measures the constraint rather than a
        # silently truncated grid.
        if float(alpha_lo) >= float(alpha_hi):
            raise ValueError("alpha_lo must be below alpha_hi, got %r and %r"
                             % (alpha_lo, alpha_hi))
        self.alpha_lo, self.alpha_hi = float(alpha_lo), float(alpha_hi)
        grid = {"alpha": np.logspace(self.alpha_lo, self.alpha_hi, n_alpha)}

        # A list of strings, never a bare string: ParameterSampler samples
        # individual CHARACTERS out of a bare string as soon as any sibling
        # entry is a scipy distribution.
        grid["structure"] = list(structures)
        if "group" in structures:
            # Bottom of the range is a measured no-op, top is the fused limit
            # where a group collapses to one effective coefficient.
            grid["group_penalty"] = np.logspace(-2, 4, 25)

        if w_schemes != ("uniform",):
            grid["w_scheme"] = list(w_schemes)
            # Added PER SCHEME, not whenever the default is overridden: asking
            # for ("uniform", "exp_decay") must not drag a tail_weight dimension
            # into the grid that no draw can ever use, burn search budget on it
            # and then record it in the hyperparams CSV as if it meant something.
            if "tail" in w_schemes:
                # A convex mixing weight, so LINEAR and anchored at exactly 0.0
                # -- the W = I reduction, so the family nests vanilla ridge.
                grid["tail_weight"] = np.linspace(0.0, 0.8, 9)
            if "exp_decay" in w_schemes:
                # In DATES. Floored at 32 because a 130-date block decayed at a
                # 21-date half-life retains 59 effective dates -- about 2.9
                # effective independent windows at a 20-day overlapping target,
                # against 6.5 uniform. Shorter than 32 is fittable noise and
                # belongs in a PINNED arm, never in a searched dimension.
                grid["half_life"] = np.asarray(self.half_lives, dtype=float)

        if b0_schemes != ("zero",):
            grid["b0_scheme"] = list(b0_schemes)
            # Anchored at 0.0 so the family is a proper superset of b0 = 0.
            grid["prior_shrinkage"] = np.linspace(0.0, 1.5, 16)

        if self.group_specs:
            # A LIST OF STRINGS, so each draw picks one whole specification.
            # to_dimension turns it into a Categorical, which both search
            # backends handle, and it lands in best_params_, the metrics JSON
            # and one hyperparams CSV cell intact.
            grid["group_spec"] = list(self.group_specs)

        self.param_grid = grid


# --------------------------------------------------------------------------- #
# Gradient boosting
# --------------------------------------------------------------------------- #

class SurRidgeWrapper:
    """
    Multi-symbol multivariate ridge: feasible GLS across the cross-section.

    The contrast to `ridge_glmnet` (one shared beta on the STACKED panel, errors
    treated i.i.d.). `sur_ridge` fits the same shared beta but partials out k
    common CROSS-SYMBOL factors from the residuals -- the GLS adjustment under a
    factor covariance -- so it exploits the fact that coins' errors co-move within
    a date. n_factors=0 is an exact stacked-ridge control, so the value of the
    method is the searched k>0 against it. See src/prediction/sur_ridge.py.

    The symbol of each row is routed to fit() (WANTS_SYMBOL) alongside the date
    (WANTS_PERIOD); neither is needed at predict time. Both regression and the
    pipeline's binary classification are implemented. --model-options tune the
    factor grid, the min-symbol-rows floor and the alpha span WITHOUT collapsing
    the search (unlike --fixed-params).
    """

    WANTS_PERIOD = True
    WANTS_SYMBOL = True

    def __init__(self, task="regression", seed=None, n_jobs=None,
                 n_factors=(0, 1, 2, 3, 5), min_symbol_rows=60,
                 alpha_lo=0.0, alpha_hi=5.0, n_alpha=50):
        del n_jobs                       # the solve is a single Cholesky
        self.model_name = "sur_ridge"
        self.search_type = "random"
        self.task = task
        self.n_factors = tuple(int(k) for k in n_factors)
        if any(k < 0 for k in self.n_factors):
            raise ValueError("n_factors must be non-negative, got %r" % (n_factors,))
        self.min_symbol_rows = int(min_symbol_rows)
        if float(alpha_lo) >= float(alpha_hi):
            raise ValueError("alpha_lo must be below alpha_hi, got %r and %r"
                             % (alpha_lo, alpha_hi))
        self.ModelClass = SurRidge(task=task, min_symbol_rows=self.min_symbol_rows,
                                   random_state=seed)
        # alpha is on the internally STANDARDISED design, so its scale tracks the
        # Gram diagonal (~ n rows); the span brackets near-OLS to strong shrinkage
        # on the ~39k-row blocks here. n_factors is a small discrete list; k=0 is
        # the stacked-ridge control the search is free to pick.
        self.param_grid = {
            "alpha": np.logspace(float(alpha_lo), float(alpha_hi), int(n_alpha)),
            "n_factors": list(self.n_factors),
        }


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


class LGBKtzWrapper(_BoosterWrapper):
    """
    LightGBM with the E27 (Kalfa-Timmermann-van der Zwan) priors compiled into a
    search grid -- the booster analogue of MLPKtzWrapper's E17 grid.

    E27 swept a 315-cell factorial and read off, from the payout marginals, where
    the good models live and which hyperparameters actually move the outcome:

        num_leaves        the stump (2) has the highest, tightest marginal and
                          NEVER drops below the overall median; payout falls
                          monotonically to num_leaves=255. So favour small.
        learning_rate     0.001 best, 0.3 worst; the single largest median-shift.
        colsample_bytree  0.05 tight and high, 1.0 locks in a bad median; second
                          largest shift.
        n_estimators      matters only through L = learning_rate * n_estimators,
                          which is very nearly a sufficient statistic (equal-L
                          configs agree to ~1e-3) and whose median payout is
                          monotone decreasing. L <= ~40 is the good side.

    THE EFFORT ORDERING IS THE GRID SHAPE, not an assertion. RandomizedSearchCV
    samples the discrete grid uniformly, so an axis with MORE levels gets more of
    its range explored per fixed --n-iter. The level counts encode
    learning_rate >= colsample_bytree > num_leaves >= n_estimators:

        learning_rate     12 log values in [0.001, 0.01]   the tuning focus
        colsample_bytree   6 values in [0.05, 0.20]         second focus
        num_leaves         3 values {2, 4, 8}               coarse, stump-favoured
        n_estimators       2 values {2000, 4000}            fewest; L<=40 either way

    The two known bad corners are UNREACHABLE by construction: learning_rate
    stops at 0.01 (0.02-0.3 locked in worse medians), colsample_bytree stops at
    0.20 (0.2 and 1.0 dropped), and with n_estimators<=4000 the product L never
    exceeds 40, so the sweep cannot wander into the monotone-bad L region.

    min_child_samples is PINNED at 100 (E27's fixed value), so fit_grid does NOT
    rescale it the way the plain LGBWrapper does -- that rescale is overridden
    below. max_depth=-1 in every cell makes num_leaves the sole complexity axis,
    exactly as E27 ran it.
    """

    leaf_param = "min_child_samples"

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        self.model_name = "lgb_ktz"
        self.search_type = "random"
        self.param_grid = {
            "learning_rate": list(np.logspace(-3, -2, 12)),
            "colsample_bytree": [0.05, 0.08, 0.11, 0.14, 0.17, 0.20],
            "num_leaves": [2, 4, 8],
            "n_estimators": [2000, 4000],
            "max_depth": [-1],
            "min_child_samples": [100],
            "subsample": [0.8],
            "subsample_freq": [1],
            "reg_lambda": [10.0],
        }

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

    def fit_grid(self, X, y):
        """
        Return the static prior grid unchanged.

        The E27 method pins min_child_samples at 100, so -- unlike the plain
        booster wrapper, which rescales the leaf floor to each refit's block --
        this grid must NOT be touched per refit, or the pinned value the priors
        were derived under would be silently overwritten.
        """
        return self.param_grid


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


class CatBoostKtzWrapper:
    """
    CatBoost with the KTZ/E27 booster priors -- the SEARCH sibling of lgb_ktz.

    The booster analogue of E29's lgb_ktz and E28's mlp_ktz: a compact grid that
    maps the three Numerai booster idioms plus the E27 "favour shallow, span the
    learning rate" finding onto CatBoost's parameter names, searched by
    RandomizedSearchCV. It is the select-best BASELINE that E32's grid-ensemble
    arm (catboost_ktz_ens) is measured against.

    ONE GRID, TWO CONSUMERS. The 12 cells live in CatBoostGridEnsemble as class
    constants and are read here so the searched space and the ensembled space are
    provably identical -- ensemble-of-12 vs CV-pick-1-of-12 with nothing else
    changed. The grid is deliberately compact (|lr| x |depth| x |rsm| = 3x2x2)
    precisely so the ensemble variant can fit ALL of it every refit; lgb_ktz's
    432-cell grid is meant to be SAMPLED, this one to be swept. Run the search
    with --n-iter 12 to sweep it exhaustively.

    A SINGLE dict, not a list of per-depth sub-grids: CatBoost's depth is a
    scalar (symmetric trees, leaf count 2^depth), so there is no num_leaves to
    keep coherent with it the way the plain CatBoostWrapper and the LightGBM
    grids pair the two.

    Regression only, like tabnet/tabpfn: the study scores E[y|x] on the binned
    target. The grid is exposed through --model-options (learning_rates, depths,
    rsms, iterations, l2_leaf_reg, min_data_in_leaf) so a scout can shrink it.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1,
                 learning_rates=None, depths=None, rsms=None,
                 iterations=None, l2_leaf_reg=None, min_data_in_leaf=None):
        if task != "regression":
            raise ValueError(
                "catboost_ktz implements regression only; a CatBoost classifier "
                "grid is a separate, untested question no study has asked for.")

        from src.prediction.catboost_ensemble import CatBoostGridEnsemble
        from catboost import CatBoostRegressor

        lrs = (CatBoostGridEnsemble.LEARNING_RATES if learning_rates is None
               else learning_rates)
        depths_ = CatBoostGridEnsemble.DEPTHS if depths is None else depths
        rsms_ = CatBoostGridEnsemble.RSMS if rsms is None else rsms
        iterations_ = (CatBoostGridEnsemble.ITERATIONS if iterations is None
                       else int(iterations))
        l2 = (CatBoostGridEnsemble.L2_LEAF_REG if l2_leaf_reg is None
              else float(l2_leaf_reg))
        min_leaf = (CatBoostGridEnsemble.MIN_DATA_IN_LEAF if min_data_in_leaf is None
                    else int(min_data_in_leaf))

        self.model_name = "catboost_ktz"
        self.search_type = "random"
        self.ModelClass = CatBoostRegressor(
            loss_function="RMSE", random_seed=seed, thread_count=n_jobs,
            allow_writing_files=False, verbose=0)
        self.param_grid = {
            "learning_rate": [float(x) for x in lrs],
            "depth": [int(d) for d in depths_],
            "rsm": [float(r) for r in rsms_],
            "iterations": [iterations_],
            "l2_leaf_reg": [l2],
            "min_data_in_leaf": [min_leaf],
        }


class CatBoostKtzEnsembleWrapper:
    """
    The catboost_ktz grid as an ENSEMBLE, not a search (E32).

    The CatBoost member of the "ensemble the ktz grid instead of searching it"
    family -- the sibling of mlp_ktz_ens (E31) and the lgb grid ensemble (E30).
    It fits every one of the same 12 cells CatBoostKtzWrapper searches and
    rank-averages their predictions rather than letting an inner CV keep one, so
    no fit is discarded and no selection variance is paid on ~26 effective
    windows. See prediction.catboost_ensemble.CatBoostGridEnsemble for the
    combination rule (block-level rank-average, matching mlp_ktz_ens) and the
    cost note: run with --n-splits 2 so the unused inner CV costs one fold, not
    five.

    Unlike GridEnsembleRegressor (which rebuilds the mlp_ktz grid and clones a
    Pipeline per cell), the CatBoost members are constructed directly by the
    estimator from the shared grid constants -- there is no per-cell scaler to
    fit and CatBoost is imported lazily there. A fitted ensemble survives
    joblib.dump, so a catboost_ktz_ens slot is deployable through the same
    build_wrapper + joblib.dump(best_estimator_) path as every other MODELS entry.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1, combine="rank",
                 learning_rates=None, depths=None, rsms=None,
                 iterations=None, l2_leaf_reg=None, min_data_in_leaf=None):
        if task != "regression":
            raise ValueError(
                "catboost_ktz_ens is regression-only; got task=%r" % (task,))

        from src.prediction.catboost_ensemble import CatBoostGridEnsemble

        self.model_name = "catboost_ktz_ens"
        self.search_type = "random"
        self.ModelClass = CatBoostGridEnsemble(
            task=task, seed=seed, thread_count=n_jobs, combine=combine,
            learning_rates=learning_rates, depths=depths, rsms=rsms,
            iterations=iterations, l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf)
        # Nothing to search: one candidate. `combine` is a real estimator
        # parameter, so pinning it lands it in best_params_ and the hyperparameter
        # history rather than searching an empty grid.
        self.param_grid = {"combine": [combine]}


# --------------------------------------------------------------------------- #
# Random forest -- bagged trees, the one tree ensemble that does not boost
# --------------------------------------------------------------------------- #

class RFWrapper:
    """
    Random forest, wrapped like the boosters but WITHOUT a learning rate.

    A forest is bagging, not boosting: every tree is fit independently on a
    bootstrap sample and their predictions are AVERAGED, so there is no
    learning_rate and no n_estimators/learning_rate coupling to worry about --
    more trees only ever reduces variance (with diminishing returns), it never
    overfits the way a booster's late trees do. That single difference is why
    this is a separate wrapper rather than a fourth `_BoosterWrapper` subclass,
    and it reshapes which knobs matter.

    --------------------------------------------------------------------------
    The three Numerai idioms, mapped onto a forest
    --------------------------------------------------------------------------

    The module header names three settings that make Numerai different from a
    generic tabular benchmark. Two map cleanly onto a forest; the third does not,
    and its budget goes to the two axes a forest has instead.

      * Very low column sampling -> `max_features`. This is the DIRECT analogue
        of the boosters' colsample_bytree and the most distinctive Numerai
        setting: the features are heavily collinear, so starving each split of
        columns is what buys the ensemble its diversity. A forest already
        subsamples columns by default (regression default is all columns, but
        the classic RF recipe is a fraction), so the useful region here runs from
        ~0.1 up, not the 0.6-1.0 a general benchmark reaches for. The search
        spans 0.05..1.0 so the low corner is reachable.
      * Heavy leaf regularisation -> `min_samples_leaf`. The forest analogue of
        min_data_in_leaf, and rescaled to the training block the same way, for
        the same reason: a fixed 10,000 is 0.4% of Classic's rows but 25% of a
        crypto block, where it would force every leaf to a constant. fit_grid
        rescales it per refit via scaled_leaf_range, exactly as the boosters do.
      * Low learning rate against many trees -> NO forest analogue. There is no
        learning rate to lower. `n_estimators` is still here -- more trees lower
        the averaging variance -- but it is a variance knob, not the coupled
        capacity knob it is for a booster, so it is swept coarsely.

    `max_depth` is the forest's own capacity control (a booster leans on
    num_leaves + a low learning rate instead). None -- grow every tree out fully
    and lean entirely on bagging + column subsampling to control variance -- is
    the textbook forest default and is offered alongside shallow caps so the
    search can find where on that axis this noisy target actually wants to sit.

    --------------------------------------------------------------------------
    The search grid
    --------------------------------------------------------------------------

    A SINGLE dict, not a list of per-depth sub-grids: a forest has no
    num_leaves to keep coherent with max_depth (the boosters pair the two via
    depth_leaf_grids), so there is nothing to pin per depth and one flat grid is
    correct. max_depth carries an explicit None so the search can reach the
    grow-out-fully regime; RandomizedSearchCV samples the mixed int/None list
    fine, and a pinned run takes JSON null for the same value.

    Regression and classification both, like the boosters. The forest classifier
    exposes predict_proba, which is the continuous signal numerai_corr needs;
    the studies that use this wrapper score regression on the binned target.
    """

    leaf_param = "min_samples_leaf"

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        self.model_name = "rf"
        self.search_type = "random"
        # max_features is the colsample idiom (0.05..1.0, centred low); max_depth
        # spans shallow caps through grow-out-fully (None); n_estimators is the
        # coarse variance knob; min_samples_leaf is rescaled per refit by fit_grid.
        self.param_grid = {
            "n_estimators": sp_randint(100, 1001),          # 100 .. 1000
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 16, None],  # None = grow fully
            "max_features": sp_uniform(loc=0.05, scale=0.95),  # 0.05 .. 1.00
            "min_samples_leaf": MIN_LEAF,                    # rescaled, see fit_grid
            "bootstrap": [True],
        }

        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            self.ModelClass = RandomForestClassifier(
                random_state=seed, n_jobs=n_jobs)
        else:
            from sklearn.ensemble import RandomForestRegressor
            # The split criterion is left at its default -- squared error in
            # every sklearn version -- rather than named explicitly: the label
            # changed ("mse" -> "squared_error") at sklearn 1.0, so pinning
            # either string breaks on the other side of that release. The default
            # is the value we want on both.
            self.ModelClass = RandomForestRegressor(
                random_state=seed, n_jobs=n_jobs)

    def fit_grid(self, X, y):
        """
        Rescale the leaf floor to this refit's block, like _BoosterWrapper.

        min_samples_leaf is an absolute row count, so a value tuned on one block
        size is a different constraint on another. scaled_leaf_range expresses
        the "heavy leaf regularisation" intent as a fraction of the rows, capped
        by the absolute values the forum and configs evidence. The rest of the
        grid is scale-free and passes through untouched.

        A PINNED run (--fixed-params) never reaches this hook -- run_forecast
        disables the search and sets the value verbatim -- so a sweep's pinned
        min_samples_leaf means exactly what it says, which is what the E48
        sensitivity study relies on.
        """
        grid = dict(self.param_grid)
        grid[self.leaf_param] = scaled_leaf_range(len(X))
        return grid


class RFKtzWrapper(RFWrapper):
    """
    Random forest with the E48 (Kalfa-Timmermann-van der Zwan) priors compiled
    into a search grid -- the forest analogue of MLPKtzWrapper (E17) and
    LGBKtzWrapper (E27). E48 swept a 192-cell factorial on the raw 22 and read
    off, from the payout marginals and a functional-ANOVA decomposition, where
    the good forests live and which knobs actually move the outcome. This wrapper
    is that reading turned into a grid.

    --------------------------------------------------------------------------
    What E48 found, and how each finding becomes a grid choice
    --------------------------------------------------------------------------

    n_estimators is a RED HERRING (fANOVA main-effect share 0.0002; medians 100
    vs 300 vs 1000 agree to 0.0015 payout). A forest averages independent trees,
    so past ~100 the variance is already gone. So it is PINNED at 100 -- the grid
    spends none of its search budget there, exactly the E17-style "do not tune
    the axis that does not move" discipline.

    max_depth and min_samples_leaf are the two axes that matter (fANOVA 0.28 and
    0.31) AND they interact strongly (0.25) -- a deep tree is fine only if the
    leaf floor is heavy, because the two are alternate routes to the same
    complexity control. Both point the Numerai way: SHALLOW (depth 3 best,
    monotone worse to None) and HEAVY LEAF (monotone better to the grid's
    ceiling). So depth is a shallow gate {3, 4, 6} and the leaf floor is a heavy
    sweep.

    THE LEAF CEILING WAS BINDING. min_samples_leaf's marginal rose monotonically
    to 1000 -- the largest value E48 tried -- so 1000 is the grid EDGE, not a
    measured interior optimum. This is the same trap E07's alpha ceiling was
    (best value pinned to the boundary means the boundary bound the optimum), and
    the E17/E33 remedy is to OPEN it. So this grid runs the leaf floor to 4000
    (~10% of a 39k-row block, still splittable at these depths) to find where it
    actually peaks rather than assuming 1000.

    max_features is the Numerai colsample idiom and matters moderately (0.12);
    low wins monotonically (0.1 best, 1.0 worst, 0/48 of the good cells at 1.0).
    So the grid is a DENSE LOW sweep {0.05 .. 0.30} and 0.6/1.0 are dropped --
    the bad corner is unreachable by construction.

    Grid size 1 x 3 x 5 x 4 = 60 cells; every cell sits in E48's good region, so
    a random search only ever fine-tunes. Run it with --n-iter ~40.

    Like LGBKtzWrapper, fit_grid returns the static grid UNCHANGED: the leaf
    values are the prior itself (absolute counts the CURUPIRA 130-period block
    makes ~39k rows), so the plain RFWrapper's per-block leaf rescale must NOT
    fire here or it would overwrite the very values the study identified.
    """

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        super().__init__(task=task, seed=seed, n_jobs=n_jobs)
        self.model_name = "rf_ktz"
        self.param_grid = {
            "n_estimators": [100],                       # red herring -- pinned
            "max_depth": [3, 4, 6],                       # shallow gate
            "max_features": [0.05, 0.10, 0.15, 0.20, 0.30],  # dense low colsample
            "min_samples_leaf": [300, 1000, 2000, 4000],     # heavy; ceiling OPENED past E48's 1000
            "bootstrap": [True],
        }

    def fit_grid(self, X, y):
        """Return the static prior grid unchanged -- see the class docstring on
        why the leaf floor must not be rescaled per block here."""
        return self.param_grid


# --------------------------------------------------------------------------- #
# MLP -- the one non-linear, non-tree function class
# --------------------------------------------------------------------------- #

class ScaleContinuousOnly(BaseEstimator, TransformerMixin):
    """
    StandardScaler for real features; 0/1 dummy columns pass through untouched.

    Standardising a binary column divides it by sqrt(p(1-p)), so a RARE dummy
    becomes an enormous spike. Measured on a real 130-period block of
    crypto/v2.0-onehot, which carries 740 one-hot ticker columns:

        the raw 22 features after scaling   max |z| =   2.29
        the 740 ticker dummies              max |z| = 197.48

    A ticker present on 3 rows of 39,000 lands at z = 197 beside features that
    never leave +-2.3, so the first layer sees an input 86x larger than anything
    real. 346 of the 740 dummies do not occur at ALL in a given block. The
    failure is silent: the run completes and simply scores worse, which reads as
    "the one-hot block does not help" rather than "the preprocessing broke it".

    Dummies are detected FROM THE DATA -- every value in {0, 1} -- not from a
    name prefix. build_onehot happens to name them feature_sym_*, but a
    name-based rule would miss any future dummy block and is crypto-specific.
    One pass over the array, no nunique(): 0.48s on 39,000 x 762.

    On a dataset with NO binary columns this is exactly StandardScaler, which is
    what keeps every existing arm comparable -- raw, tenc, interact, fenccausal,
    cumlag3, rollsel and spread reproduce unchanged. Only the four one-hot
    datasets were ever affected, and no MLP arm had been run on them.

    NOTE TabNetWrapper still uses a plain StandardScaler and has the same
    exposure on those datasets; it belongs to another line of work and is left
    alone rather than changed underneath it.
    """

    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler

        values = np.asarray(X, dtype="float64")
        self.binary_ = ((values == 0) | (values == 1)).all(axis=0)
        self.scaler_ = StandardScaler().fit(values[:, ~self.binary_])
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype="float64").copy()
        values[:, ~self.binary_] = self.scaler_.transform(values[:, ~self.binary_])
        return values


class MLPWrapper:
    """
    A classical feedforward net, wrapped exactly like ridge and the boosters.

    This is a ROW-WISE estimator: one row in, one score out, no state carried
    between rows. That is the same shape as every other wrapper in this module,
    which is why it belongs here rather than in prediction/nn/ -- that package is
    a sequence model with its own trainer and its own walk-forward, and nothing in
    src/live/ can deploy it. Anything in MODELS is deployable to a CURUPIRA slot
    for free, because live/refit.py builds slots through build_wrapper().

    Backed by sklearn rather than torch on purpose. torch is an OPTIONAL
    dependency (see setup.sh) and this module is currently torch-free, so an
    import here would make `--model mlp` fail on any box where the wheel did not
    install -- for a model that needs none of what torch offers.

    --------------------------------------------------------------------------
    Why the estimator is a Pipeline
    --------------------------------------------------------------------------

    Ridge and the trees are indifferent to feature scale. An MLP is not: unscaled
    inputs put the first layer's activations wherever the raw units happen to
    land, and gradient descent then spends its budget undoing that. So ModelClass
    is a Pipeline of StandardScaler + the net, which also gets the scaler FIT
    INSIDE each CV fold rather than once over the whole block -- fitting it up
    front would leak the validation fold's mean and variance into training.

    Crypto's 22 shipped features are already binned to {0, .25, .5, .75, 1}, so
    there the scaler does little. It matters on the wide derived sets (786 and 812
    columns), where a target encoding and a one-hot share a matrix, and getting it
    wrong is silent -- the run completes and simply scores worse.

    Grid keys are therefore prefixed `mlp__`, which is how sklearn addresses a
    step's parameters. clone() handles Pipelines, so hyper_params_search() needs
    no special case.

    --------------------------------------------------------------------------
    This is the one wrapper --search-type bayes cannot search
    --------------------------------------------------------------------------

    hidden_layer_sizes is a TUPLE per architecture, and skopt has no
    representation for a category that is not a scalar -- even a single-category
    dimension dies inside BayesSearchCV.fit(). run_forecast.py refuses the
    combination up front and search_spaces.build_bayes_search() refuses it again;
    see search_spaces.sequence_valued_params() for why it is reported rather than
    worked around. --search-type random is the default and searches this grid
    exactly as written.

    --------------------------------------------------------------------------
    early_stopping is OFF, and must stay off
    --------------------------------------------------------------------------

    sklearn's `early_stopping=True` holds out a RANDOM `validation_fraction` of
    rows. On a panel that draws the holdout from the same periods it trains on,
    which is precisely the leak purged_period_splits() exists to prevent -- and it
    would not announce itself, it would just make the CV score optimistic. The
    regularisation here is `alpha` plus a fixed `max_iter`; the honest cost is
    that a candidate can stop short of convergence, which sklearn reports as a
    ConvergenceWarning and which the search sees as a worse score. That is the
    right trade: an under-trained candidate loses fairly, a leaked one wins
    unfairly.

    --------------------------------------------------------------------------
    Where these ranges come from
    --------------------------------------------------------------------------

    Reasoned rather than measured, apart from `batch_size` below -- treat the
    rest as a starting point, the same caveat this module's header already
    applies to the xgb and catboost grids. Two of the three Numerai idioms in
    that header do carry over:

      * Heavy regularisation. `alpha` spans 1e-4 to 1e+1, six orders of
        magnitude, log-spaced because a penalty is a scale parameter. The upper
        end is extreme by general-ML standards and is there for the same reason
        min_data_in_leaf runs into the thousands for the boosters: the target is
        mostly noise, and the useful region is much closer to a linear model than
        a generic tabular benchmark would suggest. The ceiling is MEASURED: it
        was raised to 1e2, cost ~0.007 payout across three studies, and was put
        back. See ALPHAS for the numbers.
      * Low learning rate. 1e-4 to 1e-2, log-spaced, against the iteration
        ceiling described below.

    The third -- very low column sampling -- has no MLP analogue; a dense layer
    sees every column by construction. `alpha` is doing that job instead.

    Architectures are deliberately small and span narrow-shallow to wider-deep.
    On 22 features a large net is all capacity and no signal, and the widest entry
    here is aimed at the 786/812-column sets.

    `batch_size` is the one entry here that IS measured. On a real 130-period
    rolling block (39,000 rows x 22 features, which is exactly what the CURUPIRA
    cadence hands each refit) the small batches every general-ML default reaches
    for are three times more expensive for nothing:

        arch        batch 128    batch 1024    batch 4096
        (32,)            6.5s          2.4s          2.6s
        (128,)          21.1s         10.1s          3.7s
        (256, 128)     110.6s         33.4s         32.6s

    The iteration COUNT barely moves (30-115 across all of them); what changes is
    the number of gradient steps inside each one, since a 39,000-row block at
    batch 128 is 305 steps per iteration and at batch 4096 is 10. A search draws
    250 fits per refit and there are four refits, so a mispriced batch range is
    the difference between a three-hour arm and a ten-hour one. The grid starts
    at 1024 for that reason.

    --------------------------------------------------------------------------
    What actually stops training
    --------------------------------------------------------------------------

    `max_iter=200` is a ceiling, not a target -- the measured runs above stopped
    at 30 to 115. What terminates them is sklearn's `n_iter_no_change` (10) and
    `tol` (1e-4), which watch the TRAINING loss. That is a different mechanism
    from `early_stopping`, and it is the reason this wrapper can leave early
    stopping off without simply burning the full budget every time: monitoring
    the training loss needs no holdout, so there is nothing to leak.

    --------------------------------------------------------------------------
    Why param_grid is a LIST
    --------------------------------------------------------------------------

    One sub-grid per architecture, with hidden_layer_sizes pinned inside each --
    the same idiom the boosters use for max_depth. It matters for the Bayesian
    path: skopt fits one surrogate per sub-grid, so this gives each architecture
    its own model of alpha and learning_rate_init rather than asking one surrogate
    to interpolate between shapes that have no interior. See search_spaces.to_dimension,
    which keeps a tuple-valued entry categorical for the same reason.
    """

    # Six orders of magnitude, and the ceiling is 1e1 on EVIDENCE, having been
    # raised to 1e2 and put back.
    #
    # The raise looked justified: every E07 regression refit selected 8.2 or 10.0
    # against a 1e1 ceiling, and run_forecast.hyperparams_table's own reasoning
    # says a value pinned to the edge means the GRID was binding. It was not. On
    # the SAME 514 periods:
    #
    #     ceiling   alpha the search chose      payout
    #     1e2       [39.2, 79.1, 12.2, 39.2]   0.075302   (e09_sklmlp_raw)
    #     pinned    [8.2,  8.2,  8.2,  8.2]    0.082438   (e13_raw)
    #
    # So the CV score keeps improving as alpha grows past ~10 while out-of-sample
    # payout falls -- a CV/OOS mismatch, not a binding constraint. Widening the
    # grid let the search walk into that region and cost roughly 0.007 payout
    # across E09, E11 and E12, every one of which landed at 0.0753-0.0766 where
    # E07 reached 0.0904.
    #
    # The lesson generalises: an edge-pinned hyperparameter means the grid bounds
    # the CV optimum. Whether that optimum is worth reaching is a separate
    # question, and on this panel it was not.
    ALPHAS = np.logspace(-4, 1, 60)
    LEARNING_RATES = np.logspace(-4, -2, 40)
    # Measured, not assumed -- see the docstring's table. Below 1024 the cost
    # roughly triples and the fit is no better.
    BATCH_SIZES = [1024, 2048, 4096, 8192]

    # (width,) ... (wide, narrower). Small on purpose -- 22 features on a mostly
    # noise target -- with the last two sized for the wide derived datasets.
    ARCHITECTURES = [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (256, 128),
    ]

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        # n_jobs is accepted and dropped. sklearn's MLP has no worker count --
        # it parallelises through whatever BLAS numpy is linked against -- but
        # build_wrapper() hands one to every non-linear wrapper, and taking it
        # here is cheaper than teaching that dispatch about a third category.
        #
        # The consequence is that --n-jobs does NOT bound this model: left alone
        # it takes every core on the box, which matters on a machine already
        # running other arms. OMP_NUM_THREADS / MKL_NUM_THREADS /
        # OPENBLAS_NUM_THREADS are the only knobs that do; experiments/
        # e07_mlp_reg_crypto.sh exports all three.
        del n_jobs

        self.model_name = "mlp"
        self.search_type = "random"

        common = {
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "early_stopping": False,        # see the docstring; do not turn on
            "random_state": seed,
        }

        if task == "classification":
            from sklearn.neural_network import MLPClassifier
            estimator = MLPClassifier(**common)
        else:
            from sklearn.neural_network import MLPRegressor
            estimator = MLPRegressor(**common)

        from sklearn.pipeline import Pipeline

        self.ModelClass = Pipeline(
            [("scale", ScaleContinuousOnly()), ("mlp", estimator)]
        )

        self.param_grid = [
            {"mlp__hidden_layer_sizes": [architecture],
             "mlp__alpha": self.ALPHAS,
             "mlp__learning_rate_init": self.LEARNING_RATES,
             "mlp__batch_size": self.BATCH_SIZES}
            for architecture in self.ARCHITECTURES
        ]


# --------------------------------------------------------------------------- #
# MLP -- the same architecture on torch, to separate framework from model
# --------------------------------------------------------------------------- #

class MLPKtzWrapper(MLPWrapper):
    """
    The MLP with a search grid set from EVIDENCE, not convention -- E17.

    E17 ran the Kalfa-Timmermann-van der Zwan hyperparameter-sensitivity
    framework on this exact estimator (a 160-cell factorial, then marginal
    distributions, functional-ANOVA importance, corners and priors). This wrapper
    is that study's recommendation compiled into a grid. Named for the paper
    ("ktz"). It subclasses MLPWrapper, so the estimator, the scaler
    (ScaleContinuousOnly -- these one-hot book datasets need it), the n_jobs
    handling and the classification path are all inherited; only the search grid
    changes.

    --------------------------------------------------------------------------
    What E17 found, and how each finding becomes a grid choice
    --------------------------------------------------------------------------

    LEARNING RATE is the one continuous knob worth tuning (median-payout range
    0.025 at a fixed architecture, the largest of any axis). So it gets a DENSE
    log grid and the search budget concentrates here. The range is bounded by two
    measured corners: 1e-4 never appeared in the good quartile (0 of 40 -- it
    underfits), and 1e-1 fails unless alpha is simultaneously high. The grid runs
    ~2e-4 to ~5e-2 -- fine resolution across the good region 1e-3..1e-2, stopping
    short of the 1e-1 danger zone, which is what makes the high-lr/low-alpha bad
    corner UNREACHABLE rather than merely discouraged.

    ARCHITECTURE is a shallow/deep GATE, not a sweep: three coarse depths, all
    within the 1-3 layers E17 found safe. Four- and five-layer ReLU nets collapse
    to a constant prediction (verified pred std ~5e-17), so they are excluded
    entirely -- not a value the search can waste a draw on.

    ALPHA is the least important axis (range 0.011, under half of lr) AND its CV
    optimum diverges from its out-of-sample optimum -- the exact mismatch that
    made widening its ceiling to 1e2 cost ~0.007 payout across three studies
    before it was reverted. So it is nearly pinned: two values, leaning high, and
    the search is told not to spend budget here.

    ACTIVATION: identity is the safe default (never collapses); relu is offered
    but only meaningful with the shallow gate above. Two values.

    Grid size 20 x 3 x 2 x 1 = 120 combinations. ACTIVATION is PINNED for a
    mechanical reason, not a stylistic one: RandomizedSearchCV samples the
    discrete grid uniformly (ParameterSampler, no replacement), so EVERY axis
    gets an even split over its own values -- giving lr 20 levels does NOT
    down-weight a two-value activation or alpha axis. A free activation would
    therefore eat ~half the draws regardless of the lr grid. Pinning activation
    (and near-pinning alpha to 2 values) is what actually concentrates the draw
    budget on the learning rate: at --n-iter 80 over 120 cells, each of the 20 lr
    levels gets ~4 draws instead of ~2. That is the "spend the budget on the
    learning rate" the E17 priors -- and the ask -- call for, made true by the
    grid's SHAPE rather than merely asserted.
    """

    # The tuning focus: dense across the good region, short of the 1e-1 corner.
    LEARNING_RATES = np.logspace(-3.7, -1.3, 20)   # ~2.0e-4 .. 5.0e-2
    # A shallow/deep gate, all <= 3 layers (E17: 4-5 layers collapse).
    ARCHITECTURES = [(32,), (32, 16), (32, 16, 8)]
    # Least important; near-pinned, leaning high.
    ALPHAS = [1e-2, 1e-1]
    # PINNED, not searched. E17 review caught the mechanism: RandomizedSearchCV's
    # ParameterSampler draws uniformly over the discrete grid, so a two-value
    # activation axis would take a fixed ~50% of the draws no matter how many lr
    # levels exist -- HALVING the learning-rate density in the branch that wins.
    # identity is E17's safe default (median 0.068 vs relu 0.034; the largest
    # fANOVA share, but as a GOOD corner -- identity's worst 0.044 beats relu's
    # median) and it never collapses. Pinning it hands the whole draw budget to
    # the learning rate, which is the axis worth tuning and the one the ask
    # names. Probing relu on the wide one-hot sets E17 never covered is a
    # separate question, not this budget's.
    ACTIVATION = "identity"
    BATCH = 1024

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        super().__init__(task=task, seed=seed, n_jobs=n_jobs)
        self.model_name = "mlp_ktz"
        # One flat grid -- unlike MLPWrapper's per-architecture sub-grids -- so a
        # random search samples lr, arch, alpha and activation INDEPENDENTLY and
        # the dense lr axis naturally dominates the variation. batch is fixed.
        self.param_grid = {
            "mlp__hidden_layer_sizes": self.ARCHITECTURES,
            "mlp__activation": [self.ACTIVATION],   # pinned; see ACTIVATION
            "mlp__learning_rate_init": self.LEARNING_RATES,
            "mlp__alpha": self.ALPHAS,
            "mlp__batch_size": [self.BATCH],
        }


class GridEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Fit EVERY cell of a base wrapper's grid and rank-average their predictions.

    This is the "ensemble instead of select" estimator. A RandomizedSearchCV
    over the mlp_ktz grid fits ~n_iter candidates on inner CV folds and KEEPS
    ONE -- the argmax of a noisy 26-effective-window CV score -- and throws the
    other 119 fits away. This estimator instead fits all 120 cells on the whole
    training block and combines their out-of-sample predictions, so no fit is
    discarded and no selection variance is incurred. It is the same move the
    deployed book makes -- CURUPIRA_SACI_*, CU_SA_IA_BO_01 are rank-average
    blends -- applied to a model's OWN hyperparameter grid rather than to
    distinct feature sets.

    --------------------------------------------------------------------------
    Why rank-average, and why at the block level
    --------------------------------------------------------------------------

    run_ensemble.py combines the book's members by ranking each to [0, 1]
    WITHIN EACH PERIOD and averaging (see its docstring): raw-averaging is wrong
    because members on different scales would let the widest-spread member
    dominate. The same hazard exists here -- a low-alpha cell has a wider output
    spread than a high-alpha one -- so members are rank-transformed before
    averaging, not raw-averaged.

    The one difference from run_ensemble is the ranking GRANULARITY. This
    estimator is called through predict_signal(best_estimator_, X_block) with a
    feature matrix and no period labels (functions.py routes the period into
    fit() for opt-in wrappers, but never into predict), so it ranks each member
    over the whole predicted block rather than per period. That is an
    APPROXIMATION of run_ensemble's per-period rank-average, exact only when a
    member has no systematic level drift across the block's dates; and since the
    final numerai_corr re-ranks the ensemble PER PERIOD at scoring time, any
    residual block-vs-period difference survives only through such drift.
    test_pipeline quantifies the gap on a synthetic panel and asserts it is
    negligible against per-period run_ensemble; if a future feature set breaks
    that, the members must be dumped and combined through run_ensemble instead.

    --------------------------------------------------------------------------
    Cost, and why it must run at a low --n-splits
    --------------------------------------------------------------------------

    The machinery has no "one candidate => skip CV" path: a selection-free
    estimator is still fit (usable folds) + 1 times per refit (inner-CV folds to
    score a candidate that is never chosen against anything, plus the refit).
    Every one of those fits trains all 120 members. purged_period_splits skips
    its first fold (nothing precedes period 0), so n_splits=1 yields ZERO usable
    folds and errors; n_splits=2 yields exactly one (1 CV fit + 1 refit = 2
    ensemble fits per refit, the floor). So this must run at --n-splits 2 --
    anything higher multiplies 120 * refits * datasets member-fits by dead CV
    folds. The CV score it produces is meaningless here (nothing is selected on
    it); it is paid only because RandomizedSearchCV always scores before it refits.

    --------------------------------------------------------------------------
    sklearn contract
    --------------------------------------------------------------------------

    __init__ stores its arguments unchanged (task, seed, n_jobs) so clone() --
    which functions.py applies before every fit -- reconstructs it exactly; the
    grid and the base estimator are rebuilt from MLPKtzWrapper at fit time, not
    stored, so there is no mutable state to deep-copy. Fitted members live on
    self.members_ (trailing underscore), invisible to get_params. n_jobs is
    accepted and dropped for the same reason MLPWrapper drops it -- sklearn's MLP
    has no worker count and threads through BLAS; the members are fit in a plain
    loop.
    """

    # Regression only. The book pays on the 20-day regression label and E28 is
    # a regression study; a classification ensemble is a separate, untested
    # question and is not offered rather than shipped half-checked.
    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1, base="mlp_ktz"):
        self.task = task
        self.seed = seed
        self.n_jobs = n_jobs
        # Which wrapper's grid to ensemble. Default mlp_ktz (E31); mlp_ktz2 is the
        # E33-reprior grid (E35). Stored unchanged so clone() reconstructs it.
        self.base = base

    def _cells(self):
        """The base wrapper's grid, expanded into one param dict per cell."""
        from sklearn.model_selection import ParameterGrid

        base = build_wrapper(self.base, task="regression", seed=self.seed)
        return base.ModelClass, list(ParameterGrid(base.param_grid))

    def fit(self, X, y):
        from sklearn.base import clone

        if self.task != "regression":
            raise ValueError(
                "GridEnsembleRegressor is regression-only; got task=%r"
                % (self.task,))

        base_estimator, cells = self._cells()
        self.n_members_ = len(cells)
        self.members_ = []
        for params in cells:
            member = clone(base_estimator).set_params(**params)
            member.fit(X, y)
            self.members_.append(member)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        import warnings
        from scipy.stats import rankdata

        if not getattr(self, "members_", None):
            raise RuntimeError("GridEnsembleRegressor.predict before fit")

        n = np.asarray(X).shape[0]
        if n == 0:
            return np.zeros(0, dtype="float64")

        # Rank each member to (0, 1] over the block, then average. Ranking first
        # is the scale normalisation run_ensemble uses; averaging ranks is a
        # no-op with respect to the per-period re-ranking numerai_corr applies.
        #
        # A DIVERGED member is dropped, not propagated. An adam MLP at the top of
        # the lr grid can return a non-finite prediction on a block without
        # raising; scipy.rankdata would then turn that member's whole rank vector
        # to NaN, the average to NaN, and summarize() would silently dropna the
        # entire block from the scored record. run_ensemble isolates a NaN to its
        # own row; a diverged MLP is bad on every row, so the honest move is to
        # exclude that member and average the survivors. The identity/high-alpha/
        # capped-lr grid makes this rare, but silent is the failure mode this repo
        # exists to prevent, so it is guarded rather than assumed away.
        total = np.zeros(n, dtype="float64")
        used = 0
        for member in self.members_:
            preds = np.asarray(member.predict(X), dtype="float64")
            if not np.isfinite(preds).all():
                continue
            total += rankdata(preds, method="average") / n
            used += 1
        if used == 0:
            # Every member diverged -- essentially impossible on this grid, but a
            # loud constant is a visible zero-corr score, not a silent NaN drop.
            warnings.warn(
                "GridEnsembleRegressor: all %d members returned non-finite "
                "predictions on a block; emitting a constant" % len(self.members_))
            return np.full(n, 0.5, dtype="float64")
        if used < len(self.members_):
            warnings.warn(
                "GridEnsembleRegressor: dropped %d of %d members with non-finite "
                "predictions on a block" % (len(self.members_) - used, len(self.members_)))
        return total / used

    def predict_members(self, X):
        """
        Each member's RAW prediction, as an (n_rows, n_members) array.

        The ensemble discards these -- predict() returns only the rank-average --
        but they are what the arm-board skill's ensemble-vs-best-member question
        needs, and what a per-period run_ensemble cross-check of the combiner
        needs. Kept here (rather than reconstructed by re-fitting) so the members
        scored are EXACTLY the ones a walk-forward block averaged. Column j is
        cells()[j]; order is ParameterGrid's, stable across a fixed grid.
        """
        if not getattr(self, "members_", None):
            raise RuntimeError("GridEnsembleRegressor.predict_members before fit")
        return np.column_stack(
            [np.asarray(m.predict(X), dtype="float64") for m in self.members_])


class MLPKtzEnsembleWrapper:
    """
    The mlp_ktz grid as an ENSEMBLE, not a search (E31).

    Same 120 cells MLPKtzWrapper's search ranges over -- the E17 priors
    (dense lr, shallow/deep architecture gate, near-pinned alpha, identity
    activation) -- but every cell is fitted and their predictions are
    rank-averaged, rather than one being selected by inner CV. See
    GridEnsembleRegressor for the combination rule and the cost note.

    The wrapper exposes the standard four attributes with a DEGENERATE search:
    an empty param_grid means RandomizedSearchCV has exactly one candidate (the
    ensemble as constructed) and nothing to tune -- run it with --n-iter 1
    --n-splits 2 (2 is the CV floor; n_splits=1 skips its only fold). There is
    deliberately no fit_grid and no searchable dimension; the point of the study
    is to remove selection, not relocate it.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        if task != "regression":
            raise ValueError(
                "mlp_ktz_ens is regression-only; got task=%r" % (task,))
        self.model_name = "mlp_ktz_ens"
        self.search_type = "random"
        self.ModelClass = GridEnsembleRegressor(
            task=task, seed=seed, n_jobs=n_jobs)
        # Empty grid: one candidate, nothing to sample. The ensemble is the
        # model; the search machinery only refits it on each block.
        self.param_grid = {}


class MLPKtz2Wrapper(MLPWrapper):
    """
    The MLP grid rebuilt from the E33 sweep -- the priors that actually transfer.

    E17 (-> mlp_ktz) swept alpha only to 0.1 and froze the first layer at 32; its
    priors then failed on the richer feature sets, because the best region (where
    IA_01 lives) was OUTSIDE that box. E33 re-swept with alpha opened to 100 and
    width unfrozen, on raw / fenccausal / tenc_onehot, and the three datasets
    AGREED on the same priors. This wrapper is that agreement compiled into a
    grid.

    What E33 found, and how each becomes a grid choice
    --------------------------------------------------------------------------
    ALPHA is the axis E17 got wrong. alpha 10 is decisively best on every dataset
    (fenccausal median 0.087 at alpha 10 vs ~0.05 at alpha <=0.1); the peak is
    ~1-10 and alpha 100 falls off. So alpha is {1, 10} -- roughly 100x E17's
    ceiling, and the single change that matters.

    ACTIVATION is un-pinned to {identity, relu}. identity has the higher MEDIAN
    (robust), but the top cells on every dataset are relu+alpha10 -- alpha 10
    suppresses the high-lr relu collapse that dragged relu's median down in E17.
    A search should see both; an ensemble should hold both.

    WIDTH barely moved the median across 32/64/128/256, so it is NOT the strong
    axis E17's frozen-32 made it look -- IA_01's width was incidental to its
    alpha. Kept small: {32, 64, 128}. DEPTH is shallow (1 > 2 > 3): {1, 2}. LR is
    low, peak 1e-3: a 5-point log sweep 1e-4..1e-2.

    Grid size 6 arch x 2 act x 5 lr x 2 alpha = 120 -- the same footprint as
    mlp_ktz, so the E34 search (draws from it) and the E35 ensemble (fits all of
    it) cost what E28/E31 did, and ensemble-of-120 vs search-of-120 is the same
    clean contrast. Subclasses MLPWrapper, so the scaler, n_jobs handling and
    classification path are inherited; only the grid changes.
    """

    # width {32,64,128} x depth {1,2}, halving. Width is a weak axis (E33); depth
    # shallow. First layer stays modest -- the alpha is what does the work.
    ARCHITECTURES = [(32,), (32, 16), (64,), (64, 32), (128,), (128, 64)]
    ACTIVATIONS = ["identity", "relu"]        # E33: identity robust, relu peaks
    LEARNING_RATES = np.logspace(-4, -2, 5)   # 1e-4..1e-2, peak 1e-3
    ALPHAS = [1.0, 10.0]                       # E33 peak ~10; ~100x E17's ceiling
    BATCH = 1024

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        super().__init__(task=task, seed=seed, n_jobs=n_jobs)
        self.model_name = "mlp_ktz2"
        self.param_grid = {
            "mlp__hidden_layer_sizes": self.ARCHITECTURES,
            "mlp__activation": self.ACTIVATIONS,
            "mlp__learning_rate_init": self.LEARNING_RATES,
            "mlp__alpha": self.ALPHAS,
            "mlp__batch_size": [self.BATCH],
        }


class MLPKtz2EnsembleWrapper:
    """
    The mlp_ktz2 (E33-reprior) grid as an ENSEMBLE, not a search (E35).

    Identical in construction to MLPKtzEnsembleWrapper (see GridEnsembleRegressor
    for the rank-average combiner, the collapse-drop, and the --n-splits 2 cost
    note) but points the ensemble at the mlp_ktz2 grid via base="mlp_ktz2". Its
    120 members carry alpha {1,10} and both activations -- the region E31's
    ensemble could not reach because its grid stopped at alpha 0.1 and identity.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1):
        if task != "regression":
            raise ValueError(
                "mlp_ktz2_ens is regression-only; got task=%r" % (task,))
        self.model_name = "mlp_ktz2_ens"
        self.search_type = "random"
        self.ModelClass = GridEnsembleRegressor(
            task=task, seed=seed, n_jobs=n_jobs, base="mlp_ktz2")
        self.param_grid = {}


class LossColumnScaler(BaseEstimator, TransformerMixin):
    """
    Scale the feature block; pass the last `n_passthrough` columns through unscaled.

    torch_mlp_loss appends a period code and (for the bmc family) a reference
    series as TRAILING columns the loss needs but the network must not see -- and
    that must NOT be scaled (a scaled integer period code stops being a group id;
    the reference is peeled off before the net regardless). This scales the
    leading feature block with ScaleContinuousOnly (so one-hot dummies still pass
    through correctly) and re-appends the trailing loss columns untouched; the
    estimator's `_split_ids` then peels them. With n_passthrough=0 it is exactly
    ScaleContinuousOnly, so the mse control arm is unaffected.
    """

    def __init__(self, n_passthrough=0):
        self.n_passthrough = n_passthrough

    def fit(self, X, y=None):
        values = np.asarray(X, dtype="float64")
        k = int(self.n_passthrough)
        self.n_features_in_ = values.shape[1]
        self.scaler_ = ScaleContinuousOnly().fit(
            values[:, :values.shape[1] - k] if k else values)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype="float64")
        k = int(self.n_passthrough)
        if not k:
            return self.scaler_.transform(values)
        return np.hstack([self.scaler_.transform(values[:, :-k]), values[:, -k:]])


class TorchMLPWrapper:
    """
    MLPWrapper's architecture, trained by prediction/torch_mlp.py instead.

    The point is a controlled comparison. MLPWrapper's numbers are the sklearn
    MLP's training loop as much as they are "a feedforward net on this panel",
    and the two cannot be separated without an implementation that differs in the
    framework and in nothing else. torch_mlp.TorchMLPRegressor reproduces every
    documented detail of sklearn's loop -- the half-MSE loss, the L2 term scaled
    by batch size and applied to weights only, Glorot-uniform init over
    fan_in + fan_out, Adam's exact betas, index shuffling with a short final
    batch, and the stop-on-training-loss rule including its off-by-one.

    Measured agreement at matched hyperparameters, five seeds, on a real
    130-period block:

        impl      epochs   final loss   OOS spearman      pred std
        sklearn     28.6     0.025217   0.11541 +- 0.00797  0.00789
        torch       26.0     0.025262   0.11638 +- 0.00928  0.01022

    Final losses agree to four decimals, and the spearman gap is an eighth of the
    seed-to-seed spread. Treat the two as the same model; a divergence at RUN
    level would therefore be about the walk-forward, not the estimator.

    THE GRID IS NOT COPIED, IT IS REFERENCED. Every constant comes from
    MLPWrapper, so the two can never drift apart and a comparison between them can
    never be a comparison of two different search spaces. Change MLPWrapper's
    ranges and this one follows.

    It is roughly 2x sklearn's wall clock per fit on CPU (5.0s against 2.4s on a
    39,000-row block), so budget about double. `device="cuda"` is available and
    untested on this panel -- the walk-forward runs many small fits, and host-to-
    device transfers can cost more than the kernels save at this size.
    """

    # Declared, not discovered by exception: supported_tasks() reads this, so a
    # caller can ask what a wrapper does without constructing one and catching.
    TASKS = ("regression",)

    ALPHAS = MLPWrapper.ALPHAS
    LEARNING_RATES = MLPWrapper.LEARNING_RATES
    BATCH_SIZES = MLPWrapper.BATCH_SIZES
    ARCHITECTURES = MLPWrapper.ARCHITECTURES

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu"):
        # Dropped for the same reason MLPWrapper drops it: there is no worker
        # count here either. torch threading is set by torch.set_num_threads or
        # OMP_NUM_THREADS, which the runners export.
        del n_jobs

        if task != "regression":
            raise ValueError(
                "torch_mlp implements regression only; the sklearn `mlp` wrapper "
                "covers classification. Adding it here would mean a second head "
                "and a second loss, and no study has asked for one."
            )

        self.model_name = "torch_mlp"
        self.search_type = "random"

        # Imported here, not at module scope: torch is OPTIONAL (see setup.sh) and
        # this module is imported by run_forecast and by src/live/refit.py, which
        # must both work on a box where the wheel never installed. Same lazy
        # pattern LGBWrapper uses for lightgbm.
        from src.prediction.torch_mlp import TorchMLPRegressor
        from sklearn.pipeline import Pipeline

        estimator = TorchMLPRegressor(
            activation="relu", max_iter=200, random_state=seed, device=device,
        )
        self.ModelClass = Pipeline(
            [("scale", ScaleContinuousOnly()), ("mlp", estimator)]
        )
        self.param_grid = [
            {"mlp__hidden_layer_sizes": [architecture],
             "mlp__alpha": self.ALPHAS,
             "mlp__learning_rate_init": self.LEARNING_RATES,
             "mlp__batch_size": self.BATCH_SIZES}
            for architecture in self.ARCHITECTURES
        ]


class TorchMLPEmbWrapper(TorchMLPWrapper):
    """
    TorchMLPWrapper plus a learned ticker embedding, and nothing else changed.

    The one thing that varies against `torch_mlp` is whether the network can see
    which symbol a row belongs to. Everything else -- architecture set, alpha,
    learning rate, batch sizes, optimiser, stopping rule, seed -- is inherited,
    so a difference between the two arms is attributable to the embedding.

    Needs `--data-version crypto/v2.0-symid`, which is the raw 22 plus one
    integer `feature_symbol_id` column (src/data_mani/build_symbol_id.py). The
    id must reach the estimator UNSCALED, so the Pipeline's first step is
    IdPassthroughScaler rather than StandardScaler; with no id column present the
    two are the same transformation, which is what keeps the no-embedding arm
    comparable.

    The table is built the way prediction/nn/model.py builds the LSTM's -- one
    row per training symbol plus a zeroed padding row for codes the block never
    saw. See TorchMLPRegressor's docstring for the full correspondence.

    embed_dim is searched rather than pinned, because there is no evidence for a
    value here. The LSTM's 50 is inherited from iwtsf and is far too wide beside
    22 features; the grid spans 4 to 32 so the search can say. Note what an
    embedding costs: 1,316 symbols at embed_dim 32 is 42,144 parameters against
    roughly 1,500 in a (64, 32) stack over 22 features, so nearly all of the
    model's capacity becomes ticker identity. alpha covers the table as well as
    the dense weights, deliberately -- see TorchMLPRegressor.fit.
    """

    EMBED_DIMS = [4, 8, 16, 32]
    ID_COLUMN = "feature_symbol_id"

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu",
                 embed_dim=None):
        """
        :param embed_dim: PIN the table width instead of searching it. None
                          searches EMBED_DIMS, which is what E11 does; an int
                          fixes it and lets the search cover everything else,
                          which is what a per-width arm needs. Any positive int
                          is accepted, including widths outside EMBED_DIMS --
                          the grid is where the SEARCH looks, not a whitelist.
        :type embed_dim: int or None
        """
        super().__init__(task=task, seed=seed, n_jobs=n_jobs, device=device)
        self.model_name = "torch_mlp_emb"

        if embed_dim is not None:
            embed_dim = int(embed_dim)
            if embed_dim < 1:
                raise ValueError(
                    "embed_dim must be at least 1, got {}; use --model "
                    "torch_mlp for no embedding at all, which is a different "
                    "network rather than a zero-width one".format(embed_dim))
        self.embed_dim = embed_dim

        from src.prediction.torch_mlp import IdPassthroughScaler, TorchMLPRegressor
        from sklearn.pipeline import Pipeline

        estimator = TorchMLPRegressor(
            activation="relu", max_iter=200, random_state=seed, device=device,
            n_embedding_columns=1,
        )
        self.ModelClass = Pipeline([
            ("scale", IdPassthroughScaler(id_columns=(self.ID_COLUMN,))),
            ("mlp", estimator),
        ])
        widths = self.EMBED_DIMS if self.embed_dim is None else [self.embed_dim]
        self.param_grid = [
            dict(grid, **{"mlp__embed_dim": widths}) for grid in self.param_grid
        ]


class TorchMLPLossWrapper(TorchMLPWrapper):
    """
    torch_mlp with a configurable training LOSS (E42 pinned / E43 searched).

    The plain torch_mlp trains on half-MSE; this carries loss in {mse (=control),
    corr, bmc, sharpe, calmar} via --model-options, e.g.
    `--model-options '{"loss": "corr"}'`. The loss is baked into the estimator
    instance, so --fixed-params can pin architecture/lr/alpha/batch and leave the
    loss untouched (E42, isolating the loss's effect on learning), or the grid can
    be searched per loss (E43).

    Plumbing (all gated on this wrapper, so no other model is affected): losses
    that score per period need a period code, and the bmc family needs a reference
    series (the real meta model under --bmc-reference benchmark, or the full-
    coverage meta_proxy). Both are carried as TRAILING passthrough columns like
    the ticker id -- appended by the walk-forward for any wrapper declaring
    FIT_APPEND, scaled-around by LossColumnScaler, and peeled by the estimator's
    _split_ids -- never fed to the network. The mmc-sparse vs bmc-proxy contrast
    is this same loss=bmc arm run under the two --bmc-reference values.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu",
                 loss="mse", bmc_reference=None, **loss_kwargs):
        del n_jobs
        if task != "regression":
            raise ValueError("torch_mlp_loss implements regression only")

        from src.prediction.torch_mlp_losses import loss_meta
        from src.prediction.torch_mlp import TorchMLPRegressor
        from sklearn.pipeline import Pipeline

        meta = loss_meta(loss)               # validates the loss name up front
        self.model_name = "torch_mlp_loss"
        self.search_type = "random"
        self.loss = loss
        self.bmc_reference = bmc_reference    # informational; --bmc-reference drives it
        n_period = 1 if meta["needs_period"] else 0
        n_reference = 1 if meta["needs_reference"] else 0

        estimator = TorchMLPRegressor(
            activation="relu", max_iter=200, random_state=seed, device=device,
            loss=loss, loss_kwargs=(dict(loss_kwargs) or None),
            n_period_columns=n_period, n_reference_columns=n_reference,
        )
        self.ModelClass = Pipeline([
            ("scale", LossColumnScaler(n_passthrough=n_period + n_reference)),
            ("mlp", estimator),
        ])
        self.param_grid = [
            {"mlp__hidden_layer_sizes": [architecture],
             "mlp__alpha": self.ALPHAS,
             "mlp__learning_rate_init": self.LEARNING_RATES,
             "mlp__batch_size": self.BATCH_SIZES}
            for architecture in self.ARCHITECTURES
        ]
        # Which df columns the walk-forward appends as trailing passthrough, in the
        # order the estimator peels them: [period, reference].
        append = []
        if n_period:
            append.append("period")
        if n_reference:
            append.append("reference")
        self.FIT_APPEND = tuple(append)


# --------------------------------------------------------------------------- #
# TabNet -- the one deep-tabular architecture that is sklearn-complete
# --------------------------------------------------------------------------- #

class TabNetWrapper:
    """
    TabNet (dreamquark-ai `pytorch-tabnet`), wrapped like ridge and the boosters.

    A ROW-WISE estimator -- one row in, one score out -- so it belongs in this
    module rather than prediction/nn/, and because pytorch-tabnet's fitted model
    survives joblib.dump/load identically, a `tabnet` arm is deployable to a live
    slot through the same build_wrapper + joblib.dump(best_estimator_) path
    src/live/refit.py uses for every other MODELS entry. The estimator itself and
    the four sklearn-contract adaptations live in src/prediction/tabnet.py.

    --------------------------------------------------------------------------
    Why a Pipeline, and why torch is imported here
    --------------------------------------------------------------------------

    StandardScaler in front, for the same reason MLPWrapper has one: a net is not
    scale-invariant, and the scaler is fit INSIDE each CV fold so no validation
    statistic leaks into training. On crypto's 5-binned features it does little;
    it matters on the 763-812 column derived sets. The tabnet import is lazy,
    inside __init__, because torch is OPTIONAL (setup.sh) and this module is
    imported by run_forecast and src/live/refit.py -- both must import on a box
    with no torch. Same lazy pattern as TorchMLPWrapper.

    --------------------------------------------------------------------------
    Regression only, and how "default parameters" is expressed
    --------------------------------------------------------------------------

    TASKS is regression only: the task needs no classifier head, and adding one
    would mean a second estimator (TabNetClassifier) and a second predict path
    for no study that has asked for it. The estimator's constructor defaults ARE
    pytorch-tabnet's library defaults, so a run that pins nothing -- or pins this
    grid's values, which equal them -- reproduces the paper's configuration. That
    is what the tabnet study does: --fixed-params collapses the grid to one
    candidate (see hyper_params_search), so the walk-forward fits the default
    network and searches nothing.

    --------------------------------------------------------------------------
    The grid is for a LATER search, and is reasoned, not measured
    --------------------------------------------------------------------------

    This panel is small and noisy for a model this expressive (see tabnet.py), so
    the default-parameter arm is a viability probe and this grid is only exercised
    if that probe justifies tuning. It encodes the two Numerai idioms that carry
    over from the module header -- heavy regularisation and a low learning rate --
    and refuses to over-search on ~26 effective independent windows:

      * lambda_sparse spans 1e-4..1e-1 log-spaced. It is the mask-sparsity
        penalty, the one knob doing the job MLPWrapper's alpha does, and the
        useful region on this target sits toward the strong-penalty end.
      * learning_rate spans 1e-3..2e-2 log-spaced, against a fixed epoch budget.
      * n_a is TIED to n_d (the paper ties them); one sub-grid per (n_d, n_steps)
        capacity shape, the same list-of-dicts idiom the boosters use for
        max_depth, so a surrogate never interpolates between shapes.

    Architectures are small: on 22 features a wide TabNet is all capacity and no
    signal. There is no column-subsampling analogue -- a dense TabNet step sees
    every column -- but its learned attention masks ARE a per-step feature
    selection, so that idiom is expressed by the architecture rather than a knob.
    """

    TASKS = ("regression",)

    # Library defaults, named so the study's pinned config is self-documenting.
    LAMBDA_SPARSE = np.logspace(-4, -1, 40)              # 1e-4 .. 1e-1
    LEARNING_RATES = np.logspace(-3, np.log10(2e-2), 30)  # 1e-3 .. 2e-2
    # One (n_d == n_a, n_steps) capacity shape per sub-grid; small on purpose.
    SHAPES = [(8, 3), (8, 4), (16, 3), (16, 4), (24, 5)]

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu"):
        # No worker count: torch threads through OMP_NUM_THREADS, which the
        # runners export. Dropped like MLPWrapper/TorchMLPWrapper drop it.
        del n_jobs

        if task != "regression":
            raise ValueError(
                "tabnet implements regression only; no study needs a TabNet "
                "classifier head, and adding one would mean a second estimator "
                "and a second predict path."
            )

        self.model_name = "tabnet"
        self.search_type = "random"

        # Lazy: torch is OPTIONAL. See the class docstring.
        from src.prediction.tabnet import TabNetSkRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = TabNetSkRegressor(seed=seed, device=device)
        self.ModelClass = Pipeline(
            [("scale", StandardScaler()), ("net", estimator)]
        )
        self.param_grid = [
            {"net__n_d": [n_d], "net__n_a": [n_d], "net__n_steps": [n_steps],
             "net__lambda_sparse": self.LAMBDA_SPARSE,
             "net__learning_rate": self.LEARNING_RATES}
            for (n_d, n_steps) in self.SHAPES
        ]


class TabNetKtzWrapper(TabNetWrapper):
    """
    TabNet with a WIDE, prior-shaped exploration grid (study E153).

    E18 ran TabNet pinned at library defaults and it lost to the deployed slot on
    every dataset (payout delta -0.015..-0.043, t -2.0..-5.3). Its write-up named
    the only remaining lever as tuning. This wrapper is that tuning, built in
    the KTZ style used for lgb_ktz/catboost_ktz/rf_ktz/mlp_ktz: every axis the
    estimator exposes is opened, and the priors shape the RANGES and the grid's
    STRUCTURE instead of pinning values.

    Priors carried over:

      * LEARNING RATE is the axis that mattered most for every net here (E17,
        E105 mlp_ktz): a dense log grid, 3e-4 .. 5e-2. The library default 0.02
        sits inside it, so E18's configuration stays reachable.
      * REGULARISATION. lambda_sparse spans 1e-6 .. 1e-1. Below 1e-4 the masks
        are effectively unpenalised, and that is opened deliberately, since
        the 22 shipped features are already selected and binned (E18's
        mechanism). Momentum (BatchNorm) runs 0.005 .. 0.3; that axis is
        known to move TabNet a lot on small data.
      * CAPACITY. n_a is TIED to n_d (the paper ties them). One sub-grid per
        width, so the random search samples a width and then everything else
        independently, the list-of-dicts idiom the boosters use. Widths go up
        to 64 and n_steps up to 7. That is wider than TabNetWrapper.SHAPES,
        because a search that never sees large nets cannot rule them out.
      * TRAINING LENGTH. With no eval_set (see tabnet.py: leak-free), max_epochs
        is exact rather than a cap, so it is the early-stopping axis and is
        searched: 30 .. 150.
      * BATCH. Ghost batch norm needs virtual_batch_size <= batch_size, and
        every pair here satisfies that (largest virtual 512 <= smallest batch
        1024). Larger batches are also cheaper per epoch.

    Grid per width: 20 lr x 11 lambda x 5 steps x 4 gamma x 3 x 3 GLU x 5
    momentum x 2 mask x 3 batch x 3 virtual x 4 epochs = 28.5M cells, and with
    4 widths about 114M. RandomizedSearchCV samples it without enumerating it,
    so the size costs nothing. --n-iter alone sets the budget.
    """

    LEARNING_RATES = np.logspace(np.log10(3e-4), np.log10(5e-2), 20)
    LAMBDA_SPARSE = np.logspace(-6, -1, 11)
    WIDTHS = [8, 16, 32, 64]                 # n_d == n_a
    N_STEPS = [2, 3, 4, 5, 7]
    GAMMAS = [1.0, 1.3, 1.5, 2.0]
    N_INDEPENDENT = [1, 2, 3]
    N_SHARED = [1, 2, 3]
    MOMENTUM = [0.005, 0.02, 0.05, 0.1, 0.3]
    MASK_TYPES = ["sparsemax", "entmax"]
    BATCH_SIZES = [1024, 2048, 4096]
    VIRTUAL_BATCH_SIZES = [128, 256, 512]
    MAX_EPOCHS = [30, 60, 100, 150]

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu"):
        super().__init__(task=task, seed=seed, n_jobs=n_jobs, device=device)
        self.model_name = "tabnet_ktz"
        shared = {
            "net__n_steps": self.N_STEPS,
            "net__gamma": self.GAMMAS,
            "net__n_independent": self.N_INDEPENDENT,
            "net__n_shared": self.N_SHARED,
            "net__lambda_sparse": self.LAMBDA_SPARSE,
            "net__momentum": self.MOMENTUM,
            "net__mask_type": self.MASK_TYPES,
            "net__learning_rate": self.LEARNING_RATES,
            "net__batch_size": self.BATCH_SIZES,
            "net__virtual_batch_size": self.VIRTUAL_BATCH_SIZES,
            "net__max_epochs": self.MAX_EPOCHS,
        }
        self.param_grid = [
            dict(shared, **{"net__n_d": [w], "net__n_a": [w]})
            for w in self.WIDTHS
        ]


class TabNetKtzTVWrapper(TabNetWrapper):
    """
    tabnet_ktz with the epoch count TIED to model size: the Timmermann-Vulicevic
    schedule (src/prediction/tabnet_tv.py) and no max_epochs axis.

    Each candidate trains for E = floor(0.2 * P**0.8) + 10 epochs, where P is the
    trainable parameter count of the network ITS sampled settings build on THIS
    dataset's column count. The pairing is guaranteed by construction rather
    than by grid design.

    The space is capped where TV becomes unaffordable inside a search. Width
    stays at 8-16, steps at 2-5, and GLU depth at 1-2 per kind. That bounds TV at
    roughly 150-840 epochs on raw (22 columns) and 180-910 on fenccausal (44),
    against ~1,400-11,700 for tabnet_ktz's width 32-64 and 7-step corners.
    Every other axis keeps tabnet_ktz's range. epoch_mult and epoch_cap are
    pinned at 1 / None, so each fit runs exactly its TV count. Unpin
    epoch_mult (e.g. 0.5, 1, 2) to test budgets around the schedule.
    """

    WIDTHS = [8, 16]
    N_STEPS = [2, 3, 4, 5]
    GAMMAS = TabNetKtzWrapper.GAMMAS
    N_INDEPENDENT = [1, 2]
    N_SHARED = [1, 2]
    LAMBDA_SPARSE = TabNetKtzWrapper.LAMBDA_SPARSE
    MOMENTUM = TabNetKtzWrapper.MOMENTUM
    MASK_TYPES = TabNetKtzWrapper.MASK_TYPES
    LEARNING_RATES = TabNetKtzWrapper.LEARNING_RATES
    BATCH_SIZES = TabNetKtzWrapper.BATCH_SIZES
    VIRTUAL_BATCH_SIZES = TabNetKtzWrapper.VIRTUAL_BATCH_SIZES
    EPOCH_MULT = [1.0]

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu"):
        super().__init__(task=task, seed=seed, n_jobs=n_jobs, device=device)
        self.model_name = "tabnet_ktz_tv"
        from src.prediction.tabnet_tv import TabNetTVRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.ModelClass = Pipeline(
            [("scale", StandardScaler()),
             ("net", TabNetTVRegressor(seed=seed, device=device))])
        shared = {
            "net__n_steps": self.N_STEPS,
            "net__gamma": self.GAMMAS,
            "net__n_independent": self.N_INDEPENDENT,
            "net__n_shared": self.N_SHARED,
            "net__lambda_sparse": self.LAMBDA_SPARSE,
            "net__momentum": self.MOMENTUM,
            "net__mask_type": self.MASK_TYPES,
            "net__learning_rate": self.LEARNING_RATES,
            "net__batch_size": self.BATCH_SIZES,
            "net__virtual_batch_size": self.VIRTUAL_BATCH_SIZES,
            "net__epoch_rule": ["tv"],
            "net__epoch_mult": self.EPOCH_MULT,
        }
        self.param_grid = [
            dict(shared, **{"net__n_d": [w], "net__n_a": [w]})
            for w in self.WIDTHS
        ]


# --------------------------------------------------------------------------- #
# TabPFN -- a tabular FOUNDATION model, no per-dataset training at all
# --------------------------------------------------------------------------- #

class TabPFNWrapper:
    """
    TabPFN (Prior-Labs) wrapped like the other estimators. See tabpfn_fm.py.

    The point of difference from every other MODELS entry: there is no training
    and nothing to search. TabPFN is pre-trained; "fit" loads the block as
    context and "predict" is one forward pass. So the grid is a SINGLE candidate
    -- the default configuration -- and a study runs it as-is (or pins it with
    --fixed-params, which collapses to the same one candidate). It is the
    cheapest arm to ADD after TabNet, and the most expensive to RUN on CPU: the
    forward pass attends every query row to the whole context, so it wants a GPU
    (device="cuda"). See tabpfn_fm.py for the context-subsampling and offline
    checkpoint handling.

    No StandardScaler in front, unlike MLP/TabNet: TabPFN does its own input
    normalisation as part of the foundation model, so an outer scaler is
    redundant and would only fight its categorical handling. The estimator is
    therefore the ModelClass directly, and grid keys are unprefixed.
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu",
                 n_estimators=4, max_context=4096, softmax_temperature=0.9,
                 memory_saving_mode="auto", model_path=None):
        del n_jobs                                       # torch threads via OMP

        if task != "regression":
            raise ValueError(
                "tabpfn uses the v2 REGRESSOR checkpoint; classification would "
                "need the separate classifier weights and a predict_proba path, "
                "which no study has asked for."
            )

        self.model_name = "tabpfn"
        self.search_type = "random"

        # Lazy: torch/tabpfn are OPTIONAL. Same pattern as TabNetWrapper.
        from src.prediction.tabpfn_fm import (
            TabPFNSkRegressor, DEFAULT_MODEL_PATH)

        estimator = TabPFNSkRegressor(
            model_path=(model_path or DEFAULT_MODEL_PATH),
            n_estimators=n_estimators, max_context=max_context,
            softmax_temperature=softmax_temperature,
            memory_saving_mode=memory_saving_mode,
            device=device, random_state=seed,
        )
        self.ModelClass = estimator
        # One candidate: there is nothing to SEARCH in a foundation model. The
        # keys that move cost or the inference temperature -- n_estimators,
        # max_context, softmax_temperature -- are set via --model-options above
        # (a --fixed-params pin then collapses the grid to exactly this one).
        # n_estimators may be the string "auto", so it is not cast here.
        self.param_grid = [{"max_context": [int(max_context)]}]


class TabPFNClfWrapper:
    """
    TabPFN's classifier head on the 5-binned target, exposed as a regressor.

    Same foundation model as TabPFNWrapper, different head: it fits
    TabPFNClassifier on the target's five ordered bins and returns E[y|x] as the
    continuous signal (see TabPFNSkClassifierRegressor). It is a distinct MODELS
    entry rather than a task on `tabpfn` because it needs the separate CLASSIFIER
    checkpoint and a different estimator class -- the arm this exists to measure
    is exactly "classifier head vs regressor head, same data, same cadence".
    """

    TASKS = ("regression",)

    def __init__(self, task="regression", seed=None, n_jobs=-1, device="cpu",
                 n_estimators=4, max_context=4096, softmax_temperature=0.9,
                 memory_saving_mode="auto", model_path=None):
        del n_jobs

        if task != "regression":
            raise ValueError(
                "tabpfn_clf returns a continuous E[y|x] signal and runs as a "
                "regression arm; there is no separate classification path."
            )

        self.model_name = "tabpfn_clf"
        self.search_type = "random"

        from src.prediction.tabpfn_fm import (
            TabPFNSkClassifierRegressor, DEFAULT_CLF_MODEL_PATH)

        estimator = TabPFNSkClassifierRegressor(
            model_path=(model_path or DEFAULT_CLF_MODEL_PATH),
            n_estimators=n_estimators, max_context=max_context,
            softmax_temperature=softmax_temperature,
            memory_saving_mode=memory_saving_mode,
            device=device, random_state=seed,
        )
        self.ModelClass = estimator
        self.param_grid = [{"max_context": [int(max_context)]}]


# --------------------------------------------------------------------------- #
# Gaussian Process -- the one kernel model, and the one that carries a posterior
# --------------------------------------------------------------------------- #

class GPWrapper:
    """
    Subsampled exact Gaussian Process, regression and classification.

    The estimators live in src.prediction.gp; see that module for why the
    training rows are subsampled (a GP is O(n^3) and a pooled block is 20k-500k
    rows), why the kernel is chosen by a STRING rather than a kernel object (a
    kernel object is a silent --search-type bayes trap), and why it is frozen by
    default (so the OUTER search sets and RECORDS the scale, and a live slot can
    pin it with --fixed-params, exactly like ridge's alpha).

    --------------------------------------------------------------------------
    The search grid
    --------------------------------------------------------------------------

    Regression tunes three scalars/strings: the kernel FAMILY (Matern-3/2,
    Matern-5/2, RBF -- roughest to smoothest), the LENGTH-SCALE (how fast the
    return surface varies; log-spaced ~0.05..5 on [0,1]-scaled features), and
    ALPHA, the observation-noise floor. alpha leans HIGH -- the same "the target
    is mostly noise" lesson that pushes the boosters' min_data_in_leaf and the
    MLP's alpha up, and what also keeps K + alpha*I positive definite.

    Classification (GaussianProcessClassifier) drops alpha -- the Laplace
    approximation has no noise diagonal -- and keeps kernel + length-scale.

    ARD (one length-scale per feature) is NOT tuned here: it is 22-44 extra
    dimensions that cannot be swept as scalars, so it stays a constructor default
    (isotropic) that E54 pins per cell via --fixed-params and gp_ktz compiles in
    only if the sweep says it wins. `subsample` (the O(n^3) row cap) and
    `max_iter_predict` arrive via --model-options so they are fixed per run and
    recorded in the metrics json, not searched.

    Every grid entry is a numeric array or a string list, so the grid translates
    under both --search-type random and bayes with no search_spaces change.
    """

    def __init__(self, task="regression", seed=None, n_jobs=-1, subsample=None,
                 max_iter_predict=100):
        # n_jobs is accepted and dropped, like MLP/TabPFN: GaussianProcessRegressor
        # has no worker count, and GaussianProcessClassifier only parallelises
        # across one-vs-rest classes -- binary is a single class, so there is
        # nothing to parallelise. BLAS threads (OMP/MKL/OPENBLAS_NUM_THREADS) are
        # the only knob that bounds a Cholesky; the runners export them.
        del n_jobs

        self.model_name = "gp"
        self.search_type = "random"

        # kernel FAMILY as a string (never a kernel object) and a log-spaced
        # length-scale; shared by both tasks.
        kernel_grid = ["matern32", "matern52", "rbf"]
        length_scale_grid = np.logspace(-1.3, 0.7, 20)      # ~0.05 .. ~5, log

        if task == "classification":
            from src.prediction.gp import SubsampledGPClassifier
            kwargs = {"random_state": seed,
                      "max_iter_predict": int(max_iter_predict)}
            if subsample is not None:
                kwargs["subsample"] = int(subsample)
            self.ModelClass = SubsampledGPClassifier(**kwargs)
            self.param_grid = {
                "kernel": kernel_grid,
                "length_scale": length_scale_grid,
            }
        else:
            from src.prediction.gp import SubsampledGPRegressor
            kwargs = {"random_state": seed}
            if subsample is not None:
                kwargs["subsample"] = int(subsample)
            self.ModelClass = SubsampledGPRegressor(**kwargs)
            self.param_grid = {
                "kernel": kernel_grid,
                "length_scale": length_scale_grid,
                "alpha": np.logspace(-3, 0.5, 20),          # NOISE floor, high
            }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#
# Same plain-dict dispatch the CLI uses for tournaments. Adding a model means
# writing a wrapper and adding one line here.
#
class _SurTreeWrapper:
    """
    A tree family (lgb/rf/catboost) with the cross-symbol factor-GLS treatment.

    The E61/`sur_ridge` idea generalised to a non-linear base: SurAugment adds k
    common cross-symbol factor scores as EXTRA FEATURES during fit and predicts
    with them zeroed, so the tree explains away common shocks in-sample and scores
    on real features out-of-sample. n_factors=0 is an EXACT plain-tree control.

    It reuses an existing base wrapper (LGBKtzWrapper etc.) for the estimator AND
    its tuned grid, wrapping the estimator in SurAugment and prefixing the grid
    keys with `estimator__` so the search ranges over base hyperparameters x
    n_factors. fit_grid is delegated to the base (lgb_ktz rescales
    min_child_samples per block there), so nothing about the base tune changes
    except the added factor columns. Symbol and date are routed to fit().
    """

    WANTS_PERIOD = True
    WANTS_SYMBOL = True

    def __init__(self, model_name, base, task, n_factors=(0, 1, 2, 3, 5),
                 min_symbol_rows=60):
        self.model_name = model_name
        self.search_type = "random"
        self.task = task
        self._base = base
        self.n_factors = tuple(int(k) for k in n_factors)
        if any(k < 0 for k in self.n_factors):
            raise ValueError("n_factors must be non-negative, got %r" % (n_factors,))
        self.ModelClass = SurAugment(estimator=base.ModelClass, task=task,
                                     min_symbol_rows=int(min_symbol_rows))

    def _prefix(self, grid):
        def one(g):
            d = {"estimator__" + k: v for k, v in g.items()}
            d["n_factors"] = list(self.n_factors)
            return d
        return [one(g) for g in grid] if isinstance(grid, list) else one(grid)

    @property
    def param_grid(self):
        # The un-rescaled grid, for anything that inspects a wrapper's search
        # space without data in hand (the bayes-translation tests, --search-type
        # bayes); fit_grid is what a real fit searches.
        return self._prefix(self._base.param_grid)

    def fit_grid(self, X, y):
        base_grid = (self._base.fit_grid(X, y) if hasattr(self._base, "fit_grid")
                     else self._base.param_grid)
        return self._prefix(base_grid)


class SurLgbWrapper(_SurTreeWrapper):
    def __init__(self, task="regression", seed=None, n_jobs=-1, **options):
        base = LGBKtzWrapper(task=task, seed=seed, n_jobs=n_jobs)
        super().__init__("sur_lgb", base, task, **options)


class SurRfWrapper(_SurTreeWrapper):
    def __init__(self, task="regression", seed=None, n_jobs=-1, **options):
        base = RFKtzWrapper(task=task, seed=seed, n_jobs=n_jobs)
        super().__init__("sur_rf", base, task, **options)


class SurCatboostWrapper(_SurTreeWrapper):
    # catboost_ktz is regression-only; the plain catboost wrapper carries the
    # classifier, so each task keeps a matched stacked baseline of the same base.
    def __init__(self, task="regression", seed=None, n_jobs=-1, **options):
        base = (CatBoostKtzWrapper(task=task, seed=seed, n_jobs=n_jobs)
                if task == "regression"
                else CatBoostWrapper(task=task, seed=seed, n_jobs=n_jobs))
        super().__init__("sur_catboost", base, task, **options)


MODELS = {
    "ridge": RidgeWrapper,
    "ridge_glmnet": RidgeGlmnetWrapper,
    "gen_ridge": GenRidgeWrapper,
    "sur_ridge": SurRidgeWrapper,
    "sur_lgb": SurLgbWrapper,
    "sur_rf": SurRfWrapper,
    "sur_catboost": SurCatboostWrapper,
    "lgb": LGBWrapper,
    "lgb_ktz": LGBKtzWrapper,
    "xgb": XGBWrapper,
    "catboost": CatBoostWrapper,
    "catboost_ktz": CatBoostKtzWrapper,
    "catboost_ktz_ens": CatBoostKtzEnsembleWrapper,
    "rf": RFWrapper,
    "rf_ktz": RFKtzWrapper,
    "mlp": MLPWrapper,
    "mlp_ktz": MLPKtzWrapper,
    "mlp_ktz_ens": MLPKtzEnsembleWrapper,
    "mlp_ktz2": MLPKtz2Wrapper,
    "mlp_ktz2_ens": MLPKtz2EnsembleWrapper,
    "torch_mlp": TorchMLPWrapper,
    "torch_mlp_emb": TorchMLPEmbWrapper,
    "torch_mlp_loss": TorchMLPLossWrapper,
    "tabnet": TabNetWrapper,
    "tabnet_ktz": TabNetKtzWrapper,
    "tabnet_ktz_tv": TabNetKtzTVWrapper,
    "tabpfn": TabPFNWrapper,
    "tabpfn_clf": TabPFNClfWrapper,
    "gp": GPWrapper,
}


def supported_tasks(model_name):
    """
    Which tasks a wrapper implements.

    Nearly every wrapper here does both, so the default is both and only the
    exceptions declare a TASKS attribute. Exposed because the alternative -- a
    caller building the wrapper and catching ValueError -- cannot tell "this model
    does not do classification" apart from "this model failed to construct".

    :param model_name: key into MODELS
    :type model_name: str
    :return: the tasks it accepts
    :rtype: tuple of str
    """
    if model_name not in MODELS:
        raise ValueError(
            "unknown model '{}', expected one of {}".format(
                model_name, sorted(MODELS))
        )
    return tuple(getattr(MODELS[model_name], "TASKS",
                         ("regression", "classification")))


def build_wrapper(model_name, task, seed=None, n_jobs=-1, **options):
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
    :param options: extra keyword arguments for this wrapper's constructor, from
                    run_forecast's --model-options. These configure the SEARCH
                    SPACE rather than an estimator parameter, which is what
                    --fixed-params cannot do: pinning a value there collapses the
                    grid to one candidate and disables the search entirely, so
                    "hold embed_dim at 8 and search everything else" had no way
                    to be expressed. Unknown keys raise from the constructor.
    :return: a configured wrapper
    """
    if model_name not in MODELS:
        raise ValueError(
            "unknown model '{}', expected one of {}".format(
                model_name, sorted(MODELS)
            )
        )

    if task not in supported_tasks(model_name):
        raise ValueError(
            "model '{}' does not implement task '{}'; it supports {}".format(
                model_name, task, ", ".join(supported_tasks(model_name)))
        )

    Wrapper = MODELS[model_name]

    # The linear wrappers have no use for a worker count.
    # Do NOT add "gen_ridge" here: this branch forbids --model-options, and
    # choosing which W / b0 / Delta families the search may range over is
    # exactly what that flag is for. gen_ridge accepts and drops n_jobs instead.
    if model_name in ("ridge", "ridge_glmnet"):
        if options:
            raise ValueError(
                "model '{}' takes no options; got {}".format(
                    model_name, ", ".join(sorted(options)))
            )
        return Wrapper(task=task, seed=seed)
    try:
        return Wrapper(task=task, seed=seed, n_jobs=n_jobs, **options)
    except TypeError as error:
        # A misspelled key is otherwise a bare TypeError from deep inside a
        # constructor, hours after the flag was typed.
        raise ValueError(
            "model '{}' rejected --model-options {}: {}".format(
                model_name, sorted(options), error)
        )
