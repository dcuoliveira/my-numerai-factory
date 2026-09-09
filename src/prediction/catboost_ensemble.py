"""
A CatBoost ensemble that AVERAGES a hyperparameter grid instead of selecting one.

The rest of this pipeline tunes a model by RandomizedSearchCV: it scores every
candidate under an inner CV and keeps the single best. On a panel with ~26
effective independent windows (510 daily periods at a 20-day, ~95%-overlapping
target) that selection is noisy -- the difference between the best CV config and
the third-best is routinely inside one standard error, so which one "wins" is
partly a draw of the folds. See the crypto-experiment-reporting skill's "26
effective windows" note.

This estimator takes the other branch of that fork. It fits ONE CatBoost per
cell of a small, KTZ-informed grid and, at predict time, AVERAGES their outputs.
Nothing is selected, so there is no selection variance to pay -- the cost is
fitting the whole grid every refit rather than searching a sample of it, which
CatBoost's column-starved trees (rsm ~= 0.1) make cheap: ~5-7s per member fit on
a 39,000-row block even at 812 features.

------------------------------------------------------------------------------
It is a drop-in MODELS estimator, so it deploys for free
------------------------------------------------------------------------------

The whole grid lives INSIDE one sklearn estimator: `.fit()` fits every member,
`.predict()` averages them. To `run_forecast` and to `src/live/refit.py` this is
just another regressor -- one call to `joblib.dump(search.best_estimator_)`
persists the entire ensemble, exactly as it does for ridge or a single booster.
That is the same contract CLAUDE.md names as the reason the torch LSTM cannot be
submitted; a grid ensemble honours it, so a catboost_ktz slot is deployable.

The consequence is that the walk-forward's inner hyperparameter search has
nothing to search -- the wrapper hands it a one-candidate grid. Run such an arm
with `--n-splits 2` (one usable fold): the inner CV then produces a single
unused score instead of five, halving the wasted work, exactly as the TabNet and
TabPFN default-parameter arms do (E18/E19).

------------------------------------------------------------------------------
Why rank-average by default
------------------------------------------------------------------------------

`combine="rank"` is the default, matching mlp_ktz_ens's GridEnsembleRegressor
and run_ensemble.py: each member is ranked to (0, 1] over the predicted block
and the ranks are averaged. Ranking first is the scale normalisation the book's
own blends use -- a low-learning-rate or shallow member has a wider output
spread than a deep one, and a raw mean would let the widest-spread member
dominate. It is a BLOCK-level rank (predict sees a feature matrix with no period
labels), an approximation of run_ensemble's per-period rank that is exact when a
member has no systematic level drift across the block; the final numerai_corr
re-ranks per period at scoring time, so any residual difference survives only
through such drift, and test_pipeline quantifies it as negligible.

`combine="mean"` (plain average -- every member is an RMSE fit on the same
target, so the outputs do share a scale) and `combine="zscore"` (average after
per-member standardisation) are offered so the rank rule's block-vs-period
approximation can be checked against alternatives rather than assumed.
"""

import itertools

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


COMBINERS = ("mean", "rank", "zscore")


class CatBoostGridEnsemble(BaseEstimator, RegressorMixin):
    """
    Fit one CatBoost per (learning_rate, depth, rsm) cell; average the outputs.

    A plain sklearn regressor: clone -> set_params -> fit -> predict, which is
    the whole path `hyper_params_search` exercises. Every constructor argument is
    stored VERBATIM (sklearn's clone contract), and the grid defaults are
    resolved in `fit`, never in `__init__` -- so `get_params` returns exactly
    what was passed and a cloned estimator is identical.

    The grid axes map the three Numerai booster idioms (see models.py's header)
    plus the E27/KTZ "favour shallow, span the learning rate" finding onto
    CatBoost's parameter names:

        learning_rate   the axis worth spanning; the ensemble AVERAGES over it
                        rather than picking one, so a range is the point
        depth           shallow symmetric trees (2, 4 -> 4, 16 leaves), the
                        CatBoost analogue of lgb_ktz's num_leaves {2, 4, 8}
        rsm             low column sampling (~0.1), the most distinctive Numerai
                        booster setting, kept as a small spread for diversity
        iterations      pinned; with lr in [0.005, 0.02] the total learning
                        L = lr * iterations lands in [10, 40], the good side of
                        the E27 L-is-a-sufficient-statistic finding
        l2_leaf_reg     pinned; CatBoost's L2, the analogue of lgb_ktz's
                        reg_lambda = 10
        min_data_in_leaf pinned at 100, the E27 leaf floor
    """

    # Resolved in fit() when the corresponding argument is None. Kept as class
    # attributes so the wrapper and the test read one source of truth.
    LEARNING_RATES = (0.005, 0.01, 0.02)
    DEPTHS = (2, 4)
    RSMS = (0.1, 0.2)
    ITERATIONS = 2000
    L2_LEAF_REG = 10.0
    MIN_DATA_IN_LEAF = 100

    def __init__(self, task="regression", seed=None, thread_count=-1,
                 combine="rank", learning_rates=None, depths=None, rsms=None,
                 iterations=None, l2_leaf_reg=None, min_data_in_leaf=None):
        # Stored verbatim -- do NOT resolve None here; sklearn.clone requires
        # get_params() to round-trip the exact values __init__ received.
        self.task = task
        self.seed = seed
        self.thread_count = thread_count
        self.combine = combine
        self.learning_rates = learning_rates
        self.depths = depths
        self.rsms = rsms
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf

    def _grid(self):
        """The (learning_rate, depth, rsm) cells, defaults resolved."""
        lrs = self.LEARNING_RATES if self.learning_rates is None else self.learning_rates
        depths = self.DEPTHS if self.depths is None else self.depths
        rsms = self.RSMS if self.rsms is None else self.rsms
        return list(itertools.product(
            [float(x) for x in lrs],
            [int(d) for d in depths],
            [float(r) for r in rsms],
        ))

    def fit(self, X, y):
        """
        Fit every grid member on (X, y).

        CatBoost is imported here, not at module scope, so `import
        src.prediction.models` stays light and torch-free-style on a box that
        has no CatBoost wheel -- the same lazy pattern CatBoostWrapper uses.
        """
        from catboost import CatBoostRegressor

        if self.task != "regression":
            raise ValueError(
                "catboost_ktz implements regression only; the grid-ensemble "
                "study scores E[y|x] on the binned target and no arm has asked "
                "for a classifier ensemble.")
        if self.combine not in COMBINERS:
            raise ValueError("combine must be one of {}, got {!r}".format(
                COMBINERS, self.combine))

        iterations = self.ITERATIONS if self.iterations is None else int(self.iterations)
        l2 = self.L2_LEAF_REG if self.l2_leaf_reg is None else float(self.l2_leaf_reg)
        min_leaf = (self.MIN_DATA_IN_LEAF if self.min_data_in_leaf is None
                    else int(self.min_data_in_leaf))

        values = np.asarray(X, dtype="float32")
        target = np.asarray(y, dtype="float32")
        self.n_features_in_ = values.shape[1]

        self.members_ = []
        self.member_params_ = []
        for learning_rate, depth, rsm in self._grid():
            model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                rsm=rsm,
                l2_leaf_reg=l2,
                min_data_in_leaf=min_leaf,
                random_seed=self.seed,
                thread_count=self.thread_count,
                allow_writing_files=False,   # no catboost_info/ per fit
                verbose=0,
            )
            model.fit(values, target)
            self.members_.append(model)
            self.member_params_.append(
                {"learning_rate": learning_rate, "depth": depth, "rsm": rsm})

        self.n_members_ = len(self.members_)
        return self

    def predict(self, X):
        """
        Combine the members' predictions into one signal per row.

        The default `rank` averages each member's within-block rank, the same
        rule mlp_ktz_ens (GridEnsembleRegressor) and run_ensemble.py use, so a
        wide-spread member cannot dominate. `mean` and `zscore` are alternatives;
        all three feed a per-period re-ranking at scoring time.
        """
        from scipy.stats import rankdata

        if not getattr(self, "members_", None):
            raise RuntimeError("CatBoostGridEnsemble.predict called before fit")

        values = np.asarray(X, dtype="float32")
        n = values.shape[0]
        if n == 0:
            return np.zeros(0, dtype="float64")

        # rows x members
        preds = np.column_stack(
            [np.asarray(m.predict(values), dtype="float64") for m in self.members_])

        if self.combine == "mean":
            combined = preds.mean(axis=1)
        elif self.combine == "zscore":
            mu = preds.mean(axis=0, keepdims=True)
            sd = preds.std(axis=0, keepdims=True)
            sd[sd == 0.0] = 1.0
            combined = ((preds - mu) / sd).mean(axis=1)
        else:   # rank -- each member ranked to (0, 1] over the block, averaged
            total = np.zeros(n, dtype="float64")
            for j in range(preds.shape[1]):
                total += rankdata(preds[:, j], method="average") / n
            combined = total / preds.shape[1]

        return combined.astype("float64")
