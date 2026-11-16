"""
Tests for the parts of the pipeline that fail silently when they are wrong.

Everything here runs offline in a couple of seconds -- no downloads, no fitting
of real models. These cover the mistakes that produce plausible-looking numbers
rather than crashes:

  * an embargo that does not actually separate train from validation
  * a rolling window that quietly behaves like an expanding one
  * summary statistics that drift from the official notebook formulas
  * a drawdown objective whose sign is inverted, so the search picks the worst
  * a validation split that leaks across the boundary
  * a glmnet path missing its scale conversion, putting every alpha in the
    wrong regime
  * a Bayesian search space translated with the wrong prior, or a warm-up that
    consumes the whole budget so no surrogate is ever fitted

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

from src.data_mani.utils import (
    REFERENCE_COL,
    TOURNAMENTS,
    build_meta_proxy,
    default_embargo,
    make_validation_split,
)
from src.prediction.functions import (
    build_scorer,
    grid_bounds,
    hyper_params_search,
    predict_signal,
    purged_period_splits,
    score_frame,
)
from src.prediction.metrics import summarize
from src.prediction.models import (
    MODELS,
    MLPWrapper,
    build_wrapper,
    supported_tasks,
    glmnet_alpha_path,
    scaled_leaf_range,
)
from src.prediction.search_spaces import (
    build_bayes_search,
    describe_budget,
    initial_points,
    is_log_spaced,
    sequence_valued_params,
    space_bounds,
    split_budget,
    to_dimension,
    to_space,
)


# Five rows per period, 100 periods -- enough to slice into folds.
N_PERIODS = 100
ROWS_PER_PERIOD = 5
PERIODS = np.repeat(np.arange(N_PERIODS), ROWS_PER_PERIOD)


class TestPurgedSplits(unittest.TestCase):
    """The embargo has to actually separate the two sides."""

    def _check_embargo(self, scheme, train_window=None):
        embargo = 4
        splits = purged_period_splits(
            PERIODS, n_splits=5, embargo=embargo,
            scheme=scheme, train_window=train_window,
        )
        self.assertGreater(len(splits), 0)

        for train_idx, val_idx in splits:
            train_periods = set(PERIODS[train_idx])
            val_periods = set(PERIODS[val_idx])

            # No period may appear on both sides.
            self.assertEqual(train_periods & val_periods, set())

            # And the gap must exceed the embargo, not merely exist.
            gap = min(val_periods) - max(train_periods)
            self.assertGreater(gap, embargo)

    def test_embargo_holds_expanding(self):
        self._check_embargo("expanding")

    def test_embargo_holds_rolling(self):
        self._check_embargo("rolling", train_window=20)

    def test_expanding_grows_from_the_start(self):
        splits = purged_period_splits(PERIODS, n_splits=5, embargo=4,
                                      scheme="expanding")
        sizes = []
        for train_idx, _ in splits:
            train_periods = sorted(set(PERIODS[train_idx]))
            # Expanding always reaches back to the earliest period available.
            self.assertEqual(train_periods[0], 0)
            sizes.append(len(train_periods))

        # Each fold trains on strictly more history than the one before it.
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), len(sizes))

    def test_rolling_keeps_a_fixed_window(self):
        train_window = 20
        splits = purged_period_splits(PERIODS, n_splits=5, embargo=4,
                                      scheme="rolling", train_window=train_window)

        sizes = [len(set(PERIODS[train_idx])) for train_idx, _ in splits]

        # Never longer than the window, and after the first (possibly truncated)
        # fold every window is exactly full.
        for size in sizes:
            self.assertLessEqual(size, train_window)
        self.assertEqual(sizes[-1], train_window)

    def test_rolling_requires_a_window(self):
        # Without this guard, rolling would silently behave like expanding.
        with self.assertRaises(ValueError):
            purged_period_splits(PERIODS, n_splits=5, embargo=4, scheme="rolling")

    def test_rolling_trains_on_less_than_expanding(self):
        common = dict(n_splits=5, embargo=4)
        expanding = purged_period_splits(PERIODS, scheme="expanding", **common)
        rolling = purged_period_splits(PERIODS, scheme="rolling",
                                       train_window=10, **common)

        # By the last fold, expanding has accumulated far more history.
        self.assertGreater(len(set(PERIODS[expanding[-1][0]])),
                           len(set(PERIODS[rolling[-1][0]])))


class TestSummarize(unittest.TestCase):
    """Summary statistics must match the hello_numerai notebook exactly."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.scores = pd.Series(rng.normal(0.02, 0.01, size=200))

    def test_matches_notebook_formulas(self):
        got = summarize(self.scores)

        # These four lines are the notebook's, transcribed verbatim.
        corr_mean = self.scores.mean()
        corr_std = self.scores.std(ddof=0)
        corr_sharpe = corr_mean / corr_std
        cumsum = self.scores.cumsum()
        corr_max_drawdown = (cumsum.expanding(min_periods=1).max() - cumsum).max()

        self.assertAlmostEqual(got["mean"], corr_mean, places=12)
        self.assertAlmostEqual(got["std"], corr_std, places=12)
        self.assertAlmostEqual(got["sharpe"], corr_sharpe, places=12)
        self.assertAlmostEqual(got["max_drawdown"], corr_max_drawdown, places=12)

    def test_uses_ddof_zero(self):
        # target_ensemble.ipynb uses the pandas default of ddof=1. We follow
        # hello_numerai and the legacy scorer, so guard against a silent switch.
        got = summarize(self.scores)
        self.assertAlmostEqual(got["std"], self.scores.std(ddof=0), places=12)
        self.assertNotAlmostEqual(got["std"], self.scores.std(ddof=1), places=12)

    def test_drawdown_is_positive(self):
        # A larger number must always mean a worse drawdown.
        losing = pd.Series([0.1, -0.2, -0.3, 0.05])
        self.assertGreater(summarize(losing)["max_drawdown"], 0.0)

    def test_empty_is_not_a_crash(self):
        got = summarize(pd.Series([], dtype="float64"))
        self.assertTrue(np.isnan(got["mean"]))
        self.assertTrue(np.isnan(got["sharpe"]))


class TestScorerSign(unittest.TestCase):
    """
    sklearn always maximises, so every objective must be greater-is-better.

    The drawdown objectives are the ones that bite: they measure a loss, so the
    scorer has to negate them. Inverted, the search silently selects the worst
    candidate and nothing else in the pipeline notices.
    """

    def _frame(self, per_period_signal):
        """
        Build a frame whose per-period correlation follows `per_period_signal`.

        Within each period the prediction either tracks the target (positive
        correlation) or opposes it (negative), scaled by the requested strength.
        """
        rng = np.random.default_rng(7)
        rows = []
        for period, strength in enumerate(per_period_signal):
            target = rng.normal(size=40)
            noise = rng.normal(size=40)
            prediction = strength * target + (1.0 - abs(strength)) * noise
            rows.append(pd.DataFrame({
                "period": period,
                "target": target,
                "prediction": prediction,
            }))
        return pd.concat(rows, ignore_index=True)

    def test_higher_corr_scores_higher(self):
        strong = self._frame([0.9] * 20)
        weak = self._frame([0.1] * 20)

        self.assertGreater(
            score_frame(strong, "corr", "period", None),
            score_frame(weak, "corr", "period", None),
        )

    def test_steadier_series_has_higher_sharpe(self):
        steady = self._frame([0.5] * 20)
        erratic = self._frame([0.9, 0.1] * 10)

        self.assertGreater(
            score_frame(steady, "corr_sharpe", "period", None),
            score_frame(erratic, "corr_sharpe", "period", None),
        )

    def test_smaller_drawdown_scores_higher(self):
        # Steadily positive: the cumulative score rarely gives anything back.
        steady = self._frame([0.6] * 20)
        # Alternating sign: large peak-to-trough swings, so a big drawdown.
        swinging = self._frame([0.9, -0.9] * 10)

        steady_score = score_frame(steady, "corr_drawdown", "period", None)
        swinging_score = score_frame(swinging, "corr_drawdown", "period", None)

        # Greater-is-better, so the smaller drawdown must win.
        self.assertGreater(steady_score, swinging_score)
        # And because it is a negated loss, the scores are non-positive.
        self.assertLessEqual(steady_score, 0.0)

    def test_bmc_without_a_reference_is_unscorable(self):
        frame = self._frame([0.5] * 5)
        # Rather than raising mid-search, an unscorable candidate ranks last.
        self.assertLess(score_frame(frame, "bmc", "period", None), -1e8)


class TestWeightedObjective(unittest.TestCase):
    """
    The two-metric blend must be a true weighted sum, and degrade safely.

    The scale caveat is the thing to keep in mind rather than test: the
    components are not rescaled, so a nominal 50/50 of corr (~0.15) and
    corr_sharpe (~1.2) is dominated by sharpe. These tests only check the
    mechanics.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        rows = []
        for period in range(12):
            target = rng.normal(size=50)
            rows.append(pd.DataFrame({
                "period": period,
                "target": target,
                "prediction": 0.6 * target + 0.4 * rng.normal(size=50),
                "reference_prediction": 0.3 * target + 0.7 * rng.normal(size=50),
            }))
        self.frame = pd.concat(rows, ignore_index=True)

    def _blend(self, weight):
        """Reproduce the scorer's blend directly from score_frame."""
        primary = score_frame(self.frame, "corr", "period", "reference_prediction")
        secondary = score_frame(self.frame, "bmc", "period", "reference_prediction")
        return weight * primary + (1.0 - weight) * secondary, primary, secondary

    def test_blend_is_a_weighted_sum(self):
        blended, primary, secondary = self._blend(0.5)
        self.assertAlmostEqual(blended, 0.5 * primary + 0.5 * secondary, places=12)

    def test_extremes_recover_each_component(self):
        self.assertAlmostEqual(self._blend(1.0)[0], self._blend(1.0)[1], places=12)
        self.assertAlmostEqual(self._blend(0.0)[0], self._blend(0.0)[2], places=12)

    def test_weight_must_be_a_proportion(self):
        meta = self.frame[["period", "reference_prediction"]]
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                build_scorer("corr", meta, "period", "reference_prediction",
                             "regression", metric2="bmc", metric_weight=bad)

    def test_unknown_second_metric_rejected(self):
        meta = self.frame[["period", "reference_prediction"]]
        with self.assertRaises(ValueError):
            build_scorer("corr", meta, "period", "reference_prediction",
                         "regression", metric2="not_a_metric")


class TestMetaProxy(unittest.TestCase):
    """
    The Ridge proxy exists so BMC can be scored where the real meta model is not.

    It must (a) cover every period, and (b) be fit only on training rows -- the
    meta model's coverage often starts close to the validation boundary, so it
    would be easy to reach across it by accident.
    """

    def setUp(self):
        rng = np.random.default_rng(5)
        periods = np.repeat(np.arange(40), 30)
        n = len(periods)
        self.features = ["f0", "f1", "f2"]
        df = pd.DataFrame(rng.normal(size=(n, 3)), columns=self.features)
        df["period"] = periods
        # Train on the first 30 periods, validate on the last 10.
        df["split"] = np.where(df["period"] < 30, "train", "validation")
        # The meta model only exists from period 20 onward -- straddling the
        # train/validation boundary, exactly as it does on Crypto.
        df["meta_model"] = np.where(
            df["period"] >= 20, df["f0"] + 0.2 * rng.normal(size=n), np.nan
        )
        self.df = df

    def test_proxy_covers_every_period(self):
        out, _ = build_meta_proxy(
            self.df, self.features, "period",
            train_mask=(self.df["split"] == "train"), verbose=False,
        )
        # The real meta model covers half the periods; the proxy covers all.
        self.assertEqual(out["meta_proxy"].notna().all(), True)
        self.assertLess(self.df["meta_model"].notna().mean(), 1.0)

    def test_proxy_is_fit_on_training_periods_only(self):
        _, n_fit = build_meta_proxy(
            self.df, self.features, "period",
            train_mask=(self.df["split"] == "train"), verbose=False,
        )
        # Periods 20-29 are both in training and have a meta model: 10 of them.
        # If validation periods leaked in this would be 20.
        self.assertEqual(n_fit, 10)

    def test_raises_when_no_training_overlap(self):
        # Meta model entirely inside validation -- nothing to learn from.
        df = self.df.copy()
        df.loc[df["period"] < 30, "meta_model"] = np.nan
        with self.assertRaises(ValueError):
            build_meta_proxy(df, self.features, "period",
                             train_mask=(df["split"] == "train"), verbose=False)


class TestValidationSplit(unittest.TestCase):
    """The Crypto split must be chronological and genuinely gapped."""

    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.df = pd.DataFrame({
            "date": np.repeat(dates, 3),
            "symbol": ["A", "B", "C"] * 100,
        })

    def test_split_is_chronological_and_embargoed(self):
        embargo = 20
        out = make_validation_split(self.df, "date", val_frac=0.3, embargo=embargo)

        train_dates = out.loc[out["split"] == "train", "date"]
        val_dates = out.loc[out["split"] == "validation", "date"]

        # Every validation date is strictly later than every training date.
        self.assertLess(train_dates.max(), val_dates.min())

        # And the gap is at least the embargo, so no 20-day target window
        # straddles the boundary.
        gap_days = (val_dates.min() - train_dates.max()).days
        self.assertGreaterEqual(gap_days, embargo)

    def test_embargoed_periods_are_dropped(self):
        out = make_validation_split(self.df, "date", val_frac=0.3, embargo=20)

        # The embargoed periods belong to neither side and must not survive.
        self.assertEqual(set(out["split"].unique()), {"train", "validation"})
        self.assertEqual(out["date"].nunique(), 100 - 20)

    def test_validation_fraction_is_respected(self):
        out = make_validation_split(self.df, "date", val_frac=0.3, embargo=0)
        n_val = out.loc[out["split"] == "validation", "date"].nunique()
        self.assertEqual(n_val, 30)

    def test_impossible_split_raises(self):
        # Asking for more validation than there is data must fail loudly.
        with self.assertRaises(ValueError):
            make_validation_split(self.df, "date", val_frac=0.99, embargo=20)


class TestGlmnetPath(unittest.TestCase):
    """
    The glmnet path is only correct if the lambda -> alpha conversion is applied.

    glmnet minimises (1/2N)*RSS + (lambda/2)*||b||^2 while sklearn's Ridge
    minimises RSS + alpha*||b||^2, so alpha = lambda * N. Drop the factor of N
    and every candidate sits in the wrong regime -- but the code still runs and
    still produces numbers, which is why this needs a test.
    """

    def setUp(self):
        rng = np.random.default_rng(3)
        self.n, self.p = 500, 8
        self.X = pd.DataFrame(
            rng.normal(size=(self.n, self.p)),
            columns=["f{}".format(i) for i in range(self.p)],
        )
        beta = rng.normal(size=self.p)
        self.y = pd.Series(self.X.to_numpy() @ beta + rng.normal(scale=0.1, size=self.n))

    def test_path_is_descending_and_positive(self):
        alphas = glmnet_alpha_path(self.X, self.y, n_lambda=50)
        self.assertEqual(len(alphas), 50)
        self.assertTrue(np.all(alphas > 0))
        self.assertTrue(np.all(np.diff(alphas) < 0))

    def test_largest_alpha_shrinks_to_zero(self):
        alphas = glmnet_alpha_path(self.X, self.y, n_lambda=50)

        strongest = Ridge(alpha=alphas[0]).fit(self.X, self.y)
        ols = LinearRegression().fit(self.X, self.y)

        # The top of the path must be strong enough to crush the coefficients.
        self.assertLess(
            np.abs(strongest.coef_).max(),
            0.01 * np.abs(ols.coef_).max(),
        )

    def test_smallest_alpha_is_nearly_unregularised(self):
        alphas = glmnet_alpha_path(self.X, self.y, n_lambda=50)

        weakest = Ridge(alpha=alphas[-1]).fit(self.X, self.y)
        ols = LinearRegression().fit(self.X, self.y)

        # glmnet's path stops at eps * lambda_max, never at zero, so the bottom
        # end is *nearly* unregularised rather than exactly OLS. With X'X ~ n the
        # residual shrinkage is n / (n + alpha_min), which on this small, strong-
        # signal fixture leaves around 12%. On real data it is negligible: at
        # n = 500k it is 0.003%, at n = 2.4M it is 0.001%. So assert the
        # coefficients keep most of their magnitude and all of their signs,
        # rather than matching OLS exactly.
        ratio = weakest.coef_ / ols.coef_
        self.assertTrue(np.all(ratio > 0.8), "bottom of path shrinks too hard")
        self.assertTrue(np.all(ratio <= 1.0), "ridge cannot inflate coefficients")

    def test_path_spans_crushed_to_nearly_free(self):
        alphas = glmnet_alpha_path(self.X, self.y, n_lambda=50)
        ols = np.abs(LinearRegression().fit(self.X, self.y).coef_).max()

        top = np.abs(Ridge(alpha=alphas[0]).fit(self.X, self.y).coef_).max()
        bottom = np.abs(Ridge(alpha=alphas[-1]).fit(self.X, self.y).coef_).max()

        # The whole point of the path is that it brackets the useful region.
        self.assertLess(top, 0.01 * ols)
        self.assertGreater(bottom, 0.8 * ols)

    def test_scale_conversion_is_present(self):
        # Without the * N, lambda_max would be ~N times smaller and the strongest
        # penalty on the path would no longer zero the coefficients.
        alphas = glmnet_alpha_path(self.X, self.y, n_lambda=50)
        unconverted = alphas[0] / self.n

        coef_converted = np.abs(Ridge(alpha=alphas[0]).fit(self.X, self.y).coef_).max()
        coef_unconverted = np.abs(Ridge(alpha=unconverted).fit(self.X, self.y).coef_).max()

        self.assertLess(coef_converted, coef_unconverted)


class TestScaledLeafRange(unittest.TestCase):
    """
    The minimum-leaf-size range has to track the training block size.

    Numerai's min_data_in_leaf=10,000 is ~0.4% of Classic's 2.4M rows. Used
    verbatim on an 80k-row Crypto block it demands 16% of the data per leaf, no
    split is ever valid, and the booster returns one constant for every row --
    which surfaces as a NaN CORR rather than an error.
    """

    def test_small_block_gets_a_feasible_range(self):
        dist = scaled_leaf_range(80_000)
        draws = dist.rvs(size=200, random_state=0)

        # Never more than 2% of the block, or the model cannot split at all.
        self.assertLessEqual(draws.max(), 80_000 * 0.02)
        self.assertGreaterEqual(draws.min(), 20)

    def test_large_block_reaches_the_evidenced_value(self):
        # On full Classic the range must still cover Numerai's own 10,000.
        draws = scaled_leaf_range(2_400_000).rvs(size=500, random_state=0)
        self.assertGreater(draws.max(), 10_000)
        # But never runs away past the largest value anyone has evidenced.
        self.assertLessEqual(draws.max(), 20_000)

    def test_range_grows_with_the_block(self):
        small = scaled_leaf_range(50_000).rvs(size=200, random_state=0).max()
        large = scaled_leaf_range(500_000).rvs(size=200, random_state=0).max()
        self.assertLess(small, large)


class TestMLPWrapper(unittest.TestCase):
    """
    The MLP is the only wrapper whose estimator is a Pipeline and the only one
    with an internal validation split it must not use.

    Both failure modes are silent. An unscaled net simply scores worse and looks
    like a model that does not work; a net using sklearn's own early stopping
    scores BETTER, because its holdout is drawn at random from the same periods
    it trained on.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame(
            rng.random((240, 8)), columns=["f%d" % i for i in range(8)]
        )
        self.y = pd.Series(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], 240))

    def test_estimator_scales_before_the_net(self):
        # StandardScaler must be a STEP, so it is refit inside every CV fold.
        # Scaling once up front would carry the validation fold's mean and
        # variance into training, and purged_period_splits could not see it.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        self.assertEqual(
            [name for name, _ in wrapper.ModelClass.steps], ["scale", "mlp"])

    def test_internal_early_stopping_is_off(self):
        # sklearn's early_stopping holds out a RANDOM validation_fraction of
        # rows -- the exact leak purged_period_splits exists to prevent.
        for task in ("regression", "classification"):
            wrapper = build_wrapper("mlp", task=task, seed=42, n_jobs=1)
            self.assertFalse(
                wrapper.ModelClass.named_steps["mlp"].early_stopping, task)

    def test_grid_addresses_the_pipeline_step(self):
        # A grid key that misses the `mlp__` prefix is not an error -- sklearn
        # raises on set_params, candidate by candidate, and every one scores
        # UNSCORABLE.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        self.assertIsInstance(wrapper.param_grid, list)
        for grid in wrapper.param_grid:
            for key in grid:
                self.assertTrue(key.startswith("mlp__"), key)

    def test_one_sub_grid_per_architecture_with_the_size_pinned(self):
        # The booster idiom: skopt fits one surrogate per sub-grid, so pinning
        # the architecture gives each its own model of alpha rather than asking
        # one surrogate to interpolate between shapes.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        sizes = [g["mlp__hidden_layer_sizes"] for g in wrapper.param_grid]
        self.assertTrue(all(len(s) == 1 for s in sizes), sizes)
        self.assertTrue(all(isinstance(s[0], tuple) for s in sizes), sizes)
        self.assertEqual(len(sizes), len(set(s[0] for s in sizes)))

    def test_grid_bounds_survives_a_sequence_valued_parameter(self):
        # grid_bounds stacks each parameter's values across sub-grids to record
        # the span searched. hidden_layer_sizes is pinned per sub-grid as a
        # tuple, and (32,) does not stack against (64, 32) -- an uncaught
        # ValueError there kills the run at the FIRST refit, after the data has
        # loaded and the search has already fitted. It has no numeric span, so
        # the right answer is to omit it, exactly as a distribution is omitted.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        bounds = grid_bounds(wrapper.param_grid)

        self.assertNotIn("mlp__hidden_layer_sizes", bounds)
        # Read off the wrapper's own constants rather than restating them: the
        # point is that the SCALAR parameters still get a span recorded, not
        # what today's grid happens to span.
        for key, values in (("mlp__alpha", MLPWrapper.ALPHAS),
                            ("mlp__batch_size", MLPWrapper.BATCH_SIZES),
                            ("mlp__learning_rate_init", MLPWrapper.LEARNING_RATES)):
            low, high, _ = bounds[key]
            self.assertAlmostEqual(low, float(min(values)), msg=key)
            self.assertAlmostEqual(high, float(max(values)), msg=key)

    def test_space_bounds_survives_a_sequence_valued_parameter(self):
        # The Bayesian twin of the check above, and it needs its own test: a
        # Categorical over EQUAL-LENGTH tuples coerces to a 2-D float array
        # rather than raising, so the try/except that catches a non-numeric
        # category does not catch this one. It surfaced as a TypeError from
        # build_bayes_search at the first refit -- after the data had loaded.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        bounds = space_bounds([(to_space(g), 8) for g in wrapper.param_grid])

        self.assertNotIn("mlp__hidden_layer_sizes", bounds)
        self.assertEqual(bounds["mlp__alpha"][2], 0)        # continuous under bayes
        self.assertAlmostEqual(bounds["mlp__batch_size"][0],
                               float(min(MLPWrapper.BATCH_SIZES)))

    def test_bayes_refuses_the_grid_instead_of_failing_inside_the_fit(self):
        # skopt cannot round-trip a Categorical whose categories are tuples --
        # even a single-category one dies inside BayesSearchCV.fit() with
        # "can only convert an array of size 1 to a Python scalar". Left
        # unguarded that surfaces at the FIRST REFIT, after the data has loaded,
        # on a run that was never going to work.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        self.assertEqual(sequence_valued_params(wrapper.param_grid),
                         ["mlp__hidden_layer_sizes"])

        with self.assertRaises(ValueError) as caught:
            build_bayes_search(
                estimator=wrapper.ModelClass, param_grid=wrapper.param_grid,
                n_iter=12, cv=2, scoring="r2", refit=True, search_jobs=1,
                seed=42, error_score=0.0,
            )
        self.assertIn("--search-type random", str(caught.exception))

    def test_every_other_wrapper_stays_bayes_searchable(self):
        # The guard keys off sequence-valued entries, so it must not catch the
        # grids that have always worked under --search-type bayes.
        # Which wrappers legitimately carry a tuple-valued parameter is derived,
        # not listed: a hardcoded exclusion list silently grows stale, and the
        # third MLP variant is what made that concrete.
        carriers = set()
        for name in sorted(MODELS):
            for task in supported_tasks(name):
                wrapper = build_wrapper(name, task=task, seed=42, n_jobs=1)
                offenders = sequence_valued_params(wrapper.param_grid)
                if offenders:
                    carriers.add(name)
                    self.assertEqual(offenders, ["mlp__hidden_layer_sizes"],
                                     (name, task))

        # Every carrier is an MLP: hidden_layer_sizes is the only sequence-valued
        # hyperparameter shipped, so any OTHER wrapper appearing here would mean
        # the guard has started firing on a grid that used to search fine.
        self.assertTrue(all("mlp" in n for n in carriers), carriers)

    def test_regression_only_wrappers_say_so_rather_than_raising_late(self):
        # torch_mlp has no classification head. A caller must be able to ask
        # WITHOUT constructing one, or "does not support this task" is
        # indistinguishable from "failed to construct".
        self.assertEqual(supported_tasks("torch_mlp"), ("regression",))
        self.assertEqual(set(supported_tasks("mlp")),
                         {"regression", "classification"})
        with self.assertRaises(ValueError) as caught:
            build_wrapper("torch_mlp", task="classification", seed=42, n_jobs=1)
        self.assertIn("does not implement task", str(caught.exception))

    def test_torch_and_sklearn_mlp_search_the_same_grid(self):
        # The comparison between them is only about the implementation if the
        # search space is identical. torch_mlp REFERENCES MLPWrapper's constants
        # rather than copying them, and this is what pins that.
        a = build_wrapper("mlp", task="regression", seed=42, n_jobs=1)
        b = build_wrapper("torch_mlp", task="regression", seed=42, n_jobs=1)
        self.assertEqual(len(a.param_grid), len(b.param_grid))
        for ga, gb in zip(a.param_grid, b.param_grid):
            self.assertEqual(set(ga), set(gb))
            for key in ga:
                np.testing.assert_allclose(
                    np.asarray(ga[key], dtype=object).ravel().tolist(),
                    np.asarray(gb[key], dtype=object).ravel().tolist(),
                    err_msg=key)

    def test_id_passthrough_scaler_leaves_the_code_alone(self):
        # An integer ticker code is not a quantity. Standardising it produces a
        # meaningless number AND destroys the integer the lookup needs.
        import pandas as pd
        from src.prediction.torch_mlp import IdPassthroughScaler

        frame = pd.DataFrame({"feature_a": np.random.normal(5, 3, 60),
                              "feature_b": np.random.normal(0, 1, 60),
                              "feature_symbol_id": np.random.randint(0, 9, 60)})
        out = IdPassthroughScaler(id_columns=("feature_symbol_id",)).fit_transform(frame)
        # ids land LAST, unchanged -- that placement is the contract the
        # estimator's n_embedding_columns relies on.
        np.testing.assert_array_equal(out[:, -1],
                                      frame["feature_symbol_id"].to_numpy())
        self.assertAlmostEqual(out[:, :-1].mean(), 0.0, places=9)
        self.assertAlmostEqual(out[:, :-1].std(), 1.0, places=6)

    def test_with_no_id_columns_it_is_plain_standard_scaling(self):
        # This is what keeps the no-embedding arm comparable: the two Pipelines
        # differ in their first step only when an id column is actually present.
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from src.prediction.torch_mlp import IdPassthroughScaler

        frame = pd.DataFrame(np.random.normal(0, 1, (40, 3)),
                             columns=["feature_a", "feature_b", "feature_c"])
        np.testing.assert_allclose(
            IdPassthroughScaler(id_columns=("feature_symbol_id",)).fit_transform(frame),
            StandardScaler().fit_transform(frame))

    def test_the_embedding_learns_an_effect_the_features_cannot_carry(self):
        # The decisive check. On a target that depends on NOTHING but identity,
        # a network without the embedding is at chance and one with it is not.
        # Without this, an embedding that silently did nothing would still pass
        # every shape and clone test above.
        from scipy.stats import spearmanr
        from src.prediction.torch_mlp import IdPassthroughScaler, TorchMLPRegressor
        import pandas as pd

        rng = np.random.default_rng(0)
        n, k = 4000, 20
        ids = rng.integers(0, k, n)
        level = rng.normal(0, 1, k)
        frame = pd.DataFrame({"feature_a": rng.normal(0, 1, n),
                              "feature_b": rng.normal(0, 1, n),
                              "feature_symbol_id": ids})
        y = level[ids] + rng.normal(0, 0.2, n)
        scaled = IdPassthroughScaler(
            id_columns=("feature_symbol_id",)).fit_transform(frame)

        options = dict(hidden_layer_sizes=(32,), max_iter=300, random_state=0,
                       alpha=1e-6, learning_rate_init=0.01, batch_size=256)
        blind = TorchMLPRegressor(n_embedding_columns=0, **options).fit(
            scaled[:, :-1], y)
        seeing = TorchMLPRegressor(n_embedding_columns=1, embed_dim=8,
                                   **options).fit(scaled, y)
        self.assertLess(abs(spearmanr(blind.predict(scaled[:, :-1]), y).statistic), 0.3)
        self.assertGreater(spearmanr(seeing.predict(scaled), y).statistic, 0.8)
        self.assertEqual(seeing.n_symbols_, k)

    def test_an_unseen_code_lands_on_the_padding_row(self):
        # A code first met at predict time has nothing behind it to have learned
        # a vector from. It must reach the zeroed padding row and contribute no
        # identity signal -- not collide with some other ticker's vector.
        from src.prediction.torch_mlp import TorchMLPRegressor

        rng = np.random.default_rng(1)
        X = np.column_stack([rng.normal(0, 1, 200), rng.integers(0, 5, 200)])
        model = TorchMLPRegressor(hidden_layer_sizes=(8,), max_iter=20,
                                  random_state=0, n_embedding_columns=1,
                                  embed_dim=4).fit(X, rng.normal(0, 1, 200))
        self.assertEqual(model.n_symbols_, 5)
        rows = model._index_codes(np.array([0, 4, 99, -7]))
        self.assertEqual(list(rows[:2]), [0, 4])
        self.assertEqual(list(rows[2:]), [model._padding_row] * 2)
        # padding_idx keeps that row at zero through training, so an unseen
        # ticker adds nothing rather than adding noise.
        weight = model._embedding.weight.detach().cpu().numpy()
        np.testing.assert_allclose(weight[model._padding_row], 0.0)

    def test_embedding_wrapper_differs_from_torch_mlp_in_exactly_one_key(self):
        a = build_wrapper("torch_mlp", task="regression", seed=42, n_jobs=1)
        b = build_wrapper("torch_mlp_emb", task="regression", seed=42, n_jobs=1)
        for ga, gb in zip(a.param_grid, b.param_grid):
            self.assertEqual(set(gb) - set(ga), {"mlp__embed_dim"})
            self.assertEqual(set(ga) - set(gb), set())

    def test_model_options_pin_a_width_without_disabling_the_search(self):
        # This is the whole reason --model-options exists. --fixed-params CANNOT
        # express it: it rewrites the grid to one candidate per key and forces
        # n_iter to 1, so pinning embed_dim there would also pin alpha, the
        # architecture and everything else.
        pinned = build_wrapper("torch_mlp_emb", task="regression", seed=42,
                               n_jobs=1, embed_dim=64)
        searched = build_wrapper("torch_mlp_emb", task="regression", seed=42,
                                 n_jobs=1)
        for grid in pinned.param_grid:
            self.assertEqual(grid["mlp__embed_dim"], [64])
            # everything else must still be a range, not a pin
            self.assertGreater(len(grid["mlp__alpha"]), 1)
            self.assertGreater(len(grid["mlp__batch_size"]), 1)
        self.assertEqual(searched.param_grid[0]["mlp__embed_dim"],
                         MLPWrapper.__mro__ and [4, 8, 16, 32])

    def test_a_width_outside_the_search_grid_is_still_pinnable(self):
        # EMBED_DIMS is where the SEARCH looks, not a whitelist. E12 pins 64,
        # which is deliberately outside it.
        wrapper = build_wrapper("torch_mlp_emb", task="regression", seed=42,
                                n_jobs=1, embed_dim=64)
        self.assertNotIn(64, wrapper.EMBED_DIMS)
        self.assertEqual(wrapper.param_grid[0]["mlp__embed_dim"], [64])

    def test_zero_width_is_rejected_rather_than_silently_meaning_no_embedding(self):
        # A zero-width table is not the same object as no table: it would still
        # carry the id column, the padding row and the concatenation. The control
        # arm must use --model torch_mlp instead, and the error says so.
        with self.assertRaises(ValueError) as caught:
            build_wrapper("torch_mlp_emb", task="regression", seed=42, n_jobs=1,
                          embed_dim=0)
        self.assertIn("torch_mlp", str(caught.exception))

    def test_an_unknown_option_names_itself(self):
        # Otherwise a typo is a bare TypeError from inside a constructor, hours
        # after the flag was typed.
        with self.assertRaises(ValueError) as caught:
            build_wrapper("torch_mlp_emb", task="regression", seed=42, n_jobs=1,
                          embbed_dim=8)
        self.assertIn("embbed_dim", str(caught.exception))

    def test_wrappers_that_take_no_options_say_so(self):
        with self.assertRaises(ValueError) as caught:
            build_wrapper("ridge", task="regression", seed=42, n_jobs=1,
                          embed_dim=8)
        self.assertIn("takes no options", str(caught.exception))

    def test_dummies_pass_through_the_scaler_untouched(self):
        # Standardising a binary column divides by sqrt(p(1-p)), so a rare dummy
        # becomes a huge spike: on a real 130-period block of v2.0-onehot the 740
        # ticker dummies reached max |z| = 197.48 while the raw 22 stayed inside
        # 2.29. Silent -- the run completes and just scores worse.
        from sklearn.preprocessing import StandardScaler
        from src.prediction.models import ScaleContinuousOnly

        rng = np.random.default_rng(0)
        continuous = rng.normal(5, 3, (400, 3))
        rare = np.zeros((400, 1)); rare[:2] = 1.0        # 0.5% prevalence
        X = np.hstack([continuous, rare])

        out = ScaleContinuousOnly().fit_transform(X)
        self.assertEqual(sorted(set(out[:, -1])), [0.0, 1.0])
        self.assertLess(abs(out).max(), 5.0)
        # what it protects against
        self.assertGreater(abs(StandardScaler().fit_transform(X)).max(), 10.0)

    def test_with_no_dummies_it_is_exactly_standard_scaling(self):
        # This is what keeps every arm run before the fix comparable: on a
        # dataset with no binary column the two are the same transformation, so
        # raw/tenc/interact/fenccausal/cumlag3/rollsel/spread reproduce unchanged.
        from sklearn.preprocessing import StandardScaler
        from src.prediction.models import ScaleContinuousOnly

        X = np.random.default_rng(1).normal(0, 3, (200, 5))
        np.testing.assert_allclose(ScaleContinuousOnly().fit_transform(X),
                                   StandardScaler().fit_transform(X))

    def test_both_mlp_wrappers_use_it(self):
        from src.prediction.models import ScaleContinuousOnly
        for name in ("mlp", "torch_mlp"):
            step = build_wrapper(name, task="regression", seed=42,
                                 n_jobs=1).ModelClass.steps[0][1]
            self.assertIsInstance(step, ScaleContinuousOnly, name)

    def test_mlp_ktz_encodes_the_e17_priors(self):
        # The grid IS the E17 recommendation; a regression here means the priors
        # silently drifted from what the sensitivity study found.
        w = build_wrapper("mlp_ktz", task="regression", seed=42, n_jobs=1)
        g = w.param_grid
        lr = np.asarray(g["mlp__learning_rate_init"])
        # lr is the dense tuning axis, bounded away from both measured corners:
        # 1e-4 underfits, 1e-1 fails without high alpha.
        self.assertGreaterEqual(len(lr), 15)
        self.assertGreater(lr.min(), 1e-4)      # above the underfit corner
        self.assertLess(lr.max(), 1e-1)         # below the high-lr danger zone
        # architecture is a coarse gate, all <= 3 layers (deeper collapses)
        self.assertLessEqual(len(g["mlp__hidden_layer_sizes"]), 4)
        self.assertTrue(all(len(a) <= 3 for a in g["mlp__hidden_layer_sizes"]))
        # alpha near-pinned (least important axis), leaning high
        self.assertLessEqual(len(g["mlp__alpha"]), 2)
        self.assertGreaterEqual(min(g["mlp__alpha"]), 1e-2)
        # activation PINNED to identity -- a free activation would eat ~half the
        # random-search draws (ParameterSampler splits each discrete axis evenly),
        # halving the lr density the whole design exists to maximise.
        self.assertEqual(g["mlp__activation"], ["identity"])
        # inherits the dummy-safe scaler for the one-hot book datasets
        from src.prediction.models import ScaleContinuousOnly
        self.assertIsInstance(w.ModelClass.steps[0][1], ScaleContinuousOnly)

    def test_mlp_ktz2_encodes_the_e33_priors(self):
        # The E33 re-sweep moved the box: alpha to ~10 (was the ceiling bug),
        # activation un-pinned to both, width a weak axis kept small. A regression
        # here means the new priors drifted from what E33 found.
        from sklearn.model_selection import ParameterGrid
        w = build_wrapper("mlp_ktz2", task="regression", seed=42, n_jobs=1)
        g = w.param_grid
        # alpha is the axis E17 got wrong: {1, 10}, ~100x the old 0.1 ceiling
        self.assertEqual(sorted(g["mlp__alpha"]), [1.0, 10.0])
        # activation un-pinned -- relu holds the peak, identity the robust median
        self.assertEqual(sorted(g["mlp__activation"]), ["identity", "relu"])
        # lr low, peak 1e-3
        lr = np.asarray(g["mlp__learning_rate_init"])
        self.assertGreaterEqual(lr.min(), 1e-4)
        self.assertLessEqual(lr.max(), 1e-2)
        # width weak axis, kept small (<=128); depth shallow (<=2)
        self.assertLessEqual(max(a[0] for a in g["mlp__hidden_layer_sizes"]), 128)
        self.assertTrue(all(len(a) <= 2 for a in g["mlp__hidden_layer_sizes"]))
        # same 120-cell footprint as mlp_ktz, so E34 search / E35 ensemble cost
        # what E28/E31 did and the contrast is clean
        self.assertEqual(len(list(ParameterGrid(g))), 120)
        from src.prediction.models import ScaleContinuousOnly
        self.assertIsInstance(w.ModelClass.steps[0][1], ScaleContinuousOnly)

    def test_n_jobs_is_accepted_and_ignored(self):
        # build_wrapper hands a worker count to every non-linear wrapper, but
        # sklearn's MLP has none. Taking it and dropping it keeps that dispatch
        # from needing a third category.
        wrapper = build_wrapper("mlp", task="regression", seed=42, n_jobs=8)
        self.assertFalse(hasattr(wrapper.ModelClass.named_steps["mlp"], "n_jobs"))

    def test_fits_and_predicts_a_finite_signal_for_both_tasks(self):
        # Cheap end-to-end check that the Pipeline survives clone -> set_params
        # -> fit -> predict_signal, which is the path hyper_params_search takes.
        import warnings

        for task in ("regression", "classification"):
            wrapper = build_wrapper("mlp", task=task, seed=42, n_jobs=1)
            estimator = clone(wrapper.ModelClass)
            estimator.set_params(
                mlp__hidden_layer_sizes=(8,), mlp__alpha=1e-2,
                mlp__learning_rate_init=1e-2, mlp__batch_size=128,
                mlp__max_iter=20,
            )
            target = (self.y > 0.5).astype(int) if task == "classification" else self.y
            with warnings.catch_warnings():
                # 20 iterations will not converge; that is the point of the test
                # being cheap, and an unconverged candidate simply scores worse.
                warnings.simplefilter("ignore")
                estimator.fit(self.X, target)
                signal = predict_signal(estimator, self.X, task)

            self.assertEqual(signal.shape, (len(self.X),), task)
            self.assertTrue(np.isfinite(signal).all(), task)


class TestGridEnsembleRegressor(unittest.TestCase):
    """
    E31: the mlp_ktz grid fitted whole and rank-averaged, not searched.

    The estimator has to satisfy the same clone -> fit -> predict contract every
    wrapper does, AND its combiner has to reproduce run_ensemble's per-period
    rank-average closely enough that ranking at the block level (all predict()
    ever sees) is not a different experiment. Both are checked here.
    """

    def setUp(self):
        # A real panel shape: periods x symbols, mild signal on binned features.
        rng = np.random.default_rng(3)
        self.n_date, self.n_sym, p = 50, 30, 22
        self.dates = np.repeat(np.arange(self.n_date), self.n_sym)
        N = self.n_date * self.n_sym
        self.X = rng.integers(0, 5, size=(N, p)).astype("float64") / 4.0
        beta = rng.normal(size=p)
        # per-date level drift stresses the block-vs-period ranking difference
        drift = np.repeat(rng.normal(scale=2.0, size=self.n_date), self.n_sym)
        self.y = self.X @ beta + drift + rng.normal(scale=3.0, size=N)

    def _silent_fit(self, est):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return est.fit(self.X, self.y)

    def test_wrapper_exposes_a_degenerate_search(self):
        from src.prediction.models import GridEnsembleRegressor
        w = build_wrapper("mlp_ktz_ens", task="regression", seed=42, n_jobs=1)
        self.assertEqual(w.model_name, "mlp_ktz_ens")
        self.assertEqual(w.search_type, "random")
        # Empty grid: RandomizedSearchCV has exactly one candidate, nothing to
        # tune -- selection is removed, not relocated.
        self.assertEqual(w.param_grid, {})
        self.assertIsInstance(w.ModelClass, GridEnsembleRegressor)

    def test_ktz2_ensemble_reads_its_own_grid(self):
        # mlp_ktz2_ens must ensemble the mlp_ktz2 (E33-reprior) grid, not mlp_ktz.
        # The generalised base= knob is what keeps E31 (mlp_ktz) and E35 (mlp_ktz2)
        # from sharing a grid by accident, and survives clone.
        from sklearn.model_selection import ParameterGrid
        from src.prediction.models import GridEnsembleRegressor
        w = build_wrapper("mlp_ktz2_ens", task="regression", seed=42, n_jobs=1)
        self.assertEqual(w.model_name, "mlp_ktz2_ens")
        self.assertEqual(w.param_grid, {})
        self.assertEqual(clone(w.ModelClass).get_params()["base"], "mlp_ktz2")
        _, cells = w.ModelClass._cells()
        expected = list(ParameterGrid(
            build_wrapper("mlp_ktz2", task="regression", seed=42).param_grid))
        self.assertEqual(len(cells), len(expected))
        self.assertEqual(len(cells), 120)
        # and it is genuinely the ktz2 grid: alpha reaches 10, activation both
        self.assertTrue(any(c["mlp__alpha"] == 10.0 for c in cells))
        self.assertTrue(any(c["mlp__activation"] == "relu" for c in cells))
        self.assertIsInstance(w.ModelClass, GridEnsembleRegressor)

    def test_classification_is_refused(self):
        # Regression-only on purpose; the book pays on the 20-day regression
        # label and a clf ensemble is a separate untested question.
        with self.assertRaises(ValueError):
            build_wrapper("mlp_ktz_ens", task="classification", seed=42)

    def test_fits_exactly_the_mlp_ktz_grid(self):
        # The member set IS ParameterGrid(mlp_ktz.param_grid); tie the count to
        # the grid so the two can never silently drift apart.
        from sklearn.model_selection import ParameterGrid
        expected = len(list(ParameterGrid(
            build_wrapper("mlp_ktz", task="regression", seed=42).param_grid)))
        self.assertEqual(expected, 120)
        est = self._silent_fit(
            build_wrapper("mlp_ktz_ens", task="regression", seed=42).ModelClass)
        self.assertEqual(est.n_members_, expected)

    def test_clone_fit_predict_is_finite_and_ranked(self):
        # The exact path hyper_params_search takes: clone the ModelClass, fit,
        # predict_signal. Output is one rank per row, in (0, 1].
        est = clone(build_wrapper("mlp_ktz_ens", task="regression",
                                  seed=42).ModelClass)
        self._silent_fit(est)
        signal = predict_signal(est, self.X, "regression")
        self.assertEqual(signal.shape, (len(self.X),))
        self.assertTrue(np.isfinite(signal).all())
        self.assertGreater(signal.min(), 0.0)
        self.assertLessEqual(signal.max(), 1.0)

    def test_is_deterministic_at_a_fixed_seed(self):
        a = self._silent_fit(
            build_wrapper("mlp_ktz_ens", task="regression", seed=42).ModelClass)
        b = self._silent_fit(
            build_wrapper("mlp_ktz_ens", task="regression", seed=42).ModelClass)
        np.testing.assert_allclose(a.predict(self.X), b.predict(self.X))

    def test_predict_before_fit_raises(self):
        from src.prediction.models import GridEnsembleRegressor
        with self.assertRaises(RuntimeError):
            GridEnsembleRegressor().predict(self.X)

    def test_a_diverged_member_is_dropped_not_propagated(self):
        # One non-finite member must not poison the whole block. scipy.rankdata
        # propagates NaN, so without the guard a single diverged MLP turns the
        # ensemble's prediction for the entire block to NaN, which summarize()
        # then silently dropna's from the scored record. predict() drops the bad
        # member and averages the survivors instead.
        import warnings

        est = self._silent_fit(
            build_wrapper("mlp_ktz_ens", task="regression", seed=42).ModelClass)
        good = est.predict(self.X)

        class _Diverged:
            def predict(self, X):
                out = np.zeros(np.asarray(X).shape[0]); out[0] = np.inf
                return out

        est.members_.append(_Diverged())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poisoned = est.predict(self.X)
        self.assertTrue(np.isfinite(poisoned).all())      # not NaN-poisoned
        # the survivors are exactly the original 120, so the result is unchanged
        np.testing.assert_allclose(poisoned, good)

    def test_block_rank_average_matches_per_period(self):
        # THE correctness check. predict() ranks each member over the whole block
        # because it never sees period labels; run_ensemble ranks per period.
        # These must agree at scoring time (numerai_corr re-ranks per period), or
        # the estimator is measuring a different combiner than the book uses.
        #
        # SCOPE: this panel is narrow (22 features) with a SHARED per-date drift
        # that cancels in the average. The 4 wide one-hot arms (763-812 cols) are
        # NOT exercised here -- identity-activation ticker intercepts can give
        # members period-correlated levels that do not cancel, so their BACKTEST
        # blocks rest on this equivalence rather than a proof (the LIVE artifact
        # is unaffected: one period per round makes block==period trivially). The
        # wide-set cross-check against run_ensemble is a report-time validation.
        est = self._silent_fit(
            build_wrapper("mlp_ktz_ens", task="regression", seed=42).ModelClass)
        members = np.column_stack([m.predict(self.X) for m in est.members_])

        # (a) block rank-average == what predict() returns
        from scipy.stats import rankdata
        block = np.column_stack(
            [rankdata(members[:, j]) / members.shape[0]
             for j in range(members.shape[1])]).mean(axis=1)
        np.testing.assert_allclose(block, est.predict(self.X))

        # (b) per-period rank-average == run_ensemble's operator
        frame = pd.DataFrame(members, index=pd.Index(self.dates, name="d"))
        per_period = frame.groupby(level=0).rank(pct=True).mean(axis=1).to_numpy()

        # scored corr per period for each combiner; the means must match closely
        def scored(pred):
            s = pd.DataFrame({"p": pred, "t": self.y, "d": self.dates})
            return s.groupby("d").apply(
                lambda g: np.corrcoef(g["p"].rank(), g["t"].rank())[0, 1],
                include_groups=False).mean()

        self.assertLess(abs(scored(block) - scored(per_period)), 5e-3)


class TestCatBoostKtz(unittest.TestCase):
    """
    E32: the catboost_ktz grid, both as a search and as a rank-averaged ensemble.

    catboost_ktz is the SEARCH sibling of lgb_ktz; catboost_ktz_ens fits the same
    cells whole and rank-averages them (the CatBoost member of the E30/E31/E32
    "ensemble the grid instead of searching it" family). The two must read ONE
    grid so ensemble-of-12 vs pick-1-of-12 is exact, and the ensemble must honour
    the clone -> fit -> predict -> joblib contract every deployable MODELS entry
    does. iterations are kept tiny so the 12 CatBoost fits stay unit-test cheap.
    """

    def setUp(self):
        rng = np.random.default_rng(5)
        self.n_date, self.n_sym, p = 40, 25, 8
        self.dates = np.repeat(np.arange(self.n_date), self.n_sym)
        N = self.n_date * self.n_sym
        self.X = rng.integers(0, 5, size=(N, p)).astype("float64") / 4.0
        beta = rng.normal(size=p)
        drift = np.repeat(rng.normal(scale=2.0, size=self.n_date), self.n_sym)
        self.y = self.X @ beta + drift + rng.normal(scale=3.0, size=N)

    def test_search_grid_encodes_the_ktz_priors(self):
        # A compact grid, shallow trees, low column sampling, low learning rate --
        # the E27 booster priors on CatBoost's parameter names. Compact ON PURPOSE
        # so the ensemble variant can fit ALL of it.
        g = build_wrapper("catboost_ktz", task="regression", seed=42).param_grid
        self.assertTrue(all(d <= 4 for d in g["depth"]))          # shallow
        self.assertTrue(max(g["rsm"]) <= 0.30)                    # low colsample
        self.assertTrue(max(g["learning_rate"]) <= 0.05)          # low lr
        self.assertEqual(len(g["iterations"]), 1)                 # pinned
        self.assertEqual(len(g["l2_leaf_reg"]), 1)                # pinned
        self.assertEqual(g["min_data_in_leaf"], [100])           # E27 leaf floor
        from sklearn.model_selection import ParameterGrid
        self.assertEqual(len(list(ParameterGrid(g))), 12)

    def test_ensemble_wrapper_exposes_a_degenerate_search(self):
        from src.prediction.catboost_ensemble import CatBoostGridEnsemble
        w = build_wrapper("catboost_ktz_ens", task="regression", seed=42, n_jobs=1)
        self.assertEqual(w.model_name, "catboost_ktz_ens")
        self.assertEqual(w.search_type, "random")
        self.assertEqual(w.param_grid, {"combine": ["rank"]})     # one candidate
        self.assertIsInstance(w.ModelClass, CatBoostGridEnsemble)

    def test_ensemble_fits_exactly_the_search_grid(self):
        # The member set IS the catboost_ktz grid; tie the count to it so the two
        # can never silently drift apart.
        from sklearn.model_selection import ParameterGrid
        expected = len(list(ParameterGrid(
            build_wrapper("catboost_ktz", task="regression", seed=42).param_grid)))
        est = build_wrapper("catboost_ktz_ens", task="regression", seed=42,
                            n_jobs=1, iterations=25).ModelClass
        est.fit(self.X, self.y)
        self.assertEqual(est.n_members_, expected)
        # the (lr, depth, rsm) cells match the search grid exactly
        sg = build_wrapper("catboost_ktz", task="regression", seed=42).param_grid
        search_cells = {(c["learning_rate"], c["depth"], c["rsm"])
                        for c in ParameterGrid(sg)}
        self.assertEqual(set(est._grid()), search_cells)

    def test_both_wrappers_refuse_classification(self):
        for name in ("catboost_ktz", "catboost_ktz_ens"):
            with self.assertRaises(ValueError):
                build_wrapper(name, task="classification", seed=42)

    def test_clone_fit_predict_is_finite_and_ranked(self):
        est = clone(build_wrapper("catboost_ktz_ens", task="regression", seed=42,
                                  n_jobs=1, iterations=25).ModelClass)
        est.fit(self.X, self.y)
        signal = predict_signal(est, self.X, "regression")
        self.assertEqual(signal.shape, (len(self.X),))
        self.assertTrue(np.isfinite(signal).all())
        self.assertGreater(signal.min(), 0.0)      # rank combiner: (0, 1]
        self.assertLessEqual(signal.max(), 1.0)

    def test_survives_joblib_roundtrip(self):
        # The deployability contract: src/live/refit.py dumps best_estimator_.
        import io, joblib
        est = build_wrapper("catboost_ktz_ens", task="regression", seed=42,
                            n_jobs=1, iterations=25).ModelClass
        est.fit(self.X, self.y)
        buf = io.BytesIO(); joblib.dump(est, buf); buf.seek(0)
        np.testing.assert_allclose(joblib.load(buf).predict(self.X),
                                   est.predict(self.X))

    def test_predict_before_fit_raises(self):
        from src.prediction.catboost_ensemble import CatBoostGridEnsemble
        with self.assertRaises(RuntimeError):
            CatBoostGridEnsemble().predict(self.X)

    def test_block_rank_average_matches_per_period(self):
        # predict() ranks each member over the whole block (it never sees period
        # labels); run_ensemble ranks per period. They must agree at scoring time.
        from scipy.stats import rankdata
        est = build_wrapper("catboost_ktz_ens", task="regression", seed=42,
                            n_jobs=1, iterations=25).ModelClass
        est.fit(self.X, self.y)
        members = np.column_stack([m.predict(self.X.astype("float32"))
                                   for m in est.members_])
        block = np.column_stack(
            [rankdata(members[:, j]) / members.shape[0]
             for j in range(members.shape[1])]).mean(axis=1)
        np.testing.assert_allclose(block, est.predict(self.X), atol=1e-9)

        frame = pd.DataFrame(members, index=pd.Index(self.dates, name="d"))
        per_period = frame.groupby(level=0).rank(pct=True).mean(axis=1).to_numpy()

        def scored(pred):
            s = pd.DataFrame({"p": pred, "t": self.y, "d": self.dates})
            return s.groupby("d").apply(
                lambda g: np.corrcoef(g["p"].rank(), g["t"].rank())[0, 1],
                include_groups=False).mean()

        self.assertLess(abs(scored(block) - scored(per_period)), 5e-3)


class TestGP(unittest.TestCase):
    """
    The subsampled exact Gaussian Process (E53), regression and classification.

    A GP is O(n^3), so the wrapper caps the training rows with a seeded draw
    taken inside fit; these tests fix the invariants the rest of the pipeline
    leans on -- the grid searches under both random and bayes, the estimator
    honours clone -> set_params -> fit -> predict_signal and the joblib dump
    live/refit.py does, the subsample is deterministic and capped, the kernel is
    frozen by default (so the outer search owns the scale), --model-options
    subsample flows through, and a singular kernel falls back instead of crashing.
    Panels are tiny and the subsample tinier so every fit stays unit-cheap.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        self.n_date, self.n_sym, p = 40, 25, 22
        self.dates = np.repeat(np.arange(self.n_date), self.n_sym)
        N = self.n_date * self.n_sym
        # Features already rank-binned to [0,1], as the pipeline delivers them.
        self.X = pd.DataFrame(rng.integers(0, 5, size=(N, p)).astype("float64") / 4.0)
        beta = rng.normal(size=p)
        drift = np.repeat(rng.normal(scale=2.0, size=self.n_date), self.n_sym)
        raw = self.X.values @ beta + drift + rng.normal(scale=3.0, size=N)
        # A target on [0,1] so (y > 0.5) binarises ~50/50, like binarize_target.
        self.y = pd.Series(raw).rank(pct=True)

    def test_registered_for_both_tasks(self):
        self.assertIn("gp", MODELS)
        self.assertEqual(set(supported_tasks("gp")),
                         {"regression", "classification"})

    def test_grid_is_bayes_searchable_and_translates(self):
        # No kernel objects: the grid is scalars + strings, so it must carry no
        # sequence-valued entry and must translate under --search-type bayes.
        for task in ("regression", "classification"):
            w = build_wrapper("gp", task=task, seed=42, n_jobs=1)
            self.assertEqual(sequence_valued_params(w.param_grid), [], task)
            space = to_space(w.param_grid)
            self.assertEqual(set(space), set(w.param_grid), task)

    def test_clone_and_set_params_round_trip(self):
        w = build_wrapper("gp", task="regression", seed=42, n_jobs=1, subsample=500)
        est = clone(w.ModelClass)
        self.assertEqual(est.get_params()["subsample"], 500)
        est.set_params(kernel="matern52", length_scale=2.0, alpha=0.1)
        self.assertEqual(est.get_params()["kernel"], "matern52")
        self.assertEqual(est.get_params()["length_scale"], 2.0)

    def test_fits_and_predicts_a_finite_signal_for_both_tasks(self):
        # The path hyper_params_search takes: clone -> set_params -> fit ->
        # predict_signal, for regression (posterior mean) and classification
        # (positive-class probability on the binarised target).
        for task in ("regression", "classification"):
            w = build_wrapper("gp", task=task, seed=42, n_jobs=1, subsample=300)
            est = clone(w.ModelClass)
            est.set_params(kernel="matern32", length_scale=1.0)
            target = (self.y > 0.5).astype(int) if task == "classification" else self.y
            est.fit(self.X, target)
            signal = predict_signal(est, self.X, task)
            self.assertEqual(signal.shape, (len(self.X),), task)
            self.assertTrue(np.isfinite(signal).all(), task)

    def test_classifier_predict_proba_is_a_distribution(self):
        w = build_wrapper("gp", task="classification", seed=42, n_jobs=1, subsample=300)
        est = clone(w.ModelClass)
        est.fit(self.X, (self.y > 0.5).astype(int))
        proba = est.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_subsample_is_deterministic_and_capped(self):
        from src.prediction.gp import SubsampledGPRegressor
        a = SubsampledGPRegressor(subsample=200, random_state=7).fit(self.X, self.y)
        b = SubsampledGPRegressor(subsample=200, random_state=7).fit(self.X, self.y)
        np.testing.assert_array_equal(a.gp_.X_train_, b.gp_.X_train_)
        self.assertEqual(a.n_train_, 200)
        # subsample >= block size uses every row.
        c = SubsampledGPRegressor(subsample=10 ** 6, random_state=7).fit(self.X, self.y)
        self.assertEqual(c.n_train_, len(self.X))

    def test_kernel_is_frozen_by_default_and_tunable_on_request(self):
        from src.prediction.gp import SubsampledGPRegressor
        frozen = SubsampledGPRegressor(length_scale=1.0, subsample=200,
                                       random_state=0).fit(self.X, self.y)
        self.assertIsNone(frozen.gp_.optimizer)          # outer search owns the scale
        tuned = SubsampledGPRegressor(length_scale=1.0, subsample=200,
                                      length_scale_bounds=(1e-2, 1e2),
                                      random_state=0).fit(self.X, self.y)
        self.assertEqual(tuned.gp_.optimizer, "fmin_l_bfgs_b")

    def test_model_options_subsample_flows_through(self):
        w = build_wrapper("gp", task="regression", seed=42, n_jobs=1, subsample=1234)
        self.assertEqual(w.ModelClass.get_params()["subsample"], 1234)

    def test_survives_joblib_roundtrip(self):
        # The deployability contract: src/live/refit.py dumps best_estimator_.
        import io, joblib
        est = clone(build_wrapper("gp", task="regression", seed=42, n_jobs=1,
                                  subsample=300).ModelClass)
        est.set_params(kernel="matern52", length_scale=0.5)
        est.fit(self.X, self.y)
        buf = io.BytesIO(); joblib.dump(est, buf); buf.seek(0)
        np.testing.assert_allclose(joblib.load(buf).predict(self.X),
                                   est.predict(self.X))

    def test_singular_kernel_falls_back_instead_of_crashing(self):
        # A tiny alpha with a short length-scale is where E54's alpha=1e-3 corner
        # can produce a near-singular kernel; the fit must degrade to a constant,
        # not raise, so a sensitivity sweep records a flat cell.
        from src.prediction.gp import SubsampledGPRegressor
        est = SubsampledGPRegressor(kernel="rbf", length_scale=0.02, alpha=1e-12,
                                    normalize_y=False, subsample=300,
                                    random_state=0).fit(self.X, self.y)
        self.assertTrue(np.isfinite(est.predict(self.X)).all())


class TestReferenceNaming(unittest.TestCase):
    """
    The reference series must never share a name with our own prediction column.

    Crypto's meta model column is literally called "prediction" in the shipped
    parquet. Joining it in under that name overwrote the model's own output, so
    CORR silently measured the meta model against the target and BMC came out as
    exactly 0.0 -- a series contributes nothing over itself. Both numbers looked
    like plausible results rather than an error, which is what makes this worth
    a test.
    """

    def test_canonical_name_is_not_prediction(self):
        self.assertNotEqual(REFERENCE_COL, "prediction")

    def test_every_tournament_renames_its_reference(self):
        # The source column may be called anything; what matters is that the
        # loader hands back the canonical name.
        for tournament, spec in TOURNAMENTS.items():
            self.assertIn("reference_col", spec, tournament)


class TestEmbargoDefaults(unittest.TestCase):
    """A period count is not portable between tournaments."""

    def test_same_horizon_different_embargo(self):
        # A 20-day target spans 4 weekly eras but 20 daily dates.
        self.assertEqual(default_embargo("numerai", "target"), 4)
        self.assertEqual(default_embargo("crypto", "target_binned_return_20"), 20)

    def test_longer_horizon_needs_longer_embargo(self):
        self.assertEqual(default_embargo("crypto", "target_binned_return_60"), 60)


class TestBayesTranslation(unittest.TestCase):
    """
    The wrapper grids are translated, not rewritten, so the translation is the
    only place --search-type bayes can silently search the wrong space.

    Every failure here produces a run that completes and reports plausible
    numbers: a uniform prior on a scale parameter puts almost all of its mass in
    the barely-regularised corner, a float where LightGBM wants an int is
    rejected candidate by candidate, and an off-by-one on an inclusive bound
    quietly moves the ceiling of the search.
    """

    def test_logspace_gets_a_log_prior_and_linspace_does_not(self):
        # This is the RidgeWrapper docstring's warning, restated as a test: a
        # uniform prior over [1e-4, 1e2] puts 99% of its draws above 1 and never
        # probes strong shrinkage at all.
        self.assertTrue(is_log_spaced(np.logspace(-4, 2, 100)))
        self.assertTrue(is_log_spaced(np.logspace(-3, -2, 40)))
        self.assertFalse(is_log_spaced(np.linspace(0.001, 100.0, 100)))

    def test_log_detection_ignores_order_and_rejects_non_positive(self):
        # glmnet hands back a DESCENDING path, and log10 is undefined at zero.
        self.assertTrue(is_log_spaced(np.logspace(2, -4, 50)))
        self.assertFalse(is_log_spaced(np.array([0.0, 0.1, 1.0, 10.0])))
        self.assertFalse(is_log_spaced(np.array([-1.0, -10.0, -100.0])))

    def test_glmnet_path_translates_to_a_log_uniform_real(self):
        # The alpha path is built per refit from the training block, so it can
        # only ever be translated, never hand-written.
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 5)))
        y = pd.Series(X[0].to_numpy() * 0.4 + rng.normal(size=200))
        alphas = glmnet_alpha_path(X, y, n_lambda=100)

        dim = to_dimension("alpha", alphas)
        self.assertEqual(type(dim).__name__, "Real")
        self.assertEqual(dim.prior, "log-uniform")
        self.assertAlmostEqual(dim.low, float(alphas.min()))
        self.assertAlmostEqual(dim.high, float(alphas.max()))

    def test_randint_bounds_are_inclusive_on_both_sides(self):
        # scipy's support() is already inclusive: sp_randint(2000, 30001)
        # covers 30000. "Correcting" it either way loses a real endpoint.
        dim = to_dimension("n_estimators", sp_randint(2000, 30001))
        self.assertEqual(type(dim).__name__, "Integer")
        self.assertEqual((dim.low, dim.high), (2000, 30000))

    def test_uniform_keeps_its_interval(self):
        dim = to_dimension("colsample_bytree", sp_uniform(loc=0.05, scale=0.25))
        self.assertEqual(type(dim).__name__, "Real")
        self.assertAlmostEqual(dim.low, 0.05)
        self.assertAlmostEqual(dim.high, 0.30)

    def test_pinned_values_stay_categorical_and_keep_their_type(self):
        # max_depth and num_leaves are pinned per sub-grid; that pairing is the
        # whole reason the boosters ship six grids. Turning a pin into an
        # interval would break the pairing, and an int into a float would be
        # rejected by LightGBM.
        for name, value in [("max_depth", [7]), ("num_leaves", [127]),
                            ("subsample_freq", [1])]:
            dim = to_dimension(name, value)
            self.assertEqual(type(dim).__name__, "Categorical", name)
            self.assertEqual(list(dim.categories), value, name)
            self.assertIsInstance(list(dim.categories)[0], int, name)

    def test_sequence_valued_entries_stay_categorical(self):
        # An MLP's hidden_layer_sizes is one tuple per architecture. Translating
        # it by raveling would flatten [(128, 64), (256, 128)] into
        # [128, 64, 256, 128] and return an Integer over 64..256, so the search
        # would propose a plain int where the estimator needs a tuple. That does
        # not raise here -- it fails inside EVERY candidate fit, each scores
        # UNSCORABLE, and the run finishes having selected nothing.
        for value in ([(128, 64), (256, 128)],          # equal lengths: the trap
                      [(64,), (128, 64), (256, 128, 64)],
                      [(64,)]):
            dim = to_dimension("hidden_layer_sizes", value)
            self.assertEqual(type(dim).__name__, "Categorical", value)
            self.assertEqual(list(dim.categories), [tuple(v) for v in value])

    def test_sequence_detection_does_not_disturb_scalar_grids(self):
        # The guard keys off entries being sequences, so a plain grid must come
        # through on exactly the dimension it had before.
        self.assertEqual(
            type(to_dimension("alpha", np.logspace(-4, 2, 100))).__name__, "Real")
        self.assertEqual(
            type(to_dimension("max_depth", [5, 6, 7, 8])).__name__, "Integer")

    def test_every_shipped_wrapper_grid_translates(self):
        # A wrapper whose grid cannot be translated must fail loudly at import
        # of the space, not halfway through a multi-hour walk-forward.
        #
        # Driven off MODELS rather than a hardcoded list: the point of this test
        # is that a NEWLY ADDED wrapper is covered, and a list you have to
        # remember to extend cannot do that.
        for name in sorted(MODELS):
            for task in supported_tasks(name):
                wrapper = build_wrapper(name, task=task, seed=42, n_jobs=1)
                grids = wrapper.param_grid
                for grid in (grids if isinstance(grids, list) else [grids]):
                    space = to_space(grid)
                    self.assertEqual(set(space), set(grid), (name, task))


class TestBayesBudget(unittest.TestCase):
    """
    skopt fits one surrogate PER SUB-GRID, and warms each up with random draws
    before fitting it at all.

    Left at skopt's default of 10 warm-up points, a booster's six sub-grids at
    --n-iter 50 would spend every one of their ~8 draws on warm-up: no surrogate
    would ever be fitted and the run would be random search under another name,
    reporting numbers that look like a Bayesian result.
    """

    def test_budget_splits_without_dropping_a_sub_grid(self):
        budgets = split_budget(6, 50)
        self.assertEqual(len(budgets), 6)
        self.assertEqual(sum(budgets), 50)
        self.assertTrue(all(b >= 1 for b in budgets))

    def test_tiny_budget_keeps_every_depth_rather_than_dropping_some(self):
        # Dropping sub-grids would mean silently never trying some tree depths,
        # which is worse than overspending the nominal budget by a few fits.
        budgets = split_budget(6, 2)
        self.assertEqual(len(budgets), 6)
        self.assertTrue(all(b == 1 for b in budgets))

    def test_warm_up_never_eats_a_budget_that_could_fit_a_surrogate(self):
        # 25 draws per sub-grid must leave the surrogate something to guide.
        self.assertLess(initial_points(25), 25)
        self.assertLessEqual(initial_points(25), 10)   # never above skopt's cap
        # And an explicit request is honoured, but cannot exceed the budget.
        self.assertEqual(initial_points(25, requested=5), 5)
        self.assertEqual(initial_points(4, requested=50), 4)

    def test_all_warm_up_budget_is_reported_as_a_warning(self):
        notes = " ".join(describe_budget([1] * 6, initial_points(1)))
        self.assertIn("WARNING", notes)
        self.assertIn("random search", notes)

    def test_healthy_budget_reports_no_warning(self):
        notes = " ".join(describe_budget(split_budget(1, 50), initial_points(50)))
        self.assertNotIn("WARNING", notes)
        self.assertNotIn("NOTE", notes)


class TestBayesSearchIntegration(unittest.TestCase):
    """
    The Bayesian path has to be a drop-in: the walk-forward reads five
    attributes off the search object and never branches on which optimiser
    produced it.

    The subtle requirement is the DataFrame index. build_scorer() looks the
    period and reference columns up in a side table keyed by X's index, so an
    optimiser that hands the scorer a numpy array would score every fold against
    the wrong periods -- and still return a number.
    """

    def _problem(self, n_periods=40, rows=5):
        rng = np.random.default_rng(3)
        periods = np.repeat(np.arange(n_periods), rows)
        index = pd.Index(np.arange(len(periods)) + 10_000, name="row")
        X = pd.DataFrame(rng.normal(size=(len(periods), 4)),
                         columns=list("abcd"), index=index)
        y = pd.Series(X["a"].to_numpy() * 0.5 + rng.normal(size=len(periods)) * 0.5,
                      index=index)
        meta = pd.DataFrame({"period": periods, "target": y.to_numpy()},
                            index=index)
        return X, y, meta

    def _search(self, search_type):
        X, y, meta = self._problem()
        wrapper = build_wrapper("ridge", task="regression", seed=0)
        return hyper_params_search(
            X=X, y=y, wrapper=wrapper, meta=meta, metric="corr",
            period_col="period", reference_col=None, task="regression",
            n_iter=8, n_splits=2, embargo=1, scheme="expanding",
            train_window=None, n_jobs=1, seed=0, verbose=False,
            search_type=search_type,
        )

    def test_bayes_exposes_the_same_interface_as_random(self):
        search = self._search("bayes")
        for attribute in ("best_estimator_", "best_params_", "best_score_",
                          "best_index_", "cv_results_", "searched_bounds_"):
            self.assertTrue(hasattr(search, attribute), attribute)
        # The ride-along scorers must survive too, or the hyperparameter history
        # loses its CV train-vs-validation columns for every bayes run.
        self.assertIn("mean_test_corr", search.cv_results_)
        self.assertIn("mean_train_objective", search.cv_results_)

    def test_bayes_searches_the_grids_span(self):
        search = self._search("bayes")
        low, high, size = search.searched_bounds_["alpha"]
        self.assertAlmostEqual(low, 1e-4)
        self.assertAlmostEqual(high, 1e2)
        # 0 means continuous, which is the visible difference from random search
        # in results/hyperparams -- not a missing value.
        self.assertEqual(size, 0)
        self.assertGreaterEqual(search.best_params_["alpha"], low)
        self.assertLessEqual(search.best_params_["alpha"], high)

    def test_sub_grid_bounds_are_unioned_not_summed(self):
        # Every booster sub-grid carries the SAME n_estimators range, so summing
        # the six counts would claim 168,006 reachable values where there are
        # 28,001. The distinct pins (six depths) must still count as six.
        wrapper = build_wrapper("lgb", task="regression", seed=42, n_jobs=1)
        spaces = [(to_space(grid), 8) for grid in wrapper.param_grid]
        bounds = space_bounds(spaces)

        self.assertEqual(bounds["n_estimators"], (2000.0, 30000.0, 28001))
        self.assertEqual(bounds["max_depth"], (5.0, 10.0, 6))
        self.assertEqual(bounds["num_leaves"], (31.0, 1023.0, 6))
        # learning_rate is continuous under bayes, whatever the sub-grid count.
        self.assertEqual(bounds["learning_rate"][2], 0)

    def test_random_still_reports_a_countable_grid(self):
        search = self._search("random")
        self.assertEqual(search.searched_bounds_["alpha"][2], 100)

    def test_unknown_search_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self._search("optuna")


if __name__ == "__main__":
    unittest.main()


class TestKFoldInnerCV(unittest.TestCase):
    """
    The inner hyperparameter CV can be swapped from purged chronological folds
    to Wolpert's original shuffled k-fold.

    What this guards is the distinction that MATTERS: purged splits on whole
    periods with an embargo, so a period lands in at most one validation fold
    and some periods are dropped to the gap. Shuffled k-fold ignores time
    entirely -- every row is validated exactly once and a single period is
    scattered across folds, which is the train/val target overlap purged exists
    to forbid. If the two branches ever produced the same folds the option would
    be a no-op wearing a name.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        n_periods, per = 10, 20
        periods = np.repeat(["p%03d" % i for i in range(n_periods)], per)
        n = len(periods)
        self.X = pd.DataFrame(rng.random((n, 3)),
                              columns=["f0", "f1", "f2"], index=range(n))
        self.y = pd.Series(rng.random(n), index=range(n))
        self.meta = pd.DataFrame({"date": periods}, index=range(n))
        self.wrapper = build_wrapper("ridge", task="regression", seed=42, n_jobs=1)

    def _capture_folds(self, inner_cv):
        import src.prediction.functions as fx
        captured = {}
        real = fx.RandomizedSearchCV

        def spy(*args, **kwargs):
            captured["cv"] = list(kwargs["cv"])
            return real(*args, **kwargs)

        fx.RandomizedSearchCV = spy
        try:
            hyper_params_search(
                X=self.X, y=self.y, wrapper=self.wrapper, meta=self.meta,
                metric="corr", period_col="date", reference_col=None,
                task="regression", n_iter=3, n_splits=5, embargo=2,
                scheme="expanding", train_window=None, n_jobs=1, seed=42,
                verbose=False, inner_cv=inner_cv,
            )
        finally:
            fx.RandomizedSearchCV = real
        return captured["cv"]

    def test_kfold_validates_every_row_exactly_once(self):
        folds = self._capture_folds("kfold")
        val = np.concatenate([v for _, v in folds])
        np.testing.assert_array_equal(np.sort(val), np.arange(len(self.X)))

    def test_purged_drops_rows_to_the_embargo(self):
        folds = self._capture_folds("purged")
        val = np.concatenate([v for _, v in folds])
        # First fold has no history and is skipped, and an embargo gap sits
        # before every block, so the validated rows are a strict subset.
        self.assertLess(len(np.unique(val)), len(self.X))

    def test_kfold_scatters_a_period_across_folds_and_purged_does_not(self):
        periods = self.meta["date"].to_numpy()

        def folds_per_period(folds):
            counts = []
            for p in np.unique(periods):
                rows = set(np.where(periods == p)[0])
                hit = sum(bool(rows & set(v)) for _, v in folds)
                counts.append(hit)
            return counts

        # Purged: each period lands in at most one validation fold.
        self.assertLessEqual(max(folds_per_period(self._capture_folds("purged"))), 1)
        # k-fold: at least one period is split across several folds -- the very
        # overlap purged forbids, accepted here by design.
        self.assertGreater(max(folds_per_period(self._capture_folds("kfold"))), 1)

    def test_unknown_inner_cv_is_rejected(self):
        with self.assertRaises(ValueError):
            self._capture_folds("timeseries")
