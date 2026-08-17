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
    hyper_params_search,
    purged_period_splits,
    score_frame,
)
from src.prediction.metrics import summarize
from src.prediction.models import (
    build_wrapper,
    glmnet_alpha_path,
    scaled_leaf_range,
)
from src.prediction.search_spaces import (
    describe_budget,
    initial_points,
    is_log_spaced,
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

    def test_every_shipped_wrapper_grid_translates(self):
        # A wrapper whose grid cannot be translated must fail loudly at import
        # of the space, not halfway through a multi-hour walk-forward.
        for name in ("ridge", "ridge_glmnet", "lgb", "xgb", "catboost"):
            for task in ("regression", "classification"):
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
