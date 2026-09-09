"""
Tests for prediction.generalised_ridge and its wrapper.

Run with:  python3 -m unittest src.tests.test_generalised_ridge -v

The load-bearing test is the first one: GRR reduces EXACTLY to sklearn's Ridge
when W = I, b0 = 0 and Delta = lambda*I. Everything else in the module is a
generalisation of that point, so if the point is wrong the generalisations are
decoration.

One fixture detail that is not optional: the frames are float64. The parquets on
disk are float32 and sklearn's Ridge PRESERVES float32, so on raw dtypes the
comparison agrees only to about 2e-6 and the test would fail for a reason that
has nothing to do with the arithmetic.
"""

import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.prediction.generalised_ridge import (
    GeneralisedRidge,
    feature_groups,
    parse_group_spec,
    solve_box_constrained,
    stem_index,
    stem_of,
)
from src.prediction.models import MODELS, build_wrapper, supported_tasks
from src.prediction.search_spaces import sequence_valued_params, to_space

#: The real v2.0 feature names. Hard-coded so a dataset rebuild cannot quietly
#: change what this test asserts.
RAW_22 = [
    "feature_bollinger_20d", "feature_bollinger_60d",
    "feature_close_avg_20d", "feature_close_avg_60d",
    "feature_close_ewa_20d", "feature_close_ewa_60d",
    "feature_market_cap_avg_20d", "feature_market_cap_avg_60d",
    "feature_market_cap_ewa_20d", "feature_market_cap_ewa_60d",
    "feature_momentum_20d", "feature_momentum_60d",
    "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
    "feature_sharpe_ratio_20d", "feature_sharpe_ratio_60d",
    "feature_volatility_20d", "feature_volatility_60d",
    "feature_volume_avg_20d", "feature_volume_avg_60d",
    "feature_volume_ewa_20d", "feature_volume_ewa_60d",
]

#: The tournament's five bins and their published mass, used wherever a test
#: needs a realistic label rather than a gaussian.
BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
MASS = np.array([0.05, 0.20, 0.50, 0.20, 0.05])


def frame(n=4000, p=22, seed=3, names=None, dtype="float64"):
    """A DataFrame fixture with real-looking column names."""
    rng = np.random.default_rng(seed)
    names = names or RAW_22[:p]
    X = pd.DataFrame(rng.normal(size=(n, len(names))).astype(dtype), columns=names)
    y = pd.Series((X.to_numpy() @ rng.normal(size=len(names))
                   + rng.normal(size=n)).astype(dtype))
    return X, y


class TestGeneralisedRidge(unittest.TestCase):

    def test_reduces_to_sklearn_ridge_exactly(self):
        # The whole claim. W = I, b0 = 0, Delta = alpha*I must BE ridge.
        X, y = frame()
        for fit_intercept in (True, False):
            for alpha in (1e-4, 1.0, 137.0, 1e5):
                mine = GeneralisedRidge(
                    alpha=alpha, alpha_scale="absolute", structure="identity",
                    w_scheme="uniform", b0_scheme="zero",
                    fit_intercept=fit_intercept).fit(X, y)
                theirs = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
                np.testing.assert_allclose(
                    mine.coef_, theirs.coef_, rtol=1e-10, atol=1e-12,
                    err_msg="alpha=%g intercept=%s" % (alpha, fit_intercept))
                self.assertAlmostEqual(mine.intercept_, theirs.intercept_, places=10)

    def test_W_matches_sklearn_sample_weight(self):
        # W is not a bespoke convention: it is exactly sklearn's sample_weight,
        # so the weighted case has a reference implementation too.
        X, y = frame(n=1500, p=8)
        rng = np.random.default_rng(11)
        w = rng.uniform(0.2, 3.0, size=len(y))

        class Fixed(GeneralisedRidge):
            def _row_weights(self, target, period=None):
                return w

        mine = Fixed(alpha=2.0, alpha_scale="absolute").fit(X, y)
        theirs = Ridge(alpha=2.0).fit(X, y, sample_weight=w)
        np.testing.assert_allclose(mine.coef_, theirs.coef_, rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(mine.intercept_, theirs.intercept_, places=9)

    def test_unweighted_centring_would_be_wrong_under_weights(self):
        # Guards the subtle half of the intercept: the centring must use the
        # WEIGHTED means. With recency-like weights, unweighted centring is off
        # by ~5e-4, which is far above anything else in this file.
        X, y = frame(n=1200, p=6)
        w = np.exp(np.linspace(0.0, 2.0, len(y)))       # correlated with row order

        class Fixed(GeneralisedRidge):
            def _row_weights(self, target, period=None):
                return w

        good = Fixed(alpha=3.0, alpha_scale="absolute").fit(X, y).coef_
        reference = Ridge(alpha=3.0).fit(X, y, sample_weight=w).coef_
        np.testing.assert_allclose(good, reference, rtol=1e-9)

        A = X.to_numpy()
        Ac, yc = A - A.mean(0), y.to_numpy() - y.to_numpy().mean()   # UNweighted
        bad = np.linalg.solve(Ac.T @ (w[:, None] * Ac) + 3.0 * np.eye(A.shape[1]),
                              Ac.T @ (w * yc))
        self.assertGreater(np.abs(bad - reference).max(), 1e-5)

    def test_group_at_zero_penalty_is_the_identity_case(self):
        X, y = frame()
        for fit_intercept in (True, False):
            fused = GeneralisedRidge(alpha=0.5, structure="group",
                                     group_penalty=0.0,
                                     fit_intercept=fit_intercept).fit(X, y)
            plain = GeneralisedRidge(alpha=0.5, structure="identity",
                                     fit_intercept=fit_intercept).fit(X, y)
            np.testing.assert_allclose(fused.coef_, plain.coef_, rtol=1e-12)

    def test_alpha_is_a_ratio_to_the_gram_scale(self):
        X, y = frame()
        A = X.to_numpy()
        Ac = A - A.mean(0)
        scale = np.trace(Ac.T @ Ac) / A.shape[1]
        mine = GeneralisedRidge(alpha=0.03, alpha_scale="gram").fit(X, y)
        theirs = Ridge(alpha=0.03 * scale).fit(X, y)
        np.testing.assert_allclose(mine.coef_, theirs.coef_, rtol=1e-10)

        # The property that licenses a STATIC grid: doubling the rows leaves a
        # gram-scaled alpha's solution alone, and moves an absolute one.
        X2 = pd.concat([X, X], ignore_index=True)
        y2 = pd.concat([y, y], ignore_index=True)
        ratio = GeneralisedRidge(alpha=0.03, alpha_scale="gram").fit(X2, y2)
        np.testing.assert_allclose(mine.coef_, ratio.coef_, rtol=1e-8)
        absolute_1 = GeneralisedRidge(alpha=50.0, alpha_scale="absolute").fit(X, y)
        absolute_2 = GeneralisedRidge(alpha=50.0, alpha_scale="absolute").fit(X2, y2)
        self.assertGreater(np.abs(absolute_1.coef_ - absolute_2.coef_).max(), 1e-5)

    def test_delta_stays_positive_definite_across_the_whole_grid(self):
        # A fixture with a dead column and two near-duplicates: the corners of
        # the grid must still produce a finite, solvable system.
        X, y = frame(n=800)
        X = X.copy()
        X["feature_bollinger_20d"] = 0.7                       # constant
        X["feature_close_ewa_60d"] = X["feature_close_ewa_20d"] + 1e-9
        for alpha in (1e-4, 1.0, 1e2):
            for rho in (1e-2, 1.0, 1e2, 1e4):
                model = GeneralisedRidge(alpha=alpha, structure="group",
                                         group_penalty=rho).fit(X, y)
                self.assertTrue(np.isfinite(model.coef_).all(),
                                "alpha=%g rho=%g" % (alpha, rho))
                self.assertTrue(np.isfinite(model.predict(X)).all())

    def test_a_dead_column_is_dropped_from_its_group_not_zero_scaled(self):
        # Leaving a constant column in its group drags the group mean toward a
        # coefficient nobody is estimating, and roughly halves the survivors.
        # A delisted token's column going constant on a 130-date block is the
        # case this actually protects.
        X, y = frame(n=1500)
        dead_name = "feature_close_ewa_60d"                  # a real group member
        with_dead = X.copy()
        with_dead[dead_name] = 0.25                          # constant
        without = X.drop(columns=[dead_name])

        fused_dead = GeneralisedRidge(alpha=0.5, structure="group",
                                      group_penalty=10.0).fit(with_dead, y)
        fused_drop = GeneralisedRidge(alpha=0.5, structure="group",
                                      group_penalty=10.0).fit(without, y)

        # The group is reported at its LIVE size, and the dead member is inert:
        # after centring its Gram row is zero, so it decouples from the system
        # and the survivors are exactly what a fit without it produces.
        self.assertEqual(fused_dead.groups_["feature_close"], 3)
        keep = [with_dead.columns.get_loc(c) for c in without.columns]
        np.testing.assert_allclose(fused_dead.coef_[keep], fused_drop.coef_,
                                   rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(fused_dead.coef_[with_dead.columns.get_loc(dead_name)],
                               0.0, places=12)

        # And the thing being guarded against: had the dead column been kept in
        # the group at scale zero, it would pull the group mean toward zero.
        # Reproduce that variant explicitly and show it moves the survivors.
        s = np.ones(4)
        s[3] = 0.0                                            # the zero-scaled way
        block_wrong = np.diag(s * s) - np.outer(s, s) / 4      # divides by 4, not 3
        live_s = np.ones(3)
        block_right = np.diag(live_s * live_s) - np.outer(live_s, live_s) / 3
        self.assertNotAlmostEqual(float(block_wrong[0, 0]), float(block_right[0, 0]))

    def test_group_penalty_spans_no_op_to_fused(self):
        X, y = frame(n=3000)
        plain = GeneralisedRidge(alpha=0.5, structure="identity").fit(X, y)
        groups = feature_groups(list(X.columns))

        def within_group_spread(coef):
            return float(np.mean([coef[idx].std() for idx in groups.values()]))

        base = within_group_spread(plain.coef_)
        soft = GeneralisedRidge(alpha=0.5, structure="group",
                                group_penalty=1e-2).fit(X, y)
        hard = GeneralisedRidge(alpha=0.5, structure="group",
                                group_penalty=1e4).fit(X, y)
        self.assertGreater(within_group_spread(soft.coef_) / base, 0.95)
        self.assertLess(within_group_spread(hard.coef_) / base, 0.05)

    def test_group_structure_raises_rather_than_degrading_to_ridge(self):
        X, y = frame(n=400, p=6)
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge(structure="group").fit(X.to_numpy(), y)
        self.assertIn("column NAMES", str(caught.exception))

        anonymous = pd.DataFrame(X.to_numpy(), columns=["f%d" % i for i in range(6)])
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge(structure="group").fit(anonymous, y)
        self.assertIn("no multi-column feature group", str(caught.exception))

    def test_stem_rule_partitions_the_raw_22(self):
        groups = feature_groups(RAW_22)
        sizes = {stem: len(idx) for stem, idx in groups.items()}
        self.assertEqual(sizes, {
            "feature_close": 4, "feature_market_cap": 4, "feature_volume": 4,
            "feature_bollinger": 2, "feature_momentum": 2,
            "feature_relative_strength_index": 2, "feature_sharpe_ratio": 2,
            "feature_volatility": 2})
        self.assertEqual(sum(sizes.values()), 22)          # zero singletons

    def test_stem_rule_never_cliques_a_ticker_one_hot_or_a_code(self):
        # The non-ASCII ticker is the regression case: a [A-Za-z0-9]+ character
        # class files it as a raw core feature and nothing fails loudly.
        strays = ["feature_sym_BTC", "feature_sym_币安人生", "feature_sym_tenc",
                  "feature_sym_emb_3", "feature_ae_07",
                  "feature_close_avg_20d_symmean", "feature_market_cap_avg_spread",
                  "feature_close_avg_20d_x_feature_volume_avg_20d"]
        groups = feature_groups(RAW_22 + strays)
        cliqued = {(RAW_22 + strays)[i] for idx in groups.values() for i in idx}
        for name in strays:
            self.assertNotIn(name, cliqued, name)

    def test_lag_and_difference_axes_join_their_parent_group(self):
        names = ["feature_volatility_20d", "feature_volatility_60d",
                 "feature_volatility_20d_diff1", "feature_volatility_20d_diff21",
                 "feature_volatility_20d_lag3", "feature_volatility_20d_dlag20",
                 "feature_volatility_20d_arlag40"]
        self.assertEqual(len(feature_groups(names)), 1)
        self.assertEqual(len(feature_groups(names)["feature_volatility"]), 7)

    def test_tail_weights_are_bounded_and_average_one(self):
        rng = np.random.default_rng(5)
        y = rng.choice(BINS, size=20000, p=MASS)
        for tau in (0.0, 0.25, 0.5, 0.8):
            w = GeneralisedRidge(w_scheme="tail", tail_weight=tau)._row_weights(y)
            if tau == 0.0:
                self.assertIsNone(w)       # exactly the reduction case
                continue
            self.assertGreaterEqual(w.min(), 0.19)
            # mean(w) == 1 by construction but only in exact arithmetic.
            self.assertAlmostEqual(float(w.mean()), 1.0, places=12)

    def test_zero_prior_shrinkage_is_the_zero_prior_under_every_scheme(self):
        X, y = frame(n=1200)
        plain = GeneralisedRidge(alpha=0.5).fit(X, y)
        for scheme in ("marginal", "group_mean"):
            model = GeneralisedRidge(alpha=0.5, b0_scheme=scheme,
                                     prior_shrinkage=0.0).fit(X, y)
            np.testing.assert_allclose(model.coef_, plain.coef_, rtol=1e-12,
                                       err_msg=scheme)

    def test_marginal_prior_only_rescales_when_columns_share_a_scale(self):
        # Why b0 is off by default: on equal-variance columns -- which is what
        # the rank-binned raw 22 are -- a marginal prior moves the coefficient
        # vector's LENGTH and not its DIRECTION, and numerai_corr ranks the
        # prediction, so it cannot change the score.
        def cosine(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        # Columns standardised to EXACTLY equal variance, which is what the raw
        # 22 effectively are (measured sd spread 1.011x). Then g is constant,
        # Delta @ b0 = alpha*kappa*c, the right-hand side collapses to
        # (1 + alpha*kappa) * c and the left-hand side is kappa-free -- so
        # beta(kappa) is a positive scalar multiple of beta(0), exactly.
        X, y = frame(n=2000)
        X = (X - X.mean()) / X.std(ddof=0)
        base = GeneralisedRidge(alpha=1.0).fit(X, y).coef_
        moved = GeneralisedRidge(alpha=1.0, b0_scheme="marginal",
                                 prior_shrinkage=1.0).fit(X, y).coef_
        self.assertGreater(cosine(base, moved), 1.0 - 1e-12)
        # The multiple is 1 + alpha*kappa = 2, and it is the SAME for every
        # coefficient -- which is why a rank metric cannot see it.
        np.testing.assert_allclose(moved / base, 2.0, rtol=1e-9)

        spread = X * np.linspace(1.0, 40.0, X.shape[1])       # 40x sd spread
        base = GeneralisedRidge(alpha=1.0).fit(spread, y).coef_
        moved = GeneralisedRidge(alpha=1.0, b0_scheme="marginal",
                                 prior_shrinkage=1.0).fit(spread, y).coef_
        self.assertLess(cosine(base, moved), 0.9)

    def test_predict_requires_the_fitted_columns(self):
        X, y = frame(n=500, p=8)
        model = GeneralisedRidge(alpha=1.0).fit(X, y)
        np.testing.assert_allclose(model.predict(X.to_numpy()), model.predict(X))
        with self.assertRaises(ValueError):
            model.predict(X[list(reversed(X.columns))])
        with self.assertRaises(ValueError):
            model.predict(X.iloc[:, :4])

    def test_predictions_stay_finite_and_non_constant_at_the_grid_corners(self):
        # A constant prediction makes the period's corr NaN and scores the
        # candidate UNSCORABLE, so the corners must not produce one.
        X, y = frame(n=900)
        X = X.copy()
        X["feature_momentum_60d"] = 1.0
        for alpha, rho in ((1e-4, 1e4), (1e2, 1e-2)):
            pred = GeneralisedRidge(alpha=alpha, structure="group",
                                    group_penalty=rho).fit(X, y).predict(X)
            self.assertTrue(np.isfinite(pred).all(), (alpha, rho))
            self.assertGreater(pred.std(), 0.0, (alpha, rho))

    def test_every_column_constant_raises(self):
        X, y = frame(n=300, p=5)
        dead = pd.DataFrame(np.ones((300, 5)), columns=X.columns)
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge().fit(dead, y)
        self.assertIn("constant", str(caught.exception))

    def test_estimator_stores_data_not_behaviour_and_survives_joblib(self):
        # A closure or a lambda would survive every candidate fit under loky and
        # explode only at joblib.dump in the live refit, which run_round
        # downgrades to "refit failed, using the cached fit".
        X, y = frame(n=600, p=8)
        model = GeneralisedRidge(alpha=0.7, structure="group",
                                 group_penalty=5.0).fit(X, y)
        self.assertFalse([k for k, v in vars(model).items() if callable(v)])
        path = "/tmp/_gen_ridge_roundtrip.joblib"
        joblib.dump(model, path, compress=3)
        reloaded = joblib.load(path)
        np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))

    def test_clone_rebuilds_every_parameter(self):
        from sklearn.base import clone
        model = GeneralisedRidge(alpha=0.3, structure="group", group_penalty=7.0,
                                 w_scheme="tail", tail_weight=0.4,
                                 b0_scheme="marginal", prior_shrinkage=0.9)
        self.assertEqual(clone(model).get_params(), model.get_params())


class TestGenRidgeWrapper(unittest.TestCase):

    def test_wrapper_exposes_the_contract(self):
        from sklearn.pipeline import Pipeline
        wrapper = build_wrapper("gen_ridge", task="regression", seed=42, n_jobs=8)
        self.assertEqual(wrapper.model_name, "gen_ridge")
        self.assertEqual(wrapper.search_type, "random")
        self.assertIsInstance(wrapper.ModelClass, GeneralisedRidge)
        self.assertNotIsInstance(wrapper.ModelClass, Pipeline)
        self.assertIsInstance(wrapper.param_grid, dict)
        self.assertIn("gen_ridge", MODELS)
        self.assertEqual(supported_tasks("gen_ridge"), ("regression",))
        with self.assertRaises(ValueError) as caught:
            build_wrapper("gen_ridge", task="classification", seed=42, n_jobs=1)
        self.assertIn("does not implement task", str(caught.exception))

    def test_constructible_from_defaults_alone(self):
        # run_forecast.check_args pre-flights the bayes path with no seed and no
        # options, so this must not need either.
        self.assertIsInstance(MODELS["gen_ridge"]().param_grid, dict)

    def test_model_options_restrict_the_searched_families(self):
        grid = build_wrapper("gen_ridge", task="regression",
                             structures=["identity"]).param_grid
        self.assertEqual(grid["structure"], ["identity"])
        self.assertNotIn("group_penalty", grid)

        grid = build_wrapper("gen_ridge", task="regression",
                             structures=["group"]).param_grid
        self.assertEqual(grid["structure"], ["group"])
        self.assertIn("group_penalty", grid)

        grid = build_wrapper("gen_ridge", task="regression",
                             w_schemes=["uniform", "tail"]).param_grid
        self.assertIn("w_scheme", grid)
        self.assertIn("tail_weight", grid)

        grid = build_wrapper("gen_ridge", task="regression",
                             b0_schemes=["zero", "group_mean"]).param_grid
        self.assertIn("b0_scheme", grid)
        self.assertIn("prior_shrinkage", grid)

        # The defaults carry no dead column into the hyperparams CSV.
        default = build_wrapper("gen_ridge", task="regression").param_grid
        self.assertEqual(sorted(default), ["alpha", "group_penalty", "structure"])

    def test_an_unknown_family_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as caught:
            build_wrapper("gen_ridge", task="regression", structures=["fused"])
        self.assertIn("fused", str(caught.exception))
        with self.assertRaises(ValueError):
            build_wrapper("gen_ridge", task="regression", structures=[])

    def test_every_option_combination_stays_bayes_searchable(self):
        # The MODELS-driven tests build every wrapper from DEFAULTS only, so the
        # non-default modes -- where every extra grid entry lives -- are
        # otherwise never translated.
        import itertools
        combos = itertools.product(
            [["identity"], ["group"], ["identity", "group"]],
            [["uniform"], ["uniform", "tail"]],
            [["zero"], ["zero", "marginal"], ["zero", "marginal", "group_mean"]])
        for structures, w_schemes, b0_schemes in combos:
            grid = build_wrapper("gen_ridge", task="regression",
                                 structures=structures, w_schemes=w_schemes,
                                 b0_schemes=b0_schemes).param_grid
            self.assertEqual(sequence_valued_params(grid), [],
                             (structures, w_schemes, b0_schemes))
            self.assertEqual(set(to_space(grid)), set(grid),
                             (structures, w_schemes, b0_schemes))

    def test_the_grid_gives_each_structure_a_real_share_of_the_budget(self):
        # Pins the single-dict decision: a LIST of sub-grids allocates draws by
        # cardinality, which starves an arm without saying so.
        from sklearn.model_selection import ParameterSampler
        grid = build_wrapper("gen_ridge", task="regression").param_grid
        for seed in (0, 1, 2):
            draws = list(ParameterSampler(grid, n_iter=50, random_state=seed))
            counts = {s: sum(d["structure"] == s for d in draws)
                      for s in ("identity", "group")}
            self.assertGreaterEqual(min(counts.values()), 15, (seed, counts))


class TestExponentialWeighting(unittest.TestCase):
    """Hypothesis 1: exponential decay in time, on a PANEL."""

    @staticmethod
    def panel(n_dates=60, per_date=25):
        X, y = frame(n=n_dates * per_date)
        period = np.repeat(
            pd.date_range("2024-01-01", periods=n_dates).to_numpy(), per_date)
        return X, y, period

    def test_weight_is_constant_within_a_date(self):
        # The panel carries ~300 symbols per date. A weight derived from ROW
        # POSITION would give them 300 different weights -- a cross-sectional
        # reweighting nobody asked for.
        X, y, period = self.panel()
        w = GeneralisedRidge(w_scheme="exp_decay",
                             half_life=20.0)._row_weights(y.to_numpy(), period)
        for date in np.unique(period):
            self.assertEqual(len(np.unique(np.round(w[period == date], 12))), 1)

    def test_weight_halves_every_half_life_and_averages_one(self):
        X, y, period = self.panel(n_dates=41, per_date=3)
        w = GeneralisedRidge(w_scheme="exp_decay",
                             half_life=10.0)._row_weights(y.to_numpy(), period)
        newest = w[period == period.max()][0]
        older = w[period == np.unique(period)[-11]][0]     # ten dates earlier
        self.assertAlmostEqual(older / newest, 0.5, places=10)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=12)

    def test_exp_decay_without_a_period_raises_rather_than_falling_back(self):
        # Falling back to uniform would make a live slot silently drop the very
        # thing it was deployed for.
        X, y, _ = self.panel()
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge(w_scheme="exp_decay", half_life=20.0).fit(X, y)
        self.assertIn("period", str(caught.exception))

    def test_period_must_be_positional(self):
        X, y, period = self.panel()
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge(w_scheme="exp_decay",
                             half_life=20.0).fit(X, y, period=period[:-5])
        self.assertIn("positional", str(caught.exception))

    def test_a_very_long_half_life_approaches_the_uniform_fit(self):
        X, y, period = self.panel()
        flat = GeneralisedRidge(alpha=1.0).fit(X, y).coef_
        slow = GeneralisedRidge(alpha=1.0, w_scheme="exp_decay",
                                half_life=1e6).fit(X, y, period=period).coef_
        # Compared against the coefficient VECTOR's norm, not elementwise: a
        # near-zero coefficient fails any relative tolerance for a reason that
        # is not the weighting. The weight spread here is 4e-05.
        self.assertLess(np.abs(slow - flat).max() / np.linalg.norm(flat), 1e-5)
        cosine = float(slow @ flat / (np.linalg.norm(slow) * np.linalg.norm(flat)))
        self.assertGreater(cosine, 1.0 - 1e-12)

    def test_a_uniform_rescale_of_the_weights_is_absorbed_under_gram(self):
        # This is what makes per-fold slicing safe: an inner CV fold whose rows
        # are all old receives uniformly tiny weights, and the fit is unchanged.
        X, y, period = self.panel()
        w = GeneralisedRidge(w_scheme="exp_decay",
                             half_life=15.0)._row_weights(y.to_numpy(), period)

        class Fixed(GeneralisedRidge):
            def __init__(self, wv=None, **kw):
                super().__init__(**kw)
                self.wv = wv

            def _row_weights(self, target, period=None):
                return self.wv

        a = Fixed(wv=w, alpha=0.5, alpha_scale="gram").fit(X, y).coef_
        b = Fixed(wv=w * 1e6, alpha=0.5, alpha_scale="gram").fit(X, y).coef_
        np.testing.assert_allclose(a, b, rtol=1e-9)


class TestGroupSpec(unittest.TestCase):
    """Hypotheses 2 and 4: per-group priors and per-group shrinkage."""

    def test_grammar_accepts_composed_fields(self):
        self.assertEqual(
            parse_group_spec("feature_momentum:s+,p0.25;feature_volatility:s-"),
            {"feature_momentum": {"s": 1, "p": 0.25},
             "feature_volatility": {"s": -1}})
        for empty in (None, "", "  ", "none"):
            self.assertEqual(parse_group_spec(empty), {})

    def test_grammar_rejects_every_malformed_spec(self):
        for bad in ("feature_close", "feature_close:z1", "feature_close:s*",
                    "feature_close:p1,p2", "feature_close:", "feature_close:pX",
                    "feature_close:p1;feature_close:p2", "feature_close:k-1"):
            with self.assertRaises(ValueError, msg=bad):
                parse_group_spec(bad)

    def test_a_zero_penalty_multiplier_is_refused_with_the_remedy(self):
        # p0 removes the penalty entirely and makes the system singular on a
        # collinear group -- which is every group here, that is why they exist.
        with self.assertRaises(ValueError) as caught:
            parse_group_spec("feature_close:p0")
        self.assertIn("p1e-6", str(caught.exception))

    def test_a_dict_is_refused_with_the_reason(self):
        with self.assertRaises(ValueError) as caught:
            parse_group_spec({"feature_close": {"p": 0.5}})
        self.assertIn("STRING", str(caught.exception))

    def test_an_unknown_stem_raises_and_names_the_vocabulary_trap(self):
        # experiments/feature_groups_crypto.sh defines SEVEN semantic groups by
        # regex and 57 result files already use those names; "oscillator" there
        # is two stems here. Silently ignoring the name would score the arm and
        # record a comparison that never happened.
        X, y = frame(n=400)
        with self.assertRaises(ValueError) as caught:
            GeneralisedRidge(group_spec="oscillator:p0.5").fit(X, y)
        message = str(caught.exception)
        self.assertIn("oscillator", message)
        self.assertIn("feature_groups_crypto.sh", message)

    def test_per_group_multiplier_moves_only_that_group(self):
        X, y = frame(n=3000)
        stems = stem_index(list(X.columns))
        target = stems["feature_volatility"]
        others = [i for i in range(X.shape[1]) if i not in target]
        base = GeneralisedRidge(alpha=1.0).fit(X, y)
        relaxed = GeneralisedRidge(alpha=1.0,
                                   group_spec="feature_volatility:p0.05").fit(X, y)
        self.assertEqual(relaxed.group_alpha_levels_, {"feature_volatility": 0.05})
        self.assertGreater(np.abs(relaxed.coef_[target]).mean(),
                           np.abs(base.coef_[target]).mean())
        # The other groups are not FROZEN -- the columns are correlated, so
        # relaxing one group reallocates a little everywhere. What must hold is
        # that the treated group moves by far the most; asserting the rest are
        # unchanged fails on a coefficient that merely sits near zero.
        moved_target = np.abs(relaxed.coef_[target] - base.coef_[target]).mean()
        moved_others = np.abs(relaxed.coef_[others] - base.coef_[others]).mean()
        self.assertGreater(moved_target, 10 * moved_others)

    def test_a_uniform_multiplier_is_only_alpha_rescaled(self):
        # The identification trap, pinned. A multiplier on EVERY group is
        # exactly alpha rescaled, and alpha is already searched -- so such an
        # arm would be a relabelling. Only a RELATIVE spec is a real axis.
        X, y = frame(n=2000)
        stems = stem_index(list(X.columns))
        spec = ";".join("%s:p3" % stem for stem in stems)
        uniform = GeneralisedRidge(alpha=1.0, group_spec=spec).fit(X, y)
        rescaled = GeneralisedRidge(alpha=3.0).fit(X, y)
        np.testing.assert_allclose(uniform.coef_, rescaled.coef_, rtol=1e-12)

    def test_zero_prior_on_every_group_equals_the_zero_prior_scheme(self):
        X, y = frame(n=2000)
        stems = stem_index(list(X.columns))
        spec = ";".join("%s:k0" % stem for stem in stems)
        masked = GeneralisedRidge(alpha=1.0, b0_scheme="marginal",
                                  prior_shrinkage=1.0, group_spec=spec).fit(X, y)
        plain = GeneralisedRidge(alpha=1.0, b0_scheme="zero").fit(X, y)
        np.testing.assert_allclose(masked.coef_, plain.coef_, rtol=1e-12)

    def test_zero_prior_on_one_group_moves_only_it(self):
        X, y = frame(n=2000)
        stems = stem_index(list(X.columns))
        target = stems["feature_volatility"]
        full = GeneralisedRidge(alpha=1.0, b0_scheme="marginal",
                                prior_shrinkage=1.0).fit(X, y)
        masked = GeneralisedRidge(alpha=1.0, b0_scheme="marginal",
                                  prior_shrinkage=1.0,
                                  group_spec="feature_volatility:k0").fit(X, y)
        self.assertEqual(masked.prior_levels_, {"feature_volatility": 0.0})
        self.assertGreater(np.abs(full.coef_[target] - masked.coef_[target]).max(),
                           1e-6)


class TestSignConstraints(unittest.TestCase):
    """Hypothesis 3: sign constraints on groups."""

    def test_unconstrained_fits_still_take_the_closed_form(self):
        X, y = frame(n=1500)
        model = GeneralisedRidge(alpha=1.0, group_spec="feature_close:p2").fit(X, y)
        self.assertEqual(model.solver_, "cholesky")
        self.assertEqual(model.n_constrained_, 0)

    def test_a_constraint_the_data_already_satisfies_is_a_bit_exact_no_op(self):
        # THE diagnostic. Without it a dead arm and a null result look the same:
        # both report a small t against the control. n_binding_ == 0 says the arm
        # never applied its treatment.
        X, y = frame(n=3000)
        stems = stem_index(list(X.columns))
        base = GeneralisedRidge(alpha=1.0).fit(X, y)
        satisfied = None
        for stem, index in stems.items():
            if all(base.coef_[i] < 0 for i in index):
                satisfied = stem
                break
        self.assertIsNotNone(satisfied, "fixture has no all-negative group")
        model = GeneralisedRidge(alpha=1.0,
                                 group_spec="%s:s-" % satisfied).fit(X, y)
        self.assertEqual(model.n_binding_, 0)
        np.testing.assert_allclose(model.coef_, base.coef_, rtol=0, atol=1e-12)

    def test_a_binding_constraint_pins_the_wrong_signed_coefficients(self):
        X, y = frame(n=3000)
        stems = stem_index(list(X.columns))
        index = stems["feature_volatility"]
        base = GeneralisedRidge(alpha=1.0).fit(X, y)
        model = GeneralisedRidge(alpha=1.0,
                                 group_spec="feature_volatility:s+").fit(X, y)
        self.assertEqual(model.solver_, "coordinate_descent")
        self.assertTrue((model.coef_[index] >= -1e-12).all())
        expected = int((base.coef_[index] < 0).sum())
        self.assertEqual(model.n_binding_, expected)

    def test_a_dead_column_is_never_counted_as_binding(self):
        # n_binding_ is the only signal that separates a no-op arm from a real
        # one, so counting a constant column's zero coefficient would inflate it.
        X, y = frame(n=1500)
        X = X.copy()
        X["feature_momentum_20d"] = 1.0
        X["feature_momentum_60d"] = 2.0
        model = GeneralisedRidge(alpha=1.0,
                                 group_spec="feature_momentum:s+").fit(X, y)
        self.assertEqual(model.n_constrained_, 2)
        self.assertEqual(model.n_binding_, 0)

    def test_the_constrained_solve_matches_an_independent_optimiser(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(9)
        p = 14
        A = rng.normal(size=(500, p))
        system = A.T @ A + 3.0 * np.eye(p)
        rhs = A.T @ rng.normal(size=500)
        lower = np.where(np.arange(p) < 5, 0.0, -np.inf)
        upper = np.full(p, np.inf)
        mine, _ = solve_box_constrained(system, rhs, lower, upper)
        objective = lambda b: 0.5 * b @ system @ b - rhs @ b
        reference = minimize(objective, np.zeros(p),
                             jac=lambda b: system @ b - rhs, method="L-BFGS-B",
                             bounds=list(zip(lower, upper)),
                             options={"ftol": 1e-16, "gtol": 1e-14,
                                      "maxiter": 50000}).x
        np.testing.assert_allclose(mine, reference, atol=1e-8)
        self.assertLessEqual(objective(mine), objective(reference) + 1e-9)

    def test_infinite_bounds_reproduce_the_closed_form(self):
        import scipy.linalg
        rng = np.random.default_rng(2)
        p = 20
        A = rng.normal(size=(600, p))
        system = A.T @ A + 2.0 * np.eye(p)
        rhs = A.T @ rng.normal(size=600)
        mine, _ = solve_box_constrained(system, rhs, np.full(p, -np.inf),
                                        np.full(p, np.inf))
        np.testing.assert_allclose(
            mine, scipy.linalg.solve(system, rhs, assume_a="pos"), rtol=1e-12)


class TestExtendedWrapper(unittest.TestCase):

    def test_declares_it_wants_the_period(self):
        # The gate that keeps every OTHER model untouched. Measured: ridge,
        # ridge_glmnet, lgb, xgb and catboost would ACCEPT a fit param and
        # silently change; the three Pipeline-backed MLPs REJECT one outright.
        self.assertTrue(build_wrapper("gen_ridge", "regression").WANTS_PERIOD)
        for name in MODELS:
            if name == "gen_ridge":
                continue
            for task in supported_tasks(name):
                wrapper = build_wrapper(name, task=task, seed=0, n_jobs=1)
                self.assertFalse(getattr(wrapper, "WANTS_PERIOD", False), name)

    def test_each_w_scheme_brings_only_its_own_dimension(self):
        # Asking for exp_decay must not drag a tail_weight into the grid that no
        # draw can use, burn budget on it, and then record it as if it meant
        # something.
        grid = build_wrapper("gen_ridge", "regression",
                             w_schemes=["uniform", "exp_decay"]).param_grid
        self.assertIn("half_life", grid)
        self.assertNotIn("tail_weight", grid)
        grid = build_wrapper("gen_ridge", "regression",
                             w_schemes=["uniform", "tail"]).param_grid
        self.assertIn("tail_weight", grid)
        self.assertNotIn("half_life", grid)

    def test_half_life_grid_is_floored_at_a_defensible_length(self):
        # A 130-date block at a 21-date half-life keeps 59 effective dates, about
        # 2.9 effective independent windows against 6.5 uniform. Shorter belongs
        # in a pinned arm, never in a searched dimension.
        grid = build_wrapper("gen_ridge", "regression",
                             w_schemes=["uniform", "exp_decay"]).param_grid
        self.assertGreaterEqual(float(np.min(grid["half_life"])), 32.0)

    def test_group_specs_become_one_categorical_dimension(self):
        specs = ["none", "feature_volatility:p0.1", "feature_close:s+"]
        grid = build_wrapper("gen_ridge", "regression",
                             group_specs=specs).param_grid
        self.assertEqual(grid["group_spec"], specs)
        self.assertEqual(sequence_valued_params(grid), [])
        self.assertEqual(set(to_space(grid)), set(grid))

    def test_a_malformed_group_spec_is_rejected_at_construction(self):
        # Not at the first refit, hours after the flag was typed.
        with self.assertRaises(ValueError):
            build_wrapper("gen_ridge", "regression", group_specs=["feature_close:p0"])

    def test_every_extended_grid_stays_bayes_searchable(self):
        import itertools
        for structures, w_schemes, b0_schemes, specs in itertools.product(
                [["identity"], ["identity", "group"]],
                [["uniform"], ["uniform", "tail"], ["uniform", "exp_decay"],
                 ["uniform", "tail", "exp_decay"]],
                [["zero"], ["zero", "marginal", "group_mean"]],
                [None, ["none", "feature_volatility:p0.1,s+"]]):
            grid = build_wrapper("gen_ridge", "regression",
                                 structures=structures, w_schemes=w_schemes,
                                 b0_schemes=b0_schemes,
                                 group_specs=specs).param_grid
            combo = (structures, w_schemes, b0_schemes, specs)
            self.assertEqual(sequence_valued_params(grid), [], combo)
            self.assertEqual(set(to_space(grid)), set(grid), combo)


if __name__ == "__main__":
    unittest.main()
