"""
Tests for walk_forward's tune_window: SEARCH on a long block, FIT on a short one.

Run with:  python3 -m unittest src.tests.test_tune_window -v

Why the feature exists: a purged inner fold needs training periods + the
embargo + validation periods inside the block it searches, so a training window
of 22 periods or fewer at embargo 20 cannot hold one and its search raises.
tune_window searches on a longer block ending at the same train_end, with
folds that each train on train_window periods, then refits the winner on the
training window alone.
"""

import unittest

import numpy as np
from sklearn.linear_model import Ridge

from src.prediction.functions import (
    hyper_params_search,
    purged_period_splits,
    walk_forward_fit_and_predict,
)
from src.prediction.models import build_wrapper
from src.tests.test_lambda_selection import ALPHAS, panel

FEATURES = ["f0", "f1", "f2", "f3"]


def run(df, train_window, tune_window=None, fixed_params=None, model="ridge"):
    wrapper = build_wrapper(model, task="regression", seed=42, n_jobs=1)
    if model == "ridge":
        wrapper.param_grid = {"alpha": ALPHAS}
    return walk_forward_fit_and_predict(
        df=df.copy(), wrapper=wrapper, features=FEATURES, target_col="target",
        period_col="date", id_col="symbol", meta=df[["date", "target"]],
        reference_col=None, task="regression", metric="corr",
        n_iter=len(ALPHAS), n_splits=3, n_jobs=1, seed=42, verbose=False,
        refit_every=7, embargo=3, scheme="rolling", train_window=train_window,
        fixed_params=fixed_params, tune_window=tune_window)


class TestTuneWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = panel()

    def test_default_is_unchanged(self):
        # tune_window=None and tune_window <= train_window must both be the old
        # path exactly: same choices, same predictions.
        base, base_refits, _, _ = run(self.df, 24)
        same, same_refits, _, _ = run(self.df, 24, tune_window=24)
        self.assertEqual([r["best_params"] for r in base_refits],
                         [r["best_params"] for r in same_refits])
        np.testing.assert_array_equal(base["prediction"].to_numpy(),
                                      same["prediction"].to_numpy())
        self.assertNotIn("tune_rows", base_refits[0])

    def test_a_window_too_short_for_a_fold_raises_without_it(self):
        # The case the feature exists for: 2 training periods + embargo 3 cannot
        # hold a purged fold, so a plain search refuses.
        with self.assertRaises(ValueError):
            run(self.df, 2)

    def test_with_it_the_short_window_searches_and_fits(self):
        predictions, refits, _, _ = run(self.df, 2, tune_window=24)
        self.assertGreater(len(predictions), 0)
        for r in refits:
            self.assertEqual(r["train_rows"], 2 * 25)        # FIT on 2 periods
            self.assertEqual(r["tune_rows"], 24 * 25)        # SEARCHED on 24
            self.assertEqual(r["tune_periods"], 24)
            self.assertIn(r["best_params"]["alpha"], list(ALPHAS))

    def test_the_final_model_is_fitted_on_the_training_window_only(self):
        # The search refits its winner on the TUNING block; the walk-forward
        # must replace that with a fit on the training window. Reproduce one
        # block's predictions from a plain Ridge on the last 2 periods.
        predictions, refits, _, _ = run(self.df, 2, tune_window=24)
        r = refits[0]
        periods = sorted(self.df["date"].unique())
        train = [p for p in periods if r["train_start"] <= p <= r["train_end"]]
        self.assertEqual(len(train), 2)
        rows = self.df[self.df["date"].isin(train)]
        model = Ridge(alpha=r["best_params"]["alpha"], random_state=42).fit(
            rows[FEATURES], rows["target"])
        block = predictions[predictions["refit"] == 0]
        source = self.df.set_index(["symbol", "date"]).loc[
            list(zip(block["symbol"], block["date"])), FEATURES]
        np.testing.assert_allclose(block["prediction"].to_numpy(),
                                   model.predict(source), rtol=1e-9, atol=1e-12)

    def test_inner_folds_train_on_the_training_window(self):
        # Every fold but the first (which has no history before it) trains on
        # exactly train_window periods, so a candidate is scored as it is used.
        tune = np.repeat(np.arange(24), 3)
        folds = purged_period_splits(tune, n_splits=3, embargo=3,
                                     scheme="rolling", train_window=2)
        lengths = [len(np.unique(tune[tr])) for tr, _ in folds]
        self.assertTrue(all(n == 2 for n in lengths[1:]), lengths)

    def test_the_glmnet_grid_is_scaled_to_the_fit_block_not_the_tune_block(self):
        # ridge_glmnet's alphas are lambda x N. Built on the 24-period tuning
        # block they would be 12x too strong for a 2-period fit.
        df = self.df
        periods = sorted(df["date"].unique())
        tune_rows = df[df["date"].isin(periods[:24])]
        fit_rows = df[df["date"].isin(periods[22:24])]
        wrapper = build_wrapper("ridge_glmnet", task="regression", seed=42, n_jobs=1)
        common = dict(X=tune_rows[FEATURES], y=tune_rows["target"], wrapper=wrapper,
                      meta=tune_rows[["date", "target"]], metric="corr",
                      period_col="date", reference_col=None, task="regression",
                      n_iter=5, n_splits=3, embargo=3, scheme="rolling",
                      train_window=2, n_jobs=1, seed=42, verbose=False)
        scaled = hyper_params_search(grid_from=(fit_rows[FEATURES],
                                                fit_rows["target"]), **common)
        unscaled = hyper_params_search(**common)
        ratio = unscaled.searched_bounds_["alpha"][1] / scaled.searched_bounds_["alpha"][1]
        self.assertGreater(ratio, 5.0)        # ~N_tune / N_fit, data-dependent

    def test_pinned_configs_ignore_it(self):
        # A pinned configuration has nothing to tune, so tune_window must not
        # route it through the tuning block.
        a, ra, _, _ = run(self.df, 24, fixed_params={"alpha": 10.0})
        b, rb, _, _ = run(self.df, 24, tune_window=40, fixed_params={"alpha": 10.0})
        np.testing.assert_array_equal(a["prediction"].to_numpy(),
                                      b["prediction"].to_numpy())
        self.assertNotIn("tune_rows", rb[0])


if __name__ == "__main__":
    unittest.main()
