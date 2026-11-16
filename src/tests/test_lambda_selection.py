"""
Tests for src.prediction.lambda_selection (E115), the parts that would produce a
plausible wrong number rather than a crash:

  * a closed-form ridge that drifts from sklearn's Ridge
  * a per-period loss that is not the CORR the pipeline scores
  * a prequential choice that reads a target before it has resolved
  * a TSCV rule that is not the pipeline's own search, so the comparison would
    pit prequential against a strawman
  * warm-up blocks that overlap, leave gaps, or train on a short window
  * a payout utility that is not the payout the report scores, or that lets a
    period with no meta model vote

Run with:  python -m unittest src.tests.test_lambda_selection
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.analysis.results_table import CORR_WEIGHT, payout_series
from src.prediction.functions import (
    hyper_params_search,
    sorted_unique,
    walk_forward_fit_and_predict,
)
from src.prediction.lambda_selection import (
    PAYOUT_CORR_WEIGHT,
    lambda_walk_forward,
    period_losses,
    prequential_choice,
    refit_blocks,
    ridge_path,
    tscv_losses,
)
from src.prediction.metrics import per_period_bmc, per_period_corr
from src.prediction.models import build_wrapper


ALPHAS = np.logspace(-1, 4, 6)


def panel(n_periods=70, per=25, n_val=14, seed=0, meta_from=None):
    """
    A small panel shaped like the crypto frame: binned target, split column.

    The features are deliberately COLLINEAR. With orthogonal, equal-variance
    columns ridge only rescales least squares, every alpha ranks every period
    identically, CORR cannot tell them apart, and a test of which alpha a rule
    picks would be a test of tie-breaking.

    meta_from adds a "meta" column that starts at that period index and is NaN
    before it -- the crypto meta model starts in 2024, long after the data.
    """
    rng = np.random.default_rng(seed)
    periods = np.repeat(["p%03d" % i for i in range(n_periods)], per)
    symbols = np.tile(["s%02d" % j for j in range(per)], n_periods)
    n = len(periods)
    a, b = rng.random(n), rng.random(n)
    X = np.column_stack([a, b, a + 0.05 * rng.normal(size=n),
                         0.5 * a + 0.5 * b + 0.05 * rng.normal(size=n)])
    signal = X @ np.array([1.0, -0.5, -0.8, 0.6]) + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["date"] = periods
    df["symbol"] = symbols
    df["target"] = (df.assign(s=signal).groupby("date")["s"]
                    .transform(lambda s: pd.qcut(s, 5, labels=False) / 4.0))
    df["split"] = np.where(df["date"] >= "p%03d" % (n_periods - n_val),
                           "validation", "train")
    if meta_from is not None:
        meta = X @ np.array([0.8, -0.2, -0.5, 0.4]) + rng.normal(scale=0.3, size=n)
        df["meta"] = np.where(df["date"] >= "p%03d" % meta_from, meta, np.nan)
    return df


class TestRidgePath(unittest.TestCase):

    def test_matches_sklearn_ridge_for_every_alpha(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(400, 3))
        # Collinear columns, so the penalty actually moves the solution.
        X = np.hstack([base, base @ rng.normal(size=(3, 4)) + 0.01 * rng.normal(size=(400, 4))])
        y = X @ rng.normal(size=7) + rng.normal(size=400)
        alphas = np.array([1e-3, 1.0, 1e3, 1e6])
        coef, intercept = ridge_path(X, y, alphas)
        for k, alpha in enumerate(alphas):
            model = Ridge(alpha=alpha).fit(X, y)
            np.testing.assert_allclose(coef[:, k], model.coef_, rtol=1e-6, atol=1e-9)
            self.assertAlmostEqual(intercept[k], model.intercept_, places=8)


class TestPeriodLosses(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(2)
        self.period = np.repeat(["a", "b", "c", "d"], 40)
        self.target = rng.integers(0, 5, size=160) / 4.0
        self.pred = rng.normal(size=(160, 3))

    def test_corr_is_minus_the_pipelines_per_period_corr(self):
        losses = period_losses(self.pred, self.target, self.period, "corr")
        for j in range(3):
            frame = pd.DataFrame({"p": self.pred[:, j], "t": self.target,
                                  "date": self.period})
            expected = per_period_corr(frame, "p", "t", "date")
            np.testing.assert_allclose(-losses[j].to_numpy(), expected.to_numpy(),
                                       atol=1e-12)

    def test_mse_is_the_mean_within_each_period(self):
        losses = period_losses(self.pred, self.target, self.period, "mse")
        for j in range(3):
            frame = pd.DataFrame({"e": (self.pred[:, j] - self.target) ** 2,
                                  "date": self.period})
            np.testing.assert_allclose(losses[j].to_numpy(),
                                       frame.groupby("date")["e"].mean().to_numpy())

    def test_a_constant_prediction_is_charged_zero_corr(self):
        pred = self.pred.copy()
        pred[self.period == "b", 1] = 0.3
        losses = period_losses(pred, self.target, self.period, "corr")
        self.assertEqual(losses.loc["b", 1], 0.0)
        self.assertTrue(np.isfinite(losses.to_numpy()).all())

    def test_payout_is_minus_the_reported_payout(self):
        rng = np.random.default_rng(3)
        meta = rng.normal(size=160)
        meta[self.period == "a"] = np.nan          # no meta model at all
        meta[np.where(self.period == "c")[0][:15]] = np.nan   # partly covered
        losses = period_losses(self.pred, self.target, self.period, "payout",
                               meta=meta)
        for j in range(3):
            frame = pd.DataFrame({"p": self.pred[:, j], "t": self.target,
                                  "m": meta, "date": self.period})
            table = pd.DataFrame({"corr": per_period_corr(frame, "p", "t", "date"),
                                  "mmc": per_period_bmc(frame, "p", "m", "t", "date")})
            expected = payout_series(table).reindex(losses.index)
            np.testing.assert_allclose(-losses[j].to_numpy(), expected.to_numpy(),
                                       atol=1e-12)
        # A period with no meta model has no payout: it must not vote.
        self.assertTrue(losses.loc["a"].isna().all())
        self.assertTrue(np.isfinite(losses.loc[["b", "c", "d"]].to_numpy()).all())

    def test_payout_needs_the_meta_model(self):
        with self.assertRaises(ValueError):
            period_losses(self.pred, self.target, self.period, "payout")

    def test_payout_weight_is_the_reports(self):
        self.assertEqual(PAYOUT_CORR_WEIGHT, CORR_WEIGHT)


class TestPrequentialCutoff(unittest.TestCase):
    """The cutoff is the only guard between the choice and unresolved targets."""

    def setUp(self):
        index = pd.Index(["p%02d" % i for i in range(10)])
        # Alpha 0 is better on the first six periods, alpha 1 by far on the rest.
        self.losses = pd.DataFrame({0: [0.1] * 6 + [9.0] * 4,
                                    1: [0.2] * 6 + [-9.0] * 4}, index=index)

    def test_only_periods_before_the_cutoff_count(self):
        pick, mean, counted = prequential_choice(self.losses, "p06")
        self.assertEqual(pick, 0)
        self.assertEqual(counted, 6)
        np.testing.assert_allclose(mean, [0.1, 0.2])

    def test_the_future_flips_the_choice_once_it_has_resolved(self):
        pick, _, counted = prequential_choice(self.losses, "p99")
        self.assertEqual((pick, counted), (1, 10))

    def test_nothing_resolved_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError):
            prequential_choice(self.losses, "p00")

    def test_undefined_periods_neither_vote_nor_count(self):
        losses = self.losses.copy()
        losses.iloc[:4] = np.nan               # e.g. payout before the meta model
        pick, mean, counted = prequential_choice(losses, "p06")
        self.assertEqual((pick, counted), (0, 2))
        np.testing.assert_allclose(mean, [0.1, 0.2])
        with self.assertRaises(ValueError):
            prequential_choice(losses, "p04")


class TestRefitBlocks(unittest.TestCase):

    def setUp(self):
        self.all_periods = np.array(["p%03d" % i for i in range(200)])
        self.val = self.all_periods[150:]

    def test_layout(self):
        blocks, skipped = refit_blocks(self.all_periods, self.val, refit_every=7,
                                       embargo=5, scheme="rolling", train_window=40)
        self.assertEqual(skipped, [])
        warm = [b for b in blocks if not b["scored"]]
        scored = [b for b in blocks if b["scored"]]
        # Scored blocks are the validation blocks the pipeline lays out.
        self.assertEqual([b["start"] for b in scored], list(range(150, 200, 7)))
        self.assertEqual([b["refit"] for b in scored], list(range(len(scored))))
        # Warm-up and scored blocks tile the timeline without gaps or overlaps.
        for a, b in zip(blocks, blocks[1:]):
            self.assertEqual(a["stop"], b["start"])
        self.assertEqual(warm[-1]["stop"], 150)
        # Every block trains on a full window that ends `embargo` before it.
        for b in blocks:
            self.assertEqual(b["train_end"], b["start"] - 5)
            self.assertEqual(b["train_end"] - b["train_start"], 40)
        # And the warm-up runs back as far as a full window allows.
        self.assertLess(warm[0]["start"] - 7 - 5, 40)

    def test_validation_must_be_the_tail(self):
        with self.assertRaises(ValueError):
            refit_blocks(self.all_periods, self.all_periods[100:150], 7, 5,
                         "rolling", 40)


class TestAgainstThePipeline(unittest.TestCase):
    """
    The TSCV arm must BE the pipeline's ridge search over the same grid, or the
    study compares prequential against something nobody runs.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = panel()
        cls.features = ["f0", "f1", "f2", "f3"]
        cls.kwargs = dict(refit_every=7, embargo=3, scheme="rolling",
                          train_window=24)
        cls.wf = lambda_walk_forward(
            cls.df, cls.features, "target", "date", "symbol", ALPHAS,
            n_splits=3, verbose=False, **cls.kwargs)

    def pipeline(self, fixed_params=None):
        wrapper = build_wrapper("ridge", task="regression", seed=42, n_jobs=1)
        wrapper.param_grid = {"alpha": ALPHAS}
        predictions, refits, _, _ = walk_forward_fit_and_predict(
            df=self.df.copy(), wrapper=wrapper, features=self.features,
            target_col="target", period_col="date", id_col="symbol",
            meta=self.df[["date", "target"]], reference_col=None,
            task="regression", metric="corr", n_iter=len(ALPHAS), n_splits=3,
            n_jobs=1, seed=42, verbose=False, fixed_params=fixed_params,
            **self.kwargs)
        return predictions, refits

    def test_tscv_corr_arm_reproduces_the_pipeline_search(self):
        expected, refits = self.pipeline()
        got, records = self.wf["arms"][("tscv", "corr")]
        self.assertEqual([r["best_params"]["alpha"] for r in records],
                         [r["best_params"]["alpha"] for r in refits])
        for key in ("train_start", "train_end", "predict_start", "predict_end"):
            self.assertEqual([r[key] for r in records], [r[key] for r in refits])
        merged = got.merge(expected, on=["symbol", "date"], suffixes=("", "_pipe"))
        self.assertEqual(len(merged), len(expected))
        np.testing.assert_allclose(merged["prediction"], merged["prediction_pipe"],
                                   rtol=1e-7, atol=1e-10)

    def test_tscv_mean_loss_is_minus_the_search_score(self):
        df = self.df[self.df["date"] < "p024"].reset_index(drop=True)
        X, y, period = (df[self.features].to_numpy(), df["target"].to_numpy(),
                        df["date"].to_numpy())
        mine = tscv_losses(X, y, period, ALPHAS, ["corr"], n_splits=3, embargo=3,
                           scheme="rolling", train_window=24)["corr"][0]
        wrapper = build_wrapper("ridge", task="regression", seed=42, n_jobs=1)
        wrapper.param_grid = {"alpha": ALPHAS}
        search = hyper_params_search(
            X=df[self.features], y=df["target"], wrapper=wrapper,
            meta=df[["date", "target"]], metric="corr", period_col="date",
            reference_col=None, task="regression", n_iter=len(ALPHAS), n_splits=3,
            embargo=3, scheme="rolling", train_window=24, n_jobs=1, seed=42,
            verbose=False)
        score = pd.Series(search.cv_results_["mean_test_objective"],
                          index=np.asarray(search.cv_results_["param_alpha"], float))
        np.testing.assert_allclose(mine, -score.sort_index().to_numpy(), atol=1e-12)

    def test_every_alpha_column_is_the_pipelines_fixed_alpha_fit(self):
        alpha = ALPHAS[4]
        expected, _ = self.pipeline(fixed_params={"alpha": alpha})
        oos = self.wf["oos"].assign(prediction=self.wf["matrix"][:, 4])
        merged = oos.merge(expected, on=["symbol", "date"], suffixes=("", "_pipe"))
        self.assertEqual(len(merged), len(expected))
        np.testing.assert_allclose(merged["prediction"], merged["prediction_pipe"],
                                   rtol=1e-7, atol=1e-10)

    def test_prequential_counts_only_resolved_out_of_sample_periods(self):
        all_periods = sorted_unique(self.df["date"].to_numpy())
        oos_periods = sorted_unique(self.wf["oos"]["date"].to_numpy())
        _, records = self.wf["arms"][("prequential", "corr")]
        for record in records:
            # train_end in the record is the LAST training period, inclusive.
            last = record["train_end"]
            expected = int((oos_periods <= last).sum())
            self.assertEqual(record["selection_periods"], expected)
            self.assertGreater(expected, 0)
            self.assertLess(last, record["predict_start"])
        self.assertIn(records[0]["train_end"], all_periods)

    def test_the_four_arms_share_their_rows_and_differ_only_in_alpha(self):
        frames = [p for p, _ in self.wf["arms"].values()]
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(
                frame[["symbol", "date", "target", "refit"]],
                frames[0][["symbol", "date", "target", "refit"]])


class TestPayoutWalkForward(unittest.TestCase):
    """
    The crypto meta model starts in 2024, long after the data, so the payout
    arms must choose on the periods that HAVE one and say how many that was.

    Layout here: refit 7, embargo 3, window 24, meta model from p045. The first
    scored block trains on p029..p052; the warm-up predicts from p028.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = panel(meta_from=45)
        cls.wf = lambda_walk_forward(
            cls.df, ["f0", "f1", "f2", "f3"], "target", "date", "symbol", ALPHAS,
            refit_every=7, embargo=3, scheme="rolling", train_window=24,
            n_splits=3, losses=("corr", "payout"), meta_col="meta",
            extra_cols=("meta",), verbose=False)

    def test_prequential_counts_only_resolved_periods_with_a_meta_model(self):
        oos = sorted_unique(self.wf["oos"]["date"].to_numpy())
        _, records = self.wf["arms"][("prequential", "payout")]
        for record in records:
            expected = int(((oos <= record["train_end"]) & (oos >= "p045")).sum())
            self.assertEqual(record["selection_periods"], expected)
        self.assertEqual(records[0]["selection_periods"], 8)    # p045..p052

    def test_tscv_votes_only_with_folds_that_have_a_meta_model(self):
        _, records = self.wf["arms"][("tscv", "payout")]
        # Of the first window's two scorable folds (p037..p044, p045..p052)
        # only the second has a meta model.
        self.assertEqual(records[0]["selection_periods"], 8)
        _, corr_records = self.wf["arms"][("tscv", "corr")]
        self.assertEqual(corr_records[0]["selection_periods"], 16)

    def test_a_block_with_no_meta_model_falls_back_to_the_corr_choice(self):
        # Meta model from p055: the first scored block trains on p029..p052, so
        # neither rule has any payout evidence there; the second (to p059) does.
        wf = lambda_walk_forward(
            panel(meta_from=55), ["f0", "f1", "f2", "f3"], "target", "date",
            "symbol", ALPHAS, refit_every=7, embargo=3, scheme="rolling",
            train_window=24, n_splits=3, losses=("payout",), meta_col="meta",
            verbose=False)
        for rule in ("prequential", "tscv"):
            _, records = wf["arms"][(rule, "payout")]
            self.assertEqual(records[0]["selection_loss"], "corr")
            self.assertEqual(records[1]["selection_loss"], "payout")
            _, corr_records = self.wf["arms"][(rule, "corr")]
            # Same panel apart from the meta column, so the corr pick is shared.
            self.assertEqual(records[0]["alpha_index"], corr_records[0]["alpha_index"])

    def test_narrowing_the_arms_changes_nothing_in_the_ones_kept(self):
        narrow = lambda_walk_forward(
            self.df, ["f0", "f1", "f2", "f3"], "target", "date", "symbol", ALPHAS,
            refit_every=7, embargo=3, scheme="rolling", train_window=24,
            n_splits=3, arms=[("tscv", "corr"), ("prequential", "payout")],
            meta_col="meta", extra_cols=("meta",), verbose=False)
        self.assertEqual(list(narrow["arms"]), [("tscv", "corr"), ("prequential", "payout")])
        for key in narrow["arms"]:
            pd.testing.assert_frame_equal(narrow["arms"][key][0], self.wf["arms"][key][0])
            self.assertEqual(narrow["arms"][key][1], self.wf["arms"][key][1])
        # Prequential payout needs payout and its corr fallback, nothing else.
        self.assertEqual(sorted(narrow["losses"]), ["corr", "payout"])

    def test_payout_without_a_meta_column_is_refused(self):
        with self.assertRaises(ValueError):
            lambda_walk_forward(
                self.df, ["f0", "f1", "f2", "f3"], "target", "date", "symbol",
                ALPHAS, refit_every=7, embargo=3, scheme="rolling",
                train_window=24, n_splits=3, losses=("payout",), verbose=False)


if __name__ == "__main__":
    unittest.main()
