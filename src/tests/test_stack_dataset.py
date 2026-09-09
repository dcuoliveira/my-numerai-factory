"""
Tests for the stacking dataset builder.

Stacking is the easiest place in this repo to leak, because the leak does not
crash -- it produces a better number. These cover the failures that look like
success:

  * member predictions averaged on their raw scales rather than ranked
  * an outer join, so the stack spans dates a member never scored
  * a feature prefix the Crypto loader cannot see, fitting on nothing
  * two members colliding into one column
  * stacking across two different labels
  * a split that is not a strict, embargoed tail of the members' window

Everything runs offline in a second -- no downloads, no fitting.

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.build_stack_dataset import (
    RESERVED,
    assemble,
    attach_targets,
    member_column,
    rank_within_period,
    split_member,
)
from src.data_mani.utils import make_validation_split


def member_frame(dates, symbols, values):
    """A forecast-shaped frame indexed on (symbol, date)."""
    frame = pd.DataFrame({
        "symbol": np.tile(symbols, len(dates)),
        "date": np.repeat(dates, len(symbols)),
        "prediction": values,
    })
    return frame.set_index(["symbol", "date"])


class TestMemberNaming(unittest.TestCase):
    """Members are addressed model/run, and must not collide."""

    def test_explicit_model_wins(self):
        self.assertEqual(split_member("mlp/e10_raw", "ridge"), ("mlp", "e10_raw"))

    def test_bare_name_uses_the_default(self):
        self.assertEqual(split_member("e10_raw", "ridge"), ("ridge", "e10_raw"))

    def test_bare_name_without_a_default_is_fatal(self):
        with self.assertRaises(SystemExit):
            split_member("e10_raw", None)

    def test_same_run_in_two_trees_gets_two_columns(self):
        # The case that motivates the naming: stacking across wrappers, where
        # the same run name exists in both trees.
        a = member_column("ridge", "reg_corr_expanding")
        b = member_column("mlp", "reg_corr_expanding")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("feature_"))


class TestRanking(unittest.TestCase):
    """Members must be normalised within period, never across the panel."""

    def setUp(self):
        self.dates = pd.date_range("2020-01-01", periods=3, freq="D")
        self.symbols = ["A", "B", "C", "D"]

    def test_ranks_are_per_period(self):
        # Date 2's values are an order of magnitude larger. Ranked within
        # period they are indistinguishable from date 1's; pooled they would
        # dominate, which is exactly the bug.
        values = np.concatenate([[1, 2, 3, 4], [10, 20, 30, 40], [4, 3, 2, 1]])
        frame = member_frame(self.dates, self.symbols, values.astype(float))

        ranked = rank_within_period(frame, "date")

        for date in self.dates:
            block = ranked.xs(date, level="date")
            self.assertEqual(sorted(block.to_numpy()), [0.25, 0.5, 0.75, 1.0])

    def test_ranks_lie_in_the_unit_interval(self):
        values = np.random.RandomState(0).normal(size=len(self.dates) * len(self.symbols))
        ranked = rank_within_period(member_frame(self.dates, self.symbols, values), "date")

        self.assertGreater(ranked.min(), 0.0)
        self.assertLessEqual(ranked.max(), 1.0)

    def test_scale_is_irrelevant(self):
        # A member on a proba scale and the same ordering on a regression scale
        # must produce identical columns.
        order = np.array([0.1, 0.4, 0.2, 0.3])
        small = member_frame(self.dates[:1], self.symbols, order)
        large = member_frame(self.dates[:1], self.symbols, order * 1000 - 500)

        pd.testing.assert_series_equal(
            rank_within_period(small, "date"), rank_within_period(large, "date"))


class TestAssemble(unittest.TestCase):
    """The join is what bounds the panel to the members' shared window."""

    def setUp(self):
        self.symbols = ["A", "B"]
        self.early = pd.date_range("2020-01-01", periods=10, freq="D")
        self.late = pd.date_range("2020-01-06", periods=10, freq="D")

        rng = np.random.RandomState(1)
        self.one = rank_within_period(
            member_frame(self.early, self.symbols, rng.normal(size=20)), "date")
        self.two = rank_within_period(
            member_frame(self.late, self.symbols, rng.normal(size=20)), "date")

    def test_join_is_the_intersection(self):
        panel = assemble([("feature_stk_a", self.one),
                          ("feature_stk_b", self.two)], "symbol", "date")

        dates = set(panel.index.get_level_values("date"))
        self.assertEqual(dates, set(self.early) & set(self.late))
        self.assertFalse(panel.isna().any().any())

    def test_panel_never_exceeds_a_member_window(self):
        # The build-time no-lookahead invariant: a stack cannot be scored on a
        # date some member never predicted.
        panel = assemble([("feature_stk_a", self.one),
                          ("feature_stk_b", self.two)], "symbol", "date")
        dates = set(panel.index.get_level_values("date"))

        for member in (self.one, self.two):
            self.assertTrue(dates <= set(member.index.get_level_values("date")))

    def test_one_column_per_member(self):
        panel = assemble([("feature_stk_a", self.one),
                          ("feature_stk_b", self.two)], "symbol", "date")
        self.assertEqual(list(panel.columns), ["feature_stk_a", "feature_stk_b"])

    def test_duplicate_columns_are_fatal(self):
        with self.assertRaises(SystemExit):
            assemble([("feature_stk_a", self.one),
                      ("feature_stk_a", self.two)], "symbol", "date")

    def test_prefix_the_loader_cannot_see_is_fatal(self):
        # Without the feature_ prefix the dataset loads with zero features and
        # the run still "succeeds", which is the worst possible outcome.
        with self.assertRaises(SystemExit):
            assemble([("stk_a", self.one)], "symbol", "date", prefix="stk_")


class TestAttachTargets(unittest.TestCase):
    """Targets come off the source dataset, and unresolved ones survive."""

    def setUp(self):
        self.symbols = ["A", "B"]
        self.dates = pd.date_range("2020-01-01", periods=4, freq="D")
        ranked = rank_within_period(
            member_frame(self.dates, self.symbols, np.arange(8.0)), "date")
        self.panel = assemble([("feature_stk_a", ranked)], "symbol", "date")

        self.source = pd.DataFrame({
            "symbol": np.tile(self.symbols, len(self.dates)),
            "date": np.repeat(self.dates, len(self.symbols)),
            "target_binned_return_20": np.linspace(0, 1, 8),
            "target_binned_return_60": np.linspace(0, 1, 8),
        }).set_index(["symbol", "date"])

    def test_targets_are_joined_on(self):
        out = attach_targets(self.panel, self.source, verbose=False)

        self.assertIn("target_binned_return_20", out.columns)
        self.assertEqual(len(out), len(self.panel))
        np.testing.assert_allclose(
            out["target_binned_return_20"].to_numpy(),
            self.source["target_binned_return_20"].reindex(out.index).to_numpy())

    def test_rows_absent_from_the_source_are_dropped(self):
        trimmed = self.source.iloc[:-2]
        out = attach_targets(self.panel, trimmed, verbose=False)

        self.assertEqual(len(out), len(trimmed))
        self.assertFalse(out["target_binned_return_20"].isna().any())

    def test_unresolved_targets_are_kept(self):
        # The 60-day label is genuinely NaN over the tail. Dropping here would
        # silently shorten every 20-day stack too; load_tournament_data drops on
        # whichever target is actually modelled.
        self.source.loc[:, "target_binned_return_60"] = np.nan
        out = attach_targets(self.panel, self.source, verbose=False)

        self.assertEqual(len(out), len(self.panel))
        self.assertTrue(out["target_binned_return_60"].isna().all())
        self.assertFalse(out["target_binned_return_20"].isna().any())

    def test_reserved_columns_are_not_produced(self):
        out = attach_targets(self.panel, self.source, verbose=False)
        for column in RESERVED:
            self.assertNotIn(column, out.reset_index().columns)


class TestStackSplitIsAStrictTail(unittest.TestCase):
    """
    End-to-end no-lookahead: the stack is scored on a tail of the member window.

    This is the guarantee that makes the whole design defensible, and the one
    that is invisible in the output -- a stack scored on dates its members
    trained through would simply look excellent.
    """

    def setUp(self):
        self.symbols = ["A", "B"]
        self.member_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        ranked = rank_within_period(
            member_frame(self.member_dates, self.symbols,
                         np.random.RandomState(2).normal(size=200)), "date")
        self.panel = assemble([("feature_stk_a", ranked)], "symbol", "date")

    def test_validation_is_a_strict_embargoed_tail(self):
        frame = self.panel.reset_index()
        out = make_validation_split(frame, "date", val_frac=0.3, embargo=20)

        train = out.loc[out["split"] == "train", "date"]
        val = out.loc[out["split"] == "validation", "date"]

        self.assertLess(train.max(), val.min())
        self.assertGreaterEqual((val.min() - train.max()).days, 20)

    def test_scored_window_stays_inside_the_member_window(self):
        frame = self.panel.reset_index()
        out = make_validation_split(frame, "date", val_frac=0.3, embargo=20)
        scored = set(out.loc[out["split"] == "validation", "date"])

        # Inside the members' window, and a strict subset of it -- the shrinkage
        # from 100 member dates to 30 scored dates is the cost of stacking.
        self.assertTrue(scored < set(self.member_dates))
        self.assertEqual(len(scored), 30)


if __name__ == "__main__":
    unittest.main()
