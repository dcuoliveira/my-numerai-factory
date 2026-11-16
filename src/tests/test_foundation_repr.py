"""
Tests for E109's foundation-model representation block: the three places a leak
could hide without crashing.

  - the TabPFN context window (foundation_repr.refresh_blocks): every context
    label must have resolved by the first date it predicts, and no row may sit
    in its own context;
  - the spot cut for TimesFM / Chronos-2 (foundation_repr_ts.context_at): row D
    may read only closes dated <= D-1, newest within 4 days -- the same rule as
    momentum_features.attach_signals_asof;
  - the embedding reduction (foundation_repr_ts.burnin_pca): the basis must come
    from the burn-in rows alone, so a later row cannot move an earlier one.

Offline, no model is loaded. Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np

from src.data_mani.foundation_repr import EMBARGO, refresh_blocks
from src.data_mani.foundation_repr_ts import (LAG_DAYS, MAX_STALE_DAYS, MIN_OBS,
                                              burnin_pca, context_at)


class RefreshBlocks(unittest.TestCase):

    def setUp(self):
        self.periods = np.arange(400)
        self.blocks = list(refresh_blocks(self.periods, first=200, refresh=20, window=130, embargo=EMBARGO))

    def test_every_context_label_resolves_before_the_block(self):
        # a label at t resolves at t + EMBARGO; the block's first row is at b
        for context, block in self.blocks:
            self.assertLessEqual(context.max() + EMBARGO, block.min())

    def test_no_row_in_its_own_context(self):
        for context, block in self.blocks:
            self.assertEqual(len(np.intersect1d(context, block)), 0)

    def test_full_window_and_contiguous_cover(self):
        for context, _ in self.blocks:
            self.assertEqual(len(context), 130)
        predicted = np.concatenate([b for _, b in self.blocks])
        np.testing.assert_array_equal(predicted, np.arange(200, 400))

    def test_short_history_block_is_skipped(self):
        blocks = list(refresh_blocks(self.periods, first=100, refresh=20, window=130, embargo=EMBARGO))
        self.assertGreaterEqual(blocks[0][1].min(), 130 + EMBARGO)


class SpotContext(unittest.TestCase):

    def setUp(self):
        self.dates = np.arange(0, 200)                # one close a day
        self.history = (self.dates, np.log(100.0 + self.dates))

    def test_never_reads_day_d(self):
        ctx = context_at(self.history, 150)
        self.assertEqual(ctx[-1], np.log(100.0 + 150 - LAG_DAYS))

    def test_weekend_gap_is_fresh_enough(self):
        dates = np.r_[np.arange(0, 100), np.arange(103, 110)]
        ctx = context_at((dates, np.log(1.0 + dates)), 103)   # newest usable = day 99
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx[-1], np.log(1.0 + 99))

    def test_stale_and_short_series_are_left_out(self):
        self.assertIsNone(context_at(self.history, 199 + MAX_STALE_DAYS + 1))
        self.assertIsNotNone(context_at(self.history, MIN_OBS + LAG_DAYS - 1))   # exactly MIN_OBS
        self.assertIsNone(context_at(self.history, MIN_OBS + LAG_DAYS - 2))      # one short


class BurninPCA(unittest.TestCase):

    def test_later_rows_do_not_move_the_basis(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 12))
        dates = np.repeat(np.arange(50), 10)
        burnin = np.arange(20)
        before = burnin_pca(x, dates, burnin, 4)
        x2 = x.copy()
        x2[dates >= 20] *= 50.0                       # wreck every post-burn-in row
        after = burnin_pca(x2, dates, burnin, 4)
        np.testing.assert_allclose(before[dates < 20], after[dates < 20])


if __name__ == "__main__":
    unittest.main()
