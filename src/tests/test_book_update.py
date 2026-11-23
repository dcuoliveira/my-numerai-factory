"""Tests for src.analysis.book_update (no data files or API needed)."""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis import book_update as bu
from src.analysis import factor_momentum_stake as fm


class AddToBook(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "fm.py")
        shutil.copy(fm.__file__, self.path)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _book(self):
        scope = {}
        src = open(self.path).read()
        block = bu._BOOK_RE.search(src).group(0)
        exec(block, scope)
        return scope["BOOK"]

    def test_appends_once_and_keeps_the_rest(self):
        before = self._book()
        self.assertEqual(bu.add_to_book(["NEW_SLOT_01"], self.path), ["NEW_SLOT_01"])
        self.assertEqual(self._book(), before + ["NEW_SLOT_01"])
        self.assertEqual(bu.add_to_book(["NEW_SLOT_01", before[0]], self.path), [])
        self.assertEqual(self._book(), before + ["NEW_SLOT_01"])
        # the rest of the module is untouched
        self.assertIn("def watch(", open(self.path).read())


class Writable(unittest.TestCase):
    def _cov(self, tail, common):
        return {"book_last": "2026-05-13",
                "candidates": {"X": {"last": "2026-02-19", "rows_missing_at_tail": tail,
                                     "n_common": common}}}

    def test_tail_gap_blocks(self):
        ok, reasons = bu.writable(self._cov(59, 400))
        self.assertFalse(ok)
        self.assertIn("2026-02-19", reasons[0])

    def test_short_history_blocks(self):
        self.assertFalse(bu.writable(self._cov(0, 100))[0])

    def test_full_coverage_passes(self):
        self.assertEqual(bu.writable(self._cov(0, 400)), (True, []))


class RankFrame(unittest.TestCase):
    def test_signal_is_resolved_trailing_mean(self):
        idx = pd.bdate_range("2025-01-01", periods=400)
        rng = np.random.default_rng(0)
        panel = pd.DataFrame(rng.normal(0, 0.01, (400, 3)), index=idx, columns=list("abc"))
        panel["c"] += 0.02
        df = bu.rank_frame(panel, list("abc"), lookback=250)
        self.assertEqual(df.index[0], "c")
        want = panel["a"].shift(fm.LAG).iloc[-250:].mean()
        self.assertAlmostEqual(df.loc["a", "signal"], want, places=12)
        self.assertEqual(list(df["rank"]), [1, 2, 3])
        # a payout inside the unresolved last LAG rows cannot move the signal
        bumped = panel.copy()
        bumped.iloc[-1, 0] += 100.0
        self.assertAlmostEqual(bu.rank_frame(bumped, list("abc"))["signal"]["a"],
                               df.loc["a", "signal"], places=12)


class AllocationPaths(unittest.TestCase):
    def test_a_never_ranked_candidate_leaves_the_book_unchanged(self):
        idx = pd.bdate_range("2024-01-01", periods=420)
        rng = np.random.default_rng(1)
        panel = pd.DataFrame(rng.normal(0, 0.01, (420, 4)), index=idx, columns=list("abcz"))
        panel[["a", "b", "c"]] += [0.03, 0.02, 0.01]
        panel["z"] -= 0.05                       # the candidate is always last
        out = bu.allocation_paths(panel, list("abc"), list("abcz"))
        for d in out.values():
            pd.testing.assert_series_equal(d["previous"], d["recent"])
            self.assertEqual(float(d["cand_weight"].max()), 0.0)
        # paths begin one ladder after the first complete signal
        ready = fm.signal(panel, lookback=250).dropna(how="any").index
        self.assertEqual(next(iter(out.values()))["recent"].index[0], ready[fm.LADDER])


if __name__ == "__main__":
    unittest.main()
