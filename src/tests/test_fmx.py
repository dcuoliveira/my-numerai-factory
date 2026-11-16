"""
Tests for E118's representation builders: the places a look-ahead could hide.

  - fmx_tabpfn.ahead: a nowcast label is the SAME coin's value `steps` periods
    later, never another coin's, and missing when that row does not exist;
  - fmx_tabpfn.regimes: the BTC regime of date D reads closes dated <= D-1 only, so
    moving the close OF day D (or later) cannot change it;
  - refresh_blocks with the 60-date wall: every 60-day label in a context resolves
    before the block it serves;
  - build_fmx.rank_uncertainty: one expected rank, spread and tail per row, in range.

Offline, no model is loaded. Run with:  python -m unittest discover -s src/tests
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.data_mani.fmx_tabpfn as T
from src.data_mani.build_fmx import rank_uncertainty
from src.data_mani.foundation_repr import refresh_blocks


class Ahead(unittest.TestCase):

    def test_same_coin_steps_later(self):
        df = pd.DataFrame({"symbol": ["A"] * 5 + ["B"] * 5, "_pos": list(range(5)) * 2,
                           "v": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]})
        got = T.ahead(df, "v", steps=2)
        np.testing.assert_array_equal(got[:3], [3, 4, 5])
        np.testing.assert_array_equal(got[5:8], [30, 40, 50])
        self.assertTrue(np.isnan(got[3]) and np.isnan(got[4]) and np.isnan(got[9]))


class Regimes(unittest.TestCase):

    def _regimes(self, closes):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spot.parquet")
            closes.to_parquet(path, index=False)
            old, T.SPOT = T.SPOT, path
            try:
                return T.regimes(pd.date_range("2024-03-01", "2024-03-20").to_numpy())
            finally:
                T.SPOT = old

    def test_day_d_close_does_not_move_day_d_regime(self):
        days = pd.date_range("2024-01-01", "2024-03-25")
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
        spot = pd.DataFrame({"ucid": "1", "date": days, "close": close})
        base = self._regimes(spot)
        d = pd.Timestamp("2024-03-10")
        shocked = spot.copy()
        shocked.loc[shocked["date"] >= d, "close"] *= 50.0       # day D and after
        again = self._regimes(shocked)
        self.assertTrue((base[base.index <= d] == again[again.index <= d]).all())
        # ...and the shock IS visible once it is in the past, so the check above
        # is not passing because regimes never move.
        self.assertTrue((base[base.index > d] != again[again.index > d]).any())


class SixtyDayWall(unittest.TestCase):

    def test_context_labels_resolve_before_block(self):
        for context, block in refresh_blocks(np.arange(500), first=300, refresh=20, window=130, embargo=60):
            self.assertLessEqual(context.max() + 60, block.min())


class RankUncertainty(unittest.TestCase):

    def test_shape_and_range(self):
        rng = np.random.default_rng(1)
        n = 120
        p = rng.dirichlet(np.ones(5), size=n)
        main = pd.DataFrame(p, columns=["tp_p%d" % k for k in range(5)])
        main["date"] = np.repeat(pd.to_datetime(["2024-01-01", "2024-01-02"]), n // 2)
        out = rank_uncertainty(main)
        self.assertEqual(out.shape, (n, 3))
        self.assertTrue(((out[:, 0] >= 0) & (out[:, 0] <= 1)).all())
        self.assertTrue((out[:, 1] >= 0).all())
        self.assertTrue(((out[:, 2] >= -1) & (out[:, 2] <= 1)).all())
        # A coin that is sure to land in the top bin ranks high.
        main.loc[0, ["tp_p%d" % k for k in range(5)]] = [0, 0, 0, 0, 1]
        self.assertGreater(rank_uncertainty(main)[0, 0], 0.8)


if __name__ == "__main__":
    unittest.main()
