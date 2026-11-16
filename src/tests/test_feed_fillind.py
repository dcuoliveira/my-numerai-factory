"""
Test the feed-fillind drop rule: a coin goes only if it has no price on ANY of its
panel rows, judged per (symbol, ucid), so a reused ticker's other coin survives and
a coin with a partial price history is kept whole.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.build_feed_fillind import PRICE, coins_without_prices


class TestCoinsWithoutPrices(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({
            "symbol": ["A", "A", "B", "B", "S", "S"],
            "ucid":   ["1", "1", "2", "2", "6868", "39955"],
            PRICE:    [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
        })

    def test_partial_history_is_kept_whole(self):
        self.assertEqual(coins_without_prices(self._df()).tolist()[:2], [False, False])

    def test_coin_with_no_price_is_dropped(self):
        self.assertEqual(coins_without_prices(self._df()).tolist()[2:4], [True, True])

    def test_reused_ticker_is_judged_per_ucid(self):
        # same symbol: the 2020 coin has no price, the 2026 coin does
        self.assertEqual(coins_without_prices(self._df()).tolist()[4:], [True, False])


if __name__ == "__main__":
    unittest.main()
