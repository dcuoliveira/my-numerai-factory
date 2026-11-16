"""
Test the binned causal encoding: same means, Numerai's scale, and no later date reaching back.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.build_feature_encoding import causal_feature_means
from src.data_mani.history_features import bin_ties_preserved

F = "feature_x_20d"
E = F + "_symmean"


def panel(n_symbols=40, n_dates=30, seed=0):
    values = np.random.default_rng(seed).choice([0, .25, .5, .75, 1], size=n_symbols * n_dates)
    return pd.DataFrame({
        "symbol": np.repeat(["S%02d" % i for i in range(n_symbols)], n_dates),
        "date": np.tile(pd.bdate_range("2024-01-01", periods=n_dates), n_symbols),
        F: values})


def binned(df):
    encoded, _ = causal_feature_means(df, [F], "symbol", "date", 50.0)
    return bin_ties_preserved(encoded, df["date"])[E]


class TestBinnedEncoding(unittest.TestCase):

    def test_values_on_numerai_scale(self):
        self.assertTrue(set(binned(panel()).unique()) <= {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_first_date_is_neutral(self):
        df = panel()
        self.assertTrue((binned(df)[df["date"] == df["date"].min()] == 0.5).all())

    def test_later_dates_do_not_change_earlier_bins(self):
        df = panel()
        cut = df["date"].sort_values().unique()[20]
        early = df[df["date"] <= cut]
        encoded, _ = causal_feature_means(df, [F], "symbol", "date", 50.0, prior={F: 0.5})
        short, _ = causal_feature_means(early, [F], "symbol", "date", 50.0, prior={F: 0.5})
        full = bin_ties_preserved(encoded, df["date"])[E].loc[early.index]
        pd.testing.assert_series_equal(full, bin_ties_preserved(short, early["date"])[E])

    def test_order_within_a_date_follows_the_mean(self):
        df = panel()
        encoded, _ = causal_feature_means(df, [F], "symbol", "date", 50.0)
        last = df["date"] == df["date"].max()
        pairs = pd.DataFrame({"mean": encoded[E][last], "bin": binned(df)[last]})
        self.assertTrue(pairs.groupby("bin")["mean"].min().is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
