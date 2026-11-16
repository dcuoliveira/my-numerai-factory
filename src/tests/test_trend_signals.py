"""
Tests for the Carver/Clenow trend-following signals. The one that needs real
checking is Clenow -- a vectorised rolling exponential regression that must match
the textbook slope x R^2 -- plus the usual causality and per-symbol isolation.

Offline, sub-second.

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.trend_signals import TREND_SPECS, compute_trend_signals


def _one(close, symbol="A", start="2020-01-01"):
    dates = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": np.asarray(close, float)})


class TestClenow(unittest.TestCase):

    def test_perfect_exponential_trend_gives_slope_and_r2_one(self):
        k, N = 0.01, 60
        close = np.exp(k * np.arange(100))          # log-price is exactly linear
        spec = {"feature_tf_clenow_60": TREND_SPECS["feature_tf_clenow_60"]}
        v = compute_trend_signals(_one(close), spec)["feature_tf_clenow_60"].to_numpy()
        self.assertTrue(np.isnan(v[:N - 1]).all())  # needs a full window
        expected = np.expm1(k * 252)                # slope=k, R^2=1 -> annualised*1
        np.testing.assert_allclose(v[N - 1:], expected, rtol=1e-6)

    def test_downtrend_is_negative_uptrend_positive(self):
        spec = {"feature_tf_clenow_60": TREND_SPECS["feature_tf_clenow_60"]}
        up = compute_trend_signals(_one(np.exp(0.01 * np.arange(80))), spec)
        dn = compute_trend_signals(_one(np.exp(-0.01 * np.arange(80))), spec)
        self.assertGreater(up["feature_tf_clenow_60"].iloc[-1], 0)
        self.assertLess(dn["feature_tf_clenow_60"].iloc[-1], 0)


class TestOtherSignals(unittest.TestCase):

    def test_bbz_sign_matches_side_of_the_mean(self):
        # last point jumps well above its recent mean -> positive z-score
        close = np.r_[np.full(40, 100.0), [130.0]]
        spec = {"feature_tf_bbz_20": TREND_SPECS["feature_tf_bbz_20"]}
        v = compute_trend_signals(_one(close), spec)["feature_tf_bbz_20"].to_numpy()
        self.assertGreater(v[-1], 0)

    def test_newhigh_counts_new_20d_highs_in_last_5(self):
        spec = {"feature_tf_newhigh20in5": TREND_SPECS["feature_tf_newhigh20in5"]}
        up = compute_trend_signals(_one(np.arange(1, 61, dtype=float)), spec)
        dn = compute_trend_signals(_one(np.arange(60, 0, -1, dtype=float)), spec)
        self.assertEqual(up["feature_tf_newhigh20in5"].iloc[-1], 5.0)   # every day a new high
        self.assertEqual(dn["feature_tf_newhigh20in5"].iloc[-1], 0.0)   # never a new high

    def test_ewmac_positive_on_an_uptrend(self):
        rng = np.random.RandomState(0)
        close = 100 * np.cumprod(1 + (0.01 + rng.normal(0, 0.003, 80)))
        spec = {"feature_tf_ewmac_8_32": TREND_SPECS["feature_tf_ewmac_8_32"]}
        v = compute_trend_signals(_one(close), spec)["feature_tf_ewmac_8_32"].dropna()
        self.assertGreater(v.iloc[-1], 0)

    def test_breakout_positive_near_the_top_of_the_range(self):
        close = np.r_[np.linspace(100, 90, 40), np.linspace(90, 120, 20)]  # ends at a high
        spec = {"feature_tf_breakout_20": TREND_SPECS["feature_tf_breakout_20"]}
        v = compute_trend_signals(_one(close), spec)["feature_tf_breakout_20"].dropna()
        self.assertGreater(v.iloc[-1], 0)


class TestCausalityAndIsolation(unittest.TestCase):

    def test_future_price_never_moves_a_past_signal(self):
        rng = np.random.RandomState(1)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 80))
        spec = {"feature_tf_bbz_20": TREND_SPECS["feature_tf_bbz_20"]}
        before = compute_trend_signals(_one(close), spec)["feature_tf_bbz_20"]
        spiked = close.copy(); spiked[70] *= 5
        after = compute_trend_signals(_one(spiked), spec)["feature_tf_bbz_20"]
        pd.testing.assert_series_equal(before.iloc[:70], after.iloc[:70])

    def test_two_symbols_do_not_mix(self):
        up = _one(np.exp(0.01 * np.arange(80)), symbol="A")
        dn = _one(np.exp(-0.01 * np.arange(80)), symbol="B")
        spec = {"feature_tf_clenow_60": TREND_SPECS["feature_tf_clenow_60"]}
        out = compute_trend_signals(pd.concat([up, dn]), spec)
        self.assertGreater(out[out.symbol == "A"]["feature_tf_clenow_60"].iloc[-1], 0)
        self.assertLess(out[out.symbol == "B"]["feature_tf_clenow_60"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
