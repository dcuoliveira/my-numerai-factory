"""
Tests for the crypto reversion feature pipeline: the signal math, that nothing
reads a future bar, and that the built columns land on Numerai's scale.
Run with:  python -m unittest src.tests.test_reversion_features
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_LABELS, bin_cross_sectional
from src.data_mani.reversion_features import (REVERSION_SPECS,
                                              compute_reversion_signals)


def _panel(o, h, l, c, v=None, symbol="A", start="2020-01-01"):
    c = np.asarray(c, dtype="float64")
    n = len(c)
    v = np.full(n, 1e6) if v is None else np.asarray(v, dtype="float64")
    return pd.DataFrame({"symbol": symbol, "date": pd.date_range(start, periods=n, freq="D"),
                         "open": o, "high": h, "low": l, "close": c, "volume": v})


def _from_close(c, **kw):
    c = np.asarray(c, dtype="float64")
    return _panel(c, c, c, c, **kw)


class TestSpecGrid(unittest.TestCase):
    def test_count_and_prefix(self):
        self.assertEqual(len(REVERSION_SPECS), 14)
        self.assertTrue(all(k.startswith("feature_rev_") for k in REVERSION_SPECS))


class TestMath(unittest.TestCase):

    def test_short_term_reversal_is_prior_return(self):
        c = 2.0 ** np.arange(20)                      # doubles daily
        out = compute_reversion_signals(_from_close(c),
                                        {"feature_rev_str_7d": REVERSION_SPECS["feature_rev_str_7d"]})
        v = out["feature_rev_str_7d"].to_numpy()
        self.assertTrue(np.isnan(v[:7]).all())
        np.testing.assert_allclose(v[7:], 2.0 ** 7 - 1.0, rtol=1e-9)

    def test_rsi_is_100_on_pure_uptrend(self):
        c = np.cumprod(np.r_[100.0, np.full(20, 1.01)])    # every day a gain
        out = compute_reversion_signals(_from_close(c),
                                        {"feature_rev_rsi_14d": REVERSION_SPECS["feature_rev_rsi_14d"]})
        v = out["feature_rev_rsi_14d"].dropna().to_numpy()
        np.testing.assert_allclose(v, 100.0, atol=1e-9)

    def test_stochastic_is_one_at_window_high(self):
        # strictly increasing close with high==close -> close is always the window max -> %K = 1
        c = np.linspace(100, 160, 30)
        out = compute_reversion_signals(_panel(c, c, c - 1.0, c),
                                        {"feature_rev_stoch_14d": REVERSION_SPECS["feature_rev_stoch_14d"]})
        v = out["feature_rev_stoch_14d"].dropna().to_numpy()
        np.testing.assert_allclose(v, 1.0, atol=1e-9)

    def test_distance_from_ma_is_exact(self):
        c = np.r_[np.full(19, 100.0), 110.0]               # flat then a jump on the last day
        out = compute_reversion_signals(_from_close(c),
                                        {"feature_rev_dist_ma_20d": REVERSION_SPECS["feature_rev_dist_ma_20d"]})
        v = out["feature_rev_dist_ma_20d"].to_numpy()
        sma = np.mean(c[-20:])                              # mean over the trailing 20
        self.assertAlmostEqual(v[-1], c[-1] / sma - 1.0, places=9)

    def test_bbz_sign_above_mean(self):
        rng = np.random.RandomState(0)
        c = 100 + np.cumsum(rng.normal(0, 1, 60))
        out = compute_reversion_signals(_from_close(c),
                                        {"feature_rev_bbz_20d": REVERSION_SPECS["feature_rev_bbz_20d"]})
        bbz = out["feature_rev_bbz_20d"].to_numpy()
        sma = pd.Series(c).rolling(20).mean().to_numpy()
        ok = ~np.isnan(bbz)
        # sign of z-score matches sign of (close - SMA)
        self.assertTrue(np.all(np.sign(bbz[ok]) == np.sign((c - sma)[ok])))


class TestCausality(unittest.TestCase):
    def test_no_lookahead(self):
        rng = np.random.RandomState(1)
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
        o = c * (1 + rng.normal(0, 0.003, 60))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, 60)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, 60)))
        specs = {n: REVERSION_SPECS[n] for n in
                 ("feature_rev_str_7d", "feature_rev_rsi_14d", "feature_rev_bbz_20d",
                  "feature_rev_cci_20d", "feature_rev_dist_ma_50d")}
        before = compute_reversion_signals(_panel(o, hi, lo, c), specs)
        c2, o2, hi2, lo2 = c.copy(), o.copy(), hi.copy(), lo.copy()
        c2[50] *= 1.5; hi2[50] *= 1.5; lo2[50] /= 1.5
        after = compute_reversion_signals(_panel(o2, hi2, lo2, c2), specs)
        for col in specs:
            pd.testing.assert_series_equal(before[col].iloc[:50], after[col].iloc[:50])


class TestScale(unittest.TestCase):
    def test_binned_onto_numerai_scale(self):
        rng = np.random.RandomState(2)
        frames = []
        for i in range(300):
            c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 40))
            frames.append(_from_close(c, symbol="s%03d" % i))
        panel = pd.concat(frames, ignore_index=True)
        cols = ["feature_rev_str_7d", "feature_rev_rsi_14d"]
        raw = compute_reversion_signals(panel, {c: REVERSION_SPECS[c] for c in cols})
        binned = bin_cross_sectional(raw, cols, "date")
        for c in cols:
            self.assertTrue(set(np.unique(binned[c].dropna())).issubset(set(BIN_LABELS)))
            self.assertEqual(binned[c].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
