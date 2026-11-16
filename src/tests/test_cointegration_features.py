"""
Tests for the cointegration-residual pipeline: the AR(1) stationarity math, the
residual z-score, causality, and the Numerai scale.
Run with:  python -m unittest src.tests.test_cointegration_features
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_LABELS, bin_cross_sectional
from src.data_mani.cointegration_features import (COINT_SPECS, _ar1_stats,
                                                  compute_cointegration_signals)


def _spot(close, symbol, start="2019-01-01"):
    close = np.asarray(close, dtype="float64")
    d = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame({"symbol": symbol, "ucid": "0", "date": d,
                         "close": close, "volume": 1e6})


def _macro(n, start="2019-01-01"):
    d = pd.date_range(start, periods=n, freq="D")
    base = 100 + np.cumsum(np.random.RandomState(0).normal(0, 0.2, n))
    return pd.DataFrame({"date": d, "gold": base, "dxy": base * 0.9 + 10,
                         "sp500": base * 3, "silver": base, "y1": 1, "y2": 1,
                         "y5": 1, "y10": 1, "y30": 1})


class TestAR1(unittest.TestCase):

    def test_half_life_of_known_ar1(self):
        # r_t = phi r_{t-1} + e, phi=0.9  ->  half-life = -ln2/ln(0.9) ~= 6.58
        rng = np.random.RandomState(0)
        phi = 0.9
        r = np.zeros(4000)
        for t in range(1, len(r)):
            r[t] = phi * r[t - 1] + rng.normal(0, 1)
        hl, adf = _ar1_stats(r, w=500)
        hl = hl[~np.isnan(hl)]; adf = adf[~np.isnan(adf)]
        self.assertTrue(abs(np.median(hl) - (-np.log(2) / np.log(phi))) < 2.0)
        self.assertLess(np.median(adf), -3.0)          # stationary -> strongly negative DF stat

    def test_random_walk_is_not_stationary(self):
        rng = np.random.RandomState(1)
        rw = np.cumsum(rng.normal(0, 1, 4000))          # unit root
        hl, adf = _ar1_stats(rw, w=500)
        adf = adf[~np.isnan(adf)]
        self.assertGreater(np.median(adf), -2.0)        # DF stat not significantly negative


class TestResidualAndScale(unittest.TestCase):

    def _panel(self):
        rng = np.random.RandomState(3)
        n = 400
        # a BTC-like factor; two coins cointegrated with it + stationary residuals
        logbtc = np.cumsum(rng.normal(0, 0.03, n))
        frames = [_spot(np.exp(logbtc), "BTC")]
        for i in range(60):
            resid = np.zeros(n)
            for t in range(1, n):
                resid[t] = 0.85 * resid[t - 1] + rng.normal(0, 0.05)
            coin = np.exp(0.5 + 1.1 * logbtc + resid + 0.001 * i)
            frames.append(_spot(coin, "c%02d" % i))
        return pd.concat(frames, ignore_index=True)

    def test_residual_z_and_adf_are_finite_and_binnable(self):
        spot = self._panel()
        out = compute_cointegration_signals(spot, _macro(spot["date"].nunique()))
        self.assertEqual(set(COINT_SPECS).issubset(set(out.columns)), True)
        z = out["feature_coint_resid_z_120d"].dropna()
        self.assertGreater(len(z), 0)
        self.assertLess(abs(z.std() - 1.0), 0.6)        # z-score ~ unit std over valid rows
        # cointegrated coins -> many strongly-negative DF stats
        adf = out["feature_coint_adf_120d"].dropna()
        self.assertLess(adf.median(), -1.5)
        binned = bin_cross_sectional(out.dropna(subset=["feature_coint_resid_z_60d"]),
                                     ["feature_coint_resid_z_60d"], "date")
        self.assertTrue(set(np.unique(binned["feature_coint_resid_z_60d"].dropna()))
                        .issubset(set(BIN_LABELS)))


class TestCausality(unittest.TestCase):

    def test_future_bar_does_not_move_past_residual(self):
        rng = np.random.RandomState(5)
        n = 300
        logbtc = np.cumsum(rng.normal(0, 0.03, n))
        btc = _spot(np.exp(logbtc), "BTC")
        coin = np.exp(1.0 * logbtc + np.cumsum(rng.normal(0, 0.02, n)))
        spot = pd.concat([btc, _spot(coin, "c0")], ignore_index=True)
        mac = _macro(n)
        before = compute_cointegration_signals(spot, mac)
        spot2 = spot.copy()
        mask = (spot2["symbol"] == "c0")
        idx = spot2[mask].index[200]
        spot2.loc[idx, "close"] *= 3.0                  # spike a future bar for c0
        after = compute_cointegration_signals(spot2, mac)
        b = before[(before.symbol == "c0")].reset_index(drop=True)["feature_coint_resid_z_120d"]
        a = after[(after.symbol == "c0")].reset_index(drop=True)["feature_coint_resid_z_120d"]
        pd.testing.assert_series_equal(b.iloc[:200], a.iloc[:200])


if __name__ == "__main__":
    unittest.main()
