"""Tests for the perp/spot basis + funding features.
Run with:  python -m unittest src.tests.test_basis_features"""
import unittest
import numpy as np, pandas as pd

from src.data_mani.binning import BIN_LABELS, bin_cross_sectional
from src.data_mani.basis_features import BASIS_FEATURES, compute_basis_signals


def _panel(spot, perp, funding, symbol="A", start="2020-01-01"):
    n = len(spot)
    return pd.DataFrame({"symbol": symbol, "date": pd.date_range(start, periods=n, freq="D"),
                         "spot_close": np.asarray(spot, float),
                         "perp_close": np.asarray(perp, float),
                         "funding": np.asarray(funding, float)})


class TestBasisMath(unittest.TestCase):

    def test_basis_pct_is_exact(self):
        spot = np.full(30, 100.0); perp = spot * 1.05
        out = compute_basis_signals(_panel(spot, perp, np.zeros(30)))
        np.testing.assert_allclose(out["feature_basis_pct"].to_numpy(), 0.05, rtol=1e-12)

    def test_funding_passthrough_and_ma(self):
        n = 40; f = np.full(n, 0.0003)
        out = compute_basis_signals(_panel(np.full(n, 100.0), np.full(n, 100.0), f))
        np.testing.assert_allclose(out["feature_funding_rate"].to_numpy(), 0.0003, rtol=1e-12)
        ma = out["feature_funding_ma_7d"].dropna().to_numpy()
        np.testing.assert_allclose(ma, 0.0003, rtol=1e-9)

    def test_basis_change_is_diff(self):
        spot = np.full(30, 100.0)
        perp = spot * (1 + np.linspace(0, 0.029, 30))     # basis rises linearly
        out = compute_basis_signals(_panel(spot, perp, np.zeros(30)))
        chg = out["feature_basis_chg_5d"].to_numpy()
        self.assertTrue(np.all(chg[5:] > 0))              # rising basis -> positive change

    def test_missing_perp_gives_nan_basis(self):
        spot = np.full(10, 100.0); perp = np.full(10, np.nan)
        out = compute_basis_signals(_panel(spot, perp, np.zeros(10)))
        self.assertTrue(out["feature_basis_pct"].isna().all())


class TestCausality(unittest.TestCase):
    def test_no_lookahead(self):
        rng = np.random.RandomState(0); n = 60
        spot = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        perp = spot * (1 + rng.normal(0, 0.002, n))
        fund = rng.normal(0, 0.0002, n)
        before = compute_basis_signals(_panel(spot, perp, fund))
        perp2 = perp.copy(); perp2[50] *= 1.1
        after = compute_basis_signals(_panel(spot, perp2, fund))
        for c in ("feature_basis_pct", "feature_basis_z_20d", "feature_basis_chg_5d"):
            pd.testing.assert_series_equal(before[c].iloc[:50], after[c].iloc[:50])


class TestScale(unittest.TestCase):
    def test_binnable(self):
        rng = np.random.RandomState(1); frames = []
        for i in range(300):
            spot = 100 * np.cumprod(1 + rng.normal(0, 0.02, 40))
            perp = spot * (1 + rng.normal(0, 0.003, 40))
            frames.append(_panel(spot, perp, rng.normal(0, 0.0002, 40), symbol="s%03d" % i))
        out = compute_basis_signals(pd.concat(frames, ignore_index=True))
        binned = bin_cross_sectional(out, ["feature_basis_pct", "feature_funding_rate"], "date")
        for c in ("feature_basis_pct", "feature_funding_rate"):
            self.assertTrue(set(np.unique(binned[c].dropna())).issubset(set(BIN_LABELS)))


if __name__ == "__main__":
    unittest.main()
