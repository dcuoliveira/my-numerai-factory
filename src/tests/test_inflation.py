"""
Tests for the inflation block: break-even levels (nominal - real), the 5y5y
forward, and the momentum (change) of the levels. Checked on the RAW values
(normalize=False) so the arithmetic is exact.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.macro_features import INFLATION_SPECS, compute_macro_signals


class TestInflation(unittest.TestCase):

    def _macro(self, n=80):
        dates = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({
            "date": dates,
            "y5": np.full(n, 4.0), "y10": np.full(n, 4.5), "y30": np.full(n, 5.0),
            "r5": np.full(n, 1.5), "r10": np.full(n, 2.0), "r30": np.full(n, 2.5),
        })

    def test_breakeven_is_nominal_minus_real(self):
        out = compute_macro_signals(self._macro(), INFLATION_SPECS, normalize=False)
        self.assertTrue(np.allclose(out["feature_macro_bei_5y"], 4.0 - 1.5))    # 2.5
        self.assertTrue(np.allclose(out["feature_macro_bei_10y"], 4.5 - 2.0))   # 2.5
        self.assertTrue(np.allclose(out["feature_macro_bei_30y"], 5.0 - 2.5))   # 2.5

    def test_5y5y_forward_formula(self):
        out = compute_macro_signals(self._macro(), INFLATION_SPECS, normalize=False)
        bei5, bei10 = 2.5, 2.5
        expected = (10 * bei10 - 5 * bei5) / 5.0
        self.assertTrue(np.allclose(out["feature_macro_bei_5y5y"], expected))

    def test_momentum_of_level_is_its_change(self):
        # a rising 10y break-even: y10 climbs, real flat -> BEI climbs linearly
        n = 80
        m = self._macro(n)
        m["y10"] = np.linspace(4.5, 6.5, n)      # +2.0 over 79 steps
        out = compute_macro_signals(m, INFLATION_SPECS, normalize=False)
        v = out["feature_macro_bei_10y_chg20"].to_numpy()
        self.assertTrue(np.isnan(v[:20]).all())
        step = 2.0 / (n - 1)                     # per-day change in BEI
        np.testing.assert_allclose(v[20:], 20 * step, rtol=1e-9)

    def test_normalized_is_bounded(self):
        out = compute_macro_signals(self._macro(400), INFLATION_SPECS, normalize=True,
                                    window=60, min_periods=20)
        # constant levels -> percentile is degenerate but must stay in [0,1] or NaN
        for c in [c for c in out.columns if c != "date"]:
            v = out[c].dropna().to_numpy()
            if v.size:
                self.assertTrue(np.all((v >= 0) & (v <= 1.0)), c)


if __name__ == "__main__":
    unittest.main()
