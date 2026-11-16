"""
Test the macro-volatility interaction builder: strictly-prior normalisation, macro change definitions,
the regime gates and the coin-level exposures.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani import build_vol_macrovol as M


class TestPriorScores(unittest.TestCase):

    def frame(self, values, key="A"):
        return pd.DataFrame({"k": key, "date": pd.bdate_range("2024-01-01", periods=len(values)), "v": values})

    def test_uses_only_earlier_rows(self):
        f = self.frame(np.arange(1, 200, dtype=float))
        _, p, z = M.prior_scores(f, ["v"], "k", 50, 20, calendar=False)
        # an increasing series beats every earlier value: percentile 1, never more
        self.assertTrue((p["v"].dropna() == 1.0).all())
        self.assertTrue(p["v"].iloc[:20].isna().all())         # fewer than 20 earlier rows
        g = f.copy()
        g.loc[150:, "v"] = 1e9                                   # a later change cannot move an earlier score
        _, p2, _ = M.prior_scores(g, ["v"], "k", 50, 20, calendar=False)
        pd.testing.assert_series_equal(p["v"].iloc[:150], p2["v"].iloc[:150])

    def test_probit_is_finite_and_centred(self):
        rng = np.random.default_rng(0)
        f = self.frame(rng.lognormal(size=600))
        _, p, z = M.prior_scores(f, ["v"], "k", 252, 120, calendar=False)
        self.assertTrue(np.isfinite(z["v"].dropna()).all())
        self.assertLess(abs(z["v"].dropna().mean()), 0.15)


class TestMacro(unittest.TestCase):

    def store(self):
        d = pd.bdate_range("2023-01-02", periods=400)
        rng = np.random.default_rng(1)
        rows = []
        for s, base in (("gold", 180), ("sp500", 4000), ("dxy", 100), ("vix", 18), ("y10", 4.0)):
            x = base * np.exp(np.cumsum(rng.normal(0, 0.01, len(d)))) if s != "y10" else base + np.cumsum(rng.normal(0, 0.05, len(d)))
            rows.append(pd.DataFrame({"date": d, "series": s, "value": x}))
        return pd.concat(rows)

    def test_yield_changes_are_basis_points(self):
        st = self.store()
        ch, _ = M.macro_changes(st, "main")
        y = st[st["series"] == "y10"].set_index("date")["value"]
        np.testing.assert_allclose(ch["y10"].dropna().to_numpy(), (100 * y.diff()).dropna().to_numpy())

    def test_future_placebo_reads_later_values(self):
        st = self.store()
        a, _ = M.macro_changes(st, "main")
        b, _ = M.macro_changes(st, "future")
        np.testing.assert_allclose(b["gold"].dropna().iloc[:50].to_numpy(), a["gold"].dropna().iloc[M.FUTURE_SHIFT:M.FUTURE_SHIFT + 50].to_numpy())

    def test_exposure_recovers_a_planted_beta(self):
        d = pd.bdate_range("2024-01-01", periods=300)
        rng = np.random.default_rng(2)
        rm = pd.Series(rng.normal(0, 0.01, len(d)), index=d)
        rc = 2.0 * rm + rng.normal(0, 0.002, len(d))
        close = pd.DataFrame({"symbol": "C", "date": d, "close": 100 * np.exp(np.cumsum(rc.to_numpy()))})
        ex = M.exposures(close, {"sp500": rm}).set_index("date")
        self.assertAlmostEqual(ex["feature_mvx_sp500_beta_120"].dropna().iloc[-1], 2.0, delta=0.05)
        self.assertGreater(ex["feature_mvx_sp500_r2_120"].dropna().iloc[-1], 0.95)


class TestColumns(unittest.TestCase):

    def test_names_and_pairing(self):
        doc = M.columns()
        self.assertEqual(len(doc), 6 * 22 * 4 + 5 * 2 * 5)
        self.assertEqual(M.partner("feature_vol_yz_60d"), "sqret_60d")
        self.assertEqual(M.partner("feature_vol_ewma_lam97"), "ewma_lam97")


if __name__ == "__main__":
    unittest.main()
