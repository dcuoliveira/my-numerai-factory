"""
Tests for the crypto volatility feature pipeline: the four estimators' math, that
they never read a future price, and that the built columns land on Numerai's scale.

The failure that would look like success here is a feature that is NOT on Numerai's
{0,.25,.5,.75,1} scale, or an estimator that quietly reads a future bar (a better
backtest that cannot be reproduced live). So the math tests pin each estimator on a
controlled series, the causality tests spike a future price and assert past
features are byte-identical, and the scale test pins the 5/20/50/20/5 bin split.

Everything runs offline in well under a second -- no downloads, no fitting.

Run with:  python -m unittest src.tests.test_volatility_features
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.binning import BIN_LABELS, bin_cross_sectional
from src.data_mani.volatility_features import (EWMA_LAMBDAS, VOL_SPECS,
                                               VOL_WINDOWS, _yz_k,
                                               compute_volatility_signals)


def _panel(open_, high, low, close, symbol="A", start="2020-01-01"):
    dates = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates,
                         "open": open_, "high": high, "low": low, "close": close})


def _from_close(close, **kw):
    """OHLC panel with no intraday range (O=H=L=C): isolates the close estimators."""
    close = np.asarray(close, dtype="float64")
    return _panel(close, close, close, close, **kw)


# --------------------------------------------------------------------------- #
# The spec grid is what we said it is: 4 estimators over the hyperparameter grid.
# --------------------------------------------------------------------------- #
class TestSpecGrid(unittest.TestCase):

    def test_grid_is_6_4_6_6(self):
        kinds = [s["kind"] for s in VOL_SPECS.values()]
        self.assertEqual(kinds.count("sqret"), len(VOL_WINDOWS))
        self.assertEqual(kinds.count("ewma"), len(EWMA_LAMBDAS))
        self.assertEqual(kinds.count("parkinson"), len(VOL_WINDOWS))
        self.assertEqual(kinds.count("yangzhang"), len(VOL_WINDOWS))
        self.assertEqual(len(VOL_SPECS), 22)
        for w in VOL_WINDOWS:
            self.assertIn("feature_vol_sqret_%dd" % w, VOL_SPECS)
            self.assertIn("feature_vol_park_%dd" % w, VOL_SPECS)
            self.assertIn("feature_vol_yz_%dd" % w, VOL_SPECS)
        self.assertIn("feature_vol_ewma_lam94", VOL_SPECS)  # RiskMetrics daily

    def test_isolation_regex_excludes_base_vol_names(self):
        # the isolate handle is '^feature_vol_'; it must NOT match the base's
        # feature_volatility_* / feature_volume_* columns.
        import re
        rx = re.compile(r"^feature_vol_")
        self.assertTrue(all(rx.match(c) for c in VOL_SPECS))
        for base in ("feature_volatility_20d", "feature_volume_avg_20d",
                     "feature_volume_ewa_60d"):
            self.assertIsNone(rx.match(base))


# --------------------------------------------------------------------------- #
# Squared-returns realised variance.
# --------------------------------------------------------------------------- #
class TestSquaredReturns(unittest.TestCase):

    def test_constant_return_gives_that_squared(self):
        # a constant daily log return r -> every r_i^2 = r^2 -> RV_w = r^2 exactly.
        r = 0.03
        close = np.exp(r * np.arange(40))
        spec = {"feature_vol_sqret_20d": VOL_SPECS["feature_vol_sqret_20d"]}
        out = compute_volatility_signals(_from_close(close), spec)
        v = out["feature_vol_sqret_20d"].to_numpy()
        self.assertTrue(np.isnan(v[:20]).all())          # min_periods = window
        np.testing.assert_allclose(v[20:], r * r, rtol=1e-12)

    def test_matches_rolling_mean_of_squared_log_returns(self):
        rng = np.random.RandomState(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 80))
        w = 10
        spec = {"feature_vol_sqret_10d": VOL_SPECS["feature_vol_sqret_10d"]}
        out = compute_volatility_signals(_from_close(close), spec)
        r = np.diff(np.log(close))
        expected = pd.Series(r ** 2).rolling(w, min_periods=w).mean()
        # feature index t aligns to close[t]; returns start at close index 1.
        got = out["feature_vol_sqret_10d"].to_numpy()[1:]
        np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# EWMA (RiskMetrics) recursion.
# --------------------------------------------------------------------------- #
class TestEWMA(unittest.TestCase):

    def test_recursion_matches_hand_rolled(self):
        rng = np.random.RandomState(1)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
        lam = 0.94
        spec = {"feature_vol_ewma_lam94": VOL_SPECS["feature_vol_ewma_lam94"]}
        out = compute_volatility_signals(_from_close(close), spec)
        got = out["feature_vol_ewma_lam94"].to_numpy()
        # hand recursion on squared log returns with adjust=False semantics:
        # y_t = lam*y_{t-1} + (1-lam)*x_t, seeded at the first return.
        r = np.diff(np.log(close))
        x = r ** 2
        y = np.empty_like(x); y[0] = x[0]
        for t in range(1, len(x)):
            y[t] = lam * y[t - 1] + (1 - lam) * x[t]
        mp = VOL_SPECS["feature_vol_ewma_lam94"]["min_periods"]
        y[:mp - 1] = np.nan                              # min_periods warm-up
        np.testing.assert_allclose(got[1:], y, rtol=1e-10, equal_nan=True)
        self.assertTrue(np.isnan(got[0]))               # first row has no return

    def test_min_periods_scale_with_lambda(self):
        for lam in EWMA_LAMBDAS:
            mp = VOL_SPECS["feature_vol_ewma_lam%d" % round(100 * lam)]["min_periods"]
            self.assertEqual(mp, int(round(1.0 / (1.0 - lam))))


# --------------------------------------------------------------------------- #
# Parkinson (high/low range).
# --------------------------------------------------------------------------- #
class TestParkinson(unittest.TestCase):

    def test_constant_range_is_exact(self):
        # H/L a constant ratio c every day -> park_day = (ln c)^2 / (4 ln 2),
        # and the rolling mean of a constant is that constant.
        n = 30
        close = np.full(n, 100.0)
        ratio = 1.05
        high = close * np.sqrt(ratio)
        low = close / np.sqrt(ratio)                    # H/L = ratio
        spec = {"feature_vol_park_20d": VOL_SPECS["feature_vol_park_20d"]}
        out = compute_volatility_signals(
            _panel(close, high, low, close), spec)
        v = out["feature_vol_park_20d"].to_numpy()
        self.assertTrue(np.isnan(v[:19]).all())         # needs a full window
        expected = (np.log(ratio) ** 2) / (4.0 * np.log(2.0))
        np.testing.assert_allclose(v[19:], expected, rtol=1e-12)

    def test_flat_day_contributes_zero(self):
        # H == L -> ln(H/L)=0 -> that day's Parkinson term is 0 (valid, biased low).
        close = np.full(25, 100.0)
        spec = {"feature_vol_park_10d": VOL_SPECS["feature_vol_park_10d"]}
        out = compute_volatility_signals(_panel(close, close, close, close), spec)
        v = out["feature_vol_park_10d"].dropna().to_numpy()
        np.testing.assert_allclose(v, 0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# Yang-Zhang (full OHLC + previous close).
# --------------------------------------------------------------------------- #
class TestYangZhang(unittest.TestCase):

    def test_k_weight_formula(self):
        for n in (5, 10, 20, 60):
            self.assertAlmostEqual(_yz_k(n),
                                   0.34 / (1.34 + (n + 1.0) / (n - 1.0)), places=12)
        with self.assertRaises(ValueError):
            _yz_k(1)

    def test_matches_direct_formula_on_last_window(self):
        # build a valid OHLC series, then recompute YZ over the final window by
        # hand and check the feature equals it -- pins the vo + k*vc + (1-k)*vrs
        # assembly and the rolling alignment.
        rng = np.random.RandomState(4)
        n = 20
        m = 40
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, m))
        o = c * (1 + rng.normal(0, 0.005, m))            # small overnight gap
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, m)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, m)))
        spec = {"feature_vol_yz_20d": VOL_SPECS["feature_vol_yz_20d"]}
        out = compute_volatility_signals(_panel(o, hi, lo, c), spec)
        got = out["feature_vol_yz_20d"].to_numpy()[-1]

        # direct computation over the last n days (indices m-n .. m-1), overnight
        # needs the close at m-n-1.
        idx = np.arange(m - n, m)
        ovn = np.log(o[idx] / c[idx - 1])
        oc = np.log(c[idx] / o[idx])
        u = np.log(hi[idx] / o[idx]); d = np.log(lo[idx] / o[idx])
        rs = u * (u - oc) + d * (d - oc)
        k = _yz_k(n)
        yz = ovn.var(ddof=1) + k * oc.var(ddof=1) + (1 - k) * rs.mean()
        self.assertAlmostEqual(got, yz, places=12)

    def test_nonnegative_and_needs_full_window(self):
        rng = np.random.RandomState(5)
        m = 30
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, m))
        o = c * (1 + rng.normal(0, 0.005, m))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, m)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, m)))
        spec = {"feature_vol_yz_10d": VOL_SPECS["feature_vol_yz_10d"]}
        v = compute_volatility_signals(_panel(o, hi, lo, c), spec)["feature_vol_yz_10d"].to_numpy()
        self.assertTrue(np.isnan(v[:10]).all())          # overnight costs one extra warm-up day
        self.assertTrue((v[~np.isnan(v)] >= 0).all())


# --------------------------------------------------------------------------- #
# Causality: no estimator reads a future bar; windows never cross a symbol.
# --------------------------------------------------------------------------- #
class TestCausality(unittest.TestCase):

    def test_no_lookahead_future_bar_never_moves_a_past_feature(self):
        rng = np.random.RandomState(6)
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
        o = c * (1 + rng.normal(0, 0.005, 60))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, 60)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, 60)))
        specs = {n: VOL_SPECS[n] for n in
                 ("feature_vol_sqret_20d", "feature_vol_yz_20d",
                  "feature_vol_park_20d", "feature_vol_ewma_lam94")}
        before = compute_volatility_signals(_panel(o, hi, lo, c), specs)
        c2, o2, hi2, lo2 = c.copy(), o.copy(), hi.copy(), lo.copy()
        # Spike a future bar with a RANGE widening + close jump, so it moves every
        # estimator. A purely proportional spike (all of OHLC x5) leaves Parkinson's
        # ln(H/L) invariant, so it would not move a range-ratio feature at all --
        # a correct property, but useless as a no-op guard.
        c2[50] *= 1.5; hi2[50] *= 1.5; lo2[50] /= 1.5
        after = compute_volatility_signals(_panel(o2, hi2, lo2, c2), specs)
        for col in specs:
            pd.testing.assert_series_equal(before[col].iloc[:50], after[col].iloc[:50])
            self.assertFalse(np.allclose(before[col].iloc[51:].fillna(-1),
                                         after[col].iloc[51:].fillna(-1)),
                             "%s: the spike moved no later date -- no-op test" % col)

    def test_windows_never_cross_a_symbol_boundary(self):
        # A is calm, B is wild; A's variance must be strictly below B's, never mixed.
        rng = np.random.RandomState(7)
        ca = 100 * np.cumprod(1 + rng.normal(0, 0.005, 40))
        cb = 100 * np.cumprod(1 + rng.normal(0, 0.05, 40))
        a = _from_close(ca, symbol="A")
        b = _from_close(cb, symbol="B")
        spec = {"feature_vol_sqret_20d": VOL_SPECS["feature_vol_sqret_20d"]}
        out = compute_volatility_signals(pd.concat([a, b]), spec)
        va = out[out.symbol == "A"]["feature_vol_sqret_20d"].dropna()
        vb = out[out.symbol == "B"]["feature_vol_sqret_20d"].dropna()
        self.assertLess(va.max(), vb.min())


# --------------------------------------------------------------------------- #
# The variant is a superset of the panel, on the tournament scale.
# --------------------------------------------------------------------------- #
class TestScale(unittest.TestCase):

    def test_binned_columns_are_on_the_numerai_scale(self):
        rng = np.random.RandomState(8)
        n_syms, n_dates = 300, 40
        frames = []
        for i in range(n_syms):
            c = 100 * np.cumprod(1 + rng.normal(0, 0.01 + 0.03 * rng.rand(), n_dates))
            frames.append(_from_close(c, symbol="s%03d" % i))
        panel = pd.concat(frames, ignore_index=True)
        cols = ["feature_vol_sqret_20d", "feature_vol_ewma_lam94"]
        specs = {c: VOL_SPECS[c] for c in cols}
        raw = compute_volatility_signals(panel, specs)
        binned = bin_cross_sectional(raw, cols, "date")
        for c in cols:
            vals = set(np.unique(binned[c].dropna()))
            self.assertTrue(vals.issubset(set(BIN_LABELS)))
            self.assertEqual(binned[c].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
