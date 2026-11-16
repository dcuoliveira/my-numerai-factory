"""
Tests for the crypto momentum feature pipeline: binning, the momentum math, and
the merge onto the tournament panel.

The failure that would look like success here is a feature that is NOT on
Numerai's scale (a booster splits on raw magnitude, the neutral-fill stops
meaning "middle of the field") or a signal that quietly reads a future price (a
better backtest that cannot be reproduced live). So the binning tests pin the
exact 5/20/50/20/5 mass split, and the momentum tests pin causality directly.

Everything runs offline in well under a second -- no downloads, no fitting.

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.binning import (BIN_LABELS, BIN_MASS, bin_cross_sectional,
                                    bin_mass, rank_to_bins)
from src.data_mani.momentum_features import (MOMENTUM_SPECS,
                                             attach_signals_asof,
                                             compute_momentum_signals)


# --------------------------------------------------------------------------- #
# Binning: it must reproduce Numerai's cross-sectional rank-bin exactly.
# --------------------------------------------------------------------------- #
class TestBinning(unittest.TestCase):

    def test_five_labels_and_mass_split_at_n300(self):
        # A continuous, strictly increasing signal over 300 symbols must bin to
        # exactly 15/60/150/60/15 -- the counts the shipped feature_momentum_20d
        # shows on a 300-symbol date.
        binned = rank_to_bins(np.arange(300, dtype="float64"))
        counts = [int(np.isclose(binned, lab).sum()) for lab in BIN_LABELS]
        self.assertEqual(counts, [15, 60, 150, 60, 15])
        self.assertTrue(set(np.unique(binned)).issubset(set(BIN_LABELS)))

    def test_mass_split_holds_across_sizes(self):
        for n in (241, 300, 512, 1000):
            mass = bin_mass(rank_to_bins(np.random.RandomState(0).randn(n)))
            self.assertTrue(np.max(np.abs(mass - BIN_MASS)) < 0.01,
                            "mass off at n=%d: %s" % (n, mass))

    def test_monotone_in_the_raw_signal(self):
        rng = np.random.RandomState(1)
        raw = rng.randn(500)
        binned = rank_to_bins(raw)
        order = np.argsort(raw)
        # sorted by raw, the bins never decrease
        self.assertTrue(np.all(np.diff(binned[order]) >= 0))

    def test_nan_is_preserved_not_ranked(self):
        raw = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        binned = rank_to_bins(raw)
        self.assertTrue(binned.isna().equals(raw.isna()))
        # the three present values are ranked among themselves only
        self.assertEqual(int(binned.notna().sum()), 3)

    def test_all_nan_cross_section_stays_nan(self):
        binned = rank_to_bins([np.nan, np.nan, np.nan])
        self.assertTrue(binned.isna().all())

    def test_bins_each_date_independently(self):
        # date d1 ascending, date d2 descending on the same symbols; the top
        # symbol on d1 is the bottom symbol on d2, so their bins must flip.
        n = 300
        syms = ["s%03d" % i for i in range(n)]
        d1 = pd.DataFrame({"date": "2020-01-01", "symbol": syms,
                           "feature_mom_x": np.arange(n, dtype="float64")})
        d2 = pd.DataFrame({"date": "2020-01-02", "symbol": syms,
                           "feature_mom_x": np.arange(n, dtype="float64")[::-1]})
        out = bin_cross_sectional(pd.concat([d1, d2]), ["feature_mom_x"], "date")
        top_s = syms[-1]
        v1 = out[(out.date == "2020-01-01") & (out.symbol == top_s)]["feature_mom_x"].iloc[0]
        v2 = out[(out.date == "2020-01-02") & (out.symbol == top_s)]["feature_mom_x"].iloc[0]
        self.assertEqual(v1, 1.0)   # highest on d1
        self.assertEqual(v2, 0.0)   # lowest on d2
        # output is on the tournament dtype/scale
        self.assertEqual(out["feature_mom_x"].dtype, np.float32)


# --------------------------------------------------------------------------- #
# Momentum math: values are right and never read a future price.
# --------------------------------------------------------------------------- #
def _one_symbol(close, symbol="A", start="2020-01-01"):
    dates = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": close})


class TestMomentumMath(unittest.TestCase):

    def test_cumulative_return_is_exact(self):
        # doubling every day: ret over w days = 2**w - 1, exactly.
        close = 2.0 ** np.arange(40)
        spec = {"feature_mom_ret_7d": MOMENTUM_SPECS["feature_mom_ret_7d"]}
        out = compute_momentum_signals(_one_symbol(close), spec)
        v = out["feature_mom_ret_7d"].to_numpy()
        self.assertTrue(np.isnan(v[:7]).all())            # min_periods = window
        np.testing.assert_allclose(v[7:], 2.0 ** 7 - 1.0, rtol=1e-9)

    def test_max1_is_the_window_max_daily_return(self):
        # flat except one +50% jump; MAX(1) over any window covering it is 0.5
        close = np.full(40, 100.0)
        close[10:] *= 1.5                                  # a single +50% day at t=10
        spec = {"feature_mom_max1_28d": MOMENTUM_SPECS["feature_mom_max1_28d"]}
        out = compute_momentum_signals(_one_symbol(close), spec)
        v = out["feature_mom_max1_28d"].to_numpy()
        # windows fully after the jump see only 0% days -> max 0
        self.assertAlmostEqual(v[39], 0.0, places=9)
        # a window covering the jump sees the 0.5 day
        self.assertAlmostEqual(v[28], 0.5, places=9)

    def test_skip_month_excludes_the_recent_window(self):
        spec = {"feature_mom_ret_12_1": MOMENTUM_SPECS["feature_mom_ret_12_1"]}
        w, skip = 330, 30
        close = np.exp(np.linspace(0, 3, w + skip + 5))   # smooth log-ramp
        out = compute_momentum_signals(_one_symbol(close), spec)
        v = out["feature_mom_ret_12_1"].to_numpy()
        self.assertTrue(np.isnan(v[:w + skip]).all())
        # value at t = close(t-skip)/close(t-skip-w) - 1
        t = w + skip
        expected = close[t - skip] / close[t - skip - w] - 1.0
        self.assertAlmostEqual(v[t], expected, places=9)

    def test_no_lookahead_a_future_price_never_moves_a_past_feature(self):
        # Compute ret_7d, then 10x a future price and recompute: every feature
        # at a date strictly before that price must be byte-identical.
        rng = np.random.RandomState(3)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
        spec = {"feature_mom_ret_7d": MOMENTUM_SPECS["feature_mom_ret_7d"]}
        before = compute_momentum_signals(_one_symbol(close), spec)["feature_mom_ret_7d"]
        spiked = close.copy(); spiked[50] *= 10.0
        after = compute_momentum_signals(_one_symbol(spiked), spec)["feature_mom_ret_7d"]
        # dates with index < 50 use only prices <= their own date, all < 50
        pd.testing.assert_series_equal(before.iloc[:50], after.iloc[:50])
        # and the spike DOES move a later date (guards against a no-op test)
        self.assertFalse(np.isclose(before.iloc[50], after.iloc[50]))

    def test_windows_never_cross_a_symbol_boundary(self):
        # two symbols interleaved by date; A doubles, B halves. A's ret must be
        # positive and B's negative -- never contaminated by the other.
        up = _one_symbol(2.0 ** np.arange(20), symbol="A")
        down = _one_symbol(0.5 ** np.arange(20), symbol="B")
        spec = {"feature_mom_ret_7d": MOMENTUM_SPECS["feature_mom_ret_7d"]}
        out = compute_momentum_signals(pd.concat([up, down]), spec)
        a = out[out.symbol == "A"]["feature_mom_ret_7d"].dropna()
        b = out[out.symbol == "B"]["feature_mom_ret_7d"].dropna()
        self.assertTrue((a > 0).all())
        self.assertTrue((b < 0).all())


# --------------------------------------------------------------------------- #
# Tzotchev (2018) robust trend signal.
# --------------------------------------------------------------------------- #
class TestRobustTrend(unittest.TestCase):

    def _from_dret(self, dret, symbol="A"):
        close = 100.0 * np.cumprod(1.0 + np.asarray(dret, dtype="float64"))
        return _one_symbol(close, symbol=symbol)

    def test_clean_uptrend_saturates_near_plus_one(self):
        rng = np.random.RandomState(0)
        dret = 0.02 + rng.normal(0, 0.002, 30)          # strong, low-noise up-trend
        spec = {"feature_mom_trend_14d": MOMENTUM_SPECS["feature_mom_trend_14d"]}
        v = compute_momentum_signals(self._from_dret(dret), spec)["feature_mom_trend_14d"].to_numpy()
        self.assertTrue(np.isnan(v[:14]).all())          # needs a full 14-window of returns
        self.assertGreater(v[-1], 0.9)

    def test_clean_downtrend_saturates_near_minus_one(self):
        rng = np.random.RandomState(1)
        dret = -0.02 + rng.normal(0, 0.002, 30)
        spec = {"feature_mom_trend_14d": MOMENTUM_SPECS["feature_mom_trend_14d"]}
        v = compute_momentum_signals(self._from_dret(dret), spec)["feature_mom_trend_14d"].to_numpy()
        self.assertLess(v[-1], -0.9)

    def test_zero_mean_trend_is_exactly_zero(self):
        # exactly alternating returns -> window mean 0 -> t-stat 0 -> S = 0
        dret = np.tile([0.01, -0.01], 20)
        spec = {"feature_mom_trend_14d": MOMENTUM_SPECS["feature_mom_trend_14d"]}
        v = compute_momentum_signals(self._from_dret(dret), spec)["feature_mom_trend_14d"].dropna().to_numpy()
        self.assertTrue(np.all(np.abs(v) < 1e-9))

    def test_bounded_in_minus_one_one(self):
        rng = np.random.RandomState(2)
        dret = rng.normal(0.005, 0.03, 60)
        specs = {n: MOMENTUM_SPECS[n] for n in
                 ["feature_mom_trend_7d", "feature_mom_trend_14d", "feature_mom_trend_30d"]}
        out = compute_momentum_signals(self._from_dret(dret), specs)
        for c in specs:
            v = out[c].dropna().to_numpy()
            self.assertTrue(np.all((v >= -1.0) & (v <= 1.0)))

    def test_composite_is_the_sum_of_its_horizons(self):
        rng = np.random.RandomState(3)
        dret = rng.normal(0.001, 0.02, 320)             # long enough for the 240d horizon
        windows = MOMENTUM_SPECS["feature_mom_trend_robust"]["windows"]
        specs = {("feature_mom_trend_%dd" % T): MOMENTUM_SPECS["feature_mom_trend_%dd" % T]
                 for T in windows}
        specs["feature_mom_trend_robust"] = MOMENTUM_SPECS["feature_mom_trend_robust"]
        out = compute_momentum_signals(self._from_dret(dret), specs)
        per = out[["feature_mom_trend_%dd" % T for T in windows]].sum(axis=1, skipna=False)
        pd.testing.assert_series_equal(out["feature_mom_trend_robust"], per, check_names=False)

    def test_no_lookahead_future_return_never_moves_a_past_signal(self):
        rng = np.random.RandomState(4)
        dret = rng.normal(0.001, 0.02, 60)
        spec = {"feature_mom_trend_14d": MOMENTUM_SPECS["feature_mom_trend_14d"]}
        before = compute_momentum_signals(self._from_dret(dret), spec)["feature_mom_trend_14d"]
        spiked = dret.copy(); spiked[50] = 5.0
        after = compute_momentum_signals(self._from_dret(spiked), spec)["feature_mom_trend_14d"]
        pd.testing.assert_series_equal(before.iloc[:50], after.iloc[:50])


# --------------------------------------------------------------------------- #
# Merge: the variant is a superset of the panel, on the tournament scale.
# --------------------------------------------------------------------------- #
class TestMergeAndScale(unittest.TestCase):

    def _panel(self, n_syms=300, n_dates=8):
        syms = ["s%03d" % i for i in range(n_syms)]
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
        rows = []
        rng = np.random.RandomState(7)
        for d in dates:
            rows.append(pd.DataFrame({
                "symbol": syms, "date": d,
                "feature_close_avg_20d": rng.choice(BIN_LABELS, n_syms),
                "target_binned_return_20": rng.choice(BIN_LABELS, n_syms),
            }))
        return pd.concat(rows, ignore_index=True)

    def test_merge_preserves_rows_and_base_columns(self):
        base = self._panel()
        raw = base[["symbol", "date"]].copy()
        raw["feature_mom_ret_7d"] = np.random.RandomState(9).randn(len(raw))
        merged = base.merge(raw, on=["symbol", "date"], how="left", validate="m:1")
        self.assertEqual(len(merged), len(base))
        # base feature + target columns are untouched
        pd.testing.assert_series_equal(merged["feature_close_avg_20d"],
                                       base["feature_close_avg_20d"])
        binned = bin_cross_sectional(merged, ["feature_mom_ret_7d"], "date")
        # the new column is on the {0,.25,.5,.75,1} scale and prefix-visible
        self.assertTrue(set(np.unique(binned["feature_mom_ret_7d"].dropna()))
                        .issubset(set(BIN_LABELS)))
        feats = [c for c in binned.columns if c.startswith("feature_")]
        self.assertIn("feature_mom_ret_7d", feats)

    def test_missing_symbol_bins_to_nan_then_neutral_fill(self):
        # a symbol absent from the spot feed -> NaN signal -> NaN bin; the loader
        # fills 0.5. Here we check the NaN survives binning (fill is tested in
        # test_pipeline against fill_missing_features).
        base = self._panel(n_syms=10, n_dates=3)
        raw = base[["symbol", "date"]].copy()
        raw["feature_mom_ret_7d"] = 1.0
        raw.loc[raw.symbol == "s000", "feature_mom_ret_7d"] = np.nan
        merged = base.merge(raw, on=["symbol", "date"], how="left")
        binned = bin_cross_sectional(merged, ["feature_mom_ret_7d"], "date")
        self.assertTrue(binned.loc[binned.symbol == "s000",
                                   "feature_mom_ret_7d"].isna().all())


# --------------------------------------------------------------------------- #
# Submission timing: the join must never attach a price we would not have.
# --------------------------------------------------------------------------- #
class TestAsofSubmissionLag(unittest.TestCase):

    def _signals(self, dates, symbol="A"):
        # value on date[i] == i, so the attached value reveals which date won.
        return pd.DataFrame({"symbol": symbol, "date": dates,
                             "feature_mom_ret_7d": np.arange(len(dates), dtype=float)})

    def test_row_D_uses_D_minus_1_not_D(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        sig = self._signals(dates)
        panel = pd.DataFrame({"symbol": "A", "date": dates})
        out = attach_signals_asof(panel, sig, ["feature_mom_ret_7d"],
                                  lag_days=1, max_staleness_days=4)
        v = out["feature_mom_ret_7d"].to_numpy()
        self.assertTrue(np.isnan(v[0]))                       # no D-1 for the first date
        np.testing.assert_array_equal(v[1:], np.arange(0, 9, dtype=float))

    def test_same_day_signal_is_never_used(self):
        # only a same-day signal exists -> NaN, because close(D) is not final at
        # submission; using it would be look-ahead.
        d = pd.Timestamp("2020-01-05")
        panel = pd.DataFrame({"symbol": ["A"], "date": [d]})
        sig = pd.DataFrame({"symbol": ["A"], "date": [d], "feature_mom_ret_7d": [9.0]})
        out = attach_signals_asof(panel, sig, ["feature_mom_ret_7d"],
                                  lag_days=1, max_staleness_days=4)
        self.assertTrue(out["feature_mom_ret_7d"].isna().all())

    def test_freshest_within_lag_includes_the_weekend(self):
        # a Monday round should use Sunday's close (D-1), not Friday's.
        monday = pd.Timestamp("2020-01-06")
        sdates = pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-05"])  # Fri,Sat,Sun
        sig = pd.DataFrame({"symbol": "A", "date": sdates,
                            "feature_mom_ret_7d": [1.0, 2.0, 3.0]})
        panel = pd.DataFrame({"symbol": ["A"], "date": [monday]})
        out = attach_signals_asof(panel, sig, ["feature_mom_ret_7d"],
                                  lag_days=1, max_staleness_days=4)
        self.assertEqual(out["feature_mom_ret_7d"].iloc[0], 3.0)   # Sunday

    def test_staleness_cap_blocks_old_carry_forward(self):
        # newest usable signal is 12 days old, cap is 4 -> NaN, not a stale carry.
        d = pd.Timestamp("2020-01-20")
        sig = pd.DataFrame({"symbol": "A", "date": [pd.Timestamp("2020-01-08")],
                            "feature_mom_ret_7d": [5.0]})
        panel = pd.DataFrame({"symbol": ["A"], "date": [d]})
        out = attach_signals_asof(panel, sig, ["feature_mom_ret_7d"],
                                  lag_days=1, max_staleness_days=4)
        self.assertTrue(out["feature_mom_ret_7d"].isna().all())

    def test_row_order_and_base_columns_preserved(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        panel = pd.DataFrame({"symbol": "A", "date": dates, "keep": [10, 20, 30, 40, 50]})
        out = attach_signals_asof(panel, self._signals(dates),
                                  ["feature_mom_ret_7d"], lag_days=1)
        pd.testing.assert_series_equal(out["keep"], panel["keep"])
        self.assertEqual(list(out["date"]), list(panel["date"]))

    def test_two_symbols_do_not_borrow_each_others_prices(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        a = pd.DataFrame({"symbol": "A", "date": dates, "feature_mom_ret_7d": 1.0})
        b = pd.DataFrame({"symbol": "B", "date": dates, "feature_mom_ret_7d": 9.0})
        panel = pd.DataFrame({"symbol": ["A", "B"], "date": [dates[3], dates[3]]})
        out = attach_signals_asof(panel, pd.concat([a, b]),
                                  ["feature_mom_ret_7d"], lag_days=1)
        got = dict(zip(out["symbol"], out["feature_mom_ret_7d"]))
        self.assertEqual(got["A"], 1.0)
        self.assertEqual(got["B"], 9.0)


# --------------------------------------------------------------------------- #
# The SWAP dataset: --replace-proxies must drop exactly the base momentum proxies.
# --------------------------------------------------------------------------- #
class TestSwapProxyRegex(unittest.TestCase):

    BASE_22 = [
        "feature_bollinger_20d", "feature_bollinger_60d",
        "feature_close_avg_20d", "feature_close_avg_60d",
        "feature_close_ewa_20d", "feature_close_ewa_60d",
        "feature_market_cap_avg_20d", "feature_market_cap_avg_60d",
        "feature_market_cap_ewa_20d", "feature_market_cap_ewa_60d",
        "feature_momentum_20d", "feature_momentum_60d",
        "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
        "feature_sharpe_ratio_20d", "feature_sharpe_ratio_60d",
        "feature_volatility_20d", "feature_volatility_60d",
        "feature_volume_avg_20d", "feature_volume_avg_60d",
        "feature_volume_ewa_20d", "feature_volume_ewa_60d",
    ]

    def test_regex_drops_exactly_the_eight_momentum_proxies(self):
        import re
        from src.data_mani.build_momentum import OLD_MOMENTUM_PROXIES_RX
        dropped = [c for c in self.BASE_22 if re.match(OLD_MOMENTUM_PROXIES_RX, c)]
        expected = {
            "feature_momentum_20d", "feature_momentum_60d",
            "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
            "feature_sharpe_ratio_20d", "feature_sharpe_ratio_60d",
            "feature_bollinger_20d", "feature_bollinger_60d",
        }
        self.assertEqual(set(dropped), expected)
        self.assertEqual(len(dropped), 8)
        # volatility is risk, NOT a momentum proxy -- it must survive the swap
        self.assertNotIn("feature_volatility_20d", dropped)


if __name__ == "__main__":
    unittest.main()
