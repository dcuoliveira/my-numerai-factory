"""
Tests for the shared macro feature pipeline: the causal cross-time normalisation,
the raw compute (returns / yield levels / slopes), and the date-level broadcast
as-of merge.

The failures that would look like success: a percentile that peeks at a future
value (a better, unreproducible backtest), a macro column that is NOT actually the
same for every symbol on a date (then it is not a macro feature), or a same-date
attach that reads a value not known at submission.

Offline, sub-second, no network.

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.macro_features import (attach_macro_asof,
                                          causal_trailing_percentile,
                                          compute_macro_signals)


class TestCausalPercentile(unittest.TestCase):

    def _s(self, vals):
        return pd.Series(vals, index=pd.date_range("2020-01-01", periods=len(vals), freq="D"))

    def test_increasing_series_is_always_top_percentile(self):
        p = causal_trailing_percentile(self._s([1, 2, 3, 4, 5]), window=3, min_periods=1)
        np.testing.assert_allclose(p.to_numpy(), [1, 1, 1, 1, 1])

    def test_trailing_window_forgets_an_old_extreme(self):
        # a big early value must drop out of the window (unlike expanding)
        p = causal_trailing_percentile(self._s([10, 1, 2, 3]), window=2, min_periods=1)
        np.testing.assert_allclose(p.to_numpy(), [1.0, 0.5, 1.0, 1.0])

    def test_bounded_and_min_periods(self):
        rng = np.random.RandomState(0)
        p = causal_trailing_percentile(self._s(rng.randn(300)), window=126, min_periods=50)
        self.assertTrue(p.iloc[:49].isna().all())            # < min_periods -> NaN
        v = p.dropna().to_numpy()
        self.assertTrue(np.all((v > 0) & (v <= 1.0)))

    def test_is_causal_a_future_value_never_moves_a_past_percentile(self):
        base = self._s([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
        before = causal_trailing_percentile(base, window=4, min_periods=1)
        bumped = base.copy(); bumped.iloc[5] = 100.0          # change a future point
        after = causal_trailing_percentile(bumped, window=4, min_periods=1)
        pd.testing.assert_series_equal(before.iloc[:5], after.iloc[:5])


class TestComputeMacro(unittest.TestCase):

    def _macro(self, n=80):
        dates = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({
            "date": dates,
            "gold": 2.0 ** np.arange(n),                      # doubles each step
            "silver": np.linspace(10, 20, n),
            "sp500": np.linspace(100, 200, n),
            "dxy": np.linspace(90, 110, n),
            "y1": np.full(n, 1.0), "y2": np.full(n, 2.0),
            "y5": np.full(n, 3.0), "y10": np.full(n, 4.0),
            "y30": np.full(n, 5.0),
        })

    def test_return_level_and_slope_raw(self):
        specs = {
            "feature_macro_gold_ret_20d": {"kind": "ret", "series": "gold", "window": 20},
            "feature_macro_y10": {"kind": "level", "series": "y10"},
            "feature_macro_slope_2s10s": {"kind": "slope", "long": "y10", "short": "y2"},
        }
        out = compute_macro_signals(self._macro(), specs, normalize=False)
        g = out["feature_macro_gold_ret_20d"].to_numpy()
        self.assertTrue(np.isnan(g[:20]).all())
        np.testing.assert_allclose(g[20:], 2.0 ** 20 - 1.0, rtol=1e-9)   # doubling
        self.assertTrue(np.allclose(out["feature_macro_y10"], 4.0))
        self.assertTrue(np.allclose(out["feature_macro_slope_2s10s"], 4.0 - 2.0))

    def test_normalized_columns_are_bounded(self):
        out = compute_macro_signals(self._macro(200), None, normalize=True,
                                    window=60, min_periods=20)
        for c in [c for c in out.columns if c != "date"]:
            v = out[c].dropna().to_numpy()
            if v.size:
                self.assertTrue(np.all((v >= 0) & (v <= 1.0)), c)


class TestAttachBroadcast(unittest.TestCase):

    def _panel(self, dates, symbols=("A", "B", "C")):
        rows = [pd.DataFrame({"symbol": list(symbols), "date": d}) for d in dates]
        return pd.concat(rows, ignore_index=True)

    def test_every_symbol_on_a_date_gets_the_same_value(self):
        dates = pd.date_range("2020-02-01", periods=5, freq="D")
        panel = self._panel(dates)
        macro = pd.DataFrame({"date": pd.date_range("2020-01-25", periods=12, freq="D"),
                              "feature_macro_x": np.arange(12, dtype=float)})
        out = attach_macro_asof(panel, macro, ["feature_macro_x"], lag_days=1, max_staleness_days=7)
        for d in dates:
            vals = out[out.date == d]["feature_macro_x"].unique()
            self.assertEqual(len(vals), 1)                    # shared across symbols

    def test_row_D_uses_D_minus_1_not_D(self):
        d = pd.Timestamp("2020-02-10")
        panel = self._panel([d])
        # macro value on d itself must NOT be used (not final at submission)
        macro = pd.DataFrame({"date": [pd.Timestamp("2020-02-09"), d],
                              "feature_macro_x": [7.0, 99.0]})
        out = attach_macro_asof(panel, macro, ["feature_macro_x"], lag_days=1, max_staleness_days=7)
        self.assertTrue((out["feature_macro_x"] == 7.0).all())   # D-1, never 99

    def test_staleness_bridges_a_weekend(self):
        saturday = pd.Timestamp("2020-02-08")                 # Sat; macro is business-day
        panel = self._panel([saturday])
        friday = pd.Timestamp("2020-02-07")
        macro = pd.DataFrame({"date": [friday], "feature_macro_x": [3.3]})
        out = attach_macro_asof(panel, macro, ["feature_macro_x"], lag_days=1, max_staleness_days=7)
        self.assertTrue((out["feature_macro_x"] == 3.3).all())   # Friday reaches Saturday

    def test_staleness_cap_blocks_ancient_carry(self):
        # macro spans Jan..Mar so the daily grid is long; a value 40+ days stale
        # must NOT carry (ffill limit = staleness), it falls to NaN -> neutral.
        d = pd.Timestamp("2020-02-15")
        panel = self._panel([d])
        macro = pd.DataFrame({"date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")],
                              "feature_macro_x": [5.0, 6.0]})
        out = attach_macro_asof(panel, macro, ["feature_macro_x"], lag_days=1, max_staleness_days=7)
        self.assertTrue(out["feature_macro_x"].isna().all())

    def test_per_column_fill_is_independent(self):
        # column A missing on the lookup date but present the day before; column B
        # present. A must carry its prior value, not go null because of the row.
        d = pd.Timestamp("2020-02-10")
        panel = self._panel([d])
        macro = pd.DataFrame({
            "date": [pd.Timestamp("2020-02-08"), pd.Timestamp("2020-02-09")],
            "feature_macro_a": [1.0, np.nan],   # A missing on D-1 (a holiday for A)
            "feature_macro_b": [2.0, 2.5],      # B present on D-1
        })
        out = attach_macro_asof(panel, macro, ["feature_macro_a", "feature_macro_b"],
                                lag_days=1, max_staleness_days=7)
        self.assertTrue((out["feature_macro_a"] == 1.0).all())   # carried from 02-08
        self.assertTrue((out["feature_macro_b"] == 2.5).all())   # its own 02-09

    def test_rows_and_order_preserved(self):
        dates = pd.date_range("2020-02-01", periods=4, freq="D")
        panel = self._panel(dates).reset_index(drop=True)
        panel["keep"] = np.arange(len(panel))
        macro = pd.DataFrame({"date": pd.date_range("2020-01-20", periods=20, freq="D"),
                              "feature_macro_x": np.arange(20, dtype=float)})
        out = attach_macro_asof(panel, macro, ["feature_macro_x"], lag_days=1, max_staleness_days=7)
        pd.testing.assert_series_equal(out["keep"], panel["keep"])
        self.assertEqual(len(out), len(panel))


if __name__ == "__main__":
    unittest.main()
