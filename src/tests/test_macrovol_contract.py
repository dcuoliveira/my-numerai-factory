"""Adversarial checks for release timing, definitions and observed-only fitting."""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.data_mani.fetch_alfred_macro import initial_records, vintage_bound, fetch_series
from src.data_mani.macrovol_contract import validate_store, attach_available, covered_rows
from src.data_mani.build_vol_macrovol import exposures, prior_scores


class TestAlfred(unittest.TestCase):
    def test_vintage_bound_handles_dst(self):
        x = vintage_bound(pd.Series(["2024-03-09", "2024-03-10"]))
        self.assertEqual(x.iloc[0], pd.Timestamp("2024-03-10T06:00:00Z"))
        self.assertEqual(x.iloc[1], pd.Timestamp("2024-03-11T05:00:00Z"))

    def test_initial_release_keeps_vintage_not_observation_as_availability(self):
        rows = [dict(date="2024-01-05", value="4.05", realtime_start="2024-01-08", realtime_end="2024-01-10")]
        x = initial_records(rows, "y10", "2024-02-01T00:00:00Z", "a" * 64)
        self.assertEqual(x.available_at.iloc[0], pd.Timestamp("2024-01-09T06:00:00Z"))
        self.assertEqual(x.series_version.iloc[0], "initial-release:2024-01-08")
        self.assertTrue(pd.isna(x.source_timestamp.iloc[0]))

    def test_http_error_does_not_expose_api_key(self):
        session = Mock()
        session.get.return_value.status_code = 400
        with self.assertRaisesRegex(RuntimeError, "HTTP 400") as error:
            fetch_series(session, "SECRET", "vix", "2020-01-01", "2024-01-01", None)
        self.assertNotIn("SECRET", str(error.exception))


class TestContract(unittest.TestCase):
    def store(self):
        definitions = dict(gold="gold_spot", dxy="ice_dxy", sp500="sp500_price_index",
                           vix="vix_close", y10="10y_constant_maturity_percent")
        return pd.DataFrame([dict(date=pd.Timestamp("2024-01-05"), series=k, value=4.0,
                                 source="archive", source_id=k, available_at="2024-01-05T22:00:00Z",
                                 retrieved_at="2024-01-06T00:00:00Z", series_version="v1",
                                 availability_basis="archived_market_close", raw_sha256="a" * 64,
                                 series_definition=v, source_timestamp="2024-01-05T21:00:00Z",
                                 source_timestamp_precision="timestamp") for k, v in definitions.items()])

    def test_accepts_complete_evidence(self):
        self.assertEqual(len(validate_store(self.store())), 5)

    def test_rejects_latest_history_and_wrong_substitutes(self):
        with self.assertRaisesRegex(ValueError, "historical availability"):
            validate_store(self.store()[["series", "date", "value"]])
        for source, substitute in (("gold", "gold_etf"), ("dxy", "broad_trade_weighted_dollar")):
            f = self.store()
            f.loc[f.series == source, "series_definition"] = substitute
            with self.assertRaisesRegex(ValueError, "Wrong series definition"):
                validate_store(f)

    def test_rejects_unselected_revisions(self):
        f = self.store()
        with self.assertRaisesRegex(ValueError, "duplicate revisions"):
            validate_store(pd.concat([f, f.iloc[:1]]))

    def test_delayed_release_weekends_four_day_staleness_and_no_backfill(self):
        panel = pd.DataFrame(dict(symbol="A", date=pd.date_range("2024-01-05", "2024-01-11")))
        signals = pd.DataFrame(dict(symbol="A", date=pd.to_datetime(["2024-01-05"]), feature_x=[0.75]))
        obs = pd.DataFrame(dict(date=pd.to_datetime(["2024-01-05"]), available_at=["2024-01-07T12:00:00Z"]))
        x = attach_available(panel, signals, ["feature_x"], obs).feature_x
        self.assertTrue(x.iloc[:3].isna().all())  # not known by Jan 7 cutoff
        self.assertEqual(x.iloc[3], 0.75)       # Jan 8 uses released Friday value
        self.assertEqual(x.iloc[4], 0.75)       # age exactly four calendar days
        self.assertTrue(x.iloc[5:].isna().all())

    def test_late_history_cannot_leak_through_a_newer_signal(self):
        panel = pd.DataFrame(dict(symbol="A", date=pd.to_datetime(["2024-01-08"])))
        signals = pd.DataFrame(dict(symbol="A", date=pd.to_datetime(["2024-01-04", "2024-01-05"]), feature_x=[1., 2.]))
        obs = pd.DataFrame(dict(date=signals.date, available_at=["2024-01-09T00:00:00Z", "2024-01-05T21:00:00Z"]))
        self.assertTrue(attach_available(panel, signals, ["feature_x"], obs).feature_x.isna().all())

    def test_gates_have_observed_only_training_masks(self):
        f = pd.DataFrame(dict(feature_hi=[0.25, np.nan], feature_lo=[np.nan, 0.75]))
        self.assertEqual(covered_rows(f, ["feature_hi"]).tolist(), [True, False])
        self.assertFalse(covered_rows(f, ["feature_hi", "feature_lo"]).any())

    def test_zero_and_negative_volatility_are_missing(self):
        f = pd.DataFrame(dict(k="A", date=pd.date_range("2024-01-01", periods=8), v=[1., 2., 3., 4., 0., -1., np.inf, np.nan]))
        _, p, z = prior_scores(f, ["v"], "k", 5, 2, calendar=False)
        self.assertTrue(p.v.iloc[4:].isna().all())
        self.assertTrue(z.v.iloc[4:].isna().all())

    def test_exposures_do_not_accumulate_weekend_coin_returns(self):
        days = pd.date_range("2023-01-01", periods=400)
        rng = np.random.default_rng(4)
        daily = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        business = days[days.dayofweek < 5]
        macro = daily.reindex(business)
        # Giant weekend moves should not alter the daily aligned beta of 2.
        coin = 2 * daily
        coin.loc[days.dayofweek >= 5] = 0.5
        close = pd.DataFrame(dict(symbol="A", date=days, close=100 * np.exp(coin.cumsum().to_numpy())))
        ex = exposures(close, {"sp500": macro})
        self.assertAlmostEqual(ex.feature_mvx_sp500_beta_120.dropna().iloc[-1], 2.0, places=8)


if __name__ == "__main__":
    unittest.main()
