"""
Test crypto/v2.0-vol-causal's construction: per-coin series keys, the identity gate, coverage masks,
and that a row never depends on anything after its own date.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.data_mani import build_vol_causal as B

DATA = "data/crypto/v2.0-vol-causal/train.parquet"


def toy(n_coins=30, n_days=220, seed=0):
    """A panel whose shipped vol features are the true cross-sectional vol ranks of a spot store."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    sig = np.exp(rng.normal(-3.5, 0.6, n_coins))
    spot = []
    for i in range(n_coins):
        r = rng.normal(0, sig[i], n_days)
        c = 100 * np.exp(np.cumsum(r))
        spot.append(pd.DataFrame({"symbol": "C%02d" % i, "date": days, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c}))
    spot = pd.concat(spot, ignore_index=True)
    panel = spot[spot["date"].dt.dayofweek < 5][["symbol", "date"]].copy()
    panel["ucid"] = panel["symbol"].str[1:]
    ranks = pd.Series(sig, index=["C%02d" % i for i in range(n_coins)]).rank(pct=True)
    for f in B.GATE.values():
        panel[f] = panel["symbol"].map(ranks).round(2)
    return panel.reset_index(drop=True), spot


class TestGate(unittest.TestCase):

    def run_gate(self, panel, spot):
        from src.data_mani.momentum_features import attach_signals_asof
        from src.data_mani.volatility_features import compute_volatility_signals
        raw = compute_volatility_signals(spot, {k: B.VOL_SPECS[k] for k in B.GATE})
        att = attach_signals_asof(panel[["date", "symbol"]], raw, list(B.GATE), lag_days=1, max_staleness_days=4)
        return B.identity_gate(panel, panel["symbol"], att)[0]

    def test_own_coin_passes_and_swapped_coin_fails(self):
        panel, spot = toy()
        lo, hi = panel.groupby("symbol")[B.GATE["feature_vol_sqret_20d"]].first().agg(["idxmin", "idxmax"])
        # give the calmest coin the wildest coin's prices
        swapped = spot[spot["symbol"] != lo].copy()
        swapped = pd.concat([swapped, spot[spot["symbol"] == hi].assign(symbol=lo)])
        stat = pd.Series(self.run_gate(panel, swapped), index=panel.index)
        own = stat[(panel["symbol"] != lo) & stat.notna()]
        bad = stat[(panel["symbol"] == lo) & stat.notna()]
        self.assertLess(own.median(), B.GATE_THRESHOLD)
        self.assertTrue((bad > B.GATE_THRESHOLD).all())

    def test_gate_needs_min_rows(self):
        panel, spot = toy()
        stat = pd.Series(self.run_gate(panel, spot), index=panel.index)
        first = panel.assign(stat=stat).sort_values("date").groupby("symbol").head(B.GATE_MIN - 1)
        self.assertTrue(first["stat"].isna().all())


class TestKeys(unittest.TestCase):

    def test_archive_coin_gets_its_own_series_key(self):
        p = pd.DataFrame({"symbol": ["ONE", "ONE", "BTC"], "ucid": ["3945", "2324", "1"], "date": pd.Timestamp("2024-01-02")})
        self.assertEqual(B.series_key(p).tolist(), ["ONE@3945", "ONE", "BTC"])

    def test_arms_partition_the_block(self):
        self.assertEqual(sorted(c for a in B.ARMS.values() for c in a), sorted(B.COLS))


@unittest.skipUnless(os.path.exists(DATA), "dataset not built")
class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(DATA).reset_index()

    def test_masks_match_the_nan_pattern(self):
        for a, cols in list(B.ARMS.items()) + list(B.TSR_ARMS.items()):
            np.testing.assert_array_equal(self.df["cov_" + a].to_numpy(), self.df[cols].notna().all(axis=1).to_numpy())

    def test_values_on_numerai_scale(self):
        v = self.df[B.COLS + B.TSR_COLS].stack().unique()
        self.assertTrue(set(np.round(v, 2)) <= {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_a_cell_never_passes_without_a_gate_statistic(self):
        ok = self.df["cov_all"] == 1
        self.assertTrue((self.df.loc[ok, "vol_gate_stat"] <= B.GATE_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()


class TestOwnHistory(unittest.TestCase):

    def test_rank_by_hand(self):
        days = pd.date_range("2024-01-01", periods=10, freq="D")
        raw = pd.DataFrame({"symbol": "A", "date": days})
        for c in B.COLS:
            raw[c] = [1, 2, 3, 4, 5, 6, 7, 3, 3, 9.0]
        out = B.own_history_ranks(raw).set_index("date")
        c = B.TSR[B.COLS[0]][0]                         # 7-day window, needs ceil(0.6*7) = 5 values
        self.assertTrue(out[c].iloc[:4].isna().all())    # fewer than 5 days of history
        # day 9 (value 3): window days 3..9 = 3,4,5,6,7,3,3 -> below 0, equal 3 -> (0 + 4/2) / 7
        self.assertAlmostEqual(out[c].iloc[8], 2 / 7)
        # day 10 (value 9): window 4..10 = 4,5,6,7,3,3,9 -> below 6, equal 1 -> (6 + 1) / 7
        self.assertAlmostEqual(out[c].iloc[9], 1.0)

    def test_a_later_value_never_changes_an_earlier_rank(self):
        days = pd.date_range("2024-01-01", periods=400, freq="D")
        v = np.random.default_rng(1).lognormal(size=400)
        full = pd.DataFrame({"symbol": "A", "date": days, **{c: v for c in B.COLS}})
        cut = full.iloc[:300]
        a = B.own_history_ranks(full).iloc[:300].reset_index(drop=True)
        b = B.own_history_ranks(cut).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_tsr_arms_partition_the_ranks(self):
        self.assertEqual(sorted(c for a in B.TSR_ARMS.values() for c in a), sorted(B.TSR_COLS))
        self.assertEqual(len(B.TSR_COLS), 132)
