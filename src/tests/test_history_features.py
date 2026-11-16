"""
Test the own-history block: hand-checked values on a small panel, tie handling,
calendar windows, coverage, causality, and that the fast binner is the repo's rule.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.history_features import (add_history_features, bin_ties_preserved,
                                            compute_history_signals, history_columns,
                                            is_short)
from src.live.features import add_histfe

F = "feature_x_20d"
N = "feature_hfe_x_20d"


def panel(values, ucid="1", start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"symbol": "A" + ucid, "ucid": ucid, "date": dates, F: values})


class TestOwnHistory(unittest.TestCase):

    def test_rank_preserves_ties(self):
        # Mon..Fri; on Friday the 7-day window holds all five values
        out = compute_history_signals(panel([0.25, 0.5, 0.5, 0.75, 0.5]), [F])
        # one below, three equal including today: (1 + (3 + 1) / 2) / 5
        self.assertAlmostEqual(out[N + "_rank7"].iloc[4], 0.6)

    def test_rank_uses_a_calendar_window(self):
        # 2024-01-01 is a Monday. 7 days back from the next Monday reaches Tuesday,
        # so the first Monday is out: 7 calendar days hold 5 weekdays, not 7 rows.
        out = compute_history_signals(panel([1.0, 0.5, 0.5, 0.5, 0.5, 0.75]), [F])
        self.assertAlmostEqual(out[N + "_rank7"].iloc[5], 1.0)

    def test_ma_difference(self):
        v = [0.0] * 40 + [1.0] * 5
        out = compute_history_signals(panel(v), [F])
        # last row (Fri): 3 days = Wed..Fri mean 1, 7 days = Mon..Fri mean 1
        self.assertAlmostEqual(out[N + "_ma3_7"].iloc[-1], 0.0)
        self.assertAlmostEqual(out[N + "_ma7_14"].iloc[-1], 1.0 - 5 / 10)

    def test_time_since_max_takes_the_latest_maximum(self):
        out = compute_history_signals(panel([1.0, 0.5, 1.0, 0.5, 0.25]), [F])
        self.assertEqual(out[N + "_tsmax7"].iloc[4], 2)     # Wed -> Fri

    def test_time_since_max_counts_calendar_days(self):
        out = compute_history_signals(panel([0.5, 0.5, 0.5, 0.5, 1.0, 0.5]), [F])
        self.assertEqual(out[N + "_tsmax7"].iloc[5], 3)     # Fri -> Mon

    def test_time_since_change(self):
        out = compute_history_signals(panel([0.5, 0.5, 0.75, 0.75, 0.75]), [F])
        got = out[N + "_tschg"].tolist()
        self.assertTrue(np.isnan(got[0]) and np.isnan(got[1]))   # no change seen yet
        self.assertEqual(got[2:], [0, 1, 2])

    def test_coverage_keeps_nan(self):
        # two values in a 5-date window is below 60%
        out = compute_history_signals(panel([0.5, 0.5, np.nan, np.nan, np.nan]), [F])
        self.assertTrue(np.isnan(out[N + "_ma3_7"].iloc[4]))
        short = compute_history_signals(panel([0.5, 0.5]), [F])
        self.assertTrue(np.isnan(short[N + "_rank364"]).all())

    def test_tokens_never_mix_and_tickers_can_repeat(self):
        a = panel([0.0] * 10, "1")
        b = panel([1.0] * 10, "2").assign(symbol="A1")   # reused ticker
        both = pd.concat([a, b], ignore_index=True)
        out = compute_history_signals(both, [F])
        self.assertEqual(out[N + "_rank7"].iloc[9], out[N + "_rank7"].iloc[19])
        self.assertAlmostEqual(out[N + "_ma7_14"].iloc[19], 0.0)

    def test_causal_truncation(self):
        rng = np.random.default_rng(0)
        vals = rng.choice([0, .25, .5, .75, 1], size=400)
        full = panel(list(vals))
        cut = full.iloc[:300]
        a = compute_history_signals(full, [F]).iloc[:300]
        pd.testing.assert_frame_equal(a, compute_history_signals(cut, [F]))


class TestBinning(unittest.TestCase):

    def test_reproduces_numerai_counts_below_300_tokens(self):
        # shipped cross-sections: n=241 is 13/48/120/48/12, n=245 is 13/49/122/48/13
        for n, want in ((241, [13, 48, 120, 48, 12]), (245, [13, 49, 122, 48, 13]),
                        (300, [15, 60, 150, 60, 15])):
            s = pd.DataFrame({"a": np.arange(n, dtype="float64")})
            out = bin_ties_preserved(s, np.zeros(n))["a"]
            got = out.value_counts().sort_index().tolist()
            self.assertEqual(got, want)

    def test_a_lone_value_is_neutral_and_nan_stays(self):
        s = pd.DataFrame({"a": [3.0, np.nan]})
        out = bin_ties_preserved(s, np.array(["d", "d"]))["a"]
        self.assertEqual(out.iloc[0], 0.5)
        self.assertTrue(np.isnan(out.iloc[1]))

    def test_ties_share_a_bin(self):
        s = pd.DataFrame({"a": [0, 0, 0, 0, 1, 2, 3, 4, 5, 6.0] * 3})
        out = bin_ties_preserved(s, np.array(["d"] * 30))
        self.assertEqual(out.loc[s["a"] == 0, "a"].nunique(), 1)

    def test_columns_and_split(self):
        g = history_columns([F])
        self.assertEqual({k: len(v) for k, v in g.items()},
                         {"rank": 5, "ma": 5, "tsmax": 5, "tschg": 1})
        self.assertEqual(sum(is_short(c) is True for c in sum(g.values(), [])), 9)
        self.assertEqual(sum(is_short(c) is False for c in sum(g.values(), [])), 6)
        self.assertIsNone(is_short(g["tschg"][0]))

    def test_entry_point_binned(self):
        rows = pd.concat([panel(list(np.linspace(0, 1, 30)), str(i)) for i in range(40)],
                         ignore_index=True)
        out = add_history_features(rows, [F])
        self.assertTrue(set(np.unique(out.to_numpy()[~np.isnan(out.to_numpy())]))
                        <= {0.0, 0.25, 0.5, 0.75, 1.0})


class TestLiveParity(unittest.TestCase):

    def test_live_rows_match_the_offline_block(self):
        rng = np.random.default_rng(2)
        rows = pd.concat([panel(list(rng.choice([0, .25, .5, .75, 1], 300)), str(i))
                          for i in range(30)], ignore_index=True)
        offline = add_history_features(rows, [F])
        live = rows.assign(is_live=rows["date"] > rows["date"].iloc[285])
        live.loc[live["is_live"] & (live["date"] < live["date"].max()), "ucid"] = np.nan
        out, added = add_histfe(live.copy())
        last = rows["date"] == rows["date"].max()
        pd.testing.assert_frame_equal(out.loc[last, added], offline.loc[last, added])


if __name__ == "__main__":
    unittest.main()
