"""
Test the momentum-gated mask: non-momentum features are neutralised exactly on the
rows where the momentum block is incomplete, momentum is left untouched, and the
'require' modes gate the right rows.
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.build_mommask import (apply_fillind, apply_fillmask,
                                          apply_mommask, is_created)


class TestCreatedPrefixes(unittest.TestCase):

    def test_engineered_blocks_are_created(self):
        for c in ("feature_mom_ret_7d", "feature_macro_gold_20d", "feature_ix_x",
                  "feature_tf_ewmac_2_8", "feature_vol_sqret_20d",
                  "feature_vol_yz_120d", "feature_vol_ewma_lam94",
                  "feature_coint_resid_z_60d", "feature_coint_adf_120d"):
            self.assertTrue(is_created(c), c)

    def test_shipped_features_are_not_created(self):
        # the shipped volatility/volume/momentum features must NOT be mistaken for
        # our engineered blocks -- feature_vol_ must not swallow feature_volatility_.
        for c in ("feature_volatility_20d", "feature_volatility_60d",
                  "feature_volume_avg_20d", "feature_volume_ewa_60d",
                  "feature_momentum_20d", "feature_market_cap_avg_20d",
                  "feature_bollinger_20d"):
            self.assertFalse(is_created(c), c)


class TestMomMask(unittest.TestCase):

    def _df(self):
        return pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "feature_mom_a": [0.25, 0.75, np.nan],   # C: absent
            "feature_mom_b": [0.5, np.nan, np.nan],  # B, C: absent
            "feature_close_x": [0.0, 1.0, 0.25],     # non-momentum
            "feature_close_y": [1.0, 0.0, 0.75],
            "target_binned_return_20": [0.5, 0.5, 0.5],
        })

    MOM = ["feature_mom_a", "feature_mom_b"]
    OTHER = ["feature_close_x", "feature_close_y"]

    def test_require_all_gates_any_missing(self):
        out, gated = apply_mommask(self._df(), self.MOM, self.OTHER, require="all", fill_value=0.5)
        self.assertEqual(gated.tolist(), [False, True, True])   # B, C incomplete
        # gated rows -> non-momentum neutralised
        self.assertTrue((out.loc[[1, 2], self.OTHER] == 0.5).all().all())
        # ungated row A untouched
        self.assertEqual(out.loc[0, "feature_close_x"], 0.0)
        self.assertEqual(out.loc[0, "feature_close_y"], 1.0)
        # momentum never touched
        pd.testing.assert_frame_equal(out[self.MOM], self._df()[self.MOM])
        # targets never touched
        self.assertTrue((out["target_binned_return_20"] == 0.5).all())

    def test_require_none_gates_only_fully_absent(self):
        out, gated = apply_mommask(self._df(), self.MOM, self.OTHER, require="none", fill_value=0.5)
        self.assertEqual(gated.tolist(), [False, False, True])  # only C is fully absent
        self.assertTrue((out.loc[2, self.OTHER] == 0.5).all())
        self.assertEqual(out.loc[1, "feature_close_x"], 1.0)    # B kept (partial momentum)

    def test_fill_value_is_configurable(self):
        out, _ = apply_mommask(self._df(), self.MOM, self.OTHER, require="all", fill_value=0.0)
        self.assertTrue((out.loc[[1, 2], self.OTHER] == 0.0).all().all())


class TestFillMask(unittest.TestCase):
    """Cell-level mask: only the missing values are replaced; present ones kept."""

    def _df(self):
        return pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "feature_vol_a": [0.25, 0.75, np.nan],   # C: missing this signal
            "feature_vol_b": [0.5, np.nan, np.nan],  # B, C: missing
            "feature_close_x": [0.0, 1.0, 0.25],     # shipped, always present
            "target_binned_return_20": [0.5, 0.5, 0.5],
        })

    COLS = ["feature_vol_a", "feature_vol_b", "feature_close_x"]

    def test_only_missing_cells_are_filled(self):
        out = apply_fillmask(self._df(), self.COLS, fill_value=0.5)
        # the two NaN cells become 0.5
        self.assertEqual(out.loc[2, "feature_vol_a"], 0.5)   # was NaN
        self.assertEqual(out.loc[1, "feature_vol_b"], 0.5)   # was NaN
        self.assertEqual(out.loc[2, "feature_vol_b"], 0.5)   # was NaN
        # every PRESENT value is kept exactly (unlike apply_mommask which would
        # have zeroed the whole incomplete row)
        self.assertEqual(out.loc[0, "feature_vol_a"], 0.25)
        self.assertEqual(out.loc[1, "feature_vol_a"], 0.75)  # present on a partial row
        self.assertEqual(out.loc[0, "feature_vol_b"], 0.5)
        self.assertEqual(out.loc[2, "feature_close_x"], 0.25)  # shipped kept on all-missing-signal row
        # no NaN left in the filled columns; target untouched
        self.assertFalse(out[self.COLS].isna().any().any())
        self.assertTrue((out["target_binned_return_20"] == 0.5).all())

    def test_partial_row_keeps_its_present_signal(self):
        # row B has feature_vol_a present (0.75) but feature_vol_b missing: fill mode
        # keeps the 0.75, gate mode would have neutralised it.
        out = apply_fillmask(self._df(), self.COLS, fill_value=0.5)
        gate, _ = apply_mommask(self._df(), ["feature_vol_a", "feature_vol_b"],
                                self.COLS, require="all", fill_value=0.5)
        self.assertEqual(out.loc[1, "feature_vol_a"], 0.75)   # fill: kept
        self.assertEqual(gate.loc[1, "feature_vol_a"], 0.5)   # gate: neutralised


class TestFillInd(unittest.TestCase):
    """Fill plus a 0/1 missingness indicator per feature that has any hole."""

    COLS = TestFillMask.COLS

    def test_fill_matches_fillmask_and_indicators_mark_holes(self):
        df = TestFillMask()._df()
        out, inds = apply_fillind(df, self.COLS, fill_value=0.5)
        # the filled values are exactly apply_fillmask's
        pd.testing.assert_frame_equal(out[self.COLS],
                                      apply_fillmask(df, self.COLS, fill_value=0.5)[self.COLS])
        # one indicator per holed feature; the always-present shipped one gets none
        self.assertEqual(inds, ["feature_isna_vol_a", "feature_isna_vol_b"])
        self.assertEqual(out["feature_isna_vol_a"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(out["feature_isna_vol_b"].tolist(), [0.0, 1.0, 1.0])
        # indicators are not swallowed by the engineered-block prefixes
        self.assertFalse(any(is_created(c) for c in inds))
        self.assertTrue((out["target_binned_return_20"] == 0.5).all())


if __name__ == "__main__":
    unittest.main()
