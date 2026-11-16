"""
Test the load-bearing property of the macro x crypto interaction block: forming
the product on the RAW signals and binning AFTER turns a per-date-constant macro
momentum into a SIGN switch on the crypto momentum ranking (magnitude washes out,
sign survives). If this ever stops holding, the interactions are not what the
dataset claims.

Offline, sub-second.

Run with:  python -m unittest discover -s src/tests
"""

import unittest

import numpy as np
import pandas as pd

from src.data_mani.binning import bin_cross_sectional


class TestInteractionSignFlip(unittest.TestCase):

    def _frame(self, macro_value, date):
        # 300 symbols with a distinct crypto raw signal; one macro scalar for the
        # whole date. product = crypto_raw * macro_value.
        n = 300
        crypto = np.arange(n, dtype="float64")
        return pd.DataFrame({
            "symbol": ["s%03d" % i for i in range(n)],
            "date": date,
            "prod": crypto * macro_value,
        })

    def test_positive_macro_keeps_ranking_negative_reverses(self):
        pos = self._frame(+2.0, "2020-01-01")   # macro up
        neg = self._frame(-3.0, "2020-01-02")   # macro down
        out = bin_cross_sectional(pd.concat([pos, neg]), ["prod"], "date")
        top = "s299"                              # highest crypto raw signal
        v_pos = out[(out.date == "2020-01-01") & (out.symbol == top)]["prod"].iloc[0]
        v_neg = out[(out.date == "2020-01-02") & (out.symbol == top)]["prod"].iloc[0]
        self.assertEqual(v_pos, 1.0)             # macro>0: top stays top
        self.assertEqual(v_neg, 0.0)             # macro<0: top becomes bottom
        # the two dates are exact reverses of each other, bin for bin
        a = out[out.date == "2020-01-01"].set_index("symbol")["prod"]
        b = out[out.date == "2020-01-02"].set_index("symbol")["prod"]
        np.testing.assert_allclose(a.to_numpy(), (1.0 - b).to_numpy())

    def test_percentile_style_positive_macro_would_be_degenerate(self):
        # a macro input that is ALWAYS positive (like the [0,1] percentile) leaves
        # the crypto ranking unchanged on every date -- i.e. no macro info. This is
        # why the builder uses the SIGNED raw return, not the percentile.
        d1 = self._frame(0.2, "2020-01-01")      # both positive, different sizes
        d2 = self._frame(0.9, "2020-01-02")
        out = bin_cross_sectional(pd.concat([d1, d2]), ["prod"], "date")
        a = out[out.date == "2020-01-01"].set_index("symbol")["prod"]
        b = out[out.date == "2020-01-02"].set_index("symbol")["prod"]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())   # identical -> degenerate


if __name__ == "__main__":
    unittest.main()
