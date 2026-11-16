"""Scoring acceleration must retain the frozen study's numerical definitions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import e150_finish_scoring as F


def test_matrix_scores_match_reference_with_ties_missing_meta_and_float32_targets():
    rng = np.random.default_rng(150)
    n = 300
    predictions = rng.integers(0, 40, size=(n, 7)).astype(float)
    predictions[:, -1] = 1
    y = rng.choice([0, .25, .5, .75, 1], size=n).astype("float32")
    meta = rng.normal(size=n)
    meta[::7] = np.nan
    corr, mmc = F.matrix_scores(predictions, y, meta)
    for col in range(predictions.shape[1]):
        frame = pd.DataFrame({"prediction": predictions[:, col], F.M.H.T20: y, "meta_model": meta},
                             index=["S%03d" % i for i in range(n)])
        expected = [F.M.E.corr_one(frame), F.M.E.contrib_one(frame, "meta_model")]
        np.testing.assert_allclose([corr[col], mmc[col]], expected, atol=1e-12, rtol=1e-11, equal_nan=True)


def test_block_sum_bootstrap_uses_identical_draws_and_truncation():
    rng = np.random.default_rng(2)
    for n in (19, 40, 41, 80, 143, 1300):
        x = rng.normal(0, .01, size=n)
        np.testing.assert_allclose(F.fast_bootstrap(x), F.M.E.block_bootstrap_ci(x), atol=1e-14,
                                   rtol=1e-11, equal_nan=True)
