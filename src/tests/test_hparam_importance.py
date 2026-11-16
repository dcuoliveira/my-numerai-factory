"""hparam_importance recovers a planted effect and ignores a null axis."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis import hparam_importance as H


def synthetic(path, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for refit in range(4):
        for i in range(150):
            lr = rng.choice([1e-3, 3e-3, 1e-2, 3e-2])
            width = rng.choice([8, 16, 32])
            noise_axis = rng.choice(["a", "b", "c"])
            corr = 0.05 + 0.03 * np.log10(lr / 1e-3) + rng.normal(0, 0.01) + 0.02 * refit
            mmc = 0.02 - 0.002 * (width / 8) + rng.normal(0, 0.003)
            rows.append({"refit": refit, "candidate": i, "is_best": False,
                         "param_net__learning_rate": lr, "param_net__n_d": width,
                         "param_net__n_a": width, "param_net__mask_type": noise_axis,
                         "param_net__fixed": 1,
                         "val_corr_mean": corr, "val_mmc_mean": mmc})
    pd.DataFrame(rows).to_csv(path, index=False)


class TestHparamImportance(unittest.TestCase):
    def test_planted_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x_cv.csv")
            synthetic(p, 0)
            r = H.analyse("syn", [p])
            # n_a duplicates n_d and fixed never varies: both dropped.
            self.assertEqual(sorted(H.pretty(x) for x in r["params"]),
                             ["learning_rate", "mask_type", "n_d"])
            corr = r["axes"]["corr"].set_index("axis")
            mmc = r["axes"]["mmc"].set_index("axis")
            self.assertEqual(corr.index[0], "learning_rate")
            self.assertLess(corr.loc["learning_rate", "p"], 0.01)
            self.assertGreater(corr.loc["learning_rate", "kendall_w"], 0.9)
            self.assertEqual(mmc.index[0], "n_d")
            self.assertGreater(corr.loc["mask_type", "p"], 0.01)
            grid = r["grid"].set_index("axis")
            self.assertEqual(grid.loc["learning_rate", "best (corr)"], "0.03")
            self.assertEqual(grid.loc["mask_type", "action"], "pin")
            out = H.render([r], "t", os.path.join(tmp, "rep"))
            self.assertTrue(os.path.getsize(out) > 10000)


if __name__ == "__main__":
    unittest.main()
