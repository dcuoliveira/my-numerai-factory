"""Grid invariants for tabnet_ktz (E153). Skips when torch / pytorch-tabnet are absent."""
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    import pytorch_tabnet  # noqa: F401
    HAVE_TABNET = True
except ImportError:
    HAVE_TABNET = False


@unittest.skipUnless(HAVE_TABNET, "torch / pytorch-tabnet not installed")
class TestTabNetKtzGrid(unittest.TestCase):
    def setUp(self):
        from src.prediction.models import build_wrapper
        self.w = build_wrapper("tabnet_ktz", "regression", seed=0)

    def test_registered_regression_only(self):
        from src.prediction.models import supported_tasks
        self.assertEqual(supported_tasks("tabnet_ktz"), ("regression",))
        self.assertEqual(self.w.model_name, "tabnet_ktz")

    def test_n_a_tied_to_n_d(self):
        for sub in self.w.param_grid:
            self.assertEqual(list(sub["net__n_d"]), list(sub["net__n_a"]))
            self.assertEqual(len(sub["net__n_d"]), 1)

    def test_ghost_batch_never_exceeds_batch(self):
        for sub in self.w.param_grid:
            self.assertLessEqual(max(sub["net__virtual_batch_size"]),
                                 min(sub["net__batch_size"]))

    def test_e18_defaults_reachable(self):
        sub = next(s for s in self.w.param_grid if s["net__n_d"] == [8])
        self.assertIn(3, sub["net__n_steps"])
        self.assertIn(1.3, sub["net__gamma"])
        self.assertIn(0.02, sub["net__momentum"])
        self.assertIn(100, sub["net__max_epochs"])
        lr = np.asarray(sub["net__learning_rate"])
        self.assertTrue(lr.min() <= 0.02 <= lr.max())
        lam = np.asarray(sub["net__lambda_sparse"])
        self.assertTrue(np.isclose(lam, 1e-3).any())

    def test_every_key_is_an_estimator_param(self):
        params = set(self.w.ModelClass.get_params())
        for sub in self.w.param_grid:
            self.assertLessEqual(set(sub), params)


if __name__ == "__main__":
    unittest.main()
