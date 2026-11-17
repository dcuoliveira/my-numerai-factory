"""TV-epoch TabNet (tabnet_ktz_tv): the epoch count is tied to model size and dataset."""
import itertools
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    import pytorch_tabnet  # noqa: F401
    HAVE_TABNET = True
except ImportError:
    HAVE_TABNET = False


class TestTVSchedule(unittest.TestCase):
    def test_schedule_matches_skill_example(self):
        from src.prediction.tabnet_tv import tv_epochs
        # timmermann-mlp-compute: P = 4,033 -> 163 epochs, 10x -> 1,630.
        self.assertEqual(tv_epochs(4033), 163)
        self.assertEqual(tv_epochs(4033, mult=10), 1630)
        self.assertEqual(tv_epochs(4033, cap=100), 100)
        self.assertEqual(tv_epochs(1), 11)


@unittest.skipUnless(HAVE_TABNET, "torch / pytorch-tabnet not installed")
class TestTabNetTV(unittest.TestCase):
    def test_e18_default_counts(self):
        from src.prediction.tabnet_tv import count_tabnet_params, tv_epochs
        self.assertEqual(count_tabnet_params(22), 7092)
        self.assertEqual(tv_epochs(count_tabnet_params(22)), 250)
        self.assertEqual(tv_epochs(count_tabnet_params(44)), 289)

    def test_fit_uses_own_tv_count_and_dataset(self):
        from src.prediction.tabnet_tv import TabNetTVRegressor
        rng = np.random.default_rng(0)
        y = rng.normal(size=600)
        # epoch_cap keeps the test fast; the uncapped count is checked separately.
        est = TabNetTVRegressor(n_d=8, n_a=8, n_steps=2, batch_size=256,
                                virtual_batch_size=128, epoch_cap=2)
        est.fit(rng.normal(size=(600, 22)), y)
        self.assertEqual(est.epochs_, 2)
        self.assertTrue(est.epoch_cap_bound_)
        raw = est.planned_epochs(22)
        fenc = est.planned_epochs(44)
        self.assertGreater(fenc[1], raw[1])            # more inputs, more params
        self.assertEqual(est.n_params_, raw[1])        # matches the fitted network
        self.assertEqual(est.predict(rng.normal(size=(5, 22))).shape, (5,))

    def test_clone_does_not_carry_epochs(self):
        from sklearn.base import clone
        from src.prediction.tabnet_tv import TabNetTVRegressor
        est = TabNetTVRegressor()
        self.assertNotIn("max_epochs", est.get_params())
        est.max_epochs = 999
        self.assertFalse(hasattr(clone(est), "max_epochs"))

    def test_wrapper_grid_bounds_epochs(self):
        from src.prediction.models import build_wrapper, supported_tasks
        from src.prediction.tabnet_tv import count_tabnet_params, tv_epochs
        w = build_wrapper("tabnet_ktz_tv", "regression", seed=0)
        self.assertEqual(supported_tasks("tabnet_ktz_tv"), ("regression",))
        params = set(w.ModelClass.get_params())
        epochs = {22: [], 44: []}
        for sub in w.param_grid:
            self.assertLessEqual(set(sub), params)
            self.assertNotIn("net__max_epochs", sub)
            self.assertEqual(list(sub["net__n_d"]), list(sub["net__n_a"]))
            for nin in epochs:
                for s, ni, ns in itertools.product(sub["net__n_steps"],
                                                   sub["net__n_independent"],
                                                   sub["net__n_shared"]):
                    epochs[nin].append(tv_epochs(count_tabnet_params(
                        nin, n_d=sub["net__n_d"][0], n_a=sub["net__n_d"][0],
                        n_steps=s, n_independent=ni, n_shared=ns)))
        self.assertLessEqual(max(epochs[22]), 840)
        self.assertLessEqual(max(epochs[44]), 910)
        self.assertGreaterEqual(min(epochs[22]), 100)


if __name__ == "__main__":
    unittest.main()
