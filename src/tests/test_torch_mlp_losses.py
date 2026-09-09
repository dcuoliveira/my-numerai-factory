"""
Tests for the torch_mlp training-loss study (E42 / E43).

The load-bearing test is the MSE regression guard: adding a `loss=` knob must
leave the default (mse) estimator BYTE-IDENTICAL, or the control arm moves and
every loss-vs-control delta is contaminated.
"""
import unittest
import warnings

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False

from src.prediction.models import build_wrapper


def _panel(seed=0, n_period=25, n_sym=40, p=22):
    rng = np.random.default_rng(seed)
    n = n_period * n_sym
    X = rng.random((n, p))
    y = X @ rng.normal(size=p) + rng.normal(scale=2.0, size=n)
    period = np.repeat(np.arange(n_period), n_sym).astype("float64")
    reference = rng.random(n)
    return X, y, period, reference


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestMseControlUnchanged(unittest.TestCase):
    """The mse arm must not move -- it is the baseline every loss is measured against."""

    def test_default_is_mse_and_byte_identical(self):
        from src.prediction.torch_mlp import TorchMLPRegressor
        X, y, _, _ = _panel()
        common = dict(hidden_layer_sizes=(16,), max_iter=30, random_state=42,
                      batch_size=256)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default = TorchMLPRegressor(**common).fit(X, y).predict(X)
            explicit = TorchMLPRegressor(loss="mse", **common).fit(X, y).predict(X)
        # default loss IS mse, and naming it explicitly changes nothing
        self.assertTrue(np.array_equal(default, explicit))

    def test_mse_needs_no_passthrough_columns(self):
        w = build_wrapper("torch_mlp_loss", task="regression", loss="mse")
        self.assertEqual(w.FIT_APPEND, ())
        self.assertEqual(w.ModelClass.named_steps["mlp"].n_period_columns, 0)
        self.assertEqual(w.ModelClass.named_steps["mlp"].n_reference_columns, 0)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestLossesBuildAndDifferentiate(unittest.TestCase):

    def test_every_loss_forwards_to_a_finite_scalar_with_grad(self):
        import torch
        from src.prediction.torch_mlp_losses import build_torch_mlp_loss, LOSSES
        _, y, period, reference = _panel()
        yt = torch.as_tensor(y, dtype=torch.float64)
        per = torch.as_tensor(period, dtype=torch.float64)
        ref = torch.as_tensor(reference, dtype=torch.float64)
        for name in LOSSES:
            if name == "mse":
                self.assertIsNone(build_torch_mlp_loss(name)[0])
                continue
            fn, meta = build_torch_mlp_loss(name)
            sig = torch.randn(len(y), 1, dtype=torch.float64, requires_grad=True)
            loss = fn(sig, yt, reference=ref if meta["needs_reference"] else None,
                      period=per if meta["needs_period"] else None)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(torch.isfinite(sig.grad).all(), name)
            self.assertTrue(bool((sig.grad != 0).any()), name)

    def test_degenerate_batch_returns_grad_carrying_zero(self):
        import torch
        from src.prediction.torch_mlp_losses import build_torch_mlp_loss
        _, y, period, _ = _panel()
        fn, _ = build_torch_mlp_loss("corr")
        sig = torch.ones(len(y), 1, dtype=torch.float64, requires_grad=True)  # constant
        loss = fn(sig, torch.as_tensor(y), period=torch.as_tensor(period))
        loss.backward()                                    # must not raise
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(sig.grad).all())

    def test_per_period_equals_pooled_when_one_period(self):
        import torch
        from src.prediction.torch_mlp_losses import PerPeriodLoss
        from src.prediction.nn.losses import CorrLoss
        rng = np.random.default_rng(1)
        sig = torch.as_tensor(rng.random((200, 1)))
        tgt = torch.as_tensor(rng.random(200))
        one = torch.zeros(200, dtype=torch.float64)        # all one period
        pp = PerPeriodLoss(CorrLoss())(sig, tgt, period=one)
        pooled = CorrLoss()(sig, tgt)
        self.assertAlmostEqual(float(pp), float(pooled), places=10)

    def test_soft_drawdown_approaches_true_max_drawdown(self):
        import torch
        from src.prediction.torch_mlp_losses import _soft_max_drawdown
        from src.prediction.metrics import summarize
        # a series that rises then falls -> a real, known drawdown
        series = np.array([0.02, 0.03, -0.05, -0.04, 0.01, -0.02])
        cum = torch.cumsum(torch.as_tensor(series), dim=0)
        soft = float(_soft_max_drawdown(cum, beta_max=400.0, beta_dd=400.0))
        true_dd = summarize(series)["max_drawdown"]
        self.assertLess(abs(soft - true_dd), 5e-3)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestPassthroughAndWrapper(unittest.TestCase):

    def test_wrapper_declares_the_right_passthrough(self):
        for loss, appends, n_ref in [("corr", ("period",), 0),
                                     ("bmc", ("period", "reference"), 1),
                                     ("sharpe", ("period",), 0),
                                     ("calmar", ("period",), 0)]:
            w = build_wrapper("torch_mlp_loss", task="regression", loss=loss)
            self.assertEqual(w.FIT_APPEND, appends, loss)
            mlp = w.ModelClass.named_steps["mlp"]
            self.assertEqual(mlp.n_period_columns, 1, loss)
            self.assertEqual(mlp.n_reference_columns, n_ref, loss)

    def test_passthrough_columns_never_enter_the_network(self):
        # n_features_in_ counts the trailing columns, but the net's first layer
        # sees only the real features -- the extras are peeled by _split_ids.
        from sklearn.base import clone
        X, y, period, reference = _panel()
        w = build_wrapper("torch_mlp_loss", task="regression", loss="bmc")
        est = clone(w.ModelClass)
        est.set_params(mlp__hidden_layer_sizes=(8,), mlp__max_iter=5,
                       mlp__batch_size=256)
        Xe = np.column_stack([X, period, reference])       # [features | period | reference]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xe, y)
        mlp = est.named_steps["mlp"]
        self.assertEqual(mlp.n_features_in_, X.shape[1] + 2)   # counts the 2 extras
        first_linear = [m for m in mlp._net if hasattr(m, "in_features")][0]
        self.assertEqual(first_linear.in_features, X.shape[1])  # net saw only features

    def test_diverged_flag_and_diagnostics_populate(self):
        from sklearn.base import clone
        X, y, period, _ = _panel()
        w = build_wrapper("torch_mlp_loss", task="regression", loss="corr")
        est = clone(w.ModelClass)
        est.set_params(mlp__hidden_layer_sizes=(8,), mlp__max_iter=10,
                       mlp__batch_size=256)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(np.column_stack([X, period]), y)
        mlp = est.named_steps["mlp"]
        for attr in ("loss_curve_", "n_iter_", "best_loss_", "final_loss_",
                     "pred_std_", "diverged_"):
            self.assertTrue(hasattr(mlp, attr), attr)
        self.assertFalse(mlp.diverged_)                    # clean run does not diverge
        self.assertEqual(len(mlp.loss_curve_), mlp.n_iter_)


if __name__ == "__main__":
    unittest.main()
