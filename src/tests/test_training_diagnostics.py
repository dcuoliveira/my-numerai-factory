"""
Tests for the training-diagnostics panel (src/prediction/training_diagnostics.py).

The load-bearing test is the first: taking a snapshot every epoch must leave
training BIT-IDENTICAL to a run that takes none. A panel that nudges the run it
describes is a panel of some other run. The rest pin the numbers the panel's
reading rests on -- the gradient's data/penalty split, the net-map norm that shows
a linear stack decaying into its regulariser, and a constant predictor being
charged a CORR of 0 rather than dropped.
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


def _part(seed, n_days, n_sym=30, p=6, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    n = n_days * n_sym
    X = rng.normal(size=(n, p))
    y = np.clip(np.round((0.5 + 0.08 * X[:, 0] + rng.normal(scale=0.2, size=n)) * 4) / 4, 0, 1)
    dates = np.repeat(pd.date_range(start, periods=n_days).to_numpy(), n_sym)
    return {"X": X, "y": y.astype("float64"), "y_score": y.astype("float32"), "date": dates,
            "meta": rng.random(n)}


def _net(activation=None, seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(6, 8)]
    if activation:
        layers.append(activation())
    layers += [nn.Linear(8, 1)]
    return nn.Sequential(*layers).to(dtype=torch.float64)


def _train(net, X, y, epochs, alpha, on_epoch=None):
    """A minibatch Adam loop of the kind the repo's MLPs use."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    weights = [m.weight for m in net if isinstance(m, nn.Linear)]
    Xt, yt = torch.tensor(X), torch.tensor(y).reshape(-1, 1)
    rng = np.random.RandomState(3)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X))
        for s in range(0, len(X), 32):
            idx = torch.as_tensor(order[s:s + 32], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = 0.5 * ((yt[idx] - net(Xt[idx])) ** 2).mean() \
                + (0.5 * alpha) * sum((w * w).sum() for w in weights) / 32
            loss.backward(); opt.step()
        if on_epoch:
            on_epoch(epoch)
    return net


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestSnapshotsDoNotMoveTraining(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_bit_identical_with_and_without_snapshots(self):
        from src.prediction.training_diagnostics import Objective, Snapshotter
        train, held = _part(0, 8), _part(1, 4, start="2024-03-01")
        plain = _train(_net(nn.ReLU), train["X"], train["y"], 6, alpha=0.5)

        watched = _net(nn.ReLU)
        snap = Snapshotter(self.dir, Objective(0.5, 32), train, held,
                           probe={"rows": np.arange(32)}, target_col="t")
        snap.snapshot(0, watched)
        _train(watched, train["X"], train["y"], 6, alpha=0.5,
               on_epoch=lambda e: snap.snapshot(e, watched))
        for a, b in zip(plain.parameters(), watched.parameters()):
            self.assertTrue(torch.equal(a, b), "a snapshot moved the training")
        self.assertTrue(watched.training, "the network's mode must be restored")
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in watched.parameters()))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestWhatASnapshotMeasures(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        from src.prediction.training_diagnostics import Objective, Snapshotter
        self.train, self.held = _part(0, 8), _part(1, 4, start="2024-03-01")
        self.objective = Objective(alpha=2.0, denominator=32)
        self.snap = Snapshotter(self.dir, self.objective, self.train, self.held,
                                probe={"rows": np.arange(40)}, target_col="t",
                                meta={"activation": "identity"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_penalty_gradient_is_alpha_w_over_denominator(self):
        net = _net()
        record = self.snap.snapshot(0, net)
        expected = sum(float(((2.0 * m.weight.detach() / 32) ** 2).sum())
                       for m in net if isinstance(m, nn.Linear)) ** 0.5
        self.assertAlmostEqual(record["gradients"]["penalty"], expected, places=12)

    def test_total_gradient_is_the_sum_of_its_parts(self):
        net = _net()
        record = self.snap.snapshot(0, net)
        X = torch.tensor(self.train["X"][:40]); y = torch.tensor(self.train["y"][:40]).reshape(-1, 1)
        weights = [m.weight for m in net if isinstance(m, nn.Linear)]
        loss = self.objective.data(net(X), y) + self.objective.penalty(weights)
        grads = torch.autograd.grad(loss, list(net.parameters()))
        self.assertAlmostEqual(record["gradients"]["total"],
                               sum(float((g * g).sum()) for g in grads) ** 0.5, places=12)
        self.assertEqual(len(record["gradients"]["groups"]), 2)

    def test_net_map_norm_only_for_a_pure_linear_stack(self):
        from src.prediction.training_diagnostics import net_map_norm
        linear = _net()
        product = linear[1].weight.detach() @ linear[0].weight.detach()
        self.assertAlmostEqual(net_map_norm(linear), float(product.norm()), places=12)
        self.assertIsNone(net_map_norm(_net(nn.ReLU)))

    def test_constant_predictor_is_charged_zero_not_dropped(self):
        net = _net()
        with torch.no_grad():
            for m in net:
                m.weight.zero_()
        record = self.snap.snapshot(0, net)
        self.assertEqual(record["metric"]["held_corr"], 0.0)
        self.assertEqual(record["predictions"]["distinct"], 1.0)
        self.assertEqual(record["predictions"]["spread"], 0.0)

    def test_log_is_written_and_the_panel_draws(self):
        from src.analysis.plot_training_diagnostics import plot_run
        net = _net()
        for epoch in range(3):
            self.snap.snapshot(epoch, net, lr=1e-3, running_loss=0.1)
        self.snap.finish(net, extra={"stop_rule_epoch": None})
        log = json.load(open(os.path.join(self.dir, "log.json")))
        self.assertEqual([s["epoch"] for s in log["snapshots"]], [0, 1, 2])
        for key in ("train", "held", "gradients", "weights", "predictions", "metric"):
            self.assertIn(key, log["snapshots"][-1])
        self.assertIsNotNone(log["snapshots"][1]["predictions"]["movement"])
        root, tag = os.path.split(self.dir)
        self.assertTrue(os.path.exists(plot_run(tag, root=root)))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestCorrObjective(unittest.TestCase):
    """The -corr loss: per-day Pearson, each day once, finite on a constant day."""

    def setUp(self):
        from src.prediction.training_diagnostics import Objective, day_codes
        rng = np.random.default_rng(3)
        self.dates = np.repeat(pd.date_range("2024-01-01", periods=4).to_numpy(), 25)
        self.days = torch.as_tensor(day_codes(self.dates))
        self.y = rng.random(100)
        self.p = rng.normal(size=100) + 2 * self.y
        self.objective = Objective(0.0, 1024, kind="corr")

    def test_is_minus_the_mean_per_day_pearson(self):
        expected = -np.mean([np.corrcoef(self.p[d == self.dates], self.y[d == self.dates])[0, 1]
                             for d in np.unique(self.dates)])
        got = float(self.objective.data(torch.tensor(self.p), torch.tensor(self.y), days=self.days))
        self.assertAlmostEqual(got, expected, places=12)

    def test_ignores_level_and_scale(self):
        a = self.objective.data(torch.tensor(self.p), torch.tensor(self.y), days=self.days)
        b = self.objective.data(torch.tensor(3 * self.p + 7), torch.tensor(self.y), days=self.days)
        self.assertAlmostEqual(float(a), float(b), places=12)

    def test_constant_day_scores_zero_with_a_finite_gradient(self):
        p = torch.tensor(self.p, requires_grad=True)
        q = p.clone()
        q = torch.where(self.days == 1, torch.zeros_like(q), q)
        loss = self.objective.data(q, torch.tensor(self.y), days=self.days)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_needs_the_days(self):
        with self.assertRaises(ValueError):
            self.objective.data(torch.tensor(self.p), torch.tensor(self.y))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestMMCObjective(unittest.TestCase):
    """
    The -MMC loss is Numerai's correlation_contribution with one change: the
    prediction is z-scored within the day instead of rank-gaussianised (which has no
    gradient). The meta model and the target are treated exactly as the scorer does.
    """

    def setUp(self):
        from src.prediction.training_diagnostics import Objective, day_codes, meta_gaussian
        rng = np.random.default_rng(7)
        n_days, n_sym = 40, 50
        self.dates = np.repeat(pd.date_range("2024-05-07", periods=n_days).to_numpy(), n_sym)
        self.days = torch.as_tensor(day_codes(self.dates))
        self.y = np.round(rng.random(n_days * n_sym) * 4) / 4
        self.meta = self.y + rng.normal(scale=0.5, size=len(self.y))
        self.p = 0.5 * self.meta + self.y + rng.normal(scale=0.7, size=len(self.y))
        self.mg = meta_gaussian(self.meta, self.dates)
        self.objective = Objective(0.0, 1024, kind="mmc")

    def _per_day(self):
        from src.prediction.training_diagnostics import day_contribution
        return day_contribution(torch.tensor(self.p), torch.tensor(self.y), self.days,
                                torch.tensor(self.mg)).numpy()

    def test_meta_is_gaussianised_as_the_scorer_does(self):
        from numerai_tools.scoring import gaussian, tie_kept_rank
        day = self.dates == self.dates[0]
        expected = gaussian(tie_kept_rank(pd.DataFrame({"m": self.meta[day]})))["m"].to_numpy()
        np.testing.assert_allclose(self.mg[day], expected, rtol=0, atol=1e-15)

    def test_is_the_contribution_formula_with_a_z_scored_prediction(self):
        got = self._per_day()
        for i, d in enumerate(pd.unique(self.dates)):
            day = self.dates == d
            p = self.p[day]; z = (p - p.mean()) / p.std()
            m = self.mg[day]
            neutral = z - (z @ m) / (m @ m) * m
            t = 4 * self.y[day]; t = t - t.mean()
            self.assertAlmostEqual(got[i], t @ neutral / day.sum(), places=12)

    def test_tracks_the_scorer_day_by_day(self):
        from src.prediction.metrics import per_period_bmc
        frame = pd.DataFrame({"date": self.dates, "prediction": self.p, "meta": self.meta,
                              "target": self.y})
        scorer = per_period_bmc(frame, "prediction", "meta", "target", "date").to_numpy()
        self.assertGreater(np.corrcoef(self._per_day(), scorer)[0, 1], 0.95)

    def test_loss_is_minus_the_mean_and_needs_the_meta(self):
        loss = self.objective.data(torch.tensor(self.p), torch.tensor(self.y), days=self.days,
                                   meta=torch.tensor(self.mg))
        self.assertAlmostEqual(float(loss), -self._per_day().mean(), places=12)
        with self.assertRaises(ValueError):
            self.objective.data(torch.tensor(self.p), torch.tensor(self.y), days=self.days)

    def test_missing_meta_refuses(self):
        from src.prediction.training_diagnostics import meta_gaussian
        meta = self.meta.copy(); meta[3] = np.nan
        with self.assertRaises(ValueError):
            meta_gaussian(meta, self.dates)


if __name__ == "__main__":
    unittest.main()
