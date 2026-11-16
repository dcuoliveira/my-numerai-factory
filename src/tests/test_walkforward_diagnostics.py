"""
Tests for the walk-forward procedure level (src/analysis/walkforward_diagnostics.py)
and the next-block predictions it reads (src/prediction/diagnose.next_block_store).

The load-bearing test is the last: predicting the next refit's block every epoch must
leave training BIT-IDENTICAL, or the churn it measures is of some other run. The rest
pin the arithmetic the skill's reading rests on -- churn is 0 for one function and 2
for its reverse, a flat-topped fold reports its range, the gradient screen flags a fold
that leaves the envelope, and the folds to panel are chosen by the gap.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis import walkforward_diagnostics as wf

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


def _days(n_days, n_sym):
    return wf._day_rows(np.repeat(pd.date_range("2024-01-01", periods=n_days).to_numpy(), n_sym))


def _fold(refit, corr, grad=None, stop=None, finished=True):
    epochs = np.arange(len(corr))
    return {"refit": refit, "tag": "x_r%02d" % refit, "finished": finished,
            "epoch": epochs, "held_corr": np.asarray(corr, dtype="float64"),
            "grad": np.ones(len(corr)) if grad is None else np.asarray(grad, dtype="float64"),
            "summary": {"stop_rule_epoch": stop} if finished else None,
            "config": {"held_start": "2024-01-01", "held_end": "2024-06-01"}}


class TestChurn(unittest.TestCase):
    def setUp(self):
        self.days = _days(5, 40)
        self.a = np.random.default_rng(0).normal(size=200)

    def test_same_function_has_no_churn(self):
        self.assertAlmostEqual(wf.churn(self.a, self.a * 3 + 1, self.days)[0], 0.0, places=12)

    def test_reversed_ranking_churns_two(self):
        self.assertAlmostEqual(wf.churn(self.a, -self.a, self.days)[0], 2.0, places=12)

    def test_a_constant_day_is_undefined_not_zero(self):
        b = self.a.copy()
        b[self.days[2]] = 0.5
        value, undefined = wf.churn(self.a, b, self.days)
        self.assertEqual(undefined, 1)
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_row_index_mismatch_refuses(self):
        r = pd.DataFrame({"date": [1, 1], "symbol": ["a", "b"]})
        with self.assertRaises(SystemExit):
            wf._same_rows(r, r.iloc[::-1], "swapped")


class TestFoldReading(unittest.TestCase):
    def test_flat_top_reports_its_range(self):
        corr = [0.0, 0.05, 0.100, 0.098, 0.097, 0.02, 0.099]
        self.assertEqual(wf.best_epoch(_fold(0, corr), flat=0.005), (2, 2, 6))

    def test_unscored_epochs_are_skipped(self):
        corr = [0.0, np.nan, 0.08, np.nan, 0.05]
        self.assertEqual(wf.best_epoch(_fold(0, corr), flat=0.001), (2, 2, 2))

    def test_kept_epoch_rule(self):
        f = _fold(0, [0.1] * 10, stop=4)
        self.assertEqual(wf.kept_epoch(f, "last"), 9)
        self.assertEqual(wf.kept_epoch(f, "stop"), 4)
        self.assertEqual(wf.kept_epoch(_fold(0, [0.1] * 10, stop=None), "stop"), 9)

    def test_median_iqr_ignores_missing(self):
        self.assertEqual(wf.median_iqr([1.0, np.nan, 3.0, 2.0]), (2.0, 1.5, 2.5))

    def test_gradient_screen_flags_the_odd_fold(self):
        folds = [_fold(r, [0.1] * 6, grad=np.full(6, 0.1)) for r in range(4)]
        folds.append(_fold(4, [0.1] * 6, grad=np.full(6, 3.0)))
        folds.append(_fold(5, [0.1] * 6, finished=False))
        screen = wf.gradient_screen(folds)
        self.assertEqual(list(screen["flags"]), [4])
        self.assertEqual(screen["refits"], [0, 1, 2, 3, 4])

    def test_picks_by_gap(self):
        table = pd.DataFrame({"refit": [0, 1, 2, 3, 4], "gap_kept": [0.01, -0.05, 0.02, 0.0, 0.03]})
        self.assertEqual(wf.pick_folds(table, {}), {"first": 0, "worst": 1, "median": 0})
        self.assertEqual(wf.pick_folds(table, {3: {}})["flagged r03"], 3)


class TestMetaDayRescoring(unittest.TestCase):
    """meta_scores() is per_period_corr / per_period_bmc at every epoch, on the meta days only."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        n_days, n_sym, n_ep = 6, 30, 4
        dates = np.repeat(pd.date_range("2024-05-01", periods=n_days).to_numpy(), n_sym)
        meta = rng.random(n_days * n_sym)
        meta[: 2 * n_sym] = np.nan                       # the first two days have no meta model
        self.rows = pd.DataFrame({
            "date": dates, "symbol": np.tile(np.arange(n_sym).astype(str), n_days),
            "target": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], n_days * n_sym).astype("float32"),
            "reference_prediction": meta, "meta_model": meta})
        self.preds = rng.normal(size=(n_ep, n_days * n_sym))
        self.preds[3] = np.nan                           # an epoch the run never reached
        self.rows.to_parquet(self.tmp + "/val_rows.parquet", index=False)
        np.save(self.tmp + "/pred_val.npy", self.preds)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_matches_the_per_day_scorer_on_the_covered_days(self):
        from src.prediction.metrics import per_period_bmc, per_period_corr
        got = wf.meta_scores(self.tmp)
        self.assertEqual((int(got["n_days"]), int(got["n_days_block"])), (4, 6))
        covered = self.rows[self.rows["meta_model"].notna()]
        for e in range(3):
            frame = covered.assign(prediction=self.preds[e][self.rows["meta_model"].notna().to_numpy()])
            corr = per_period_corr(frame, "prediction", "target", "date").fillna(0.0).mean()
            mmc = per_period_bmc(frame, "prediction", "meta_model", "target", "date").mean()
            self.assertAlmostEqual(got["corr"][e], corr, places=12)
            self.assertAlmostEqual(got["mmc"][e], mmc, places=12)
        self.assertTrue(np.isnan(got["corr"][3]) and np.isnan(got["mmc"][3]))
        cached = wf.meta_scores(self.tmp)                 # the second call reads the cache
        np.testing.assert_array_equal(cached["mmc"], got["mmc"])

    def test_a_fold_without_meta_days_has_none(self):
        self.rows.assign(meta_model=np.nan).to_parquet(self.tmp + "/val_rows.parquet", index=False)
        os.remove(self.tmp + "/meta_scores.npz") if os.path.exists(self.tmp + "/meta_scores.npz") else None
        self.assertIsNone(wf.meta_scores(self.tmp))


class TestAgreement(unittest.TestCase):
    def test_runs_are_named_as_ranges(self):
        self.assertEqual(wf._runs([6, 0, 1, 2, 3, 5]), "r00–r03, r05–r06")
        self.assertEqual(wf._runs([7]), "r07")

    def test_sign_agreement_by_fold_and_month(self):
        days = pd.date_range("2024-06-01", periods=60)
        nn = pd.DataFrame({"date": days, "corr": 0.10, "mmc": 0.05})
        ridge = pd.DataFrame({"date": days, "corr": 0.08, "mmc": np.r_[np.full(30, 0.06), np.full(30, 0.04)]})
        table = pd.DataFrame({"refit": [8, 9], "corr_gap_meta_kept": [0.02, 0.02],
                              "gap_kept": [-0.01, 0.01], "meta_days": [30, 30]})
        g = wf.gap_agreement(nn, ridge, table)
        self.assertEqual((g["folds_same_sign"], g["n_folds"]), (1, 2))
        self.assertEqual(g["n_days"], 60)
        june, july = g["months"]
        self.assertAlmostEqual(june["corr_gap"], 0.02, places=12)
        self.assertLess(june["mmc_gap"], 0)               # June: ahead on CORR, behind on MMC


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestNextBlockDoesNotMoveTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _train(self, record):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)).double()
        rng = np.random.default_rng(1)
        X = torch.tensor(rng.normal(size=(64, 4)))
        y = torch.tensor(rng.normal(size=(64, 1)))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        if record:
            record(0, net)
        for epoch in range(1, 6):
            for s in range(0, 64, 16):
                opt.zero_grad(); loss = ((net(X[s:s + 16]) - y[s:s + 16]) ** 2).mean()
                loss.backward(); opt.step()
            if record:
                record(epoch, net)
        return [p.detach().clone() for p in net.parameters()]

    def test_bit_identical_with_and_without_next_block(self):
        from src.prediction.diagnose import next_block_store
        rng = np.random.default_rng(2)
        n = 30
        nxt = {"X": rng.normal(size=(n, 4)), "y": rng.normal(size=n), "y_score": rng.random(n),
               "reference": rng.random(n), "meta": rng.random(n), "symbol": np.arange(n).astype(str),
               "date": np.repeat(pd.date_range("2024-01-01", periods=3).to_numpy(), 10)}
        record = next_block_store(self.tmp, nxt, 5)
        with_next = self._train(record)
        record.flush()
        without = self._train(None)
        for a, b in zip(with_next, without):
            self.assertTrue(torch.equal(a, b))
        stored = np.load(self.tmp + "/pred_next.npy")
        self.assertEqual(stored.shape, (6, n))
        self.assertTrue(np.isfinite(stored).all())


if __name__ == "__main__":
    unittest.main()
