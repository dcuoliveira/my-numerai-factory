"""
Tests for the PDF results report.

The data-dependent ones are skipped when results/ is absent, because that tree
is gitignored -- a checkout without it should not report failures it cannot fix.
What they guard is not the rendering but the arithmetic and the conventions:
that payout is rebuilt per period, that a slot still resolves to the run it is
supposed to, that two targets never share a table, and that nothing is dropped
without being named.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analysis import ensemble_screen, report_data, report_pdf, slots
from src.analysis.ablation_report import newey_west as canonical_newey_west
from src.analysis.results_table import (payout_series, row_from_frame,
                                        short_target, target_of, scored_target)
from src.live import registry
from src.prediction.metrics import summarize

HAS_RESULTS = os.path.isdir("results/per_period/crypto/lgb")
needs_results = unittest.skipUnless(HAS_RESULTS, "results/ tree not present")

#: The two slots whose recorded figure is knowingly not their stored series.
#: CURUPIRA_05_60's registry payout is a 20-day RE-SCORE out of results/cache/;
#: CURUPIRA_07's is a screen over payout series, not a run. Asserting the set
#: exactly means a THIRD slot drifting shows up as a failure rather than as a
#: footnote nobody reads.
KNOWN_MISMATCHES = {"CURUPIRA_05_60", "CURUPIRA_07"}


class PayoutArithmetic(unittest.TestCase):
    """Payout is a series first and a summary second."""

    def test_payout_is_rebuilt_per_period(self):
        # corr and mmc are correlated, so the payout Sharpe is not any function
        # of the two summaries. Build a case where combining them would be
        # visibly wrong and check the row takes the series route.
        rng = np.random.default_rng(0)
        corr = pd.Series(rng.normal(0.15, 0.08, 200))
        mmc = pd.Series(-0.9 * corr.to_numpy() * 0.5 + rng.normal(0.07, 0.01, 200))
        frame = pd.DataFrame({"corr": corr, "mmc": mmc})

        row = row_from_frame("synthetic", frame)
        want = summarize(payout_series(frame).dropna())
        self.assertAlmostEqual(row["payout"], want["mean"], places=12)
        self.assertAlmostEqual(row["p.sharpe"], want["sharpe"], places=12)
        self.assertAlmostEqual(row["dd(pay)"], want["max_drawdown"], places=12)

        # The wrong route, for contrast: it must not coincide by accident.
        naive = 0.1 * summarize(corr)["mean"] + summarize(mmc)["mean"]
        naive_sharpe = (0.1 * summarize(corr)["sharpe"]
                        + summarize(mmc)["sharpe"])
        self.assertAlmostEqual(row["payout"], naive, places=12)
        self.assertNotAlmostEqual(row["p.sharpe"], naive_sharpe, places=2)

    def test_calmar_is_infinite_against_a_zero_drawdown(self):
        frame = pd.DataFrame({"corr": [1.0] * 10, "mmc": [1.0] * 10})
        self.assertTrue(np.isinf(row_from_frame("flat", frame)["calmar"]))

    def test_n_counts_scored_periods(self):
        # Two different holes, deliberately treated differently.
        #
        # Row 3 has no corr but DOES have mmc, which proves the period had rows
        # and a reference -- so the missing corr means the prediction was
        # degenerate. The model earned 0 that day and is charged for it, so the
        # period is scored, not dropped.
        #
        # Row 4 has no mmc, so the payout cannot be formed at all and the period
        # drops. Charging a 0 there would invent a number.
        frame = pd.DataFrame({"corr": [0.1, 0.2, np.nan, 0.3],
                              "mmc": [0.05, 0.06, 0.07, np.nan]})
        row = row_from_frame("gappy", frame)
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["dead"], 1)

    def test_a_degenerate_period_is_charged_not_dropped(self):
        # The E10 case: an arm that predicts one constant value on half its
        # dates. Dropping those scores it only on the days it worked, which
        # ranked a collapsed arm FIRST in its group.
        frame = pd.DataFrame({"corr": [0.2] * 5 + [np.nan] * 5,
                              "mmc": [0.0] * 10})
        row = row_from_frame("collapsed", frame)

        self.assertEqual(row["dead"], 5)
        self.assertEqual(row["n"], 10)
        # Charged: 5 periods at 0.02 and 5 at 0.0.
        self.assertAlmostEqual(row["payout"], 0.01, places=12)
        self.assertAlmostEqual(row["CORR"], 0.1, places=12)
        # Dropped, the old behaviour, would have doubled both.
        self.assertNotAlmostEqual(row["payout"], 0.02, places=6)

    def test_a_healthy_arm_is_untouched(self):
        frame = pd.DataFrame({"corr": [0.1, 0.2, 0.3], "mmc": [0.01, 0.02, 0.03]})
        row = row_from_frame("healthy", frame)
        self.assertEqual(row["dead"], 0)
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["payout"], (0.02 + 0.04 + 0.06) / 3, places=12)

    def test_a_degenerate_period_does_not_shrink_the_common_window(self):
        # The subtler half of the bug: Block intersects the non-null payout
        # indices, so one arm's undefined periods vanished from EVERY arm's
        # window. In E10 that took the common window from 111 dates to 67 and
        # changed every number in the table, not just the collapsed arm's.
        dates = ["2026-01-%02d" % d for d in range(1, 11)]
        healthy = pd.DataFrame({"corr": [0.2] * 10, "mmc": [0.0] * 10}, index=dates)
        collapsed = pd.DataFrame({"corr": [0.2] * 5 + [np.nan] * 5,
                                  "mmc": [0.0] * 10}, index=dates)

        self.assertEqual(len(payout_series(healthy).dropna()), 10)
        self.assertEqual(len(payout_series(collapsed).dropna()), 10)
        shared = payout_series(healthy).dropna().index.intersection(
            payout_series(collapsed).dropna().index)
        self.assertEqual(len(shared), 10)


class NeweyWestHasOneDefinition(unittest.TestCase):
    """Four copies of this estimator already exist; a fifth would be a bug."""

    def test_screen_uses_the_canonical_estimator(self):
        rng = np.random.default_rng(7)
        series = pd.Series(rng.normal(0, 1, 300)).rolling(5).mean().dropna()
        self.assertIs(ensemble_screen.newey_west, canonical_newey_west)
        mean, _, t = ensemble_screen.newey_west(series, lag=20)
        self.assertAlmostEqual(mean, float(series.mean()), places=12)
        self.assertTrue(np.isfinite(t))


class TargetLabelling(unittest.TestCase):
    """The scored target comes from the forecast, not the config."""

    def test_short_target(self):
        self.assertEqual(short_target("target_binned_return_60"), "t60")
        self.assertEqual(short_target(None), "-")

    @needs_results
    def test_ensembles_are_not_mislabelled_as_20_day(self):
        # An ensemble's config carries no `target` key, so the old config
        # fallback reported every 60-day blend as 20-day.
        self.assertEqual(target_of("crypto", "ridge_glmnet", "t60_c05_blend"),
                         "target_binned_return_60")
        self.assertEqual(target_of("crypto", "lgb", "saci01_c03_c0560"),
                         "target_binned_return_20")

    @needs_results
    def test_scored_target_survives_a_reclaimed_forecast(self):
        # The forecast header is the primary source and must still win: a blend
        # writes no target into its config, so config-first would mislabel it.
        self.assertEqual(
            scored_target("crypto", "ridge_glmnet", "t60_c05_blend"),
            "target_binned_return_60")
        # But when the forecast is gone -- which is what pruning a concluded
        # study's grid cells does -- the config has to answer, or report_data
        # groups the arm under "unknown" and the block mixes 20-day with
        # 60-day. Simulated by naming a run with no forecast on disk.
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            here = os.getcwd()
            metrics = os.path.join(tmp, "results", "metrics", "crypto", "lgb")
            os.makedirs(metrics)
            with open(os.path.join(metrics, "gone.json"), "w") as handle:
                json.dump({"config": {"target": "target_binned_return_60"}}, handle)
            try:
                os.chdir(tmp)
                self.assertIsNone(target_of("crypto", "lgb", "gone"))
                self.assertEqual(scored_target("crypto", "lgb", "gone"),
                                 "target_binned_return_60")
            finally:
                os.chdir(here)

    @needs_results
    def test_blocks_never_mix_targets(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        self.assertGreater(len(blocks), 1, "expected a 20-day and a 60-day block")
        for block in blocks:
            targets = {arm.target for arm in block.arms}
            self.assertEqual(len(targets), 1, "block %s mixes %s"
                             % (block.target, targets))

    @needs_results
    def test_lag_follows_the_target(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        lags = {block.target: block.lag for block in blocks}
        self.assertEqual(lags["target_binned_return_20"], 20)
        # 470 sixty-day periods overlap ~98%, so they are roughly eight
        # effective independent windows and lag 20 would inflate every t.
        self.assertEqual(lags["target_binned_return_60"], 60)


class SlotResolution(unittest.TestCase):
    """Every deployed slot must reach a real series."""

    @needs_results
    def test_every_slot_resolves(self):
        unresolved = [name for name in registry.ORDER
                      if slots.slot_run(name)[0] is None]
        self.assertEqual(unresolved, [], "slots with no backtest series")

    @needs_results
    def test_slots_reproduce_the_registry(self):
        resolved, dropped = slots.slot_rows("all")
        self.assertEqual(dropped, [])
        self.assertEqual(len(resolved), len(registry.ORDER))
        mismatched = {s.name for s in resolved if not s.matches_registry}
        self.assertEqual(mismatched, KNOWN_MISMATCHES)
        for slot in resolved:
            if slot.name in KNOWN_MISMATCHES:
                self.assertTrue(slot.note, "%s must carry its reason" % slot.name)
            else:
                self.assertAlmostEqual(float(slot.payout.mean()),
                                       slot.registry_payout, places=6)

    @needs_results
    def test_curupira_07_is_the_reconstruction_not_the_screen(self):
        model, run, note = slots.slot_run("CURUPIRA_07")
        self.assertEqual((model, run), ("ridge_glmnet", "xae_c07_base"))
        self.assertIn("SCREEN", note)


class NothingIsDroppedSilently(unittest.TestCase):
    """A blank row reads as 'tested and null' when it means 'never run'."""

    @needs_results
    def test_unmatched_patterns_are_returned(self):
        pairs, missing = report_data.expand_runs(
            "crypto", "lgb", ["e05_cumlag3", "no_such_run", "no_such_glob_*"])
        self.assertEqual(sorted(missing), ["no_such_glob_*", "no_such_run"])
        self.assertEqual(pairs, [("lgb", "e05_cumlag3")])

    @needs_results
    def test_a_run_that_is_a_deployed_slot_is_named_not_hidden(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3", "--no-pdf"])
        _, _, dropped = report_data.build(args)
        reasons = dict(dropped)
        self.assertIn("e05_cumlag3", reasons)
        self.assertIn("SACI_05", reasons["e05_cumlag3"])

    @needs_results
    def test_main_survives_a_bad_run_name(self):
        code = report_pdf.main(["--model", "lgb", "--runs", "e05_cumlag3",
                                "no_such_run", "--no-pdf"])
        self.assertEqual(code, 0)


class Windowing(unittest.TestCase):
    """Calmar is only comparable at equal n, so the block makes n equal."""

    @needs_results
    def test_common_window_is_the_intersection(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "fe_lgb_reg_raw", "e01_tenc_onehot_m50",
            "--submitted", "none", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        by_target = {b.target: b for b in blocks}
        rows = by_target["target_binned_return_20"].rows()
        self.assertEqual({r["n"] for r in rows}, {510})

    @needs_results
    def test_own_window_reproduces_the_registry(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "fe_lgb_reg_interact",
            "--submitted", "none", "--window", "own", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        row = blocks[0].rows()[0]
        self.assertAlmostEqual(row["payout"],
                               registry.get("SACI_01")["backtest"]["payout"],
                               places=6)

    @needs_results
    def test_rows_keep_their_native_length(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "fe_lgb_reg_raw", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        block = next(b for b in blocks if b.target == "target_binned_return_20")
        lengths = {r["n(own)"] for r in block.rows()}
        self.assertIn(510, lengths)
        self.assertIn(470, lengths)
        self.assertEqual({r["n"] for r in block.rows()}, {470})


class BaselineChoice(unittest.TestCase):
    """The naive maximum is a trap; the filters are the point."""

    @needs_results
    def test_best_single_submitted_excludes_blends_and_non_slots(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        block = next(b for b in blocks if b.target == "target_binned_return_20")
        self.assertEqual(block.baseline.label, "SACI_01")
        self.assertEqual(block.baseline.kind, "sub")
        rejected = dict(block.rejected)
        # The highest-paying arm overall is a slot blend, so it must appear as
        # rejected with a reason rather than quietly winning.
        self.assertIn("CURUPIRA_SACI_03", rejected)
        self.assertIn("single model", rejected["CURUPIRA_SACI_03"])

    @needs_results
    def test_a_hand_override_is_annotated(self):
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3",
            "--baseline", "CURUPIRA_02", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        block = next(b for b in blocks if b.target == "target_binned_return_20")
        self.assertEqual(block.baseline.label, "CURUPIRA_02")
        self.assertIn("overridden by hand", dict(block.rejected)["--baseline"])


class ToleranttCorrelationLoader(unittest.TestCase):
    """corr_heatmap's SystemExit is right there and fatal here."""

    @needs_results
    def test_a_short_panel_is_excluded_not_fatal(self):
        # xgb's only run covers 128 dates on a panel that is a strict subset of
        # the main one, so an intersection with it succeeds QUIETLY and cuts the
        # matrix to 128 dates. It must be dropped and named instead.
        if not os.path.exists("results/forecast/crypto/xgb/"
                              "regression_corr_expanding.csv"):
            self.skipTest("xgb run not present")
        def arm(model, run):
            return report_data.Arm(label=run, model=model, run=run, kind="prop",
                                  target="target_binned_return_20", frame=None,
                                  payout=None, staked=False, live_status="-",
                                  note=None)

        arms = [arm("lgb", "fe_lgb_reg_raw"), arm("lgb", "fe_lgb_reg_interact"),
                arm("xgb", "regression_corr_expanding")]
        wide, n_dates, excluded = report_data.load_predictions_common(
            "crypto", arms, min_shared_dates=400)
        self.assertGreaterEqual(n_dates, 400)
        self.assertIn("regression_corr_expanding", dict(excluded))
        self.assertNotIn("regression_corr_expanding", wide.columns)


class PdfOutput(unittest.TestCase):
    """The document itself."""

    @needs_results
    def test_a_pdf_is_produced(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "report.pdf")
            code = report_pdf.main(["--model", "lgb", "--runs", "e05_cumlag3",
                                    "--out", out])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))
            with open(out, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")
            self.assertGreater(os.path.getsize(out), 20000)

    @needs_results
    def test_the_printed_table_and_the_pdf_share_one_row_list(self):
        # The PDF and stdout must not drift: that is exactly how CLAUDE.md came
        # to document a `status` column that render() has never emitted.
        args = report_pdf.parse_args([
            "--model", "lgb", "--runs", "e05_cumlag3", "--no-pdf"])
        blocks, _, _ = report_data.build(args)
        header, body = report_data.render_block(blocks[0], "payout")
        self.assertEqual(len(body), len(blocks[0].arms))
        self.assertIn("dd(pay)", header[0])
        self.assertIn("dd(corr)", header[0])
        self.assertIn("n(own)", header[0])


if __name__ == "__main__":
    unittest.main()
