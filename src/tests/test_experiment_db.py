"""
Tests for the experiment index.

The index is only useful if its status is TRUE, and every failure mode here is
silent: a stale row, a study reporting as running when it finished, a scout arm
counted as a result. None of those raise -- they just make the table wrong while
it still looks authoritative.

Run with:  python3 -m pytest src/tests/test_experiment_db.py -q
"""

import json
import os
import sqlite3
import tempfile
import unittest
import csv

from src.analysis import experiment_db as edb


class TestStudiesTable(unittest.TestCase):
    """The declared intent has to stay consistent with the repo around it."""

    def test_every_study_names_a_runner_that_exists(self):
        # A study whose runner has been renamed cannot be rerun, and the index
        # would keep pointing at the old path indefinitely.
        for study_id, spec in edb.STUDIES.items():
            self.assertTrue(os.path.exists(spec["runner"]),
                            "%s names a missing runner: %s"
                            % (study_id, spec["runner"]))

    def test_prefixes_are_unique_and_scouts_do_not_collide(self):
        # e08_ must not swallow e08scout_. The scan sorts prefixes longest-first
        # for exactly this reason; this pins the property it relies on.
        #
        # The identifying key is (prefix, model tree), not the prefix alone --
        # E28 and E29 both write e28_ and are separated by tree, which is how
        # scan_arms has always read them. The uniqueness assertion lives in
        # test_no_two_studies_share_a_prefix_and_a_model_tree; a bare-prefix
        # assertion here would encode a rule the scan does not use.
        for study_id, spec in edb.STUDIES.items():
            scout = spec.get("scout_prefix")
            if scout:
                self.assertTrue(scout.startswith(spec["prefix"].rstrip("_")),
                                "%s scout prefix is unrelated to its own" % study_id)
                self.assertNotEqual(scout, spec["prefix"])

    def test_every_study_has_an_idea_worth_reading(self):
        # The whole point of the idea column is that it says what the study ASKS.
        # A placeholder defeats it, and nothing else would catch one.
        for study_id, spec in edb.STUDIES.items():
            self.assertGreater(len(spec["idea"]), 60, study_id)
            self.assertIn(" ", spec["title"], study_id)


class TestScanAndState(unittest.TestCase):
    """Scanning is where a scout arm could be mistaken for a result."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.metrics = os.path.join(self.tmp, "results", "metrics", "crypto", "toy")
        os.makedirs(self.metrics)
        self._root = edb.METRICS_ROOT
        edb.METRICS_ROOT = os.path.join(self.tmp, "results", "metrics", "crypto")

    def tearDown(self):
        edb.METRICS_ROOT = self._root

    def test_checkpoint_study_waits_for_complete_task_and_final_scores(self):
        directory = os.path.join(self.tmp, 'custom')
        arm = os.path.join(directory, 'fits', 'toy__cs')
        os.makedirs(arm)
        spec = {'prefix': 'ex_', 'models': ['toy'], 'artifact_root': directory}
        json.dump({'unfinished': True}, open(os.path.join(arm, 'block0.json'), 'w'))
        self.assertEqual(edb.scan_arms('EX', spec), [])
        json.dump({'task': {'tag': 'toy__cs', 'model': 'toy'}},
                  open(os.path.join(arm, 'COMPLETE.json'), 'w'))
        partial = edb.scan_arms('EX', spec)
        self.assertEqual(len(partial), 1)
        self.assertIsNone(partial[0][7])
        with open(os.path.join(directory, 'all_model_window_rows.csv'), 'w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['tag', 'corr', 'mmc', 'payout', 'n_dates'])
            writer.writeheader()
            writer.writerow({'tag': 'toy__cs', 'corr': .1, 'mmc': .05, 'payout': .06, 'n_dates': 468})
        finished = edb.scan_arms('EX', spec)
        self.assertEqual(finished[0][1], 'ex_toy__cs')
        self.assertEqual(finished[0][6:10], (468., .1, .05, .06))
        self.assertIsNone(finished[0][5])  # per-fold active-ID dimension is not guessed

    def _write(self, name, corr=0.1, mmc=0.05):
        json.dump(
            {"corr": {"mean": corr, "sharpe": 1.0, "max_drawdown": 0.5},
             "mmc": {"mean": mmc}, "n_periods": 510, "runtime_minutes": 3.0,
             "config": {"target": "target_binned_return_20", "n_features": 22}},
            open(os.path.join(self.metrics, name + ".json"), "w"))

    def test_a_scout_arm_is_never_counted_as_a_full_one(self):
        # ex_ is a prefix of exscout_, so a naive startswith would label the
        # scout 'full' and inflate the completion count.
        spec = {"prefix": "ex_", "scout_prefix": "exscout_", "models": ["toy"]}
        self._write("ex_alpha")
        self._write("exscout_alpha")
        kinds = {row[1]: row[3] for row in edb.scan_arms("EX", spec)}
        self.assertEqual(kinds["ex_alpha"], "full")
        self.assertEqual(kinds["exscout_alpha"], "scout")

    def test_seed_checkpoints_are_not_arms(self):
        # run_nn_forecast writes <run>.seeds.json beside the metrics; counting it
        # would double every LSTM study's arm count.
        spec = {"prefix": "ex_", "scout_prefix": None, "models": ["toy"]}
        self._write("ex_alpha")
        json.dump({"1": 0.5}, open(os.path.join(self.metrics, "ex_alpha.seeds.json"), "w"))
        self.assertEqual(len(edb.scan_arms("EX", spec)), 1)

    def test_state_reports_partial_rather_than_done_when_arms_are_missing(self):
        self.assertEqual(edb.state_of("E", 0, 12, set()), "not started")
        self.assertEqual(edb.state_of("E", 5, 12, set()), "partial")
        self.assertEqual(edb.state_of("E", 12, 12, set()), "done")
        # A live runner outranks the artifact count: a study on its last arm has
        # every earlier artifact on disk and is still running.
        self.assertEqual(edb.state_of("E", 12, 12, {"E"}), "running")


class TestRefreshIsDestructive(unittest.TestCase):
    """
    Status must never survive the thing that justified it.

    refresh() empties the tables rather than upserting, so a run deleted from
    results/ disappears here too. An index that keeps rows for artifacts that no
    longer exist fails the same way a hand-maintained status field does.
    """

    def test_a_deleted_arm_disappears_from_the_index(self):
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "experiments.db")
        edb.refresh(db)
        connection = sqlite3.connect(db)
        try:
            connection.execute(
                "INSERT INTO arms VALUES ('E01','ghost_run','lgb','full',NULL,"
                "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)")
            connection.commit()
        finally:
            connection.close()

        edb.refresh(db)
        connection = sqlite3.connect(db)
        try:
            left = connection.execute(
                "SELECT COUNT(*) FROM arms WHERE run_name='ghost_run'").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(left, 0, "a refresh kept a row with no artifact behind it")


class TestStudyOf(unittest.TestCase):
    """
    The prefix lookup the report's `exp` column is built on.

    EXPERIMENTS.md's naming rule exists so `e13_*` retrieves one study years
    later; this is what turns that into an index. Every failure here is silent:
    a scout labelled as a full arm, or a study's rows scattered under "-".
    """

    def test_a_full_arm_and_its_scout_resolve_to_the_same_study(self):
        self.assertEqual(edb.study_of("e13_adl"), ("E13", False))
        self.assertEqual(edb.study_of("e13scout_adl"), ("E13", True))

    def test_a_scout_is_never_read_as_a_full_arm(self):
        # e13_ is a prefix of e13scout_, so the longest-prefix-first ordering is
        # load-bearing: without it a scout would be labelled E13 and read as a
        # full-resolution result.
        for study_id, spec in edb.STUDIES.items():
            scout = spec.get("scout_prefix")
            if not scout:
                continue
            resolved, is_scout = edb.study_of(scout + "somearm")
            self.assertEqual(resolved, study_id)
            self.assertTrue(is_scout, "%s scout resolved as a full arm" % study_id)

    def test_runs_outside_every_study_resolve_to_nothing(self):
        # Deployed slots' backtests predate the numbering and must not be
        # forced into a study.
        for run in ("w6r6_fixed_reg", "saci04_c05_c08", "w6r6_glmnet_reg_tenc"):
            self.assertEqual(edb.study_of(run), (None, False), run)

    def test_every_declared_study_resolves_its_own_prefix(self):
        # Pass the tree: E28 and E29 share the prefix e28_, and the tree is what
        # separates them -- the same rule scan_arms has always applied.
        for study_id, spec in edb.STUDIES.items():
            got = edb.study_of(spec["prefix"] + "x", model=spec["models"][0])[0]
            self.assertEqual(got, study_id, study_id)

    def test_a_shared_prefix_is_split_by_model_tree(self):
        # E29 launched as "E28" while a concurrent session held the id. The
        # artifacts never collided because they live in different trees; this is
        # that fact made available to the name lookup.
        self.assertEqual(edb.study_of("e28_allfeat", model="mlp_ktz")[0], "E28")
        self.assertEqual(edb.study_of("e28_allfeat", model="ridge_glmnet")[0], "E29")

    def test_no_two_studies_share_a_prefix_and_a_model_tree(self):
        # A genuine collision -- same prefix AND same tree -- is unresolvable by
        # any rule, so it has to fail here rather than silently mis-attribute.
        seen = {}
        for study_id, spec in edb.STUDIES.items():
            for model in spec["models"]:
                key = (spec["prefix"], model)
                self.assertNotIn(
                    key, seen, "%s collides with %s on %s" % (study_id, seen.get(key), key))
                seen[key] = study_id

    def test_the_retrospectively_registered_studies_resolve(self):
        # Both ran under an id another study already held, so their artifacts
        # belonged to no entry and read as orphaned. E50 matters beyond the
        # count: e29_tenc_onehot is SACI_06's backtest_run.
        self.assertEqual(edb.study_of("ktz_nl0031_lr0p001_ne02000_cs0p20", model="lgb")[0], "E49")
        self.assertEqual(edb.study_of("e29_tenc_onehot", model="lgb_ktz")[0], "E50")
        # ktz_ is a RUN prefix, not a tree name -- it must not swallow the
        # ktz_farm tree, whose runs belong to E46.
        self.assertEqual(edb.study_of("e46_pool_a", model="ktz_farm")[0], "E46")


class TestRunningDetection(unittest.TestCase):
    def test_the_detector_does_not_find_itself(self):
        # The first version matched the runner name anywhere in a command line,
        # so the shell writing this module -- carrying every runner name in its
        # own argv -- made all nine studies report as running.
        live = edb.running_studies()
        self.assertLessEqual(
            len(live), len(edb.STUDIES),
            "running_studies() cannot report more studies than exist")
        self.assertNotIn("E01", live,
                         "E01 finished long ago; a hit means self-matching is back")


if __name__ == "__main__":
    unittest.main()
