"""
Tests for the cluster resource manager -- the ways it could overrun a shared
machine while looking like it worked.

Everything here is offline: no ssh, no probing, no launching. The arithmetic in
src/cluster/budget.py is pure for exactly this reason, so the decisions that
matter can be tested without a cluster to test them on. These cover the
mistakes that produce a plausible placement rather than an error:

  * a cap computed by rounding, so 30% of 32 cores becomes 10
  * work placed on a box whose load already exceeds its cores
  * two jobs pinned to the same cpus, each believing it has the slot
  * a second batch that re-spends cpus the first batch is already holding
  * a GPU job sent to a box whose driver is unreachable, or onto a card
    somebody else is using
  * a 30% promise quietly rounded up to one of two cards
  * jobs that did not fit being dropped instead of reported
  * a thread count left at the machine's core count inside a 4-cpu slot

Run with:  python -m unittest discover -s src/tests
"""

import unittest

from src.cluster.budget import (SHARE, Host, Job, Placement, available_cores,
                                core_cap, distribute, format_cpu_list, free_gpus,
                                gpu_cap, parse_cpu_list, pick_cpus, ram_cap_gb)
from src.cluster.manager import job_script, launch_script, parse_jobs, slug


def host(name="h", cores=32, ram_gb=251, gpus=0, load1=0.0, **kw):
    return Host(name=name, cores=cores, ram_gb=ram_gb, gpus=gpus, load1=load1, **kw)


class TestTheCap(unittest.TestCase):
    """30% means 30%, floored -- never rounded up to be helpful."""

    def test_real_core_counts(self):
        self.assertEqual(core_cap(host(cores=32)), 9)    # 9.6 floored
        self.assertEqual(core_cap(host(cores=64)), 19)   # 19.2 floored
        self.assertEqual(ram_cap_gb(host(ram_gb=251)), 75)

    def test_cap_is_floor_not_round(self):
        self.assertEqual(core_cap(host(cores=33)), 9)    # 9.9 -> 9, not 10
        self.assertEqual(core_cap(host(cores=3)), 0)     # 0.9 -> 0, not 1

    def test_share_is_configurable_but_defaults_to_30pc(self):
        self.assertAlmostEqual(SHARE, 0.30)
        self.assertEqual(core_cap(host(cores=64), share=0.5), 32)

    def test_saturated_host_offers_nothing(self):
        """deepsix on 2026-10-23: load 44 on 32 cores. Its 9-core budget is
        unreachable, and handing out 9 anyway is how you make it worse."""
        self.assertEqual(available_cores(host(cores=32, load1=44.0)), 0)

    def test_partly_loaded_host_offers_what_is_left(self):
        # 32 cores, 28 of them busy: 4 free, under the 9-core budget.
        self.assertEqual(available_cores(host(cores=32, load1=28.0)), 4)

    def test_budget_binds_before_free_cores_do(self):
        # Idle 64-core box: 64 free, but we may still only take 19.
        self.assertEqual(available_cores(host(cores=64, load1=0.0)), 19)

    def test_unreachable_host_offers_nothing(self):
        self.assertEqual(available_cores(host(reachable=False, cores=64)), 0)

    def test_our_own_cpus_do_not_count_as_other_load(self):
        """Our pinned job shows up in load1. Subtracting it twice would make
        the host look busier than it is and starve the next placement."""
        h = host(cores=32, load1=4.0, our_cpus=(0, 1, 2, 3))
        self.assertEqual(h.others_load, 0.0)
        self.assertEqual(available_cores(h), 5)        # 9 budget - 4 held


class TestCpuSets(unittest.TestCase):
    def test_roundtrip_including_taskset_prefix(self):
        self.assertEqual(parse_cpu_list("pid 12's current affinity list: 0-3,8"),
                         (0, 1, 2, 3, 8))
        self.assertEqual(format_cpu_list((0, 1, 2, 3, 8)), "0-3,8")
        self.assertEqual(format_cpu_list(()), "")
        self.assertEqual(parse_cpu_list(""), ())

    def test_sets_avoid_cpus_we_already_hold(self):
        h = host(cores=32, our_cpus=(0, 1, 2))
        self.assertEqual(pick_cpus(h, 2), (3, 4))

    def test_two_jobs_on_one_host_get_disjoint_sets(self):
        jobs = [Job("a", "x", cores=4), Job("b", "y", cores=4)]
        placed, unplaced = distribute(jobs, [host(cores=32)])
        self.assertEqual(unplaced, [])
        a, b = (p.cpus for p in placed)
        self.assertFalse(set(a) & set(b), "jobs share cpus: %s vs %s" % (a, b))

    def test_total_pinned_never_exceeds_the_cap(self):
        jobs = [Job("j%d" % i, "x", cores=3) for i in range(10)]   # 30 cores asked
        placed, unplaced = distribute(jobs, [host(cores=32)])
        used = sum(len(p.cpus) for p in placed)
        self.assertLessEqual(used, core_cap(host(cores=32)))
        self.assertTrue(unplaced, "should refuse what does not fit")

    def test_a_second_batch_respects_the_first(self):
        """The cap is per machine, not per invocation. Occupancy is read back
        from live affinities and passed in; ignoring it doubles our footprint."""
        h = host(cores=32, our_cpus=tuple(range(9)), load1=9.0)
        placed, unplaced = distribute([Job("late", "x", cores=1)], [h])
        self.assertEqual(placed, [])
        self.assertEqual(len(unplaced), 1)


class TestGpus(unittest.TestCase):
    def test_strict_default_forbids_gpu_work_on_a_two_card_box(self):
        self.assertEqual(gpu_cap(host(gpus=2)), 0)
        self.assertEqual(free_gpus(host(gpus=2)), ())

    def test_override_is_explicit_and_capped_by_the_hardware(self):
        self.assertEqual(free_gpus(host(gpus=2), gpu_slots=1), (0,))
        self.assertEqual(gpu_cap(host(gpus=2), gpu_slots=9), 2)

    def test_card_in_use_by_anyone_is_not_offered(self):
        self.assertEqual(free_gpus(host(gpus=2, busy_gpus=(0,)), gpu_slots=1), (1,))

    def test_broken_driver_host_never_takes_a_gpu_job(self):
        """deepfour has two cards and an unreachable driver, so the probe
        reports gpus=0. Counting nvidia-smi's error message as a card would
        send the job to a box that cannot run it."""
        dead = host(name="deepfour", gpus=0, note="gpu driver unreachable")
        placed, unplaced = distribute([Job("g", "x", gpus=1)], [dead], gpu_slots=1)
        self.assertEqual(placed, [])
        self.assertIn("GPU", unplaced[0][1])

    def test_gpu_job_goes_to_the_gpu_box(self):
        boxes = [host(name="hamilton", cores=64), host(name="deepfive", gpus=2)]
        placed, unplaced = distribute([Job("g", "x", gpus=1)], boxes, gpu_slots=1)
        self.assertEqual(unplaced, [])
        self.assertEqual(placed[0].host, "deepfive")
        self.assertEqual(placed[0].gpus, (0,))


class TestDistribution(unittest.TestCase):
    def test_work_spreads_rather_than_packs(self):
        """Worst-fit: the request is to use the FREE machines. Best-fit would
        fill one box to its ceiling and leave the others idle."""
        boxes = [host(name="a", cores=64), host(name="b", cores=64)]
        placed, _ = distribute([Job("j1", "x", cores=8), Job("j2", "y", cores=8)], boxes)
        self.assertEqual(len({p.host for p in placed}), 2)

    def test_biggest_job_first(self):
        """A 19-core job placed after small ones never fits anywhere."""
        boxes = [host(name="a", cores=64)]
        jobs = [Job("small%d" % i, "x", cores=1) for i in range(5)] + [Job("big", "x", cores=19)]
        placed, unplaced = distribute(jobs, boxes)
        self.assertIn("big", [p.job.name for p in placed])

    def test_ram_can_be_the_binding_constraint(self):
        boxes = [host(cores=64, ram_gb=100)]      # cap 30 G
        placed, unplaced = distribute([Job("hog", "x", cores=1, ram_gb=40)], boxes)
        self.assertEqual(placed, [])
        self.assertIn("RAM", unplaced[0][1])

    def test_nothing_is_silently_dropped(self):
        jobs = [Job("fits", "x", cores=1), Job("huge", "x", cores=99)]
        placed, unplaced = distribute(jobs, [host(cores=32)])
        self.assertEqual([p.job.name for p in placed], ["fits"])
        self.assertEqual([j.name for j, _ in unplaced], ["huge"])

    def test_no_hosts_means_everything_unplaced(self):
        placed, unplaced = distribute([Job("a", "x")], [])
        self.assertEqual(placed, [])
        self.assertEqual(len(unplaced), 1)


class TestJobsFile(unittest.TestCase):
    def _write(self, text):
        import tempfile
        fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        fh.write(text)
        fh.close()
        return fh.name

    def test_defaults_and_overrides(self):
        jobs, errors = parse_jobs(self._write(
            "cores=8 ram=16 name=ridge -- python3 -m src.run_forecast crypto ridge\n"
            "# a comment\n"
            "\n"
            "python3 -m src.analysis.report_html --model lgb\n"))
        self.assertEqual(errors, [])
        self.assertEqual((jobs[0].name, jobs[0].cores, jobs[0].ram_gb), ("ridge", 8, 16))
        self.assertEqual((jobs[1].cores, jobs[1].ram_gb, jobs[1].gpus), (1, 4, 0))
        self.assertTrue(jobs[1].command.startswith("python3 -m src.analysis"))

    def test_duplicate_names_are_an_error(self):
        _, errors = parse_jobs(self._write("name=x -- a\nname=x -- b\n"))
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_malformed_line_is_reported_not_skipped(self):
        _, errors = parse_jobs(self._write("cores -- a\n"))
        self.assertTrue(any("key=value" in e for e in errors))

    def test_missing_command_is_reported(self):
        _, errors = parse_jobs(self._write("cores=4 --\n"))
        self.assertTrue(any("no command" in e for e in errors))

    def test_slug_makes_a_safe_unit_name(self):
        self.assertEqual(slug("E120 ridge/glmnet"), "e120-ridge-glmnet")


class TestLaunchScript(unittest.TestCase):
    def _placement(self, cores=4, gpus=(), command="python3 -m src.run_forecast crypto ridge"):
        return Placement(job=Job("j", command, cores=cores), host="h",
                         cpus=tuple(range(cores)), gpus=gpus)

    def script(self, **kw):
        """The systemd-run invocation."""
        return launch_script(self._placement(**kw), repo="/repo")

    def body(self, **kw):
        """The command file it runs."""
        return job_script(self._placement(**kw), repo="/repo")

    def test_pins_the_allocated_cpus(self):
        self.assertIn("taskset -c 0-3", self.script())

    def test_thread_counts_match_the_slot_not_the_machine(self):
        """Affinity caps the cpus; without this a LightGBM search opens one
        thread per machine core and thrashes inside its 4-cpu slot."""
        s = self.script(cores=4)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertIn("--setenv=%s=4" % var, s)

    def test_cpu_job_cannot_grab_a_card(self):
        self.assertIn("--setenv=CUDA_VISIBLE_DEVICES= ", self.script(gpus=()) + " ")

    def test_gpu_job_sees_only_its_card(self):
        self.assertIn("--setenv=CUDA_VISIBLE_DEVICES=1", self.script(gpus=(1,)))

    def test_runs_in_the_repo_with_the_right_interpreter_on_path(self):
        self.assertIn("miniconda3/bin", self.script())
        self.assertIn("cd /repo", self.body())

    def test_command_never_reaches_systemds_command_line(self):
        """systemd expands `$` in its own ExecStart and collapses `$$` to `$`.
        A command containing either must go through a file, not the unit."""
        p = Placement(job=Job("j", "python3 x.py --pid $$ --home $HOME", cores=2),
                      host="h", cpus=(0, 1))
        self.assertNotIn("$$", launch_script(p, repo="/repo"))
        self.assertIn("--pid $$ --home $HOME", job_script(p, repo="/repo"))

    def test_named_unit_so_stop_needs_no_pattern(self):
        """`pkill -f` over ssh matches the ssh command carrying the pattern."""
        self.assertIn("--unit=nmr-j", self.script())

    def test_output_goes_to_the_shared_log_dir(self):
        self.assertIn("logs/cluster/j.out", self.body())

    def test_command_with_shell_metacharacters_is_not_reinterpreted(self):
        body = self.body(command="python3 x.py --name 'a b' && echo $HOME")
        self.assertIn("python3 x.py --name 'a b' && echo $HOME", body)


if __name__ == "__main__":
    unittest.main()
