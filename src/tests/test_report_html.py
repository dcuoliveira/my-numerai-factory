"""
Tests for the interactive HTML report.

The one that matters is `JavaScriptMatchesPython`. Every figure on the page is
recomputed in the browser so that the date filter moves the table, the score
correlations, the ensemble screen and the chart together -- which means the same
arithmetic now exists twice, once in `report_data` and once in `report_html`'s
embedded script. Two implementations drift. This runs the embedded script under
node against the embedded data and asserts it reproduces Python's rows, its
paired Newey-West deltas, its ensemble screen and its score matrix to 1e-9, so
the drift fails a test rather than a decision.

The rest guard the properties a file that gets mailed around has to keep: it
loads nothing off the network, its payload is real JSON, and a run name is
treated as text rather than markup.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from src.analysis import report_data, report_html

HAS_RESULTS = os.path.isdir("results/per_period/crypto/lgb")
needs_results = unittest.skipUnless(HAS_RESULTS, "results/ tree not present")

#: Anything bigger than this and the page stops being something you can mail.
MAX_MB = 12.0

#: The arithmetic is float, the transport is decimal. Twelve decimals in the
#: payload keeps the browser within this of Python after a 510-period cumsum.
TOLERANCE = 1e-9


def build_html(runs=("e05_cumlag[0-4]",), extra=()):
    """
    Render the report to a temporary file and return (path, payload, args).

    :param runs: run patterns
    :type runs: (str,)
    :param extra: further CLI arguments
    :type extra: (str,)
    :return: (html path, parsed payload, parsed args)
    :rtype: (str, dict, argparse.Namespace)
    """
    handle, path = tempfile.mkstemp(suffix=".html")
    os.close(handle)
    argv = ["--model", "lgb", "--runs"] + list(runs) + ["--out", path] + list(extra)
    assert report_html.main(argv) == 0
    text = open(path).read()
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        text, re.S)
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    return path, payload, report_html.parse_args(argv)


def node_available():
    """Whether node can run the compute module at all."""
    if not shutil.which("node"):
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(report_html.COMPUTE_JS)
        name = handle.name
    try:
        done = subprocess.run(["node", "--check", name], capture_output=True)
        return done.returncode == 0
    finally:
        os.unlink(name)


NODE = node_available()


class SelfContained(unittest.TestCase):
    """A file that gets mailed around cannot depend on a network."""

    @needs_results
    def test_nothing_is_fetched(self):
        path, _, _ = build_html()
        try:
            text = open(path).read()
        finally:
            os.unlink(path)
        for pattern in ('<script src=', '<link ', "@import", "https://",
                        "//cdn", "fonts.googleapis", "fonts.gstatic"):
            self.assertNotIn(pattern, text, "page reaches out for %r" % pattern)
        # The only http: in the file is the SVG namespace, which is an
        # identifier the parser matches on, never a URL anything fetches.
        for hit in re.findall(r"http://[^\s\"']+", text):
            self.assertTrue(hit.startswith("http://www.w3.org/"),
                            "page reaches out for %r" % hit)

    @needs_results
    def test_size_is_mailable(self):
        path, _, _ = build_html()
        try:
            megabytes = os.path.getsize(path) / 1e6
        finally:
            os.unlink(path)
        self.assertLess(megabytes, MAX_MB, "%.1f MB is too big to mail" % megabytes)

    @needs_results
    def test_payload_is_strict_json(self):
        # allow_nan=False at the Python end: a bare NaN in the payload parses in
        # node and throws in every browser, which would be a blank page.
        path, payload, _ = build_html()
        try:
            self.assertTrue(payload["blocks"])
            self.assertTrue(payload["dates"])
            for block in payload["blocks"]:
                for arm in block["arms"]:
                    self.assertEqual(len(arm["corr"]), len(block["dates"]))
                    for value in arm["corr"]:
                        self.assertTrue(value is None or value == value)
        finally:
            os.unlink(path)

    @needs_results
    def test_a_run_name_is_text_not_markup(self):
        path, _, _ = build_html()
        try:
            text = open(path).read()
        finally:
            os.unlink(path)
        # The static shell escapes; the dynamic half uses textContent. Neither
        # should ever be built by pasting a label into an HTML string.
        self.assertNotIn(".innerHTML", report_html.UI_JS)
        self.assertIn("textContent", report_html.UI_JS)
        self.assertIn("&lt;", report_html.esc("<script>"))
        self.assertNotIn("<script>alert", text)


class WorksWithoutScripts(unittest.TestCase):
    """
    The page must carry its numbers before any script runs.

    This is the bug this class exists for: the first version rendered every
    section from JavaScript, so in a sandboxed preview pane or behind a policy
    that refuses inline script the reader got headings, captions and empty
    cards -- a results report with no results in it. Every section is now
    written out server-side and the script replaces it, so the failure mode is
    "you cannot sort it", not "there is nothing there".
    """

    @staticmethod
    def _without_scripts(path):
        return re.sub(r"<script[^>]*>.*?</script>", "", open(path).read(), flags=re.S)

    @needs_results
    def test_every_section_has_its_numbers(self):
        path, payload, args = build_html()
        try:
            text = self._without_scripts(path)
        finally:
            os.unlink(path)

        blocks, _, _ = report_data.build(args)
        block = [b for b in blocks
                 if report_data.short_target(b.target) == payload["blocks"][0]["id"]][0]

        # the results table, cell by cell for a row nobody could fake
        rows = {r["run"]: r for r in block.rows()}
        for name in ("SACI_01", "CURUPIRA_08"):
            self.assertIn(name, text)
            self.assertIn("%.5f" % rows[name]["payout"], text)
            self.assertIn("%.2f" % rows[name]["p.sharpe"], text)

        # two correlation matrices per block that has them -- prediction and
        # score. Both blocks here do, so this is 4, not 2.
        with_corr = sum(1 for b in payload["blocks"] if b["predCorr"])
        self.assertGreater(with_corr, 0)
        self.assertEqual(text.count('<table class="heat">'), 2 * with_corr)

        # the ensemble screen, ranked, with its warning intact
        self.assertIn("screened payout", text)
        # `>` is escaped in the rendered page, as it must be
        self.assertIn("beat their own best member on payout at |t| &gt; 2", text)

        # the chart, as real geometry rather than an empty box
        self.assertIn('<path class="baseline"', text)
        self.assertIn('<path class="series"', text)
        self.assertGreater(text.count('<path class='), 6)

    @needs_results
    def test_the_baseline_is_drawn_as_the_baseline(self):
        # block.baseline is an Arm, not a label. Comparing it to a label
        # silently never matched, and the baseline drew as a grey context line.
        path, payload, _ = build_html()
        try:
            text = self._without_scripts(path)
        finally:
            os.unlink(path)
        chart = re.search(r'<div class="chart-wrap"[^>]*>(.*?)</svg>', text, re.S)
        self.assertIsNotNone(chart)
        # one baseline path in the cumulative panel, one in the underwater panel
        self.assertEqual(chart.group(1).count('<path class="baseline"'), 2)
        self.assertIn("(baseline)", text)

    @needs_results
    def test_a_reader_without_scripts_is_told_what_is_missing(self):
        path, _, _ = build_html()
        try:
            text = open(path).read()
        finally:
            os.unlink(path)
        self.assertIn("<noscript>", text)
        self.assertIn("Scripts are off in this viewer", text)

    @needs_results
    def test_static_and_interactive_agree(self):
        # Both halves read the same rows, so the number the reader sees before
        # the script runs must be the number they see after it.
        path, payload, args = build_html()
        try:
            text = self._without_scripts(path)
        finally:
            os.unlink(path)
        blocks, _, _ = report_data.build(args)
        block = [b for b in blocks
                 if report_data.short_target(b.target) == payload["blocks"][0]["id"]][0]
        for row in block.rows():
            self.assertIn("%.5f" % row["payout"], text,
                          "%s's payout is not in the static table" % row["run"])


class StaysLightUnderInteraction(unittest.TestCase):
    """
    Interaction must not queue one full re-render per input event.

    This is the bug this class exists for: a range slider fires `input` on every
    pixel of a drag, and each one called renderAll -- rows, both matrices, an
    8,436-combination ensemble screen and two SVG panels. A two-second drag
    queued minutes of synchronous work and enough DOM garbage to take a machine
    down. Continuous controls now schedule one coalesced render instead.
    """

    CONTINUOUS = ("onRange", "search.addEventListener")

    def test_continuous_controls_are_coalesced(self):
        ui = report_html.UI_JS
        self.assertIn("function scheduleRender", ui)
        self.assertIn("clearTimeout(pendingRender)", ui)

        # the slider and the search box must go through the scheduler
        for marker in self.CONTINUOUS:
            start = ui.index(marker)
            window = ui[start:start + 400]
            self.assertIn("scheduleRender", window,
                          "%s does not coalesce its renders" % marker)
            self.assertNotIn("renderAll();", window,
                             "%s calls renderAll directly" % marker)

    def test_the_page_is_held_rather_than_blanked(self):
        # No skeleton, no layout jump: the previous render stays put at reduced
        # opacity until the new one lands.
        self.assertIn('classList.add("stale")', report_html.UI_JS)
        self.assertIn('classList.remove("stale")', report_html.UI_JS)
        self.assertIn("body.stale", report_html.CSS)

    @needs_results
    def test_the_ensemble_screen_is_capped(self):
        path, payload, args = build_html(runs=("e05_*",))
        os.unlink(path)
        biggest = max(payload["blocks"], key=lambda b: len(b["arms"]))
        self.assertGreater(len(biggest["arms"]), args.ens_max_arms,
                           "need more arms than the cap for this to mean anything")
        self.assertEqual(payload["ensMaxArms"], args.ens_max_arms)

    @needs_results
    def test_the_cap_keeps_both_proposals_and_the_book(self):
        # Taking proposals first and filling afterwards looks right until a
        # study has twenty arms, at which point the screen holds nothing to
        # compare them against.
        args = report_html.parse_args(["--model", "lgb", "--runs", "e05_*",
                                       "--no-html"])
        blocks, _, _ = report_data.build(args)
        block = max(blocks, key=lambda b: len(b.arms))
        kept, cut = report_data.matrix_arms(block, args.ens_max_arms)
        self.assertEqual(len(kept), args.ens_max_arms)
        self.assertTrue(cut)
        kinds = {a.kind for a in kept}
        self.assertIn("prop", kinds)
        self.assertTrue(kinds - {"prop"}, "the cap crowded out the whole book")
        self.assertIn(block.baseline.label, [a.label for a in kept])


class Structure(unittest.TestCase):
    """The page carries the same sections, and the same warnings, as the PDF."""

    @needs_results
    def test_sections_and_warnings_are_present(self):
        path, payload, _ = build_html()
        try:
            text = open(path).read()
        finally:
            os.unlink(path)
        for needed in ("Results", "Correlation", "Ensembles",
                       "Cumulative backtested payout", "Notes"):
            self.assertIn(needed, text)
        self.assertIn("SCREEN, NOT A PRICE", text)
        self.assertIn("dd(pay)", text)
        self.assertEqual(len(payload["blocks"]), 2, "expected a t20 and a t60 block")
        self.assertEqual({b["id"] for b in payload["blocks"]}, {"t20", "t60"})

    @needs_results
    def test_blocks_never_mix_targets(self):
        path, payload, _ = build_html()
        os.unlink(path)
        for block in payload["blocks"]:
            self.assertEqual({a["tgt"] for a in block["arms"]}, {block["tgt"]})

    @needs_results
    def test_the_lag_follows_the_target(self):
        path, payload, _ = build_html()
        os.unlink(path)
        lags = {b["id"]: b["lag"] for b in payload["blocks"]}
        self.assertEqual(lags["t20"], 20)
        self.assertEqual(lags["t60"], 60)

    @needs_results
    def test_series_colour_is_stable_per_arm(self):
        # `order` is what the page indexes the palette by. If it were rebuilt
        # from whatever is currently visible, hiding one arm would repaint the
        # others and a reader who learnt "cumlag3 is blue" would be misled.
        path, payload, _ = build_html()
        os.unlink(path)
        for block in payload["blocks"]:
            self.assertEqual(block["order"], [a["label"] for a in block["arms"]])
            self.assertEqual(len(set(block["order"])), len(block["order"]))


@unittest.skipUnless(NODE, "node is not available to run the embedded script")
class JavaScriptMatchesPython(unittest.TestCase):
    """The embedded script must reproduce report_data, or the page lies."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.compute = os.path.join(cls.dir, "compute.js")
        with open(cls.compute, "w") as handle:
            handle.write(report_html.COMPUTE_JS)

    def _run(self, script, payload):
        data = os.path.join(self.dir, "payload.json")
        with open(data, "w") as handle:
            json.dump(payload, handle)
        runner = os.path.join(self.dir, "run.js")
        with open(runner, "w") as handle:
            handle.write(
                'var C = require(%s);\n'
                'var R = JSON.parse(require("fs").readFileSync(%s, "utf8"));\n'
                % (json.dumps(self.compute), json.dumps(data)) + script)
        done = subprocess.run(["node", runner], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    @staticmethod
    def _close(a, b):
        # A missing value is None from JSON and NaN from pandas -- the same
        # thing said two ways. It happens for real: two arms with identical
        # predictions give a delta series of exact zeros, so the Newey-West
        # standard error is 0 and t is undefined on both sides.
        missing = lambda v: v is None or (isinstance(v, float) and v != v)
        if missing(a) or missing(b):
            return missing(a) and missing(b)
        if a in (float("inf"), float("-inf")) or b in (float("inf"), float("-inf")):
            return a == b
        return abs(float(a) - float(b)) < TOLERANCE

    @needs_results
    def test_rows_and_paired_deltas(self):
        path, payload, args = build_html()
        os.unlink(path)
        got = self._run('''
var b = R.blocks[0], base = null, i;
for (i = 0; i < b.arms.length; i++) if (b.arms[i].label === b.baseline) base = b.arms[i];
var out = b.arms.map(function (a) {
  var r = C.rowFor(a, 0, b.dates.length);
  if (base && a.label !== b.baseline) {
    var nw = C.pairedDelta(a, base, 0, b.dates.length, b.lag);
    r.dpay = nw.mean; r.t = nw.t;
  }
  return r;
});
process.stdout.write(JSON.stringify(out));
''', payload)

        blocks, _, _ = report_data.build(args)
        block = [b for b in blocks
                 if report_data.short_target(b.target) == payload["blocks"][0]["id"]][0]
        want = {r["run"]: r for r in block.rows()}
        have = {r["run"]: r for r in got}
        self.assertEqual(set(want), set(have))

        for name in want:
            for key in ("n", "n(own)", "CORR", "MMC", "c.sharpe", "dd(corr)",
                        "payout", "p.sharpe", "dd(pay)", "calmar", "dpay", "t"):
                self.assertTrue(
                    self._close(want[name].get(key), have[name].get(key)),
                    "%s.%s: python %r, browser %r"
                    % (name, key, want[name].get(key), have[name].get(key)))

    @needs_results
    def test_ensemble_screen(self):
        path, payload, args = build_html()
        os.unlink(path)
        got = self._run('''
var b = R.blocks[0];
var res = C.screenEnsembles(b.arms, 0, b.dates.length, R.ensSizes, b.lag, 999999);
process.stdout.write(JSON.stringify(res));
''', payload)

        blocks, _, _ = report_data.build(args)
        block = [b for b in blocks
                 if report_data.short_target(b.target) == payload["blocks"][0]["id"]][0]
        rows, _, periods = report_data.screen(
            {a.label: block.payout(a) for a in block.arms},
            tuple(args.ens_size), block.lag)
        self.assertEqual(got["n"], periods)
        self.assertEqual(len(got["rows"]), len(rows))

        key = lambda r: tuple(sorted(r["members"]))
        want = {key(r): r for r in rows}
        have = {key(r): r for r in got["rows"]}
        self.assertEqual(set(want), set(have), "different combination sets")
        for combo in want:
            self.assertEqual(want[combo]["best"], have[combo]["best"])
            for field in ("payout", "sharpe", "maxdd", "delta", "t", "rho"):
                self.assertTrue(
                    self._close(want[combo][field], have[combo][field]),
                    "%s.%s: python %r, browser %r"
                    % (combo, field, want[combo][field], have[combo][field]))

    @needs_results
    def test_score_correlation_matrix(self):
        path, payload, args = build_html()
        os.unlink(path)
        got = self._run('''
var b = R.blocks[0];
process.stdout.write(JSON.stringify(C.scoreMatrix(b.arms, 0, b.dates.length)));
''', payload)

        blocks, _, _ = report_data.build(args)
        block = [b for b in blocks
                 if report_data.short_target(b.target) == payload["blocks"][0]["id"]][0]
        order = [a.label for a in block.arms]
        matrix, periods = report_data.score_series_matrix(block)
        matrix = matrix.loc[order, order]
        self.assertEqual(got["n"], periods)
        for i in range(len(order)):
            for j in range(len(order)):
                self.assertTrue(
                    self._close(float(matrix.iat[i, j]), got["matrix"][i][j]),
                    "%s vs %s" % (order[i], order[j]))

    @needs_results
    def test_a_narrowed_window_actually_changes_the_numbers(self):
        # The whole point of computing in the browser: if the filter did not
        # move the figures this would all be theatre.
        path, payload, _ = build_html()
        os.unlink(path)
        got = self._run('''
var b = R.blocks[0], n = b.dates.length;
process.stdout.write(JSON.stringify({
  full: C.rowFor(b.arms[0], 0, n),
  half: C.rowFor(b.arms[0], Math.floor(n / 2), n)
}));
''', payload)
        self.assertLess(got["half"]["n"], got["full"]["n"])
        self.assertNotAlmostEqual(got["half"]["payout"], got["full"]["payout"],
                                  places=9)


if __name__ == "__main__":
    unittest.main()
