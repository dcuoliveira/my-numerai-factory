"""
Tests for the live submission pipeline.

    python3 -m pytest src/tests/test_live.py -q

This code moves real money on a deadline, so the tests concentrate on the three
failure modes that would cost a round rather than on happy-path coverage:

  * a submission file that Numerai would reject (NaN, duplicates, out of range)
  * a fallback ladder that does not actually catch a fall
  * a refit cadence or embargo that quietly stops matching the backtest

The heavy fixtures (a fitted estimator, the real panel) are avoided on purpose:
these run in about a second and can sit in a pre-submission hook.
"""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.live import registry, run_round, submit
from src.live.features import (ONEHOT_MIN_DATES, ROLL_WINDOWS, TENC_LAG,
                               add_derived,
                               add_onehot, add_rolling, add_tenc, build_features)
from src.live.refit import feature_signature, refit_due, training_block

SYMBOL, PERIOD, TARGET = "symbol", "date", "target_binned_return_20"


def make_panel(n_symbols=6, n_periods=60, n_live=1, seed=0):
    """A small synthetic panel with the same shape as the real one."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_periods + n_live, freq="D")
    rows = []
    for s in range(n_symbols):
        for i, date in enumerate(dates):
            live = i >= n_periods
            row = {SYMBOL: "SYM%d" % s, PERIOD: date, "is_live": live,
                   TARGET: np.nan if live else rng.choice([0, .25, .5, .75, 1.0])}
            for feature in ("feature_sharpe_ratio", "feature_relative_strength_index",
                            "feature_volatility", "feature_momentum"):
                for horizon in ("20d", "60d"):
                    row["%s_%s" % (feature, horizon)] = rng.choice([0, .25, .5, .75, 1.0])
            rows.append(row)
    return pd.DataFrame(rows).sort_values([SYMBOL, PERIOD]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Submission file validity -- the guards that stop a bad file leaving the box
# --------------------------------------------------------------------------

def test_rank_normalise_is_strictly_inside_the_unit_interval():
    values = np.array([-5.0, 0.0, 0.0, 0.0, 12.0])
    out = submit.rank_normalise(values)
    assert (out > 0).all() and (out < 1).all()
    # method="first" keeps ties as a strict permutation rather than collapsing
    # them to one rank, which would throw away ordering the model did express.
    assert len(set(out)) == len(values)


def test_rank_normalise_preserves_order():
    values = np.array([3.0, 1.0, 2.0])
    out = submit.rank_normalise(values)
    assert list(np.argsort(out)) == list(np.argsort(values))


@pytest.mark.parametrize("frame,reason", [
    (pd.DataFrame({SYMBOL: ["A", "A"], "prediction": [0.1, 0.2]}), "duplicate"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [np.nan]}), "NaN"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [1.5]}), "outside"),
    (pd.DataFrame({SYMBOL: ["A"], "prediction": [0.0]}), "outside"),
])
def test_write_csv_rejects_invalid_submissions(frame, reason, tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        with pytest.raises(ValueError, match=reason):
            submit.write_csv(frame, 1, "T")
    finally:
        submit.SUBMISSION_DIR = original


# --------------------------------------------------------------------------
# Fallback ladder
# --------------------------------------------------------------------------

class ConstantEstimator(object):
    def predict(self, X):
        return np.arange(len(X), dtype="float64")


def write_round(root, round_number, name, source, symbols=("A",)):
    """A submitted CSV plus the round report that says what produced it."""
    directory = os.path.join(str(root), str(round_number))
    os.makedirs(directory, exist_ok=True)
    pd.DataFrame({SYMBOL: list(symbols),
                  "prediction": np.linspace(0.1, 0.9, len(symbols))}
                 ).to_csv(os.path.join(directory, "%s.csv" % name), index=False)
    with open(os.path.join(directory, "round_report.json"), "w") as handle:
        json.dump({"dry_run": False,
                   "slots": [{"name": name, "status": "submitted",
                              "source": source}]}, handle)


def test_fallback_ladder_descends_one_rung_at_a_time(tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        panel = make_panel()
        live_date = panel[PERIOD].max()
        columns = [c for c in panel.columns if c.startswith("feature_")]
        symbols = panel.loc[panel["is_live"], SYMBOL].tolist()

        def context(estimators):
            return {"estimators": estimators, "live_date": live_date,
                    "round_number": 10, "live_symbols": symbols}

        entry = (ConstantEstimator(), columns, panel, {})

        # 1. own model
        _, source, _ = submit.build_predictions("CURUPIRA_03", context(
            {"CURUPIRA_03": entry}))
        assert source == "model"

        # 2. own model absent -> the raw-feature sibling stands in
        _, source, note = submit.build_predictions("CURUPIRA_03", context(
            {"CURUPIRA_02": entry}))
        assert source == "sibling" and "CURUPIRA_02" in note

        # 3. nothing fitted -> last round's file, but only because round 9
        #    recorded it as a genuine "model" submission.
        previous = os.path.join(str(tmp_path), "9")
        write_round(tmp_path, 9, "CURUPIRA_03", "model", symbols)
        _, source, _ = submit.build_predictions("CURUPIRA_03", context({}))
        assert source == "previous_round"

        # 4. no history at all -> Numerai's example predictions
        shutil.rmtree(previous)
        example = pd.DataFrame({SYMBOL: symbols, "prediction": 0.5})
        example_path = os.path.join(str(tmp_path), "example.csv")
        example.to_csv(example_path, index=False)
        original_example = submit.EXAMPLE_PREDS
        submit.EXAMPLE_PREDS = example_path
        try:
            _, source, _ = submit.build_predictions("CURUPIRA_03", context({}))
            assert source == "example"
        finally:
            submit.EXAMPLE_PREDS = original_example
    finally:
        submit.SUBMISSION_DIR = original


def test_previous_submission_never_reaches_forward_in_time(tmp_path):
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        for round_number in (9, 12):
            write_round(tmp_path, round_number, "CURUPIRA_01", "model")
        found = submit.previous_submission("CURUPIRA_01", 10)
        assert found.endswith(os.path.join("9", "CURUPIRA_01.csv")), (
            "round 12 is in the future of round 10 and must never be reused")
    finally:
        submit.SUBMISSION_DIR = original


def test_a_fallback_is_never_laundered_forward_as_a_previous_round(tmp_path):
    """
    Round 1338's CURUPIRA_SACI_03.csv WAS Numerai's example file, byte for byte.
    Round 1339 re-served it under source "previous_round", which reads clean
    against CLAUDE.md's own detection recipe -- grep the report for
    "source": "example" -- while the numbers going out were unchanged benchmark
    ranks. Each hop also re-intersects with the universe without padding, so the
    book decays: 280 rows, then 275.
    """
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        write_round(tmp_path, 9, "CURUPIRA_SACI_03", "example")
        assert submit.previous_submission("CURUPIRA_SACI_03", 10) is None

        write_round(tmp_path, 9, "CURUPIRA_SACI_03", "previous_round")
        assert submit.previous_submission("CURUPIRA_SACI_03", 10) is None, (
            "a chain of previous_round hops is the same laundering, one step on")

        write_round(tmp_path, 9, "CURUPIRA_SACI_03", "sibling")
        assert submit.previous_submission("CURUPIRA_SACI_03", 10) is not None, (
            "a sibling stand-in is a real model's prediction and may be reused")
    finally:
        submit.SUBMISSION_DIR = original


def test_a_stale_submission_is_not_reused_for_ever(tmp_path):
    """
    The 95%-overlap argument for reusing yesterday's file holds for ONE round on
    a 20-day target. It does not hold for five, and max() over every earlier
    round had no bound at all.
    """
    original = submit.SUBMISSION_DIR
    submit.SUBMISSION_DIR = str(tmp_path)
    try:
        write_round(tmp_path, 5, "CURUPIRA_01", "model")
        assert submit.previous_submission("CURUPIRA_01", 10) is None
    finally:
        submit.SUBMISSION_DIR = original


def test_upload_retries_then_succeeds(monkeypatch, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("symbol,prediction\nA,0.5\n")
    monkeypatch.setattr(submit.time, "sleep", lambda _: None)

    calls = {"n": 0}

    class Flaky(object):
        def upload_predictions(self, file_path, model_id):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("502 Bad Gateway")
            return "sub-id"

    submission_id, attempts = submit.upload(Flaky(), str(path), "mid", "T")
    assert submission_id == "sub-id" and attempts == 3


def test_upload_gives_up_after_the_last_attempt(monkeypatch, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("symbol,prediction\nA,0.5\n")
    monkeypatch.setattr(submit.time, "sleep", lambda _: None)

    class Broken(object):
        def upload_predictions(self, file_path, model_id):
            raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="after %d attempts" % submit.MAX_UPLOAD_ATTEMPTS):
        submit.upload(Broken(), str(path), "mid", "T")


def test_dry_run_never_touches_the_api():
    class Exploding(object):
        def upload_predictions(self, *a, **k):
            raise AssertionError("dry run must not upload")

    assert submit.upload(Exploding(), "x", "mid", "T", dry_run=True) == ("DRY_RUN", 0)


# --------------------------------------------------------------------------
# Crash survival -- what a failed write and a mid-round abort must not destroy
# --------------------------------------------------------------------------

def test_a_failed_history_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """
    The 2026-09-04 regression. live_history.parquet is the ONLY copy of the
    accumulated live history and rebuilds one date per round, so an in-place
    write that dies mid-flight costs weeks. A disk quota ran out inside
    to_parquet and took three weeks of dates with it, leaving one date behind.
    """
    from src.live import features

    path = str(tmp_path / "live_history.parquet")
    first = make_panel(n_symbols=3, n_periods=0, n_live=2)
    features.append_live_history(first, path=path)
    before = pd.read_parquet(path)
    assert len(before) > 0

    def explode(self, *a, **k):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    later = make_panel(n_symbols=3, n_periods=0, n_live=2, seed=7)
    with pytest.raises(OSError):
        features.append_live_history(later, path=path)

    monkeypatch.undo()
    after = pd.read_parquet(path)
    pd.testing.assert_frame_equal(before, after)
    assert not os.path.exists(path + ".partial"), "staging file left behind"


def test_run_reports_slot_progress_to_its_caller():
    """
    A crash is not the same as "nothing was submitted". Round 1348 died on a
    disk quota at slot 19 of 19 -- 18 already uploaded and confirmed on-time --
    and mailed CRASHED -- NOTHING SUBMITTED. main() reads this list to say what
    actually landed, so run() must append to the caller's object, not a copy.
    """
    import inspect
    from src.live import run_round

    assert "progress" in inspect.signature(run_round.run).parameters
    assert "round_info" in inspect.signature(run_round.run).parameters

    source = inspect.getsource(run_round.run)
    assert "slots = progress if progress is not None else []" in source, \
        "run() must append into the caller's list, not rebind it"

    handler = inspect.getsource(run_round.main)
    assert "NOTHING WAS SUBMITTED" in handler
    assert 'status") == "submitted"' in handler, \
        "the crash mail must count what landed before claiming nothing did"


# --------------------------------------------------------------------------
# Which round is open -- and which source is allowed to say so
# --------------------------------------------------------------------------

# The published round table on 2026-08-17, verbatim. It tops out at 1334, the
# weekend round: open Saturday, staking closed Sunday 16:00Z, still "current" by
# closeTime until Tuesday.
SCHEDULE = [
    {"number": 1333, "openTime": "2026-08-14T12:00:00Z",
     "closeStakingTime": "2026-08-14T13:06:14Z"},
    {"number": 1334, "openTime": "2026-08-15T12:00:00Z",
     "closeStakingTime": "2026-08-16T16:00:00Z"},
]

# What the submission service saw at the same moment: round 1335, open since
# 12:00Z, accepting until 16:00Z. The table had not heard of it 49 minutes later.
SUBMISSIONS = [
    {"roundNumber": 1335, "roundOpenTime": "2026-08-17T12:00:00.000000Z",
     "roundCloseStakingTime": "2026-08-17T16:00:00.000000Z", "status": "none"},
    {"roundNumber": 1334, "roundOpenTime": "2026-08-15T12:00:00Z",
     "roundCloseStakingTime": "2026-08-16T16:00:00Z", "status": "on-time"},
]


class FakeApi(object):
    """Answers the round table and the submission history."""

    def __init__(self, schedule=SCHEDULE, submissions=SUBMISSIONS, failures=0):
        self.schedule, self.submissions = schedule, submissions
        self.failures, self.calls = failures, 0

    def raw_query(self, query, arguments=None, authorization=True):
        self.calls += 1
        if self.calls <= self.failures:
            raise KeyError("data")          # how numerapi surfaces a blip
        if "latestSubmissions" in query:
            if self.submissions is None:
                raise KeyError("data")
            return {"data": {"model": {"latestSubmissions": list(self.submissions)}}}
        return {"data": {"rounds": list(self.schedule)}}

    def get_current_round(self):
        raise AssertionError(
            "get_current_round answers with closeTime, which is not the "
            "submission cutoff -- it must not decide the target round")


def at(when):
    return pd.Timestamp(when)


def test_open_round_finds_the_round_inside_its_staking_window():
    number, _, deadline, minutes_left, state, _ = run_round.open_round(
        FakeApi(submissions=None), now=at("2026-08-15 12:05:00"), verbose=False)
    assert (number, state) == (1334, "open")
    assert deadline == at("2026-08-16 16:00:00")
    assert minutes_left == pytest.approx(28 * 60 - 5, abs=1)


def test_a_long_round_is_still_open_the_next_day():
    number, _, _, _, state, _ = run_round.open_round(
        FakeApi(submissions=None), now=at("2026-08-16 12:05:00"), verbose=False)
    assert (number, state) == (1334, "open")


def test_the_submission_service_outranks_the_lagging_round_table():
    """
    The regression, and it cuts the opposite way to how it looked.

    At 12:05Z on 2026-08-17 round 1335 was open and took all six slots on time.
    The published table still topped out at 1334 -- and still did 44 minutes
    later -- so the pipeline decided the window had shut on the Sunday and
    mailed LATE about six submissions Numerai had accepted. Believing the table
    and skipping would have been worse still: a missed round.
    """
    number, _, deadline, minutes_left, state, via = run_round.open_round(
        FakeApi(), model_id="mid", now=at("2026-08-17 12:05:00"), verbose=False)
    assert (number, state) == (1335, "open")
    assert deadline == at("2026-08-17 16:00:00")
    assert minutes_left == pytest.approx(235, abs=1)
    assert via == "submission history"


def test_a_genuinely_closed_window_is_reported_closed():
    number, _, _, minutes_left, state, _ = run_round.open_round(
        FakeApi(), model_id="mid", now=at("2026-08-17 16:30:00"), verbose=False)
    assert (number, state) == (1335, "closed")
    assert minutes_left < 0


def test_a_round_that_has_not_published_its_deadline_yet_counts_as_open():
    schedule = [dict(SCHEDULE[1], closeStakingTime=None)]
    number, _, deadline, _, state, _ = run_round.open_round(
        FakeApi(schedule, submissions=None), now=at("2026-08-15 12:00:30"),
        verbose=False)
    assert (number, deadline, state) == (1334, None, "open")


def test_an_unreadable_api_is_unknown_rather_than_closed():
    """Never infer "closed" from an API failure -- that would skip a live round."""
    api = FakeApi(failures=99)
    assert run_round.open_round(api, model_id="mid", verbose=False)[4] == "unknown"


def test_the_round_queries_are_retried_before_giving_up():
    api = FakeApi(submissions=None, failures=run_round.ROUND_QUERY_ATTEMPTS - 1)
    number, _, _, _, state, _ = run_round.open_round(
        api, now=at("2026-08-15 12:05:00"), verbose=False)
    assert (number, state) == (1334, "open")


# Round numbers repeat. On 2026-08-18 the API served two rounds both numbered
# 1335: the one that opened on the 17th and closed 16:00Z, and the one that
# opened on the 18th and closed 13:04Z. Both come back in the same list.
REISSUED = [
    {"roundNumber": 1335, "roundOpenTime": "2026-08-17T12:00:00Z",
     "roundCloseStakingTime": "2026-08-17T16:00:00Z", "status": "queued"},
    {"roundNumber": 1335, "roundOpenTime": "2026-08-18T12:00:00Z",
     "roundCloseStakingTime": "2026-08-18T13:04:01Z", "status": "none"},
    {"roundNumber": 1334, "roundOpenTime": "2026-08-15T12:00:00Z",
     "roundCloseStakingTime": "2026-08-16T16:00:00Z", "status": "on-time"},
]


def test_a_reissued_round_number_resolves_to_the_window_that_is_open():
    """
    The second regression, and it cost a live round.

    Ranking candidates by round number picked whichever 1335 came first -- the
    stale one -- so on 2026-08-18 the 09:05 and 09:25 passes both reported "no
    open round" while a round sat open with 38 minutes left on it.
    """
    api = FakeApi(schedule=[], submissions=REISSUED)
    number, opened, deadline, minutes_left, state, _ = run_round.open_round(
        api, model_id="mid", now=at("2026-08-18 12:26:00"), verbose=False)
    assert (number, state) == (1335, "open")
    assert opened == at("2026-08-18 12:00:00")
    assert deadline == at("2026-08-18 13:04:01")
    assert minutes_left == pytest.approx(38, abs=1)


def test_with_both_windows_closed_the_later_one_is_reported():
    api = FakeApi(schedule=[], submissions=REISSUED)
    _, opened, _, _, state, _ = run_round.open_round(
        api, model_id="mid", now=at("2026-08-18 14:00:00"), verbose=False)
    assert state == "closed" and opened == at("2026-08-18 12:00:00")


def test_status_is_read_from_the_right_window_of_a_reissued_number():
    """Matching on the number alone returned yesterday's verdict for 6 of 7 slots."""
    api = FakeApi(schedule=[], submissions=REISSUED)
    stale = run_round.confirm_statuses(api, {"C01": "mid"}, 1335, verbose=False)
    assert stale == {"C01": "queued"}, "no open time given -- picks the first match"

    api = FakeApi(schedule=[], submissions=REISSUED)
    fresh = run_round.confirm_statuses(
        api, {"C01": "mid"}, 1335, opened=at("2026-08-18 12:00:00"), verbose=False)
    assert fresh == {"C01": "none"}


def test_the_numerai_verdict_is_stamped_once_however_many_passes_run():
    """Three passes inside one window must not stack three verdicts on a line."""
    slot = {"name": "C01", "detail": "cached fit | C01.csv | 300 rows"}
    for status in ("queued", "on-time", "on-time"):
        base = (slot.get("detail") or "-").split(" | numerai: ")[0]
        slot["detail"] = "%s | numerai: %s" % (base, status)
    assert slot["detail"] == "cached fit | C01.csv | 300 rows | numerai: on-time"
    assert slot["detail"].count("numerai:") == 1


def test_the_late_verdict_comes_from_numerai_not_from_the_clock():
    api = FakeApi(submissions=[
        {"roundNumber": 1335, "roundOpenTime": "2026-08-17T12:00:00Z",
         "roundCloseStakingTime": "2026-08-17T16:00:00Z", "status": "queued"}])
    statuses = run_round.confirm_statuses(
        api, {"CURUPIRA_01": "mid"}, 1335, verbose=False)
    assert statuses == {"CURUPIRA_01": "queued"}


# --------------------------------------------------------------------------
# What the round downloads, and how forgiving it is about each file
# --------------------------------------------------------------------------

def _dataset_specs():
    return {os.path.basename(d["remote"]): d for d in run_round.DATASETS}


def test_the_training_file_is_downloaded_every_round():
    """
    train.parquet was in no download list at all, so the "rolling" 130-period
    window could not roll. Every cached fit shared one training block ending
    2026-07-08 while fitted_at spanned nine days, and the live embargo had
    reached 32 periods against a backtested 20, widening by one every round --
    while Numerai was already serving 1711 dates to the 1701 sitting on disk.
    """
    from src.live.features import TRAIN_PARQUET
    specs = _dataset_specs()
    assert "train.parquet" in specs, "the training file must be fetched each round"
    train = specs["train.parquet"]
    assert train["refresh"], "an existing copy must not short-circuit the download"
    assert train["local"] == TRAIN_PARQUET, "must land where load_panel reads it"


def test_the_example_predictions_are_refreshed_rather_than_cached_forever():
    """
    The rule used to be `name != "live.parquet" and exists(local): continue`,
    which pinned the example file to whichever round first fetched it. By
    2026-08-22 that was round 1332, nine rounds stale, covering 279 of the 300
    live symbols -- and write_csv checks NaN, duplicates and range but never
    coverage, so the short book uploaded clean.
    """
    example = _dataset_specs()["live_example_preds.csv"]
    assert example["refresh"]
    # But a last-resort fallback failing to download must not kill the round.
    assert not example["required"]


def test_a_short_example_book_is_refused_rather_than_submitted(tmp_path):
    """A rung that cannot cover the universe has not worked; say so."""
    path = tmp_path / "example.csv"
    pd.DataFrame({SYMBOL: ["A", "B"], "prediction": [0.4, 0.6]}).to_csv(path, index=False)
    original = submit.EXAMPLE_PREDS
    submit.EXAMPLE_PREDS = str(path)
    try:
        assert len(submit.load_example_predictions(["A", "B"])) == 2
        with pytest.raises(ValueError, match="cover 2 of 3"):
            submit.load_example_predictions(["A", "B", "C"])
    finally:
        submit.EXAMPLE_PREDS = original


# --------------------------------------------------------------------------
# A retry must never make a round worse than it already is
# --------------------------------------------------------------------------

def test_a_retry_pass_never_replaces_a_better_submission():
    """
    Rounds 1338 and 1339 both lost a good submission this way.

    Pass 1 landed CURUPIRA_SACI_03 on 'sibling' -- a real CURUPIRA_02 ridge over
    the full 300-symbol universe. The 09:25 retry ran that slot alone, so the
    sibling was not in context and the chain fell to 'example'. Numerai keeps the
    LAST upload, so the benchmark replaced the model.
    """
    stood = {"name": "CURUPIRA_SACI_03", "status": "submitted", "source": "sibling"}
    assert run_round._would_downgrade(stood, "example")
    assert run_round._would_downgrade(stood, "previous_round")
    assert not run_round._would_downgrade(stood, "model")
    # Equal quality is not a downgrade -- a re-upload of the same rung is fine.
    assert not run_round._would_downgrade(stood, "sibling")


def test_nothing_standing_means_any_source_may_upload():
    """The guard must not block the FIRST submission of a round."""
    assert not run_round._would_downgrade(None, "example")
    assert not run_round._would_downgrade({}, "example")
    assert not run_round._would_downgrade({"name": "X", "status": "failed"}, "example")


def test_a_rehearsal_never_counts_as_a_standing_submission(tmp_path):
    """
    Same reason already_submitted ignores dry runs: a rehearsal records
    status "submitted" for every slot having uploaded nothing. Trusting it would
    let the guard block a real upload and leave the round empty.
    """
    original = run_round.REPORT_DIR
    run_round.REPORT_DIR = str(tmp_path)
    try:
        directory = os.path.join(str(tmp_path), "1400")
        os.makedirs(directory)
        report = {"dry_run": True,
                  "slots": [{"name": "CURUPIRA_01", "status": "submitted",
                             "source": "model"}]}
        with open(os.path.join(directory, "round_report.json"), "w") as handle:
            json.dump(report, handle)
        assert run_round.standing_records(1400) == {}

        report["dry_run"] = False
        with open(os.path.join(directory, "round_report.json"), "w") as handle:
            json.dump(report, handle)
        assert set(run_round.standing_records(1400)) == {"CURUPIRA_01"}
    finally:
        run_round.REPORT_DIR = original


# --------------------------------------------------------------------------
# Fidelity to the backtest protocol
# --------------------------------------------------------------------------

def test_registry_protocol_matches_the_backtest_config():
    fit = registry.FIT
    assert fit["train_window"] == 130 and fit["refit_every"] == 130
    assert fit["embargo"] == 20, "embargo must equal the 20-day target horizon"
    assert fit["cv_scheme"] == "rolling" and fit["n_splits"] == 5
    assert fit["n_iter"] == 50 and fit["seed"] == 42


def _touches_the_60_day_label(name, seen=None):
    """
    True when a slot is fitted on the 60-day target, or reuses one that is.

    Recursive because slot blends nest: CURUPIRA_07 and _08 are 20-day slots
    that carry CURUPIRA_05_60, and a slot blend of a slot blend would inherit it
    one level deeper still.
    """
    seen = seen or set()
    if name in seen:
        return False
    seen.add(name)

    # The BACKTEST's label, not the live fit target. SACI_03 fits the 20-day
    # label live while its backtest block is a 60-day study, so reading
    # fit["target"] here would expect 510 periods from a 470-period arm.
    if registry.backtest_target(name).endswith("_60"):
        return True
    spec = registry.get(name)
    return any(_touches_the_60_day_label(m["name"], seen)
               for m in spec.get("slot_members") or [])


def test_every_slot_declares_a_backtest_run_and_stats():
    """
    The period count is not one number, and hard-coding 510 was wrong.

    The 60-day target is NaN over its last 40 dates, so anything fitted on it --
    or blended from something that is -- scores 470 periods rather than 510.
    Asserting 510 for every slot failed the moment CURUPIRA_05_60 was added, and
    the fix is to assert the count the slot's own label implies, so that a slot
    silently scored on the wrong panel is still caught.
    """
    for spec in registry.all_specs():
        name = spec["name"]
        assert spec["backtest_run"], name
        assert spec["why"], name

        expected = 470 if _touches_the_60_day_label(name) else 510
        assert spec["backtest"]["n_periods"] == expected, (
            "%s declares %d periods; a slot %s the 60-day label should have %d"
            % (name, spec["backtest"]["n_periods"],
               "touching" if expected == 470 else "clear of", expected))


def test_every_submitted_slot_fits_the_label_numerai_pays_on():
    """
    Numerai scores live submissions on target_binned_return_20, so a slot that is
    uploaded in its own right must FIT that label -- whatever label its
    hyperparameters were searched on. Fitting the 60-day target and being scored
    on the 20-day one is predicting the wrong quantity, however good the 60-day
    study looked.

    SACI_03 is the case that forced this: searched on the 60-day label, promoted
    to a submitted slot on 2026-08-22, retargeted to 20 with its params kept.
    CURUPIRA_05_60 is the deliberate exception and is in NOT_SUBMITTED -- it is
    never uploaded, and its horizon is its entire contribution to four blends.
    """
    for name in registry.ORDER:
        if name in registry.NOT_SUBMITTED:
            continue
        target = registry.get(name)["fit"]["target"]
        assert target == "target_binned_return_20", (
            "%s is submitted but fits %s" % (name, target))


def test_the_retarget_kept_the_hyperparameters_the_60_day_study_selected():
    """
    The retarget changes the LABEL, not the model. Re-searching on the 20-day
    label would discard the study that justified deploying the slot, and the
    embargo has to follow the target or the guard is sized for the wrong horizon.
    """
    spec = registry.get("SACI_03")
    fixed = spec["fit_override"]["fixed_params"]
    assert fixed["n_estimators"] == 4327
    assert fixed["learning_rate"] == pytest.approx(0.002572)
    assert fixed["min_child_samples"] == 50, "50 keeps the 740 one-hot columns usable"
    assert spec["fit"]["embargo"] == 20, "embargo must follow the target horizon"
    # And the divergence stays legible: the backtest is still a 60-day study.
    assert registry.backtest_target("SACI_03") == "target_binned_return_60"
    assert spec["backtest"]["n_periods"] == 470


def test_submission_order_is_cheapest_first():
    """
    The submission window is ~66 minutes, so slots run in ascending FIT COST:
    if the deadline bites it must bite the most expensive slot, not the core.

    Cost, not feature count. Those tracked each other until CURUPIRA_05, a blend
    whose 763 union columns are cheaper to store but five times more expensive to
    fit than CURUPIRA_04's 812 -- five searches instead of one. Testing the proxy
    rather than the thing would have forced the wrong ordering.
    """
    from src.live.run_round import DEFAULT_FIT_SECONDS

    missing = [n for n in registry.ORDER if n not in DEFAULT_FIT_SECONDS]
    assert not missing, "no fit-cost estimate for %s" % missing

    # Slot blends are exempt from cheapest-first: they fit nothing, but they
    # load their members' caches, so a DEPENDENCY constraint overrides the cost
    # one. They are checked separately below.
    fitting = [n for n in registry.ORDER
               if not registry.is_slot_blend(registry.get(n))]
    costs = [DEFAULT_FIT_SECONDS[n] for n in fitting]
    assert costs == sorted(costs), (
        "fitting slots must run cheapest-first, got %s" % list(zip(fitting, costs)))
    assert registry.ORDER[0] == "CURUPIRA_02", "the no-build slot must go first"
    assert set(registry.ORDER) == set(registry.MODELS)


def test_every_slot_blend_can_be_fed_from_its_own_declared_feature_set():
    """
    load_slot_blend builds ONE frame from the blend's declared feature_set and
    promises it is a superset of every member's columns. Nothing checked that.

    CURUPIRA_SACI_04 broke the promise silently: it declared allfeat_pp, which
    calls add_cumlags from nowhere, so member SACI_05's 66 lag columns were
    absent, RankBlend.predict raised on them, and the slot uploaded Numerai's
    example predictions instead. The ordering test below would not have caught
    it -- position and weights were both fine.
    """
    panel = make_panel(n_periods=ONEHOT_MIN_DATES + 25)
    built = {}

    def columns_for(feature_set):
        if feature_set not in built:
            _, cols, _ = build_features(panel.copy(), feature_set, verbose=False)
            built[feature_set] = set(cols)
        return built[feature_set]

    for spec in registry.all_specs():
        if not registry.is_slot_blend(spec):
            continue
        outer = columns_for(spec["feature_set"])
        for member in spec["slot_members"]:
            member_set = registry.get(member["name"])["feature_set"]
            missing = columns_for(member_set) - outer
            assert not missing, (
                "%s builds '%s', which lacks %d columns its member %s ('%s') "
                "needs, e.g. %s" % (spec["name"], spec["feature_set"],
                                    len(missing), member["name"], member_set,
                                    sorted(missing)[:3]))


def test_a_slot_blend_may_be_a_member_of_another_slot_blend():
    """
    CURUPIRA_08 sits inside CURUPIRA_SACI_03, and a slot blend fits nothing and
    so persists no joblib. Looking one up on disk therefore returns nothing, and
    load_cached's docstring still asserts slot blends "never reach here" -- true
    only until a blend was made a member of another blend. load_slot_blend has to
    ASSEMBLE a nested member, recursively, rather than load it.

    Until it did, CURUPIRA_SACI_03 reported "CURUPIRA_08 (no cached fit)" and
    uploaded Numerai's example predictions every round from 1338 onward.
    """
    import inspect
    from src.live import refit as refit_module

    nested = [(spec["name"], member["name"])
              for spec in registry.all_specs()
              for member in (spec.get("slot_members") or [])
              if registry.is_slot_blend(registry.get(member["name"]))]
    assert nested, ("no slot blend nests another any more -- if that is "
                    "deliberate, delete this test rather than weakening it")
    # The recursion carries a cycle guard, because a membership loop would
    # otherwise recurse until the stack gives out.
    assert "seen" in inspect.signature(refit_module.load_slot_blend).parameters


def test_slot_blends_run_after_every_slot_they_reuse():
    """
    A slot blend loads its members' fitted caches, so it must come later in
    ORDER than all of them -- otherwise it would assemble from a stale or
    missing fit on the very round its member was refreshed.
    """
    position = {name: i for i, name in enumerate(registry.ORDER)}
    for name in registry.ORDER:
        spec = registry.get(name)
        if not registry.is_slot_blend(spec):
            continue
        for member in spec["slot_members"]:
            assert member["name"] in position, (
                "%s reuses unknown slot %s" % (name, member["name"]))
            assert position[member["name"]] < position[name], (
                "%s must run after its member %s" % (name, member["name"]))
        assert abs(sum(m["weight"] for m in spec["slot_members"]) - 1.0) < 1e-9
        # It must not also declare its own fittable members -- the two
        # composition types need opposite refit handling.
        assert not registry.is_blend(spec)


def test_training_block_refuses_to_train_inside_the_embargo():
    panel = make_panel(n_periods=40)
    columns = [c for c in panel.columns if c.startswith("feature_")]
    live_date = panel[PERIOD].max()
    # Only 1 day between the last training period and the live date, against a
    # 20-day horizon: those training rows' targets overlap the prediction date.
    with pytest.raises(ValueError, match="overlap"):
        training_block(panel, columns, train_window=130, embargo=20,
                       live_date=live_date)


def test_training_block_takes_exactly_the_last_window_periods():
    panel = make_panel(n_periods=60)
    columns = [c for c in panel.columns if c.startswith("feature_")]
    live_date = panel[PERIOD].max() + pd.Timedelta(days=40)
    X, y, meta, periods, gap = training_block(panel, columns, 30, 20, live_date)
    assert len(periods) == 30
    assert meta[PERIOD].nunique() == 30
    assert X.index.equals(y.index), "the scorer aligns through this index"
    assert not panel.loc[X.index, "is_live"].any(), "live rows must never train"


def test_refit_cadence():
    live = pd.Timestamp("2026-08-12")
    assert refit_due(None, live, 130, "sig")[0] is True
    fresh = {"live_date": "2026-08-05", "fitted_through": "2026-07-08",
             "feature_signature": "sig"}
    assert refit_due(fresh, live, 130, "sig")[0] is False
    stale = {"live_date": "2025-01-01", "fitted_through": "2024-12-01",
             "feature_signature": "sig"}
    assert refit_due(stale, live, 130, "sig")[0] is True
    # A changed one-hot vocabulary invalidates the cache regardless of the clock.
    due, reason = refit_due(fresh, live, 130, "different")
    assert due is True and "feature set changed" in reason


def test_the_cadence_is_counted_in_periods_not_calendar_days():
    """
    registry.FIT declares refit_every in PERIODS and this compared it against
    .days. The panel is weekday-only -- 1709 dates over 2424 calendar days -- so
    130 calendar days is 93 periods, a 29% short cadence. That lands between the
    r6m and r3m arms of the repo's own cadence sweep, whose conclusion is that
    refitting more often is actively worse.
    """
    assert (pd.bdate_range("2025-01-01", periods=131)[-1] - pd.Timestamp("2025-01-01")).days > 175, (
        "130 business periods should span ~181 calendar days, not 130")
    fitted = {"live_date": str(pd.Timestamp("2025-01-01").date()),
              "feature_signature": "sig"}
    # 130 CALENDAR days on: the old code refit here. It must not.
    day_130 = pd.Timestamp("2025-01-01") + pd.Timedelta(days=130)
    assert refit_due(fitted, day_130, 130, "sig")[0] is False
    # 130 PERIODS on: now it is genuinely due.
    assert refit_due(fitted, pd.bdate_range("2025-01-01", periods=131)[-1], 130, "sig")[0] is True


def test_the_cadence_clock_starts_at_zero_on_the_day_of_the_fit():
    """
    It used to measure from fitted_through -- the last TRAINING period -- so a
    model refit today reported itself 44 days old the same afternoon, because
    train.parquet lags live by that much. Each fit's counter started at the
    embargo rather than zero, making the effective cadence `refit_every - gap`
    and shrinking it as the gap grew; the 60-day slots started at 93 and 99, so
    their service life was set by their target horizon, not by the protocol.
    """
    live = pd.Timestamp("2026-02-25")
    just_fitted = {"live_date": str(live.date()),
                   "fitted_through": str(pd.Timestamp("2026-01-14").date()),
                   "feature_signature": "sig"}
    due, reason = refit_due(just_fitted, live, 130, "sig")
    assert due is False
    assert "0 of 130 periods" in reason, reason


def test_feature_signature_tracks_column_identity_and_order():
    assert feature_signature(["a", "b"]) == feature_signature(["a", "b"])
    assert feature_signature(["a", "b"]) != feature_signature(["b", "a"])
    assert feature_signature(["a", "b"]) != feature_signature(["a", "b", "c"])


# --------------------------------------------------------------------------
# Live feature construction
# --------------------------------------------------------------------------

def test_tenc_never_sees_a_live_row_target():
    panel = make_panel(n_periods=60, n_live=3)
    encoded, added = add_tenc(panel)
    assert added == ["feature_sym_tenc"]
    assert np.isfinite(encoded["feature_sym_tenc"]).all()
    # The lag means the live rows' support stops 20 periods short of the end.
    live_support = encoded.loc[encoded["is_live"], "_tenc_support"]
    assert (live_support <= 60 - TENC_LAG + 3).all()


def test_tenc_lag_equals_the_target_horizon():
    assert TENC_LAG == registry.FIT["embargo"] == 20


def test_onehot_vocabulary_comes_from_training_rows_only():
    panel = make_panel(n_symbols=4, n_periods=ONEHOT_MIN_DATES + 5, n_live=1)
    # A token that only ever appears live must not earn a column -- live, you
    # cannot have fit a coefficient for something you have never trained on.
    newcomer = panel[panel["is_live"]].head(1).copy()
    newcomer[SYMBOL] = "NEWCOIN"
    panel = pd.concat([panel, newcomer], ignore_index=True)

    encoded, columns, cold, vocabulary = add_onehot(panel, ~panel["is_live"])
    assert "feature_sym_NEWCOIN" not in columns
    assert vocabulary == 4 and cold == 1
    assert encoded.loc[encoded[SYMBOL] == "NEWCOIN", columns].to_numpy().sum() == 0


def test_onehot_respects_the_minimum_history_threshold():
    panel = make_panel(n_symbols=3, n_periods=ONEHOT_MIN_DATES - 1, n_live=1)
    _, columns, _, vocabulary = add_onehot(panel, ~panel["is_live"])
    assert vocabulary == 0 and columns == []


def test_rolling_adds_one_column_per_feature_and_window():
    panel = make_panel(n_periods=30)
    rolled, added = add_rolling(panel)
    assert len(added) == 8 * len(ROLL_WINDOWS)      # 4 indicators x 2 horizons
    assert not rolled[added].isna().any().any(), "min_periods=1 leaves no NaN"


def test_rolling_never_crosses_symbols():
    panel = make_panel(n_symbols=2, n_periods=10)
    rolled, added = add_rolling(panel)
    column = "feature_momentum_20d"
    for symbol in panel[SYMBOL].unique():
        block = rolled[rolled[SYMBOL] == symbol].sort_values(PERIOD)
        expected = block[column].rolling(5, min_periods=1).mean().to_numpy()
        np.testing.assert_allclose(block[column + "_rm5"].to_numpy(), expected, rtol=1e-5)


def test_build_features_leaves_no_missing_values():
    panel = make_panel()
    panel.loc[panel.index[:5], "feature_momentum_20d"] = np.nan
    built, columns, diagnostics = build_features(panel, "tenc", verbose=False)
    assert not built[columns].isna().any().any()
    assert diagnostics["n_features"] == len(columns)


def test_allfeat_pp_is_the_union_of_every_other_slot_plus_derived():
    """
    CURUPIRA_04's live feature set must be a strict superset of the others.

    This is the property that makes it "everything at once", and it is easy to
    break: the offline `allfeat` dataset (788) looked like a superset because it
    had MORE columns than CURUPIRA_03 (786), while actually lacking all 24
    rolling means. Counting columns does not establish containment.
    """
    panel = make_panel(n_symbols=3, n_periods=ONEHOT_MIN_DATES + 25, n_live=1)
    sets = {}
    for name in ("raw", "tenc", "onehot_roll", "allfeat_pp"):
        _, columns, _ = build_features(panel.copy(), name, verbose=False)
        sets[name] = set(columns)

    for name in ("raw", "tenc", "onehot_roll"):
        missing = sets[name] - sets["allfeat_pp"]
        assert not missing, "allfeat_pp lacks %d columns of %s" % (len(missing), name)

    # And it adds exactly the derived block on top of that union -- no more, no
    # less. The expectation is derived from add_derived on the same panel rather
    # than hardcoded, because the synthetic fixture carries 4 indicators where
    # the real data carries 11, so the absolute count is a property of the
    # fixture (8 here, 25 live) while the containment is the real invariant.
    union = sets["raw"] | sets["tenc"] | sets["onehot_roll"]
    _, derived = add_derived(panel.copy())
    assert sets["allfeat_pp"] - union == set(derived)
    assert all(c.endswith("_spread") or "_x_" in c or "_accel_" in c
               for c in derived)


def test_derived_columns_need_no_history():
    """
    Spreads and interactions must be same-row transforms.

    They are what lets CURUPIRA_04 avoid the rolling block's dependence on
    contiguous history, so a row's derived values must not change when its
    neighbours are removed.
    """
    panel = make_panel(n_symbols=4, n_periods=40, n_live=1)
    full, columns = add_derived(panel.copy())
    derived = [c for c in columns]

    # Recompute on a shuffled, decimated panel: same rows, no neighbours.
    subset = panel.iloc[::3].copy().sample(frac=1.0, random_state=0)
    partial, _ = add_derived(subset)

    common = full.loc[partial.index, derived]
    np.testing.assert_allclose(
        common.to_numpy(), partial[derived].to_numpy(), rtol=1e-6,
        err_msg="a derived column changed when neighbouring rows were dropped")


# --------------------------------------------------------------------------
# Ablation report sign convention
# --------------------------------------------------------------------------

def test_ablation_importance_sign_is_not_inverted(tmp_path, monkeypatch):
    """
    Removing a block that helps must report POSITIVE importance and a positive
    t, and be labelled as contributing.

    This is the single easiest thing in the analysis to get backwards, and it
    was: the table printed the importance's t while the verdict tested the
    delta's, so the one block that actually contributed was labelled "drop it".
    A sign error here does not crash or look wrong -- it just reverses every
    conclusion.
    """
    from src.analysis import ablation_report as ar

    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, len(dates))

    def write(directory, name, corr):
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame({"date": dates, "corr": corr, "bmc": 0.0,
                      "mmc": 0.0, "pnl": 0.0}).to_csv(
            os.path.join(directory, name + ".csv"), index=False)

    metrics_dir = str(tmp_path / "metrics")
    per_dir = str(tmp_path / "per_period")
    os.makedirs(metrics_dir, exist_ok=True)

    # Reference scores higher than the arm -> the removed block was helping.
    write(per_dir, "REF", 0.10 + noise)
    write(per_dir, "w6r6_lobo_useful", 0.05 + noise)
    for name in ("REF", "w6r6_lobo_useful"):
        with open(os.path.join(metrics_dir, name + ".json"), "w") as handle:
            handle.write('{"config": {"n_features": 10}, "n_periods": 200}')

    monkeypatch.setattr(ar, "METRICS", metrics_dir)
    monkeypatch.setattr(ar, "PER_PERIOD", per_dir)
    monkeypatch.setitem(ar.FAMILIES, "lobo",
                        {"prefix": "w6r6_lobo_", "reference": "REF"})

    rows = ar.report("lobo", verbose=False)
    row = rows[0]
    assert row["importance"] > 0, "removing a helpful block must cost, not gain"
    assert row["t"] > 0, "the importance t must share the importance's sign"
    assert row["t"] > 2, "a 0.05 gap on 200 periods should be unambiguous"


# --------------------------------------------------------------------------
# Blend slots
# --------------------------------------------------------------------------

def test_rank_blend_weights_are_renormalised_and_order_preserving():
    from src.live.blend import RankBlend

    class Const(object):
        def __init__(self, v): self.v = np.asarray(v, dtype="float64")
        def predict(self, X): return self.v[:len(X)]

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    # Weights that do not sum to 1 must be renormalised, not taken literally.
    blend = RankBlend([("m1", Const([1, 2, 3]), ["a"], 2.0),
                       ("m2", Const([1, 2, 3]), ["b"], 2.0)])
    out = blend.predict(X)
    assert np.all(np.diff(out) > 0), "two agreeing members must preserve order"
    assert out.max() <= 1.0 + 1e-9

    # A member pulling the other way must move the blend toward a tie.
    opposed = RankBlend([("m1", Const([1, 2, 3]), ["a"], 0.5),
                         ("m2", Const([3, 2, 1]), ["b"], 0.5)])
    assert np.allclose(opposed.predict(X), opposed.predict(X)[0])


def test_rank_blend_rejects_a_member_missing_its_columns():
    from src.live.blend import RankBlend

    class Const(object):
        def predict(self, X): return np.arange(len(X), dtype="float64")

    blend = RankBlend([("m", Const(), ["absent"], 1.0)])
    with pytest.raises(ValueError, match="missing"):
        blend.predict(pd.DataFrame({"present": [1.0, 2.0]}))


def test_blend_member_patterns_select_the_widths_the_backtest_fitted():
    """
    Each member's include pattern must pick exactly the column count its
    backtest run used.

    The two raw groups are the trap: they were fitted on the 22-column dataset,
    where '^feature_market_cap_(avg|ewa)_' matches 4 columns. Against the
    812-column live build the same pattern also catches
    feature_market_cap_avg_spread, giving 6. The registry anchors them for that
    reason, and this test is what stops the anchor being "tidied" away.
    """
    from src.prediction.functions import filter_features

    spec = registry.get("CURUPIRA_05")
    assert registry.is_blend(spec)

    # Stand-in for the real 812-column set: the raw pairs plus the derived
    # columns whose names would collide with an unanchored pattern.
    columns = ([f"feature_{n}_{h}" for n in ("market_cap_avg", "market_cap_ewa",
                                             "close_avg", "close_ewa")
                for h in ("20d", "60d")]
               + ["feature_market_cap_avg_spread", "feature_close_avg_spread",
                  "feature_sym_tenc", "feature_sym_BTC", "feature_sym_ETH",
                  "feature_momentum_x_volatility_20d", "feature_close_accel_20d"])

    widths = {m["name"]: len(filter_features(columns, m["include"], None, verbose=False))
              for m in spec["members"]}
    assert widths["size"] == 4, "size caught the spread column: %d" % widths["size"]
    assert widths["price"] == 4, "price caught the spread column: %d" % widths["price"]
    assert widths["tenc"] == 1
    assert widths["onehot"] == 2, "one-hot must exclude feature_sym_tenc"
    assert widths["interact"] == 2

    assert abs(sum(m["weight"] for m in spec["members"]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# staking
# ---------------------------------------------------------------------------

def test_stake_weights_match_the_solver_they_claim_to_come_from():
    """
    The hardcoded WEIGHTS must equal what --solve produces today.

    They are a cache of a solve over the backtested series, and the whole file
    is written as though they are. If a member run is ever re-scored the cache
    goes stale silently -- the plan still prints, still looks solved, and
    allocates real NMR on numbers nobody re-derived. This is that alarm.
    """
    from src.live import stake

    assert set(stake.WEIGHTS) == set(stake.STAKED)
    solved = stake.solve(stake.SCHEME)
    for name, weight in stake.WEIGHTS.items():
        assert abs(float(solved[name]) - float(weight)) < 5e-4, (
            "%s: stored %s, solver now says %s -- re-run "
            "`python3 -m src.live.stake --solve --scheme %s`"
            % (name, weight, solved[name], stake.SCHEME))


def test_staked_slots_are_actually_submitted():
    """Staking a slot that never uploads pays fees on nothing."""
    from src.live import stake

    for name in stake.STAKED:
        assert name in registry.ORDER, "%s is not in the submission order" % name
        assert name not in registry.NOT_SUBMITTED, "%s is never uploaded" % name
        assert registry.per_period_path(name) is not None, (
            "%s has no scored run, so no scheme can weight it" % name)


def test_mmc_budget_beats_equal_weight_on_mmc_and_respects_correlation():
    """
    The two properties the scheme is chosen for, asserted separately.

    Tilting toward MMC must actually raise the book's MMC, and the tilt must run
    through the covariance rather than straight off the means -- otherwise the
    'while also taking correlation into account' half of the design is decoration.
    """
    from src.live import stake

    names = stake.STAKED
    payout = stake.payout_series(names)
    mmc = stake.mmc_series(names)[payout.columns]
    means = mmc.mean().to_numpy()

    equal = {n: 1.0 / len(names) for n in payout.columns}
    tilted = {n: float(w) for n, w in stake.solve("mmc").items()}
    assert (stake.describe_weights(tilted)["mmc"]
            > stake.describe_weights(equal)["mmc"])

    # Risk share, not weight, is what the budget targets: it must be ordered the
    # same way as mean MMC.
    share = stake.describe_weights(tilted)["risk_share"]
    order_by_mmc = [n for _, n in sorted(zip(means, payout.columns))]
    assert [n for n in sorted(share, key=share.get)] == order_by_mmc

    # And the weights must differ from the naive mean-proportional answer,
    # because the covariance is not a multiple of the identity.
    naive = means / means.sum()
    assert max(abs(tilted[n] - w) for n, w in zip(payout.columns, naive)) > 0.01


# --------------------------------------------------------------------------- #
# fenccausal: the causal per-symbol feature encoding in the live path
# --------------------------------------------------------------------------- #

def _fenc_panel(n_dates=40, n_symbols=12, live_dates=1, seed=0):
    """A live-shaped panel: history plus a trailing live date."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rows = []
    for s in range(n_symbols):
        for d in dates:
            row = {"symbol": "SYM%02d" % s, "date": d}
            for f in ("feature_a", "feature_b"):
                row[f] = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel["is_live"] = panel["date"].isin(dates[-live_dates:])
    return panel


def test_fenccausal_is_registered_and_global_is_not():
    """
    `global` mode is lookahead by construction -- build_feature_encoding calls it
    "a diagnostic upper bound, not a candidate". It must never be live-buildable.
    """
    from src.live.features import FEATURE_SETS
    assert "fenccausal" in FEATURE_SETS
    assert "fencglobal" not in FEATURE_SETS


def test_fenccausal_never_lets_live_rows_move_a_historical_encoding():
    """
    REGRESSION TEST. The shrinkage prior defaulted to the mean over the whole
    frame, live rows included, so today's features shifted every historical row's
    encoding -- measured at 8.3e-3 before the fix. The live path now computes the
    prior from non-live rows only.
    """
    import numpy as np
    from src.live.features import add_fenccausal

    panel = _fenc_panel()
    built, added = add_fenccausal(panel)
    column = added[0]
    raw = column.replace("_symmean", "")

    for extreme in (999.0, 1.0, 0.0):
        tampered = panel.copy()
        tampered.loc[tampered["is_live"], raw] = extreme
        after, _ = add_fenccausal(tampered)

        joined = built[["symbol", "date", column]].merge(
            after[["symbol", "date", column]], on=["symbol", "date"],
            suffixes=("_before", "_after"))
        historical = joined[~joined["date"].isin(
            panel.loc[panel["is_live"], "date"].unique())]
        shift = float(np.nanmax(np.abs(
            historical[column + "_before"] - historical[column + "_after"])))
        assert shift == 0.0, (
            "live value %s moved a historical encoding by %.3e" % (extreme, shift))


def test_fenccausal_row_never_contributes_to_its_own_encoding():
    """
    The defining property of a causal expanding mean.

    The prior is pinned here on purpose. It is a statistic over the training rows,
    so perturbing a training row legitimately moves it and would move every
    encoding with it -- that is fit-on-train, not lookahead. Holding it fixed
    isolates the property actually under test: the expanding mean itself must
    exclude the row's own value.
    """
    import numpy as np
    from src.data_mani.build_feature_encoding import causal_feature_means

    panel = _fenc_panel()
    base = ["feature_a", "feature_b"]
    prior = {"feature_a": 0.5, "feature_b": 0.5}

    before, _ = causal_feature_means(panel, base, "symbol", "date", 50.0, prior=prior)

    target = panel.index[(panel["symbol"] == "SYM03") & (~panel["is_live"])][10]
    tampered = panel.copy()
    tampered.loc[target, "feature_a"] = 999.0
    after, _ = causal_feature_means(tampered, base, "symbol", "date", 50.0, prior=prior)

    assert float(before.loc[target, "feature_a_symmean"]) == pytest.approx(
        float(after.loc[target, "feature_a_symmean"]), abs=1e-9), (
        "a row contributed to its own encoding")

    # And it MUST move strictly later rows of that same symbol -- otherwise the
    # expanding mean is not accumulating anything.
    later = panel.index[(panel["symbol"] == "SYM03") & (~panel["is_live"])][11]
    assert abs(float(before.loc[later, "feature_a_symmean"]) -
               float(after.loc[later, "feature_a_symmean"])) > 1e-6


def test_fenccausal_adds_one_column_per_raw_feature():
    from src.live.features import add_fenccausal
    panel = _fenc_panel()
    _, added = add_fenccausal(panel)
    assert sorted(added) == ["feature_a_symmean", "feature_b_symmean"]
