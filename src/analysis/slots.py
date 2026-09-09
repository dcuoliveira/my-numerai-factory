"""
The deployed slots as scored series, with every gotcha resolved in one place.

    python3 -m src.analysis.slots

A slot's backtest series is not simply `results/per_period/<wrapper>/<backtest_run>`.
Four things get in the way, and each one has silently produced a wrong table before:

  * A slot blend has no wrapper, so the tree cannot be derived from it. The
    registry carries `scored_under` for exactly that -- and CURUPIRA_SACI_01..04
    were missing it until this module was written, so registry.per_period_path
    returned None for four slots whose CSVs were sitting on disk.

  * CURUPIRA_07's `backtest_run` is the string "(screened, not run as a single
    arm)". There is no such file and there never was. Its recorded 0.084782 is a
    payout-SERIES average -- a screen. The prediction-level blend of the same
    four slots exists as ridge_glmnet/xae_c07_base and pays 0.098594, 16% more.
    Both numbers are real; they answer different questions, so this module
    reports the run and names the difference rather than picking one quietly.

  * CURUPIRA_05_60's registry payout (0.074914) is its 60-day predictions
    RE-SCORED on the payable 20-day label, out of results/cache/. Its own stored
    series is scored on the 60-day label and pays 0.092201. Same slot, two
    scales, and neither is wrong -- but putting them in one column is.

  * A slot can be in the registry and not on the Numerai account. NOT_SUBMITTED
    now covers only CURUPIRA_05_60, the one slot fitted purely to be a component.
    SACI_03, SACI_04 and SACI_05 are dual-role -- submitted in their own right
    AND reused as members -- so they are cached like components and uploaded like
    models. The only current source for what exists on the account is the newest
    round report, so it is read rather than hardcoded.

Everything is summarised through results_table.row_from_frame, so a slot row and
an experiment row come out of the same code path.
"""

import collections
import glob
import json
import os

import pandas as pd

from src.analysis.results_table import payout_series, row_from_frame, target_of
from src.live import registry

#: Slots whose registry `backtest_run` does not name a real run, and the run that
#: reconstructs them. The note travels with the number wherever it is printed.
RECONSTRUCTED = {
    "CURUPIRA_07": (
        "ridge_glmnet", "xae_c07_base",
        "registry backtest_run is a SCREEN, not a run: it averages the four "
        "members' payout SERIES. This row is the prediction-level rank average "
        "of the same four slots, which is what the slot actually submits."),
}

#: How the round report's `source` field reads in a one-token column.
LIVE_TOKENS = {
    "model": "ok",
    "sibling": "sibling!",
    "previous_round": "prev!",
    "example": "example!",
    "not-submitted": "not-sub",
}

SlotSeries = collections.namedtuple("SlotSeries", [
    "name", "model", "run", "target", "payout", "row", "kind", "staked",
    "on_account", "live_status", "note", "registry_payout", "matches_registry",
])


def _staked():
    """
    The staked slot names.

    Imported rather than copied: the amounts drift every time stake is moved,
    and a hardcoded list goes stale without anything failing.

    :return: slot names carrying stake
    :rtype: set
    """
    try:
        from src.live.stake import STAKED
        return set(STAKED)
    except Exception:
        return set()


def latest_round_report(root="results/live"):
    """
    The newest round report, or None.

    :param root: directory holding one subdirectory per round
    :type root: str
    :return: (round_number, report) or (None, None)
    :rtype: (int, dict) or (None, None)
    """
    rounds = []
    for path in glob.glob(os.path.join(root, "*", "round_report.json")):
        name = os.path.basename(os.path.dirname(path))
        if name.isdigit():
            rounds.append((int(name), path))
    if not rounds:
        return None, None
    number, path = max(rounds)
    try:
        return number, json.load(open(path))
    except (ValueError, IOError):
        return None, None


def live_state(report=None):
    """
    Per-slot live state from a round report.

    Two things are read, because they answer different questions: what the slot
    submitted (its fallback source -- a slot can report "submitted" while having
    uploaded Numerai's own example predictions), and which slots the account
    holds at all.

    :param report: a round report, or None to read the newest
    :type report: dict or None
    :return: (status by slot, set of slots on the account or None if unknown)
    :rtype: (dict, set or None)
    """
    if report is None:
        _, report = latest_round_report()
    if not report:
        return {}, None

    status = {}
    for slot in report.get("slots") or []:
        if slot.get("status") == "failed":
            status[slot["name"]] = "failed!"
        else:
            status[slot["name"]] = LIVE_TOKENS.get(slot.get("source"), "ok")

    on_account = None
    for warning in report.get("warnings") or []:
        marker = "Slots currently on the account:"
        if marker in warning:
            tail = warning.split(marker, 1)[1]
            on_account = {n.strip() for n in tail.split(",") if n.strip()}
    return status, on_account


def slot_run(name):
    """
    The (model tree, run name) a slot's backtest series lives at.

    :param name: slot name
    :type name: str
    :return: (model, run, note) -- model is None when no series exists
    :rtype: (str or None, str or None, str or None)
    """
    if name in RECONSTRUCTED:
        model, run, note = RECONSTRUCTED[name]
        return model, run, note

    spec = registry.get(name)
    run = spec.get("backtest_run")
    model = spec.get("scored_under") or spec.get("wrapper")
    if not (model and run):
        return None, None, "no scored run: backtest_run is %r" % (run,)
    if not os.path.exists("results/per_period/crypto/%s/%s.csv" % (model, run)):
        return None, None, "%s/%s has no per-period file" % (model, run)
    return model, run, None


def slot_series(name, tournament="crypto"):
    """
    One slot as a scored series plus its row.

    :param name: slot name
    :type name: str
    :param tournament: e.g. "crypto"
    :type tournament: str
    :return: the resolved slot, or None if it has no series
    :rtype: SlotSeries or None
    """
    model, run, note = slot_run(name)
    if not model:
        return None

    spec = registry.get(name)
    frame = pd.read_csv("results/per_period/%s/%s/%s.csv" % (tournament, model, run))
    if "date" in frame.columns:
        frame = frame.set_index("date")
    payout = payout_series(frame).dropna()
    target = target_of(tournament, model, run)

    kind = ("slotblend" if registry.is_slot_blend(spec)
            else "blend" if registry.is_blend(spec) else "sub")

    recorded = (spec.get("backtest") or {}).get("payout")
    matches = recorded is not None and abs(float(payout.mean()) - recorded) < 1e-6
    if not matches and recorded is not None and note is None:
        note = ("registry records %.6f; this series pays %.6f. The registry "
                "figure is the 20-day RE-SCORE from results/cache/"
                "cross_target_payout.csv, this one is the stored %s series."
                % (recorded, payout.mean(), (target or "?").rsplit("_", 1)[-1] + "-day"))

    status, on_account = live_state()
    return SlotSeries(
        name=name, model=model, run=run, target=target, payout=payout,
        row=row_from_frame(name, frame, target=target),
        kind=kind, staked=name in _staked(),
        on_account=(name in on_account) if on_account is not None else None,
        live_status=status.get(name, "-"), note=note,
        registry_payout=recorded, matches_registry=matches)


def slot_rows(which="all", tournament="crypto"):
    """
    Every slot that has a series, with the ones that do not named separately.

    :param which: "all", "on-account", "staked" or "none"
    :type which: str
    :param tournament: e.g. "crypto"
    :type tournament: str
    :return: (resolved slots, [(name, why it was left out)])
    :rtype: ([SlotSeries], [(str, str)])
    """
    if which == "none":
        return [], []

    staked, (_, on_account) = _staked(), live_state()
    resolved, dropped = [], []
    for name in registry.ORDER:
        got = slot_series(name, tournament)
        if got is None:
            dropped.append((name, slot_run(name)[2]))
            continue
        if which == "staked" and name not in staked:
            continue
        if which == "on-account":
            if name in registry.NOT_SUBMITTED:
                dropped.append((name, "component by design, never uploaded"))
                continue
            if on_account is not None and name not in on_account:
                dropped.append((name, "not a slot on the Numerai account"))
                continue
        resolved.append(got)
    return resolved, dropped


def main():
    """
    Print the resolution of every slot -- the self-check.

    :return: process exit code
    :rtype: int
    """
    number, _ = latest_round_report()
    rows, dropped = slot_rows("all")
    print("%d slots resolved, live state from round %s\n"
          % (len(rows), number if number else "(none on disk)"))

    print("%-18s %-13s %-30s %4s %6s %10s %10s %-8s"
          % ("slot", "tree", "run", "tgt", "n", "payout", "registry", "live"))
    for s in rows:
        print("%-18s %-13s %-30s %4s %6d %10.6f %10s %-8s%s"
              % (s.name, s.model, s.run,
                 "t" + (s.target or "?").rsplit("_", 1)[-1], len(s.payout),
                 s.payout.mean(),
                 "match" if s.matches_registry else "%.6f" % s.registry_payout,
                 s.live_status, " $" if s.staked else ""))

    mismatched = [s for s in rows if not s.matches_registry]
    print("\n%d of %d reproduce the registry to 1e-6."
          % (len(rows) - len(mismatched), len(rows)))
    for s in mismatched:
        print("  %s: %s" % (s.name, s.note))
    for name, why in dropped:
        print("  %s: %s" % (name, why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
