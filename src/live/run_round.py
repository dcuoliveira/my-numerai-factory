"""
Run one live round: refit what is due, predict, submit, email.

    python3 -m src.live.run_round --dry-run          # full rehearsal, no upload
    python3 -m src.live.run_round                    # for real
    python3 -m src.live.run_round --deadline-minutes 25 --force-refit

Designed to be the only command a cron entry needs.

------------------------------------------------------------------------------
The deadline is the design constraint
------------------------------------------------------------------------------

A submission that arrives after the round closes is worth exactly nothing, so
every decision here is subordinate to finishing on time.

Measured costs on this box (deepsix, 130-period window, 39,000 training rows):

    load panel + build features      ~25 s   once, shared by all slots
    refit CURUPIRA_02  (22 feat)     ~55 s
    refit CURUPIRA_01  (23 feat)     ~55 s
    refit CURUPIRA_03  (786 feat)    ~75 s
    predict + write + upload         ~2 s    per slot

so a worst case where all three refit is around 4 minutes. The default
60-minute budget is therefore enormous -- which is the point. The budget exists
for the bad day: a slow API, a machine under load, a refit that hits a pathological
alpha path. On that day the pipeline degrades in a defined order instead of
hanging.

Three mechanisms enforce it:

  1. Slots run cheapest-first (CURUPIRA_02, 01, 03), so if time runs out it runs
     out on the most expensive and least essential slot.
  2. A refit only starts if the time remaining covers its ESTIMATED cost (taken
     from the last recorded fit_seconds for that slot, x1.5 for headroom) plus a
     reserve big enough to predict and upload every slot still outstanding. A
     refit that would not finish in time is skipped and the cached model is used.
  3. Each slot uploads as soon as it is ready rather than batching at the end, so
     a failure in slot three cannot cost slot one its submission.

--safety-first inverts step 3 for a genuinely tight window: every slot submits
from cache immediately, and only then does anything refit and resubmit. It costs
one extra upload per slot and guarantees that a submission is in the door within
seconds of starting.
"""

import argparse
import json
import os
import time

import pandas as pd

from src.live import registry, submit
from src.live.env import get_api, load_env
from src.live.features import (LIVE_DIR, PERIOD, SYMBOL, append_live_history,
                               build_features, load_panel)
from src.live.notify import send
from src.live.refit import (feature_signature, load_cached, load_slot_blend,
                            refit, refit_blend, refit_due)

# Seconds to reserve per outstanding slot for predict + write + upload, including
# the retry ladder in submit.upload (5 attempts, ~75s worst case).
RESERVE_PER_SLOT = 90

# Dataset downloads hit an S3 accelerate host that fails independently of the
# GraphQL API. 4 attempts over ~35s.
DOWNLOAD_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 5


class StaleLiveData(Exception):
    """Raised when live.parquet could not be refreshed but an old copy exists."""

# Used only until a slot has been fit once and has a real measurement.
DEFAULT_FIT_SECONDS = {"CURUPIRA_02": 60, "CURUPIRA_01": 60,
                       "CURUPIRA_03": 90, "CURUPIRA_04": 95,
                       "CURUPIRA_05": 440,
                       "CURUPIRA_06": 1}

REPORT_DIR = "results/live"


def download_live(api, verbose=True):
    """
    Fetch this round's live file and example predictions, with retries.

    The datasets come from an S3 accelerate endpoint, which is a different host
    from the GraphQL API and fails independently of it. A DNS blip there cost a
    whole round on 2026-08-14: the API authenticated fine, handed back a signed
    URL, and the download then failed to resolve the hostname.

    live.parquet is downloaded to a temporary path and only moved into place once
    complete. Deleting the old file first and downloading over it -- the previous
    approach -- means a mid-download failure leaves no live file at all, so the
    retry has nothing to fall back on either.
    """
    os.makedirs(LIVE_DIR, exist_ok=True)
    for remote in ("crypto/v2.0/live.parquet", "crypto/v2.0/live_example_preds.csv"):
        name = os.path.basename(remote)
        local = os.path.join(LIVE_DIR, name)
        # The example predictions rarely change and are only ever a last-resort
        # fallback, so an existing copy is good enough.
        if name != "live.parquet" and os.path.exists(local):
            continue

        staging = local + ".partial"
        last_error = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                if os.path.exists(staging):
                    os.remove(staging)
                api.download_dataset(remote, staging)
                os.replace(staging, local)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt == DOWNLOAD_ATTEMPTS:
                    break
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                if verbose:
                    print("   %s download attempt %d/%d failed (%s), retrying in %ds"
                          % (name, attempt, DOWNLOAD_ATTEMPTS, error, delay))
                time.sleep(delay)

        if last_error is not None:
            if name == "live.parquet" and os.path.exists(local):
                # A stale live file is normally the most dangerous thing here,
                # so this is not silent: the caller warns, and load_panel prints
                # the live date so a stale round is visible in the report.
                raise StaleLiveData(
                    "could not download live.parquet after %d attempts (%s); "
                    "a previous round's file is still on disk"
                    % (DOWNLOAD_ATTEMPTS, last_error))
            raise
    if verbose:
        print("live data ready in %s" % LIVE_DIR)


def round_window(api, round_number):
    """
    (is_open, open_time, deadline, minutes_left) for a round.

    THE DEADLINE IS closeStakingTime, NOT closeTime.

    This cost two late submissions before it was understood. A crypto round
    carries four timestamps and only one of them is the submission cutoff:

        openTime          12:00 UTC daily
        closeStakingTime  ~13:05 UTC -- SUBMISSIONS CLOSE HERE, ~66 min later
        closeTime         12:00 UTC the NEXT day, when the round stops being
                          "current" and the next one opens. Reading this as the
                          deadline suggests a 24-hour window that does not exist.
        scoreTime         ~6 days later, when the 20-day target resolves enough
                          to score

    Weekend rounds (isDaily=False) are the exception: round 1329 opened Sat
    12:00Z with a closeStakingTime of Sun 16:00Z, a 28-hour window. So the
    margin cannot be assumed either -- it is read per round.

    numerapi's check_round_open() is no help here: it reports False for rounds
    that are demonstrably open, answering for a different tournament's schedule.
    """
    try:
        rounds = api.raw_query(
            "query { rounds(tournament:12) { number openTime closeStakingTime } }",
            authorization=True)["data"]["rounds"]
    except Exception:
        return None, None, None, None

    for entry in rounds:
        if entry["number"] != round_number:
            continue
        opened = pd.Timestamp(entry["openTime"]).tz_localize(None)
        if not entry.get("closeStakingTime"):
            return None, opened, None, None
        deadline = pd.Timestamp(entry["closeStakingTime"]).tz_localize(None)
        now = pd.Timestamp.now("UTC").tz_localize(None)
        minutes_left = (deadline - now).total_seconds() / 60.0
        return (opened <= now < deadline), opened, deadline, minutes_left
    return None, None, None, None


def resolve_model_ids(api, names):
    """
    Map slot names to Numerai model ids.

    Slots must already exist on the account -- they are created in the Numerai
    UI, not through the API. A missing slot is reported rather than raised so
    the other slots still submit.
    """
    available = {k.upper(): v for k, v in (api.get_models() or {}).items()}
    resolved, missing = {}, []
    for name in names:
        if name.upper() in available:
            resolved[name] = available[name.upper()]
        else:
            missing.append(name)
    return resolved, missing, sorted(available)


def prepare_slot(name, panel, live_date, budget, warnings, force_refit=False):
    """
    Build features and load or refit the estimator for one slot.

    Returns (estimator, columns, built_panel, metadata, info) where info records
    what happened for the report. Never raises: a slot that cannot be prepared
    returns estimator=None and the fallback chain takes over.
    """
    spec = registry.get(name)
    info = {"refit": False, "detail": "", "note": None, "diagnostics": None}

    try:
        built, columns, diagnostics = build_features(
            panel.copy(), spec["feature_set"], verbose=False)
        info["diagnostics"] = diagnostics
    except Exception as error:
        info["note"] = "feature build failed: %s" % error
        return None, None, None, None, info

    # A slot blend reuses other slots' fits and must never refit. It also has to
    # run AFTER its members in ORDER, which the cheapest-first ordering already
    # gives since it costs nothing.
    if registry.is_slot_blend(spec):
        estimator, metadata = load_slot_blend(name, panel)
        if estimator is None:
            info["note"] = "members unavailable (%s)" % ", ".join(
                m["name"] for m in spec["slot_members"])
            return None, None, None, None, info
        info["detail"] = "reuses %s" % "+".join(
            m["name"].replace("CURUPIRA_", "C") for m in spec["slot_members"])
        return estimator, columns, built, metadata, info

    estimator, metadata = load_cached(name, columns)
    signature = feature_signature(columns)
    due, reason = refit_due(metadata, live_date, spec["fit"]["refit_every"], signature)
    if force_refit and not due:
        due, reason = True, "forced by --force-refit"

    if due:
        estimate = (metadata or {}).get("fit_seconds") or DEFAULT_FIT_SECONDS.get(name, 90)
        needed = estimate * 1.5 + budget.reserve()
        if budget.remaining() < needed:
            info["note"] = ("refit due (%s) but skipped: needs ~%.0fs, only %.0fs "
                            "left in the budget" % (reason, needed, budget.remaining()))
            warnings.append("%s ran on a stale fit -- %s" % (name, info["note"]))
            if estimator is None:
                info["note"] += "; and there is no cached model to fall back on"
                return None, None, None, None, info
            info["detail"] = "stale cache"
        else:
            try:
                if registry.is_blend(spec):
                    metadata = refit_blend(name, built, columns, live_date, verbose=True)
                else:
                    metadata = refit(name, built, columns, live_date, verbose=True)
                estimator, _ = load_cached(name, columns)
                info["refit"] = True
                info["detail"] = ("%d members" % len(metadata["members"])
                                  if "members" in metadata else
                                  "alpha=%.4g" % metadata["best_params"].get("alpha", float("nan")))
            except Exception as error:
                info["note"] = "refit failed: %s" % error
                if estimator is None:
                    return None, None, None, None, info
                warnings.append("%s refit failed, using the cached fit" % name)
                info["detail"] = "cache after failed refit"
    else:
        info["detail"] = reason

    if estimator is None:
        info["note"] = info["note"] or "no estimator available"
        return None, None, None, None, info

    return estimator, columns, built, metadata, info


class Budget(object):
    """Wall-clock budget for the round, in seconds."""

    def __init__(self, minutes, n_slots):
        self.total = minutes * 60.0
        self.started = time.time()
        self.outstanding = n_slots

    def elapsed(self):
        return time.time() - self.started

    def remaining(self):
        return max(0.0, self.total - self.elapsed())

    def reserve(self):
        """Time that must stay available to finish every outstanding slot."""
        return RESERVE_PER_SLOT * self.outstanding


def already_submitted(round_number, names):
    """
    Slots that already have a non-fallback submission recorded for this round.

    The submission window is only ~66 minutes, so the schedule runs several
    times inside it to survive a transient failure. Without this check every
    retry would re-upload work that already landed; with it, a retry after a
    clean run is a no-op costing one API call.
    """
    path = os.path.join(REPORT_DIR, str(round_number), "round_report.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as handle:
            report = json.load(handle)
    except Exception:
        return set()
    # A dry run writes the same report file with status "submitted" for every
    # slot, having uploaded nothing. Trusting it would make a later retry skip
    # real work and leave the round empty -- so a rehearsal never counts as a
    # submission.
    if report.get("dry_run"):
        return set()
    return {slot["name"] for slot in report.get("slots", [])
            if slot.get("status") == "submitted"
            and slot.get("source") == "model"
            and slot["name"] in names}


def run(deadline_minutes=60, dry_run=False, safety_first=False,
        force_refit=False, models=None, email=True, skip_if_submitted=False):
    load_env()
    api = get_api()

    names = models or registry.ORDER
    budget = Budget(deadline_minutes, len(names))
    warnings = []
    report_late = False
    started_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 78)
    print("live round | deadline %d min | %s" % (
        deadline_minutes, "DRY RUN" if dry_run else "LIVE SUBMIT"))
    print("=" * 78)

    round_number = api.get_current_round() or 0
    if skip_if_submitted:
        done = already_submitted(round_number, names)
        if done == set(names):
            print("round %s: all %d slots already submitted from their own models "
                  "-- nothing to do" % (round_number, len(names)))
            return {"round": round_number, "skipped": True, "slots": [],
                    "live_date": "-", "started_at": started_at,
                    "elapsed_minutes": budget.elapsed() / 60.0,
                    "deadline_minutes": deadline_minutes, "dry_run": dry_run,
                    "warnings": [], "diagnostics": {}}
        if done:
            names = [n for n in names if n not in done]
            budget.outstanding = len(names)
            print("round %s: %s already submitted, retrying %s"
                  % (round_number, ", ".join(sorted(done)), ", ".join(names)))

    try:
        download_live(api, verbose=True)
    except StaleLiveData as error:
        warnings.append(str(error))
        print("   WARNING: %s" % error)
    is_open, opened, deadline, minutes_left = round_window(api, round_number)
    if is_open is None:
        warnings.append("could not read the round window; proceeding anyway")
    elif not is_open:
        # Numerai accepts a late submission but flags it Late, and a late
        # submission does not score. Say so loudly rather than reporting "OK".
        warnings.append(
            "round %s SUBMISSIONS CLOSED at %s UTC (%.0f min ago) -- anything "
            "uploaded now is accepted but flagged Late and will not score"
            % (round_number, deadline, -minutes_left))
        report_late = True
    else:
        print("round %s: submissions close %s UTC (%.0f min left)"
              % (round_number, deadline, minutes_left))
        # The budget must never outlive the window it is submitting into.
        if minutes_left < deadline_minutes:
            deadline_minutes = max(2, int(minutes_left) - 2)
            budget.total = deadline_minutes * 60.0
            warnings.append("budget trimmed to %d min -- only %.0f min until the "
                            "submission deadline" % (deadline_minutes, minutes_left))
        elif minutes_left < 15:
            warnings.append("only %.0f min of margin left" % minutes_left)

    live_raw = pd.read_parquet(os.path.join(LIVE_DIR, "live.parquet"))
    append_live_history(live_raw)
    panel, live_date, gap_days = load_panel(verbose=True)
    live_symbols = panel.loc[panel["is_live"] & (panel[PERIOD] == live_date),
                             SYMBOL].tolist()

    if gap_days > 45:
        warnings.append("train.parquet lags live by %d days; CURUPIRA_03's rolling "
                        "features are computed across that gap" % gap_days)

    model_ids, missing, available = resolve_model_ids(api, names)
    if missing:
        warnings.append(
            "these slots do not exist on the account and cannot be submitted: %s. "
            "Create them at numer.ai (Crypto tournament) with exactly these names. "
            "Slots currently on the account: %s"
            % (", ".join(missing), ", ".join(available) or "(none)"))

    context = {"estimators": {}, "live_date": live_date,
               "round_number": round_number, "live_symbols": live_symbols}
    slots, diagnostics = [], {}

    # --- optional guard pass: get something in the door immediately -----------
    if safety_first and not dry_run:
        print("\n--- safety-first pass: submitting from cache ---")
        for name in names:
            estimator, metadata = load_cached(name, columns)
            if estimator is None or name not in model_ids:
                continue
            try:
                built, columns, _ = build_features(
                    panel.copy(), registry.get(name)["feature_set"], verbose=False)
                frame = submit.predict(estimator, built, columns, live_date)
                path = submit.write_csv(frame, round_number, name)
                submit.upload(api, path, model_ids[name], name)
                print("   %s: guard submission in" % name)
            except Exception as error:
                print("   %s: guard submission failed (%s)" % (name, error))

    # --- main pass -----------------------------------------------------------
    for name in names:
        slot_started = time.time()
        print("\n--- %s (%s, %s) | %.0fs left ---" % (
            name, registry.get(name)["wrapper"],
            registry.get(name)["feature_set"], budget.remaining()))

        # --force-refit asks for a fresh fit; it must NOT delete the cached
        # metadata to get one. If the budget then refuses the refit, that cache
        # is the only thing standing between this slot and a stand-in.
        estimator, columns, built, metadata, info = prepare_slot(
            name, panel, live_date, budget, warnings, force_refit=force_refit)
        if info["diagnostics"]:
            diagnostics[name] = info["diagnostics"]
        if estimator is not None:
            context["estimators"][name] = (estimator, columns, built, metadata)

        record = {"name": name, "refit": info["refit"], "detail": info["detail"],
                  "note": info["note"]}

        try:
            frame, source, note = submit.build_predictions(name, context)
            path = submit.write_csv(frame, round_number, name)
            record["source"] = source
            if note:
                record["note"] = "; ".join(filter(None, [record["note"], note]))

            if name not in model_ids:
                record["status"] = "failed"
                record["error"] = ("no such model slot on the account; CSV written "
                                   "to %s but not uploaded" % path)
            else:
                submission_id, attempts = submit.upload(
                    api, path, model_ids[name], name, dry_run=dry_run)
                record["status"] = "submitted"
                record["detail"] = "%s | %s | %d rows%s" % (
                    record["detail"] or "-", os.path.basename(path), len(frame),
                    "" if attempts <= 1 else " | %d upload attempts" % attempts)
                print("   submitted from '%s' -> %s" % (source, submission_id))
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["source"] = "-"
            print("   FAILED: %s" % error)

        record["seconds"] = round(time.time() - slot_started, 1)
        slots.append(record)
        budget.outstanding -= 1

    # A retry only ran the outstanding slots; merge in what earlier runs did so
    # the report and the email describe the whole round, not just this pass.
    if skip_if_submitted:
        previous = os.path.join(REPORT_DIR, str(round_number), "round_report.json")
        if os.path.exists(previous):
            try:
                with open(previous) as handle:
                    earlier = json.load(handle)
                have = {s["name"] for s in slots}
                slots = slots + [s for s in earlier.get("slots", [])
                                 if s["name"] not in have]
                slots.sort(key=lambda s: registry.ORDER.index(s["name"])
                           if s["name"] in registry.ORDER else 99)
            except Exception:
                pass

    report = {
        "round": round_number, "live_date": str(live_date.date()),
        "started_at": started_at, "elapsed_minutes": budget.elapsed() / 60.0,
        "deadline_minutes": deadline_minutes, "dry_run": dry_run,
        "slots": slots, "diagnostics": diagnostics, "warnings": warnings,
        "train_live_gap_days": gap_days, "submitted_late": report_late,
    }

    os.makedirs(os.path.join(REPORT_DIR, str(round_number)), exist_ok=True)
    report_path = os.path.join(REPORT_DIR, str(round_number), "round_report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print("\n" + "=" * 78)
    for slot in slots:
        print("%-13s %-9s %-14s %6.1fs  %s" % (
            slot["name"], slot["status"], slot.get("source", "-"),
            slot["seconds"], slot.get("error") or slot.get("detail") or ""))
    print("finished in %.1f min -> %s" % (budget.elapsed() / 60.0, report_path))
    print("=" * 78)

    if email:
        send(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run one live Numerai Crypto round.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--deadline-minutes", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true",
                        help="do everything except upload")
    parser.add_argument("--safety-first", action="store_true",
                        help="submit from cache before refitting anything")
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--model", action="append", dest="models", default=None,
                        help="restrict to one slot; repeatable")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--skip-if-submitted", action="store_true",
                        help="no-op for slots that already submitted this round; "
                             "use on the retry runs inside the submission window")
    args = parser.parse_args()

    try:
        report = run(deadline_minutes=args.deadline_minutes, dry_run=args.dry_run,
                     safety_first=args.safety_first, force_refit=args.force_refit,
                     models=args.models, email=not args.no_email,
                     skip_if_submitted=args.skip_if_submitted)
    except Exception as error:
        # Anything that escapes run() -- a DNS failure on the dataset host, a
        # corrupt parquet, an API outage -- happens BEFORE the per-slot handling
        # and so produces no round report and, without this, no email either.
        # A silent miss is the worst outcome available: you find out days later
        # from a Late flag. This is exactly how round 1333 was lost.
        import traceback
        traceback.print_exc()
        if not args.no_email:
            send({"round": "?", "live_date": "-", "started_at": "-",
                  "elapsed_minutes": 0.0, "deadline_minutes": args.deadline_minutes,
                  "dry_run": args.dry_run, "slots": [], "diagnostics": {},
                  "crash": "%s: %s" % (type(error).__name__, error),
                  "warnings": ["the round aborted before any slot was attempted; "
                               "NOTHING WAS SUBMITTED"]})
        raise SystemExit(2)

    if report.get("skipped"):
        raise SystemExit(0)
    failed = [s for s in report["slots"] if s["status"] != "submitted"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
