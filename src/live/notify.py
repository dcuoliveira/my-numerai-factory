"""
Email the per-round status of every slot.

    python3 -m src.live.notify --test     # send a sample report to NOTIFY_TO

Two rules shape this module.

First, it sends on success as well as failure. A pipeline that only mails on
failure is indistinguishable from a pipeline that has silently stopped running,
and on a weekly cadence you would not find out for a month. Every round produces
a mail; the subject line carries the verdict so it can be triaged without
opening it.

Second, a mail failure must never break a submission. SMTP is the least reliable
thing in this pipeline and the least important -- if the report cannot be sent,
the round still succeeded. Everything here is wrapped so the caller gets a
boolean, never an exception.

Credentials come from the environment (see .env, which is gitignored). For Gmail
you need an App Password, not the account password: SMTP_PASSWORD is what
'Manage your Google Account -> Security -> App passwords' issues.
"""

import argparse
import os
import smtplib
import socket
from email.message import EmailMessage

# Set by src.live.env.load_env() before this module is used.
ENV = os.environ

SUBJECT_PREFIX = "[numerai-crypto]"


def _config():
    """SMTP settings, or None if the mailer is not configured."""
    host = ENV.get("SMTP_HOST")
    user = ENV.get("SMTP_USER")
    password = ENV.get("SMTP_PASSWORD")
    to = ENV.get("NOTIFY_TO") or user
    if not (host and user and password and to):
        return None
    return {"host": host, "port": int(ENV.get("SMTP_PORT", 587)),
            "user": user, "password": password, "to": to}


def render(report):
    """
    Turn a round report into (subject, body).

    `report` is the dict the orchestrator assembles: round number, live date,
    elapsed time, and one entry per slot.
    """
    slots = report["slots"]
    failed = [s for s in slots if s["status"] == "failed"]
    fell_back = [s for s in slots if s["status"] == "submitted"
                 and s["source"] != "model"]
    submitted = [s for s in slots if s["status"] == "submitted"]

    if report.get("crash"):
        verdict = "CRASHED -- NOTHING SUBMITTED"
    elif report.get("submitted_late"):
        verdict = "LATE -- submissions had already closed"
    elif failed:
        verdict = "FAILED (%d/%d slots)" % (len(failed), len(slots))
    elif fell_back:
        verdict = "OK with fallbacks (%d/%d)" % (len(fell_back), len(slots))
    else:
        verdict = "OK (%d/%d)" % (len(submitted), len(slots))

    subject = "%s round %s -- %s" % (SUBJECT_PREFIX, report["round"], verdict)

    lines = [
        "Round        : %s" % report["round"],
        "Live date    : %s" % report["live_date"],
        "Started      : %s" % report["started_at"],
        "Elapsed      : %.1f min of a %s min budget" % (
            report["elapsed_minutes"], report.get("deadline_minutes", "-")),
        "Host         : %s" % socket.gethostname(),
        "Mode         : %s" % ("DRY RUN -- nothing was uploaded"
                               if report.get("dry_run") else "live"),
        "",
    ]
    if report.get("crash"):
        lines += ["!" * 78,
                  "THE ROUND ABORTED BEFORE ANY SLOT WAS ATTEMPTED.",
                  "Nothing was submitted. The submission window is ~66 minutes;",
                  "if it has not passed, rerun now:",
                  "    python3 -m src.live.run_round",
                  "",
                  "  %s" % report["crash"],
                  "!" * 78, ""]
    lines += [
        "=" * 78,
        "%-13s %-9s %-14s %-8s %6s  %s" % (
            "SLOT", "STATUS", "SOURCE", "REFIT", "SEC", "DETAIL"),
        "=" * 78,
    ]
    for slot in slots:
        lines.append("%-13s %-9s %-14s %-8s %6.1f  %s" % (
            slot["name"], slot["status"], slot.get("source", "-"),
            "yes" if slot.get("refit") else "no",
            slot.get("seconds", 0.0), slot.get("detail", "") or ""))

    # Anything that fell back or failed gets its reason spelled out in full --
    # the table truncates, and the reason is the only actionable part.
    notable = [s for s in slots if s.get("note") or s["status"] == "failed"]
    if notable:
        lines += ["", "-" * 78, "WHY:", "-" * 78]
        for slot in notable:
            lines.append("%s: %s" % (slot["name"], slot.get("note") or slot.get("error")))

    if report.get("diagnostics"):
        lines += ["", "-" * 78, "FEATURE BUILD:", "-" * 78]
        for name, diagnostics in report["diagnostics"].items():
            lines.append("%-13s %s" % (name, diagnostics))

    lines += ["", "-" * 78, "WHAT THESE MODELS ARE:", "-" * 78]
    from src.live import registry
    lines.append(registry.describe())

    if report.get("warnings"):
        lines += ["", "-" * 78, "WARNINGS:", "-" * 78]
        lines += ["  - %s" % w for w in report["warnings"]]

    return subject, "\n".join(lines)


def send(report, verbose=True):
    """
    Send the round report. Returns True if it went out, False otherwise.

    Never raises: the submission has already happened by the time this runs, and
    a broken mailer must not turn a good round into a failed one.
    """
    subject, body = render(report)

    config = _config()
    if config is None:
        if verbose:
            print("   email not configured (set SMTP_* and NOTIFY_TO in .env) "
                  "-- report follows\n")
            print(body)
        return False

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = config["user"]
        message["To"] = config["to"]
        message.set_content(body)

        with smtplib.SMTP(config["host"], config["port"], timeout=30) as server:
            server.starttls()
            server.login(config["user"], config["password"])
            server.send_message(message)
        if verbose:
            print("   emailed %s: %s" % (config["to"], subject))
        return True
    except Exception as error:
        if verbose:
            print("   EMAIL FAILED (%s) -- the round itself was unaffected" % error)
            print(body)
        return False


def main():
    parser = argparse.ArgumentParser(description="Send a sample round report.")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    from src.live.env import load_env
    load_env()

    report = {
        "round": 9999, "live_date": "2026-08-12",
        "started_at": "2026-08-13 17:00:00", "elapsed_minutes": 3.2,
        "deadline_minutes": 60, "dry_run": True,
        "slots": [
            {"name": "CURUPIRA_02", "status": "submitted", "source": "model",
             "refit": True, "seconds": 55.4, "detail": "alpha=65.79"},
            {"name": "CURUPIRA_01", "status": "submitted", "source": "model",
             "refit": False, "seconds": 2.1, "detail": "cached fit, 12 days old"},
            {"name": "CURUPIRA_03", "status": "submitted", "source": "sibling",
             "refit": False, "seconds": 3.0, "detail": "stand-in",
             "note": "rolling feature build failed: history gap too large"},
        ],
        "diagnostics": {"CURUPIRA_03": {"live_cold_start": 73}},
        "warnings": ["sample report -- not a real round"],
    }
    if args.test:
        send(report)
    else:
        subject, body = render(report)
        print(subject); print(); print(body)


if __name__ == "__main__":
    main()
