"""E159 chain: fits -> score -> audit -> report -> PDF, with email at every milestone (src.live.notify).

Runs as its own 1-core cluster unit (nmr-e159_orch). Fit units are launched separately from
experiments/e159_fit.jobs; this waits for them, launches scoring from e159_score.jobs, waits again,
then renders. Any failed e159 unit stops the chain and mails which one.
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from src.cluster import manager as mgr  # noqa: E402
from src.live.notify import send_text  # noqa: E402

PY = "/home/danielco/miniconda3/bin/python3"
ROOT = "results/reports/crypto/e159_volumealpha"
TAG = "[crypto] E159 volumealpha"


def mail(subject, body):
    try:
        send_text("%s %s" % (TAG, subject), body)
    except Exception as e:  # never let mail stop the chain
        print("mail failed:", e, flush=True)


def units(prefix):
    rows = []
    for host, rs in mgr.status().items():
        rows += [(host, r) for r in rs if r["unit"].startswith("nmr-" + prefix) and not r["unit"].startswith("nmr-e159_orch")]
    return rows


def wait(prefix, what):
    t0 = time.time()
    seen = False
    while True:
        rows = units(prefix)
        seen = seen or bool(rows)
        bad = [(h, r["unit"]) for h, r in rows if r["state"].startswith("failed")]
        live = [(h, r["unit"]) for h, r in rows if r["state"].startswith(("active", "activating"))]
        if bad:
            tails = []
            for h, u in bad:
                log = "logs/cluster/%s.out" % u[4:].replace(".service", "")
                tail = subprocess.run(["tail", "-n", "15", log], capture_output=True, text=True).stdout if os.path.exists(log) else "(no log)"
                tails.append("--- %s on %s\n%s" % (u, h, tail))
            mail("FAILED during %s" % what, "Failed units:\n" + "\n".join(tails) + "\nChain stopped; nothing further runs.")
            sys.exit(1)
        if seen and not live:
            print("%s done after %.1f h" % (what, (time.time() - t0) / 3600), flush=True)
            return
        if not seen and time.time() - t0 > 1800:
            mail("FAILED: no %s units found" % what, "No nmr-%s* unit appeared within 30 min." % prefix)
            sys.exit(1)
        time.sleep(300)


def run(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if r.returncode:
        mail("FAILED: %s" % cmd.split()[1] if len(cmd.split()) > 1 else cmd, "Command:\n%s\n\nstderr tail:\n%s" % (cmd, r.stderr[-3000:]))
        sys.exit(1)
    return r.stdout


def headline():
    import pandas as pd
    R = pd.read_csv(os.path.join(ROOT, "routing.csv"))
    R = R[(R["rule"] == "R_SLOT")]
    lines = ["Routed (R_SLOT) arm minus v2.0, NW lag-20 t; MMC is the proxy:", ""]
    lines.append("%-8s %-10s %-5s %18s %18s" % ("model", "arm", "scope", "dCORR (t)", "dpayout (t)"))
    for _, r in R[R["arm"].isin(["FULL", "F_TECH", "F_FCST", "F_RAW", "F_MODEL"])].iterrows():
        lines.append("%-8s %-10s %-5s %10.4f (%+5.1f) %10.4f (%+5.1f)" % (r["model"], r["arm"], r["scope"], r["corr_delta"], r["corr_t"], r["payout_delta"], r["payout_t"]))
    return "\n".join(lines)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "fits"
    if stage == "fits":
        mail("started", "E159: crypto/v2.0-volumealpha-causal (22 + 70 feature_va_*) vs v2.0, routed protocol of E149.\n"
             "Fits: experiments/e159_fit.jobs (ridge, lgb_ktz x5 seeds, mlp_ktz x5 seeds, retunes). Logs logs/cluster/e159_*.out.\n"
             "Next mail: fits done / scoring, then the verdict with the PDF path.")
        wait("e159_", "fits")
        out = run("%s -m src.cluster run --jobs experiments/e159_score.jobs" % PY)
        if "1 started, 0 failed, 0 not placed" not in out:
            mail("FAILED: scoring not placed", out[-3000:])
            sys.exit(1)
        mail("fits done, scoring", "All E159 fit units finished cleanly. Scoring launched (nmr-e159s_score).")
        time.sleep(120)
        stage = "score"
    if stage == "score":
        wait("e159s_", "scoring")
        stage = "report"
    if stage == "report":
        run("%s -c \"import sys; sys.path.insert(0,'experiments'); import e159_report_parts as X; X.assemble()\"" % PY)
        run("%s experiments/e138_report.py E159" % PY)
        htmls = sorted(f for f in os.listdir(ROOT) if f.startswith("E159") and f.endswith(".html"))
        if not htmls:
            mail("FAILED: no report HTML", "e138_report wrote no E159*.html into %s" % ROOT)
            sys.exit(1)
        html = os.path.join(ROOT, htmls[-1])
        run("%s -m src.analysis.html_pdf %s" % (PY, html))
        pdf = html[:-5] + ".pdf"
        mail("COMPLETE", "%s\n\nReport: %s\n(HTML beside it.) The verdict line in the PDF is the reading-free template;"
             " the written reading is added after the numbers are read." % (headline(), os.path.abspath(pdf)))


if __name__ == "__main__":
    main()
