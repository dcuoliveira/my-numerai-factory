#!/usr/bin/env python3
"""
E202 orchestrator: 60-day state extraction -> 60-day ridge fits -> score -> export/residualise -> ensembles -> reports.

    systemd-run --user --unit=nmr-e202-orchestrate -p WorkingDirectory=<repo> \
        /home/danielco/miniconda3/bin/python3 -u experiments/e202_orchestrate.py

Every LOOP seconds: a model whose 34 part files exist and whose T60 preds do not gets its fit submitted
through `src.cluster run` (a running unit's name refuses a duplicate). Extraction itself runs in the
nmr-e202-gpu-* units (experiments/e202_gpu_chain.sh) and the nmr-e202_cpu_* claim workers. When the four
T60 fits and B_ALL60 exist: --score, --export, --ensembles, --report, html_pdf, the repo-standard
report_pdf over the 16 exported runs, then exit.
"""
import glob
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = "/home/danielco/miniconda3/bin/python3"
LOOP = 600
MODELS = ["timestral", "tabpfnts", "toto", "t0"]
STATES = "/scratch/danielco/tsfmstates60"
ROOT = "results/reports/crypto/e202_tsfm_t60"
N_PARTS = 34


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def parts(m):
    return len(glob.glob(os.path.join(STATES, m, "part_*.npz")))


def have(arm):
    return os.path.exists(os.path.join(ROOT, "preds", arm + ".parquet"))


def main():
    while True:
        status, todo = [], []
        for m in MODELS:
            p = parts(m)
            status.append("%s %d/%d%s" % (m, p, N_PARTS, " fitted" if have("T60_" + m) else ""))
            if p >= N_PARTS and not have("T60_" + m):
                todo.append("cores=4 ram=28 name=e202_fit_%s -- %s -u experiments/e202_tsfm_t60.py --fit --arms T60_%s" % (m, PY, m))
        log(" | ".join(status))
        if todo:
            jobs = "logs/tsfm/e202_fit_todo.jobs"
            open(jobs, "w").write("# E202 60-day ridge fits (written by e202_orchestrate.py)\n" + "\n".join(todo) + "\n")
            rc, out = sh("%s -m src.cluster run --jobs %s --only hamilton hopper lovelace dostoievski tolstoi deepfour"
                         % (PY, jobs), timeout=900)
            started = [l.strip() for l in out.splitlines() if l.strip().startswith("ok")]
            if started:
                log("cluster", started)
        if all(have("T60_" + m) for m in MODELS) and have("B_ALL60"):
            for step, t in (("--score --workers 4", 3600), ("--export", 7200), ("--ensembles --workers 4", 7200), ("--report", 3600)):
                rc, out = sh("%s experiments/e202_tsfm_t60.py %s" % (PY, step), timeout=t)
                log(step, "rc", rc, out[-1500:])
                if rc:
                    return
            rc, out = sh("%s -m src.analysis.html_pdf results/reports/crypto/e202_tsfm_t60.html" % PY, timeout=1800)
            log("pdf", rc, out[-300:])
            rc, out = sh("%s -m src.analysis.report_pdf --model tsfm_states --runs 'e202_*' --sort-by payout "
                         "--title 'E202 TSFM 20d/60d plain and residualised vs the book' --out results/reports/crypto/e202_book.pdf"
                         % PY, timeout=3600)
            log("report_pdf", rc, out[-2500:])
            return
        time.sleep(LOOP)


if __name__ == "__main__":
    main()
