#!/usr/bin/env python3
"""
E193-E197 orchestrator: chains state extraction -> setup -> ridge arms -> scoring -> tables, unattended.

    systemd-run --user --unit=nmr-e193-orchestrate -p WorkingDirectory=<repo> \
        /home/danielco/miniconda3/bin/python3 -u experiments/e193_orchestrate.py

Every LOOP seconds, for each model:
  states complete (35 part files)   -> --setup once, then `src.cluster run` its jobs file (re-run each loop:
                                       fit_arm skips finished arms, a running unit's name refuses a duplicate)
  all 38 preds present, no scores   -> --score (here, sequentially)
When every model is scored: --tables, then html_pdf, then exit. Also moves freed GPUs onto unfinished
extractions (Toto / TabPFN-TS claim workers) once TempoPFN, Timestral and t0 are done.
"""
import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
PY = "/home/danielco/miniconda3/bin/python3"
LOOP = 600
MODELS = {"tempopfn": "e195", "timestral": "e197", "t0": "e196", "toto": "e193", "tabpfnts": "e194"}
STATES = "/scratch/danielco/tsfmstates"
N_ARMS = 38


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-2000:]


def parts(m):
    return len(glob.glob(os.path.join(STATES, m, "part_*.npz")))


def root(m):
    return "results/reports/crypto/%s_%snorm" % (MODELS[m], m)


def n_preds(m):
    return len(glob.glob(os.path.join(root(m), "preds", "*.parquet")))


def gpu_launch(host, gpu, unit, model, venv, extra=""):
    rc, out = sh("ssh %s 'systemctl --user is-active %s'" % (host, unit))
    if out.strip() == "active":
        return
    cmd = ("cd %s && systemd-run --user --unit=%s --setenv=CUDA_VISIBLE_DEVICES=%d --setenv=HF_HOME=/scratch/danielco/hf "
           "--setenv=PYTHONUNBUFFERED=1 %s -p WorkingDirectory=%s bash -c '%s -m src.data_mani.tsfm_states --model %s --claim "
           "> logs/tsfm/%s.log 2>&1'" % (REPO, unit, gpu, extra, REPO, venv, model, unit))
    sh("ssh %s \"systemctl --user reset-failed %s 2>/dev/null; %s\"" % (host, unit, cmd.replace('"', '\\"')))
    log("launched", unit, "on", host, "gpu", gpu)


def main():
    done_setup, launched_gpu = set(), set()
    while True:
        status = []
        for m, sid in MODELS.items():
            p = parts(m)
            status.append("%s %d/35 preds %d %s" % (m, p, n_preds(m), "scored" if os.path.exists(os.path.join(root(m), "scores.parquet")) else ""))
            if p < 35:
                continue
            if m not in done_setup and not os.path.exists(os.path.join(root(m), "setup.json")):
                rc, out = sh("%s experiments/e193_tsfm_norm.py --tsfm %s --setup" % (PY, m))
                log("setup", m, rc, out[-300:])
                if rc:
                    continue
            done_setup.add(m)
            if n_preds(m) < N_ARMS:
                rc, out = sh("%s -m src.cluster run --jobs experiments/%s_%snorm.jobs --extra-cores 8 --only hamilton hopper lovelace "
                             "dostoievski tolstoi" % (PY, sid, m), timeout=900)
                started = [l for l in out.splitlines() if l.strip().startswith("ok")]
                if started:
                    log("cluster", m, "started", len(started))
            elif not os.path.exists(os.path.join(root(m), "scores.parquet")):
                rc, out = sh("%s experiments/e193_tsfm_norm.py --tsfm %s --score --workers 8" % (PY, m), timeout=6 * 3600)
                log("score", m, rc, out[-400:])
        log(" | ".join(status))
        # freed GPUs -> slow extractions
        if parts("t0") >= 35 and "g1" not in launched_gpu:
            if parts("toto") < 35:
                gpu_launch("deepfive", 1, "nmr-e193-toto-claim1", "toto", "/scratch/danielco/tsfm2/venv_toto/bin/python",
                           "--setenv=TSFM_AMP=fp32 --setenv=TSFM_CHUNK=16")
            else:
                gpu_launch("deepfive", 1, "nmr-e194-tabpfnts-claim1", "tabpfnts", "/scratch/danielco/tsfm/envs/tabpfnts/bin/python")
            launched_gpu.add("g1")
        if parts("toto") >= 35 and parts("tabpfnts") < 35 and "g0" not in launched_gpu:
            gpu_launch("deepfive", 0, "nmr-e194-tabpfnts-claim0", "tabpfnts", "/scratch/danielco/tsfm/envs/tabpfnts/bin/python")
            launched_gpu.add("g0")
        if all(os.path.exists(os.path.join(root(m), "scores.parquet")) for m in MODELS):
            rc, out = sh("%s experiments/e193_tsfm_norm.py --tables" % PY, timeout=3600)
            log("tables", rc, out[-600:])
            rc, out = sh("%s -m src.analysis.html_pdf results/reports/crypto/e193_e197_tsfm_tables.html" % PY, timeout=1800)
            log("pdf", rc, out[-300:])
            return
        time.sleep(LOOP)


if __name__ == "__main__":
    main()
