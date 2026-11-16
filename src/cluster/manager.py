"""
Spread a batch of jobs over the free machines, taking at most 30% of each.

    python3 -m src.cluster probe                  # what is free right now
    python3 -m src.cluster plan  --jobs jobs.txt  # placement, launches nothing
    python3 -m src.cluster run   --jobs jobs.txt  # launch it
    python3 -m src.cluster status                 # our units across the cluster
    python3 -m src.cluster stop --all             # stop them, by unit
    python3 -m src.cluster reap                   # clear finished units

A jobs file is one job per line. Bare lines are commands with the defaults;
`key=value` pairs before a `--` override them:

    cores=8 ram=16 name=e120_ridge -- python3 -m src.run_forecast crypto ridge ...
    cores=4 gpu=1 name=e121_tabpfn -- python3 -m src.analysis.tabpfn_lora --tag x
    python3 -m src.analysis.report_html --model lgb --runs e120_ridge

HOW THE CAP IS HELD, and what is honest about each half:

  cores   ENFORCED. Each job is pinned to its own cpu set with `taskset`, so
          the kernel refuses it more cpus however many threads it spawns. 8
          busy loops pinned to 2 cpus measured 194%, not 800%.
  threads ENFORCED by the above, and ALSO declared: OMP/MKL/OPENBLAS/NUMEXPR
          thread counts are set to the slot size. Affinity alone would let a
          LightGBM search open 64 threads onto 9 cpus and spend its time in
          context switches rather than over budget.
  RAM     ADMISSION CONTROL ONLY. Declared, summed, never exceeded in the
          placement -- but nothing stops a job that lied. These hosts run
          cgroup v1 (`/sys/fs/cgroup` is tmpfs, systemd 245), where a --user
          unit's MemoryMax is recorded and ignored. Measured, not assumed:
          4 busy loops under CPUQuota=100% took 391% between them, four times
          the cap, while systemd reported the quota as applied. This is why
          the limits are not handed to systemd at all.
  GPUs    Pinned by CUDA_VISIBLE_DEVICES, and cpu-only jobs get it set EMPTY so
          a stray torch import cannot take a card nobody budgeted.

WHAT THIS IS NOT: there is no queue and no dependency graph. `run` places what
fits at the moment it is called and prints what did not fit -- it does not wait
for capacity, retry, or order jobs against each other. A finished job's unit is
collected by systemd, so `status` showing nothing means every job either
finished or was never started; a FAILED job persists with its exit code, which
is the one state worth keeping. The logs under logs/cluster/ are the record.

systemd still earns its place, for lifetime rather than limits: a unit outlives
the ssh session that started it, is named, and can be stopped by that name.
Stopping by name is the point -- `pkill -f <pattern>` over ssh matches the ssh
command carrying the pattern, kills the connection mid-sweep and leaves you
unable to tell which targets were reached.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys

from src.cluster import hosts as hostmod
from src.cluster.budget import (SHARE, Job, available_cores, core_cap, distribute,
                                format_cpu_list, gpu_cap, ram_cap_gb)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = "logs/cluster"
#: Every runner in this repo needs it; bare python3 is a 3.8 sandbox with no
#: sklearn. Jobs inherit it on PATH rather than having to spell it out.
CONDA_BIN = "/home/danielco/miniconda3/bin"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def slug(text):
    """A systemd-safe unit name fragment."""
    out = "".join(c if (c.isalnum() or c in "-_") else "-" for c in text.strip().lower())
    return out.strip("-") or "job"


def parse_jobs(path):
    """
    Read a jobs file. Returns (jobs, errors); errors are never swallowed.

    An unparsable line is reported with its number rather than skipped -- a
    batch that silently runs 9 of 10 jobs is worse than one that refuses.
    """
    jobs, errors, auto = [], [], 0
    try:
        with open(path) as handle:
            lines = handle.readlines()
    except OSError as exc:
        return [], ["cannot read %s: %s" % (path, exc)]

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # The separator is a STANDALONE `--`, not any occurrence of two
        # dashes: `partition("--")` splits `python3 -m x --model lgb` at
        # `--model` and reads the program name as spec fields.
        cut = re.search(r"(?:^|\s)--(?=\s|$)", line)
        if cut:
            spec, command = line[: cut.start()], line[cut.end() :]
        else:                            # bare line: all defaults
            spec, command = "", line
        command = command.strip()
        if not command:
            errors.append("line %d: no command" % lineno)
            continue
        fields = {}
        bad = False
        for piece in spec.split():
            key, _, value = piece.partition("=")
            if not value:
                errors.append("line %d: %r is not key=value" % (lineno, piece))
                bad = True
                continue
            fields[key.lower()] = value
        if bad:
            continue
        auto += 1
        name = slug(fields.get("name") or "job%02d" % auto)
        try:
            jobs.append(Job(
                name=name,
                command=command,
                cores=int(fields.get("cores", 1)),
                ram_gb=int(fields.get("ram", fields.get("ram_gb", 4))),
                gpus=int(fields.get("gpu", fields.get("gpus", 0))),
            ))
        except ValueError as exc:
            errors.append("line %d: %s" % (lineno, exc))
    seen = set()
    for job in jobs:
        if job.name in seen:
            errors.append("duplicate job name %r -- unit names must be unique "
                          "or `stop` cannot tell them apart" % job.name)
        seen.add(job.name)
    return jobs, errors


def _ssh(host, script, timeout=60, local=None):
    """Run a snippet on a host; locally if it is this box. Returns CompletedProcess."""
    if local is None:
        local = os.uname().nodename
    argv = (["bash", "-s"] if host == local else
            ["ssh", *hostmod.SSH_OPTS, "-o", "ConnectTimeout=10", host, "bash", "-s"])
    return subprocess.run(argv, input=script, capture_output=True, text=True, timeout=timeout)


def script_path(placement, repo=REPO):
    """Where this job's command is written. Shared NFS, so every box sees it."""
    return os.path.join(repo, LOGDIR, "%s.cmd.sh" % placement.job.name)


def log_path(placement, repo=REPO):
    return os.path.join(repo, LOGDIR, "%s.out" % placement.job.name)


def job_script(placement, repo=REPO):
    """
    The command, in a file, run by nothing but bash.

    The command does NOT go on the systemd-run command line. systemd does its
    own `$` expansion there and collapses `$$` to a single `$` before bash ever
    sees it -- measured: a job whose command contained `taskset -pc $$` ran as
    `taskset -pc $` and failed. Anything with `$VAR`, `$$` or `%i` in it would
    be quietly rewritten. Writing the command to a file removes every layer
    between what was asked for and what runs, and leaves the exact command on
    disk next to its log for afterwards.
    """
    job = placement.job
    return (
        "#!/bin/bash\n"
        "# written by src/cluster: job %s on %s, cpus %s\n"
        "cd %s || exit 1\n"
        "exec %s > %s 2>&1\n"
        % (job.name, placement.host, format_cpu_list(placement.cpus),
           shlex.quote(repo), job.command, shlex.quote(log_path(placement, repo)))
    )


def launch_script(placement, repo=REPO):
    """
    The exact shell that starts this placement. Kept separate from running it
    so a plan can show the real command instead of a description of one.
    """
    job = placement.job
    env = ["--setenv=%s=%d" % (var, max(1, job.cores)) for var in THREAD_VARS]
    env.append("--setenv=CUDA_VISIBLE_DEVICES=%s"
               % ",".join(str(g) for g in placement.gpus))
    env.append("--setenv=PATH=%s:/usr/local/bin:/usr/bin:/bin" % CONDA_BIN)
    env.append("--setenv=NMR_SLOT_CORES=%d" % job.cores)
    return (
        'export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"\n'
        "systemd-run --user --unit=%s %s -- taskset -c %s /bin/bash %s\n"
        % (shlex.quote(placement.unit), " ".join(shlex.quote(e) for e in env),
           shlex.quote(format_cpu_list(placement.cpus)),
           shlex.quote(script_path(placement, repo)))
    )


def launch(placement, repo=REPO, local=None):
    """Start one placement. Returns (ok, message)."""
    path = script_path(placement, repo)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(job_script(placement, repo))
        os.chmod(path, 0o755)
    except OSError as exc:
        return False, "cannot write %s: %s" % (path, exc)
    try:
        done = _ssh(placement.host, launch_script(placement, repo), local=local)
    except subprocess.TimeoutExpired:
        return False, "timed out starting unit"
    text = ((done.stdout or "") + (done.stderr or "")).strip().splitlines()
    if done.returncode != 0:
        return False, (text[-1] if text else "exit %d" % done.returncode)[:120]
    return True, "%s on %s cpus %s" % (placement.unit, placement.host,
                                       format_cpu_list(placement.cpus))


STATUS_SCRIPT = r"""
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for u in $(systemctl --user list-units 'nmr-*' --all --no-legend 2>/dev/null | awk '{print $1}'); do
  state=$(systemctl --user show "$u" -p ActiveState --value 2>/dev/null)
  sub=$(systemctl --user show "$u" -p SubState --value 2>/dev/null)
  pid=$(systemctl --user show "$u" -p MainPID --value 2>/dev/null)
  code=$(systemctl --user show "$u" -p ExecMainStatus --value 2>/dev/null)
  cpus=""; el=""
  if [ -n "$pid" ] && [ "$pid" != "0" ]; then
    cpus=$(taskset -pc "$pid" 2>/dev/null | sed 's/.*: //')
    el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  fi
  printf '%s\t%s/%s\t%s\t%s\t%s\t%s\n' "$u" "$state" "$sub" "${pid:-0}" "${cpus:--}" "${el:--}" "${code:--}"
done
"""


def status(names=None, timeout=30):
    """Our units everywhere, newest state. Returns {host: [row, ...]}."""
    out = {}
    names = names or hostmod.roster()
    local = os.uname().nodename
    import concurrent.futures as cf
    def one(name):
        try:
            done = _ssh(name, STATUS_SCRIPT, timeout=timeout, local=local)
        except subprocess.TimeoutExpired:
            return name, None
        if done.returncode != 0 and not done.stdout.strip():
            return name, None
        rows = []
        for line in done.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 6:
                rows.append(dict(zip(("unit", "state", "pid", "cpus", "elapsed", "code"), parts)))
        return name, rows
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        for name, rows in pool.map(one, names):
            if rows:
                out[name] = rows
    return out


def stop(host, units, timeout=60, local=None):
    """
    Stop units by NAME, then verify and fall back to killing by PID.

    Never by pattern: under cgroup v1 a stopped user unit does not always take
    its children with it, and the obvious fix -- pkill -f -- matches the ssh
    command that carries the pattern. So: stop the unit, re-read the main pid,
    and kill that pid if it is somehow still there.
    """
    if not units:
        return True, "nothing to stop"
    quoted = " ".join(shlex.quote(u) for u in units)
    script = (
        'export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"\n'
        "for u in %s; do\n"
        '  pid=$(systemctl --user show "$u" -p MainPID --value 2>/dev/null)\n'
        '  systemctl --user stop "$u" >/dev/null 2>&1\n'
        '  if [ -n "$pid" ] && [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then\n'
        '    kill -TERM "$pid" 2>/dev/null; sleep 2\n'
        '    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null\n'
        "  fi\n"
        '  systemctl --user reset-failed "$u" >/dev/null 2>&1\n'
        '  echo "stopped $u"\n'
        "done\n" % quoted
    )
    try:
        done = _ssh(host, script, timeout=timeout, local=local)
    except subprocess.TimeoutExpired:
        return False, "timed out stopping units"
    return done.returncode == 0, (done.stdout or done.stderr or "").strip()


def disk_headroom_ok(need_gb, repo=REPO):
    """
    Can we still write `need_gb` to the shared home?

    `df` cannot answer this. /home is one 31 T NFS export with a per-user quota
    around 100 G, and nothing in df or `quota` shows the quota -- a job dies
    with ENOSPC on a filesystem that looks three-quarters empty, which is how
    24 GPU runs died silently on 2026-10-17. The only honest test is to write
    incompressible bytes and see whether they land; the pool compresses, so
    zeros would prove nothing.
    """
    if need_gb <= 0:
        return True, "skipped"
    probe = os.path.join(repo, ".nmr_quota_probe")
    try:
        done = subprocess.run(
            ["dd", "if=/dev/urandom", "of=%s" % probe, "bs=1M",
             "count=%d" % int(need_gb * 1024), "conv=fsync"],
            capture_output=True, text=True, timeout=600)
        ok = done.returncode == 0
        why = "" if ok else (done.stderr or "").strip().splitlines()[-1][:90]
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, why = False, str(exc)[:90]
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    return ok, ("%d G of headroom confirmed" % need_gb if ok
                else "cannot write %d G: %s" % (need_gb, why))
