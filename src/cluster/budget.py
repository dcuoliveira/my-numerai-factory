"""
How much of a shared machine we are allowed to take, and who gets it.

Everything here is PURE: no ssh, no subprocess, no clock. The arithmetic that
decides whether a job may run is the part worth testing, so it is kept apart
from the part that touches the cluster (`hosts.py` measures, `manager.py`
launches). `src/tests/test_cluster.py` exercises this module alone.

THE RULE: never more than SHARE (30%) of a machine's cores, and never more than
SHARE of its RAM. These boxes belong to the whole lab -- the point is not to
leave a polite margin, it is that someone else's 64-core job must still fit
after ours lands.

Two caps apply at once, and the smaller wins:

  our budget      floor(SHARE * cores) -- our ceiling, whatever else is idle
  what is free    cores - (our jobs) - (everyone else's load)

The second cap is why an idle-looking table can still refuse a host: on
2026-10-23 deepsix carried load 38 on 32 cores from an E116 run, so its budget
of 9 cores was unreachable and it correctly took no work. A job NOT launched
through this manager counts as somebody else's load, which is deliberate --
conservative is the safe direction when the alternative is oversubscribing a
box someone else is using.

GPUs do not divide. floor(0.30 * 2) is 0, so on a two-card box the strict
reading of the rule forbids GPU work outright; `gpu_slots` overrides it and the
CLI says what that costs (one of two cards is 50%, not 30%). The default stays
strict rather than quietly rounding a 30% promise up to a half.

RAM is admission control only, NOT enforcement. These hosts run cgroup v1
(`/sys/fs/cgroup` is tmpfs, systemd 245), where a `--user` unit's MemoryMax is
recorded and ignored -- measured, not assumed: 4 busy loops under CPUQuota=100%
took 391% between them, four times the cap, while systemd reported the quota as
applied. Cores ARE enforced, by pinning each job to its own cpu set with
taskset (8 loops pinned to 2 cpus measured 194%, not 800%). So the cores
figure here is a real limit and the RAM figure is a promise the job has to keep.
"""

import math
import re
from dataclasses import dataclass, field, replace

#: Fraction of any one machine we are willing to occupy.
SHARE = 0.30


@dataclass(frozen=True)
class Host:
    """One machine, as measured a moment ago. Never cache this."""

    name: str
    cores: int = 0
    ram_gb: int = 0
    gpus: int = 0
    load1: float = 0.0
    #: cpu ids already held by OUR managed units, read from live affinities.
    our_cpus: tuple = ()
    #: gpu indices with a compute app on them, anyone's.
    busy_gpus: tuple = ()
    reachable: bool = True
    note: str = ""

    @property
    def our_cores(self):
        return len(self.our_cpus)

    @property
    def others_load(self):
        """Load we did not put there. Our own pinned jobs are already counted."""
        return max(0.0, self.load1 - self.our_cores)


@dataclass(frozen=True)
class Job:
    """One command to run, with what it says it needs."""

    name: str
    command: str
    cores: int = 1
    ram_gb: int = 4
    gpus: int = 0


@dataclass
class Placement:
    """A job, the host it goes to, and the exact resources it may use."""

    job: Job
    host: str
    cpus: tuple = ()
    gpus: tuple = ()

    @property
    def unit(self):
        return "nmr-%s" % self.job.name


def core_cap(host, share=SHARE):
    """Our hard ceiling on a host: floor(share * cores), at least 0."""
    return max(0, int(math.floor(host.cores * share)))


def ram_cap_gb(host, share=SHARE):
    return max(0, int(math.floor(host.ram_gb * share)))


def gpu_cap(host, share=SHARE, gpu_slots=None):
    """
    Cards we may use. floor(share * gpus) unless gpu_slots overrides it.

    gpu_slots is the documented exception: floor(0.30 * 2) == 0 would make the
    one GPU box unusable, so the caller may raise it deliberately rather than
    have this function round a promise up on its own.
    """
    strict = max(0, int(math.floor(host.gpus * share)))
    if gpu_slots is None:
        return strict
    return max(0, min(int(gpu_slots), host.gpus))


def available_cores(host, share=SHARE):
    """
    Cores we may still start work on: our remaining budget, but never more
    than the machine physically has free after everyone else's load.
    """
    if not host.reachable:
        return 0
    budget_left = core_cap(host, share) - host.our_cores
    physically_free = host.cores - host.our_cores - host.others_load
    return max(0, int(min(budget_left, math.floor(physically_free))))


def available_ram_gb(host, share=SHARE, reserved_gb=0):
    if not host.reachable:
        return 0
    return max(0, ram_cap_gb(host, share) - reserved_gb)


def free_gpus(host, share=SHARE, gpu_slots=None):
    """Indices we may hand out: not busy for anyone, and within our cap."""
    if not host.reachable:
        return ()
    allowed = gpu_cap(host, share, gpu_slots)
    idle = [g for g in range(host.gpus) if g not in host.busy_gpus]
    return tuple(idle[:allowed])


def parse_cpu_list(text):
    """
    "0-3,8,10-11" -> (0,1,2,3,8,10,11). Accepts taskset -pc output.

    taskset prints "pid 123's current affinity list: 0-3"; only the part after
    the last colon is read, so either form works.
    """
    if not text:
        return ()
    text = text.rsplit(":", 1)[-1].strip()
    out = []
    for piece in re.split(r"[,\s]+", text):
        if not piece:
            continue
        if "-" in piece:
            lo, _, hi = piece.partition("-")
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(piece))
            except ValueError:
                continue
    return tuple(sorted(set(out)))


def format_cpu_list(cpus):
    """(0,1,2,8) -> "0-2,8". The inverse of parse_cpu_list for taskset -c."""
    cpus = sorted(set(cpus))
    if not cpus:
        return ""
    runs, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in runs)


def pick_cpus(host, want, also_taken=()):
    """
    Choose `want` cpu ids on a host, avoiding ones we already hold.

    The ids are OUR accounting, not an exclusive reservation: pinning bounds
    how many cpus our jobs can occupy, it does not stop the kernel scheduling
    anyone else there. Lowest free id first, so a placement is reproducible.

    Known simplification: ids are taken as interchangeable. On a
    hyperthreaded box two ids can be siblings of one physical core, so a
    4-core slot may get less than 4 cores of throughput. It never gets MORE
    than the cap, which is the direction that matters for a shared machine.
    """
    held = set(host.our_cpus) | set(also_taken)
    out = [c for c in range(host.cores) if c not in held][:want]
    return tuple(out)


def distribute(jobs, hosts, share=SHARE, gpu_slots=None):
    """
    Assign jobs to hosts under the share cap. Returns (placements, unplaced).

    Biggest job first onto the emptiest host that fits it -- worst-fit
    descending. Worst-fit rather than best-fit because the request is to spread
    work over the free machines: packing jobs tightly would leave one box at
    its 30% ceiling while five sat idle, and on shared hardware the spread is
    the point. Descending because a 16-core job placed last never fits.

    `unplaced` is returned, never silently dropped: a caller that cannot see
    what did not fit will think the batch ran.
    """
    pool = {}
    for h in hosts:
        pool[h.name] = {
            "host": h,
            "cores": available_cores(h, share),
            "ram": available_ram_gb(h, share),
            "gpus": list(free_gpus(h, share, gpu_slots)),
            "taken": [],
        }

    placements, unplaced = [], []
    for job in sorted(jobs, key=lambda j: (-j.cores, -j.ram_gb, j.name)):
        best, why = None, "no host had room"
        for slot in pool.values():
            if job.cores > slot["cores"]:
                continue
            if job.ram_gb > slot["ram"]:
                why = "fits on cores but not RAM"
                continue
            if job.gpus > len(slot["gpus"]):
                why = "no free GPU within the cap"
                continue
            if best is None or slot["cores"] > best["cores"]:
                best = slot
        if best is None:
            unplaced.append((job, why))
            continue
        cpus = pick_cpus(best["host"], job.cores, best["taken"])
        if len(cpus) < job.cores:            # host smaller than it claimed
            unplaced.append((job, "could not pin %d cpus" % job.cores))
            continue
        gpus = tuple(best["gpus"][: job.gpus])
        best["taken"].extend(cpus)
        best["cores"] -= job.cores
        best["ram"] -= job.ram_gb
        best["gpus"] = best["gpus"][job.gpus :]
        placements.append(Placement(job=job, host=best["host"].name, cpus=cpus, gpus=gpus))
    return placements, unplaced
