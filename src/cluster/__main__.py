"""CLI for the cluster resource manager. See src/cluster/manager.py."""

import argparse
import sys

from src.cluster import hosts as hostmod
from src.cluster import manager as mgr
from src.cluster.budget import (SHARE, available_cores, available_ram_gb, core_cap,
                                distribute, format_cpu_list, free_gpus, gpu_cap,
                                ram_cap_gb)


def _probe(args):
    found = hostmod.probe_all(timeout=args.timeout)
    live = hostmod.reachable(found)
    print("%-13s %5s %6s %7s %5s   %s" % ("host", "cores", "RAM", "load1", "gpu", "we may use"))
    total_cores = 0
    for h in live:
        cores = available_cores(h, args.share, args.extra_cores)
        total_cores += cores
        gpus = free_gpus(h, args.share, args.gpu_slots)
        note = (" (%s)" % h.note) if h.note else ""
        # RAM shown as 0 when no cores are free: nothing can be placed there,
        # and printing a spare 75 G beside "0 cores" reads as spare capacity.
        print("%-13s %5d %5dG %7.2f %5s   %2d cores, %3d G%s%s" % (
            h.name, h.cores, h.ram_gb, h.load1, h.gpus or "-",
            cores, available_ram_gb(h, args.share) if cores else 0,
            ", gpu %s" % ",".join(map(str, gpus)) if gpus else "", note))
    down = [h for h in found if h not in live]
    if down:
        print("\nunreachable: " + ", ".join("%s (%s)" % (h.name, h.note) for h in down))
    print("\n%d cores available across %d hosts at a %.0f%% share%s."
          % (total_cores, len(live), args.share * 100,
             " + %d extra cores per host (one-off)" % args.extra_cores
             if args.extra_cores else ""))
    if any(h.gpus for h in live) and not args.gpu_slots:
        box = ", ".join(h.name for h in live if h.gpus)
        print("GPU: 0 slots. floor(%.2f x 2 cards) is 0, so the 30%% rule forbids GPU work "
              "on %s.\n     Pass --gpu-slots 1 to allow one card -- that is 50%% of a "
              "two-card box, deliberately\n     above the share, which is why it is not "
              "the default." % (args.share, box))
    return 0


def _plan_common(args):
    jobs, errors = mgr.parse_jobs(args.jobs)
    if errors:
        for e in errors:
            print("jobs file: %s" % e, file=sys.stderr)
        return None, None, None
    if not jobs:
        print("no jobs in %s" % args.jobs, file=sys.stderr)
        return None, None, None
    found = hostmod.probe_all(timeout=args.timeout)
    live = [h for h in hostmod.reachable(found) if h.name not in set(args.exclude or ())]
    if args.only:
        live = [h for h in live if h.name in set(args.only)]
    placements, unplaced = distribute(jobs, live, args.share, args.gpu_slots,
                                      args.extra_cores)

    busy = set()
    for host, rows in mgr.status(names=[h.name for h in live], timeout=args.timeout).items():
        for row in rows:
            if row["state"].startswith("active"):
                busy.add(row["unit"].replace(".service", ""))
    clash = [p for p in placements if p.unit in busy]
    if clash:
        for p in clash:
            print("unit %s is already running -- rename the job or stop it first"
                  % p.unit, file=sys.stderr)
        return None, None, None
    return placements, unplaced, live


def _show(placements, unplaced):
    if placements:
        print("%-18s %5s %4s %3s   %-12s %s" % ("job", "cores", "ram", "gpu", "host", "cpus"))
        for p in placements:
            print("%-18s %5d %3dG %3d   %-12s %s" % (
                p.job.name, p.job.cores, p.job.ram_gb, p.job.gpus, p.host,
                format_cpu_list(p.cpus) + (" gpu %s" % ",".join(map(str, p.gpus)) if p.gpus else "")))
    if unplaced:
        print("\nNOT placed (%d):" % len(unplaced))
        for job, why in unplaced:
            print("  %-18s %d cores, %d G, %d gpu -- %s"
                  % (job.name, job.cores, job.ram_gb, job.gpus, why))


def _plan(args):
    placements, unplaced, _ = _plan_common(args)
    if placements is None:
        return 2
    _show(placements, unplaced)
    if args.verbose:
        print("\n--- exact commands ---")
        for p in placements:
            print("# %s on %s\n%s" % (p.unit, p.host, mgr.launch_script(p)))
    print("\nplan only; nothing launched. Re-run with `run` to start these.")
    return 0


def _run(args):
    placements, unplaced, _ = _plan_common(args)
    if placements is None:
        return 2
    _show(placements, unplaced)
    need = args.min_free_gb
    ok, why = mgr.disk_headroom_ok(need)
    if not ok:
        print("\nREFUSING to launch: %s" % why, file=sys.stderr)
        print("The shared home has a per-user quota that df does not show; a job that "
              "runs out\nof it dies with no traceback. Free space or pass "
              "--min-free-gb 0 to override.", file=sys.stderr)
        return 1
    print("\ndisk: %s" % why)
    print("--- launching ---")
    started = failed = 0
    for p in placements:
        good, message = mgr.launch(p)
        print("  %-4s %s" % ("ok" if good else "FAIL", message))
        started, failed = started + bool(good), failed + (not good)
    print("\n%d started, %d failed, %d not placed." % (started, failed, len(unplaced)))
    print("Logs: logs/cluster/<job>.out (shared NFS, readable from any box).")
    return 1 if failed else 0


def _status(args):
    everywhere = mgr.status(timeout=args.timeout)
    if not everywhere:
        print("no managed units anywhere.")
        return 0
    for host in sorted(everywhere):
        print("=== %s" % host)
        print("  %-22s %-16s %8s %-10s %6s %s" % ("unit", "state", "pid", "cpus", "elapsed", "exit"))
        for row in everywhere[host]:
            print("  %-22s %-16s %8s %-10s %6s %s" % (
                row["unit"][:22], row["state"], row["pid"], row["cpus"][:10],
                row["elapsed"], row["code"]))
    return 0


def _stop(args):
    everywhere = mgr.status(timeout=args.timeout)
    wanted = set(args.units or ())
    todo = {}
    for host, rows in everywhere.items():
        for row in rows:
            unit = row["unit"]
            bare = unit.replace(".service", "")
            if args.all or unit in wanted or bare in wanted:
                todo.setdefault(host, []).append(unit)
    if not todo:
        print("nothing matched.")
        return 0
    for host, units in todo.items():
        ok, message = mgr.stop(host, units)
        print("%s: %s" % (host, message.replace("\n", "; ") if ok else "FAILED " + message))
    return 0


def _reap(args):
    everywhere = mgr.status(timeout=args.timeout)
    todo = {}
    for host, rows in everywhere.items():
        for row in rows:
            if not row["state"].startswith("active"):
                todo.setdefault(host, []).append(row["unit"])
    if not todo:
        print("no finished units to clear.")
        return 0
    for host, units in todo.items():
        ok, message = mgr.stop(host, units)
        print("%s: cleared %d" % (host, len(units)) if ok else "%s: FAILED %s" % (host, message))
    return 0


def main(argv=None):
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--share", type=float, default=SHARE,
                        help="fraction of each machine we may use (default %.2f)" % SHARE)
    parent.add_argument("--extra-cores", type=int, default=0,
                        help="ONE-OFF allowance: this many cores per host on top of "
                             "floor(share * cores). Default 0 keeps the standing rule; "
                             "pass it only when explicitly granted for a batch")
    parent.add_argument("--gpu-slots", type=int, default=None,
                        help="cards we may use per host; default is the strict "
                             "floor(share * cards), which is 0 on a 2-card box")
    parent.add_argument("--timeout", type=int, default=12, help="per-host ssh timeout")

    ap = argparse.ArgumentParser(
        prog="python3 -m src.cluster",
        description="Spread jobs over the lab's shared machines, at most 30% of each.")
    subs = ap.add_subparsers(dest="cmd", required=True)

    subs.add_parser("probe", parents=[parent], help="what is free right now").set_defaults(fn=_probe)

    for name, fn, helptext in (("plan", _plan, "show placement, launch nothing"),
                               ("run", _run, "place and launch")):
        sub = subs.add_parser(name, parents=[parent], help=helptext)
        sub.add_argument("--jobs", required=True, help="jobs file")
        sub.add_argument("--only", nargs="*", help="restrict to these hosts")
        sub.add_argument("--exclude", nargs="*", help="skip these hosts")
        sub.add_argument("--min-free-gb", type=int, default=2,
                         help="refuse to launch unless this much is writable on the "
                              "shared home (0 skips; df cannot answer this)")
        sub.add_argument("-v", "--verbose", action="store_true", help="print exact commands")
        sub.set_defaults(fn=fn)

    subs.add_parser("status", parents=[parent], help="our units everywhere").set_defaults(fn=_status)
    sub = subs.add_parser("stop", parents=[parent], help="stop units by name")
    sub.add_argument("units", nargs="*")
    sub.add_argument("--all", action="store_true")
    sub.set_defaults(fn=_stop)
    subs.add_parser("reap", parents=[parent], help="clear finished units").set_defaults(fn=_reap)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
