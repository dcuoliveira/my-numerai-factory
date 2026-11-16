"""
Who the machines are, and what is free on them RIGHT NOW.

The roster is read from ~/.ssh/config, not hardcoded. CLAUDE.md's host table
carries a date because boxes come and go -- `curie` answers on 22 and refuses
the key, five `deep*` boxes have no route -- and a list baked into source is
wrong the first time one is renamed.

Occupancy is DERIVED, never recorded: a managed job's cpu set is read back from
the live process's own affinity mask (`taskset -pc`), not from a state file we
wrote when we launched it. Same reason `experiment_db` refuses to store a
"running" flag -- a state file outlives the process that justified it, and the
first time that happens every later placement is built on a lie. The cost is an
ssh round trip per host per call; they run in parallel and the whole cluster
answers in about a second.

Everything measured here has a shelf life of seconds. Probe, place, launch --
never probe once and reuse it.
"""

import concurrent.futures as cf
import os
import re
import subprocess

from src.cluster.budget import Host, parse_cpu_list

SSH_CONFIG = os.path.expanduser("~/.ssh/config")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]

#: Unit-name prefix for everything this manager launches. Chosen so
#: `systemctl --user list-units 'nmr-*'` finds exactly our jobs and nothing
#: else, which is what lets `stop` work by unit instead of by `pkill -f` --
#: a pattern passed over ssh matches the ssh command carrying it and kills the
#: connection mid-sweep, leaving no way to tell which targets were reached.
UNIT_PREFIX = "nmr-"

#: One round trip, every number the placer needs. `systemctl --user` over a
#: non-login ssh has no DBus address unless XDG_RUNTIME_DIR is set by hand.
PROBE = r"""
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
printf 'cores=%s;' "$(nproc 2>/dev/null || echo 0)"
printf 'ram=%s;' "$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
printf 'load1=%s;' "$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
if command -v nvidia-smi >/dev/null 2>&1; then
  # Only rows that are literally "<index>, <mib>" count. A box whose driver is
  # unreachable (deepfour) still HAS nvidia-smi, and it prints its failure on
  # stdout -- counting lines would report that error as one idle GPU and send
  # a GPU job to a box that cannot run it.
  gpuinfo=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {print $1","$2}')
  if [ -n "$gpuinfo" ]; then
    printf 'gpus=%s;' "$(printf '%s\n' "$gpuinfo" | wc -l)"
    printf 'busygpus=%s;' "$(printf '%s\n' "$gpuinfo" | awk -F, '$2>200{printf "%s,",$1}')"
  else
    printf 'gpus=0;busygpus=;driver=broken;'
  fi
else
  printf 'gpus=0;busygpus=;'
fi
occ=""
for u in $(systemctl --user list-units 'nmr-*' --state=active --no-legend 2>/dev/null | awk '{print $1}'); do
  pid=$(systemctl --user show "$u" -p MainPID --value 2>/dev/null)
  if [ -n "$pid" ] && [ "$pid" != "0" ]; then
    occ="$occ,$(taskset -pc "$pid" 2>/dev/null | sed 's/.*: //')"
  fi
done
printf 'ourcpus=%s;' "$occ"
printf 'units=%s' "$(systemctl --user list-units 'nmr-*' --state=active --no-legend 2>/dev/null | awk '{print $1}' | paste -sd, -)"
"""


def roster(path=SSH_CONFIG):
    """
    Every host named in ~/.ssh/config, in the order declared.

    Wildcard stanzas (`Host *`) are skipped -- they configure, they do not
    name a machine.
    """
    names = []
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not re.match(r"(?i)^host\s", line):
                    continue
                for name in line.split()[1:]:
                    if "*" in name or "?" in name or name in names:
                        continue
                    names.append(name)
    except OSError:
        return []
    return names


def _parse(name, text):
    """Turn the probe's `key=value;` line into a Host."""
    fields = {}
    for piece in text.strip().split(";"):
        key, _, value = piece.partition("=")
        if key:
            fields[key.strip()] = value.strip()
    if "cores" not in fields:
        return Host(name=name, reachable=False, note="probe returned nothing usable")

    def as_int(key):
        try:
            return int(float(fields.get(key) or 0))
        except ValueError:
            return 0

    try:
        load1 = float(fields.get("load1") or 0.0)
    except ValueError:
        load1 = 0.0
    note = "gpu driver unreachable" if fields.get("driver") == "broken" else ""
    return Host(
        name=name,
        cores=as_int("cores"),
        ram_gb=as_int("ram"),
        gpus=as_int("gpus"),
        load1=load1,
        our_cpus=parse_cpu_list(fields.get("ourcpus", "")),
        busy_gpus=parse_cpu_list(fields.get("busygpus", "")),
        reachable=True,
        note=note,
    )


def _reason(stderr, returncode):
    """A short, true reason a host did not answer. The distinction matters:
    'no route' is a box that is off, 'permission denied' is a box that is up
    and not ours, and only the first is worth retrying later."""
    text = (stderr or "").strip().splitlines()
    first = text[0] if text else ""
    if "No route to host" in first:
        return "no route"
    if "Permission denied" in first:
        return "up, key rejected"
    if "Connection timed out" in first or returncode == 124:
        return "timed out"
    if "Name or service not known" in first:
        return "unknown name"
    return (first or "ssh failed")[:60]


def probe(name, timeout=12, local_name=None):
    """Measure one host. Never raises: an unreachable box is a Host, not an error."""
    if local_name is None:
        local_name = os.uname().nodename
    try:
        if name == local_name:
            done = subprocess.run(
                ["bash", "-s"], input=PROBE, capture_output=True, text=True, timeout=timeout
            )
        else:
            done = subprocess.run(
                ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=%d" % max(1, timeout - 2), name,
                 "bash", "-s"],
                input=PROBE, capture_output=True, text=True, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return Host(name=name, reachable=False, note="timed out after %ds" % timeout)
    except OSError as exc:
        return Host(name=name, reachable=False, note=str(exc)[:60])
    if done.returncode != 0 and not done.stdout.strip():
        return Host(name=name, reachable=False, note=_reason(done.stderr, done.returncode))
    return _parse(name, done.stdout)


def probe_all(names=None, timeout=12, workers=16):
    """
    Measure every host at once. Order follows `names`, so output is stable.

    One slow box must not hold up the rest -- 14 hosts where five have no route
    would take a minute and a half in sequence and does take about ten seconds
    in parallel, nearly all of it waiting for the dead ones to time out.
    """
    if names is None:
        names = roster()
    local = os.uname().nodename
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, len(names) or 1))) as pool:
        futures = {pool.submit(probe, n, timeout, local): n for n in names}
        found = {}
        for fut in cf.as_completed(futures):
            host = fut.result()
            found[host.name] = host
    return [found[n] for n in names if n in found]


def reachable(hosts):
    return [h for h in hosts if h.reachable and h.cores > 0]
