"""
Distribute jobs across the lab's shared machines, never taking more than 30%
of any one of them.

    from src.cluster import probe_all, distribute, launch

`budget` holds the arithmetic (pure, tested), `hosts` measures the cluster,
`manager` launches and tracks. See `python3 -m src.cluster --help`.
"""

from src.cluster.budget import (SHARE, Host, Job, Placement, available_cores,
                                core_cap, distribute, free_gpus, gpu_cap,
                                ram_cap_gb)
from src.cluster.hosts import probe, probe_all, reachable, roster
from src.cluster.manager import launch, parse_jobs, status, stop

__all__ = ["SHARE", "Host", "Job", "Placement", "available_cores", "core_cap",
           "distribute", "free_gpus", "gpu_cap", "ram_cap_gb", "probe",
           "probe_all", "reachable", "roster", "launch", "parse_jobs",
           "status", "stop"]
