---
name: cluster-job-placement
description: "Decide WHICH of the lab's shared machines a job runs on, how much of it to take, and launch it there — the 30% share rule, the one box with a working GPU, slot sizing from --search-jobs/--n-jobs, and the four ways a cross-machine job dies with nothing in the log. Use whenever launching a study, sweep, diagnose batch or GPU fit, picking a server, asking whether a box is free or busy, spreading arms over machines, running something that must outlive the session, or explaining why a job vanished without a traceback and why experiment_db calls a running study 'partial'. Triggers on 'run this study', 'which server', 'where should this run', 'is deepfive free', 'run it on the GPU', 'distribute these arms', 'launch in the background', 'run overnight', 'my job died with no error', 'nothing is running', 'Disk quota exceeded'."
---

# Running a job on the lab's shared machines

This is the placement half of `crypto-experiment-reporting`: that skill covers
what a study runs and how long it will take, this one covers **where it runs and
how much of the box it may have**. Price the study there first — slot size
follows from the cadence and the model, not from the feature count.

## Do this

```bash
python3 -m src.cluster probe                  # cores free on every box, now
python3 -m src.cluster plan --jobs <file>     # placement, launches nothing
python3 -m src.cluster run  --jobs <file>     # launch
python3 -m src.cluster status                 # our units, every box
python3 -m src.cluster stop --all
```

`experiments/cluster_jobs.example` is the jobs-file format. `plan` is free and
read-only; run it before `run` every time, because what is free changes between
one command and the next.

## Which box

**Never pick from memory — `probe` first.** Load moves by the hour: deepsix sat
at 0.00 one hour and 44 the next, both on 2026-10-23. The reachable set as of
then, which is what `probe` confirms or contradicts:

| box | cores | the reason to pick it |
|---|---|---|
| `hamilton` `hopper` `lovelace` | 64 | widest. The only ones whose 30% share (19 cores) fits a 16-core arm |
| `deepfive` | 32 | **the only working GPU** — 2x RTX 2070, 8 G each |
| `deepfour` | 32 | cards present, driver unreachable. A CPU box in practice |
| `deepsix` `tolstoi` `dostoievski` | 32 | ordinary CPU boxes |

`curie` answers on 22 and rejects the key — up, but not ours. `puchkin` and
`deepzero`/`one`/`two`/`three` have no route. Note the spellings: `puchkin` and
`dostoievski`.

Three rules settle almost every case:

1. **GPU work goes to `deepfive` or nowhere.** TabPFN, LoRA, torch fits. Do not
   send a GPU job to `deepfive`'s twin `deepfour` expecting a graceful fallback
   — `nvidia-smi` there cannot reach the driver and prints its error on stdout,
   so anything counting lines reads that error as an idle card.
2. **A 16-core arm needs a 64-core box.** 30% of 32 is 9, which `rf_ktz` at full
   parallelism does not fit. See the sizing table below.
3. **Otherwise take whatever `probe` says is free**, and let it spread the work.

`/home` is one NFS export mounted by all of them, so there is nothing to stage
and nothing to sync: same checkout, same `/home/danielco/miniconda3`, same
`results/`, same `logs/`. Train on one box and read the output from another.

## The 30% rule, and why systemd cannot hold it

These are the whole lab's machines. The ceiling is **30% of any one box** —
9 cores of 32, 19 of 64 — and the point is not politeness, it is that someone
else's 64-core job must still fit after ours lands.

**Do not express the limit as a systemd property.** These hosts run cgroup v1
(`/sys/fs/cgroup` is tmpfs, systemd 245), where a `--user` unit's `CPUQuota` and
`MemoryMax` are accepted, recorded, and then ignored. Measured, not assumed: 4
busy loops under `CPUQuota=100%` took **391% between them** — four times the cap —
while systemd reported the quota as applied (`CPUQuotaPerSecUSec=1s`). A manager built on those
properties reports caps it is not keeping.

What does hold:

- **`taskset -c <cpus>`** — kernel affinity, inherited by children and threads.
  8 busy loops pinned to 2 cpus measured **194%**, not 800%.
- **`OMP_NUM_THREADS` / `MKL_` / `OPENBLAS_` / `NUMEXPR_`** set to the slot size.
  Affinity caps the cpus; without these a search opens one thread per machine
  core and spends its time context-switching inside its own slot.
- **RAM: admission control only.** Declared and never overbooked in the
  placement, but nothing stops a job that asked for less than it uses.

GPU cards do not divide: `floor(0.30 x 2)` is 0, so the strict share forbids GPU
work outright and `--gpu-slots 1` is needed to allow it. One of two cards is
**50%** — an exception taken deliberately, not a 30% promise rounded up.

## Sizing the slot

A pinned job sees only its slot, which makes the sizing honest: `nproc` in an
8-cpu slot on a 64-core box reports **8**, and `joblib`'s `effective_n_jobs(-1)`
reports 8 too. So **`--n-jobs -1` is safe inside a slot** — it becomes the slot
size, not the machine's.

What is not safe is `--search-jobs` multiplying it. Each search process gets the
whole slot, so `--search-jobs 8` with `--n-jobs -1` puts 64 threads on 8 cpus.
**Set both explicitly so the product is the slot.** The deployed arrays in
`experiments/mommask_core.sh` already encode this, and the product is the slot
each model wants:

| model | `--search-jobs` | `--n-jobs` | cores | fits a 32-core box at 30%? |
|---|---|---|---|---|
| `ridge` | 8 | 1 | 8 | yes (budget 9) |
| `lgb_ktz` | 4 | 2 | 8 | yes |
| `catboost_ktz` | 1 | 6 | 6 | yes |
| `mlp_ktz` | 8 | 1 | 8 | yes |
| `rf_ktz` | 8 | 2 | **16** | **no — needs a 64-core box** |

One trap in the pipeline's own guard: `src/run_forecast.py` refuses an
oversubscribing `--search-jobs`/`--n-jobs` pair and suggests
`cores / search-jobs`, but it sizes that from `os.cpu_count()`, which **ignores
affinity** — 32 inside a 4-cpu slot, where `os.sched_getaffinity(0)` correctly
says 4. Inside a slot its advice is too large by the ratio of box to slot. Size
from the slot (`NMR_SLOT_CORES` is exported into every managed job), not from
what that guard suggests.

`run_ensemble` blends finish in under a minute and only combine existing
forecasts — give them one core, or just run them locally.

## Four ways a cross-machine job dies quietly

**`experiment_db` cannot see other boxes.** `running_studies()` reads the LOCAL
`/proc`, by design, so a study running anywhere else reads as `partial` —
indistinguishable from one that died half-finished. On 2026-10-23, E116 was 33
minutes into a run on deepsix while the process table on deepfive was empty.
Before calling a study stalled, or reclaiming its forecasts, ask every box:

```bash
for h in hamilton hopper lovelace tolstoi dostoievski deepfour deepfive deepsix; do
  printf '=== %s\n' "$h"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    "ps -u danielco -o etime,cmd | grep -E 'run_forecast|run_nn|experiments/e[0-9]' \
     | grep -v grep" 2>/dev/null
done
```

**`df` lies about free space.** `/home` shows 31 T with three quarters free,
while a per-user quota near 100 G is what actually stops a write, and nothing in
`df` or `quota` shows it. A job dies with ENOSPC on a filesystem that looks
idle; that is how 24 GPU runs died at once on 2026-10-17, with no traceback,
because a memory-mapped `pred_*.npy` write on a full quota fails silently. The
only honest test writes incompressible bytes — the pool compresses, so zeros
prove nothing, and `du` on a text log overstates what deleting it frees by more
than 10x. `src.cluster run` preflights this; `--min-free-gb 0` skips it.

**`pkill -f <pattern>` over ssh kills the connection.** The ssh invocation
carries the pattern on its own command line, so it matches itself, dies
mid-sweep, and leaves no way to tell which targets were reached. Managed jobs are
`nmr-<name>` units — stop them by name. Otherwise kill by PID read from `ps`, or
use the bracket trick (`pgrep -f 'tabpfn_[l]ora'`).

**systemd rewrites `$` on a unit's command line.** It does its own expansion
there and collapses `$$` to a single `$` before bash sees it — measured: a job
whose command held `taskset -pc $$` ran as `taskset -pc $`. Anything with
`$VAR`, `$$` or `%i` is quietly altered. `src.cluster` writes the command to
`logs/cluster/<name>.cmd.sh` and runs the file for this reason; if you launch by
hand, do the same rather than inlining the command.

## Before you launch, check

- `probe` was run just now, not remembered. Load is the number that moves.
- The slot fits the model: `--search-jobs x --n-jobs` equals the cores asked
  for, and a 16-core arm is on a 64-core box.
- GPU jobs name `deepfive` and pass `--gpu-slots 1`; nothing else has a driver.
- Disk headroom confirmed by a write, not by `df`.
- The job will outlive the session: a `nmr-` unit, or `systemd-run --user
  --unit=<name> --collect` by hand. `ssh` plus `setsid` alone is not enough.
- Jobs NOT launched through the manager count as somebody else's load, so they
  shut the box down for the next placement. That is deliberate — conservative is
  the safe direction when the alternative is oversubscribing a machine in use.
- A job that finished successfully leaves no unit behind (systemd collects it),
  so an empty `status` means finished-or-never-started. The log under
  `logs/cluster/` is the record; a FAILED job persists with its exit code.
