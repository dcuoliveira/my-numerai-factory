---
name: safe-crontab-edits
description: "Add, change or remove a cron entry on a box whose crontab holds more than one unrelated job. Use whenever a task involves crontab, `crontab -e`, cron schedules, scheduling a recurring script, changing when something runs, or wiring a new pipeline into cron — and especially before running any command containing `crontab`. This crontab drives two independent live Numerai submission pipelines plus their watchdogs; replacing it wholesale silently stops the ones you were not thinking about, and nothing tells you until a round is missed."
---

# Editing a shared crontab without losing the other jobs

## The failure this prevents

`crontab FILE` and `crontab -` do not merge. They **replace the entire crontab**.
So the natural-looking way to add a job:

```bash
echo "0 3 * * * /path/to/my_new_job.sh" | crontab -      # WRONG
```

installs a crontab containing exactly one line and deletes everything else. There
is no confirmation, no backup, and no error. The next signal is a job that stopped
running days ago.

On this box that is not hypothetical. `crontab -l` currently drives **two
unrelated pipelines** and their watchdogs:

| lines | what | owner |
|---|---|---|
| `05/25/45 9 * * *` | Numerai **Crypto** submissions | `src/live/cron_round.sh` (this repo) |
| `15/35/55 9 * * *` | Numerai **Classic** `iara` submission | `../my-numerai-pipeline/src/cron_submit.sh` |
| `30 10`, `45 11` | watchdog that asks Numerai whether `iara` is in | `../my-numerai-pipeline/src/check_submission.py` |

The crontab's own comments record what losing these costs: *"round 1336 was
caught by hand with 2.5 minutes left, and round 1335 was lost outright"*. The
`iara` block carries an explicit instruction — **do not remove them without
removing the watchdog too** — and the two pipelines are deliberately interleaved
(`:05/:25/:45` against `:15/:35/:55`) so they never compete for CPU. A rewrite
that drops or reorders lines breaks that interleaving silently.

## First: check whether you need a cron change at all

Most "schedule this new thing" requests on this repo do **not** need one. Cron
calls `src/live/cron_round.sh`, which runs `src.live.run_round`, which iterates
`registry.ORDER`. So **adding a CURUPIRA slot needs no crontab edit** — put it in
`src/live/registry.py` (plus `DEFAULT_FIT_SECONDS` in `run_round.py`) and the
next scheduled round picks it up.

Ask, in this order:

1. Does an existing scheduled entry point already cover this? Then edit the code
   it calls, not cron.
2. Is this a one-off? Then it is not a cron job.
3. Only if it is genuinely a new recurring entry point does the crontab change.

## The safe procedure

Never pipe into `crontab -`. Always: snapshot, edit a copy, install the copy,
verify nothing was lost.

```bash
bash .claude/skills/safe-crontab-edits/scripts/crontab_add.sh \
    --comment "What this job is and what breaks without it" \
    --line "0 3 * * * /bin/bash /abs/path/to/script.sh"
```

The helper snapshots to `~/.crontab-backups/`, appends, refuses to install if any
pre-existing line would disappear, and prints the diff. To do it by hand, the
same four steps:

```bash
mkdir -p ~/.crontab-backups
crontab -l > ~/.crontab-backups/crontab.$(date +%Y%m%d-%H%M%S).bak   # 1. snapshot
cp ~/.crontab-backups/crontab.*.bak /tmp/crontab.new                  # 2. copy
# 3. edit /tmp/crontab.new — append, never rewrite
crontab /tmp/crontab.new                                              # 4. install
crontab -l | diff ~/.crontab-backups/crontab.*.bak - ; echo "exit $?" # 5. verify
```

Step 5 is the one people skip. The diff must show **only** your addition.

To restore: `crontab ~/.crontab-backups/crontab.<stamp>.bak`.

## Rules

- **`crontab -r` is never the answer.** It deletes the whole crontab with no
  prompt. To remove one job, filter it out of a copy and install the copy.
- **Snapshot before every change**, including ones you are sure about.
- **Verify by diff afterwards**, not by reading what you meant to write.
- **`crontab -l` exits non-zero when no crontab exists**, so a script doing
  `crontab -l > f` under `set -e` dies there. Use `crontab -l 2>/dev/null || true`.
- **Append; do not reformat.** The comment blocks in this crontab encode
  deadlines, incident history and the interleaving rationale. Reflowing them
  loses the reasons.
- **`MAILTO=""` applies to every line after it.** Adding a job below it means
  cron sends no mail on failure — so that job needs its own alerting, the way
  `check_submission.py` is the watchdog for `cron_submit.sh`.
- **Cron gives almost no environment**: `PATH=/usr/bin:/bin`, no conda, no
  working directory, no profile. Name the interpreter absolutely
  (`/home/danielco/miniconda3/bin/python3`), `cd` explicitly, and redirect to a
  log — see the header of `src/live/cron_round.sh`, which exists entirely to
  supply what cron does not.
- **Use `flock`** for anything that can overlap with its own previous run.
- **Times here are `America/Sao_Paulo` (-03)** and the crontab's comments give
  both local and UTC. State both when adding a line, and check the new time does
  not collide with the existing `:05/:15/:25/:35/:45/:55` pattern.

## Before you say a cron change is done

- The snapshot exists in `~/.crontab-backups/`.
- `crontab -l | diff <snapshot> -` shows only the intended change.
- Every pre-existing line is still present, including comments.
- The new line names an absolute interpreter and an absolute script path.
- You told the user what the schedule is in **both** local and UTC.
