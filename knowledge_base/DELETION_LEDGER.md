# DELETION LEDGER — knowledge_base
Generated from `src.analysis.retention` (authoritative) + a 7-day age guard + sibling-safety.
All sizes are ON-DISK (st_blocks); the quota frees on-disk bytes, not apparent size.
Cutoff for "past week": 2026-10-17 03:20.

## Guardrails enforced
- **No live-model results:** retention K1 (deployed-slot backtests + member closure) are KEPT. The
  flagged production studies E32/E33/E47/E102 are NOT in the delete set (verified).
- **No past-week results:** 24 recent scout files (E115/E116/E119/E127) that retention would
  auto-include are EXCLUDED by the age guard. All 56 named studies are fully >7d old.
- **Sibling safety:** a forecast is deleted only if its metrics AND per_period siblings remain.
- **Collision safety:** study_of is model-aware (e13_adl/ridge → None), so prefix collisions
  (E13, E28/E29) cannot drag an unrelated study's files in.

## What will be deleted
| set | files | on-disk | apparent | note |
|---|---|---|---|---|
| forecast (56 older Tier-A studies) | 1390 | 5.17 GB | 12.23 GB | via retention --apply, age+sibling guarded |
| diag (campaigns >7d old) | 19 tags | 1.33 GB | ~same | re-derivable; recent campaigns (22 GB) protected |
| **TOTAL** | | **6.49 GB on-disk** | | |

## What is KEPT (and why)
- 67 deployed-slot forecasts (K1) — production backtest references
- forecasts named by a published report / cited in code (K2/K5) — report re-render + reproduction
- target-record forecasts (K3) — the config-silent ones
- all metrics / per_period / hyperparams / reports / cache / live — the ground truth the KB cites
- the 19 recent studies (E106–E127) and 4 active torch campaigns — past-week work

## The retention plan for the forecast set
Named studies: E01 E02 E03 E08 E10 E15 E16 E17 E19 E20 E24 E42 E43 E48 E49 E51 E52 E57 E59 E60 E64 E65 E66 E67 E68 E69 E70 E71 E72 E73 E74 E75 E76 E77 E78 E79 E80 E81 E84 E86 E87 E88 E89 E90 E91 E92 E93 E94 E95 E96 E97 E98 E99 E100 E101 E103
(retention plan hash is regenerated at apply-time; the age guard is applied on top.)

## Executed — 2026-10-24
Moved (not unlinked) out of the quota to **`/scratch/danielco/kb_quarantine_20261024/`**
(separate NFS export, outside the per-user `/home` quota), so the space is freed and the files
remain recoverable:

| | count | on-disk |
|---|---|---|
| forecasts (56 studies older than a week) | 1,390 files | 5.17 GB |
| diag tags (campaigns older than a week) | 19 tags | 1.33 GB |
| **freed from `/home`** | | **6.49 GB** |

Skipped at move time: 24 recent scout forecasts (E115/E116/E119/E127). Every guard was
re-checked per file at the moment of the move.

Per-file log (model, run, study, scored target, bytes):
`results/retention/kb_quarantine_20261024T075550.json` (copy: `MANIFEST.json` in the quarantine dir).

Verified afterwards: `python3 -m src.analysis.slots` still reproduces 34 of 37 slots (the 3 documented
exceptions unchanged); retention still counts all 67 deployed-slot forecasts; no moved run is a slot
`backtest_run`, from E32/E33/E47/E102, or from a past-week study; every moved run kept its metrics and
per_period siblings; test_report_pdf / test_report_html / test_experiment_db show only the 4 failures
that predate the move.

**Restore one run:**
`mv /scratch/danielco/kb_quarantine_20261024/forecast/crypto/<model>/<run>.csv results/forecast/crypto/<model>/`

**Purge permanently** (once you are satisfied nothing is missing):
`rm -rf /scratch/danielco/kb_quarantine_20261024`

> **Update 2026-10-25 — the 2026-10-24 quarantine no longer exists.** By 04:43 on 2026-10-25 the whole
> `/scratch/danielco/kb_quarantine_20261024/` directory (the 1,390 forecasts, the 19 old diag tags and its
> MANIFEST.json) had been removed — a whole-directory delete, not a /scratch age purge (other files there
> back to 2026-09-06 are untouched). No copy exists elsewhere, so the "Restore one run" command above no
> longer works. What was in it is listed in `results/retention/kb_quarantine_20261024T075550.json`;
> each run's metrics, per_period and hyperparams files remain in `results/`.

## Executed — 2026-10-25: all of results/diag
The 102 remaining diag tags (the four torch walk-forward campaigns, written 2026-10-16 to 10-21; none
started this weekend) were moved to **`/scratch/danielco/kb_quarantine_20261024/diag/crypto/`**:
**21.97 GB freed**, `results/diag/crypto/` is now empty. Before the move, every tag's `config.json`,
`log.json`, `diagnostics.png`, `meta_scores.npz` and the six `*_walkforward/` read-outs were copied into
`knowledge_base/diagnostics/` (104 MB on disk). Verified after the move: all 102 tags present in /scratch
at full size, none nested, all present in the knowledge base. Per-tag log:
`results/retention/diag_quarantine_20261025T045307.json`.

Restore one tag: `mv /scratch/danielco/kb_quarantine_20261024/diag/crypto/<tag> results/diag/crypto/`
Purge permanently: `rm -rf /scratch/danielco/kb_quarantine_20261024`

## Purged — 2026-10-25
`/scratch/danielco/kb_quarantine_20261024/` (the 102 diag tags, 21.97 GB) was deleted at the user's
request. Nothing from either removal remains on disk; the per-file logs in `results/retention/` and the
summaries in `knowledge_base/diagnostics/` are the record. Any raw array or forecast needed again has to
be regenerated by re-running its study.
