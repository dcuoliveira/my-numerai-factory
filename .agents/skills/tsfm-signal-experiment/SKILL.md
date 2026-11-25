---
name: tsfm-signal-experiment
description: Run the TSFM signal experiment on a user-provided Numerai Crypto dataset, including frozen per-asset hidden states, fresh ridge heads for 20d/60d, lagged-meta prediction residualization, and exhaustive 2–5-member ensembles. Use for requests by this name or descriptions of the E206–E211 workflow on another signal dataset.
---

# TSFM signal experiment

Invoke with `$tsfm-signal-experiment`, or ask to run the **TSFM signal experiment**
on a named dataset. For example:

> Run the TSFM signal experiment on crypto/my-selected-signals with Chronos-2,
> t0 and Timestral.

This is an agent workflow, not a generic command-line launcher. Read
[the execution protocol](references/protocol.md) before implementation. Read
[the review of the original request and runners](references/review.md) when
adapting an existing experiment. Creating or editing this skill does not launch
a study. A request to run it authorizes preparing, registering, implementing,
verifying and launching the specified experiment within the applicable resource
policy; do not stop at a plan when the necessary inputs are available.

## Resolve the study

Use the dataset supplied by the user; inspect its manifest and repository paths.
If the dataset cannot be identified, ask for that missing identifier. Never
substitute momentum or copy its feature list, dates, source masks or study IDs.
Use explicitly selected foundation models; if unspecified, default to Chronos-2,
t0, TabPFN-TS and Timestral. Include a feature-only per-asset ridge baseline on
the supplied dataset. Preserve user-specified models, dates and overrides.

Use the existing
[`$tsfm-embedding-standardization`](../tsfm-embedding-standardization/SKILL.md)
defaults: Chronos/t0 `rolling`, TabPFN-TS `asset`, Timestral `robust`, and
**design C**. Use
[`$numerai-crypto-missing-values`](../numerai-crypto-missing-values/SKILL.md)
before data/fitting changes and
[`$numerai-crypto-missing-values-audit`](../numerai-crypto-missing-values-audit/SKILL.md)
before interpreting results. Register and commit the experiment before launch
as required by [AGENTS.md](../../../AGENTS.md).

## Run the complete workflow

1. Freeze dataset/feature selection, label availability, common causal coverage,
   model checkpoints, extraction specification, fold schedule and search grids.
   Estimate the run and show the actual trial count; see the protocol's counting
   rule. Inspect GPU availability and prefer GPU extraction under current
   authorization. Do not inherit an old study's resource exception automatically.
2. Generate each selected model's frozen pre-head H independently for each asset
   and for each training target (20d and 60d). Native TabPFN-TS contextual fitting
   is allowed; pretrained weights remain frozen. Batch independent origins only
   within the same asset, with compatible horizons. Audit causality and batching.
3. Apply the recommended H standardization. Fit fresh ridge models separately
   for every predicted asset, foundation model and training target, using design
   C and only resolved past labels. Also fit the matched feature-only baseline.
   Aggregate chronological forecast blocks into one full-history evaluation.
4. Retain raw and once-residualized predictions for every standalone stream.
   Use the shared lagged-meta method, strength 0.5 and snapshot at least five
   calendar days old. Train on original targets; do not residualize labels.
5. Evaluate every standalone on **both** target20 and target60 on identical
   common assets and scored dates. Build the payout-sorted summary, daily-payout
   correlation heatmap and cumulative payout plots, separately by evaluation
   target. Preserve full-panel routed forecasts and coverage diagnostics.
6. Enumerate every 2-, 3-, 4- and 5-member subset of the raw/residual standalone
   pool. Average within-date prediction ranks equally and freshly rescore each
   blend. Also produce a **separately residualized version of every ensemble**
   using the same lagged-meta method; keep both versions. This final ensemble
   transform is deliberate even when members include residualized streams.
   Apply it once, without iterative residualization.
7. Consolidate all standalone and ensemble trials, including failures and
   equivalent configurations. Sort within evaluation target by payout descending.
   Plot the **top three ensemble rows by payout plus all standalone variants**
   for each evaluation target. Label selection on the displayed history as
   exploratory. Deliver HTML/PDF, complete machine-readable tables, daily scores,
   forecast artifacts, coverage, configuration, hashes, logs and audit outcome.

Required summary columns include model/members, training target(s), evaluation
target, raw/residual view, H standardization/design, number of assets/dates,
**corr, mmc, payout, sharpe.p, maxdd.p**. Use cumulative **payout scores**, not
price-return Sharpe, price drawdown or compounded investment returns. The exact
metric definitions and completion checks are in the execution protocol.

Do not simply execute `experiments/e211_cross_target_tsfm.py` on another dataset:
it is momentum-specific, fixes the model pool and cutoff, lacks the extra
ensemble residualization pass, and plots top five rather than this skill's top
three. Adapt/reuse verified numerical components in a study-specific configurable
runner; do not modify the configuration of an unrelated running experiment.

Omit the Frozen protocol section and raw config/manifest dumps from HTML/PDF
reports. Save those details in separate configuration and audit artifacts;
retain concise explanations needed to interpret the tables and plots.

### Optional extraction host outside the shared pool

Westeros can be considered when authorized GPU work needs another host. Read
AGENTS.md's "Optional server: westeros" section (Codex) or CLAUDE.md's matching
section (Claude Code) for access and synchronization. It has independent storage;
commit/push/pull code, transfer and hash-check data/checkpoints/results explicitly,
and configure host-local environments and output paths. Do not add it to an
existing NFS-based scheduler without implementing separate-host handling.
Retain registered study IDs, resource accounting, execution-device manifests and
cache identities; verify GTX 1080 Ti/Pascal compatibility with a registered smoke
before real extraction. Documentation alone does not move a running study.
