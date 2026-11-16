---
name: numerai-crypto-missing-values
description: Design, build, extend, rebuild, or test Numerai Crypto datasets with potentially missing features. Use for incomplete external asset/date coverage, feature imputation, coverage policy, candidate training, or live prediction from partially covered features. Scoped to Numerai Crypto datasets.
---

# Numerai Crypto missing values

Apply this standard before changing a dataset or fitting or predicting with a
partially covered feature arm. Inspect the builder, training, live prediction
and evaluation entrypoints first. Use the [audit skill](../numerai-crypto-missing-values-audit/SKILL.md)
before accepting or interpreting a result.

## Preserve the panel and missingness

- Keep unavailable candidate values as `NaN` in stored datasets. Preserve genuine
  zero activity as zero. Never replace unavailable values with zero or 0.5 in
  the artifact, or backfill them from later observations.
- Forward-fill only from information genuinely available at the time and
  still valid under an explicit, source-specific staleness rule.
- Preserve every Numerai asset/date key when attaching candidate features.
  Remove or mark unavailable invalid, mis-mapped or unavailable external
  series. Use stable identifiers such as chain and contract address where
  possible, and audit ticker changes, reuse and mapping validity dates.
- Exclude retrospective coverage flags from production inputs. Examples
  include `is_missing`, `has_feed`, `never_observed`,
  `available_in_current_vendor_database`, present-day listing status and
  observation counts reconstructed from current sources. Use such fields only
  as labelled diagnostic placebos, isolated from model and feature selection.

## Define point-in-time coverage

For each arm and asset/date, set `M(i, D) = 1` only when every feature required
by that arm was available by its prediction cutoff, including its history and
staleness requirements. Keep the mask as routing and evaluation metadata.
Define coverage independently for each feature family and history window.
A spot arm needs no derivatives coverage, a complete volatility arm can
require every estimator, and a 20-day window can become eligible before a
120-day window. Test shorter histories in separate arms with their own
definitions.

Record for each external source its permanent asset identifier, source,
source and retrieval timestamps, dataset or API version, prediction cutoff,
as-of join rule, maximum staleness, first and last valid observation, and
coverage by date. Record raw-input hashes where practical. Prefer archived
snapshots or append-only raw stores. A cut-date replay of data downloaded
today can establish calculation causality. Historical availability requires
separate evidence.

## Train the three models

| Model | Inputs | Eligible rows |
| --- | --- | --- |
| `B_all` | v2.0 features | Full eligible Numerai universe |
| `B_covered` | v2.0 features | Candidate-covered sample |
| `F_covered` | v2.0 plus candidate features | Exactly the same covered sample |

Require `B_all` predictions for every live asset. For the two covered models,
match training and evaluation asset/date keys, folds, purge and embargo,
hyperparameter-search budgets, applicable seeds and downstream routing.
Exclude uncovered rows from candidate fitting and prediction. An equivalent
masked loss is allowed if uncovered rows cannot influence the learned model
through loss, preprocessing or shared statistics. If an estimator needs
neutral imputation internally, establish and enforce these exclusions first.
An all-neutral candidate block must never contribute to candidate learning
or prediction.

## Route and score as deployed

For each date, run the same procedure historically and live.

1. Predict with `B_all` and rank the complete universe using the project’s
   Numerai-compatible tie treatment. The live rank helper is `rank_normalise`
   in [src/live/submit.py](../../../src/live/submit.py).
2. Find the baseline rank positions occupied by assets with `M(i, D) = 1`.
3. Order only the covered assets using the covered model’s predictions and
   the same deterministic tie rule. Assign them back to that exact set of
   positions in order. Keep every uncovered asset at its original baseline
   rank. An empty covered set returns the complete baseline.
4. Emit one finite prediction per eligible asset. The file must explicitly
   carry baseline ranks for uncovered assets. Check that the final scoring
   or serialization step preserves the assigned ranks.

Run the identical mask, cutoffs, exclusions, ranking, routing and output
construction in backtest and live. Report absolute performance and the
paired per-date difference `Route(F_covered, B_all) - Route(B_covered, B_all)`
on the same dates and scored target. An untouched baseline can be additional
context. The two routed arms provide the marginal feature comparison.

## Controls and acceptance evidence

- Run cut-date reconstruction and wrong-asset and ticker-reuse checks.
  Compare the dataset keys to the base panel and verify candidate loss and
  predictions exclude uncovered assets.
- Check coverage stability over time and report results by coverage cohort.
  Compare routed and unrouted results, labelling their sample and purpose.
- For coverage-sensitive claims, run a causal warm-up control using only
  information available on each date. Include a retrospective
  `never_observed` placebo and a current-vendor-coverage placebo as diagnostic
  controls. Keep their results out of model and feature selection.
- A look-ahead placebo shows that a survivorship channel can be exploited.
  Its score is never a correction to subtract from candidate performance.
- Verify exact uncovered ranks, preservation of the covered rank-slot set,
  complete finite output, ties, and empty and full coverage. Compare a
  historical replay with the live path on the same snapshot before acceptance.
