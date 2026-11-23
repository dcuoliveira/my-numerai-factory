---
name: tsfm-embedding-standardization
description: Apply the user's default downstream embedding H standardization and design C when designing or configuring Chronos, t0, TabPFN-TS or Timestral forecasting experiments.
---

# Forecasting model embedding standardization

Invoke with `/tsfm-embedding-standardization`. See [repository instructions](../../../CLAUDE.md).

Use the following user-recommended defaults for downstream embeddings H:

| Model family | Recommended H standardization |
| --- | --- |
| Chronos (including Chronos-2 / `chronos2`) | `rolling` |
| t0 | `rolling` |
| TabPFN-TS / `tabpfnts` | `asset` |
| Timestral | `robust` |

- `rolling`: standardize each asset and coordinate using only its prior
  embedding observations, excluding the current observation. Follow the
  experiment's declared lookback and minimum history; select any window using
  only past validation data.
- `asset`: standardize each coordinate using that asset's training mean and
  standard deviation; freeze those statistics for its forecast block.
- `robust`: standardize each coordinate using the pooled training median and
  interquartile range divided by 1.349, matching the existing implementation;
  freeze those statistics for its forecast block.

Use **design C** by default for H: concatenate the standardized embeddings
across a fixed training asset universe, using consistent asset slots and each
asset exactly once, as input to each per-asset prediction model. This design
preference applies alongside the model-specific standardization above. Explicit
user choices and requested design comparisons take precedence.

Fit statistics on eligible, covered observations only. Preserve missing
embeddings and warm-up eligibility, and follow the existing zero-scale handling.
The reference implementation is `src/analysis/tsfm_momentum_study.py::normalize`.
Record the chosen method, window/minimum history where applicable, and training
cutoff in the experiment configuration and report.

Apply this recommendation to downstream H processing, preserving the pretrained
model's native input normalization. Explicit user instructions and requested
standardization sweeps take precedence. Treat these defaults as user preferences,
not a universal performance finding. Do not alter an already registered or
running experiment's configuration without a request to do so.

For a user-requested complete dataset-to-report study, use
[`/tsfm-signal-experiment`](../tsfm-signal-experiment/SKILL.md). A request to choose
standardization alone does not imply launching that full experiment.
