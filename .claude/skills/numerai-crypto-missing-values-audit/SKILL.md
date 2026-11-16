---
name: numerai-crypto-missing-values-audit
description: Audit Numerai Crypto feature datasets with missing values, validate candidates before acceptance, interpret missing-value experiments, investigate unusually strong results associated with coverage, or check backtest and live parity for missingness and routing.
---

# Audit Numerai Crypto missing values

Read and apply the [implementation standard](../numerai-crypto-missing-values/SKILL.md)
before completing the audit. Inspect the relevant builder, candidate fitting
and prediction code, live entrypoint, stored dataset and forecasts, and report.
Separate observed evidence from gaps in artifacts or provenance. A report’s
description alone does not establish execution parity.

| Area | Evidence to check |
| --- | --- |
| Panel and mapping | Compare base and candidate asset/date keys. Check that unavailable or incorrect external series affect features only. Inspect stable identifiers, ticker reuse, and mapping validity periods. |
| Storage and fitting | Inspect missing blocks in the stored artifact and the order of masking and imputation. Check candidate inputs and loss, including preprocessing, for contributions from uncovered rows, explicit coverage flags, or all-neutral blocks. |
| Coverage provenance | Check archived or append-only source evidence, cutoff and staleness rules, current-vendor reconstruction, and whether short histories replaced long-window values. |
| Matched comparison | Verify identical covered keys, folds, purge, embargo, search budgets, applicable seeds and routing for `B_covered` and `F_covered`. Check that retrospective placebos stayed out of selection. |
| Routing and parity | Replay representative dates through both paths. Verify matching eligibility, cutoffs, exclusions and ranking. Assert the covered rank positions are preserved as a set, uncovered assets retain their exact baseline ranks, and all assets have explicit finite predictions. Check the report scores those routed outputs on matched dates and targets. |

For each finding, cite the file, artifact or replay evidence, state the effect
on the claim, and identify any correction or rerun needed. Use these severities.

- **Blocking** invalidates the claimed marginal feature result.
- **Material** could change the conclusion and requires correction or rerun.
- **Advisory** improves robustness and does not by itself invalidate the result.

End with exactly one outcome and explain the evidence and any unresolved
conditions. Use `PASS` for demonstrated compliance, `PASS WITH CONDITIONS`
when the conclusion is supportable under stated conditions, `RERUN REQUIRED`
when a correction or missing validation needs a new run, or
`INVALID FOR MARGINAL CLAIM` when the available result cannot support marginal
attribution.
