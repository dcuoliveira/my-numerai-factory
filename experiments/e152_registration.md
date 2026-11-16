# E152 — strict-vintage admission and volume-alpha evaluation

Registered 2026-11-12 before any real-source model fitting or scoring. E151's
fetchers and registration are separate existing work in the shared checkout.
This study implements the user's stricter historical information boundary. A
modern archive download, volatility-based identity gate, or an added publication
lag is not proof of a historical vintage or permanent identity.

Dataset requested: `crypto/v2.0-volumealpha-causal`. The strict builder refuses
to overwrite any existing output and refuses to produce an all-missing candidate.
No result or claim is inferred from synthetic integration tests.

## Admission

Only evidenced original-publication/revision timestamps, dated permanent
chain/contract identity mappings and contemporaneous quote/USD conversions are
admitted. Revisions are selected as-of each cutoff, never selected retrospectively.
Price data through D-1 at D 00:00 UTC only. This is deliberately more conservative
than submitting later during D. No backfill or staleness carry for primitive daily
flows. Sources remain separate: CEX, DEX and combined; per-venue allowlists produce
additional CEX ablations. Market-wide totals explicitly mean the observed source
volume of then-known Numerai members, not a complete crypto-market census.

## Features and auxiliary forecast layer

`volumealpha_features.FAMILIES` fixes definitions before any result: eight paper
technical columns, separate liquidity extensions, size, fundamental surprises,
leverage, valuation, Dimson beta, verified-observation age, announced events and
cross-asset interactions. Optional unavailable sub-families stay missing and
their arms are reported unavailable; they do not remove base rows. Missingness
and source-coverage flags are sidecar metadata only.

Continuous inputs are winsorized using expanding strictly earlier dates, then
cross-sectionally rank-binned on the dated panel with ties preserved. No exact
bin-mass requirement overrides ties.

Volume forecasts: expanding pooled training, refit every 22 calendar dates,
minimum 130 training dates / 500 rows, seed 42. Ridge, histogram GBM and the
32/16/8 MLP are trained on the eight continuous technical inputs. Each target
has its own publication-time purge. Eta is one-day log-volume surprise versus
the previous five days. The five-day and 20-calendar-day challenge targets are
direct `log1p(sum(dollar_volume))`, not sums of overlapping one-day forecasts.
The 20-day horizon is an explicit auxiliary scenario, not an assertion about
Numerai's return-label start/end dates. Lower-tail q10 uses pinball loss;
probabilities of the causal own-history 10th/90th percentile use classifiers.
Disagreement uses the three OOF mean forecasts. Uncertainty is trailing realized
OOF log-volume error std (at least 20 resolved errors). Previous errors are used
only once their label has been released. No fitted training predictions enter.

Costs are proxies: inverse predicted volume, lambda=0.2/volume, low-tail risk,
capacity, expected-Amihud proxy, q10 and downside forecasts. Fixed USD trade-size
scenarios 1,000 / 10,000 / 100,000 remain separate. Their rank equivalences are
explicit, not treated as independent discoveries.

## Return evaluation

- Repo name mapping: requested `ridge_ktz` → `ridge`, `gbm_ktz` → `lgb_ktz`,
  `mlp_ktz` unchanged (identity-activation MLP). No new aliases are added globally.
- All three model classes; seeds 0–4. Rolling train/purge/refit = 130/22/130
  Numerai periods, first prediction after train+purge. Fifty candidate draws,
  five purged chronological partitions. Candidate draws and fold DATE boundaries
  are identical for B_ALL, B_COV and the candidate at each refit.
- B_ALL fits the full universe. For each arm, B_COV and candidate fit the exact
  same causally covered rows. A thin training block falls back to B_ALL; it is
  never dropped from scores. Both candidate and B_COV are routed through B_ALL's
  rank slots. Every output explicitly includes baseline_prediction, coverage and
  the full-universe routed prediction.
- Arms: technical, liquidity, fundamentals and their sparse sub-families, events,
  cross-asset, forecasts, costs, all raw, all derived, full, dev-selected PARSIM,
  separate trade-size scenarios. Source ablations use separately built datasets.
- Per-period official-tool CORR and proxy MMC against the historical meta model;
  payout = 0.1 CORR + MMC. Paired HAC lag 20 and 500 moving-block bootstrap draws
  of length 20; win rate, worst block, coverage, seed spread, payout Sharpe,
  drawdown, calmar and negative-payout date count. Market-cap and liquidity bins
  are formed from known-at-cutoff features. Compare raw vs derived only on an
  explicitly common covered population; differently covered arms alone cannot
  answer whether the auxiliary model adds information.
- Forward test begins **2026-11-13**, unrevealed at registration. It can only be
  evaluated after targets mature and a dev-only selection is frozen. The reused
  historical 2026-02..08 block is not described as untouched. No forward target
  is read in `--phase dev` (parquet predicate pushdown).
- PARSIM admits a family when mean seed HAC t on matched routed CORR ≥2 for at
  least two models and mean seed payout t >−2 for every model. Complete registered
  model/seed results are required. This is a dev screening rule, not an overall
  significance test. Multiple comparisons and fit noise must be reported.

## Initial source status

The local Binance and on-chain source audit is saved under
`results/reports/crypto/volumealpha_causal/source_audit.json`. At registration,
the sampled files lack the necessary vintage/identity/FX evidence fields. This
blocks strict dataset materialization and real-data fitting, not implementation
or synthetic validation. No return-alpha, execution improvement or null-result
claim can be made from unavailable source data.


## 2026-11-13 user-authorized reconstruction workaround

Before building or fitting real inputs, the user explicitly authorized alternative sources and ways of inferring provenance. The strict builder remains strict. `build_volumealpha_inferred` materializes the requested version with status `BUILT_INFERRED_RECONSTRUCTION`, never `BUILT_VERIFIED_INPUTS`. This is a transparent relaxation of historical-vintage proof, not a claim that inference establishes original publication.

Sources: existing Binance archives including delisted pairs and other CEX caches; new Coinbase direct USD stablecoin candles, with existing measured Yahoo USD rates as a labeled fallback, never a $1 peg assumption. Original retrieval dates and archive references remain in the observation store. Inputs through D-2: one full safety day beyond Binance's documented next-day publication. Identity candidates require a unique ticker-to-CMC relation (ambiguous tickers excluded), permanent CMC IDs / current chain-contract metadata where present, and a causal expanding volatility cross-check against that coin's own shipped features. This inference is explicitly not historical proof. Revised fundamentals are labeled and separable; events without dated announcements and other unavailable fields are named gaps, never fabricated.

The OOF budget is frozen before fitting: expanding refit every 90 calendar dates, seed 152, 50 iterations, at most 50,000 earlier rows per fit (deterministic uniform subsample), at least 130 training dates / 500 rows. Ridge, GBM and 32/16/8 MLP for original one-day eta and direct one/five/twenty-day log-dollar-volume targets; one-day volume LEVEL is directly fitted, not obtained using an unobserved D-1 baseline. Label publication purge follows the same two-day assumption. Lower-tail quantile/classification and error features remain OOF.

Outputs: main reconstructed dataset plus raw technical source ablations for Binance-only, other-CEX-only, and Coinbase-direct-USD-conversion-only. Every base row and column is preserved. No return-alpha claim or feature comparison is implied by a successful dataset build. A future comparison must explicitly admit this inferred mode and carry its limitations in its report.
