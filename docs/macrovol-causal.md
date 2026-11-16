# Macro volatility comparison

The requested dataset already existed as a retrospective prototype. Its histories do not establish historical availability, and its gold input is GLD, an ETF proxy. The original E150 also neutral-imputed inactive high/low gates, and its old test dates were evaluated by earlier studies. Those results cannot satisfy the strict acceptance criteria.

The observed-only rerun uses separate corrected `-retrospective` datasets, explicit complete-feature masks for training and inference, matched v2.0 and vol-causal comparators, and slot-preserving routing. High/low gates are separate experimental arms. Daily coin returns are calculated before aligning to macro trading days, avoiding weekend accumulation against Monday-only macro changes. Existing datasets and prediction outputs are preserved. The legacy final block is excluded from fitting and scoring; this run is development-only exploration.

Run preparation with `experiments/e150_observed_prepare.py`. Launch `experiments/e150_observed.jobs` through `python -m src.cluster plan` followed by `python -m src.cluster run`. Twelve model/group batches run fixed seeds and identical six-candidate inner searches. The finishing job waits for their completion, selects a parsimonious combination on development data, scores routed predictions and writes CSVs and a PDF under `results/reports/crypto/e150_macrovol_observed_dev/`. Batch states are in `batch_status/`; `COMPLETE.json` appears only after reporting succeeds. Missing-coverage arms have explicit `.unavailable.json` records.

## ALFRED path

Set `FRED_API_KEY` locally in the environment or `.env` and run:

```
python -m src.data_mani.fetch_alfred_macro
```

The adapter archives raw JSON, retrieval times, hashes, real-time periods and initial-release values for VIXCLS, SP500 and DGS10. The fixed revision policy deliberately uses initial releases rather than substituting later corrections. Date-only vintages are conservatively usable after the entire vintage day has ended in America/Chicago. This is an availability upper bound, not an invented market-close timestamp. The API key is never written to manifests or error URLs.

Gold spot/futures and ICE DXY still need separate historical sources. FRED removed its LBMA gold series, and DTWEXBGS is the Fed broad dollar index, not ICE DXY. Do not silently substitute GLD or DTWEXBGS in the strict dataset.

Combine frozen source observations into a parquet file satisfying `macrovol_contract.validate_store`, preserving identifiers, observation dates, availability bounds, source timestamp precision, retrieval timestamps, version IDs and raw SHA-256 hashes. Then use `python -m src.data_mani.build_vol_macrovol --macro <file>`. The strict builder refuses the old unverified store before replacing outputs. Publication-aware backward attachment enforces the D-1 cutoff and a four-calendar-day limit measured from the observation date. Late dependencies delay downstream signals rather than leaking through observation-date joins. Coverage by date is stored separately from model features.

Strict acceptance additionally requires a newly registered untouched test after definitions and selection rules are frozen. The inherited reconstructed crypto universe remains an explicit limitation even after macro vintages are available.

References: https://fred.stlouisfed.org/docs/api/fred/series_observations.html ; https://alfred.stlouisfed.org/help ; https://fred.stlouisfed.org/news/2022/01/ice-benchmark-administration-ltd-iba-data-to-be-removed-from-fred ; https://fred.stlouisfed.org/series/DTWEXBGS
