---
name: crypto-derived-features
description: "Build a NEW derived/engineered feature set for the Numerai Crypto pipeline that lands on Numerai's exact rank-bin scale, merges onto the panel POINT-IN-TIME (no look-ahead versus submission timing), ships as a --data-version drop-in, and is registered as a numbered study. Use whenever adding features computed from prices or any external source (momentum, volatility, on-chain, sentiment), quantizing a computed feature to {0,.25,.5,.75,1}, merging external data onto the panel without leakage, deciding what information is known at submission time, or turning a feature idea into a testable-and-deployable dataset variant. Triggers on 'add a feature', 'engineer features', 'build a dataset variant', 'merge external signals', 'bin my feature like Numerai's', 'is this look-ahead', 'make it production-ready'."
---

# Building a derived feature set for Numerai Crypto

The deliverable is always the same shape: a new `data/crypto/v2.0-<suffix>/train.parquet`
that is a **superset of the base panel** — every base row and column untouched, plus new
`feature_<name>` columns — that a predictor consumes unchanged via `--data-version
crypto/v2.0-<suffix>`, and that the live path can later reproduce exactly. Get four things
right and it is production-capable; miss any one and the backtest lies.

## The four rules, and the module that owns each

The pipeline is deliberately three importable seams so the OFFLINE builder and the LIVE
path share the same arithmetic (backtest == live by construction):

```
raw signal (prices/other) ─▶ compute (pure math) ─▶ attach as-of (timing) ─▶ bin (scale) ─▶ merge
   src/data_mani/momentum_features.py: compute_momentum_signals, attach_signals_asof
   src/data_mani/binning.py:           rank_to_bins, bin_cross_sectional
   src/data_mani/build_momentum.py:    the offline wiring + CLI  (the worked example)
```

Copy `build_momentum.py` as the template for a new builder; it does every step below.

### 1. Compute the raw signal backward-looking, per symbol

A feature at date t may read only data at or before t (it is observable at t). Compute per
symbol on the symbol's own sorted series — never let one symbol's history reach another
(group by symbol; the repo's builders all do). Windows are expressed **in observations**;
crypto trades daily so a positional window of w rows ≈ w calendar days, the same
convention as `build_rolling_features`. Require the FULL window (`min_periods=w`) so a
symbol with too little history yields NaN there rather than a short-window artifact — that
NaN becomes the neutral bin at load, exactly as the shipped 60-day features are NaN for
young tokens. Only forward-looking things (a target, a target encoding) need a resolution
lag; a trailing feature does not.

### 2. Bin to Numerai's exact scale — ALWAYS, and see [crypto-feature-binning]

Numerai ships every feature rank-binned within each date to `{0,.25,.5,.75,1}` at a fixed
**5/20/50/20/5** mass. A feature you hand a predictor un-binned is a second, silently
different feature language: a booster splits on raw magnitude, a ridge sees an
un-normalised column, and `fill_missing_features`' neutral 0.5 stops meaning "middle of the
field". So pass every derived column through `binning.bin_cross_sectional(df, cols,
"date")`. Validate with `binning.bin_mass` — every column must reproduce ≈ `[.05,.20,.50,
.20,.05]` (max drift < 0.01). **Bin AFTER the merge, over the PANEL's per-date
cross-section**, not over whatever universe the source covered — the bin is a rank against
the coins the model actually scores that day.

### 3. Merge POINT-IN-TIME, lagged by the submission constraint

This is the leak that does not crash — it just makes a better, unreproducible backtest.
On round date D you submit DURING day D, so an external daily feed's bar for day D is not
final yet: the newest usable close is **D−1**, the last completed bar. Join with
`attach_signals_asof(base, raw, cols, lag_days=1, max_staleness_days=4)` — an as-of
backward merge that gives row D the FRESHEST signal dated ≤ D−1 (so a Monday round uses the
weekend's move, which is real, known information), capped so a delisted token cannot carry
a months-stale value forward. An exact same-date join attaches close(D) to row D and leaks.
The offline builder must use the SAME lag the live path will have — that is what keeps the
backtest honest. (Numerai-native features already live at D's cutoff; only EXTERNAL sources
need the lag. A within-panel transform of already-shipped features, like a lag or rolling
mean, has no such gap — see `build_lagged`/`build_rolling_features`.)

### 4. Write the variant like every other builder

Read the base with `download_if_missing`, add `feature_*` columns, write with
`.set_index("symbol")` to `local_path("crypto", "v2.0-<suffix>", "train.parquet")`, and
**copy `historical_meta_models.parquet`** into the variant dir or `load_reference` /
`attach_meta_model` cannot find it. The Crypto loader (`utils._load_crypto`) selects
features by the `feature_` prefix, so new columns must be named `feature_...`. Keep a
config map of specs (name → definition) and a `--features name1,name2` subset flag so the
prediction team can prune; add `--drop-originals` to isolate the new block.

## Validate it is a real drop-in

```bash
PY=/home/danielco/miniconda3/bin/python3   # the interpreter with pyarrow — verify first
$PY -m src.data_mani.build_<name>                              # build the variant
$PY - <<'X'
from src.data_mani.utils import load_tournament_data
df, feats = load_tournament_data("crypto", version="crypto/v2.0-<suffix>")
print(len(feats), "features;", [f for f in feats if f.startswith("feature_")][-5:])
print("NaN after load:", bool(df[feats].isna().any().any()))   # must be False (neutral-filled)
X
```
Then a smoke forecast (give it enough periods — `--era-step 5 --train-window 520
--refit-every 6month --n-splits 3 --n-iter 3`; over-downsampling trips the embargo=20 CV
guard, which is a param problem, not a data one). Report base vs base+block with the
[arm-board-reporting] / [crypto-experiment-reporting] tools, paired, same seed — never two
means.

## Register it as a study (before it launches)

Per `CLAUDE.md`: add a `STUDIES` entry in `src/analysis/experiment_db.py` (id, title, idea,
runner, `prefix`, models, `expected_arms`) AND a row in `experiments/EXPERIMENTS.md`, and
write `experiments/e<NN>_<name>_crypto.sh` mirroring a recent runner (header stating what it
runs / measured against / what counts as a result; `DRY_RUN`/`ONLY`/`SKIP`/`SHARD`/`OF`/
`SMOKE` knobs; start+done email via `src.live.notify.send_text`). Ablate the new block with
`--feature-include '^feature_<name>_'` and `--feature-exclude` on the same dataset.

## Worked example and tests

The crypto momentum set is the reference implementation: `momentum_features.py`
(`MOMENTUM_SPECS`, 17 literature proxies), `build_momentum.py`, `test_momentum_features.py`
(pins the 5/20/50/20/5 masses, causality, and the submission-lag join), study **E63**. See
[crypto-price-data] for sourcing the spot OHLCV those features read, and
[crypto-feature-binning] for the quantization contract.

## Shared (macro / cross-asset) features — a different sub-case

A feature that is the SAME for every symbol on a date (gold, US yields, S&P — anything
not tied to a particular coin) breaks rule 2: it is constant across the cross-section, so
rank-binning within a date maps the whole column to 0.5. Normalise it across TIME instead —
`macro_features.causal_expanding_percentile` (fraction of past+present obs ≤ today, [0,1],
0.5 neutral, causal) — and broadcast it with a DATE-level as-of join
(`macro_features.attach_macro_asof`, no `by` symbol, wider staleness ~7d because macro is
business-day while crypto is 24/7). Reference build: `build_macro.py` →
`crypto/v2.0-momswap-macro`. **Honest caveat:** a within-date-constant feature has ~zero
standalone within-era `numerai_corr` (that metric is a within-date rank corr); it helps only
via regime-gating (trees/NN) or explicit macro×per-symbol interactions — say so when you add
one, and expect it to earn its keep through interactions, not alone.

## Traps

- **Skipping the bin.** The single most common mistake — an un-binned column looks fine and
  quietly changes what every model and the neutral-fill mean.
- **Binning over the source universe, not the panel.** Bin after the merge, per panel date.
- **Exact same-date merge of external data.** Look-ahead; use the as-of lag.
- **Forgetting the meta-model copy.** The variant loads but reference/BMC scoring breaks.
- **Wrong interpreter.** Bare `python3` may be the 3.8 sandbox without pyarrow; use
  `/home/danielco/miniconda3/bin/python3` and verify.
- **Missingness/coverage indicators leak survivorship.** A `feature_isna_*` or
  "has spot data" flag built from a coverage snapshot taken today marks coins
  that later died. E131 (fillind): helpful on dev, sharply harmful on test.
  Freeze coverage point-in-time (what was fetchable on each date) before any
  indicator ships, and evaluate with the placebo/tenure controls in
  `feature-set-evaluation`.
