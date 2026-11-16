---
name: crypto-feature-binning
description: "Reproduce Numerai's cross-sectional rank-binning so a feature you build lands on the exact same {0,.25,.5,.75,1} scale as the 22 shipped Crypto features (5/20/50/20/5 mass, per date). Use whenever quantizing, binning, ranking or normalising a computed feature for Numerai, matching the shipped feature scale, checking why a feature has only five values, or debugging a feature that a booster/ridge treats differently from the native ones. The one place src/data_mani/binning.py's rule lives. Triggers on 'bin my feature', 'quantize like Numerai', 'rank-bin', 'match the feature scale', '5 buckets', 'why 0.25/0.5/0.75'."
---

# Numerai's feature quantization, and how to reproduce it

## What the shipped features are

Numerai does not ship raw signals. Every Crypto feature arrives **rank-binned WITHIN each
date** to five values `{0, .25, .5, .75, 1}`, mass split fixed at **5 / 20 / 50 / 20 / 5
percent** — the two tails hold 5% each, the middle bin holds half the field. This is a
property of the download, computed on Numerai's servers; the repo never had to reproduce it
until it started building features from raw prices. Verify on the shipped data:

```python
import pandas as pd
df = pd.read_parquet("data/crypto/v2.0/train.parquet")
df["feature_momentum_20d"].value_counts(normalize=True).sort_index()
# 0.00≈.0501  0.25≈.200  0.50≈.500  0.75≈.200  1.00≈.0501   (symmetric tails)
```

On a single 300-symbol date the counts are exactly **15 / 60 / 150 / 60 / 15**. All 22
features share this; both targets too. Consequences that make the binning non-optional for
any feature you add:

- A model reads RANK, not level — the value is a within-date rank, only meaningful against
  the other coins on that same date, and not comparable across dates.
- `fill_missing_features` fills gaps with the neutral **0.5** because that is literally the
  middle of the field; a feature not on this scale breaks that meaning.
- There is no `StandardScaler` in the pipeline (`prediction/functions.py`) precisely
  because the features are already on a common rank-binned scale.

## The rule (src/data_mani/binning.py)

Within one date, rank the symbols by the raw signal, then hand the tails **exactly**
`round(0.05·n)` symbols each and the shoulders `round(0.20·n)` each, the rest to the
middle:

```
lowest round(.05 n) → 0.00 ; next round(.20 n) → 0.25 ; middle remainder → 0.50 ;
next round(.20 n) → 0.75 ; top round(.05 n) → 1.00
```

Symmetric COUNTS, not a one-sided fraction cut (which rounds the two tails differently) and
not `qcut` on the raw values (which breaks on ties) — this keeps the two 5% tails equal
exactly as Numerai's are, at any n. A NaN raw signal stays NaN (→ neutral 0.5 at load).

```python
from src.data_mani.binning import rank_to_bins, bin_cross_sectional, bin_mass, BIN_LABELS
out = bin_cross_sectional(df, ["feature_myfeat"], period_col="date")   # per-date, in place on a copy
bin_mass(out["feature_myfeat"])   # must be ≈ [.05,.20,.50,.20,.05], max drift < 0.01
```

`rank_to_bins(values)` bins one cross-section (the symbols on one date). `bin_cross_sectional`
applies it per period and casts to `float32`. `bin_mass` is the validator.

## Causality

The bin is **cross-sectional and contemporaneous**: symbol s's bin at date t uses only the
other symbols' signals AT t — the same information set the scorer uses (rank within an era).
It adds no look-ahead; ensuring the RAW signal fed in is backward-looking, and merged
point-in-time, is the caller's job (see [crypto-derived-features]).

## Live reuse

`bin_cross_sectional` is written so the live path calls it with `df` restricted to one
round's cross-section — same function, same scale — so a feature validated offline bins
identically live. Do not re-implement the rule anywhere else; import it.

## Traps

- **Binning over the wrong universe.** Bin over the tournament PANEL's symbols for that
  date, after merging, not over the source feed's coverage.
- **Two values, not five.** A sign/boolean signal cannot fill five bins — carry the
  underlying continuous quantity and let the rank spread it (e.g. cumulative return, not
  `sign(return)`).
- **Expecting exact per-date symmetry at odd n.** `round(0.05·n)` can differ by one symbol
  between tails at some n; pooled tails still match Numerai's to < 0.01.
