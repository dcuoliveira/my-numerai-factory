# E19 — TabPFN (tabular foundation model) at defaults, per submitted dataset

- **State:** done (9/9 arms)
- **Runner:** experiments/e19_tabpfn_crypto.sh
- **Models / datasets:** `tabpfn` (Prior-Labs TabPFN v2 regressor, in-context, added to prediction/models.py). 9 datasets — 5 narrow: raw, tenc, interact, fenccausal, cumlag3; 4 wide (feature-reduced to fit TabPFN's ~500-feature range): tenc_onehot, onehot_roll, allfeat (crypto/v2.0-tenc-onehot-spread-interact), allfeat_pp.
- **Artifacts on disk:** 45.80 MB across 9 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e109-foundation-repr-block.md; metrics: results/metrics/crypto/tabpfn/e19_tenc.json

## Main question
Ask E18's question of a tabular FOUNDATION model instead of a trained net: TabPFN solves each dataset by in-context learning — no per-dataset training, no search, one forward pass — so it is the cheapest arm to add. Does it beat the deployed slot on each dataset?

## What was tried
Same 9 datasets as E18 (the 5 narrow are the default set; the 4 wide exceed TabPFN's range and get feature reduction). Each 39k-row block subsampled to a few thousand context rows (a recorded fidelity/cost tradeoff). Weights are the ungated v2 checkpoint vendored at models/tabpfn/ (the box has no egress). Each arm compared to the deployed slot on the same dataset, paired Newey-West, as E18.

## Trained parameters
From results/metrics/crypto/tabpfn/e19_tenc.json: model `tabpfn`, data_version `crypto/v2.0-tenc`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 23, seed 42, n_iter 1 (single forward pass), n_splits 2, val_frac 0.3. fixed_params `{n_estimators: 4}`. No per-dataset weights are learned.

## What was found
MEMORY-BACKED, with a caveat that matters for deletion. There is no EXPERIMENTS.md finding or paired t-stat for E19. From the metric JSONs the strongest arms are e19_interact CORR 0.149611 / MMC 0.064245, e19_tenc 0.147380 / 0.064106, e19_fenccausal 0.145042 / 0.066740 (best MMC), e19_raw 0.142992; the 4 wide reduced-feature arms are weaker (onehot_roll 0.126838, allfeat_pp 0.123792). Crucially, e109-foundation-repr-block.md records that E19's ~0.145 TabPFN CORR (its 4-refit number) "looks like a lucky context draw": a later causal TabPFN refreshing every 20 dates (E109) scored only CORR 0.1095 on the 510 dates, with per-date rank correlation to E19 of just 0.745 and block-mean sd 0.08. So E19's headline is not reproducible and should not be treated as an established TabPFN level. TUPA_01 (a BACKTEST_ONLY slot) points at E19.
