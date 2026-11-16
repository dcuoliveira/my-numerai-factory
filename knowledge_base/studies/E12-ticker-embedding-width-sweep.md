# E12 — How wide should the ticker embedding be?

- **State:** done (6/6 arms, 6 scout)
- **Runner:** experiments/e12_embed_dim_crypto.sh
- **Models / datasets:** torch_mlp_emb at pinned embed widths 4/8/16/32/64 on crypto/v2.0-symid, vs torch_mlp with no embedding module (crypto/v2.0)
- **Artifacts on disk:** 27.5 MB across 6 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/torch_mlp_emb/e12_emb32_raw.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Six torch MLP arms on raw 22 differing only in embedding width — none, 4, 8, 16, 32, 64 —
each PINNED with --model-options while alpha, architecture, lr and batch size are still
searched. Where E11 put the width in the grid and let one search pick, this asks it as six
models to get a curve, five paired deltas and whether it degrades gracefully. The control
has no embedding module at all, not a zero-width one.

## What was tried
6 arms: emb0 (torch_mlp, no embedding module, crypto/v2.0, n_features 22), and
emb4/emb8/emb16/emb32/emb64 (torch_mlp_emb on crypto/v2.0-symid, n_features 23) with the
width pinned via mlp__embed_dim. 514 periods, 20-day label, real search over the rest of
the grid (n_iter 50, n_splits 5).

## Trained parameters
From results/metrics/crypto/torch_mlp_emb/e12_emb32_raw.json: model torch_mlp_emb,
regression, target_binned_return_20, data_version crypto/v2.0-symid, n_features 23,
cv_scheme rolling, n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50, embed_dim pinned to 32 (searched size 1 on that axis); refit-0 chose
alpha 19.4149, batch_size 2048, hidden [64,32], lr 0.005541. emb0 is the plain torch_mlp
on crypto/v2.0 with the same searched grid.

## What was found
No written finding; from the metrics (514 periods) no embedding width beats the no-embedding
control, and the curve is non-monotone. emb0 pays 0.076618 (CORR 0.144701), the best;
emb8 0.076595 (CORR 0.142499) ties it; emb4 0.075108 and emb32 0.076024 are close behind;
emb64 drops to 0.067754 (CORR 0.130814); and emb16 collapses to 0.052463 (CORR 0.109969).
The embedding does not degrade gracefully — the width 16 arm is far worse than its
neighbours — and does not earn its place at any width. No paired t-stats in available
artifacts.
