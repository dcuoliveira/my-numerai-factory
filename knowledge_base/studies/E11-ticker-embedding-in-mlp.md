# E11 — Does a learned ticker embedding earn its place in the MLP?

- **State:** done (2/2 arms, 2 scout)
- **Runner:** experiments/e11_symbol_embedding_crypto.sh
- **Models / datasets:** torch_mlp (no embedding) vs torch_mlp_emb (learned ticker embedding) on crypto/v2.0-symid (raw 22 + symbol id)
- **Artifacts on disk:** 8.8 MB across 2 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/torch_mlp_emb/e11_emb_raw.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Two torch MLP arms on raw 22 differing only in whether the network sees which symbol a
row is. The embedding is built the way prediction/nn/model.py builds the LSTM's — one row
per training symbol plus a zeroed padding row for unseen codes. Priors are against it:
36% of a block's 460 symbols have under 60 rows, and at embed_dim 32 the table is 4x the
dense stack.

## What was tried
2 arms: `e11_noemb_raw` (torch_mlp, raw 22, crypto/v2.0) and `e11_emb_raw`
(torch_mlp_emb, crypto/v2.0-symid, n_features 23 = raw 22 + symbol id), 514 periods,
20-day label, real search (n_iter 50, n_splits 5). The embedding width is in the grid
(mlp__embed_dim) and one search picks it.

## Trained parameters
From results/metrics/crypto/torch_mlp_emb/e11_emb_raw.json: model torch_mlp_emb,
regression, target_binned_return_20, data_version crypto/v2.0-symid, n_features 23,
cv_scheme rolling, n_splits 5, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50, no fixed_params. Refit-0 best_params: alpha 12.1547, batch_size 1024,
embed_dim 16, hidden [256,128], lr 0.000257. The no-embedding control chose alpha 19.4149,
batch_size 2048, hidden [64,32], lr 0.005541.

## What was found
No written finding; from the metrics (514 periods) the embedding HURTS badly. The
no-embedding MLP pays 0.076618 (CORR 0.144701, MMC 0.062148); adding the learned ticker
embedding roughly halves everything — `e11_emb_raw` payout 0.033465 (CORR 0.080193,
MMC 0.025445). The embedding did not earn its place, consistent with the stated priors
(many symbols with few rows). No paired t-stat in available artifacts.
