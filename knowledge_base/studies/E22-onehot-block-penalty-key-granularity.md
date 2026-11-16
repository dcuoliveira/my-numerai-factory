# E22 — The one-hot block: the penalty, the key (symbol vs ucid), the granularity

- **State:** done (13/13 arms)
- **Runner:** experiments/e22_onehot_reg_crypto.sh
- **Models / datasets:** `gen_ridge`. A raw control (crypto/v2.0) plus one-hot identity blocks keyed three ways — symbol (e22ohsymbol), ucid (e22ohucid), market-cap bucket (e22ohcap) — at penalties p0/p10/flat, and "only"-granularity variants (e22onlyucid / e22onlysymbol / e22onlycap) with no raw features.
- **Artifacts on disk:** 57.22 MB across 13 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/gen_ridge/e22_sym_p0.json

## Main question
Three axes of the one-hot identity block: how hard to penalise it (p0 = none, p10, flat), which key to one-hot on (symbol vs ucid vs coarser cap bucket), and the granularity (raw + one-hot vs the one-hot block ALONE).

## What was tried
13 gen_ridge arms crossing {penalty} × {key} × {granularity}, each on its own one-hot dataset version, searched, against the raw control e22_raw.

## Trained parameters
From results/metrics/crypto/gen_ridge/e22_sym_p0.json: model `gen_ridge`, data_version `crypto/v2.0-e22ohsymbol`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 768, seed 42, n_iter 50, n_splits 5, val_frac 0.3, fixed_params None. Each arm's dataset is its own `crypto/v2.0-e22oh*` / `e22only*` version; the penalty level is encoded by the arm.

## What was found
No written finding; from the metrics. The raw control is CORR 0.143822 / MMC 0.061533. A one-hot block added to raw only matches raw when LIGHTLY penalised, and keying barely matters: e22_ucid_p10 0.146335 (top full arm), ucid_flat 0.144690, sym_flat 0.144737, sym_p10 0.144012 — all ≈ raw. Unpenalised (p0) full arms collapse: sym_p0 0.100231, ucid_p0 0.100830. The coarse cap-keyed arms land between (cap_p0 0.137841, cap_flat 0.136662), still below raw. The "only"-one-hot arms (no raw features) are weak on CORR (only_ucid_p0 0.095387, only_sym_p0 0.094786, only_cap_p0 0.079102) yet carry the study's highest MMC (only_ucid_p0 MMC 0.063195) — the identity block alone contributes MMC-style signal but little CORR. So: ucid ≈ symbol as a key, a light penalty is necessary (p0 destroys it), and dropping to one-hot-only costs CORR. No paired t-stats recorded.
