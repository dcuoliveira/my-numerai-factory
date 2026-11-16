# E24 — Can TabPFN be pushed past its E19 default? Three Tier-1 levers

- **State:** done (11/11 arms)
- **Runner:** experiments/e24_tabpfn_tuning_crypto.sh
- **Models / datasets:** `tabpfn` and `tabpfn_clf` on fenccausal (44f), tenc, raw, interact. Three levers: softmax temperature (temp050/temp075/temp125), full/larger regression context (`reg_full`, max_context 8192), and the classifier head (`clf`).
- **Artifacts on disk:** 57.70 MB across 11 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §E24 (line 39); memory: none (E19's TabPFN caveat noted below); metrics: results/metrics/crypto/tabpfn/e24_temp125_fenccausal.json

## Main question
E19 ran TabPFN at defaults. Can three Tier-1 levers — prediction temperature, a larger in-context window, and swapping the regression head for a classifier head — push it past that default?

## What was tried
11 in-context arms (no training, single forward pass each): temperature sweep on fenccausal (0.50/0.75/1.25), `reg_full` (larger context) on fenccausal/tenc/raw/interact, and `tabpfn_clf` on fenccausal/tenc/raw/interact. Compared against E19's per-dataset TabPFN numbers.

## Trained parameters
From results/metrics/crypto/tabpfn/e24_temp125_fenccausal.json: model `tabpfn`, data_version `crypto/v2.0-fenccausal`, cv_scheme `rolling`, embargo 20, train_window 130, refit_every 130, target `target_binned_return_20`, n_features 44, seed 42, n_iter 1, n_splits 2, val_frac 0.3, fixed_params `{max_context: 8192}`. The `clf` arms use model `tabpfn_clf`; the temperature level is encoded by the arm.

## What was found
The paired verdict (EXPERIMENTS.md §E24, line 39; paired Newey-West, lag 20, each arm against its E19 regressor baseline) is that neither temperature nor the larger context helps: full strength is ~NULL and temperature is ~NULL. The only significant lever is the classifier head, and only on raw — **+0.012 MMC, t +3.03** (re-checked from per_period: clf_raw vs e19_raw dMMC +0.0115, NW t +3.24 at lag 20; dCORR only +0.0046, t +1.09, not significant) — and it hurts fenccausal (dCORR −0.0205, t −3.03). The unpaired point estimates do NOT overturn this: e24_temp075_fenccausal CORR 0.156285 / MMC 0.069461 (n=510), with temp050 0.155427, reg_full_fenccausal 0.155750, and temp125 0.152752 (study-best MMC 0.071882), all sit above E19's best fenccausal (0.145042) and E19's best arm overall (interact 0.149611) — but a cross-arm point-estimate gap of 0.156 vs 0.145 is not a paired win, so it does not establish temperature or context as an improvement, and e24_temp075_fenccausal is not "the best config" in any paid sense. On the classifier head the CORR point estimates are generally worse (clf_fenccausal 0.124557, clf_tenc 0.130305, clf_interact 0.151968) except on raw, where clf_raw 0.147589 sits above reg_full_raw 0.141086 — but that CORR gap is not significant (paired t ≈ +1.1 to +1.4); the raw-only significant gain is in MMC, not CORR. The study's prize was Win B (a strengthened TabPFN that wins the neural-member swap SACI_03+TabPFN > SACI_03+IA_01): strengthening moves the blend swap from losing (t −2.67) to tied (t −1.89 / −0.80), but it still does NOT beat CU_SA_IA_BO_01. Caveat: these are single-pass in-context numbers, and e109-foundation-repr-block.md found E19-style TabPFN CORR to be a non-reproducible "lucky context draw" — read the level accordingly.
